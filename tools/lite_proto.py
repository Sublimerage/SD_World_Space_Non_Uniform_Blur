"""Numpy prototype of World Space Non-Uniform Blur Lite: the 2D Non-Uniform Blur run on six
axis projections (+-X, +-Y, +-Z) of the mesh, then blended back per texel (triplanar, by normal).

This is a design check, not the filter. The projection lookups here are a simple stand-in
for the shared Lite core (a forward splat, which a pixel processor can't do); the bokeh
passes and the resolve are what the .sbs will do.

Per projection pixel q (a front-most surface point facing that direction):
  * the same passes as the 2D graph and v2.0: N = min(Samples, ceil(Intensity * pi)), radius
    decay exp(-2k sqrt(-ln 0.001) / N), inner rotation (1 / Samples) / (k + 1) turns, a centre
    tap plus Blades corner taps, averaged;
  * the ellipse is built in the surface's tangent plane exactly like v2.0 (Reference Axis
    projected on the surface, Anisotropy across the Angle direction), then each world offset
    is projected into the 2D projection. On a flat area this lands on the same surface points
    as the world-space taps, so all projections agree and the triplanar blend shows no seam;
  * a tap is kept only if its depth matches the depth a flat surface would have there, so
    the blur doesn't jump across silhouettes onto a farther surface (the "Depth Test").
Resolve per texel: the three projections on the side its normal faces, weighted by |n|^4,
each used only where the texel is front-most there; little coverage -> keep the input.

    python3 tools/lite_proto.py <out_dir>     (all scenes, v2.0 comparison, images, numbers)
"""
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)

C_DECAY = math.sqrt(-math.log(0.001))
DEFAULTS = dict(intensity=10.0, samples=4, anisotropy=0.0, blades=5, angle=0.0, axis=(0.0, 1.0, 0.0),
                use_blur_map=True, depth_test=True, size=0.5, sharpness=4.0, frame='3d')


# ----------------------------------------------------------------------------
# mesh maps

def dilate_nearest(img, valid):
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return img[idx[0], idx[1]]


def normals_from_pos(pos, valid, pieces=None):
    """world normals from the position map (what Painter's world-space normal map gives),
    pointed away from each piece's centre (the test pieces are convex)"""
    p = dilate_nearest(pos, valid)
    dx = np.gradient(p, axis=1)
    dy = np.gradient(p, axis=0)
    n = np.cross(dx, dy)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-30)
    pid = np.zeros(valid.shape, int) if pieces is None else pieces
    for k in np.unique(pid[valid]):
        sel = valid & (pid == k)
        c = pos[sel].mean(0)
        flip = np.sum(n * (pos - c), -1) < 0
        n[sel & flip] *= -1
    return n, np.linalg.norm(dx, axis=-1) * np.linalg.norm(dy, axis=-1)


def length_per_uv(area, valid):
    """the mesh's average 3D length per UV unit, from its total area (as v2.0's WSNB_Radius)"""
    H, W = valid.shape
    return math.sqrt(area[valid].sum() / (valid.sum() / float(H * W)))


# ----------------------------------------------------------------------------
# projections (stand-in for the Lite core)

class Projections:
    """6 orthographic views of the mesh bounding box, square pixels of size h. View v = 2a + (s<0)
    looks down -s * axis a; a pixel holds the front-most texel whose normal faces s * axis a."""

    def __init__(self, pos, nrm, valid, S, ss=3):
        P = pos[valid]
        self.lo = P.min(0)
        ext = P.max(0) - self.lo
        self.h = ext.max() / (S - 2)
        self.lo = self.lo - self.h
        self.S = S
        H, W = valid.shape
        self.idx = -np.ones((6, S, S), np.int64)
        self.depth = np.full((6, S, S), -np.inf)
        flat_pos = dilate_nearest(pos, valid)
        ti = np.flatnonzero(valid.ravel())
        ty, tx = np.divmod(ti, W)
        for oy in range(ss):
            for ox in range(ss):
                # sub-texel positions (bilinear in the dilated position map), so steep areas have no holes
                fy, fx = ty + (oy + 0.5) / ss - 0.5, tx + (ox + 0.5) / ss - 0.5
                q = np.stack([ndimage.map_coordinates(flat_pos[..., c], [fy, fx], order=1, mode='nearest')
                              for c in range(3)], -1)
                for v in range(6):
                    a, s = v // 2, (1.0 if v % 2 == 0 else -1.0)
                    face = nrm.reshape(-1, 3)[ti, a] * s > 0.0
                    b, c = [i for i in range(3) if i != a]
                    u = np.floor((q[face, b] - self.lo[b]) / self.h).astype(int)
                    w = np.floor((q[face, c] - self.lo[c]) / self.h).astype(int)
                    d = q[face, a] * s
                    lin = (w * S + u)
                    order = np.argsort(d)                      # last write wins: largest depth
                    flat_idx = self.idx[v].ravel()
                    flat_d = self.depth[v].ravel()
                    better = d[order] > flat_d[lin[order]]
                    lin_o, d_o, t_o = lin[order][better], d[order][better], ti[face][order][better]
                    flat_d[lin_o] = d_o
                    flat_idx[lin_o] = t_o
                    self.idx[v] = flat_idx.reshape(S, S)
                    self.depth[v] = flat_d.reshape(S, S)
        self.cov = (self.idx >= 0).astype(np.float64)
        self.depth = np.where(self.cov > 0, self.depth, 0.0)

    def gather(self, img):
        """img (H, W, C) -> (6, S, S, C), 0 where uncovered"""
        f = img.reshape(-1, img.shape[-1])
        out = f[np.maximum(self.idx, 0)]
        return out * self.cov[..., None]

    def plane(self, v, p):
        """world points p (..., 3) -> (pixel x, pixel y, depth) in view v"""
        a, s = v // 2, (1.0 if v % 2 == 0 else -1.0)
        b, c = [i for i in range(3) if i != a]
        return (p[..., b] - self.lo[b]) / self.h, (p[..., c] - self.lo[c]) / self.h, p[..., a] * s


def bilinear(img, x, y):
    """img (S, S, C), pixel coords (centres at i + 0.5); outside = 0"""
    S = img.shape[0]
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)))
    x = np.clip(x - 0.5 + 1.0, 0.0, S + 0.999)
    y = np.clip(y - 0.5 + 1.0, 0.0, S + 0.999)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x1, y1 = np.minimum(x0 + 1, S + 1), np.minimum(y0 + 1, S + 1)
    return ((pad[y0, x0] * (1 - fx) + pad[y0, x1] * fx) * (1 - fy) +
            (pad[y1, x0] * (1 - fx) + pad[y1, x1] * fx) * fy)


# ----------------------------------------------------------------------------
# the filter

def tangent_frame(n, axis):
    """v2.0's frame: the Reference Axis projected on the surface (another axis near its pole)"""
    D = np.asarray(axis, float)
    D = D / np.linalg.norm(D) if np.dot(D, D) > 1e-8 else np.array([0.0, 1.0, 0.0])
    t = D - n * (n @ D)[..., None]
    alt = np.array([1.0, 0.0, 0.0]) if abs(D[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    tb = alt - n * (n @ alt)[..., None]
    t = np.where((np.sum(t * t, -1) > 0.04)[..., None], t, tb)
    t1 = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-6)
    return t1, np.cross(n, t1)


def lite(col, pos, uvm, bmap, prm=None, pieces=None, stats=None, nrm=None):
    prm = dict(DEFAULTS, **(prm or {}))
    valid = uvm > 0.5
    H, W = valid.shape
    n_est, area = normals_from_pos(pos, valid, pieces)
    nrm = n_est if nrm is None else nrm
    luv = length_per_uv(area, valid)
    S = max(16, int(round(max(H, W) * prm['size'])))
    pr = Projections(pos, nrm, valid, S)
    C = col.shape[-1]
    eff = np.clip(bmap, 0, 1) if prm['use_blur_map'] else np.ones(valid.shape)
    cur = pr.gather(col.astype(np.float64))                    # (6, S, S, C), colour * coverage
    pn = pr.gather(nrm)
    pe = pr.gather(eff[..., None])[..., 0]
    I, sm, nbl = prm['intensity'], max(int(prm['samples']), 1), int(np.clip(prm['blades'], 1, 9))
    npass = int(min(sm, math.ceil(max(I, 0.0) * math.pi)))
    an, phi = float(np.clip(prm['anisotropy'], 0, 1)), prm['angle'] * -2 * math.pi
    reads = 6 * S * S * 3                                       # gather: lookup, input, blur map
    geo = np.stack([pr.depth * pr.cov, pr.cov], -1)            # static per mesh: (depth * cov, cov)
    for k in range(npass):
        nxt = cur.copy()
        for v in range(6):
            a, s = v // 2, (1.0 if v % 2 == 0 else -1.0)
            b, c = [i for i in range(3) if i != a]
            qy, qx = np.nonzero(pr.cov[v] > 0)
            n = pn[v, qy, qx]
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
            r = I / 256.0 * luv * pe[v, qy, qx] * math.exp(-2.0 * k * C_DECAY / npass)
            if prm['frame'] == '3d':
                t1, t2 = tangent_frame(n, prm['axis'])
            else:   # rejected variant: one fixed 2D frame per view (the Reference Axis on the view plane)
                ea = np.zeros((1, 3))
                ea[0, a] = s
                t1, t2 = tangent_frame(ea, prm['axis'])
                t1, t2 = np.repeat(t1, len(qy), 0), np.repeat(t2, len(qy), 0)
            d0 = pr.depth[v, qy, qx]
            acc = cur[v, qy, qx].copy()                         # centre tap, weight 1
            wsum = np.ones(len(qy))
            reads += len(qy) * 4                                # centre colour, depth, normal, blur map
            for j in range(1, nbl + 1):
                th = (1.0 / sm) * (2 * math.pi / (k + 1.0)) + j * (2 * math.pi / nbl)
                s1, c1 = (1.0 - an) * math.sin(th), math.cos(th)
                X = s1 * math.sin(phi) + c1 * math.cos(phi)
                Y = s1 * math.cos(phi) - math.sin(phi) * c1
                off = (t1 * X + t2 * Y) * r[:, None]            # world offset in the tangent plane
                tx = qx + 0.5 + off[:, b] / pr.h
                ty = qy + 0.5 + off[:, c] / pr.h
                g = bilinear(geo[v], tx, ty)
                ccol = bilinear(cur[v], tx, ty)
                reads += len(qy) * 2
                w = g[:, 1]
                if prm['depth_test']:
                    dt = g[:, 0] / np.maximum(w, 1e-9)
                    want = d0 + off[:, a] * s
                    tol = 0.5 * r + 2.0 * pr.h
                    w = w * (np.abs(dt - want) < tol)
                    ccol = ccol * (np.abs(dt - want) < tol)[:, None]
                acc += ccol
                wsum += w
            nxt[v, qy, qx] = acc / wsum[:, None]
            reads += S * S - len(qy)                            # uncovered: one read, then skip
        cur = nxt * pr.cov[..., None]
    # resolve: triplanar by normal, only where the texel is front-most in that view
    out = col.astype(np.float64).copy()
    ty, tx = np.nonzero(valid)
    p, n = pos[ty, tx], nrm[ty, tx]
    num = np.zeros((len(ty), C))
    den = np.zeros(len(ty))
    tot = np.zeros(len(ty))
    for a in range(3):
        wa = np.abs(n[:, a]) ** prm['sharpness']
        tot += wa
        for v in (2 * a, 2 * a + 1):
            side = (n[:, a] > 0) if v % 2 == 0 else (n[:, a] <= 0)
            px, py, d = pr.plane(v, p)
            g = bilinear(geo[v], px, py)
            cv = bilinear(cur[v], px, py)
            vis = side & (g[:, 1] > 0.25) & (np.abs(g[:, 0] / np.maximum(g[:, 1], 1e-9) - d) < 2.5 * pr.h)
            w = wa * vis
            num += w[:, None] * cv / np.maximum(g[:, 1], 1e-9)[:, None]
            den += w
    reads += len(ty) * (3 + 3 * 2) + (H * W - len(ty))           # position, normal, input + 3 views
    cover = den / np.maximum(tot, 1e-12)
    res = num / np.maximum(den, 1e-12)[:, None]
    al = np.clip((cover - 0.3) / 0.5, 0, 1)
    al = al * al * (3 - 2 * al)
    # like v2.0: fade the blur in over the first pixels of radius, so tiny blurs keep the input
    rp = np.clip(I / 256.0 * luv * eff[ty, tx] / pr.h * 0.5, 0, 1) if npass > 0 else np.zeros(len(ty))
    al = al * rp * rp * (3 - 2 * rp)
    out[ty, tx] = col[ty, tx] + (res - col[ty, tx]) * al[:, None]
    if stats is not None:
        stats.update(reads=reads, S=S, h=pr.h, luv=luv, passes=npass, coverage=float((cover > 0.8).mean()),
                     proj_pixels=int(pr.cov.sum()))
    return out


# ----------------------------------------------------------------------------
# 2D Non-Uniform Blur (the reference on a flat plane), anisotropy and angle included

def nub2d(img, eff, I, S, Bl, an=0.0, angle=0.0, ysign=-1.0):
    """rows follow ysign * Y: -1 matches the world frame of a plane at y = 0 seen from +Y"""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(float)
    Ns = int(min(S, math.ceil(I * math.pi)))
    out = img.astype(np.float64).copy()
    phi = angle * -2 * math.pi
    for k in range(Ns):
        r = I / 256 * img.shape[0] * math.exp(-2 * k * C_DECAY / Ns) * eff
        acc = out.copy()
        for j in range(1, Bl + 1):
            th = (1.0 / S) * (2 * math.pi / (k + 1.0)) + j * 2 * math.pi / Bl
            s1, c1 = (1 - an) * math.sin(th), math.cos(th)
            X = s1 * math.sin(phi) + c1 * math.cos(phi)
            Y = s1 * math.cos(phi) - math.sin(phi) * c1
            acc += np.stack([ndimage.map_coordinates(out[..., c], [yy + ysign * Y * r, xx + X * r], order=1,
                                                     mode='nearest') for c in range(img.shape[-1])], -1)
        out = acc / (Bl + 1)
    return out
