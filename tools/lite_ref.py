"""World Space Non-Uniform Blur Lite: the NumPy spec of world_space_non_uniform_blur_lite_v1.0.sbs
(tools/build_wsnub_lite.py builds the graph, tools/test_lite.py checks it against this).

The mesh part is the shared Lite core of World Space Mask Blur Lite (tools/wsb_lite/,
vendored from sublimerage/my-project 015857f): each piece (v2.0 box) seen from the 6 axis
directions, every surface sample owned by the view along its dominant normal axis, and per
fine view pixel the front-most hit (u, v, depth, area). See tools/wsb_lite/README.md.

Per stroke:
  1. Level: the view level L (at least 1) where the largest radius (Intensity / 256 x the
     mesh's length per UV) is at most PIXELS_PER_RADIUS level pixels, so big blurs cost little.
     The per-stroke tables are VW x VH / 2, which holds every view at level 1 or coarser.
  2. Fill: per level pixel, the front surface of its 2^L x 2^L fine pixels: the input colour
     at the hit UVs (area weighted), and with the core's own fill, the Blur Map and the depth.
     The blur amount (Blur Map or the colour's luminance, Invert, clamp) is taken from those
     averages.
  3. Geometry: the surface normal from the depth's slope across neighbouring pixels (one-sided
     where a neighbour is missing; the smaller slope where both exist). It faces the view by
     construction, so UV winding can't flip it.
  4. Passes: the 2D Non-Uniform Blur / v2.0 passes, per level pixel: N = min(Samples,
     ceil(Intensity pi)), radius decaying by exp(-2k sqrt(-ln 0.001) / N), inner rotation
     (1 / Samples) / (k + 1) turns, a centre tap plus Blades corner taps. The ellipse is
     v2.0's, built in the tangent plane (Reference Axis on the surface, Anisotropy across the
     Angle direction) and projected into the view. A tap reads the 4 pixels around it
     (bilinear); a pixel counts only if its depth is within 0.5 r + 2 p of where a flat
     surface would be, so the blur doesn't jump across silhouettes.
  5. Resolve: every view of the texel's box adds its result at the texel's projection
     (4 bilinear pixels, each times exp(-dz^2 / 2 (DEPTH_SIGMA p)^2)), then the blur is faded
     in by coverage and by the radius in pixels (tiny blurs keep the input), and dithered.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb_lite'))
import world_space_blur as w  # noqa: E402
import world_space_blur_lite as lref  # noqa: E402

C_DECAY = math.sqrt(-math.log(0.001))
MAX_PASSES = 16
PIXELS_PER_RADIUS = 16.0     # level choice: largest radius <= this many level pixels
MIN_LEVEL = 1                # per-stroke tables are VW x VH / 2: level 0 would not fit
DEPTH_SIGMA = 1.0            # resolve: depth agreement, in level pixels
COVER_FULL = 0.5             # resolve: weight sum that counts as fully covered
EMPTY = -1e20                # pass images: depth of an empty pixel
LUM = np.array([0.2126, 0.7152, 0.0722])

DEFAULTS = dict(samples=4, intensity=10.0, anisotropy=0.0, blades=5, angle=0.0, use_blur_map=True,
                blur_map_from_input=True, invert_blur_map=False, quality=4, axis_x=0.0, axis_y=1.0, axis_z=0.0,
                dither=1.0)


def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blur_amount(col, bmap, prm):
    """per-texel blur scale, as v2.0: 1, the Blur Map, or the luminance of the input"""
    if not prm['use_blur_map']:
        return np.ones(col.shape[:-1])
    e = col[..., :3] @ LUM if prm['blur_map_from_input'] else bmap
    e = 1.0 - e if prm['invert_blur_map'] else e
    return np.clip(e, 0.0, 1.0)


def pass_count(prm):
    return int(min(max(int(prm['samples']), 1), math.ceil(max(prm['intensity'], 0.0) * math.pi)))


def level(prm, luv, p0):
    """A (tolerance for the fill, L, p, largest radius), B (0, 2^L, 0, 1), like WSB_LLevel"""
    rmax = max(prm['intensity'], 0.0) / 256.0 * luv
    L, p = MIN_LEVEL, p0 * 2.0 ** MIN_LEVEL
    for _ in range(lref.MAX_LEVEL - MIN_LEVEL):
        if rmax > PIXELS_PER_RADIUS * p:
            L, p = L + 1, p * 2.0
    return dict(sigma=0.5 * rmax, L=L, p=p, rmax=rmax)


def tangent_frame(n, prm):
    """v2.0's frame: the Reference Axis projected on the surface (another axis near its pole)"""
    D = np.array([prm['axis_x'], prm['axis_y'], prm['axis_z']], float)
    l2 = D @ D
    D = D / max(math.sqrt(l2), 1e-6) if l2 > 1e-8 else np.array([0.0, 1.0, 0.0])
    t = D - n * (n @ D)[..., None]
    alt = np.array([1.0, 0.0, 0.0]) if abs(D[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    tb = alt - n * (n @ alt)[..., None]
    t = np.where((np.sum(t * t, -1) > 0.04)[..., None], t, tb)
    t1 = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-6)
    return t1, np.cross(n, t1)


def views(tab):
    for e in range(len(tab['ok'])):
        if tab['ok'][e]:
            yield e, int(tab['Wl'][e]), int(tab['Hl'][e]), int(tab['offset'][e])


def fill_colour(col, fine, tab0, tabL, lev):
    """WSNBL_FillColour: per level pixel, (sum c a) / B^2 over the front surface of its B x B fine
    pixels (the core's fill rule)"""
    B = 2 ** lev['L']
    out = np.zeros((lref.VW * lref.VH, 4))
    for e in range(len(tab0['ok'])):
        if not tab0['ok'][e]:
            continue
        W, H = int(tab0['W'][e]), int(tab0['H'][e])
        Wl, Hl = int(tabL['Wl'][e]), int(tabL['Hl'][e])
        f = fine[int(tab0['offset'][e]):int(tab0['offset'][e]) + W * H].reshape(H, W, 4)
        f = np.pad(f, ((0, Hl * B - H), (0, Wl * B - W), (0, 0)))
        cov = f[..., 3] > 0
        c = np.where(cov[..., None], w.sample_nearest(col[..., :3], f[..., 0], f[..., 1]), 0.0)
        z = np.where(cov, f[..., 2], -np.inf).reshape(Hl, B, Wl, B)
        zf = z.max((1, 3))
        keep = (z >= (zf - (lev['sigma'] + 2.0 * lev['p']))[:, None, :, None]).reshape(Hl * B, Wl * B) & cov
        a = np.where(keep, f[..., 3], 0.0)
        sm = (c * a[..., None]).reshape(Hl, B, Wl, B, 3).sum((1, 3)) / float(B * B)
        o = int(tabL['offset'][e])
        out[o:o + Hl * Wl, :3] = sm.reshape(-1, 3)
    return out


def scale_of(colour, bm, prm):
    """WSNBL blur amount from an averaged colour / blur map value: Invert, clamp, Use"""
    if not prm['use_blur_map']:
        return np.ones(colour.shape[:-1])
    e = colour @ LUM if prm['blur_map_from_input'] else bm
    e = 1.0 - e if prm['invert_blur_map'] else e
    return np.clip(e, 0.0, 1.0)


def geometry(zfill, fillc, tabL, lev, prm):
    """WSNBL_Geometry: (depth, blur amount, n_u, n_v), or (0, -1, 0, 0) where empty. n_u, n_v:
    the outward normal's components along the view's image axes, from the depth slope"""
    p = lev['p']
    out = np.zeros((lref.VW * lref.VH, 4))
    out[:, 1] = -1.0                                         # outside every view: empty
    for e, Wl, Hl, o in views(tabL):
        zf_ = zfill[o:o + Hl * Wl].reshape(Hl, Wl, 4)
        z = zf_[..., 2]
        has = zf_[..., 3] > 0.5
        a = np.maximum(zf_[..., 1], 1e-30)
        colour = fillc[o:o + Hl * Wl, :3].reshape(Hl, Wl, 3) / a[..., None]
        ef = scale_of(colour, zf_[..., 0] / a, prm)
        grads = []
        for axis in (1, 0):                                  # along x (u), then y (v)
            zp = np.roll(z, -1, axis)
            zm = np.roll(z, 1, axis)
            hp = np.roll(has, -1, axis)
            hm = np.roll(has, 1, axis)
            n = z.shape[axis]
            idx = np.arange(n)
            last = (idx == n - 1) if axis == 1 else (idx == n - 1)[:, None]
            first = (idx == 0) if axis == 1 else (idx == 0)[:, None]
            hp = hp & ~last
            hm = hm & ~first
            dp, dm = zp - z, z - zm
            g = np.where(hp & hm, np.where(np.abs(dp) <= np.abs(dm), dp, dm), np.where(hp, dp, np.where(hm, dm, 0.0)))
            grads.append(g / p)
        gu, gv = grads
        ln = np.sqrt(1.0 + gu * gu + gv * gv)
        d = np.stack([np.where(has, z, 0.0), np.where(has, ef, -1.0), np.where(has, -gu / ln, 0.0),
                      np.where(has, -gv / ln, 0.0)], -1)
        out[o:o + Hl * Wl] = d.reshape(-1, 4)
    return out


def start(zfill, fillc):
    """WSNBL_Start: (colour, depth), or (0, 0, 0, EMPTY) where empty"""
    has = zfill[:, 3:4] > 0.5
    colour = fillc[:, :3] / np.maximum(zfill[:, 1:2], 1e-30)
    return np.concatenate([np.where(has, colour, 0.0), np.where(has, zfill[:, 2:3], EMPTY)], -1)


def bokeh_pass(P, D, tabL, lev, prm, k, stats=None):
    """one pass: P (colour, depth or EMPTY) -> P"""
    out = P.copy()
    N = pass_count(prm)
    if not (k < N and prm['intensity'] > 0.0):
        return out
    p = lev['p']
    nbl = int(np.clip(int(prm['blades']), 1, 9))
    sm = max(int(prm['samples']), 1)
    an = float(np.clip(prm['anisotropy'], 0.0, 1.0))
    phi = prm['angle'] * -2.0 * math.pi
    for e, Wl, Hl, o in views(tabL):
        ax, sgn = int(tabL['ax'][e]), tabL['sgn'][e]
        _, ua, va = lref.AXES[ax]
        v = P[o:o + Hl * Wl].reshape(Hl, Wl, 4)
        d = D[o:o + Hl * Wl].reshape(Hl, Wl, 4)
        yy, xx = np.nonzero(d[..., 1] >= 0.0)
        if len(yy) == 0:
            continue
        nu, nv = d[yy, xx, 2], d[yy, xx, 3]
        n = np.zeros((len(yy), 3))
        n[:, ax] = sgn * np.sqrt(np.maximum(1.0 - nu * nu - nv * nv, 0.0))
        n[:, ua], n[:, va] = nu, nv
        t1, t2 = tangent_frame(n, prm)
        r = lev['rmax'] * d[yy, xx, 1] * math.exp(-(2.0 * k * C_DECAY) / max(N, 1))
        act = r > 0.05 * p
        zc = v[yy, xx, 3]
        acc = v[yy, xx, :3].copy()
        ws = np.ones(len(yy))
        tol = 0.5 * r + 2.0 * p
        for j in range(1, nbl + 1):
            th = (1.0 / sm) * (2.0 * math.pi / (k + 1.0)) + j * (2.0 * math.pi / nbl)
            s1, c1 = (1.0 - an) * math.sin(th), math.cos(th)
            X = s1 * math.sin(phi) + c1 * math.cos(phi)
            Y = s1 * math.cos(phi) - math.sin(phi) * c1
            off = (t1 * X + t2 * Y) * r[:, None]
            x, y = xx + off[:, ua] / p, yy + off[:, va] / p
            want = zc + sgn * off[:, ax]
            x0, y0 = np.floor(x), np.floor(y)
            fx, fy = x - x0, y - y0
            for oy in (0, 1):
                for ox in (0, 1):
                    X_, Y_ = (x0 + ox).astype(int), (y0 + oy).astype(int)
                    inb = (X_ >= 0) & (Y_ >= 0) & (X_ < Wl) & (Y_ < Hl)
                    q = v[np.clip(Y_, 0, Hl - 1), np.clip(X_, 0, Wl - 1)]
                    ok = inb & (np.abs(q[:, 3] - want) < tol)
                    wt = np.where(ok, (fx if ox else 1.0 - fx) * (fy if oy else 1.0 - fy), 0.0)
                    acc += wt[:, None] * q[:, :3]
                    ws += wt
        res = np.where(act[:, None], acc / ws[:, None], v[yy, xx, :3])
        nvw = v.copy()
        nvw[yy, xx, :3] = res
        out[o:o + Hl * Wl] = nvw.reshape(-1, 4)
    return out


def resolve(col, eff, position, uv_mask, mesh, P, tab, lev, prm):
    hgt, wid = uv_mask.shape[:2]
    yy, xx = np.mgrid[0:hgt, 0:wid]
    u, v = (xx + 0.5) / wid, (yy + 0.5) / hgt
    pos = w.sample_nearest(position, u, v)[..., :3]
    inside = w.sample_nearest(uv_mask, u, v) > 0.5
    b = w.box_of(pos, mesh['coarse'], mesh['boxes'])
    p = lev['p']
    sz = DEPTH_SIGMA * p
    num = np.zeros((hgt, wid, 3))
    den = np.zeros((hgt, wid))
    for e, Wl, Hl, o in views(tab):
        ax, sgn = int(tab['ax'][e]), tab['sgn'][e]
        _, ua, va = lref.AXES[ax]
        sel = inside & (b == e // lref.NVIEWS)
        if not sel.any():
            continue
        vw = P[o:o + Hl * Wl].reshape(Hl, Wl, 4)
        x = (pos[..., ua] - tab['ou'][e]) / p - 0.5
        y = (pos[..., va] - tab['ov'][e]) / p - 0.5
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        z = sgn * pos[..., ax]
        for oy in (0, 1):
            for ox in (0, 1):
                X, Y = (x0 + ox).astype(int), (y0 + oy).astype(int)
                ok = sel & (X >= 0) & (Y >= 0) & (X < Wl) & (Y < Hl)
                q = vw[np.clip(Y, 0, Hl - 1), np.clip(X, 0, Wl - 1)]
                f = np.exp(-(q[..., 3] - z) ** 2 / (2.0 * sz * sz))
                wt = np.where(ok, (fx if ox else 1.0 - fx) * (fy if oy else 1.0 - fy) * f, 0.0)
                num += wt[..., None] * q[..., :3]
                den += wt
    src = w.sample_nearest(col, u, v)
    e_t = w.sample_nearest(eff, u, v)
    res = num / np.maximum(den, 1e-12)[..., None]
    rp = lev['rmax'] * e_t / p * 0.5 if pass_count(prm) > 0 and prm['intensity'] > 0 else np.zeros_like(e_t)
    al = smoothstep01(den / COVER_FULL) * smoothstep01(rp)
    ok = inside & (b >= 0) & (den > 1e-6)
    al = np.where(ok, al, 0.0)
    out = src.copy()
    blended = src[..., :3] + (res - src[..., :3]) * al[..., None]
    k = np.clip(np.minimum(blended, 1.0 - blended) * 255.0, 0.0, 1.0)
    n = w.hash12(xx.astype(float), yy.astype(float)) + w.hash12(xx + 37.0, yy + 91.0) - 1.0
    dith = np.clip(blended + k * (n * prm['dither'] / 255.0)[..., None], 0.0, 1.0)
    out[..., :3] = np.where(ok[..., None] & (al[..., None] > 0), dith, src[..., :3])
    return out, dict(uncovered=float((inside & ~ok).sum() / max(inside.sum(), 1)),
                     weak=float((inside & (den < COVER_FULL)).sum() / max(inside.sum(), 1)))


def prepare(position, uv_mask, quality=4):
    return lref.prepare(position[..., :3], uv_mask, quality=int(quality))


def nub_lite(col, position, uv_mask, blur_map=None, prm=None, lite=None, stats=None, keep=None):
    """col (H, W, 4) RGBA input -> (H, W, 4) output"""
    prm = dict(DEFAULTS, **(prm or {}))
    lite = lite or prepare(position, uv_mask, prm['quality'])
    mesh = lite['mesh']
    bmap = np.zeros(col.shape[:2]) if blur_map is None else blur_map
    eff = blur_amount(col, bmap, prm)
    lev = level(prm, mesh['luv'], lite['p0'])
    tabL = lref.view_table(mesh['boxes'], lite['p0'], lev['L'])
    fz = lref.pass_fill(bmap, lite['fine'], lite['tab0'], tabL, lev)
    fc = fill_colour(col, lite['fine'], lite['tab0'], tabL, lev)
    D = geometry(fz, fc, tabL, lev, prm)
    P = start(fz, fc)
    if keep is not None:
        keep.update(fill=fc, zfill=fz, geo=D, start=P.copy())
    for k in range(MAX_PASSES):
        P = bokeh_pass(P, D, tabL, lev, prm, k)
        if keep is not None:
            keep['pass%d' % k] = P.copy()
    out, st = resolve(col, eff, position, uv_mask, mesh, P, tabL, lev, prm)
    if stats is not None:
        stats.update(st)
        stats.update(L=lev['L'], p=lev['p'], rmax=lev['rmax'], p0=lite['p0'], passes=pass_count(prm),
                     views=int(tabL['ok'].sum()), level_pixels=float(tabL['size'].sum()),
                     covered=float((D[:, 1] >= 0).sum()))
    return out
