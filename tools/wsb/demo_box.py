"""
Test: does the world-space blur remove UV seams?

Builds a box whose six faces are separate UV islands, scattered and rotated
in UV space (so every box edge is a UV seam). A hard-edged mask is defined in
3D, so it is continuous across the seams, like a mask painted in Painter's
viewport.

Then it compares three blurs of that mask:
  uv     - a normal UV-space Gaussian blur with dilated padding
           (roughly what Painter's built-in Blur filter does)
  world  - reference/world_space_blur.py (the Designer filter's algorithm)
  truth  - brute-force Euclidean Gaussian blur over the surface

and reports
  seam error     - |result - truth| at texels right next to a box edge
                   (on both islands sharing that edge)
  interior error - |result - truth| at random surface texels
A seam shows up as seam error much larger than interior error.

Run:  python reference/demo_box.py [--images]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import world_space_blur as wsb  # noqa: E402

RES = 512
BOX = np.array([1.6, 1.0, 1.2])     # box dimensions (non-uniform on purpose)
CENTER = np.array([3.0, -2.0, 5.0])  # off-origin, to exercise the bounds pass
GAP = 0.04                           # UV gap between islands
RADIUS = 0.06                        # blur radius, fraction of object size


# --------------------------------------------------------------------------
# Test mesh: 6 rectangular faces, each its own UV island
# --------------------------------------------------------------------------

def make_faces():
    h = BOX / 2
    faces = []
    for axis in range(3):
        i, j = [a for a in range(3) if a != axis]
        for sign in (1, -1):
            o = np.zeros(3)
            o[axis] = sign * h[axis]
            o[i] = -h[i]
            o[j] = -h[j]
            U = np.zeros(3); U[i] = BOX[i]
            V = np.zeros(3); V[j] = BOX[j]
            faces.append((CENTER + o, U, V))
    return faces


def layout_islands(faces):
    """Shelf-pack the six faces into UV with rotations, uniform texel density."""
    rotations = [0, 1, 2, 3, 1, 2]       # quarter turns per face
    order = [3, 0, 5, 1, 4, 2]           # scramble placement order
    gap = GAP
    dims = []
    for f, r in zip(faces, rotations):
        a, b = np.linalg.norm(f[1]), np.linalg.norm(f[2])
        dims.append((b, a) if r % 2 else (a, b))
    k = 0.22                             # UV units per world unit
    islands = [None] * 6
    x, y, row_h = gap, gap, 0.0
    for idx in order:
        w, hh = dims[idx][0] * k, dims[idx][1] * k
        if x + w > 1 - gap:
            x, y, row_h = gap, y + row_h + gap, 0.0
        islands[idx] = (x, y, w, hh, rotations[idx])
        x += w + gap
        row_h = max(row_h, hh)
    assert y + row_h < 1, "islands do not fit"
    return islands, k


def st_from_ab(a, b, r):
    """Island-local (a, b) in [0,1]^2 -> face params (s, t) for quarter turn r."""
    if r == 0:
        return a, b
    if r == 1:
        return b, 1 - a
    if r == 2:
        return 1 - a, 1 - b
    return 1 - b, a


def ab_from_st(s, t, r):
    if r == 0:
        return s, t
    if r == 1:
        return 1 - t, s
    if r == 2:
        return 1 - s, 1 - t
    return t, 1 - s


def bake_maps(faces, islands):
    """Rasterize position + UV-island mask (what Painter feeds the filter)."""
    pos = np.zeros((RES, RES, 3))
    uvm = np.zeros((RES, RES))
    face_id = -np.ones((RES, RES), int)
    yy, xx = np.mgrid[0:RES, 0:RES]
    u, v = (xx + 0.5) / RES, (yy + 0.5) / RES
    for fi, ((O, U, V), (x0, y0, w, h, r)) in enumerate(zip(faces, islands)):
        a, b = (u - x0) / w, (v - y0) / h
        inside = (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
        s, t = st_from_ab(a, b, r)
        P = O + s[..., None] * U + t[..., None] * V
        pos[inside] = P[inside]
        uvm[inside] = 1
        face_id[inside] = fi
    return pos, uvm, face_id


def mask_3d(P):
    """Hard-edged 3D pattern, continuous across UV seams."""
    q = (P - CENTER) * 9.0
    f = np.sin(q[..., 0] * 1.3 + q[..., 1] * 0.4) + np.sin(q[..., 1] * 1.7 + 1.0) + np.sin(q[..., 2] * 1.1 + q[..., 0] * 0.5 + 2.0)
    return (f > 0.0).astype(float)


# --------------------------------------------------------------------------
# Baseline: UV-space blur with dilated padding
# --------------------------------------------------------------------------

def dilate(img, valid, iters=24):
    img, valid = img.copy(), valid.copy()
    for _ in range(iters):
        acc = np.zeros_like(img)
        cnt = np.zeros_like(img)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            acc += np.roll(np.where(valid, img, 0), (dy, dx), (0, 1))
            cnt += np.roll(valid.astype(float), (dy, dx), (0, 1))
        grow = (~valid) & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow]
        valid = valid | grow
    return img


def gaussian_blur_uv(img, sigma_px):
    r = int(np.ceil(3 * sigma_px))
    x = np.arange(-r, r + 1)
    k = np.exp(-x * x / (2 * sigma_px * sigma_px))
    k /= k.sum()
    pad = np.pad(img, r, mode="edge")
    tmp = sum(k[i] * pad[:, i:i + img.shape[1]] for i in range(len(k)))
    tmp = tmp[r:-r]
    pad = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
    return sum(k[i] * pad[i:i + img.shape[0]] for i in range(len(k)))


# --------------------------------------------------------------------------
# Ground truth at arbitrary 3D points
# --------------------------------------------------------------------------

def ground_truth(points, src_pos, src_mask, sigma, chunk=512):
    out = np.zeros(len(points))
    inv = 1 / (2 * sigma * sigma)
    ss = np.sum(src_pos * src_pos, 1)
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        d2 = np.sum(p * p, 1)[:, None] + ss[None] - 2 * p @ src_pos.T
        g = np.exp(-np.maximum(d2, 0) * inv)
        out[i:i + chunk] = (g @ src_mask) / g.sum(1)
    return out


# --------------------------------------------------------------------------
# Seam probes: same 3D point seen from both islands sharing an edge
# --------------------------------------------------------------------------

def seam_probes(faces, islands, per_edge=120, inset_px=1.5):
    """Returns (uvA, uvB, P): pairs of UV coords on either side of each edge."""
    uvA, uvB, pts = [], [], []
    for fa in range(6):
        for fb in range(fa + 1, 6):
            Oa, Ua, Va = faces[fa]
            Ob, Ub, Vb = faces[fb]
            # shared edge: points on face a's border that also lie on face b
            for edge_s, edge_t in (("s", 0.0), ("s", 1.0), ("t", 0.0), ("t", 1.0)):
                c = np.linspace(0.03, 0.97, per_edge)
                s = c if edge_s == "t" else np.full_like(c, edge_t)
                t = c if edge_s == "s" else np.full_like(c, edge_t)
                P = Oa + s[:, None] * Ua + t[:, None] * Va
                # project onto face b plane params
                sb = (P - Ob) @ Ub / (Ub @ Ub)
                tb = (P - Ob) @ Vb / (Vb @ Vb)
                Pb = Ob + sb[:, None] * Ub + tb[:, None] * Vb
                on_b = (np.abs(Pb - P).max(1) < 1e-9) & (sb > -1e-9) & (sb < 1 + 1e-9) & (tb > -1e-9) & (tb < 1 + 1e-9)
                if not on_b.all():
                    continue
                for (O, U, V, ss_, tt_, isl, dst) in (
                    (Oa, Ua, Va, s, t, islands[fa], uvA),
                    (Ob, Ub, Vb, sb, tb, islands[fb], uvB),
                ):
                    x0, y0, w, h, r = isl
                    a, b = ab_from_st(np.clip(ss_, 0, 1), np.clip(tt_, 0, 1), r)
                    # push inward by inset_px texels
                    a = np.clip(a, inset_px / (w * RES), 1 - inset_px / (w * RES))
                    b = np.clip(b, inset_px / (h * RES), 1 - inset_px / (h * RES))
                    dst.append(np.stack([x0 + a * w, y0 + b * h], 1))
                pts.append(P)
    return np.concatenate(uvA), np.concatenate(uvB), np.concatenate(pts)


def seam_pos(pos, uv):
    """3D position of the texel each probe lands in (what the filter sees)."""
    return wsb.sample_nearest(pos, uv[:, 0], uv[:, 1])


def render(img, pos, uvm, path):
    """Tiny point-splat render of the box from a corner, for eyeballing seams."""
    from PIL import Image
    S = 420
    d = np.array([1.0, 0.8, 1.3]); d /= np.linalg.norm(d)
    right = np.cross([0, 1, 0], d); right /= np.linalg.norm(right)
    up = np.cross(d, right)
    valid = uvm > 0.5
    P = pos[valid] - CENTER
    col = img[valid]
    x = P @ right; y = P @ up; z = P @ d
    scale = S * 0.42 / np.abs(np.stack([x, y])).max()
    px = (S / 2 + x * scale).astype(int)
    py = (S / 2 - y * scale).astype(int)
    zbuf = np.full((S, S), -np.inf)
    out = np.full((S, S), 0.15)
    for k in np.argsort(z):
        for oy in (0, 1):
            for ox in (0, 1):
                X, Y = px[k] + ox, py[k] + oy
                if 0 <= X < S and 0 <= Y < S and z[k] >= zbuf[Y, X]:
                    zbuf[Y, X] = z[k]
                    out[Y, X] = col[k]
    Image.fromarray((out * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="write PNGs to docs/images")
    ap.add_argument("--quality", type=int, choices=sorted(wsb.QUALITY), default=3)
    ap.add_argument("--radius", type=float, default=RADIUS)
    args = ap.parse_args()
    radius = args.radius

    faces = make_faces()
    islands, k = layout_islands(faces)
    pos, uvm, _ = bake_maps(faces, islands)
    valid = uvm > 0.5
    mask = np.where(valid, mask_3d(pos), 0.0)

    size = BOX.max()
    sigma_world = radius * size * 0.5

    # UV blur: same physical sigma, converted to texels
    sigma_px = sigma_world * k * RES
    uv_blur = gaussian_blur_uv(dilate(mask, valid), sigma_px)
    uv_blur = np.where(valid, uv_blur, 0)

    world = wsb.world_space_blur(mask, pos, uvm, radius=radius, quality=args.quality)
    world = np.where(valid, world, 0)

    # truth sources: every valid texel (uniform texel density => equal area)
    src_pos, src_mask = pos[valid], mask[valid]

    uvA, uvB, P = seam_probes(faces, islands)
    rng = np.random.default_rng(0)
    vidx = np.argwhere(valid)
    pick = vidx[rng.choice(len(vidx), 3000, replace=False)]
    rand_uv = (pick[:, ::-1] + 0.5) / RES
    rand_P = pos[pick[:, 0], pick[:, 1]]

    truth_rand = ground_truth(rand_P, src_pos, src_mask, sigma_world)

    print(f"quality {args.quality}, texture {RES}x{RES}, blur radius {radius:.0%} of object size "
          f"(sigma = {sigma_world:.3f} world, {sigma_px:.1f} texels)")
    print(f"{len(P)} seam probe pairs, 3000 random surface points\n")
    truth_a = ground_truth(seam_pos(pos, uvA), src_pos, src_mask, sigma_world)
    truth_b = ground_truth(seam_pos(pos, uvB), src_pos, src_mask, sigma_world)
    print(f"{'':8} {'---- seam error ----':>22} {'-- interior error --':>22}")
    print(f"{'method':8} {'avg':>7} {'p95':>7} {'max':>6} {'avg':>7} {'p95':>7} {'max':>6}")
    for name, img in (("uv", uv_blur), ("world", world)):
        a = wsb.sample_nearest(img, uvA[:, 0], uvA[:, 1])
        b = wsb.sample_nearest(img, uvB[:, 0], uvB[:, 1])
        se = np.abs(np.concatenate([a - truth_a, b - truth_b]))
        r = wsb.sample_nearest(img, rand_uv[:, 0], rand_uv[:, 1])
        ie = np.abs(r - truth_rand)
        print(f"{name:8} {se.mean():7.3f} {np.percentile(se, 95):7.3f} {se.max():6.2f}"
              f" {ie.mean():7.3f} {np.percentile(ie, 95):7.3f} {ie.max():6.2f}")

    if args.images:
        from PIL import Image
        out = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
        os.makedirs(out, exist_ok=True)
        tmp = os.path.join(out, "_render.png")
        renders, flats = [], []
        for img in (mask, uv_blur, world):
            render(img, pos, uvm, tmp)
            renders.append(Image.open(tmp).copy())
            flats.append(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)))
        os.remove(tmp)
        for name, ims in (("comparison.png", renders), ("comparison_uv.png", flats)):
            w, h = ims[0].size
            sheet = Image.new("L", (w * 3, h))
            for i, im in enumerate(ims):
                sheet.paste(im, (i * w, 0))
            sheet.save(os.path.join(out, name))
        print(f"\nimages written to {os.path.normpath(out)}")

if __name__ == "__main__":
    main()
