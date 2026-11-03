"""Comparison image for the README: a plank from the 4-piece kit layout (pieces far apart in one
texture set), its top face seen from above. The mask (dots and a thin line), v1.0 and v2.0 at a
small Intensity. v1.0's voxels cover the whole layout, so its smallest blur washes the plank out.

    python3 tools/plank_compare.py --one <file.sbs> <intensity> <out.npy>     (one run)
    python3 tools/plank_compare.py --sheet <out.png> "<label>=<out.npy>" ...  (the image)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)
import sbsinterp as SI  # noqa: E402
import layout_mesh  # noqa: E402

RES = 512


def plank_mask(pos, uvm, tile):
    y0, x0, h = tile
    sel = np.zeros(uvm.shape, bool)
    sel[y0:y0 + h, x0:x0 + h] = uvm[y0:y0 + h, x0:x0 + h] > 0.5
    p = pos[sel]
    lo, hi = p.min(0), p.max(0)
    m = np.zeros(uvm.shape)
    q = pos
    L = hi[0] - lo[0]
    for k in range(9):                                   # dots along the plank, alternating sides
        c = np.array([lo[0] + L * (0.08 + 0.1 * k), hi[1], lo[2] + (hi[2] - lo[2]) * (0.35 if k % 2 else 0.65)])
        r = 0.035 + 0.012 * (k % 3)
        m = np.maximum(m, (np.linalg.norm(q - c, axis=-1) < r).astype(float))
    line = (np.abs(q[..., 2] - (lo[2] + 0.12 * (hi[2] - lo[2]))) < 0.012) & (q[..., 0] < lo[0] + 0.9 * L)
    m = np.maximum(m, line.astype(float))
    return np.where(sel, m, 0.0), sel, lo, hi


def top_view(img, pos, sel, lo, hi, w=480):
    """orthographic view from above of the plank's top face (texels at its max y), nearest texel"""
    from scipy.spatial import cKDTree
    top = sel & (pos[..., 1] > hi[1] - 1e-3)
    p = pos[top]
    v = img[top]
    h = int(round(w * (hi[2] - lo[2]) / (hi[0] - lo[0])))
    xs = lo[0] + (np.arange(w) + 0.5) / w * (hi[0] - lo[0])
    zs = lo[2] + (np.arange(h) + 0.5) / h * (hi[2] - lo[2])
    X, Z = np.meshgrid(xs, zs)
    _, idx = cKDTree(p[:, [0, 2]]).query(np.stack([X.ravel(), Z.ravel()], -1))
    return v[idx].reshape(h, w)


def scene():
    mask, pos, uvm = layout_mesh.make_layout(RES, 12.0)
    h = RES // 2
    m, sel, lo, hi = plank_mask(pos, uvm, (0, h, h))       # piece 1 (the plank) is in tile (row 0, col 1)
    return m, pos, uvm, sel, lo, hi


def run_one(path, intensity, out_npy):
    m, pos, uvm, sel, lo, hi = scene()
    col = np.stack([m, m, m, np.ones_like(m)], -1).astype(np.float32)
    pos4 = np.concatenate([pos, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    r = SI.run_graph(path, 'world_space_non_uniform_blur',
                     {'input': col, 'mesh_position': pos4, 'mesh_uv_mask': uvm.astype(np.float32),
                      'blur_map': np.ones_like(m, np.float32)},
                     dict(intensity=intensity, samples=4.0, quality=4.0, dither=0.0, use_blur_map=False))
    np.save(out_npy, r['output'][..., 0])


def sheet(out_png, runs):
    """runs: [(label, saved output .npy)]"""
    m, pos, uvm, sel, lo, hi = scene()
    rows = [('mask', top_view(m, pos, sel, lo, hi))] + [(name, top_view(np.load(f), pos, sel, lo, hi))
                                                        for name, f in runs]
    W = 520
    hh = rows[0][1].shape[0]
    img = Image.new('RGB', (W, (hh + 54) * len(rows) + 10), (34, 34, 34))
    d = ImageDraw.Draw(img)
    for k, (name, im) in enumerate(rows):
        y = 10 + k * (hh + 54)
        d.text((10, y), name, fill=(220, 220, 220))
        img.paste(Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)), (20, y + 24))
    img.save(out_png)
    print('wrote', out_png)


if __name__ == '__main__':
    if sys.argv[1] == '--one':
        run_one(sys.argv[2], float(sys.argv[3]), sys.argv[4])
    else:
        sheet(sys.argv[2], [(a.split('=')[0], a.split('=')[1]) for a in sys.argv[3:]])
