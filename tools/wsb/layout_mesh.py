"""
A "kit layout" test scene: several copies of the demo box, scaled into
different shapes (a cube, a plank, a beam, a slab) and placed far apart, all
sharing one texture (each copy's UVs in one quadrant), like modular pieces
laid out in a Painter scene.

    import layout_mesh
    mask, pos, uvm = layout_mesh.make_layout(256, spread=12.0)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import demo_box  # noqa: E402

# per copy: (scale xyz, offset xyz as multiples of `spread`)
PIECES = [((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
          ((2.5, 0.12, 0.25), (1.0, 0.0, 0.1)),
          ((0.3, 1.6, 0.3), (0.2, 0.0, 1.0)),
          ((1.2, 0.2, 1.4), (1.3, 0.05, 0.9))]


def make_layout(res=256, spread=12.0, pieces=None):
    """(mask, position, uv mask) of res x res maps: the pieces, `spread` apart,
    each in its own tile of the UV square (2 x 2 tiles for up to 4 pieces, ...)."""
    pieces = pieces or PIECES
    tiles = int(np.ceil(np.sqrt(len(pieces))))
    h = res // tiles
    demo_box.RES = h
    faces = demo_box.make_faces()
    islands, _ = demo_box.layout_islands(faces)
    pos1, uvm1, _ = demo_box.bake_maps(faces, islands)
    pos = np.zeros((res, res, 3))
    uvm = np.zeros((res, res))
    for k, (scale, off) in enumerate(pieces):
        y0, x0 = (k // tiles) * h, (k % tiles) * h
        p = pos1 * np.array(scale) + np.array(off) * spread
        pos[y0:y0 + h, x0:x0 + h] = np.where(uvm1[..., None] > 0.5, p, 0.0)
        uvm[y0:y0 + h, x0:x0 + h] = uvm1
    # mask: a few 3D blobs, one per piece, so every piece has some edge to blur
    mask = np.zeros((res, res))
    for k, (scale, off) in enumerate(pieces):
        y0, x0 = (k // tiles) * h, (k % tiles) * h
        p = pos[y0:y0 + h, x0:x0 + h][uvm1 > 0.5]
        lo, hi = p.min(0), p.max(0)
        c = lo + (hi - lo) * np.array([0.7, 0.8, 0.3])
        r = 0.35 * (hi - lo).max()
        mask = np.maximum(mask, (np.linalg.norm(pos - c, axis=-1) < r).astype(float))
    return np.where(uvm > 0.5, mask, 0.0), pos, uvm
