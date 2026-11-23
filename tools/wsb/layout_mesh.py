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


DOOR = [((0.15, 2.0, 0.15), (-0.6, 1.0, 0.0)), ((0.15, 2.0, 0.15), (0.6, 1.0, 0.0)),
        ((1.35, 0.15, 0.15), (0.0, 2.075, 0.0))]


def make_square_islands(res=512, pieces=None, cols=5, gap_px=4):
    """(position, uv mask) of boxes (per piece: size xyz, centre xyz) with every
    face in an equal square UV island, `cols` per row: a long thin face gets a
    strongly stretched island (the door frame's posts: about 13:1)."""
    pieces = pieces or DOOR
    faces = []
    for size, ctr in pieces:
        size, ctr = np.array(size, float), np.array(ctr, float)
        for axis in range(3):
            i, j = [a for a in range(3) if a != axis]
            for sign in (1, -1):
                o = ctr.copy()
                o[axis] += sign * size[axis] / 2
                o[i] -= size[i] / 2
                o[j] -= size[j] / 2
                U, V = np.zeros(3), np.zeros(3)
                U[i], V[j] = size[i], size[j]
                faces.append((o, U, V))
    cell = 1.0 / cols
    gap = gap_px / res
    islands = [((k % cols) * cell + gap / 2, (k // cols) * cell + gap / 2, cell - gap, cell - gap, 0)
               for k in range(len(faces))]
    demo_box.RES = res
    pos, uvm, _ = demo_box.bake_maps(faces, islands)
    return pos, uvm


def make_pieces(res=512, pieces=None, gap_px=4):
    """(position, uv mask) of boxes (per piece: size xyz, centre xyz), all faces
    shelf-packed into one UV square at one texel density, each face a
    rectangle in proportion (as an artist would unwrap a kit)."""
    pieces = pieces or DOOR
    faces = []
    for size, ctr in pieces:
        size, ctr = np.array(size, float), np.array(ctr, float)
        for axis in range(3):
            i, j = [a for a in range(3) if a != axis]
            for sign in (1, -1):
                o = ctr.copy()
                o[axis] += sign * size[axis] / 2
                o[i] -= size[i] / 2
                o[j] -= size[j] / 2
                U, V = np.zeros(3), np.zeros(3)
                U[i], V[j] = size[i], size[j]
                faces.append((o, U, V))
    gap = gap_px / res
    dims = []
    for _, U, V in faces:
        a, b = np.linalg.norm(U), np.linalg.norm(V)
        dims.append((b, a, 1) if b > a else (a, b, 0))           # long side along u
    order = sorted(range(len(faces)), key=lambda f: -dims[f][1])
    k = np.sqrt(0.7 / sum(d[0] * d[1] for d in dims))
    while True:
        islands, x, y, row_h, fits = [None] * len(faces), gap, gap, 0.0, True
        for f in order:
            w_, h_ = dims[f][0] * k, dims[f][1] * k
            if x + w_ > 1 - gap:
                x, y, row_h = gap, y + row_h + gap, 0.0
            islands[f] = (x, y, w_, h_, dims[f][2])
            x += w_ + gap
            row_h = max(row_h, h_)
            fits &= x <= 1 - gap + 1e-9
        if fits and y + row_h < 1 - gap:
            break
        k *= 0.97
    demo_box.RES = res
    pos, uvm, _ = demo_box.bake_maps(faces, islands)
    return pos, uvm
