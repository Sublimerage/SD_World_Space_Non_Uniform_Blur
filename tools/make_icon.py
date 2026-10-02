"""Generates the filter icons: the test sphere blurred by the filter itself.

  docs/icon_color.png / .svg  -> colour graph  (input: colour 3D checker)
  docs/icon_gray.png  / .svg  -> grayscale graph (input: its luminance)

The textures come from running the actual .sbs through the interpreter
(tools/sbsinterp.py) on the cube-mapped test sphere, with a blur map that is
black on the -X side (so that side stays sharp).  They are cached as
docs/icon_texture_{color,gray}.png; pass --rerun to recompute them (a few
minutes).  The sphere is then ray traced (orthographic, flat, anti-aliased) on
a transparent background, 128x128, and embedded in the .sbs by build_wsnub.py.

  python3 tools/make_icon.py [--rerun]
"""
import base64
import io
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, 'docs')
sys.path.insert(0, HERE)
import testmesh as T  # noqa: E402

SIZE = 128
SPHERE = 0.94          # sphere diameter as a fraction of the icon
VIEW = (-0.6, -0.4, -0.7)
PARAMS = dict(quality=1.0, intensity=30.0, use_blur_map=True, blur_map_from_input=False, dither=0.0)


def compute_texture(kind):
    import sbsinterp as SI
    pos, mask = T.make(shape='sphere')
    P = pos[mask > 0.5]
    mn, mx = P.min(0), P.max(0)
    pn = (pos - mn) / (mx - mn).max()
    pn4 = np.concatenate([pn, np.ones(pn.shape[:2] + (1,))], -1).astype(np.float32)
    col = T.pattern(pos)
    eff = T.effect(pos).astype(np.float32)
    sbs = os.path.join(ROOT, 'world_space_non_uniform_blur.sbs')
    if kind == 'color':
        src = np.concatenate([col, np.ones(col.shape[:2] + (1,))], -1).astype(np.float32)
        r = SI.run_graph(sbs, 'world_space_non_uniform_blur',
                         {'input': src, 'mesh_position': pn4, 'mesh_uv_mask': mask, 'blur_map': eff}, params=PARAMS)
        return r['output'][..., :3]
    src = (col @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float32)
    r = SI.run_graph(sbs, 'world_space_non_uniform_blur_grayscale',
                     {'input': src, 'mesh_position': pn4, 'mesh_uv_mask': mask, 'blur_map': eff}, params=PARAMS)
    return np.repeat(r['output'][..., None], 3, -1)


def texture(kind, rerun=False, from_npy=None):
    path = os.path.join(DOCS, 'icon_texture_%s.png' % kind)
    if from_npy is not None:
        t = np.load(from_npy)
        t = np.repeat(t[..., None], 3, -1) if t.ndim == 2 else t[..., :3]
        Image.fromarray((np.clip(t, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
    elif rerun or not os.path.exists(path):
        t = compute_texture(kind)
        Image.fromarray((np.clip(t, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
    return np.asarray(Image.open(path).convert('RGB'), np.float32) / 255.0


def sphere_uv(p, res=512):
    """inverse of testmesh.make: unit-sphere points -> texel coords (row, col)."""
    ax = np.argmax(np.abs(p), -1)
    rows = np.zeros(len(p))
    cols = np.zeros(len(p))
    tile = 128 * res // 512
    g = 4 * res // 512
    n = tile - 2 * g
    for name, (cx, cy, a, sg, rot, mir) in T.FACES.items():
        sel = (ax == a) & (np.sign(p[:, a]) == sg)
        if not sel.any():
            continue
        q = p[sel] / np.abs(p[sel, a:a + 1])
        o = [i for i in range(3) if i != a]
        A, B = q[:, o[0]].copy(), q[:, o[1]].copy()
        if mir:
            A = -A
        for _ in range(rot):
            A, B = B, -A
        ci = np.clip((A + 1) / 2 * n - 0.5, 0, n - 1)
        ri = np.clip((B + 1) / 2 * n - 0.5, 0, n - 1)
        cols[sel] = cx * tile + g + ci
        rows[sel] = cy * tile + g + ri
    return rows, cols


def render(tex, res=SIZE, ss=4):
    R = res * ss
    v = np.array(VIEW, float)
    v /= np.linalg.norm(v)
    up = np.array([0, 1.0, 0])
    r = np.cross(up, v)
    r /= np.linalg.norm(r)
    u = np.cross(v, r)
    c = ((np.arange(R) + 0.5) / R * 2 - 1) / SPHERE
    X, Y = np.meshgrid(c, -c)
    rr = X * X + Y * Y
    inside = rr < 1.0
    z = np.sqrt(np.clip(1 - rr, 0, 1))
    pts = X[inside, None] * r + Y[inside, None] * u + z[inside, None] * v
    rows, cols = sphere_uv(pts, tex.shape[0])
    rgb = np.zeros((R, R, 3))
    for k in range(3):
        rgb[inside, k] = map_coordinates(tex[..., k], [rows, cols], order=1, mode='nearest')
    a = inside.astype(float)
    # box-filter down: premultiplied colour and coverage
    rgb = (rgb * a[..., None]).reshape(res, ss, res, ss, 3).mean((1, 3))
    a = a.reshape(res, ss, res, ss).mean((1, 3))
    rgb = rgb / np.maximum(a[..., None], 1e-6)
    out = np.concatenate([rgb, a[..., None]], -1)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA')


def svg(img512):
    buf = io.BytesIO()
    img512.save(buf, 'PNG', optimize=True)
    return ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            'viewBox="0 0 %d %d" width="%d" height="%d"><image x="0" y="0" width="%d" height="%d" '
            'xlink:href="data:image/png;base64,%s"/></svg>'
            % ((SIZE,) * 6 + (base64.b64encode(buf.getvalue()).decode(),)))


if __name__ == '__main__':
    os.makedirs(DOCS, exist_ok=True)
    rerun = '--rerun' in sys.argv
    npys = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)     # e.g. color=out.npy gray=out_g.npy
    for kind in ('color', 'gray'):
        tex = texture(kind, rerun, npys.get(kind))
        render(tex).save(os.path.join(DOCS, 'icon_%s.png' % kind))
        with open(os.path.join(DOCS, 'icon_%s.svg' % kind), 'w') as f:
            f.write(svg(render(tex, 512, 2)))
    print('wrote docs/icon_color.png/.svg, docs/icon_gray.png/.svg')
