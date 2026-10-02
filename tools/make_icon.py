"""
Icon for World Space Non Uniform Blur, in the style of World Space Mask Blur
(see docs/icon-style-guide.md in sublimerage/my-project).

A neutral grey cube-mapped sphere seen at the corner where three UV islands
meet, their seams drawn as fine dashed lines. A few white spots (the mask) sit
on an arc across the seams. The filter's own Blur Map is a smooth left -> right
ramp, so the spots go from sharp to fully bokeh-blurred along the arc, by the
real graph: the grayscale graph of world_space_non_uniform_blur_v<VERSION>.sbs
run through tools/sbsinterp.py on the test sphere. Grayscale, soft key light,
transparent background.

    python3 tools/make_icon.py                 # docs/icon/world_space_non_uniform_blur{,_512}.png + preview.png
    python3 tools/make_icon.py --options       # computes every OPTIONS entry, writes a comparison sheet
"""

import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import sbsinterp as SI  # noqa: E402
from build_wsnub import OUTPUT_NAME, GRAPH_ID  # noqa: E402

# cube-mapped unit sphere, 6 UV islands in 128 px tiles of a 512 map, 4 px gutter (one island mirrored)
FACES = {  # name: (tile col, row, axis, sign, rot90 count, mirrored)
    '+X': (0, 0, 0, +1, 0, False), '-X': (1, 0, 0, -1, 1, False), '+Y': (2, 0, 1, +1, 2, False),
    '-Y': (0, 1, 1, -1, 3, False), '+Z': (1, 1, 2, +1, 0, True), '-Z': (2, 1, 2, -1, 1, False)}

VIEW = np.array([-1.0, -0.82, -1.0]) / np.linalg.norm([-1.0, -0.82, -1.0])   # towards the camera
CORNER = -np.ones(3) / np.sqrt(3)       # where the -X, -Y and -Z islands meet (the seams' "Y")
TEX = 1024                              # texture size of the test sphere
SPHERE = 0.94                           # sphere diameter / icon size
OUT_DIR = os.path.join(ROOT, "docs", "icon")
ICON_NAME = "world_space_non_uniform_blur"

# the story: spots on an arc across the "Y"; the Blur Map ramps from 0 (left) to 1 (right)
OPTIONS = {
    # the mask is the test sphere's 3D checker (continuous across the seams), drawn in two close greys;
    # blur 'blob': the Blur Map is a soft round area (the checker melts only there),
    # blur 'ramp': the Blur Map ramps from 0 (left) to 1 (right)
    'checker_blob': dict(shape='checker', blur='blob', params=dict(samples=16.0, blades=5.0, softness=0.25, intensity=14.0)),
    'checker_ramp': dict(shape='checker', blur='ramp', params=dict(samples=16.0, blades=5.0, softness=0.25, intensity=5.0)),
}
# render styles: checker greys (mask 0 / mask 1) and seam dashes (grey, width in px at the final size)
STYLES = {
    'dark_seams': dict(dark=0.38, light=0.63, seam=0.08, seam_w=1.6),
    'white_seams': dict(dark=0.36, light=0.60, seam=0.97, seam_w=1.6),
}
CHOSEN = ('checker_ramp', 'white_seams')
BLOB = (0.36, 0.02)        # blur area centre in view coordinates (right, up)
BLOB_R = (0.10, 0.78)      # blur map: 1 inside the first chord distance, 0 beyond the second
COMMON = dict(quality=4.0, use_blur_map=True, blur_map_from_input=False, invert_blur_map=False, dither=0.0)


def basis():
    v = VIEW
    r = np.cross([0, 1.0, 0], v); r /= np.linalg.norm(r)
    return v, r, np.cross(v, r)


def make_sphere(res=512, tile=128, gutter=4):
    pos = np.zeros((res, res, 3), np.float32)
    mask = np.zeros((res, res), np.float32)
    for name, (cx, cy, ax, sg, rot, mir) in FACES.items():
        s = tile * res // 512; g = gutter * res // 512; x0, y0 = cx * s + g, cy * s + g; n = s - 2 * g
        a = (np.arange(n) + 0.5) / n * 2 - 1; A, B = np.meshgrid(a, a)
        for _ in range(rot): A, B = -B, A
        if mir: A = -A
        p = np.zeros((n, n, 3)); o = [i for i in range(3) if i != ax]
        p[..., ax] = sg; p[..., o[0]] = A; p[..., o[1]] = B
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        pos[y0:y0 + n, x0:x0 + n] = p; mask[y0:y0 + n, x0:x0 + n] = 1
    idx = ndimage.distance_transform_edt(mask == 0, return_distances=False, return_indices=True)
    return pos[idx[0], idx[1]], mask


def sphere_uv(p, res=512):
    ax = np.argmax(np.abs(p), -1); rows = np.zeros(len(p)); cols = np.zeros(len(p))
    tile = 128 * res // 512; g = 4 * res // 512; n = tile - 2 * g
    for name, (cx, cy, a, sg, rot, mir) in FACES.items():
        sel = (ax == a) & (np.sign(p[:, a]) == sg)
        if not sel.any(): continue
        q = p[sel] / np.abs(p[sel, a:a + 1]); o = [i for i in range(3) if i != a]
        A, B = q[:, o[0]].copy(), q[:, o[1]].copy()
        if mir: A = -A
        for _ in range(rot): A, B = B, -A
        cols[sel] = cx * tile + g + np.clip((A + 1) / 2 * n - 0.5, 0, n - 1)
        rows[sel] = cy * tile + g + np.clip((B + 1) / 2 * n - 0.5, 0, n - 1)
    return rows, cols


def _unused_spot_centres(n):
    """n points on an arc across the seams' "Y", left to right in the view."""
    v, r, u = basis()
    s = np.linspace(0, 1, n)
    c = v[None] + (-0.70 + 1.40 * s)[:, None] * r[None] + (0.20 * np.sin(np.pi * 1.2 * s + 0.4) - 0.06)[:, None] * u[None]
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def pattern(pos, shape, period):
    """fine mask pattern defined in 3D (continuous across the seams), anti-aliased."""
    if shape == 'checker':                              # same 3D checker as the earlier test renders
        f = np.floor(pos * 3.0 + 100.37).astype(int)
        return ((f[..., 0] + f[..., 1] + f[..., 2]) % 2).astype(np.float64)
    v, r, u = basis()
    a = (pos @ (r * 0.8 + u * 0.6)) / period          # diagonal in the view
    if shape == 'stripes':
        d = np.abs(a - np.floor(a) - 0.5)               # 0 at the stripe centre, 0.5 between
        return smoothstep(0.24, 0.20, d)
    b = (pos @ (-r * 0.6 + u * 0.8)) / period
    da, db = a - np.floor(a) - 0.5, b - np.floor(b) - 0.5
    return smoothstep(0.30, 0.25, np.sqrt(da * da + db * db))


def blur_area(pos):
    v, r, u = basis()
    c = v + BLOB[0] * r + BLOB[1] * u
    c /= np.linalg.norm(c)
    return smoothstep(BLOB_R[1], BLOB_R[0], np.linalg.norm(pos - c, axis=-1))


def effect_texture(opt):
    """the mask blurred by the real filter (grayscale graph of the .sbs through tools/sbsinterp.py)."""
    pos, mask = make_sphere(TEX)
    P = pos[mask > 0.5]; mn, mx = P.min(0), P.max(0)
    position_map = (pos - mn) / (mx - mn).max()                                   # Painter-style [0,1]
    pos4 = np.concatenate([position_map, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    m = pattern(pos, opt['shape'], opt.get('period', 0.2)).astype(np.float32)
    v, r, u = basis()
    bm = (blur_area(pos) if opt['blur'] == 'blob' else np.clip((pos @ r + 0.70) / 1.55, 0, 1)).astype(np.float32)
    prm = dict(COMMON, **opt['params'])
    t0 = time.time()
    res = SI.run_graph(os.path.join(ROOT, OUTPUT_NAME), GRAPH_ID + '_grayscale',
                       {'input': m, 'mesh_position': pos4, 'mesh_uv_mask': mask, 'blur_map': bm}, params=prm)
    print('filter ran in %.0fs' % (time.time() - t0))
    return res['output']


def render(tex, res=128, ss=4, style=None):
    st = STYLES[style or CHOSEN[1]]
    R = res * ss; v, r, u = basis()
    c = ((np.arange(R) + 0.5) / R * 2 - 1) / SPHERE; X, Y = np.meshgrid(c, -c)
    rr = X * X + Y * Y; inside = rr < 1; z = np.sqrt(np.clip(1 - rr, 0, 1))
    pts = X[inside, None] * r + Y[inside, None] * u + z[inside, None] * v
    rows, cols = sphere_uv(pts, tex.shape[0])
    m = map_coordinates(tex, [rows, cols], order=1, mode='nearest')
    L = -0.35 * r + 0.55 * u + 0.75 * v; L /= np.linalg.norm(L)
    g = (st['dark'] + (st['light'] - st['dark']) * m) * (0.72 + 0.28 * np.clip(pts @ L, 0, 1))   # gently shaded
    # UV seams: where the cube face changes, dashed by the distance from the corner
    axis = np.argmax(np.abs(pts), -1)
    face = np.full((R, R), -1); face[inside] = axis * 2 + (pts[np.arange(len(pts)), axis] > 0)
    edge = np.zeros((R, R), bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        sh = np.roll(np.roll(face, dy, 0), dx, 1)
        edge |= (face != sh) & (face >= 0) & (sh >= 0)
    edge = ndimage.binary_dilation(edge, iterations=max(1, int(round(st['seam_w'] * R / res / 2 - 0.5))))
    ang = np.zeros((R, R)); ang[inside] = np.arccos(np.clip(pts @ CORNER, -1, 1))
    seam = (edge & inside & (np.sin(ang * 38.0) > -0.2))[inside]
    g = np.where(seam, st['seam'], g)
    img = np.zeros((R, R)); img[inside] = np.clip(g, 0, 1)
    a = inside.astype(float)
    img = (img * a).reshape(res, ss, res, ss).mean((1, 3)); a = a.reshape(res, ss, res, ss).mean((1, 3))
    out = np.stack([img / np.maximum(a, 1e-6)] * 3 + [a], -1)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA')


def sheet(icons_512, icons_128):
    """each icon at 256, 128 and 32 px on #2B2B2E, one row per icon."""
    w = 256 + 128 + 32 + 4 * 24
    out = Image.new('RGBA', (w, 304 * len(icons_512)), (0x2B, 0x2B, 0x2E, 255))
    for k, (big, ico) in enumerate(zip(icons_512, icons_128)):
        x = 24
        for im in (big.resize((256, 256), Image.LANCZOS), ico, ico.resize((32, 32), Image.LANCZOS)):
            out.paste(im, (x, 304 * k + 24 + (256 - im.size[1]) // 2), im)
            x += im.size[0] + 24
    return out


def cached_texture(name):
    path = os.path.join(OUT_DIR, 'texture_%s.npy' % name)
    if not os.path.exists(path):
        os.makedirs(OUT_DIR, exist_ok=True)
        np.save(path, effect_texture(OPTIONS[name]))
    return np.load(path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if '--options' in sys.argv:
        combos = [(t, st) for t in OPTIONS for st in STYLES]
        texs = {t: cached_texture(t) for t in OPTIONS}
        sheet([render(texs[t], 512, style=st) for t, st in combos],
              [render(texs[t], 128, style=st) for t, st in combos]).save(os.path.join(OUT_DIR, 'options.png'))
        print('wrote docs/icon/options.png:', ', '.join('%s/%s' % c for c in combos))
    else:
        tex = cached_texture(CHOSEN[0])
        ico, big = render(tex, 128), render(tex, 512)
        ico.save(os.path.join(OUT_DIR, ICON_NAME + ".png"))
        big.save(os.path.join(OUT_DIR, ICON_NAME + "_512.png"))
        sheet([big], [ico]).save(os.path.join(OUT_DIR, "preview.png"))
        print("wrote docs/icon/%s.png" % ICON_NAME)
