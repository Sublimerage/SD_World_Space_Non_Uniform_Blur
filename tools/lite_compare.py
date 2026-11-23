"""World Space Non-Uniform Blur Lite (tools/lite_proto.py) against v2.0, on the same scenes
and settings as tools/bench_v2.py (512 maps, work scaled to 2048^2), plus a thin plank:
dots painted on its top face only, to see how much reaches the bottom face.

    python3 tools/lite_compare.py --v2 <v2.0.sbs> <scene> <out_dir>    (one v2.0 run, cached)
    python3 tools/lite_compare.py --report <out_dir>                    (Lite runs, numbers, sheet)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)
import bench_v2 as BV  # noqa: E402
import layout_mesh  # noqa: E402
import lite_proto as L  # noqa: E402

RES = BV.RES
TARGET = BV.TARGET
PLANK = [((2.5, 0.12, 0.25), (0.0, 0.0, 0.0))]
SCENES = {'sphere': 10.0, 'layout': 10.0, 'plane': 10.0, 'plank10': 10.0, 'plank30': 30.0}
SIZES = (1.0, 0.5, 0.25)


def scene(name):
    """(colour RGBA, position, uv mask, blur map, piece id per texel, normals or None)"""
    pid = np.zeros((RES, RES), int)
    nrm = None
    if name.startswith('plank'):
        _, pos, uvm = layout_mesh.make_layout(RES, 12.0, PLANK)
        sel = uvm > 0.5
        lo, hi = pos[sel].min(0), pos[sel].max(0)
        top = sel & (pos[..., 1] > hi[1] - 1e-4)
        m = np.zeros((RES, RES))
        for k in range(9):
            c = np.array([lo[0] + (hi[0] - lo[0]) * (0.08 + 0.1 * k), hi[1],
                          lo[2] + (hi[2] - lo[2]) * (0.35 if k % 2 else 0.65)])
            m = np.maximum(m, (np.linalg.norm(pos - c, axis=-1) < 0.035 + 0.012 * (k % 3)).astype(float))
        m = np.where(top, m, 0.0)
        col = np.stack([m, m, m, np.ones_like(m)], -1) * sel[..., None]
        col[..., 3] = 1.0
        return col.astype(np.float32), pos.astype(np.float32), uvm.astype(np.float32), \
            np.ones((RES, RES), np.float32), pid, None
    col, pos, uvm, bmap = BV.scene(name)
    if name == 'layout':
        h = RES // 2
        yy, xx = np.mgrid[0:RES, 0:RES]
        pid = (yy // h) * 2 + xx // h
    if name == 'plane':
        nrm = np.zeros(pos.shape[:2] + (3,))
        nrm[..., 1] = 1.0
    return col, pos[..., :3], uvm, bmap, pid, nrm


def v2_params(name):
    return dict(BV.PARAMS, intensity=SCENES[name])


def run_v2(path, name, out_dir):
    import sbsinterp as SI
    col, pos, uvm, bmap, _, _ = scene(name)
    pos4 = np.concatenate([pos, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    SI.STATS.clear()
    r = SI.run_graph(path, 'world_space_non_uniform_blur',
                     {'input': col, 'mesh_position': pos4, 'mesh_uv_mask': uvm, 'blur_map': bmap}, v2_params(name))
    nm = BV.names(path)
    cost = {'mesh': 0.0, 'stroke': 0.0}
    for uid, st in SI.STATS.items():
        scale = (TARGET / RES) ** 2 if st['rel'] else 1.0
        cost[BV.group(nm.get(uid, '?'), False)] += st['reads'] * scale
    np.savez(os.path.join(out_dir, 'v2_%s.npz' % name), output=r['output'], stroke=cost['stroke'], mesh=cost['mesh'])
    print(name, 'v2.0 per stroke %.3f G reads, once per mesh %.3f G reads' % (cost['stroke'] / 1e9, cost['mesh'] / 1e9))


def tile(img, valid):
    x = np.clip(img[..., :3] if img.ndim == 3 else np.repeat(img[..., None], 3, -1), 0, 1)
    x = np.where(valid[..., None], x, 0.13)
    return (x * 255).astype(np.uint8)


def report(out_dir):
    rows, lines = [], []
    lines.append('| Scene | v2.0 G reads / stroke | Lite size | Lite G reads / stroke (est.) | x cheaper | '
                 'mean diff to v2.0 | blur change (v2.0) | coverage |')
    lines.append('|---|---|---|---|---|---|---|---|')
    bleed = []
    for name in SCENES:
        f = os.path.join(out_dir, 'v2_%s.npz' % name)
        if not os.path.exists(f):
            print('missing', f)
            continue
        v2 = np.load(f)
        col, pos, uvm, bmap, pid, nrm = scene(name)
        valid = uvm > 0.5
        ref = v2['output'][..., :3].astype(np.float64)
        change = np.abs(ref - col[..., :3])[valid].mean()
        row = [('input', col[..., :3]), ('v2.0', ref)]
        prm = dict(intensity=SCENES[name], samples=4, use_blur_map=True)
        for size in SIZES:
            st = {}
            out = L.lite(col[..., :3], pos, uvm, bmap, dict(prm, size=size), pieces=pid, stats=st, nrm=nrm)
            d = np.abs(out - ref)[valid].mean()
            est = st['reads'] * (TARGET / RES) ** 2
            lines.append('| %s | %.3f | %.2f (%d px at 512) | %.3f | %.1fx | %.4f | %.4f | %.1f%% |' % (
                name, v2['stroke'] / 1e9, size, st['S'], est / 1e9, v2['stroke'] / est, d, change,
                100 * st['coverage']))
            row.append(('Lite %.2f' % size, out))
            if name.startswith('plank'):
                lo, hi = pos[valid].min(0), pos[valid].max(0)
                bot = valid & (pos[..., 1] < lo[1] + 1e-4)
                top = valid & (pos[..., 1] > hi[1] - 1e-4)
                bleed.append('| %s | Lite %.2f | %.4f | %.4f | %.4f |' % (
                    name, size, out[bot][:, 0].mean(), out[bot][:, 0].max(), np.abs(out - col[..., :3])[top].mean()))
        if name.startswith('plank'):
            lo, hi = pos[valid].min(0), pos[valid].max(0)
            bot = valid & (pos[..., 1] < lo[1] + 1e-4)
            top = valid & (pos[..., 1] > hi[1] - 1e-4)
            bleed.append('| %s | v2.0 | %.4f | %.4f | %.4f |' % (
                name, ref[bot][:, 0].mean(), ref[bot][:, 0].max(), np.abs(ref - col[..., :3])[top].mean()))
        rows.append((name, [(lab, tile(im, valid)) for lab, im in row] +
                     [('|Lite 0.50 - v2.0| x4', tile(np.abs(row[3][1] - ref).max(-1) * 4, valid))]))
    print('\n'.join(lines))
    print('\n| Scene | Version | bottom face mean | bottom face max | top face change |\n|---|---|---|---|---|')
    print('\n'.join(bleed))
    # sheet: one row per scene, 256 px tiles
    n = max(len(r[1]) for r in rows)
    T = 256
    img = Image.new('RGB', (n * (T + 8) + 90, len(rows) * (T + 26) + 6), (34, 34, 34))
    d = ImageDraw.Draw(img)
    for i, (name, tiles) in enumerate(rows):
        y = 6 + i * (T + 26)
        d.text((6, y + T // 2), name, fill=(220, 220, 220))
        for j, (lab, t) in enumerate(tiles):
            x = 90 + j * (T + 8)
            img.paste(Image.fromarray(t).resize((T, T), Image.NEAREST), (x, y + 18))
            d.text((x, y + 2), lab, fill=(220, 220, 220))
    path = os.path.join(out_dir, 'lite_vs_v2.png')
    img.save(path)
    print('sheet:', path)


if __name__ == '__main__':
    if sys.argv[1] == '--v2':
        run_v2(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        report(sys.argv[2])
