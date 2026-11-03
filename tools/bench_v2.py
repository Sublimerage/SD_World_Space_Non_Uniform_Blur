"""Quality and cost of v2.0 against v1.0 (full sizes, Quality 4, through tools/sbsinterp.py).

Scenes: the cube-mapped sphere (one object), the 4-piece kit layout (pieces far apart),
and a flat plane compared with the 2D Non-Uniform Blur. For each version it reports the
finest voxel size, the work per paint stroke and once per mesh (texture reads and node
evaluations actually executed, loops at their real length, both sides of every If/Else),
with the full-resolution passes scaled to a 2048^2 texture.

    python3 tools/bench_v2.py <v1.sbs> <v2.sbs> <scene> <out.npz>      (one run)
    python3 tools/bench_v2.py --report <out dir>                        (table from the runs)
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)
import sbsinterp as SI  # noqa: E402
import testmesh as T  # noqa: E402
import layout_mesh  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

RES = 512
TARGET = 2048
PARAMS = dict(intensity=10.0, samples=4.0, quality=4.0, dither=0.0, use_blur_map=True, blur_map_from_input=False)


def scene(name):
    if name == 'sphere':
        pos, uvm = T.make(res=RES, shape='sphere')
        col = T.pattern(pos)
        bmap = T.effect(pos)
    elif name == 'layout':
        mask, pos, uvm = layout_mesh.make_layout(RES, 12.0)
        m = mask.astype(np.float32)
        col = np.stack([m, 0.3 + 0.4 * m, 1.0 - m], -1)
        bmap = np.ones_like(m)
    elif name == 'plane':
        u = (np.arange(RES) + 0.5) / RES
        U, V = np.meshgrid(u, u)
        pos = np.stack([U, np.zeros_like(U), V], -1)
        uvm = np.ones((RES, RES))
        chk = (np.floor(U * 8) + np.floor(V * 8)) % 2
        ramp = np.clip((U - 0.1) / 0.8, 0, 1)
        col = np.stack([chk, 0.3 + 0.7 * chk * ramp, (1 - chk) * 0.6], -1) * (0.25 + 0.75 * ramp[..., None])
        for (y, x) in [(100, 100), (100, 400), (400, 256), (256, 256)]:
            col[y - 3:y + 3, x - 3:x + 3] = 1
        bmap = np.clip(col @ np.array([0.2126, 0.7152, 0.0722]), 0, 1)
    col4 = np.concatenate([col, np.ones(col.shape[:2] + (1,))], -1).astype(np.float32)
    pos4 = np.concatenate([pos, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    return col4, pos4, uvm.astype(np.float32), bmap.astype(np.float32)


def names(path):
    graph = ET.parse(path).getroot().find('content/graph')
    return {o.find('GUIDependency').get('v').split('?')[1]: o.find('GUIName').get('v') for o in graph.iter('GUIObject')}


def group(name, v1):
    """what makes a pass rerun: 'mesh' (mesh / Quality), 'stroke' (input, blur sliders)"""
    if v1:
        stroke = ('WSNB_Values', 'WSNB_Splat', 'WSNB_Frame', 'WSNB_Mip', 'WSNB_Pass', 'WSNB_Resolve')
    else:
        stroke = ('WSNB_PackVals', 'WSNB_Splat', 'WSNB_Voxels', 'WSNB_Radius', 'WSNB_Mip1', 'WSNB_Mip2', 'WSNB_Mip3',
                  'WSNB_Mip4', 'WSNB_Pass', 'WSNB_Resolve')
    return 'stroke' if name.startswith(stroke) else 'mesh'


def finest_voxel(path, res, v1):
    nm = {v: k for k, v in names(path).items()}
    if v1:
        g = res['_all'][nm['WSNB_Grid']]
        return float(g[8, 4, 3])
    return float(res['_all'][nm['WSB_VoxFit']][8, 4, 0])


def run_one(path, v1, sc, out):
    col, pos, uvm, bmap = scene(sc)
    SI.STATS.clear()
    t0 = time.time()
    r = SI.run_graph(path, 'world_space_non_uniform_blur',
                     {'input': col, 'mesh_position': pos, 'mesh_uv_mask': uvm, 'blur_map': bmap}, dict(PARAMS))
    nm = names(path)
    cost = {'mesh': [0.0, 0.0], 'stroke': [0.0, 0.0]}
    for uid, st in SI.STATS.items():
        n = nm.get(uid, '?')
        scale = (TARGET / RES) ** 2 if st['rel'] else 1.0
        g = group(n, v1)
        cost[g][0] += st['reads'] * scale
        cost[g][1] += st['nodes'] * scale
    np.savez(out, output=r['output'], h=finest_voxel(path, r, v1), cost=np.array([cost['mesh'], cost['stroke']]),
             seconds=time.time() - t0)
    print(out, 'h %.4g' % finest_voxel(path, r, v1), cost, '%.0fs' % (time.time() - t0))


def ref2d(img, I, S, Bl):
    from scipy.ndimage import map_coordinates
    C = np.sqrt(-np.log(0.001))
    Ns = int(min(S, np.ceil(I * np.pi)))
    eff = np.clip(img[..., :3] @ np.array([0.2126, 0.7152, 0.0722]), 0, 1)
    out = img[..., :3].astype(np.float64).copy()
    yy, xx = np.mgrid[0:RES, 0:RES].astype(float)
    for k in range(Ns):
        r = I / 256 * RES * np.exp(-2 * k * C / Ns) * eff
        ir = (1 / S) / (k + 1)
        acc = out.copy()
        for j in range(1, Bl + 1):
            th = 2 * np.pi * ir + j * 2 * np.pi / Bl
            acc += np.stack([map_coordinates(out[..., c], [yy + np.sin(th) * r, xx + np.cos(th) * r], order=1,
                                             mode='nearest') for c in range(3)], -1)
        out = acc / (Bl + 1)
    return out


def report(d):
    rows = []
    for sc in ('sphere', 'layout', 'plane'):
        res = {}
        for v in ('v1', 'v2'):
            f = os.path.join(d, '%s_%s.npz' % (sc, v))
            if os.path.exists(f):
                res[v] = np.load(f)
        if len(res) < 2:
            continue
        line = '%-7s finest voxel v1 %.4g  v2 %.4g  (%.1fx finer)' % (sc, res['v1']['h'], res['v2']['h'],
                                                                        res['v1']['h'] / res['v2']['h'])
        for v in ('v1', 'v2'):
            c = res[v]['cost']
            line += '\n         %s per stroke: %.2f G reads, %.1f G nodes   once per mesh: %.2f G reads, %.1f G nodes' \
                % (v, c[1][0] / 1e9, c[1][1] / 1e9, c[0][0] / 1e9, c[0][1] / 1e9)
        if sc == 'plane':
            col, pos, uvm, bmap = scene('plane')
            want = ref2d(col, PARAMS['intensity'], int(PARAMS['samples']), 5)
            for v in ('v1', 'v2'):
                got = res[v]['output'][..., :3]
                line += '\n         %s vs 2D Non-Uniform Blur: mean abs diff %.4f' % (
                    v, np.abs(got - want)[40:-40, 40:-40].mean())
        rows.append(line)
    print('\n'.join(rows))


if __name__ == '__main__':
    if sys.argv[1] == '--report':
        report(sys.argv[2])
    else:
        path, v1, sc, out = sys.argv[1], sys.argv[2] == 'v1', sys.argv[3], sys.argv[4]
        run_one(path, v1, sc, out)
