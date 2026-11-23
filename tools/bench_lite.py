"""Cost of World Space Non-Uniform Blur Lite against v2.0, measured like tools/bench_v2.py:
512^2 maps through tools/sbsinterp.py, texture reads and node evaluations actually executed
(loops at their real length, both sides of every If/Else), texture-size passes scaled to
2048^2. The view tables are a fixed size (2048 x 2048 fine, 2048 x 1024 per stroke), so
their cost doesn't scale with the texture.

    python3 tools/bench_lite.py <lite.sbs> <scene> <out.npz>     (one run)
    python3 tools/bench_lite.py --report <dir> <v2 dir>           (table; v2 dir from lite_compare --v2)
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'wsb_lite'))
import sbsinterp as SI  # noqa: E402
import bench_v2 as BV  # noqa: E402
import lite_compare as C  # noqa: E402

GROUPS = (('stroke', 'WSNBL_'), ('mesh', 'WSB_'))


def group(name):
    for g, prefix in GROUPS:
        if name.startswith(prefix):
            return g
    return 'mesh'


def run_one(path, scene, out):
    col, pos, uvm, bmap, _, _ = C.scene(scene)
    pos4 = np.concatenate([pos, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    SI.STATS.clear()
    t0 = time.time()
    r = SI.run_graph(path, 'world_space_non_uniform_blur_lite',
                     {'input': col, 'mesh_position': pos4, 'mesh_uv_mask': uvm, 'blur_map': bmap},
                     dict(BV.PARAMS, intensity=C.SCENES[scene]))
    nm = BV.names(path)
    cost = {'mesh': [0.0, 0.0], 'stroke': [0.0, 0.0]}
    per = {}
    for uid, st in SI.STATS.items():
        n = nm.get(uid, '?')
        scale = (BV.TARGET / BV.RES) ** 2 if st['rel'] else 1.0
        g = group(n)
        cost[g][0] += st['reads'] * scale
        cost[g][1] += st['nodes'] * scale
        key = n.split(' ')[0].rstrip('0123456789')
        per[key] = per.get(key, 0.0) + st['reads'] * scale
    np.savez(out, output=r['output'], cost=np.array([cost['mesh'], cost['stroke']]), seconds=time.time() - t0,
             per_keys=np.array(list(per.keys())), per_vals=np.array(list(per.values())))
    print(out, scene, 'per stroke %.3f G reads %.1f G nodes, once per mesh %.3f G reads %.1f G nodes  (%.0fs)'
          % (cost['stroke'][0] / 1e9, cost['stroke'][1] / 1e9, cost['mesh'][0] / 1e9, cost['mesh'][1] / 1e9,
             time.time() - t0), flush=True)


def report(d, v2dir):
    print('| Scene | v2.0 reads / stroke | Lite reads / stroke | v2.0 node evals / stroke | Lite node evals / stroke '
          '| v2.0 once per mesh | Lite once per mesh |')
    print('|---|---|---|---|---|---|---|')
    for scene in C.SCENES:
        f = os.path.join(d, 'lite_%s.npz' % scene)
        g = os.path.join(v2dir, 'bench_%s_v2.npz' % scene)
        if not os.path.exists(f):
            continue
        lc = np.load(f)['cost']
        v2 = np.load(g)['cost'] if os.path.exists(g) else None
        fmt = lambda x: '%.3f G' % (x / 1e9)  # noqa: E731
        print('| %s | %s | %s (%s) | %s | %s | %s | %s |' % (
            scene, fmt(v2[1][0]) if v2 is not None else '-', fmt(lc[1][0]),
            '%.1fx less' % (v2[1][0] / lc[1][0]) if v2 is not None else '', '%.1f G' % (v2[1][1] / 1e9) if v2 is not None
            else '-', '%.1f G' % (lc[1][1] / 1e9), fmt(v2[0][0]) if v2 is not None else '-', fmt(lc[0][0])))


if __name__ == '__main__':
    if sys.argv[1] == '--report':
        report(sys.argv[2], sys.argv[3])
    else:
        run_one(sys.argv[1], sys.argv[2], sys.argv[3])
