"""Total work per evaluation (mesh-only passes + per-stroke passes), per node.

Painter appears to recompute the whole filter on every slider tick or paint stroke, so what
counts is the total. Runs a package through tools/sbsinterp.py at 512^2 maps and full table
sizes, on the bench scenes (sphere, 4-piece kit layout, flat plane, thin plank) with several
settings, and saves the output and the work of every node. Work is counted as executed:
texture reads and function-node evaluations, loops at their real length, both sides of every
If/Else. Passes at texture size are scaled to 2048^2 (x16) and 4096^2 (x64) from the 512^2 run;
the table passes don't depend on the texture size.

    python3 tools/bench_total.py run <file.sbs> <scene> <out dir> [setting ...]
    python3 tools/bench_total.py report <dir of variant A> <dir of variant B> ...
    python3 tools/bench_total.py quality <dir of v2.1's runs> <dir> ...    (differences to v2.1)
    python3 tools/bench_total.py nodes <out dir> <scene> <setting>         (per-node table)

Settings: default (Quality 4, Intensity 10, Samples 4, Softness 0.25), s1, s16 (Samples),
soft0 (Softness 0), q3, q2 (Quality), nomap (Use Blur Map off), i0, i4, i30 (Intensity).
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)
import sbsinterp as SI  # noqa: E402
import bench_v2 as BV  # noqa: E402
import layout_mesh  # noqa: E402

RES = BV.RES
SCENES = ('sphere', 'layout', 'plane', 'plank')
PLANK = [((2.5, 0.12, 0.25), (0.0, 0.0, 0.0))]
SETTINGS = {
    'default': {},
    's1': dict(samples=1.0),
    's16': dict(samples=16.0),
    'soft0': dict(softness=0.0),
    'q3': dict(quality=3.0),
    'q2': dict(quality=2.0),
    'nomap': dict(use_blur_map=False),
    'i0': dict(intensity=0.0),
    'i4': dict(intensity=4.0),
    'i30': dict(intensity=30.0),
}
BYTES = {0: 1, 1: 2, 2: 2, 3: 4}      # per channel, by output format


def scene(name):
    """(colour RGBA, position RGBA, uv mask, blur map)"""
    if name != 'plank':
        return BV.scene(name)
    # a thin plank, 2.5 x 0.12 x 0.25, with dots painted on its top face only
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
    col = np.stack([m, m, m, np.ones_like(m)], -1).astype(np.float32)
    pos4 = np.concatenate([pos, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    return col, pos4, uvm.astype(np.float32), np.ones((RES, RES), np.float32)


def graph_info(path):
    """(graph id, {node uid: name}, {node uid: bytes per pixel})"""
    graph = ET.parse(path).getroot().find('content/graph')
    nm = {o.find('GUIDependency').get('v').split('?')[1]: o.find('GUIName').get('v') for o in graph.iter('GUIObject')}
    bpp = {}
    for n in graph.find('compNodes'):
        impl = n.find('compImplementation')[0]
        if impl.tag != 'compFilter':
            continue
        P = {p.find('name').get('v'): p for p in impl.findall('parameters/parameter')}
        fmt = int(P['format'].find('paramValue')[0].get('v')) if 'format' in P else 3
        col = P['colorswitch'].find('paramValue')[0].get('v') == '1'
        bpp[n.find('uid').get('v')] = BYTES[fmt] * (4 if col else 1)
    return graph.find('identifier').get('v'), nm, bpp


def run(path, sc, out_dir, settings):
    os.makedirs(out_dir, exist_ok=True)
    gid, nm, bpp = graph_info(path)
    col, pos, uvm, bmap = scene(sc)
    imgs = {'input': col, 'mesh_position': pos, 'mesh_uv_mask': uvm, 'blur_map': bmap}
    cache, cache_q = None, None
    for s in settings:
        prm = dict(BV.PARAMS, **SETTINGS[s])
        if cache is None or prm['quality'] != cache_q:
            SI.STATS.clear()
            cache = None
        t0 = time.time()
        r = SI.run_graph(path, gid, imgs, prm, cache=cache)
        # the mesh-only passes don't depend on the settings (Quality aside): reuse them
        cache = {u: v for u, v in r['_all'].items() if u in nm and BV.group(nm[u], False) == 'mesh'}
        cache_q = prm['quality']
        r = {'output': r['output']}          # drop the per-stroke tables before the next run
        nodes = [dict(name=nm.get(u, '?'), reads=st['reads'], nodes=st['nodes'], w=st['size'][0], h=st['size'][1],
                      rel=st['rel'], bpp=bpp.get(u, 16)) for u, st in SI.STATS.items()]
        np.save(os.path.join(out_dir, '%s_%s.npy' % (sc, s)), r['output'])
        with open(os.path.join(out_dir, '%s_%s.json' % (sc, s)), 'w') as f:
            json.dump(dict(scene=sc, setting=s, params=prm, nodes=nodes, seconds=time.time() - t0), f)
        print(sc, s, summary(nodes, 2048), '%.0fs' % (time.time() - t0), flush=True)


def totals(nodes, target):
    """{'mesh': [reads, node evals], 'stroke': [...]} and the bytes of all the node outputs"""
    k = (target / RES) ** 2
    cost = {'mesh': [0.0, 0.0], 'stroke': [0.0, 0.0]}
    mem = 0.0
    for n in nodes:
        sc = k if n['rel'] else 1.0
        g = BV.group(n['name'], False)
        cost[g][0] += n['reads'] * sc
        cost[g][1] += n['nodes'] * sc
        mem += n['w'] * n['h'] * sc * n['bpp']
    return cost, mem


def summary(nodes, target):
    c, mem = totals(nodes, target)
    return '%d: mesh %.2f + stroke %.2f = %.2f G reads, %.1f + %.1f = %.1f G node evals, images %.0f MB' % (
        target, c['mesh'][0] / 1e9, c['stroke'][0] / 1e9, (c['mesh'][0] + c['stroke'][0]) / 1e9,
        c['mesh'][1] / 1e9, c['stroke'][1] / 1e9, (c['mesh'][1] + c['stroke'][1]) / 1e9, mem / 2 ** 20)


def load(d, sc, s):
    f = os.path.join(d, '%s_%s.json' % (sc, s))
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh)


def report(dirs):
    for target in (2048, 4096):
        print('\n%d x %d: total = mesh + stroke, G reads / G node evals' % (target, target))
        print('| Scene | Setting | ' + ' | '.join(os.path.basename(d.rstrip('/')) for d in dirs) + ' |')
        print('|---|---|' + '---|' * len(dirs))
        for sc in SCENES:
            for s in SETTINGS:
                cells, base = [], None
                for d in dirs:
                    j = load(d, sc, s)
                    if j is None:
                        cells.append('–')
                        continue
                    c, mem = totals(j['nodes'], target)
                    tr, tn = c['mesh'][0] + c['stroke'][0], c['mesh'][1] + c['stroke'][1]
                    cell = '%.2f + %.2f = **%.2f** / %.0f' % (c['mesh'][0] / 1e9, c['stroke'][0] / 1e9, tr / 1e9,
                                                             tn / 1e9)
                    if base is None:
                        base = (tr, tn)
                    else:
                        cell += ' (%.1fx / %.1fx less)' % (base[0] / tr, base[1] / tn)
                    cells.append(cell)
                if any(c != '–' for c in cells):
                    print('| %s | %s | %s |' % (sc, s, ' | '.join(cells)))
    print('\nimage memory (all node outputs together), MB at 2048 / 4096')
    for d in dirs:
        j = load(d, 'sphere', 'default')
        if j:
            print('  %s: %.0f / %.0f' % (os.path.basename(d.rstrip('/')), totals(j['nodes'], 2048)[1] / 2 ** 20,
                                       totals(j['nodes'], 4096)[1] / 2 ** 20))


def quality(ref_dir, dirs):
    """mean abs difference over the UV islands to the outputs in ref_dir (v2.1), and on the
    plane to the 2D Non-Uniform Blur; on the plank, what reaches the unpainted bottom face"""
    for sc in SCENES:
        col, pos, uvm, bmap = scene(sc)
        m = uvm > 0.5
        for s in SETTINGS:
            f = os.path.join(ref_dir, '%s_%s.npy' % (sc, s))
            if not os.path.exists(f):
                continue
            a = np.load(f)[..., :3]
            prm = dict(BV.PARAMS, **SETTINGS[s])
            line = '%-7s %-8s v2.1 changes the input by %.4f' % (sc, s, np.abs(a - col[..., :3])[m].mean())
            want = None
            if sc == 'plane' and prm['use_blur_map'] and prm['intensity'] > 0:
                want = BV.ref2d(col, prm['intensity'], int(prm['samples']), 5)
                line += ' (%.4f from the 2D filter)' % np.abs(a - want)[40:-40, 40:-40].mean()
            if sc == 'plank':
                bot = m & (pos[..., 1] < pos[m][:, 1].min() + 1e-4)
                line += ' (bottom face max %.4f)' % a[bot].max()
            for d in dirs:
                g = os.path.join(d, '%s_%s.npy' % (sc, s))
                if not os.path.exists(g):
                    continue
                b = np.load(g)[..., :3]
                line += ' | %s: %.4f from v2.1, max %.3f' % (os.path.basename(d.rstrip('/')), np.abs(a - b)[m].mean(),
                                                           np.abs(a - b)[m].max())
                if want is not None:
                    line += ', %.4f from the 2D filter' % np.abs(b - want)[40:-40, 40:-40].mean()
                if sc == 'plank':
                    line += ', bottom face max %.4f' % b[bot].max()
            print(line)


def node_table(d, sc, s, target=2048):
    j = load(d, sc, s)
    k = (target / RES) ** 2
    agg = {}
    for n in j['nodes']:
        key = n['name'].split(' ')[0]
        a = agg.setdefault(key, [0.0, 0.0, 0])
        f = k if n['rel'] else 1.0
        a[0] += n['reads'] * f
        a[1] += n['nodes'] * f
        a[2] += 1
    for key, a in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print('%-20s x%-3d %8.3f G reads %8.2f G node evals  %s' % (key, a[2], a[0] / 1e9, a[1] / 1e9,
                                                                 BV.group(key, False)))
    print(summary(j['nodes'], target))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:] or ['default'])
    elif cmd == 'report':
        report(sys.argv[2:])
    elif cmd == 'quality':
        quality(sys.argv[2], sys.argv[3:])
    else:
        node_table(*sys.argv[2:5])
