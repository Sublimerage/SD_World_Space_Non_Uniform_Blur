"""World Space Non-Uniform Blur Lite: the graph (tools/build_wsnub_lite.py, run in
tools/sbsinterp.py) against its NumPy spec (tools/lite_ref.py), at reduced table sizes
(the core's test sizes: v2.0 tables as tools/test_v2.py, views 64 x 64, level <= 4).

Cases: box, sphere, a kit layout, stacked slabs, a door frame with stretched UV islands; Intensity small to large (view levels 1-4,
with 64, 256 and 1024 px view tables),
Samples, Blades, Anisotropy and Angle, Reference Axis, each blur map mode, Quality, Dither,
and the grayscale graph. Each must match to 2e-3 (float32 graph vs float64 spec) and blur
something.

    python3 tools/test_lite.py
"""
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'wsb_lite'))
import world_space_blur as ref  # noqa: E402
import world_space_blur_lite as lref  # noqa: E402
import build_wsnub_lite as BWL  # noqa: E402
import lite_ref as R  # noqa: E402
import sbsinterp as SI  # noqa: E402
sys.path.insert(2, os.path.join(HERE, 'wsb'))
import demo_box  # noqa: E402
import layout_mesh  # noqa: E402
import testmesh  # noqa: E402

SMALL = (32, 64, 64, 32, 32, 32, 4, 4, {1: 8, 2: 12, 3: 16}, 32, 4, 8)
LITE_SMALL = (64, 64, 4)
STACK = [((1.0, 0.2, 1.0), (0.0, 0.6 * i, 0.0)) for i in range(4)]
TOL = 2e-3


def build(lite_config=LITE_SMALL):
    ref.configure(*SMALL)
    lref.configure(*lite_config)
    path = os.path.join(tempfile.mkdtemp(), 'lite.sbs')
    BWL.build(path)
    return path


def colour_input(pos, uvm):
    """a hard-edged 3D colour pattern and a blur map ramping along x"""
    lo, hi = pos[uvm > 0.5].min(0), pos[uvm > 0.5].max(0)
    q = (pos - lo) / np.maximum((hi - lo).max(), 1e-30)
    f = np.floor(q * 6.0 + 0.37).astype(int)
    chk = ((f[..., 0] + f[..., 1] + f[..., 2]) % 2).astype(np.float32)
    col = np.stack([chk, 0.4 + 0.5 * chk * q[..., 1], 1.0 - chk, np.ones_like(chk)], -1)
    col = col * (uvm[..., None] > 0.5)
    col[..., 3] = 1.0
    return col.astype(np.float32), np.clip(q[..., 0] * 1.5, 0, 1).astype(np.float32)


def scenes():
    demo_box.RES = 64
    faces = demo_box.make_faces()
    islands, _ = demo_box.layout_islands(faces)
    pos, uvm, _ = demo_box.bake_maps(faces, islands)
    sp, sm = testmesh.make(res=128, shape='sphere')     # 1 px gutter: islands don't touch the border
    _, lp, lu = layout_mesh.make_layout(96, 12.0)
    _, kp, ku = layout_mesh.make_layout(96, 1.0, STACK)     # slabs 0.6 apart: they hide each other's faces
    dp, du = layout_mesh.make_square_islands(96)              # door frame, posts stretched ~13:1 in UV
    return dict(box=(pos, uvm), sphere=(sp, sm), layout=(lp, lu), stack=(kp, ku), door=(dp, du))


def run(path, maps, prm, gray=False, cache={}):
    pos, uvm = maps
    col, bmap = colour_input(pos, uvm)
    pos4 = np.concatenate([pos[..., :3], np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    params = dict(R.DEFAULTS, **prm)
    gid = BWL.GRAPH_ID + ('_grayscale' if gray else '')
    imgs = {'input': col[..., 0] if gray else col, 'mesh_position': pos4, 'mesh_uv_mask': uvm.astype(np.float32),
            'blur_map': bmap}
    got = SI.run_graph(path, gid, imgs, params)['output']
    key = (id(maps), params['quality'], lref.VW)
    if key not in cache:
        cache[key] = R.prepare(pos, uvm, params['quality'])
    st = {}
    if gray:
        g4 = np.repeat(col[..., :1], 4, -1)
        g4[..., 3] = 1.0
        want = R.nub_lite(g4, pos, uvm, bmap, params, cache[key], st)[..., 0]
        got = got if got.ndim == 2 else got[..., 0]
        src = col[..., 0]
    else:
        want = R.nub_lite(col, pos, uvm, bmap, params, cache[key], st)
        got, want, src = got[..., :3], want[..., :3], col[..., :3]
    return got, want, src, st


P = dict(intensity=12.0, samples=4, dither=0.0, blur_map_from_input=False)
CASES = [
    ('box', P), ('box', dict(P, intensity=2.0)), ('box', dict(P, intensity=30.0)), ('box', dict(P, intensity=50.0)),
    ('box', dict(P, samples=1)), ('box', dict(P, samples=16)), ('box', dict(P, blades=3)), ('box', dict(P, blades=9)),
    ('box', dict(P, anisotropy=0.8, angle=0.15)), ('box', dict(P, axis_x=1.0, axis_y=0.0)),
    ('box', dict(P, use_blur_map=False)), ('box', dict(P, blur_map_from_input=True)),
    ('box', dict(P, blur_map_from_input=True, invert_blur_map=True)), ('box', dict(P, quality=2)),
    ('sphere', P), ('sphere', dict(P, anisotropy=0.6, angle=0.3, blades=6)), ('sphere', dict(P, intensity=40.0)),
    ('layout', P), ('layout', dict(P, intensity=4.0, samples=8)), ('stack', P), ('door', P),
]
# bigger views (256 x 256, levels up to 5): Intensity picks levels 1..4, fills average 2..16 px blocks
LITE_BIG = (256, 256, 5)
CASES_BIG = [('box', dict(P, intensity=i)) for i in (3.0, 8.0, 20.0, 50.0)] + [
    ('sphere', dict(P, intensity=25.0, anisotropy=0.5, angle=0.2)), ('layout', dict(P, intensity=30.0))]
# full-width views (1024 x 1024, levels up to 7): levels 3 and 4
LITE_FULL = (1024, 1024, 7)
CASES_FULL = [('box', dict(P, intensity=20.0)), ('box', dict(P, intensity=50.0, anisotropy=0.7, angle=0.4))]


def main():
    S = scenes()
    ok_all = True
    runs = [(LITE_SMALL, [(s, p, False) for s, p in CASES] + [('box', P, True), ('layout', P, True)]),
            (LITE_BIG, [(s, p, False) for s, p in CASES_BIG]),
            (LITE_FULL, [(s, p, False) for s, p in CASES_FULL])]
    for config, cases in runs:
        path = build(config)
        print('views %d x %d, max level %d' % config)
        for scene, prm, gray in cases:
            t0 = time.time()
            got, want, src, st = run(path, S[scene], prm, gray)
            m = S[scene][1] > 0.5
            err = np.abs(got - want)
            changed = np.abs(want - src)[m].mean()
            ok = err.max() < TOL and changed > 0.0005
            ok_all &= ok
            desc = ', '.join('%s=%g' % (k, v) for k, v in prm.items() if P.get(k) != v) or 'defaults'
            print('  %-7s %-5s %-44s max diff %.1e  mean %.1e  (level %d, blurred %.3f, uncovered %.3f)  %s  %.0fs'
                  % (scene, 'gray' if gray else '', desc, err.max(), err[m].mean(), st['L'], changed, st['uncovered'],
                     'ok' if ok else 'FAIL', time.time() - t0), flush=True)
    print('ALL PASS' if ok_all else 'SOME FAILED')
    return ok_all


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
