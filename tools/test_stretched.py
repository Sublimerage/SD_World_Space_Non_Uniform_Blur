"""Regression check for stretched UV islands (fixed in v2.1, as in World Space Mask Blur v2.1):
a door frame with every face in an equal square UV island, so the posts' faces are stretched
about 13:1. Before v2.1, at 512^2 and 1K, samples whose neighbouring texels were farther apart
in 3D than a fixed reject distance got no area, no colour and no normal: the posts came out
unblurred.

The same door unwrapped in proportion (every face at one texel density) is the reference:
the blur of a 3D point shouldn't depend on how the surface is unwrapped. Both run through the
full-size graph at 512^2, then each texel of the square-island door is compared with the
nearest texel of the proportional one (same 3D point).

    python3 tools/test_stretched.py [v2.0.sbs]     (v2.0's file: also reports v2.0's numbers)

Lite is checked the same way (it keeps v2.1's 512 x 512 surface samples).
"""
import os
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)
import layout_mesh  # noqa: E402
import sbsinterp as SI  # noqa: E402
import build_wsnub as W  # noqa: E402
import bench_v2 as BV  # noqa: E402
import bench_total as BT  # noqa: E402

RES = 512
PARAMS = dict(intensity=10.0, samples=4, quality=4.0, dither=0.0, use_blur_map=False)


def colour(pos, uvm):
    """a 3D checker with colour bands (continuous across faces)"""
    f = np.floor(pos * 6.0 + 0.37).astype(int)
    chk = ((f[..., 0] + f[..., 1] + f[..., 2]) % 2).astype(np.float32)
    col = np.stack([chk, 0.3 + 0.5 * chk * (pos[..., 1] / 2.2), 1.0 - chk, np.ones_like(chk)], -1)
    col = col * (uvm[..., None] > 0.5)
    col[..., 3] = 1.0
    return col.astype(np.float32)


def run(path, pos, uvm):
    nm = {v: k for k, v in BV.names(path).items()}
    pos4 = np.concatenate([pos, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    col = colour(pos, uvm)
    r = SI.run_graph(path, BT.graph_info(path)[0], {'input': col, 'mesh_position': pos4, 'mesh_uv_mask': uvm.astype(np.float32),
                                        'blur_map': np.ones(uvm.shape, np.float32)}, dict(PARAMS),
                     keep=[nm['WSB_PackArea'], nm['WSB_PackPos']])
    pa, pp = r['_nodes'][nm['WSB_PackArea']], r['_nodes'][nm['WSB_PackPos']]
    usable = pp[..., 3] > 0.5
    zero = float((usable & (pa[..., 0] <= 0)).sum() / max(usable.sum(), 1))
    return r['output'][..., :3], col[..., :3], zero


def check(name, path):
    sq_pos, sq_uvm = layout_mesh.make_square_islands(RES)
    pc_pos, pc_uvm = layout_mesh.make_pieces(RES)
    t0 = time.time()
    sq, sq_in, zero = run(path, sq_pos, sq_uvm)
    pc, _, _ = run(path, pc_pos, pc_uvm)
    m = sq_uvm > 0.5
    pm = pc_uvm > 0.5
    _, idx = cKDTree(pc_pos[pm]).query(sq_pos[m])
    ref_ = pc[pm][idx]
    got = sq[m]
    d = np.abs(got - ref_).mean(-1)
    post = np.abs(sq_pos[m][:, 0]) > 0.5
    changed_ref = np.abs(ref_ - sq_in[m]).mean(-1) > 0.02
    unblurred = changed_ref & (np.abs(got - sq_in[m]).mean(-1) < 1e-4)
    print('  %-5s zero-area samples %5.1f %%   posts: diff to proportional unwrap mean %.4f, unblurred %5.1f %%   '
          'lintel: mean %.4f   (%.0fs)'
          % (name, 100 * zero, d[post].mean(), 100 * unblurred[post].mean(), d[~post].mean(), time.time() - t0),
          flush=True)
    return zero, d[post].mean(), unblurred[post].mean(), d[~post].mean()


def main():
    path = os.path.join(tempfile.mkdtemp(), 'v21.sbs')
    W.build(path)
    if len(sys.argv) > 1:
        check('v2.0', sys.argv[1])
    zero, post, unbl, lintel = check('v%s' % W.VERSION, path)
    ok = zero == 0.0 and unbl < 0.001 and post < 2.0 * lintel + 0.005
    lite = os.path.join(os.path.dirname(path), 'lite.sbs')
    W.build(lite, lite=True)
    zero, post, unbl, lintel = check('Lite', lite)
    ok = ok and zero == 0.0 and unbl < 0.001 and post < 2.0 * lintel + 0.005
    print('ALL PASS' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
