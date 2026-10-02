"""Runs world_space_non_uniform_blur.sbs through the numpy interpreter on
synthetic meshes (cube-mapped sphere / box with 6 UV islands, one mirrored)
and writes before/after renders.

  python3 tools/test_wsnub.py [shape] [out_prefix] [param=value ...]
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sbsinterp as SI
import testmesh as T

from build_wsnub import OUTPUT_NAME
SBS = os.path.join(os.path.dirname(HERE), OUTPUT_NAME)


def run(shape="sphere", params=None, res=512, effect=None, verbose=False, sbs=SBS, keep=None, targets=None):
    pos, mask = T.make(res=res, shape=shape)
    P = pos[mask > 0.5]
    mn, mx = P.min(0), P.max(0)
    pn = (pos - mn) / (mx - mn).max()                     # Painter-like [0,1] position map
    pn4 = np.concatenate([pn, np.ones(pn.shape[:2] + (1,))], -1).astype(np.float32)
    col = T.pattern(pos)
    src = np.concatenate([col, np.ones(col.shape[:2] + (1,))], -1).astype(np.float32)
    eff = T.effect(pos) if effect is None else effect(pos)
    t0 = time.time()
    r = SI.run_graph(sbs, 'world_space_non_uniform_blur',
                     {'input': src, 'mesh_position': pn4, 'mesh_uv_mask': mask, 'blur_map': eff.astype(np.float32)},
                     params=params or {}, verbose=verbose, keep=keep, targets=targets)
    if verbose:
        print('total %.1fs' % (time.time() - t0))
    return pos, mask, src, r


if __name__ == '__main__':
    shape = sys.argv[1] if len(sys.argv) > 1 else 'sphere'
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'out'
    prm = {'quality': 1.0}
    for a in sys.argv[3:]:
        k, v = a.split('=')
        prm[k] = (v == '1') if k in ('use_blur_map', 'blur_map_from_input', 'invert_blur_map') else float(v)
    pos, mask, src, r = run(shape, prm, verbose=True)
    out = r['output']
    np.save(prefix + '.npy', out)
    views = [(0.6, 0.5, 0.62), (-0.6, -0.4, -0.7)]
    img = np.concatenate([np.concatenate([T.render(pos, mask, src[..., :3], view=v) for v in views], 1),
                          np.concatenate([T.render(pos, mask, out[..., :3], view=v) for v in views], 1)], 0)
    T.save(img, prefix + '.png')
