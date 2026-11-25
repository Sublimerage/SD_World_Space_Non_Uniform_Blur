"""The speed-ups (build(fast=True), what ships: the faster passes and skipping the work that
isn't used) must change no output pixel, and v2.1's stretched-island fix must change nothing
on meshes without stretched islands.

Builds v2.1 with and without the speed-ups, v2.1 with v2.0's tricubic final lookup
(lookup='cubic'), and v2.0 as released (from git, when available), at reduced table sizes,
and runs them on many cases: box, cube-mapped sphere,
the box at 1/1000 scale, kit layouts (4 pieces, 6 in a row, a column along Z, a stack
along Y), every Quality, Intensity from small to large, voxels from 1x to over 4x a sort
cell, and each option (Anisotropy, Blades, Softness, Samples, Angle, Reference Axis, blur
map off / from the input / inverted, Dither), plus a grayscale input (as in a mask). Every output pixel
must be exactly equal (max |fast - plain| == 0), and v2.1 with v2.0's lookup must equal v2.0
(only the final lookup changed elsewhere; its mean difference is printed). The fast - plain
difference maps are saved in a sheet.

The shipping build skips inactive pass slots, mip levels no tap reads, the setup of voxels and
texels that aren't blurred, the Blur Map read when it isn't used and the dither at Dither 0.
A second list of cases aims at that: Samples 1 / 2 / 4 / 8 / 16 with Softness 0 / 0.25 / 1,
Softness values where only some mip levels are read, Intensity small enough that fewer passes
run than Samples asks for, Intensity 0, the blur map off / from the input / from the Blur Map
input. On those, the plain graph (no speed-up, nothing skipped) and the shipping one must again
be exactly equal. Lite is the same graph without Softness and its mip pyramids: built at the
same sizes with v2.1's final lookup, it must equal v2.1 at Softness 0 exactly.

    python3 tools/test_fast_exact.py [out_dir]
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)
import world_space_blur as ref  # noqa: E402
import demo_box  # noqa: E402
import layout_mesh  # noqa: E402
import testmesh  # noqa: E402
import sbsinterp as SI  # noqa: E402
import build_wsnub as W  # noqa: E402
import test_v2 as TV  # noqa: E402

SMALL = TV.SMALL
# a bigger sort table with the same voxel budgets: voxels 2-4x a sort cell, wider NearX/Y/Z reach
BIG_SORT = (32, 128, 128, 32, 32, 32, 4, 4, {1: 8, 2: 12, 3: 16}, 32, 4, 8)


RELEASE = 'dde603a'        # v2.0 as released (its own builder and vendored tools/wsb)


def released_sbs(config, out):
    """build v2.0 as released at `config`, in a separate process (its vendored modules differ
    from today's), or None if git doesn't have it"""
    root = tempfile.mkdtemp()
    try:
        for f in ('tools/build_wsnub.py', 'tools/wsb/build_sbs.py', 'tools/wsb/world_space_blur.py'):
            src = subprocess.run(['git', '-C', os.path.dirname(HERE), 'show', '%s:%s' % (RELEASE, f)],
                                 capture_output=True, text=True, check=True).stdout
            os.makedirs(os.path.dirname(os.path.join(root, f)), exist_ok=True)
            with open(os.path.join(root, f), 'w') as fh:
                fh.write(src)
    except Exception:
        return None
    code = ('import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import world_space_blur as ref; '
            'ref.configure(*%r); import build_wsnub as W; W.build(%r)'
            % (os.path.join(root, 'tools', 'wsb'), os.path.join(root, 'tools'), tuple(config), out))
    subprocess.run([sys.executable, '-c', code], check=True, cwd=root)
    return out


def build_all(config):
    ref.configure(*config)
    d = tempfile.mkdtemp()
    out = {}
    for name, fast, lookup in (('fast', True, 'quadratic'), ('plain', False, 'quadratic'),
                               ('cubic', False, 'cubic')):
        out[name] = os.path.join(d, name + '.sbs')
        W.build(out[name], fast=fast, lookup=lookup)
    if released_sbs(config, os.path.join(d, 'v20.sbs')) is not None:
        out['v2.0'] = os.path.join(d, 'v20.sbs')
    # Lite at the same sizes, with v2.1's final lookup: v2.1 at Softness 0 without the mip pyramids
    out['lite'] = os.path.join(d, 'lite.sbs')
    W.build(out['lite'], lite=True, config=config)
    return out


def scenes():
    demo_box.RES = 64
    faces = demo_box.make_faces()
    islands, _ = demo_box.layout_islands(faces)
    pos, uvm, _ = demo_box.bake_maps(faces, islands)
    box = (None, pos.astype(np.float32), uvm.astype(np.float32))
    sp, sm = testmesh.make(res=128, shape='sphere')
    sphere = (None, sp.astype(np.float32), sm.astype(np.float32))
    tiny = (None, (pos * 0.001 + 0.002).astype(np.float32), uvm.astype(np.float32))
    return dict(box=box, sphere=sphere, box_tiny=tiny,
                layout=layout_mesh.make_layout(96, 12.0),
                row=layout_mesh.make_layout(96, 12.0, TV.ROW),
                column=layout_mesh.make_layout(96, 12.0, TV.COLUMN),
                stack=layout_mesh.make_layout(96, 12.0, TV.STACK))


def run(path, maps, params, gray=False, graph=None):
    mask, pos, uvm = maps
    col, bmap = TV.colour_input(pos, uvm, mask)
    pos4 = np.concatenate([pos, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    prm = dict(dither=0.0, blur_map_from_input=False)
    prm.update(params)
    imgs = {'input': col[..., 0] if gray else col, 'mesh_position': pos4,
            'mesh_uv_mask': uvm.astype(np.float32), 'blur_map': bmap}
    out = SI.run_graph(path, graph or W.GRAPH_ID, imgs, prm)['output']   # a grayscale input, as in a mask: read as grey
    return out[..., 0] if gray else out


P = dict(intensity=24.0, samples=4.0, quality=3.0)
CASES = [
    ('box', P), ('box', dict(P, quality=1.0)), ('box', dict(P, quality=2.0)), ('box', dict(P, quality=1.6)),
    ('box', dict(P, intensity=2.0)), ('box', dict(P, intensity=8.0)), ('box', dict(P, intensity=48.0)),
    ('box', dict(P, anisotropy=0.7, angle=0.3)), ('box', dict(P, blades=3.0)), ('box', dict(P, blades=9.0)),
    ('box', dict(P, softness=0.0)), ('box', dict(P, softness=1.0)), ('box', dict(P, samples=1.0)),
    ('box', dict(P, samples=16.0)), ('box', dict(P, axis_x=1.0, axis_y=0.0)), ('box', dict(P, dither=1.0)),
    ('box', dict(P, use_blur_map=False)), ('box', dict(P, blur_map_from_input=True)),
    ('box', dict(P, blur_map_from_input=True, invert_blur_map=True)),
    ('sphere', P), ('sphere', dict(P, quality=1.0, softness=0.6, anisotropy=0.5)),
    ('box_tiny', P), ('box_tiny', dict(P, quality=1.0)),
    ('layout', P), ('layout', dict(P, quality=1.0)), ('layout', dict(P, intensity=6.0, samples=8.0)),
    ('layout', dict(P, use_blur_map=False, dither=1.0)),
    ('row', P), ('column', P), ('stack', P), ('stack', dict(P, quality=1.0, blades=6.0)),
]
# what the shipping build skips (fast and plain only; Lite too where Softness is 0)
CASES_SKIP = (
    [('box', dict(P, samples=float(n), softness=sf)) for n in (1, 2, 4, 8, 16) for sf in (0.0, 0.25, 1.0)
     if (n, sf) != (4, 0.25)] +
    # mip levels: none read (the radius is too small for level 1), some, all
    [('box', dict(P, softness=0.02)), ('box', dict(P, softness=0.1)), ('box', dict(P, softness=0.5)),
     ('box', dict(P, softness=1.0, intensity=48.0, samples=2.0)),
     ('sphere', dict(P, softness=0.4, intensity=40.0, samples=8.0)),
     # fewer passes than Samples: N = min(Samples, ceil(Intensity * pi))
     ('box', dict(P, intensity=4.0, samples=16.0)), ('box', dict(P, intensity=1.5, samples=8.0, softness=1.0)),
     ('box', dict(P, intensity=0.5, samples=4.0)), ('box', dict(P, intensity=0.3, samples=16.0, softness=0.0)),
     ('layout', dict(P, intensity=4.0, samples=16.0, softness=0.0)),
     ('box', dict(P, intensity=0.0)), ('layout', dict(P, intensity=0.0, dither=1.0)),
     # blur map off / from the input / from the Blur Map input, with other settings
     ('box', dict(P, use_blur_map=False, samples=1.0, softness=0.0)),
     ('box', dict(P, use_blur_map=False, samples=16.0, softness=1.0)),
     ('sphere', dict(P, use_blur_map=False, samples=8.0)),
     ('layout', dict(P, use_blur_map=False, softness=0.0, samples=2.0)),
     ('sphere', dict(P, blur_map_from_input=True, softness=0.0, dither=2.0)),
     ('stack', dict(P, blur_map_from_input=True, invert_blur_map=True, samples=8.0, softness=1.0)),
     ('row', dict(P, softness=0.0, samples=16.0, intensity=6.0)), ('column', dict(P, softness=1.0, samples=1.0)),
     ('box_tiny', dict(P, softness=0.0, samples=8.0)),
     ('box', dict(P, softness=0.0, gray=True)), ('layout', dict(P, use_blur_map=False, samples=8.0, gray=True))])
CASES_BIG_SORT = [('box', P), ('box', dict(P, quality=1.0)), ('layout', dict(P, quality=1.0)),
                  ('sphere', dict(P, quality=2.0, softness=0.0)), ('stack', dict(P, quality=1.0))]


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    S = scenes()
    diffs, ok, n = [], True, 0
    lite_n = 0
    for cname, config, cases in (('small', SMALL, CASES), ('big sort table', BIG_SORT, CASES_BIG_SORT),
                                 ('skipping', SMALL, CASES_SKIP)):
        paths = build_all(config)
        cases = cases + ([('box', dict(P, gray=True)), ('layout', dict(P, gray=True))] if cname == 'small' else [])
        for scene, prm in cases:
            prm = dict(prm)
            gray = prm.pop('gray', False)
            t0 = time.time()
            which = ('fast', 'plain') if cname == 'skipping' else ('fast', 'plain', 'cubic', 'v2.0')
            got = {k: run(paths[k], S[scene], prm, gray) for k in which if k in paths}
            d = np.abs(got['fast'] - got['plain'])
            same = d.max() == 0.0                       # (a NaN anywhere fails this too)
            rel = ''
            if prm.get('softness') == 0.0:
                lite = run(paths['lite'], S[scene], prm, gray, graph=W.LITE_GRAPH_ID)
                eq = np.abs(lite - got['fast']).max() == 0.0
                rel += '  Lite == v2.1 at Softness 0: %s' % ('yes' if eq else 'NO')
                same &= eq
                lite_n += 1
            if 'v2.0' in got:
                fix_only = np.abs(got['cubic'] - got['v2.0']).max()
                rel += '  v2.1 with v2.0 lookup == v2.0: %s  v2.1 vs v2.0 mean %.1e' % (
                    'yes' if fix_only == 0.0 else 'NO (%.1e)' % fix_only, np.abs(got['fast'] - got['v2.0']).mean())
                same &= fix_only == 0.0
            changed = np.abs(got['fast'] - run(paths['fast'], S[scene], dict(prm, intensity=0.0), gray)).max()
            n += 1
            ok &= same
            desc = ', '.join('%s=%g' % (k, v) for k, v in prm.items() if P.get(k) != v)
            print('  %-14s %-6s %-48s fast == plain: %s (blurred: %.2f)%s  %.0fs'
                  % (cname, scene + (' gray' if gray else ''), desc or 'defaults', 'yes' if same else 'NO',
                     changed, rel, time.time() - t0), flush=True)
            dm = d if d.ndim == 2 else d.max(-1)
            diffs.append(np.pad(dm, 2, constant_values=0.25))
    # sheet of the difference maps (black = equal; grey frame around each)
    hmax = max(x.shape[0] for x in diffs)
    wmax = max(x.shape[1] for x in diffs)
    tiles = [np.pad(x, ((0, hmax - x.shape[0]), (0, wmax - x.shape[1])), constant_values=0.0) for x in diffs]
    rows = [np.concatenate(tiles[i:i + 10] + [np.zeros_like(tiles[0])] * (10 - len(tiles[i:i + 10])), 1)
            for i in range(0, len(tiles), 10)]
    Image.fromarray((np.clip(np.concatenate(rows, 0), 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(out_dir, 'fast_vs_plain_diff.png'))
    print('%d cases (%d of them also with Lite), %s' % (n, lite_n,
                                                        'ALL EQUAL (0 differing pixels)' if ok else 'SOME DIFFER'))
    print('difference maps:', os.path.join(out_dir, 'fast_vs_plain_diff.png'))
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()) else 1)
