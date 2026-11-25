"""Tests for world_space_non_uniform_blur (v2.1) and its Lite: runs the generated .sbs through
tools/sbsinterp.py at reduced table sizes (like World Space Mask Blur's
test_sbs.py) on the demo box, kit layouts of pieces placed far apart (4 pieces;
6 in a row, more slabs than the small tables hold; a column along Z; a stack
along Y), and the box scaled to 1/1000.

Checks:
  * the result is finite, and blurred where the blur map asks for it;
  * pieces in different boxes don't interact: changing the input on one piece
    leaves every other piece's result unchanged;
  * the mesh's size and units don't matter (1/1000 scale gives the same result);
  * the per-box mip pyramid equals a direct [1 3 3 1] / 8 downsample of each box's
    voxels, and every mip entry maps back to the right box; in the shipping graph the
    levels no tap reads stay empty;
  * work that isn't used isn't done: inactive pass slots read their input and nothing
    else, their mips read nothing, texels that keep the input read 3 texels;
  * Lite: blurred, pieces don't interact, size and units don't matter.

    python3 tools/test_v2.py
"""
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)
import world_space_blur as ref  # noqa: E402
import demo_box  # noqa: E402
import layout_mesh  # noqa: E402
import sbsinterp as SI  # noqa: E402
import build_wsnub as W  # noqa: E402

# PMAX, AW, AH, CW, CH, DMAX, S_MAX, R_MAX, {level: voxel budget n^3}, frame grid, reads per patch axis, area grid
SMALL = (32, 64, 64, 32, 32, 32, 4, 4, {1: 8, 2: 12, 3: 16}, 32, 4, 8)

ROW = [((0.3, 0.3, 0.4), (1.5 * i, 0.0, 0.0)) for i in range(6)]
COLUMN = [((0.6, 0.3, 1.0), (0.0, 0.0, 0.4 * i)) for i in range(3)]
STACK = [((1.0, 0.2, 1.0), (0.0, 0.6 * i, 0.0)) for i in range(4)]


def build(config, **kw):
    ref.configure(*config)
    path = os.path.join(tempfile.mkdtemp(), 'wsnub.sbs')
    W.build(path, **({'config': config} if kw.get('lite') else {}), **kw)
    return path


def node_names(path):
    """{node name: [uids]} of the package's graph"""
    import xml.etree.ElementTree as ET
    names = {}
    for o in ET.parse(path).getroot().find('content/graph').iter('GUIObject'):
        names.setdefault(o.find('GUIName').get('v'), []).append(o.find('GUIDependency').get('v').split('?')[1])
    return names


def colour_input(pos, uvm, mask=None):
    """a hard-edged 3D colour pattern (continuous across seams), and a blur map ramping along x.
    With `mask` (the layout scene's per-piece blobs), the colour follows the mask instead."""
    if mask is not None:
        m = mask.astype(np.float32)
        col = np.stack([m, 0.3 + 0.4 * m, 1.0 - m, np.ones_like(m)], -1)
        return (col * (uvm[..., None] > 0.5)).astype(np.float32), np.ones_like(m)
    lo, hi = pos[uvm > 0.5].min(0), pos[uvm > 0.5].max(0)
    q = (pos - lo) / np.maximum((hi - lo).max(), 1e-30)
    f = np.floor(q * 6.0 + 0.37).astype(int)
    chk = ((f[..., 0] + f[..., 1] + f[..., 2]) % 2).astype(np.float32)
    col = np.stack([chk, 0.4 + 0.5 * chk * q[..., 1], 1.0 - chk, np.ones_like(chk)], -1)
    bmap = np.clip(q[..., 0] * 1.5, 0, 1).astype(np.float32)
    return (col * (uvm[..., None] > 0.5)).astype(np.float32), bmap


def run(path, col, pos, uvm, bmap, params, keep=None):
    import xml.etree.ElementTree as ET
    pos4 = np.concatenate([pos, np.ones(pos.shape[:2] + (1,))], -1).astype(np.float32)
    prm = dict(dither=0.0, blur_map_from_input=False)
    prm.update(params)
    gid = ET.parse(path).getroot().find('content/graph/identifier').get('v')      # v2.1 or Lite
    r = SI.run_graph(path, gid, {'input': col, 'mesh_position': pos4, 'mesh_uv_mask': uvm.astype(np.float32),
                                 'blur_map': bmap}, prm, keep=keep)
    return r['output'], r


def piece_tiles(n, res):
    tiles = int(np.ceil(np.sqrt(n)))
    h = res // tiles
    return [((k // tiles) * h, (k % tiles) * h, h) for k in range(n)]


def report(name, ok, extra=''):
    print('  %-44s %s  %s' % (name, 'ok' if ok else 'FAIL', extra), flush=True)
    return ok


def check_blur(name, path, maps, params):
    mask, pos, uvm = maps
    col, bmap = colour_input(pos, uvm, mask)
    t0 = time.time()
    out, _ = run(path, col, pos, uvm, bmap, params)
    inside = uvm > 0.5
    finite = np.isfinite(out).all()
    changed = np.abs(out[..., :3] - col[..., :3])[inside].mean()
    sharp = np.abs(out[..., :3] - col[..., :3])[inside & (bmap < 0.02)].mean() if (inside & (bmap < 0.02)).any() else 0.0
    ok = finite and changed > 0.01 and sharp < 0.02
    return report(name, ok, 'changed %.3f, unblurred where blur map is 0: %.4f (%.0fs)' % (changed, sharp, time.time() - t0)), out


def check_independent(name, path, maps, pieces, params):
    """change the input on piece 0 only: every other piece must be unchanged"""
    mask, pos, uvm = maps
    col, bmap = colour_input(pos, uvm, mask)
    a, _ = run(path, col, pos, uvm, bmap, params)
    col2 = col.copy()
    y0, x0, h = piece_tiles(len(pieces), pos.shape[0])[0]
    col2[y0:y0 + h, x0:x0 + h, :3] = 1.0 - col2[y0:y0 + h, x0:x0 + h, :3]
    b, _ = run(path, col2, pos, uvm, bmap, params)
    other = np.ones(uvm.shape, bool)
    other[y0:y0 + h, x0:x0 + h] = False
    d_other = np.abs(a - b)[other].max()
    d_mine = np.abs(a - b)[~other].max()
    return report(name, d_other == 0.0 and d_mine > 0.1, 'other pieces max diff %.1e, edited piece %.2f' % (d_other, d_mine))


def check_scale(name, path, maps, params):
    mask, pos, uvm = maps
    col, bmap = colour_input(pos, uvm, mask)
    a, _ = run(path, col, pos, uvm, bmap, params)
    b, _ = run(path, col, (pos * 0.001 + 0.002).astype(np.float32), uvm, bmap, params)
    d = np.abs(a - b)[uvm > 0.5]
    return report(name, d.max() < 2e-3, 'max diff %.1e, mean %.1e' % (d.max(), d.mean()))


def levels_read(radius_voxels, softness):
    """how many mip levels pass 1's taps read at this largest radius (W.tap_level, W.pass_pyramid)"""
    x = float(np.clip(np.float32(softness) * np.float32(0.4) * np.float32(radius_voxels) / np.float32(0.64), 1.0, 16.0))
    lev = np.floor(np.log2(x)) + (x / 2.0 ** np.floor(np.log2(x)) - 1.0)
    return int(sum(lev > l - 1 for l in range(1, W.LEVELS))) if softness > 0 else 0


def check_mips(name, path, maps, params, skipping=True):
    """WSNB_Mip1..4 of pass 1 against a direct per-box downsample of the voxels. The shipping
    graph (skipping) only makes the levels pass 1's taps read; the others must be empty."""
    mask, pos, uvm = maps
    col, bmap = colour_input(pos, uvm, mask)
    names = node_names(path)
    keep = [names['WSNB_Voxels'][0], names['WSB_VoxBoxes'][0], names['WSNB_MipBoxes'][0], names['WSNB_MipMap'][0]] + \
        [names['WSNB_Mip%d' % l][0] for l in range(1, W.LEVELS)] + [names['WSNB_Radius'][0]]
    _, r = run(path, col, pos, uvm, bmap, params, keep=keep)
    nodes = r['_nodes']
    vox, vb, mb, mm = (nodes[k] for k in keep[:4])
    mips = [nodes[k] for k in keep[4:-1]]
    made = levels_read(nodes[keep[-1]][8, 4, 0], params.get('softness', 0.25)) if skipping else W.LEVELS - 1
    bases, QW, QH = W.mip_layout()
    R, S = ref.R_MAX, ref.S_MAX
    k = np.array([1, 3, 3, 1]) / 8.0
    worst, nbox = 0.0, 0
    q = mips[-1].reshape(-1, 4)
    qm = mm.reshape(-1, 4)
    for b in range(R * S):
        s_, r_ = divmod(b, R)
        left, right = vb[s_, r_], vb[s_, R + r_]
        if right[3] < 0.5:
            continue
        nbox += 1
        n = right[:3].astype(int)
        o = int(left[3])
        grid = vox.reshape(-1, 4)[o:o + n.prod()].reshape(n[2], n[1], n[0], 4).astype(np.float64)
        offs = mb[s_, r_]
        for l in range(1, W.LEVELS):
            d = -(-n // 2)
            pad = np.pad(grid, [(1, 2 * d[2] + 1 - grid.shape[0]), (1, 2 * d[1] + 1 - grid.shape[1]),
                                (1, 2 * d[0] + 1 - grid.shape[2]), (0, 0)])
            for ax in range(3):
                sl = lambda a, m=pad.shape[ax]: tuple(slice(a, a + 2 * d[2 - ax], 2) if i == ax else slice(None)
                                                      for i in range(4))
                pad = sum(k[a] * pad[sl(a)] for a in range(4))
            grid, n = pad, d
            ol = int(offs[l - 1])
            got = q[ol:ol + n.prod()].reshape(n[2], n[1], n[0], 4)
            if l > made:
                if np.abs(got).max() != 0.0:
                    return report(name, False, 'level %d is not read by any tap but was computed' % l)
                continue
            worst = max(worst, np.abs(got - grid).max() / max(np.abs(grid).max(), 1e-12))
            ids = qm[ol:ol + n.prod()]
            if not (ids[:, 0] == b).all():
                return report(name, False, 'mip map: entries of box %d level %d point elsewhere' % (b, l))
    # every pass-1 mip read the voxels of pass 1: worst is the half-float rounding
    return report(name, worst < 5e-3, '%d boxes, %d of %d levels made, worst relative diff %.1e'
                  % (nbox, made, W.LEVELS - 1, worst))


def check_skipped(name, path, maps):
    """Work that isn't used isn't done (counted by the interpreter as executed)."""
    mask, pos, uvm = maps
    col, bmap = colour_input(pos, uvm, mask)
    names = node_names(path)
    nvox = ref.AW * ref.AH
    ntex = uvm.size

    def reads(node, k=None):
        u = names[node]
        return [SI.STATS[x]['reads'] for x in u] if k is None else SI.STATS[u[k]]['reads']

    bad = []
    P = dict(intensity=24.0, samples=4.0, quality=3.0)
    # 4 of 16 pass slots run: the other 12 read their input (1 read per voxel), their mips nothing
    SI.STATS.clear()
    run(path, col, pos, uvm, bmap, P)
    off = [reads('WSNB_Pass%d' % k, 0) for k in range(5, W.MAX_PASSES + 1)]
    on = [reads('WSNB_Pass%d' % k, 0) for k in range(1, 5)]
    if any(x != nvox for x in off) or min(on) <= nvox:
        bad.append('pass slots: on %s, off %s reads (%d voxels)' % (on, off, nvox))
    mips_off = sum(sum(reads('WSNB_Mip%d' % l)[4:]) for l in range(1, W.LEVELS))
    if mips_off != 0 or sum(reads('WSNB_Mip1')[:4]) == 0:
        bad.append('mips of inactive slots read %d' % mips_off)
    # Softness 0: no mip is read
    SI.STATS.clear()
    run(path, col, pos, uvm, bmap, dict(P, softness=0.0))
    mips = sum(sum(reads('WSNB_Mip%d' % l)) for l in range(1, W.LEVELS))
    if mips != 0:
        bad.append('Softness 0: mips read %d' % mips)
    # Intensity 0: every pass slot is off, and a texel reads the input, the UV mask and the radius
    SI.STATS.clear()
    out, _ = run(path, col, pos, uvm, bmap, dict(P, intensity=0.0, use_blur_map=False))
    passes = [reads('WSNB_Pass%d' % k, 0) for k in range(1, W.MAX_PASSES + 1)]
    if any(x != nvox for x in passes) or reads('WSNB_Resolve', 0) != 3 * ntex or np.abs(out - col).max() > 1e-4:
        bad.append('Intensity 0: passes %s, resolve %d reads for %d texels' % (passes, reads('WSNB_Resolve', 0), ntex))
    return report(name, not bad, '; '.join(bad) or '12 idle pass slots: 1 read per voxel, their mips 0; Softness 0: '
                  'mips 0; Intensity 0: 3 reads per texel')


def main():
    path = build(SMALL)
    ok = []
    demo_box.RES = 64
    faces = demo_box.make_faces()
    islands, _ = demo_box.layout_islands(faces)
    pos, uvm, _ = demo_box.bake_maps(faces, islands)
    box = (None, pos.astype(np.float32), uvm.astype(np.float32))
    lay = layout_mesh.make_layout(96, 12.0)
    P = dict(intensity=24.0, samples=4.0, quality=3.0)
    ok.append(check_blur('box 64px', path, box, P)[0])
    ok.append(check_blur('box 64px, Softness 0, 8 samples', path, box, dict(P, softness=0.0, samples=8.0))[0])
    ok.append(check_blur('box 64px, quality 1, anisotropy', path, box, dict(P, quality=1.0, anisotropy=0.7))[0])
    ok.append(check_mips('box 64px, mip pyramid per box', path, box, P))
    ok.append(check_scale('box 64px, scaled to 1/1000', path, box, P))
    ok.append(check_blur('layout 96px, 4 pieces', path, lay, P)[0])
    ok.append(check_mips('layout 96px, 4 pieces, mip pyramid per box', path, lay, P))
    ok.append(check_independent('layout 96px, 4 pieces don\'t interact', path, lay, layout_mesh.PIECES, P))
    ok.append(check_scale('layout 96px, scaled to 1/1000', path, lay, P))
    for nm, pieces in (('6 in a row (slabs merged)', ROW), ('column along Z', COLUMN), ('stack along Y', STACK)):
        maps = layout_mesh.make_layout(96, 12.0, pieces)
        ok.append(check_blur('layout 96px, ' + nm, path, maps, P)[0])
    ok.append(check_independent('layout 96px, column along Z, independent', path,
                                layout_mesh.make_layout(96, 12.0, COLUMN), COLUMN, P))
    ok.append(check_independent('layout 96px, stack along Y, independent', path,
                                layout_mesh.make_layout(96, 12.0, STACK), STACK, P))
    # the shipping graph only makes the mip levels that are read; the plain one makes them all
    big = dict(P, softness=1.0, intensity=50.0)
    ok.append(check_mips('box 64px, mips at Softness 1, Intensity 50', path, box, big))
    plain = build(SMALL, fast=False)
    ok.append(check_mips('box 64px, mip pyramid, plain graph', plain, box, P, skipping=False))
    ok.append(check_mips('layout 96px, mip pyramid, plain graph', plain, lay, P, skipping=False))
    ok.append(check_skipped('box 64px, unused work is skipped', path, box))
    # Lite, at the same reduced sizes
    lite = build(SMALL, lite=True)
    ok.append(check_blur('Lite: box 64px', lite, box, P)[0])
    ok.append(check_blur('Lite: box 64px, quality 1, anisotropy, 8 samples', lite, box,
                         dict(P, quality=1.0, anisotropy=0.7, samples=8.0))[0])
    ok.append(check_scale('Lite: box 64px, scaled to 1/1000', lite, box, P))
    ok.append(check_blur('Lite: layout 96px, 4 pieces', lite, lay, P)[0])
    ok.append(check_independent('Lite: layout 96px, 4 pieces don\'t interact', lite, lay, layout_mesh.PIECES, P))
    ok.append(check_independent('Lite: layout 96px, stack along Y, independent', lite,
                                layout_mesh.make_layout(96, 12.0, STACK), STACK, P))
    print('ALL PASS' if all(ok) else 'SOME FAILED')
    return all(ok)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
