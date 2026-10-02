"""Generates world_space_non_uniform_blur_v<VERSION>.sbs.

Bump VERSION below for each release; it goes in the file name and the graph descriptions.

World space version of the Non-Uniform Blur: the mesh is voxelized (same
bounding box / sample sort / gather splat as world_space_mask_blur), the
Non-Uniform Blur passes (Samples x Blades taps on a rotated, anisotropic
polygon, radius decaying per pass, scaled by the blur map) run in 3D on the
voxel grid, in each voxel's tangent plane, and the result is read back at
every texel's mesh position.  Seams no longer show because neighbouring
islands share the same voxels.

Run:  python3 tools/build_wsnub.py  [output.sbs]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbsgen import (Program, E, lift, vec, floor, ceil, sqrt, exp, sin, cos, absv, fmin, fmax, clamp,
                    dot, cross, ifelse, tofloat, get, pf, pi, pb, POS, SIZE, samplecol, samplelum,
                    fmod, UID, dynamic_value, dynamic_expr, T_F1, T_F2, T_F3, T_F4, T_BOOL, T_INT)

VERSION = '1.0'
GRAPH_ID = 'world_space_non_uniform_blur'
OUTPUT_NAME = '%s_v%s.sbs' % (GRAPH_ID, VERSION)
MAX_PASSES = 16
LEVELS = 5                     # voxel mip levels: 128, 64, 32, 16, 8
C_DECAY = math.sqrt(-math.log(0.001))   # same decay as the 2D Non-Uniform Blur
F16 = 2                        # pixel processor format: 16 bits float
LEFT = (0.28125, 0.53125)      # texel (4,8) of a 16x16 info texture
RIGHT = (0.78125, 0.53125)     # texel (12,8)
SAMP = 512                     # surface samples: 512x512
HASH = 128                     # hash grid resolution (fixed)


# ----------------------------------------------------------------------------
# voxel atlas layouts: level l stores (128>>l)^3 voxels as z-slices in tiles
def level_layout(l):
    n = 128 >> l
    tiles = {0: 16, 1: 8, 2: 8, 3: 4, 4: 4}[l]
    rows = n // tiles
    return n, tiles, tiles * n, rows * n            # n, tiles per row, W, H


def atlas_size_log2(l):
    n, t, W, H = level_layout(l)
    return int(math.log2(W)), int(math.log2(H))


def voxel_uv(l, ijk):
    """ijk: float3 integer voxel coords at level l -> texel-centre uv."""
    n, T, W, H = level_layout(l)
    zr = floor(ijk.z / T)
    zc = ijk.z - zr * T
    return (vec(zc * n + ijk.x, zr * n + ijk.y) + 0.5) / vec(W, H)


def pixel_voxel(l):
    """voxel coords of the current output texel of a level-l atlas."""
    n, T, W, H = level_layout(l)
    px = floor(POS * vec(W, H))
    tile = floor(px / n)
    return vec(px - tile * n, tile.y * T + tile.x)


def idx_uv(i, size=SAMP):
    row = floor(i / size)
    return (vec(i - size * row, row) + 0.5) / size


def smoothstep01(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - x * 2.0)


# ----------------------------------------------------------------------------
# pixel processor programs.  Input indices are documented per node.

def prog_info():
    # in0 mesh position.  16x16 (input/16): stores the position map size.
    p = Program()
    return p, vec(SIZE * 16.0, 0.0, 1.0)


def prog_bbox():
    # in0 position, in1 uv mask, in2 info -> left: (hash grid origin, hash voxel), right: (extent,0,0,1)
    p = Program()
    p.set('isz', samplecol(2, (0.5, 0.5)).xy)
    p.set('mn', [1e20] * 3)
    p.set('mx', [-1e20] * 3)
    with p.loop('i', 0.0, 16392) as L:
        i = p.v('i')
        L.until(i >= 16384.0)
        isz = p.v('isz')

        def uv():
            row = floor(i / 128.0)
            return (floor(((vec(i - row * 128.0, row) + 0.47) / 128.0) * isz) + 0.5) / isz
        u = uv()
        p.set('mn', fmin(p.v('mn'), ifelse(samplelum(1, u) > 0.99, samplecol(0, u).xyz, lift([1e20] * 3))))
        u = uv()
        p.set('mx', fmax(p.v('mx'), ifelse(samplelum(1, u) > 0.99, samplecol(0, u).xyz, lift([-1e20] * 3))))
        p.set('i', i + 1.0)
    mn, mx = p.v('mn'), p.v('mx')
    d = mx - mn
    ext = fmax(fmax(fmax(d.x, d.y), d.z), 1e-8)
    hf = ext / 124.0
    left = vec((mn + mx) * 0.5 - hf * 64.0, hf)
    right = vec(ext, 0.0, 0.0, 1.0)
    return p, ifelse(POS.x < 0.5, left, right)


def quality_n():
    q = clamp(floor(pf('quality') + 0.5), 1.0, 4.0)
    return ifelse(q < 1.5, 48.0, ifelse(q < 2.5, 64.0, ifelse(q < 3.5, 96.0, 128.0)))


def blur_amount(in_idx, map_idx, uv):
    """per-texel blur scale: 1, the Blur Map input or the luminance of the input itself."""
    src = samplecol(in_idx, uv)
    e = ifelse(pb('blur_map_from_input'), dot(src.xyz, lift([0.2126, 0.7152, 0.0722])), samplelum(map_idx, uv))
    e = clamp(ifelse(pb('invert_blur_map'), 1.0 - e, e), 0.0, 1.0)
    return ifelse(pb('use_blur_map'), e, 1.0)


def prog_density():
    # in0 area, in1 info -> (hash voxels per UV unit, 0, 0, 1): average texel density of the mesh,
    # used to give Intensity the same meaning as in the 2D Non-Uniform Blur.
    p = Program()
    p.set('asum', 0.0)
    p.set('acnt', 0.0)
    with p.loop('i', 0.0, 16392) as L:
        i = p.v('i')
        L.until(i >= 16384.0)

        def area():
            row = floor(i / 128.0)
            return samplecol(0, (vec(i - row * 128.0, row) * 4.0 + 2.5) / SAMP).w
        p.set('asum', p.v('asum') + fmax(area(), 0.0))
        p.set('acnt', p.v('acnt') + ifelse(area() > 0.0, 1.0, 0.0))
        p.set('i', i + 1.0)
    isz = samplecol(1, (0.5, 0.5)).xy
    amean = p.v('asum') / fmax(p.v('acnt'), 1.0)
    # one input texel covers amean hash voxels^2 and 1/isz of UV space
    return p, vec(sqrt(amean) * sqrt(isz.x * isz.y), 0.0, 0.0, 1.0)


def prog_grid():
    # in0 bbox.  left: (blur grid origin, voxel size), right: (grid N, 0, 0, 1).
    # Depends only on the mesh and Quality, so the splats are not recomputed when the blur sliders move.
    p = Program()
    bb = samplecol(0, LEFT)
    ext = samplecol(0, RIGHT).x
    N = quality_n()
    h = ext / (N - 6.0)
    centre = bb.xyz + bb.w * 64.0
    org = centre - h * (N * 0.5)
    return p, ifelse(POS.x < 0.5, vec(org, h), vec(N, 0.0, 0.0, 1.0))


def prog_radius():
    # in0 bbox, in1 density, in2 grid -> (max blur radius in voxels, 0, 0, 1)
    # 2D units: Intensity 1 = 1/256 of the UV space, at the mesh's average texel density
    p = Program()
    rv = fmax(pf('intensity'), 0.0) / 256.0 * samplecol(1, (0.5, 0.5)).x * samplecol(0, LEFT).w / samplecol(2, LEFT).w
    return p, vec(rv, 0.0, 0.0, 1.0)


def prog_reps():
    # in0 position, in1 uv mask, in2 info -> per 512 cell: (valid position of the cell, 1) or 0
    p = Program()
    isz = p.set('isz', samplecol(2, (0.5, 0.5)).xy)
    p.set('cell', floor(POS * SAMP))
    p.set('bestUV', (floor(POS * SAMP) + 0.5) / SAMP)
    p.set('bestD', 1e20)
    with p.loop('i', 0.0, 24) as L:
        i = p.v('i')
        L.until(i >= 16.0)
        cell, isz = p.v('cell'), p.v('isz')

        def sub():
            row = floor(i / 4.0)
            s = (cell + (vec(i - row * 4.0, row) + 0.47) / 4.0) / SAMP
            d = s - (cell + 0.5) / SAMP
            ok = samplelum(1, (floor(s * isz) + 0.5) / isz) > 0.99
            return s, dot(d, d), ok
        s, dd, ok = sub()
        p.set('bestUV', ifelse(ok & (dd < p.v('bestD')), s, p.v('bestUV')))
        s, dd, ok = sub()
        p.set('bestD', ifelse(ok & (dd < p.v('bestD')), dd, p.v('bestD')))
        p.set('i', i + 1.0)
    found = p.v('bestD') < 1e19
    uv = (floor(p.v('bestUV') * p.v('isz')) + 0.5) / p.v('isz')
    return p, vec(ifelse(found, samplecol(0, uv).xyz, lift([0, 0, 0])), ifelse(found, 1.0, 0.0))


def _subsample(i, cell, isz):
    row = floor(i / 4.0)
    return (floor(((cell + (vec(i - row * 4.0, row) + 0.47) / 4.0) / SAMP) * isz) + 0.5) / isz


def _near(pos_idx, mask_idx, uv, rep, r2):
    d = samplecol(pos_idx, uv).xyz - rep.xyz
    return (samplelum(mask_idx, uv) > 0.99) & (dot(d, d) < r2)


def prog_area():
    # in0 position, in1 uv mask, in2 reps, in3 bbox, in4 info
    # -> (surface area around the sample, area weighted normal xyz), in hash voxel^2 units
    p = Program()
    p.set('isz', samplecol(4, (0.5, 0.5)).xy)
    p.set('cell', floor(POS * SAMP))
    p.set('rep', samplecol(2, (floor(POS * SAMP) + 0.5) / SAMP))
    hf = samplecol(3, LEFT).w
    p.set('r2', (hf * 2.0) * (hf * 2.0))
    p.set('a', 0.0)
    p.set('nrm', [0, 0, 0])
    with p.loop('i', 0.0, 24) as L:
        i = p.v('i')
        L.until(i >= 16.0)
        isz, cell, rep, r2 = p.v('isz'), p.v('cell'), p.v('rep'), p.v('r2')

        def crossv():
            uv = _subsample(i, cell, isz)
            c = samplecol(0, uv).xyz

            def side(du):
                up, dn = uv + du, uv - du
                dup = samplecol(0, up).xyz - c
                ddn = c - samplecol(0, dn).xyz
                inb = lambda q: (fmin(q.x, q.y) > 0.0) & (fmax(q.x, q.y) < 1.0)
                okup = (samplelum(1, up) > 0.99) & (dot(dup, dup) < r2) & inb(up)
                okdn = (samplelum(1, dn) > 0.99) & (dot(ddn, ddn) < r2) & inb(dn)
                return ifelse(okup, dup, ifelse(okdn, ddn, lift([0, 0, 0])))
            tu = side(vec(1.0 / isz.x, 0.0))
            tv = side(vec(0.0, 1.0 / isz.y))
            cr = cross(tu, tv)
            ok = _near(0, 1, uv, rep, r2) & (rep.w > 0.5)
            return cr, ok
        cr, ok = crossv()
        p.set('a', p.v('a') + ifelse(ok, sqrt(dot(cr, cr)), 0.0))
        cr, ok = crossv()
        p.set('nrm', p.v('nrm') + ifelse(ok, cr, lift([0, 0, 0])))
        p.set('i', i + 1.0)
    inv = 1.0 / (hf * hf * 16.0)
    return p, vec(p.v('nrm') * inv, p.v('a') * inv)   # (n.xyz, a) -- area in .w


def prog_vals():
    # in0 input colour, in1 position, in2 uv mask, in3 reps, in4 bbox, in5 info, in6 blur map
    # -> mean (r, g, b, blur amount) of the texels that belong to the sample
    p = Program()
    p.set('isz', samplecol(5, (0.5, 0.5)).xy)
    p.set('cell', floor(POS * SAMP))
    p.set('rep', samplecol(3, (floor(POS * SAMP) + 0.5) / SAMP))
    hf = samplecol(4, LEFT).w
    p.set('r2', (hf * 2.0) * (hf * 2.0))
    p.set('csum', [0, 0, 0, 0])
    p.set('cnt', 0.0)
    with p.loop('i', 0.0, 24) as L:
        i = p.v('i')
        L.until(i >= 16.0)
        isz, cell, rep, r2 = p.v('isz'), p.v('cell'), p.v('rep'), p.v('r2')

        def one():
            uv = _subsample(i, cell, isz)
            ok = _near(1, 2, uv, rep, r2) & (rep.w > 0.5)
            eff = blur_amount(0, 6, uv)
            return vec(samplecol(0, uv).xyz, eff), ok
        v, ok = one()
        p.set('csum', p.v('csum') + ifelse(ok, v, lift([0, 0, 0, 0])))
        v, ok = one()
        p.set('cnt', p.v('cnt') + ifelse(ok, 1.0, 0.0))
        p.set('i', i + 1.0)
    cnt = p.v('cnt')
    return p, ifelse(cnt > 0.5, p.v('csum') / fmax(cnt, 1.0), lift([0, 0, 0, 0]))


def prog_keys():
    # in0 reps, in1 bbox -> (hash voxel id or 2^23 if empty, sample index, 0, 1)
    p = Program()
    rep = samplecol(0, POS)
    bb = samplecol(1, LEFT)
    v = clamp(floor((rep.xyz - bb.xyz) / bb.w), 0.0, 127.0)
    px = floor(POS * SAMP)
    key = ifelse(rep.w > 0.5, (v.z * 128.0 + v.y) * 128.0 + v.x, 8388608.0)
    return p, vec(key, px.y * SAMP + px.x, 0.0, 1.0)


def sort_stages():
    out = []
    k = 2
    while k <= SAMP * SAMP:
        j = k // 2
        while j >= 1:
            out.append((k, j))
            j //= 2
        k *= 2
    return out


def prog_sort(k, j):
    # in0 previous keys -> one bitonic compare/exchange step
    p = Program()
    px = floor(POS * SAMP)
    t = px.y * SAMP + px.x

    def bit(m):
        return floor(t / m) - floor(t / (2.0 * m)) * 2.0
    d = bit(float(k)) - bit(float(j))
    up = d * d < 0.5
    a = samplecol(0, POS)
    q = t + (1.0 - bit(float(j)) * 2.0) * float(j)
    b = samplecol(0, idx_uv(q))
    less = (a.x < b.x) | ((a.x <= b.x) & (a.y < b.y))
    return p, ifelse((up & less) | (~up & ~less), a, b)


def prog_ranges():
    # in0 sorted keys -> per hash voxel: (first, end) index into the sorted list
    p = Program()
    ijk = pixel_voxel(0)
    p.set('cid', (ijk.z * 128.0 + ijk.y) * 128.0 + ijk.x)
    p.set('s1', [0.0, float(SAMP * SAMP)])
    p.set('s2', [0.0, float(SAMP * SAMP)])
    with p.loop('i', 0.0, 28) as L:
        s1, s2 = p.v('s1'), p.v('s2')
        L.until((s1.x >= s1.y) & (s2.x >= s2.y))

        def step(name, target):
            s = p.v(name)
            m = floor((s.x + s.y) * 0.5)
            go = samplecol(0, idx_uv(m)).x < target
            p.set(name, ifelse(s.x < s.y, ifelse(go, vec(m + 1.0, s.y), vec(s.x, m)), s))
        step('s1', p.v('cid'))
        step('s2', p.v('cid') + 1.0)
        p.set('i', p.v('i') + 1.0)
    return p, vec(p.v('s1').x, p.v('s2').x, 0.0, 1.0)


def _splat_loop(p, accumulate, active=None):
    """Gather all surface samples within one blur voxel of this voxel's centre.

    in0 sorted, in1 ranges, in2 reps, in4 params, in5 bbox.
    `accumulate(sample_uv, tent_weight)` adds the sample's contribution.
    Leaves the voxel coordinates in variable 'ijk' and the activity in 'nc'."""
    prm = samplecol(4, LEFT)
    p.set('ijk', pixel_voxel(0))
    p.set('C', prm.xyz + (p.v('ijk') + 0.5) * prm.w)

    def corner(sgn):
        bb = samplecol(5, LEFT)
        return clamp(floor((p.v('C') + samplecol(4, LEFT).w * sgn - bb.xyz) / bb.w), 0.0, 127.0)
    p.set('lo', corner(-1.0))
    p.set('dims', corner(1.0) - p.v('lo') + 1.0)
    N = samplecol(4, RIGHT).x
    ijk, dims = p.v('ijk'), p.v('dims')
    inside = (ijk.x < N) & (ijk.y < N) & (ijk.z < N)
    if active is not None:
        inside = inside & active
    p.set('nc', ifelse(inside, dims.x * dims.y * dims.z, 0.0))
    p.set('se', [0.0, 0.0])
    p.set('hv', 0.0)
    with p.loop('t', 0.0, 65536) as L:
        se, t = p.v('se'), p.v('t')
        L.until((se.x >= se.y) & (t >= p.v('nc')))
        p.set('hv', ifelse(p.v('se').x < p.v('se').y, 1.0, 0.0))
        entry = samplecol(0, idx_uv(p.v('se').x)).y
        suv = idx_uv(entry)
        d = samplecol(2, suv).xyz - p.v('C')
        h = samplecol(4, LEFT).w
        tw3 = fmax(1.0 - fmax(d, -d) / h, 0.0)
        tw = ifelse(p.v('hv') > 0.5, tw3.x * tw3.y * tw3.z, 0.0)
        accumulate(suv, tw)
        t, dims = p.v('t'), p.v('dims')
        q = floor(t / dims.x)
        cell = p.v('lo') + vec(t - dims.x * q, q - dims.y * floor(q / dims.y), floor(q / dims.y))
        p.set('se', ifelse(p.v('hv') > 0.5, p.v('se') + lift([1.0, 0.0]),
                           samplecol(1, voxel_uv(0, cell)).xy))
        p.set('t', ifelse(p.v('hv') > 0.5, p.v('t'), p.v('t') + 1.0))


WSCALE = 16.0   # keeps splat weights well inside half-float range


def prog_splat_colour():
    # in0 sorted, in1 ranges, in2 reps, in3 vals, in4 grid, in5 bbox, in6 area, in7 frame
    # -> (sum colour*w, sum blur amount*w); sum w is in frame.w.  Empty voxels (frame.w = 0) skip the loop.
    p = Program()
    p.set('acc', [0, 0, 0, 0])

    def accumulate(suv, tw):
        w = samplecol(6, suv).w * tw * WSCALE
        p.set('acc', p.v('acc') + samplecol(3, suv) * w)
    _splat_loop(p, accumulate, active=samplecol(7, POS).w > 0.0)
    return p, p.v('acc')


def prog_splat_frame():
    # in0 sorted, in1 ranges, in2 reps, in4 grid, in5 bbox, in6 area
    # -> (unit surface normal xyz, sum w) of the voxel; 0 if empty.  Depends only on the mesh.
    p = Program()
    p.set('wsum', 0.0)
    p.set('nsum', [0, 0, 0])
    p.set('tdia', [0, 0, 0])
    p.set('toff', [0, 0, 0])

    def accumulate(suv, tw):
        ar = samplecol(6, suv)

        def unit():
            n = ar.xyz
            return n / fmax(sqrt(dot(n, n)), 1e-12)
        w = ar.w * tw
        p.set('wsum', p.v('wsum') + w * WSCALE)
        p.set('nsum', p.v('nsum') + unit() * (samplecol(6, suv).w * tw))
        n = unit()
        p.set('tdia', p.v('tdia') + n * n * (samplecol(6, suv).w * tw))
        n = unit()
        p.set('toff', p.v('toff') + vec(n.x * n.y, n.x * n.z, n.y * n.z) * (samplecol(6, suv).w * tw))
    _splat_loop(p, accumulate)
    # dominant eigenvector of the normal tensor = sign-free surface normal
    dg = p.v('tdia')
    p.set('nv', ifelse((dg.x >= dg.y) & (dg.x >= dg.z), lift([1, 0, 0]),
                       ifelse(dg.y >= dg.z, lift([0, 1, 0]), lift([0, 0, 1]))))
    for _ in range(8):
        v, d, o = p.v('nv'), p.v('tdia'), p.v('toff')
        mv = vec(d.x * v.x + o.x * v.y + o.y * v.z,
                 o.x * v.x + d.y * v.y + o.z * v.z,
                 o.y * v.x + o.z * v.y + d.z * v.z)
        p.set('nv', mv / fmax(sqrt(dot(mv, mv)), 1e-12))
    nv = p.v('nv')
    nv = ifelse(dot(nv, p.v('nsum')) < 0.0, -nv, nv)
    w = p.v('wsum')
    return p, ifelse(w > 0.0, vec(nv, w), lift([0, 0, 0, 0]))


def prog_v0():
    # in0 colour splat, in1 frame -> (sum colour*w, sum w): the voxels the passes start from
    p = Program()
    return p, vec(samplecol(0, POS).xyz, samplecol(1, POS).w)


def pass_count():
    # same as the 2D graph: min(Samples, ceil(Intensity * pi))
    return fmin(fmax(pi('samples'), 1.0), ceil(fmax(pf('intensity'), 0.0) * math.pi))


def pass_on(k):
    return (pass_count() > k + 0.5) & (pf('intensity') > 0.0)


# mip levels 1..4 are packed side by side in one 1024x512 texture
QW, QH = 1024, 512
Q_OX = {1: 0, 2: 512, 3: 768, 4: 832}


def q_uv(l, ijk):
    n, T, W, H = level_layout(l)
    zr = floor(ijk.z / T)
    zc = ijk.z - zr * T
    return (vec(Q_OX[l] + zc * n + ijk.x, zr * n + ijk.y) + 0.5) / vec(QW, QH)


def prog_pyramid(l, k):
    # in0 level 0 voxels (l = 1) or the previous packed texture (l > 1) -> packed texture with levels 1..l.
    # Level l is the [1 3 3 1]/8 stride-2 filter of level l-1; lower levels are copied.
    # Only computed when pass k uses fat taps.
    p = Program()
    n, T, W, H = level_layout(l)
    np_, _, _, _ = level_layout(l - 1)
    px = floor(POS * vec(QW, QH))
    loc = px - vec(Q_OX[l], 0.0)
    tile = floor(loc / n)
    p.set('ijk', vec(loc - tile * n, tile.y * T + tile.x))
    inreg = (px.x >= Q_OX[l]) & (px.x < Q_OX[l] + W) & (px.y < H)
    p.set('acc', [0, 0, 0, 0])
    on = pass_on(k) & (pf('softness') > 0.0) & inreg
    p.set('on', ifelse(on, 1.0, 0.0))
    with p.loop('i', 0.0, 72) as L:
        i = p.v('i')
        L.until((i >= 64.0) | (p.v('on') < 0.5))
        a = fmod(i, 4.0)
        b = fmod(floor(i / 4.0), 4.0)
        c = floor(i / 16.0)
        o = vec(a, b, c)
        src = p.v('ijk') * 2.0 + o - 1.0
        wv = ifelse(o.x > 0.5, ifelse(o.x < 2.5, 3.0, 1.0), 1.0) * \
            ifelse(o.y > 0.5, ifelse(o.y < 2.5, 3.0, 1.0), 1.0) * \
            ifelse(o.z > 0.5, ifelse(o.z < 2.5, 3.0, 1.0), 1.0) / 512.0
        inb = (fmin(fmin(src.x, src.y), src.z) >= 0.0) & (fmax(fmax(src.x, src.y), src.z) <= np_ - 1.0)
        sc = clamp(src, 0.0, np_ - 1.0)
        val = samplecol(0, voxel_uv(0, sc) if l == 1 else q_uv(l - 1, sc))
        p.set('acc', p.v('acc') + ifelse(inb, val * wv, lift([0, 0, 0, 0])))
        p.set('i', i + 1.0)
    if l == 1:
        return p, ifelse(p.v('on') > 0.5, p.v('acc'), lift([0, 0, 0, 0]))
    lower = px.x < Q_OX[l]
    return p, ifelse(p.v('on') > 0.5, p.v('acc'), ifelse(lower, samplecol(0, POS), lift([0, 0, 0, 0])))


def _corners(i0, f, uvf, idx):
    acc = None
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (f.x if dx else 1.0 - f.x) * (f.y if dy else 1.0 - f.y) * (f.z if dz else 1.0 - f.z)
                v = samplecol(idx, uvf(i0 + lift([float(dx), float(dy), float(dz)]))) * w
                acc = v if acc is None else acc + v
    return acc


def tri_v(g, idx=0):
    """trilinear sample of the level-0 voxels at voxel-centre coords g."""
    gl = clamp(g, 0.0, 127.0)
    i0 = fmin(floor(gl), 126.0)
    return _corners(i0, gl - i0, lambda c: voxel_uv(0, c), idx)


def pow2_level(l):
    return ifelse(l < 0.5, 1.0, ifelse(l < 1.5, 2.0, ifelse(l < 2.5, 4.0, ifelse(l < 3.5, 8.0, 16.0))))


def tri_q(l, g, idx):
    """trilinear sample of packed mip level l (1..4, runtime value) at level-0 voxel-centre coords g."""
    s = pow2_level(l)
    n = 128.0 / s
    T = ifelse(l < 2.5, 8.0, 4.0)
    ox = ifelse(l < 1.5, 0.0, ifelse(l < 2.5, 512.0, ifelse(l < 3.5, 768.0, 832.0)))
    gl = clamp((g + 0.5) / s - 0.5, 0.0, n - 1.0)
    i0 = fmin(floor(gl), n - 2.0)

    def uvf(c):
        zr = floor(c.z / T)
        zc = c.z - zr * T
        return (vec(ox + zc * n + c.x, zr * n + c.y) + 0.5) / vec(QW, QH)
    return _corners(i0, gl - i0, uvf, idx)


def fat_tap(g, lev, iv=0, iq=1):
    """(sum c*w, sum w) at g, blurred to mip level `lev` (0..4): 2 trilinear lookups.
    Coarser levels are rescaled by 2^l so a tap on a flat surface weighs the same at any level."""
    l0 = fmin(floor(lev), 3.0)
    f = lev - l0
    a = ifelse(l0 < 0.5, tri_v(g, iv), tri_q(fmax(l0, 1.0), g, iq) * pow2_level(l0))
    b = tri_q(l0 + 1.0, g, iq) * pow2_level(l0 + 1.0)
    return a * (1.0 - f) + b * f


def approx_log2_clamped(x):
    """piecewise-linear log2(x) clamped to [0, LEVELS-1]."""
    x = clamp(x, 1.0, 2.0 ** (LEVELS - 1))
    o = ifelse(x >= 2.0, 1.0, 0.0) + ifelse(x >= 4.0, 1.0, 0.0) + ifelse(x >= 8.0, 1.0, 0.0) + \
        ifelse(x >= 16.0, 1.0, 0.0)
    base = ifelse(x >= 16.0, 16.0, ifelse(x >= 8.0, 8.0, ifelse(x >= 4.0, 4.0, ifelse(x >= 2.0, 2.0, 1.0))))
    return o + (x / base - 1.0)


def prog_pass(k):
    # in0 voxels (level 0), in1 packed mip levels, in2 frame (normal, w), in3 colour splat (.w = blur*w), in4 radius
    p = Program()
    TWO_PI = 2.0 * math.pi
    S = fmax(pi('samples'), 1.0)
    B = clamp(pi('blades'), 1.0, 9.0)
    p.set('ijk', pixel_voxel(0))
    p.set('V', samplecol(0, POS))
    eff = samplecol(3, POS).w / fmax(samplecol(2, POS).w, 1e-12)
    rv = samplecol(4, (0.5, 0.5)).x
    p.set('r', rv * clamp(eff, 0.0, 1.0) * exp(-2.0 * k * C_DECAY / fmax(pass_count(), 1.0)))
    on = pass_on(k) & (p.v('V').w > 0.0) & (p.v('r') > 0.05)
    p.set('on', ifelse(on, 1.0, 0.0))
    # tangent frame: reference axis projected on the surface
    n = samplecol(2, POS).xyz
    ax = vec(pf('axis_x'), pf('axis_y'), pf('axis_z'))
    D = ax / fmax(sqrt(dot(ax, ax)), 1e-6)
    D = ifelse(dot(ax, ax) > 1e-8, D, lift([0, 1, 0]))
    t1 = D - n * dot(D, n)
    alt = ifelse(absv(D.x) < 0.9, lift([1, 0, 0]), lift([0, 0, 1]))
    t1b = alt - n * dot(alt, n)
    t1 = ifelse(dot(t1, t1) > 0.04, t1, t1b)
    p.set('t1', t1 / fmax(sqrt(dot(t1, t1)), 1e-6))
    p.set('t2', cross(samplecol(2, POS).xyz, p.v('t1')))
    # fat-tap mip level from the tap radius
    sig = fmax(pf('softness'), 0.0) * 0.4 * p.v('r')
    p.set('lev', approx_log2_clamped(sig / 0.64))
    p.set('acc', [0, 0, 0, 0])
    ir = (1.0 / S) / (k + 1.0)
    phi = -TWO_PI * pf('angle')
    an = clamp(pf('anisotropy'), 0.0, 1.0)
    # j = 0 is the centre tap, j = 1..Blades the polygon corners
    with p.loop('j', 0.0, 12) as L:
        j = p.v('j')
        L.until((j > B) | (p.v('on') < 0.5))
        th = TWO_PI * ir + j * (TWO_PI / B)
        s1 = (1.0 - an) * sin(th)
        c1 = cos(th)
        X = s1 * sin(phi) + c1 * cos(phi)
        Y = s1 * cos(phi) - sin(phi) * c1
        off = (p.v('t1') * X + p.v('t2') * Y) * (p.v('r') * ifelse(j > 0.5, 1.0, 0.0))
        p.set('acc', p.v('acc') + fat_tap(p.v('ijk') + off, p.v('lev')))
        p.set('j', j + 1.0)
    return p, ifelse(p.v('on') > 0.5, p.v('acc') / (B + 1.0), p.v('V'))


def bspline_w(t):
    t2 = t * t
    t3 = t2 * t
    return [(1.0 - t) * (1.0 - t) * (1.0 - t) * (1.0 / 6.0),
            (t3 * 3.0 - t2 * 6.0 + 4.0) * (1.0 / 6.0),
            (t2 * 3.0 + t * 3.0 + 1.0 - t3 * 3.0) * (1.0 / 6.0),
            t3 * (1.0 / 6.0)]


def hash12(q):
    p3 = vec(q, q.x) * 0.1031
    p3 = p3 - floor(p3)
    d = dot(p3, p3.yzx + 33.33)
    p3 = p3 + d
    h = (p3.x + p3.y) * p3.z
    return h - floor(h)


def prog_resolve():
    # in0 input, in1 position, in2 uv mask, in3 blurred voxels, in4 grid, in5 blur map, in6 radius
    p = Program()
    rv = samplecol(6, (0.5, 0.5)).x
    # texels with no blur (black blur map, outside the UV mask) skip the voxel lookup
    p.set('al', ifelse(samplelum(2, POS) > 0.5, smoothstep01(rv * blur_amount(0, 5, POS) * 0.5), 0.0))
    prm = samplecol(4, LEFT)
    N = samplecol(4, RIGHT).x
    g = clamp((samplecol(1, POS).xyz - prm.xyz) / prm.w - 0.5, 1.0, N - 2.001)
    p.set('g0', floor(g))
    p.set('gf', g - floor(g))
    gf = p.v('gf')
    wx, wy, wz = bspline_w(gf.x), bspline_w(gf.y), bspline_w(gf.z)
    for a in range(4):
        p.set('wx%d' % a, wx[a])
    for a in range(4):
        p.set('wy%d' % a, wy[a])
    for a in range(4):
        p.set('wz%d' % a, wz[a])
    p.set('S', [0, 0, 0, 0])
    with p.loop('i', 0.0, 72) as L:
        i = p.v('i')
        L.until((i >= 64.0) | (p.v('al') <= 0.0))
        a = fmod(i, 4.0)
        b = fmod(floor(i / 4.0), 4.0)
        c = floor(i / 16.0)

        def pick(name, x):
            return ifelse(x < 0.5, p.v(name + '0'), ifelse(x < 1.5, p.v(name + '1'),
                                                           ifelse(x < 2.5, p.v(name + '2'), p.v(name + '3'))))
        w = pick('wx', a) * pick('wy', b) * pick('wz', c)
        v = samplecol(3, voxel_uv(0, p.v('g0') + vec(a, b, c) - 1.0))
        p.set('S', p.v('S') + v * w)
        p.set('i', i + 1.0)
    S = p.v('S')
    res = S.xyz / fmax(S.w, 1e-12)
    src = samplecol(0, POS)
    p.set('out', src.xyz + (res - src.xyz) * p.v('al'))
    # dither: triangular noise of +-dither/255, faded out at 0 and 1
    px = floor(POS * SIZE)
    nz = hash12(px) + hash12(px + 17.17) - 1.0
    o = p.v('out')
    fade = clamp(fmin(o, 1.0 - o) * 255.0, 0.0, 1.0)
    dith = clamp(o + fade * (nz * pf('dither') * (1.0 / 255.0)), 0.0, 1.0)
    ok = (p.v('al') > 0.0) & (S.w > 1e-12)
    return p, ifelse(ok, vec(dith, src.w), src)


# ----------------------------------------------------------------------------
# graph assembly
class Graph:
    def __init__(self, uid):
        self.uid = uid
        self.nodes = []
        self.out_uid = {}
        self.y = 0

    def input_bridge(self, entry, comptype, x, y, params=''):
        u = self.uid()
        o = self.uid()
        self.nodes.append('<compNode><uid v="%d"/><GUILayout><gpos v="%d %d 0"/></GUILayout><compOutputs><compOutput>'
                          '<uid v="%d"/><comptype v="%d"/></compOutput></compOutputs><compImplementation>'
                          '<compInputBridge><entry v="%d"/><parameters>%s</parameters></compInputBridge>'
                          '</compImplementation></compNode>' % (u, x, y, o, comptype, entry, params))
        self.out_uid[u] = o
        return u

    def pp(self, name, prog_ret, inputs, size, rel=0, fmt=3, colour=True, x=0, y=0):
        prog, ret = prog_ret
        u = self.uid()
        o = self.uid()
        dv, cnt = dynamic_value(self.uid, prog, ret)
        conns = ''.join('<connection><identifier v="%s"/><connRef v="%d"/><connRefOutput v="%d"/></connection>'
                        % ('input' if k == 0 else 'input:%d' % k, src, self.out_uid[src])
                        for k, src in enumerate(inputs))
        self.nodes.append(
            '<compNode><uid v="%d"/><connections>%s</connections><GUILayout><gpos v="%d %d 0"/></GUILayout>'
            '<compOutputs><compOutput><uid v="%d"/><comptype v="%d"/></compOutput></compOutputs>'
            '<compImplementation><compFilter><filter v="pixelprocessor"/><parameters>'
            '<parameter><name v="outputsize"/><relativeTo v="%d"/><paramValue><constantValueInt2 v="%d %d"/></paramValue></parameter>'
            '<parameter><name v="format"/><relativeTo v="0"/><paramValue><constantValueInt32 v="%d"/></paramValue></parameter>'
            '<parameter><name v="colorswitch"/><relativeTo v="0"/><paramValue><constantValueBool v="%d"/></paramValue></parameter>'
            '<parameter><name v="perpixel"/><relativeTo v="0"/><paramValue>%s</paramValue></parameter>'
            '</parameters></compFilter></compImplementation></compNode>'
            % (u, conns, x, y, o, 1 if colour else 2, rel, size[0], size[1], fmt, 1 if colour else 0, dv))
        self.out_uid[u] = o
        self.comments.append((u, name))
        return u

    def output_bridge(self, src, output, x, y):
        u = self.uid()
        self.nodes.append('<compNode><uid v="%d"/><connections><connection><identifier v="inputNodeOutput"/>'
                          '<connRef v="%d"/><connRefOutput v="%d"/></connection></connections><GUILayout>'
                          '<gpos v="%d %d 0"/></GUILayout><compImplementation><compOutputBridge><output v="%d"/>'
                          '</compOutputBridge></compImplementation></compNode>' % (u, src, self.out_uid[src], x, y, output))
        return u


def esc(s):
    return s.replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')


def opt(name, value):
    return '<option><name v="%s"/><value v="%s"/></option>' % (name, value)


def param_float(uid, ident, label, desc, default, lo, hi, step, group=None, widget='slider', visible=None):
    opts = opt('clamp', 1) + opt('default', default) + opt('max', hi) + opt('min', lo)
    if step is not None:
        opts += opt('step', step)
    return ('<paraminput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/><description v="%s"/>'
            '</attributes><type v="256"/><defaultValue><constantValueFloat1 v="%s"/></defaultValue>'
            '<defaultWidget><name v="%s"/><options>%s</options></defaultWidget>%s%s</paraminput>'
            % (ident, uid, esc(label), esc(desc), default, widget, opts,
               '<group v="%s"/>' % esc(group) if group else '',
               '<visibleIf v="%s"/>' % esc(visible) if visible else ''))


def param_int(uid, ident, label, desc, default, lo, hi, group=None):
    opts = opt('clamp', 1) + opt('default', default) + opt('max', hi) + opt('min', lo) + opt('step', 1)
    return ('<paraminput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/><description v="%s"/>'
            '</attributes><type v="16"/><defaultValue><constantValueInt1 v="%d"/></defaultValue>'
            '<defaultWidget><name v="slider"/><options>%s</options></defaultWidget>%s</paraminput>'
            % (ident, uid, esc(label), esc(desc), default, opts, '<group v="%s"/>' % esc(group) if group else ''))


def param_bool(uid, ident, label, desc, default, group=None, visible=None):
    return ('<paraminput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/><description v="%s"/>'
            '</attributes><type v="4"/><defaultValue><constantValueBool v="%d"/></defaultValue>'
            '<defaultWidget><name v="buttons"/><options>%s</options></defaultWidget>%s%s</paraminput>'
            % (ident, uid, esc(label), esc(desc), default, opt('default', default),
               '<group v="%s"/>' % esc(group) if group else '',
               '<visibleIf v="%s"/>' % esc(visible) if visible else ''))


def param_image(uid, ident, label, typ, desc=''):
    default = '<constantValueFloat4 v="0 0 0 0"/>' if typ == 1 else '<constantValueFloat1 v="0"/>'
    return ('<paraminput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/>%s</attributes>'
            '<isConnectable v="1"/><type v="%d"/><defaultValue>%s</defaultValue><defaultWidget><name v=""/>'
            '<options/></defaultWidget></paraminput>'
            % (ident, uid, esc(label), '<description v="%s"/>' % esc(desc) if desc else '', typ, default))


PARAM_IDS = ['samples', 'intensity', 'anisotropy', 'blades', 'angle', 'use_blur_map', 'blur_map_from_input',
             'invert_blur_map', 'softness', 'quality',
             'axis_x', 'axis_y', 'axis_z', 'dither']
PARAM_TYPES = {'samples': T_INT, 'blades': T_INT, 'use_blur_map': T_BOOL, 'blur_map_from_input': T_BOOL,
               'invert_blur_map': T_BOOL}


def params_xml(U, input_type):
    BLUR = 'Blur'
    WS = 'World Space'
    axis_desc = 'Object-space direction that Angle 0 points to (projected on the surface).'
    return ''.join([
        param_image(U['input'], 'input', 'Input', input_type),
        param_image(U['blur_map'], 'blur_map', 'Blur Map', 2,
                    'Grayscale map that scales the blur: white = full Intensity, black = no blur. '
                    'Used when "Use Blur Map" is on and "Use Input As Blur Map" is off.'),
        param_image(U['mesh_position'], 'mesh_position', 'Mesh Position', 1),
        param_image(U['mesh_uv_mask'], 'mesh_uv_mask', 'Mesh UV Mask', 2),
        param_int(U['samples'], 'samples', 'Samples',
                  'Number of blur passes. More passes fill the bokeh shape more smoothly (and cost more).',
                  4, 1, 16, BLUR),
        param_float(U['intensity'], 'intensity', 'Intensity',
                    'Blur radius, in the same units as the 2D Non-Uniform Blur (1 = 1/256 of the UV space), '
                    'converted to world space with the average texel density of the mesh.', 10, 0, 50, 0.01, BLUR),
        param_float(U['anisotropy'], 'anisotropy', 'Anisotropy',
                    'Squashes the blur shape across the Angle direction (1 = a line).', 0, 0, 1, 0.01, BLUR),
        param_int(U['blades'], 'blades', 'Blades', 'Number of sides of the bokeh shape.', 5, 1, 9, BLUR),
        param_float(U['angle'], 'angle', 'Angle',
                    'Rotation of the blur shape around the surface normal, measured from the Reference Axis.',
                    0, 0, 1, None, BLUR, widget='angle'),
        param_bool(U['use_blur_map'], 'use_blur_map', 'Use Blur Map',
                   'Scale the blur per texel (off = uniform blur).', 1, BLUR),
        param_bool(U['blur_map_from_input'], 'blur_map_from_input', 'Use Input As Blur Map',
                   'Use the luminance of the Input itself as the blur map, so no extra texture is needed.', 1, BLUR,
                   visible='input["use_blur_map"]'),
        param_bool(U['invert_blur_map'], 'invert_blur_map', 'Invert Blur Map',
                   'Blur the dark areas instead of the bright ones.', 0, BLUR, visible='input["use_blur_map"]'),
        param_float(U['softness'], 'softness', 'Softness',
                    'Blurs each tap of the bokeh. Higher values are smoother and wrap better around hard '
                    'edges and curved areas; 0 keeps the sharpest bokeh shape.', 0.25, 0, 1, 0.01, WS),
        param_float(U['quality'], 'quality', 'Quality',
                    'Voxel grid: 1 = 48^3, 2 = 64^3, 3 = 96^3, 4 = 128^3. Higher keeps finer detail '
                    'at low Intensity and is slower.', 4, 1, 4, 1, WS),
        param_float(U['axis_x'], 'axis_x', 'Reference Axis X', axis_desc, 0.0, -1, 1, 0.05, WS),
        param_float(U['axis_y'], 'axis_y', 'Reference Axis Y', axis_desc, 1.0, -1, 1, 0.05, WS),
        param_float(U['axis_z'], 'axis_z', 'Reference Axis Z', axis_desc, 0.0, -1, 1, 0.05, WS),
        param_float(U['dither'], 'dither', 'Dither',
                    'Fine noise (in 8-bit steps) that hides banding in smooth gradients. 0 = off.', 1.0, 0, 2, 0.05, WS),
    ])


def gui_comments(g, uid):
    return ''.join('<GUIObject><type v="COMMENT"/><GUIDependency v="NODE?%d"/><GUILayout><gpos v="0 60 -100"/>'
                   '<size v="140 40"/></GUILayout><GUIName v="%s"/><uid v="%d"/>'
                   '<frameColor v="0.196 0.196 0.51 0.196"/></GUIObject>' % (u, esc(name), uid())
                   for u, name in g.comments)


def icon_xml():
    """docs/icon/world_space_non_uniform_blur.png (made by make_icon.py) in the .sbs icon format:
    base64(uint32 big-endian size + zlib(png))."""
    import base64, struct, zlib
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'icon', GRAPH_ID + '.png')
    if not os.path.exists(path):
        return ''
    png = open(path, 'rb').read()
    data = base64.b64encode(struct.pack('>I', len(png)) + zlib.compress(png, 9)).decode()
    return '<icon><datalength v="%d"/><format v="png"/><strdata v="%s"/></icon>' % (len(png), data)


def graph_xml(ident, guid, label, desc, params, primary, out_uid, nodes, gui):
    return ('<graph><identifier v="%s"/><uid v="%d"/>'
            '<graphtype v="filter"/><attributes><label v="%s"/><author v="Doru Bogdan"/>'
            '<authorURL v="https://www.artstation.com/sublime"/><tags v="filter;blur;non uniform;bokeh;world space"/>'
            '<description v="%s"/>%s</attributes><paraminputs>%s</paraminputs><primaryInput v="%d"/>'
            '<graphOutputs><graphoutput><identifier v="output"/><uid v="%d"/><attributes><label v="Output"/>'
            '</attributes></graphoutput></graphOutputs><compNodes>%s</compNodes><baseParameters/>'
            '<GUIObjects>%s</GUIObjects><options><option><name v="defaultParentSize"/><value v="11x11"/></option>'
            '</options><root><rootOutputs><rootOutput><output v="%d"/><format v="0"/><usertag v=""/></rootOutput>'
            '</rootOutputs></root></graph>'
            % (ident, guid, esc(label), esc(desc), icon_xml(), params, primary, out_uid, ''.join(nodes), gui, out_uid))


IMAGE_IDS = ['input', 'blur_map', 'mesh_position', 'mesh_uv_mask']
POS_FORMAT = ('<parameter><name v="format"/><relativeTo v="0"/><paramValue><constantValueInt1 v="3"/>'
              '</paramValue></parameter>')


def build_colour(uid):
    g = Graph(uid)
    g.comments = []
    guid = uid()
    U = {k: uid() for k in IMAGE_IDS + PARAM_IDS + ['output']}

    n_in = g.input_bridge(U['input'], 1, -1600, 0)
    n_pos = g.input_bridge(U['mesh_position'], 1, -1600, 200, POS_FORMAT)
    n_uvm = g.input_bridge(U['mesh_uv_mask'], 2, -1600, 400)
    n_bm = g.input_bridge(U['blur_map'], 2, -1600, 600)

    X = -1400
    n_info = g.pp('WSNB_Size', prog_info(), [n_pos], (-4, -4), rel=1, x=X, y=800)
    n_bbox = g.pp('WSNB_Bounds', prog_bbox(), [n_pos, n_uvm, n_info], (4, 4), x=X + 150, y=700)
    n_reps = g.pp('WSNB_Samples', prog_reps(), [n_pos, n_uvm, n_info], (9, 9), x=X + 150, y=500)
    n_area = g.pp('WSNB_Area', prog_area(), [n_pos, n_uvm, n_reps, n_bbox, n_info], (9, 9), x=X + 300, y=500)
    n_den = g.pp('WSNB_Density', prog_density(), [n_area, n_info], (4, 4), x=X + 300, y=700)
    n_prm = g.pp('WSNB_Grid', prog_grid(), [n_bbox], (4, 4), x=X + 300, y=850)
    n_rad = g.pp('WSNB_Radius', prog_radius(), [n_bbox, n_den, n_prm], (4, 4), x=X + 450, y=850)
    n_vals = g.pp('WSNB_Values', prog_vals(), [n_in, n_pos, n_uvm, n_reps, n_bbox, n_info, n_bm], (9, 9),
                  x=X + 450, y=300)
    n_keys = g.pp('WSNB_Keys', prog_keys(), [n_reps, n_bbox], (9, 9), x=X + 300, y=1000)
    prev = n_keys
    for s, (k, j) in enumerate(sort_stages()):
        prev = g.pp('WSNB_Sort', prog_sort(k, j), [prev], (9, 9), x=X + 450 + 30 * (s % 40), y=1100 + 120 * (s // 40))
    n_sorted = prev
    n_rng = g.pp('WSNB_Ranges', prog_ranges(), [n_sorted], atlas_size_log2(0), x=X + 450, y=1700)
    splat_in = [n_sorted, n_rng, n_reps, n_vals, n_prm, n_bbox, n_area]
    A0 = atlas_size_log2(0)
    n_frm = g.pp('WSNB_Frame', prog_splat_frame(), splat_in, A0, fmt=F16, x=X + 600, y=700)
    n_col = g.pp('WSNB_Splat', prog_splat_colour(), splat_in + [n_frm], A0, fmt=F16, x=X + 600, y=500)
    V = g.pp('WSNB_Voxels', prog_v0(), [n_col, n_frm], A0, fmt=F16, x=X + 700, y=500)
    for k in range(MAX_PASSES):
        x0 = X + 800 + 200 * k
        q = V
        for l in range(1, LEVELS):
            q = g.pp('WSNB_Mip%d' % l, prog_pyramid(l, k), [q], (10, 9), fmt=F16, x=x0, y=300 + 120 * l)
        V = g.pp('WSNB_Pass%d' % (k + 1), prog_pass(k), [V, q, n_frm, n_col, n_rad], A0, fmt=F16, x=x0 + 100, y=200)
    n_out = g.pp('WSNB_Resolve', prog_resolve(), [n_in, n_pos, n_uvm, V, n_prm, n_bm, n_rad], (0, 0), rel=1, fmt=1,
                 x=X + 800 + 200 * MAX_PASSES + 200, y=0)
    g.output_bridge(n_out, U['output'], X + 800 + 200 * MAX_PASSES + 400, 0)
    desc = ('v' + VERSION + ' - Non-uniform (bokeh) blur done in 3D using the mesh position, so it is continuous across UV seams. '
            'The Blur Map scales the blur per texel like the 2D Non-Uniform Blur.')
    return graph_xml(GRAPH_ID, guid, 'World Space Non Uniform Blur', desc, params_xml(U, 1), U['input'],
                     U['output'], g.nodes, gui_comments(g, uid)), len(g.nodes)


def instance_param(uid, name):
    typ = PARAM_TYPES.get(name, T_F1)
    return ('<parameter><name v="%s"/><relativeTo v="0"/><paramValue>%s</paramValue></parameter>'
            % (name, dynamic_expr(uid, get(name, typ))))


def build_grayscale(uid, pkg_uid):
    """Grayscale wrapper: gray -> colour, instance of the colour graph, colour -> gray."""
    g = Graph(uid)
    g.comments = []
    guid = uid()
    U = {k: uid() for k in IMAGE_IDS + PARAM_IDS + ['output']}
    n_in = g.input_bridge(U['input'], 2, -600, 0)
    n_pos = g.input_bridge(U['mesh_position'], 1, -600, 200, POS_FORMAT)
    n_uvm = g.input_bridge(U['mesh_uv_mask'], 2, -600, 400)
    n_bm = g.input_bridge(U['blur_map'], 2, -600, 600)
    p = Program()
    n_col = g.pp('WSNB_ToColour', (p, samplecol(0, POS)), [n_in], (0, 0), rel=1, fmt=1, x=-400, y=0)
    u = uid()
    o = uid()
    conns = [('input', n_col), ('blur_map', n_bm), ('mesh_position', n_pos), ('mesh_uv_mask', n_uvm)]
    g.nodes.append(
        '<compNode><uid v="%d"/><connections>%s</connections><GUILayout><gpos v="-150 200 0"/></GUILayout>'
        '<compOutputs><compOutput><uid v="%d"/><comptype v="1"/></compOutput></compOutputs><compImplementation>'
        '<compInstance><path v="pkg:///%s?dependency=%d"/><parameters>%s</parameters><outputBridgings>'
        '<outputBridging><uid v="%d"/><identifier v="output"/></outputBridging></outputBridgings></compInstance>'
        '</compImplementation></compNode>'
        % (u, ''.join('<connection><identifier v="%s"/><connRef v="%d"/><connRefOutput v="%d"/></connection>'
                      % (k, src, g.out_uid[src]) for k, src in conns),
           o, GRAPH_ID, pkg_uid, ''.join(instance_param(uid, n) for n in PARAM_IDS), o))
    g.out_uid[u] = o
    p = Program()
    n_gray = g.pp('WSNB_ToGray', (p, samplecol(0, POS).x), [u], (0, 0), rel=1, fmt=1, colour=False, x=100, y=0)
    g.output_bridge(n_gray, U['output'], 300, 0)
    desc = 'v' + VERSION + ' - Grayscale version of World Space Non Uniform Blur (for masks).'
    return graph_xml(GRAPH_ID + '_grayscale', guid, 'World Space Non Uniform Blur Grayscale', desc,
                     params_xml(U, 2), U['input'], U['output'], g.nodes, gui_comments(g, uid))


def build(path):
    uid = UID(1700000000)
    pkg_uid = uid()
    colour, n = build_colour(uid)
    gray = build_grayscale(uid, pkg_uid)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><package><identifier v="%s"/><formatVersion v="1.1.0.202302"/>'
           '<updaterVersion v="1.1.0.202302"/><fileUID v="{4f6b2a1e-93c7-4d0a-b8e5-7c2d91a3f604}"/><versionUID v="0"/>'
           '<dependencies><dependency><filename v="?himself"/><uid v="%d"/><type v="package"/><fileUID v="0"/>'
           '<versionUID v="0"/></dependency></dependencies><content>%s%s</content></package>'
           % (GRAPH_ID, pkg_uid, colour, gray))
    with open(path, 'w') as f:
        f.write(xml)
    return n


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                             OUTPUT_NAME)
    n = build(out)
    from check_types import check
    probs = check(out)
    if probs:
        raise SystemExit('type check failed:\n' + '\n'.join(probs))
    print('wrote', out, n, 'nodes', os.path.getsize(out) // 1024, 'KB')
