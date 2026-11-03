"""Generates world_space_non_uniform_blur_v<VERSION>.sbs.

Bump VERSION below for each release; it goes in the file name and the graph descriptions.

World space version of the Non-Uniform Blur. The mesh is voxelized exactly
like World Space Mask Blur v2.0, whose passes are reused from tools/wsb/
(vendored, unmodified):
  surface samples -> coarse grid -> boxes around the pieces (slabs along X,
  runs along Z or Y) -> one sort grid and one voxel grid per box, all packed in
  linear tables with a shared cell size -> samples sorted per box.
On top of that, this filter:
  * splats the surface normals (mesh only) and the input colour / blur amount
    into each box's voxel grid, with sample areas relative to the mean sample
    area (so nothing depends on the mesh's size or units);
  * runs the Non-Uniform Blur passes (Samples x Blades taps on a rotated,
    anisotropic polygon, radius decaying per pass, scaled by the blur amount)
    in each voxel's tangent plane. Taps read a per-box mip pyramid (Softness)
    and stay inside their box: a tap that leaves it reads empty;
  * reads the result back at every texel's 3D position (cubic B-spline in the
    texel's box).
Pieces in different boxes never blur into each other.

Run:  python3 tools/build_wsnub.py  [output.sbs]
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wsb'))
sys.path.insert(0, HERE)
import world_space_blur as ref  # noqa: E402
import build_sbs as B  # noqa: E402
from build_sbs import (Ex, Var, Program, const, lift, vec, fmin, fmax, clamp, floor, fexp, dot, lt, le, gt,  # noqa: E402
                       ge, land, lor, fsqrt, select, get, pos, param, get_bool, sample_col, sample_lum,
                       frame_a, fceil, fabs, flag, on, input_size, snap, index_uv, lin_dims, lin_uv, box_uv, lin3,
                       atlas_uv, box_of, bspline_weights, hash12, my_index, my_grid2, F1, F2, F3, F4, BOOL, FMT_16,
                       FMT_16F, new_uid)

class DedupEmitter(B.FunctionEmitter):
    """B.FunctionEmitter that also merges structurally identical expression nodes. It only
    merges inside one emit() call (one statement, condition or result), so it keeps the
    vendored emitter's rule that statements and the loop's init / condition / body never
    share nodes; within one statement every Get reads the same variable values anyway."""

    def emit(self, ex, cache, depth=0):
        keys = cache.setdefault('__keys__', {})      # id(Ex) -> structure number
        intern = cache.setdefault('__intern__', {})  # (op, type, data, children's numbers) -> number
        uids = cache.setdefault('__uids__', {})      # structure number -> emitted uid

        def key(e):
            k = keys.get(id(e))
            if k is None:
                sig = (e.op, e.typ, e.data, tuple(sorted((p_, key(src)) for p_, src in e.inputs.items())))
                k = intern.setdefault(sig, len(intern))
                keys[id(e)] = k
            return k

        def go(e, d):
            k = key(e)
            if k in uids:
                return uids[k]
            conns = [(port, go(src, d + 1)) for port, src in e.inputs.items()]
            uid = new_uid()
            self.nodes.append([uid, e.op, e.typ, conns, e.data, d])
            uids[k] = uid
            return uid
        return go(ex, depth)


B.FunctionEmitter = DedupEmitter      # used by B.Graph.pixel_processor

VERSION = '2.0'
GRAPH_ID = 'world_space_non_uniform_blur'
OUTPUT_NAME = '%s_v%s.sbs' % (GRAPH_ID, VERSION)
TAGS = 'filter;blur;non uniform;bokeh;world space'
MAX_PASSES = 16
LEVELS = 5                               # voxel mip levels 0..4 per box (0 = the voxel grid itself)
C_DECAY = math.sqrt(-math.log(0.001))    # same pass decay as the 2D Non-Uniform Blur
T_INT = 16


# ----------------------------------------------------------------------------
# table sizes (derived from the reference constants, so reduced test sizes work)

def log2(n):
    return n.bit_length() - 1


def mip_layout():
    """Mip levels 1..4 of every box, packed in one linear QW x QH table: level l
    starts at bases[l]. Capacities hold the worst case of boxes at least 5 voxels
    wide (2 x MARGIN + 1) filling the whole voxel budget AW * AH at level 0, plus
    one entry per box per level."""
    n0 = ref.AW * ref.AH
    nb = ref.S_MAX * ref.R_MAX
    worst = {1: 0.22, 2: 0.066, 3: 0.012, 4: 0.009}     # max over n >= 5 of ceil(n / 2^l)^3 / n^3
    bases, acc = {}, 0
    for l in range(1, LEVELS):
        bases[l] = acc
        acc += int(worst[l] * n0) + nb + 64
    bases[LEVELS] = acc
    w, h = lin_dims(1 << (acc - 1).bit_length())
    return bases, w, h


# ----------------------------------------------------------------------------
# small DSL additions

def fsin(v): return Ex('sin', F1, {'a': lift(v)})
def fcos(v): return Ex('cos', F1, {'a': lift(v)})


def iparam(name):
    """integer graph parameter, as a float"""
    return Ex('tofloat', F1, {'value': Ex('get_integer1', T_INT, data=('constantValueString', name))})


def cross3(a, b):
    return a[1, 2, 0] * b[2, 0, 1] - a[2, 0, 1] * b[1, 2, 0]


def smoothstep01(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - x * 2.0)


def chan(v4, l):
    """component l - 1 of a float4 (l = 1..4, runtime value)"""
    return select(lt(l, 1.5), v4.x, select(lt(l, 2.5), v4.y, select(lt(l, 3.5), v4.z, v4.w)))


def pow2(l):
    """2^l for l = 0..4 (runtime value)"""
    return select(lt(l, 0.5), 1.0, select(lt(l, 1.5), 2.0, select(lt(l, 2.5), 4.0, select(lt(l, 3.5), 8.0, 16.0))))


def inside3(c, n):
    """all components of integer coords c in [0, n - 1]"""
    return land(ge(fmin(fmin(c.x, c.y), c.z), 0.0), le(fmax(fmax(c.x - n.x, c.y - n.y), c.z - n.z), -1.0))


def box_table2(slot, b):
    """entry of box b in an R_MAX x S_MAX table (WSNB_MipBoxes)"""
    R, S = ref.R_MAX, ref.S_MAX
    s = floor(lift(b) / float(R))
    return sample_col(slot, B.grid2_uv(lift(b) - s * float(R), s, R, S))


def box_dims(slot_vb, b):
    """right half of a grid table: (dims.xyz, valid)"""
    return sample_col(slot_vb, box_uv(b, 1.0))


def blur_amount(in_slot, map_slot, uv):
    """per-texel blur scale: 1, the Blur Map input or the luminance of the input itself"""
    e = select(get_bool('blur_map_from_input'), dot(sample_col(in_slot, uv).xyz, const((0.2126, 0.7152, 0.0722))),
               sample_lum(map_slot, uv))
    e = clamp(select(get_bool('invert_blur_map'), 1.0 - e, e), 0.0, 1.0)
    return select(get_bool('use_blur_map'), e, 1.0)


def pass_count():
    """same as the 2D graph: min(Samples, ceil(Intensity * pi))"""
    return fmin(fmax(iparam('samples'), 1.0), fceil(fmax(param('intensity'), 0.0) * math.pi))


def pass_on(k):
    return land(gt(pass_count(), k + 0.5), gt(param('intensity'), 0.0))


def norm_area(slot_area, slot_total, uv):
    """sample area in units of the mean sample area (scale-free, like WSB_PackMask)"""
    return sample_col(slot_area, uv).x / fmax(frame_a(slot_total).x, 1e-30) * float(ref.PMAX * ref.PMAX)


# ----------------------------------------------------------------------------
# surface samples (PMAX x PMAX)

def texel_cross(uv, S, r2, c):
    """dP/dx x dP/dy at texel uv, with the differences in coarse cells (scale-free).
    Same neighbour rules as the mask blur's texel_area (slots: 0 position, 1 uv mask)."""
    p = sample_col(0, uv).xyz

    def inside(q):
        return land(gt(fmin(q.x, q.y), 0.0), lt(fmax(q.x, q.y), 1.0))

    def diff(step):
        f, b = uv + step, uv - step
        fwd = sample_col(0, f).xyz - p
        bwd = p - sample_col(0, b).xyz
        fok = land(land(gt(sample_lum(1, f), ref.UV_INSIDE), lt(dot(fwd, fwd), r2)), inside(f))
        bok = land(land(gt(sample_lum(1, b), ref.UV_INSIDE), lt(dot(bwd, bwd), r2)), inside(b))
        return select(fok, fwd, select(bok, bwd, const(0.0, 3))) / c

    return cross3(diff(vec(1.0 / S.x, 0.0)), diff(vec(0.0, 1.0 / S.y)))


def _patch_loop(result, accs, read, isz_slot, rep_slot, coarse_slot):
    """loop over the SUB x SUB reads of this sample's patch, like WSB_PackArea.
    accs: [(Var, update(taken, uv, ctx) -> Ex)]"""
    S = ref.SUB
    i, cell, rep, r2, isz = Var('i', F1), Var('cell', F2), Var('rep', F4), Var('r2', F1), Var('isz', F2)
    ctx = dict(i=i, cell=cell, rep=rep, r2=r2, isz=isz)

    def take_and_uv():
        uv = snap(B._sub_uv(cell.get(), i.get()), isz.get())
        return B._taken(uv, rep.get(), r2.get(), *read), uv

    init = [(isz, lambda: input_size(isz_slot)), (cell, B._cell),
            (rep, lambda: sample_col(rep_slot, (B._cell() + 0.5) / float(ref.PMAX))),
            (r2, lambda: B._r2(coarse_slot))]
    body = []
    for var, upd in accs:
        init.append((var, (lambda v=var: const(0.0, B.DIM[v.typ]))))
        body.append((var, (lambda u=upd, v=var: v.get() + u(*take_and_uv(), ctx))))
    init.append((i, lambda: const(0.0)))
    body.append((i, lambda: i.get() + 1.0))
    n = float(S * S)
    return Program(result, init=init, cond=lambda: ge(i.get(), n), body=body, counter=i, max_iter=S * S + 8)


def pass_pack_normal():
    """WSNB_PackNormal (PMAX^2, mesh only) -> (sum of dP/dx x dP/dy over the reads taken, 1):
    signed (UV handedness), in coarse-cell units. Slots: 0 position, 1 uv mask, 2 PackPos, 3 Coarse, 4 Size."""
    nsum = Var('nsum', F3)

    def upd(ok, uv, ctx):
        return select(ok, texel_cross(uv, ctx['isz'].get(), ctx['r2'].get(), frame_a(3).w), const(0.0, 3))

    return _patch_loop(lambda: vec(nsum.get(), 1.0), [(nsum, upd)], (0, 1), 4, 2, 3)


def pass_pack_vals():
    """WSNB_PackVals (PMAX^2) -> mean (r, g, b, blur amount) of the reads taken.
    Slots: 0 input, 1 position, 2 uv mask, 3 PackPos, 4 Coarse, 5 Size, 6 blur map."""
    csum, cnt = Var('csum', F4), Var('cnt', F1)

    def upd_c(ok, uv, ctx):
        return select(ok, vec(sample_col(0, uv).xyz, blur_amount(0, 6, uv)), const(0.0, 4))

    def upd_n(ok, uv, ctx):
        return select(ok, 1.0, 0.0)

    def result():
        n = cnt.get()
        return select(gt(n, 0.5), csum.get() / fmax(n, 1.0), const(0.0, 4))

    return _patch_loop(result, [(csum, upd_c), (cnt, upd_n)], (1, 2), 5, 3, 4)


# ----------------------------------------------------------------------------
# splats (AW x AH voxel tables): each voxel gathers the samples of its box next to
# it through the sorted list, like WSB_Splat.
#   slots: 0 sorted keys (Sort1), 1 Cells1, 2 PackPos, 3 PackArea, 4 VoxFit (h), 5 VoxMap,
#          6 VoxBoxes, 7 SortBoxes, 8 SortFit, 9 AreaTotal, 10 PackNormal / PackVals, 11 Normals,
#   and with `fast`: 12 SortedPos, 13 SortedNormal / SortedVals, 14 NearZ (Normals only)

def unit3(n):
    return n / fmax(fsqrt(dot(n, n)), 1e-30)


def splat_program(accs, result, active=None, extra=lambda v: v, fast=False, near=False):
    """accs: [(Var, update(value, weight) -> Ex)]: value is the sample's slot-10 value after
    `extra`, weight the tent weight times the sample's area relative to the mean sample area.

    `fast` gives the same result with less work: samples are read from copies in sorted
    order (SortedPos: position and relative area, slot 13: the value after `extra`), 2 reads
    next to each other instead of 5 scattered ones; with `near`, a voxel whose sort cell has
    no occupied cell within NearZ's reach (more than the walk reaches) skips the walk, whose
    sum would be 0."""
    C, lo, dims, nc = Var('C', F3), Var('lo', F3), Var('dims', F3), Var('nc', F1)
    t, se, hv = Var('t', F1), Var('se', F2), Var('hv', F1)
    sd, so = Var('sd', F3), Var('so', F4)

    def vm():
        return sample_col(5, pos())

    def c_init():
        m = vm()
        return sample_col(6, box_uv(fmax(m.x, 0.0), 0.0)).xyz + (m[1, 2, 3] + 0.5) * frame_a(4).x

    def cell_range(sign):
        o = so.get()
        return clamp(floor((C.get() + frame_a(4).x * sign - o.xyz) / frame_a(8).x), 0.0, sd.get() - 1.0)

    def near_ok():
        o = so.get()
        c0 = clamp(floor((C.get() - o.xyz) / frame_a(8).x), 0.0, sd.get() - 1.0)
        return gt(sample_col(14, atlas_uv(o.w + lin3(c0, sd.get()))).x, 0.5)

    def nc_init():
        d = dims.get()
        ok = gt(vm().x, -0.5)
        if active is not None:
            ok = land(ok, active())
        if fast and near:
            ok = land(ok, near_ok())
        return select(ok, d.x * d.y * d.z, 0.0)

    def value_and_weight():
        if fast:
            here = index_uv(se.get().x)
            sp = sample_col(12, here)
            p, na, val = sp.xyz, sp.w, sample_col(13, here)
        else:
            uv = index_uv(sample_col(0, index_uv(se.get().x)).y)
            p, na, val = sample_col(2, uv).xyz, norm_area(3, 9, uv), extra(sample_col(10, uv))
        q = fmax(1.0 - fabs(p - C.get()) / frame_a(4).x, 0.0)
        w = q.x * q.y * q.z * na
        return val, select(gt(hv.get(), 0.5), w, 0.0)

    def upd_se():
        d = dims.get()
        tt = t.get()
        row = floor(tt / d.x)
        c = lo.get() + vec(tt - d.x * row, row - d.y * floor(row / d.y), floor(row / d.y))
        nxt = sample_col(1, atlas_uv(so.get().w + lin3(c, sd.get()))).xy
        return select(gt(hv.get(), 0.5), se.get() + vec(1.0, 0.0), nxt)

    init = [(so, lambda: sample_col(7, box_uv(fmax(vm().x, 0.0), 0.0))),
            (sd, lambda: sample_col(7, box_uv(fmax(vm().x, 0.0), 1.0)).xyz),
            (C, c_init), (lo, lambda: cell_range(-1.0)),
            (dims, lambda: cell_range(1.0) - lo.get() + 1.0), (nc, nc_init),
            (t, lambda: const(0.0)), (se, lambda: const(0.0, 2)), (hv, lambda: const(0.0))]
    # hv first: the accumulators, se and t all decide on the state before this step
    body = [(hv, lambda: select(lt(se.get().x, se.get().y), 1.0, 0.0))]
    for var, upd in accs:
        init.append((var, (lambda v=var: const(0.0, B.DIM[v.typ]))))
        body.append((var, (lambda u=upd, v=var: v.get() + u(*value_and_weight()))))
    body += [(se, upd_se), (t, lambda: select(gt(hv.get(), 0.5), t.get(), t.get() + 1.0))]
    return Program(result, init=init, cond=lambda: land(ge(se.get().x, se.get().y), ge(t.get(), nc.get())),
                   body=body, counter=t, max_iter=65536)


def pass_normals(fast=False):
    """WSNB_Normals (AW x AH, mesh + Quality) -> (unit surface normal, sum w) per voxel, 0 if empty.
    The normal is the dominant eigenvector of the area-weighted normal tensor (sign-free, so
    mirrored UV islands don't cancel out), oriented like the summed signed normals."""
    wsum, nsum, tdia, toff = Var('wsum', F1), Var('nsum', F3), Var('tdia', F3), Var('toff', F3)

    def result():
        d, o, ns, w = tdia.get(), toff.get(), nsum.get(), wsum.get()
        v = select(land(ge(d.x, d.y), ge(d.x, d.z)), const((1.0, 0.0, 0.0)),
                   select(ge(d.y, d.z), const((0.0, 1.0, 0.0)), const((0.0, 0.0, 1.0))))
        for _ in range(8):                                   # power iteration
            mv = vec(d.x * v.x + o.x * v.y + o.y * v.z, o.x * v.x + d.y * v.y + o.z * v.z,
                     o.y * v.x + o.z * v.y + d.z * v.z)
            v = mv / fmax(fsqrt(dot(mv, mv)), 1e-30)
        v = select(lt(dot(v, ns), 0.0), -v, v)
        return select(gt(w, 0.0), vec(v, w), const(0.0, 4))

    accs = [(wsum, lambda n, w: w), (nsum, lambda n, w: n.xyz * w), (tdia, lambda n, w: n.xyz * n.xyz * w),
            (toff, lambda n, w: vec(n.x * n.y, n.x * n.z, n.y * n.z) * w)]
    return splat_program(accs, result, extra=lambda v: unit3(v.xyz), fast=fast, near=True)


def pass_splat_colour(fast=False):
    """WSNB_Splat (AW x AH) -> (sum colour * w, sum blur amount * w); sum w is Normals.w.
    Voxels with no surface nearby (Normals.w = 0) skip the loop."""
    acc = Var('acc', F4)
    return splat_program([(acc, lambda v, w: v * w)], lambda: acc.get(),
                         active=lambda: gt(sample_col(11, pos()).w, 0.0), fast=fast)


def pass_sorted_copy(value=None):
    """WSNB_SortedPos / SortedNormal (mesh only) / SortedVals (PMAX^2): the splat's per-sample
    values in sorted order, so a splat reads them next to each other. value(slot, uv) -> Ex;
    default: slot 1 as is. Slot 0: sorted keys."""
    value = value or (lambda slot, uv: sample_col(slot, uv))
    return Program(lambda: value(1, index_uv(sample_col(0, pos()).y)))


def pass_voxels():
    """WSNB_Voxels (AW x AH) -> (sum colour * w, sum w): what the passes start from.
    Slots: 0 Splat, 1 Normals."""
    return Program(lambda: vec(sample_col(0, pos()).xyz, sample_col(1, pos()).w))


def pass_radius():
    """WSNB_Radius (16 x 16, Intensity / Quality) -> (max radius in voxels, length per UV, h, 1).
    Intensity is in 2D Non-Uniform Blur units: 1 = 1/256 of the UV space, at the mesh's average
    3D length per UV unit (from its total area). Slots: 0 AreaTotal, 1 VoxFit, 2 Size."""
    def result():
        tot = frame_a(0)
        S = input_size(2)
        luv = fsqrt(tot.x * S.x * S.y / fmax(tot.y, 1e-12))
        hq = frame_a(1).x
        return vec(fmax(param('intensity'), 0.0) * (1.0 / 256.0) * luv / hq, luv, hq, 1.0)
    return Program(result)


# ----------------------------------------------------------------------------
# per-box mip pyramid, levels 1..4 packed in one table

def pass_mip_boxes():
    """WSNB_MipBoxes (R_MAX x S_MAX, Quality) -> start of box b's mip levels 1..4 in the packed
    mip table (level l of a box with n voxels per axis has ceil(n / 2^l)). Slot 0: VoxBoxes."""
    bases, _, _ = mip_layout()
    R, S = ref.R_MAX, ref.S_MAX
    k, acc, box = Var('k', F1), Var('acc', F4), Var('box', F1)

    def vols():
        tb = box_dims(0, k.get())
        v = []
        for l in range(1, LEVELS):
            d = fceil(tb.xyz * (1.0 / 2 ** l))
            v.append(d.x * d.y * d.z)
        return select(gt(tb.w, 0.5), vec(*v), const(0.0, 4))

    def box_init():
        r, s = my_grid2(R, S)
        return s * float(R) + r

    return Program(lambda: acc.get() + const([float(bases[l]) for l in range(1, LEVELS)]),
                   init=[(box, box_init), (acc, lambda: const(0.0, 4)), (k, lambda: const(0.0))],
                   cond=lambda: ge(k.get(), box.get()),
                   body=[(acc, lambda: acc.get() + vols()), (k, lambda: k.get() + 1.0)],
                   counter=k, max_iter=R * S + 8)


def mip_level_of(t):
    bases, _, _ = mip_layout()
    return 1.0 + flag(ge(t, float(bases[2]))) + flag(ge(t, float(bases[3]))) + flag(ge(t, float(bases[4])))


def pass_mip_map():
    """WSNB_MipMap (packed mip table, Quality): this entry's box (binary search for the first box
    whose level ends past it) and (i, j, k) in that level -> (box or -1, i, j, k).
    Slots: 0 MipBoxes, 1 VoxBoxes."""
    _, QW, QH = mip_layout()
    nb = float(ref.R_MAX * ref.S_MAX)
    lo, hi, t, md, past = Var('lo', F1), Var('hi', F1), Var('t', F1), Var('md', F1), Var('past', F1)

    def dims_l(b):
        tb = box_dims(1, b)
        return fceil(tb.xyz / pow2(mip_level_of(t.get()))), tb.w

    def end(b):
        d, valid = dims_l(b)
        return chan(box_table2(0, b), mip_level_of(t.get())) + select(gt(valid, 0.5), d.x * d.y * d.z, 0.0)

    def result():
        b = fmin(lo.get(), nb - 1.0)
        d, _ = dims_l(b)
        loc = t.get() - chan(box_table2(0, b), mip_level_of(t.get()))
        row = floor(loc / d.x)
        kk = floor(row / d.y)
        return vec(select(lt(lo.get(), nb), b, -1.0), loc - row * d.x, row - kk * d.y, kk)

    return Program(result,
                   init=[(t, lambda: my_index(QW, QH)), (lo, lambda: const(0.0)), (hi, lambda: const(nb)),
                         (md, lambda: const(0.0)), (past, lambda: const(0.0))],
                   cond=lambda: ge(lo.get(), hi.get()),
                   # md and past first: lo and hi both decide on the same middle box
                   body=[(md, lambda: floor((lo.get() + hi.get()) * 0.5)),
                         (past, lambda: flag(gt(end(md.get()), t.get()))),
                         (hi, lambda: select(on(past.get()), md.get(), hi.get())),
                         (lo, lambda: select(on(past.get()), lo.get(), md.get() + 1.0))],
                   counter=lo, max_iter=(ref.R_MAX * ref.S_MAX).bit_length() + 8)


def pass_pyramid(l, k):
    """WSNB_Mip<l> (packed mip table): level l of every box = [1 3 3 1]/8 filter, stride 2, of
    level l - 1 of the same box (taps outside the box are empty); lower levels are copied. Only
    computed when pass k uses fat taps. Slots: 0 voxels (l = 1) or the previous mip table,
    1 MipMap, 2 MipBoxes, 3 VoxBoxes."""
    bases, QW, QH = mip_layout()
    i, acc, act, ijk = Var('i', F1), Var('acc', F4), Var('act', F1), Var('ijk', F3)
    dp, po = Var('dp', F3), Var('po', F1)

    def t():
        return my_index(QW, QH)

    def bx():
        return fmax(sample_col(1, pos()).x, 0.0)

    def parent_dims():
        n0 = box_dims(3, bx()).xyz
        return n0 if l == 1 else fceil(n0 * (1.0 / 2 ** (l - 1)))

    def parent_off():
        return sample_col(3, box_uv(bx(), 0.0)).w if l == 1 else chan(box_table2(2, bx()), float(l - 1))

    def active():
        inreg = land(ge(t(), float(bases[l])), lt(t(), float(bases[l + 1])))
        return flag(land(land(pass_on(k), gt(param('softness'), 0.0)),
                         land(inreg, gt(sample_col(1, pos()).x, -0.5))))

    def upd():
        ii = i.get()
        o = vec(ii - floor(ii / 4.0) * 4.0, floor(ii / 4.0) - floor(ii / 16.0) * 4.0, floor(ii / 16.0))

        def w1(x):
            return select(land(gt(x, 0.5), lt(x, 2.5)), 3.0, 1.0)
        wv = w1(o.x) * w1(o.y) * w1(o.z) * (1.0 / 512.0)
        src = ijk.get() * 2.0 + o - 1.0
        d = dp.get()
        idx = po.get() + lin3(clamp(src, 0.0, d - 1.0), d)
        uv = atlas_uv(idx) if l == 1 else lin_uv(idx, QW, QH)
        return acc.get() + select(inside3(src, d), sample_col(0, uv) * wv, const(0.0, 4))

    def result():
        if l == 1:
            return select(on(act.get()), acc.get(), const(0.0, 4))
        lower = lt(t(), float(bases[l]))
        return select(on(act.get()), acc.get(), select(lower, sample_col(0, pos()), const(0.0, 4)))

    return Program(result,
                   init=[(ijk, lambda: sample_col(1, pos())[1, 2, 3]), (dp, parent_dims), (po, parent_off),
                         (act, active), (acc, lambda: const(0.0, 4)), (i, lambda: const(0.0))],
                   cond=lambda: lor(ge(i.get(), 64.0), lt(act.get(), 0.5)),
                   body=[(acc, upd), (i, lambda: i.get() + 1.0)],
                   counter=i, max_iter=72)


# ----------------------------------------------------------------------------
# the Non-Uniform Blur passes (AW x AH)
#   slots: 0 voxels, 1 mip table, 2 Normals, 3 Splat, 4 Radius, 5 VoxMap, 6 VoxBoxes, 7 MipBoxes

def corners(g, d, uv_of):
    """trilinear read at voxel-centre coords g of a d-sized grid; corners outside it are empty"""
    i0 = floor(g)
    f = g - i0
    acc = None
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (f.x if dx else 1.0 - f.x) * (f.y if dy else 1.0 - f.y) * (f.z if dz else 1.0 - f.z)
                c = i0 + const([float(dx), float(dy), float(dz)])
                v = select(inside3(c, d), uv_of(clamp(c, 0.0, d - 1.0)), const(0.0, 4)) * w
                acc = v if acc is None else acc + v
    return acc


def fat_tap(g, lev, n0, o0, mo):
    """(sum c*w, sum w) at level-0 voxel coords g of the voxel's box, blurred to mip level lev
    (0..4): two trilinear reads. Coarser levels are rescaled by 2^l so a tap on a flat surface
    weighs the same at any level."""
    _, QW, QH = mip_layout()

    def tri0():
        return corners(g, n0, lambda c: sample_col(0, atlas_uv(o0 + lin3(c, n0))))

    def triq(l):
        s = pow2(l)
        d = fceil(n0 / s)
        ol = chan(mo, l)
        return corners((g + 0.5) / s - 0.5, d, lambda c: sample_col(1, lin_uv(ol + lin3(c, d), QW, QH)))

    l0 = fmin(floor(lev), 3.0)
    f = lev - l0
    a = select(lt(l0, 0.5), tri0(), triq(fmax(l0, 1.0)) * pow2(l0))
    b = triq(l0 + 1.0) * pow2(l0 + 1.0)
    return a * (1.0 - f) + b * f


def approx_log2_clamped(x):
    """piecewise-linear log2(x), clamped to [0, LEVELS - 1]"""
    x = clamp(x, 1.0, 2.0 ** (LEVELS - 1))
    o = flag(ge(x, 2.0)) + flag(ge(x, 4.0)) + flag(ge(x, 8.0)) + flag(ge(x, 16.0))
    base = select(ge(x, 16.0), 16.0, select(ge(x, 8.0), 8.0, select(ge(x, 4.0), 4.0, select(ge(x, 2.0), 2.0, 1.0))))
    return o + (x / base - 1.0)


def pass_blur(k):
    TWO_PI = 2.0 * math.pi
    vm, vv, n0, o0, mo = Var('vm', F4), Var('vv', F4), Var('n0', F3), Var('o0', F1), Var('mo', F4)
    r, act, t1, t2, lev, acc, j = (Var('r', F1), Var('act', F1), Var('t1', F3), Var('t2', F3), Var('lev', F1),
                                   Var('acc', F4), Var('j', F1))

    def bx():
        return fmax(vm.get().x, 0.0)

    def nrm():
        return sample_col(2, pos()).xyz

    def r_init():
        e = clamp(sample_col(3, pos()).w / fmax(sample_col(2, pos()).w, 1e-30), 0.0, 1.0)
        return frame_a(4).x * e * fexp(-(2.0 * k * C_DECAY) / fmax(pass_count(), 1.0))

    def act_init():
        return flag(land(land(pass_on(k), gt(vm.get().x, -0.5)), land(gt(vv.get().w, 0.0), gt(r.get(), 0.05))))

    def t1_init():
        # tangent frame: the reference axis projected on the surface (another axis near its pole)
        n = nrm()
        ax = vec(param('axis_x'), param('axis_y'), param('axis_z'))
        l2 = dot(ax, ax)
        D = select(gt(l2, 1e-8), ax / fmax(fsqrt(l2), 1e-6), const((0.0, 1.0, 0.0)))
        t = D - n * dot(D, n)
        alt = select(lt(fabs(D.x), 0.9), const((1.0, 0.0, 0.0)), const((0.0, 0.0, 1.0)))
        tb = alt - n * dot(alt, n)
        t = select(gt(dot(t, t), 0.04), t, tb)
        return t / fmax(fsqrt(dot(t, t)), 1e-6)

    def tap():
        nbl = clamp(iparam('blades'), 1.0, 9.0)
        sm = fmax(iparam('samples'), 1.0)
        jj = j.get()
        # the 2D graph's ellipse: inner rotation (1 / Samples) / (pass + 1) turns, then Blades corners
        th = (1.0 / sm) * (TWO_PI / (k + 1.0)) + jj * (TWO_PI / nbl)
        an = clamp(param('anisotropy'), 0.0, 1.0)
        phi = param('angle') * -TWO_PI
        s1 = (1.0 - an) * fsin(th)
        c1 = fcos(th)
        X = s1 * fsin(phi) + c1 * fcos(phi)
        Y = s1 * fcos(phi) - fsin(phi) * c1
        off = (t1.get() * X + t2.get() * Y) * (r.get() * select(gt(jj, 0.5), 1.0, 0.0))
        return acc.get() + fat_tap(vm.get()[1, 2, 3] + off, lev.get(), n0.get(), o0.get(), mo.get())

    def result():
        return select(on(act.get()), acc.get() / (clamp(iparam('blades'), 1.0, 9.0) + 1.0), vv.get())

    return Program(
        result,
        init=[(vm, lambda: sample_col(5, pos())), (vv, lambda: sample_col(0, pos())),
              (n0, lambda: box_dims(6, bx()).xyz), (o0, lambda: sample_col(6, box_uv(bx(), 0.0)).w),
              (mo, lambda: box_table2(7, bx())), (r, r_init), (act, act_init), (t1, t1_init),
              (t2, lambda: cross3(nrm(), t1.get())),
              (lev, lambda: approx_log2_clamped(fmax(param('softness'), 0.0) * 0.4 * r.get() / 0.64)),
              (acc, lambda: const(0.0, 4)), (j, lambda: const(0.0))],
        # j = 0 is the centre tap, j = 1..Blades the polygon corners
        cond=lambda: lor(gt(j.get(), clamp(iparam('blades'), 1.0, 9.0)), lt(act.get(), 0.5)),
        body=[(acc, tap), (j, lambda: j.get() + 1.0)],
        counter=j, max_iter=12)


# ----------------------------------------------------------------------------
# resolve (input size, colour, 16 bit)
#   slots: 0 input, 1 position, 2 uv mask, 3 blurred voxels, 4 VoxFit (h), 5 Coarse, 6 SlabX,
#          7 RunTab, 8 VoxBoxes, 9 blur map, 10 Radius

def dither3(c):
    """triangular noise of +-Dither 8-bit steps, faded out near 0 and 1 (per channel)"""
    px = floor(pos() * get('$size', F2))
    n = hash12(px) + hash12(px + const((37.0, 91.0))) - 1.0
    k = clamp(fmin(c, 1.0 - c) * 255.0, 0.0, 1.0)
    return clamp(c + k * (n * param('dither') * (1.0 / 255.0)), 0.0, 1.0)


def pass_resolve(fast=False):
    """Tricubic B-spline lookup of the texel's box grid (64 reads), blended with the input by the
    local blur amount, dithered; written as 16 bit. Texels with nothing to blur (outside the UV
    islands, in no box, or with a black blur map) skip the reads and keep the input.

    `fast` gives the same result with less work: 16 loop steps of 4 reads instead of 64 of 1,
    added in the same order with the same weights, so the sum is bit-identical."""
    p, bx, fnd, ta, n = Var('p', F3), Var('bx', F1), Var('fnd', F1), Var('ta', F4), Var('n', F3)
    g0, gf, al, S, i = Var('g0', F3), Var('gf', F3), Var('al', F1), Var('S', F4), Var('i', F1)
    W = {(ax, a): Var('w%s%d' % (ax, a), F1) for ax in 'xyz' for a in range(4)}

    def g():
        return clamp((p.get() - ta.get().xyz) / frame_a(4).x - 0.5, 1.0, n.get() - 3.0)

    def al_init():
        rv = frame_a(10).x
        ok = land(gt(sample_lum(2, pos()), 0.5), on(fnd.get()))
        return select(ok, smoothstep01(rv * blur_amount(0, 9, pos()) * 0.5), 0.0)

    init = [(p, lambda: sample_col(1, pos()).xyz),
            (bx, lambda: box_of(p.get(), 5, 6, 7)[0]), (fnd, lambda: flag(box_of(p.get(), 5, 6, 7)[1])),
            (ta, lambda: sample_col(8, box_uv(bx.get(), 0.0))), (n, lambda: box_dims(8, bx.get()).xyz),
            (g0, lambda: floor(g())), (gf, lambda: g() - floor(g()))]
    for ax in 'xyz':
        for a in range(4):
            init.append((W[(ax, a)], (lambda ax=ax, a=a: bspline_weights(gf.get()['xyz'.index(ax)])[a])))
    init += [(al, al_init), (S, lambda: const(0.0, 4)), (i, lambda: const(0.0))]

    def pick(ax, x):
        return select(lt(x, 0.5), W[(ax, 0)].get(), select(lt(x, 1.5), W[(ax, 1)].get(),
                                                             select(lt(x, 2.5), W[(ax, 2)].get(), W[(ax, 3)].get())))

    def upd():
        ii = i.get()
        a, b, c = ii - floor(ii / 4.0) * 4.0, floor(ii / 4.0) - floor(ii / 16.0) * 4.0, floor(ii / 16.0)
        w = pick('x', a) * pick('y', b) * pick('z', c)
        cell = g0.get() + vec(a, b, c) - 1.0
        return S.get() + sample_col(3, atlas_uv(ta.get().w + lin3(cell, n.get()))) * w

    def upd4():
        kk = i.get()
        c = floor(kk / 4.0)
        b = kk - c * 4.0
        acc = S.get()
        for a in range(4):
            w = pick('x', const(float(a))) * pick('y', b) * pick('z', c)
            cell = g0.get() + vec(float(a), b, c) - 1.0
            acc = acc + sample_col(3, atlas_uv(ta.get().w + lin3(cell, n.get()))) * w
        return acc

    def result():
        s = S.get()
        src = sample_col(0, pos())
        res = s.xyz / fmax(s.w, 1e-12)
        out = src.xyz + (res - src.xyz) * al.get()
        ok = land(gt(al.get(), 0.0), gt(s.w, 1e-6))          # s.w is relative to the mean sample area
        return select(ok, vec(dither3(out), src.w), src)

    steps = 16.0 if fast else 64.0
    return Program(result, init=init, cond=lambda: lor(ge(i.get(), steps), le(al.get(), 0.0)),
                   body=[(S, upd4 if fast else upd), (i, lambda: i.get() + 1.0)], counter=i, max_iter=int(steps) + 8)


# ----------------------------------------------------------------------------
# graph assembly

def esc(s):
    return s.replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')


def opt(name, value):
    return '<option><name v="%s"/><value v="%s"/></option>' % (name, value)


def param_float(ident, label, desc, default, lo, hi, step, group=None, widget='slider', visible=None):
    opts = opt('clamp', 1) + opt('default', default) + opt('max', hi) + opt('min', lo)
    if step is not None:
        opts += opt('step', step)
    return ('<paraminput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/><description v="%s"/>'
            '</attributes><type v="256"/><defaultValue><constantValueFloat1 v="%s"/></defaultValue>'
            '<defaultWidget><name v="%s"/><options>%s</options></defaultWidget>%s%s</paraminput>'
            % (ident, new_uid(), esc(label), esc(desc), default, widget, opts,
               '<group v="%s"/>' % esc(group) if group else '',
               '<visibleIf v="%s"/>' % esc(visible) if visible else ''))


def param_int(ident, label, desc, default, lo, hi, group=None):
    opts = opt('clamp', 1) + opt('default', default) + opt('max', hi) + opt('min', lo) + opt('step', 1)
    return ('<paraminput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/><description v="%s"/>'
            '</attributes><type v="16"/><defaultValue><constantValueInt1 v="%d"/></defaultValue>'
            '<defaultWidget><name v="slider"/><options>%s</options></defaultWidget>%s</paraminput>'
            % (ident, new_uid(), esc(label), esc(desc), default, opts,
               '<group v="%s"/>' % esc(group) if group else ''))


def param_bool(ident, label, desc, default, group=None, visible=None):
    return ('<paraminput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/><description v="%s"/>'
            '</attributes><type v="4"/><defaultValue><constantValueBool v="%d"/></defaultValue>'
            '<defaultWidget><name v="buttons"/><options>%s</options></defaultWidget>%s%s</paraminput>'
            % (ident, new_uid(), esc(label), esc(desc), default, opt('default', default),
               '<group v="%s"/>' % esc(group) if group else '',
               '<visibleIf v="%s"/>' % esc(visible) if visible else ''))


PARAM_IDS = ['samples', 'intensity', 'anisotropy', 'blades', 'angle', 'use_blur_map', 'blur_map_from_input',
             'invert_blur_map', 'softness', 'quality', 'axis_x', 'axis_y', 'axis_z', 'dither']
PARAM_TYPES = {'samples': T_INT, 'blades': T_INT, 'use_blur_map': BOOL, 'blur_map_from_input': BOOL,
               'invert_blur_map': BOOL}


def params_xml():
    BLUR = 'Blur'
    WS = 'World Space'
    axis_desc = 'Object-space direction that Angle 0 points to (projected on the surface).'
    levels = ', '.join('%d = %d^3' % (k, n) for k, n in sorted(ref.QUALITY.items()))
    return ''.join([
        param_int('samples', 'Samples',
                  'Number of blur passes. More passes fill the bokeh shape more smoothly (and cost more).',
                  4, 1, 16, BLUR),
        param_float('intensity', 'Intensity',
                    'Blur radius, in the same units as the 2D Non-Uniform Blur (1 = 1/256 of the UV space), '
                    "at the mesh's average texel density. Doesn't depend on how far apart the pieces are.",
                    10, 0, 50, 0.01, BLUR),
        param_float('anisotropy', 'Anisotropy',
                    'Squashes the blur shape across the Angle direction (1 = a line).', 0, 0, 1, 0.01, BLUR),
        param_int('blades', 'Blades', 'Number of sides of the bokeh shape.', 5, 1, 9, BLUR),
        param_float('angle', 'Angle',
                    'Rotation of the blur shape around the surface normal, measured from the Reference Axis.',
                    0, 0, 1, None, BLUR, widget='angle'),
        param_bool('use_blur_map', 'Use Blur Map', 'Scale the blur per texel (off = uniform blur).', 1, BLUR),
        param_bool('blur_map_from_input', 'Use Input As Blur Map',
                   'Use the luminance of the Input itself as the blur map, so no extra texture is needed.', 1, BLUR,
                   visible='input["use_blur_map"]'),
        param_bool('invert_blur_map', 'Invert Blur Map', 'Blur the dark areas instead of the bright ones.', 0, BLUR,
                   visible='input["use_blur_map"]'),
        param_float('softness', 'Softness',
                    'Blurs each tap of the bokeh. Higher values are smoother and wrap better around hard '
                    'edges and curved areas; 0 keeps the sharpest bokeh shape.', 0.25, 0, 1, 0.01, WS),
        param_float('quality', 'Quality',
                    'Voxel budget, shared by all pieces (%s voxels). Higher keeps finer detail at low '
                    'Intensity and is slower.' % levels, max(ref.QUALITY), 1, max(ref.QUALITY), 1, WS),
        param_float('axis_x', 'Reference Axis X', axis_desc, 0.0, -1, 1, 0.05, WS),
        param_float('axis_y', 'Reference Axis Y', axis_desc, 1.0, -1, 1, 0.05, WS),
        param_float('axis_z', 'Reference Axis Z', axis_desc, 0.0, -1, 1, 0.05, WS),
        param_float('dither', 'Dither',
                    'Fine noise (in 8-bit steps) that hides banding in smooth gradients. 0 = off.', 1.0, 0, 2, 0.05,
                    WS),
    ])


def image_input(g, ident, label, color, desc='', fmt32=False, primary=False, x=0, y=0):
    uid, node_uid, out_uid = new_uid(), new_uid(), new_uid()
    default = '<constantValueFloat4 v="0 0 0 0"/>' if color else '<constantValueFloat1 v="0"/>'
    g.paraminputs.append(
        '<paraminput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/>%s</attributes>'
        '<isConnectable v="1"/><type v="%d"/><defaultValue>%s</defaultValue><defaultWidget><name v=""/>'
        '<options/></defaultWidget></paraminput>'
        % (ident, uid, esc(label), '<description v="%s"/>' % esc(desc) if desc else '', 1 if color else 2, default))
    if primary:
        g.primary = uid
    params = B.fmt_param('format', 'constantValueInt1', 3) if fmt32 else ''
    g.nodes.append(
        '<compNode><uid v="%d"/><GUILayout><gpos v="%d %d 0"/></GUILayout><compOutputs><compOutput><uid v="%d"/>'
        '<comptype v="%d"/></compOutput></compOutputs><compImplementation><compInputBridge><entry v="%d"/>'
        '<parameters>%s</parameters></compInputBridge></compImplementation></compNode>'
        % (node_uid, x, y, out_uid, 1 if color else 2, uid, params))
    return node_uid, out_uid


def inputs(g, color_input):
    return dict(
        input=image_input(g, 'input', 'Input', color_input, primary=True, x=-1600, y=0),
        blur_map=image_input(g, 'blur_map', 'Blur Map', False,
                             'Grayscale map that scales the blur: white = full Intensity, black = no blur. '
                             'Used when "Use Blur Map" is on and "Use Input As Blur Map" is off.', x=-1600, y=600),
        mesh_position=image_input(g, 'mesh_position', 'Mesh Position', True, fmt32=True, x=-1600, y=200),
        mesh_uv_mask=image_input(g, 'mesh_uv_mask', 'Mesh UV Mask', False, x=-1600, y=400))


def output_color(g, ident, label, src, x=0, y=0):
    uid, node_uid = new_uid(), new_uid()
    g.outputs.append('<graphoutput><identifier v="%s"/><uid v="%d"/><attributes><label v="%s"/></attributes>'
                     '</graphoutput>' % (ident, uid, label))
    n, o = src
    g.nodes.append('<compNode><uid v="%d"/><connections><connection><identifier v="inputNodeOutput"/>'
                   '<connRef v="%d"/><connRefOutput v="%d"/></connection></connections><GUILayout>'
                   '<gpos v="%d %d 0"/></GUILayout><compImplementation><compOutputBridge><output v="%d"/>'
                   '</compOutputBridge></compImplementation></compNode>' % (node_uid, n, o, x, y, uid))
    g.root_output = uid


def graph_xml(g, desc):
    return g.xml(desc).replace('<tags v="filter;mask;blur;world space"/>', '<tags v="%s"/>' % TAGS)


def icon_path():
    return os.path.join(os.path.dirname(HERE), 'docs', 'icon', GRAPH_ID + '.png')


def build_colour(fast=True):
    g = B.Graph(GRAPH_ID, 'World Space Non Uniform Blur')
    if os.path.exists(icon_path()):
        g.set_icon(icon_path())
    I = inputs(g, True)
    g.paraminputs.append(params_xml())
    src, bmap, position, uvmask = I['input'], I['blur_map'], I['mesh_position'], I['mesh_uv_mask']

    def pp(name, prog, ins, size, x, y, **kw):
        return g.pixel_processor(name, prog, ins, size, x=x, y=y, **kw)

    def lin(n):
        w, h = lin_dims(n)
        return log2(w), log2(h)

    P, S, R, Dn = ref.PMAX, ref.S_MAX, ref.R_MAX, ref.DMAX
    lp = (log2(P), log2(P))
    atl = (log2(ref.AW), log2(ref.AH))
    one = (4, 4)
    boxt = (log2(2 * R), log2(S))
    _, QW, QH = mip_layout()
    qt = (log2(QW), log2(QH))

    # --- mesh only: World Space Mask Blur v2.0, as is ---
    size = g.pixel_processor('WSB_Size', B.pass_size(), [position], None, x=-1400, y=600,
                             rel_log2=(B.SIZE_LOG2, B.SIZE_LOG2))
    bounds = pp('WSB_Bounds', B.pass_bounds(), [position, uvmask, size], one, -1250, 500)
    coarse = pp('WSB_Coarse', B.pass_coarse(), [bounds], one, -1150, 500)
    pack_pos = pp('WSB_PackPos', B.pass_pack_pos(), [position, uvmask, size], lp, -1250, 300)
    pack_area = pp('WSB_PackArea', B.pass_pack_area(), [position, uvmask, pack_pos, coarse, size], lp, -1100, 400)
    area_blocks = pp('WSB_AreaBlocks', B.pass_area_blocks(), [pack_area], (log2(ref.AREA_GRID),) * 2, -1000, 450)
    area_total = pp('WSB_AreaTotal', B.pass_area_total(), [area_blocks], one, -900, 450)
    keys = pp('WSB_Keys0', B.pass_keys0(), [pack_pos, coarse], lp, -1100, 850)
    for n, (k, j) in enumerate(ref.sort_steps()):
        keys = pp('WSB_Sort0 %d/%d' % (k, j), B.pass_sort_step(k, j), [keys], lp, -950 + 30 * (n % 40),
                  850 + 120 * (n // 40))
    cells0 = pp('WSB_Cells0', B.pass_cells(ref.CW, ref.CH), [keys], (log2(ref.CW), log2(ref.CH)), -650, 1600)
    cellb = pp('WSB_CellBounds', B.pass_cell_bounds(), [keys, cells0, pack_pos], (log2(2 * ref.CW), log2(ref.CH)),
               -500, 1600)
    colxz = pp('WSB_ColXZ', B.pass_columns(True), [cellb, coarse], (log2(2 * Dn), log2(Dn)), -350, 1550)
    colxy = pp('WSB_ColXY', B.pass_columns(False), [cellb, coarse], (log2(2 * Dn), log2(Dn)), -350, 1650)
    occx = pp('WSB_OccX', B.pass_occ_x(), [colxz, coarse], lin(Dn), -200, 1500)
    slabx = pp('WSB_SlabX', B.pass_slab_x(), [occx], lin(Dn), -100, 1500)
    slabr = pp('WSB_SlabRange', B.pass_slab_range(), [slabx, coarse], lin(S), 0, 1500)
    occzy = pp('WSB_OccZY', B.pass_occ_zy(), [colxz, colxy, slabr, coarse], (log2(Dn), log2(S)), 100, 1600)
    runtab = pp('WSB_RunTab', B.pass_run_tab(), [occzy], (log2(Dn), log2(S)), 200, 1600)
    runr = pp('WSB_RunRange', B.pass_run_range(), [runtab], (log2(R), log2(S)), 300, 1600)
    bw, bh = lin_dims(S * R * Dn)
    boxrow = pp('WSB_BoxRow', B.pass_box_row(), [colxz, colxy, slabr, runr], (log2(2 * bw), log2(bh)), 400, 1600)
    boxes = pp('WSB_Boxes', B.pass_boxes(), [boxrow, slabr], boxt, 500, 1600)
    sortfit = pp('WSB_SortFit', B.pass_fit(lambda: const(float(ref.AW * ref.AH))), [boxes, bounds], one, 600, 1500)
    sortboxes = pp('WSB_SortBoxes', B.pass_grid_table(lambda: frame_a(1).x), [boxes, sortfit], boxt, 700, 1500)
    keys = pp('WSB_Keys1', B.pass_keys1(), [pack_pos, coarse, slabx, runtab, sortboxes, sortfit], lp, 800, 1500)
    for n, (k, j) in enumerate(ref.sort_steps()):
        keys = pp('WSB_Sort1 %d/%d' % (k, j), B.pass_sort_step(k, j), [keys], lp, 900 + 30 * (n % 40),
                  1500 + 120 * (n // 40))
    cells1 = pp('WSB_Cells1', B.pass_cells(ref.AW, ref.AH), [keys], atl, -650, 600)

    # --- Quality: the voxel grids of the boxes (the finest that fits the budget) and their mips ---
    voxfit = pp('WSB_VoxFit', B.pass_fit(B.q_budget), [boxes, bounds], one, -1000, 700)
    voxboxes = pp('WSB_VoxBoxes', B.pass_grid_table(lambda: frame_a(1).x), [boxes, voxfit], boxt, -800, 650)
    voxmap = pp('WSB_VoxMap', B.pass_vox_map(), [voxboxes], atl, -700, 650)
    mipboxes = pp('WSNB_MipBoxes', pass_mip_boxes(), [voxboxes], (log2(R), log2(S)), -700, 750)
    mipmap = pp('WSNB_MipMap', pass_mip_map(), [mipboxes, voxboxes], qt, -600, 750)

    # --- surface normals (mesh + Quality) and the input (colour, blur amount) ---
    pack_normal = pp('WSNB_PackNormal', pass_pack_normal(), [position, uvmask, pack_pos, coarse, size], lp, -950, 250)
    splat_in = [keys, cells1, pack_pos, pack_area, voxfit, voxmap, voxboxes, sortboxes, sortfit, area_total]
    pack_vals = pp('WSNB_PackVals', pass_pack_vals(), [src, position, uvmask, pack_pos, coarse, size, bmap], lp,
                   -950, 150)
    if fast:
        # same result, less work: samples in sorted order, and voxels with no surface in reach skip the walk
        sorted_pos = pp('WSNB_SortedPos', pass_sorted_copy(lambda slot, uv: vec(sample_col(slot, uv).xyz,
                                                                                norm_area(2, 3, uv))),
                        [keys, pack_pos, pack_area, area_total], lp, -650, 450)
        sorted_nrm = pp('WSNB_SortedNormal', pass_sorted_copy(lambda slot, uv: vec(unit3(sample_col(slot, uv).xyz),
                                                                                   1.0)),
                        [keys, pack_normal], lp, -650, 500)
        sorted_vals = pp('WSNB_SortedVals', pass_sorted_copy(), [keys, pack_vals], lp, -650, 150)
        sortmap = pp('WSB_SortMap', B.pass_vox_map(), [sortboxes], atl, -650, 750)
        near = cells1
        for axis, name in enumerate('XYZ'):
            near = pp('WSB_Near%s' % name, B.pass_near(axis, axis == 0), [near, sortmap, sortboxes, voxfit, sortfit],
                      atl, -620 + 40 * axis, 750)
        normals = pp('WSNB_Normals', pass_normals(True), splat_in + [pack_normal, pack_normal, sorted_pos, sorted_nrm,
                                                                     near], atl, -500, 450, fmt=FMT_16F)
        splat = pp('WSNB_Splat', pass_splat_colour(True), splat_in + [pack_vals, normals, sorted_pos, sorted_vals],
                   atl, -500, 300, fmt=FMT_16F)
    else:
        normals = pp('WSNB_Normals', pass_normals(), splat_in + [pack_normal], atl, -500, 450, fmt=FMT_16F)
        splat = pp('WSNB_Splat', pass_splat_colour(), splat_in + [pack_vals, normals], atl, -500, 300, fmt=FMT_16F)
    V = pp('WSNB_Voxels', pass_voxels(), [splat, normals], atl, -400, 300, fmt=FMT_16F)
    radius = pp('WSNB_Radius', pass_radius(), [area_total, voxfit, size], one, -400, 700)

    # --- Non-Uniform Blur passes ---
    for k in range(MAX_PASSES):
        x0 = -300 + 200 * k
        q = V
        for l in range(1, LEVELS):
            q = pp('WSNB_Mip%d' % l, pass_pyramid(l, k), [q, mipmap, mipboxes, voxboxes], qt, x0, 400 + 120 * l,
                   fmt=FMT_16F)
        V = pp('WSNB_Pass%d' % (k + 1), pass_blur(k), [V, q, normals, splat, radius, voxmap, voxboxes, mipboxes],
               atl, x0 + 100, 300, fmt=FMT_16F)
    out = g.pixel_processor('WSNB_Resolve', pass_resolve(fast),
                            [src, position, uvmask, V, voxfit, coarse, slabx, runtab, voxboxes, bmap, radius],
                            None, color=True, fmt=FMT_16, x=-300 + 200 * MAX_PASSES + 200, y=0)
    output_color(g, 'output', 'Output', out, x=-300 + 200 * MAX_PASSES + 400, y=0)
    desc = ('v%s - Non-uniform (bokeh) blur done in 3D using the mesh position, so it is continuous across UV '
            'seams. The Blur Map (or the input itself) scales the blur per texel like the 2D Non-Uniform Blur. '
            "Pieces laid out apart get their own voxel grids, so spreading them out doesn't coarsen the blur, "
            "and they don't blur into each other." % VERSION)
    return graph_xml(g, desc), len(g.nodes)


def instance_param(name):
    typ = PARAM_TYPES.get(name, F1)
    if typ == T_INT:
        ex = Ex('get_integer1', T_INT, data=('constantValueString', name))
    elif typ == BOOL:
        ex = get_bool(name)
    else:
        ex = param(name)
    em = B.FunctionEmitter()
    root = em.emit(ex, {})
    return ('<parameter><name v="%s"/><relativeTo v="0"/><paramValue>%s</paramValue></parameter>'
            % (name, em.xml(root)))


def build_grayscale(pkg_uid):
    """Grayscale wrapper: gray -> colour, instance of the colour graph, colour -> gray."""
    g = B.Graph(GRAPH_ID + '_grayscale', 'World Space Non Uniform Blur Grayscale')
    if os.path.exists(icon_path()):
        g.set_icon(icon_path())
    I = inputs(g, False)
    g.paraminputs.append(params_xml())
    col = g.pixel_processor('WSNB_ToColour', Program(lambda: sample_col(0, pos())), [I['input']], None,
                            color=True, fmt=FMT_16, x=-400, y=0)
    u, o = new_uid(), new_uid()
    conns = [('input', col), ('blur_map', I['blur_map']), ('mesh_position', I['mesh_position']),
             ('mesh_uv_mask', I['mesh_uv_mask'])]
    g.nodes.append(
        '<compNode><uid v="%d"/><connections>%s</connections><GUILayout><gpos v="-150 200 0"/></GUILayout>'
        '<compOutputs><compOutput><uid v="%d"/><comptype v="1"/></compOutput></compOutputs><compImplementation>'
        '<compInstance><path v="pkg:///%s?dependency=%d"/><parameters>%s</parameters><outputBridgings>'
        '<outputBridging><uid v="%d"/><identifier v="output"/></outputBridging></outputBridgings></compInstance>'
        '</compImplementation></compNode>'
        % (u, ''.join('<connection><identifier v="%s"/><connRef v="%d"/><connRefOutput v="%d"/></connection>'
                      % (name, n_, o_) for name, (n_, o_) in conns),
           o, GRAPH_ID, pkg_uid, ''.join(instance_param(p_) for p_ in PARAM_IDS), o))
    gray = g.pixel_processor('WSNB_ToGray', Program(lambda: sample_col(0, pos()).x), [(u, o)], None,
                             color=False, fmt=FMT_16, x=100, y=0)
    g.output('output', 'Output', gray, x=300, y=0)
    return graph_xml(g, 'v%s - Grayscale version of World Space Non Uniform Blur (for masks).' % VERSION)


def build_xml(fast=True):
    """The package XML. fast=False builds the plain graph the speed-ups are tested against
    (same output, more work)."""
    pkg_uid = new_uid()
    colour, n = build_colour(fast)
    gray = build_grayscale(pkg_uid)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><package><identifier v="%s"/><formatVersion v="1.1.0.202302"/>'
           '<updaterVersion v="1.1.0.202302"/><fileUID v="{4f6b2a1e-93c7-4d0a-b8e5-7c2d91a3f604}"/>'
           '<versionUID v="0"/><dependencies><dependency><filename v="?himself"/><uid v="%d"/><type v="package"/>'
           '<fileUID v="0"/><versionUID v="0"/></dependency></dependencies><content>%s%s</content></package>'
           % (GRAPH_ID, pkg_uid, colour, gray))
    return xml, n


def build(path, fast=True):
    xml, n = build_xml(fast)
    with open(path, 'w') as f:
        f.write(xml)
    return n


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), OUTPUT_NAME)
    n = build(out)
    from check_types import check
    probs = check(out)
    if probs:
        raise SystemExit('type check failed:\n' + '\n'.join(probs))
    print('wrote', out, n, 'nodes', os.path.getsize(out) // 1024, 'KB')
