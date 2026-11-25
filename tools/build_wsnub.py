"""Generates world_space_non_uniform_blur_v<VERSION>.sbs.

Bump VERSION below for each release; it goes in the file name and the graph descriptions.

World space version of the Non-Uniform Blur. The mesh is voxelized exactly
like World Space Mask Blur v2.1, whose passes are reused from tools/wsb/
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
  * reads the result back at every texel's 3D position (quadratic B-spline in the
    texel's box).
Pieces in different boxes never blur into each other.

Run:  python3 tools/build_wsnub.py  [--lite] [output.sbs]
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


    # --- statements beyond the vendored Program: loops as statements, and `When` ---

    def stmt(self, s, depth):
        """(uid, type) of one statement: a (Var, build) Set, a Loop or a When."""
        if isinstance(s, When):
            self.n_when = getattr(self, 'n_when', 0) + 1
            f, go = Var('when%d' % self.n_when, F1), s.go
            s = Loop([(f, lambda: const(0.0))], lambda: lor(gt(f.get(), 0.5), lt(go(), 0.5)),
                     list(s.stmts) + [(f, lambda: const(1.0))], f, 4)
        if isinstance(s, Loop):
            init = self.chain([self.stmt(x, depth + 2) for x in s.init], depth + 1)
            cond = self.emit(s.cond(), {}, depth + 1)
            body = self.chain([self.stmt(x, depth + 3) for x in s.body], depth + 2)
            tail = self.emit(s.counter.get(), {}, depth + 2)
            seq = self.raw('sequence', s.counter.typ, [('seqin', body[0]), ('seqlast', tail)], depth=depth + 1)
            return self.raw('while', s.counter.typ, [('init', init[0]), ('cond', cond), ('loop', seq)],
                            ('constantValueInt1', str(s.max_iter)), depth=depth), s.counter.typ
        return self.statement(s[0], s[1], depth)

    def program(self, prog):
        if not isinstance(prog, Block):
            return super().program(prog)
        body = self.chain([self.stmt(s, 3) for s in prog.stmts], 2)
        result = prog.result()
        res_uid = self.emit(result, {}, 2)
        seq = self.raw('sequence', result.typ, [('seqin', body[0]), ('seqlast', res_uid)], depth=1)
        zero = self.emit(const(0.0, result.dim), {}, 1)
        return self.raw('add', result.typ, [('a', seq), ('b', zero)], depth=0)


class Loop:
    """A While loop used as a statement (the vendored Program has exactly one, before its result).
    init / body: statements; cond: callable -> Ex(bool), True = stop; counter: the Var read as
    the loop's tail value."""

    def __init__(self, init, cond, body, counter, max_iter):
        self.init, self.cond, self.body, self.counter, self.max_iter = list(init), cond, list(body), counter, max_iter


class When:
    """Statement: run `stmts` once where go() > 0.5 (go: callable -> float Ex), else not at all.
    It is a While loop that runs 0 or 1 times (its own flag variable stops it after one turn):
    the engine really skips a loop that runs 0 times, where an If/Else evaluates both sides.

    Rules for the statements inside, so that the result would still be right if an engine ran
    the body once before testing (and so that nothing is read before it is set):
      * a variable read after the When is set before it (to what it must be when skipped);
      * what is read after the When doesn't change if the body runs where go() is false:
        either the value set inside is only used where go() holds, or the Set is itself
        guarded by the condition (select(cond, new, old))."""

    def __init__(self, go, stmts):
        self.go, self.stmts = go, list(stmts)


class Block:
    """A pixel function as a list of statements (Sets, Loops, Whens), then a result."""

    def __init__(self, stmts, result):
        self.stmts, self.result = list(stmts), result


def skippable(prog, go):
    """A vendored Program as a Block that only runs where go() > 0.5 (callable -> float Ex);
    elsewhere the result is 0 and nothing is read."""
    res = Var('res', F4)
    if prog.cond is None:
        inner = [(res, prog.result)]
    else:
        inner = [Loop(prog.init, prog.cond, prog.body, prog.counter, prog.max_iter), (res, prog.result)]
    return Block([(res, lambda: const(0.0, 4)), When(go, inner)], lambda: res.get())


def frame_texel():
    """1 at the two texels of a 16 x 16 global table that are ever read (B.frame_a, B.frame_b),
    0 at the other 254. tools/check_types.py checks that nothing reads those."""
    px = floor(pos() * 16.0)
    ax, bx = B.FRAME_A_UV[0] * 16.0 - 0.5, B.FRAME_B_UV[0] * 16.0 - 0.5
    row = B.FRAME_A_UV[1] * 16.0 - 0.5
    return flag(land(lt(fabs(px.y - row), 0.5), lor(lt(fabs(px.x - ax), 0.5), lt(fabs(px.x - bx), 0.5))))


def global_table(prog, skip):
    """A 16 x 16 global table (Bounds, Coarse, the fits, AreaTotal): every texel used to run
    the whole loop; with `skip` only the two texels that are read do."""
    return skippable(prog, frame_texel) if skip else prog


B.FunctionEmitter = DedupEmitter      # used by B.Graph.pixel_processor

VERSION = '2.1'
GRAPH_ID = 'world_space_non_uniform_blur'
OUTPUT_NAME = '%s_v%s.sbs' % (GRAPH_ID, VERSION)
# Lite: the same graph with tables of half the size per axis and no Softness (no mip pyramids)
LITE_VERSION = '1.0'
LITE_GRAPH_ID = 'world_space_non_uniform_blur_lite'
LITE_OUTPUT_NAME = '%s_v%s.sbs' % (LITE_GRAPH_ID, LITE_VERSION)
# ref.configure() arguments of Lite: PMAX, AW, AH, CW, CH, DMAX, S_MAX, R_MAX, Quality -> voxel budget n^3,
# frame grid, reads per patch axis, area grid
LITE_CONFIG = (256, 512, 512, 512, 512, 128, 16, 16, {1: 32, 2: 40, 3: 48, 4: 64}, 64, 4, 32)
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


def blur_amount(in_slot, map_slot, uv, colour=None, map_value=None):
    """per-texel blur scale: 1, the Blur Map input or the luminance of the input itself.
    colour / map_value: the two reads, when the caller already has them (map_value only
    matters where map_used() holds)."""
    colour = sample_col(in_slot, uv) if colour is None else colour
    map_value = sample_lum(map_slot, uv) if map_value is None else map_value
    e = select(get_bool('blur_map_from_input'), dot(colour.xyz, const((0.2126, 0.7152, 0.0722))), map_value)
    e = clamp(select(get_bool('invert_blur_map'), 1.0 - e, e), 0.0, 1.0)
    return select(get_bool('use_blur_map'), e, 1.0)


def map_used():
    """1 where the Blur Map input is read: Use Blur Map on and Use Input As Blur Map off"""
    return select(get_bool('use_blur_map'), select(get_bool('blur_map_from_input'), 0.0, 1.0), 0.0)


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
    Same neighbour rules as the mask blur's texel_area (B.texel_diffs: forward, else backward
    within r2, else the shorter in-island neighbour; slots: 0 position, 1 uv mask)."""
    dx, dy = B.texel_diffs(uv, S, r2)
    return cross3(dx / c, dy / c)


def _patch_loop(result, accs, read, isz_slot, rep_slot, area_slot):
    """loop over the SUB x SUB reads of this sample's patch, like WSB_PackMask: a read counts if
    it is inside an island and within the sample's reject distance (PackArea.z, v2.1).
    accs: [(Var, update(taken, uv, ctx) -> Ex)]"""
    S = ref.SUB
    i, cell, rep, r2, isz = Var('i', F1), Var('cell', F2), Var('rep', F4), Var('r2', F1), Var('isz', F2)
    ctx = dict(i=i, cell=cell, rep=rep, r2=r2, isz=isz)

    def take_and_uv():
        uv = snap(B._sub_uv(cell.get(), i.get()), isz.get())
        return B._taken(uv, rep.get(), r2.get(), *read), uv

    init = [(isz, lambda: input_size(isz_slot)), (cell, B._cell),
            (rep, lambda: sample_col(rep_slot, (B._cell() + 0.5) / float(ref.PMAX))),
            (r2, lambda: sample_col(area_slot, pos()).z)]
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
    signed (UV handedness), in coarse-cell units. Slots: 0 position, 1 uv mask, 2 PackPos, 3 Coarse, 4 Size,
    5 PackArea."""
    nsum = Var('nsum', F3)

    def upd(ok, uv, ctx):
        return select(ok, texel_cross(uv, ctx['isz'].get(), B._r2(3), frame_a(3).w), const(0.0, 3))

    return _patch_loop(lambda: vec(nsum.get(), 1.0), [(nsum, upd)], (0, 1), 4, 2, 5)


def pass_pack_vals():
    """WSNB_PackVals (PMAX^2) -> mean (r, g, b, blur amount) of the reads taken.
    Slots: 0 input, 1 position, 2 uv mask, 3 PackPos, 4 Coarse, 5 Size, 6 blur map, 7 PackArea."""
    csum, cnt = Var('csum', F4), Var('cnt', F1)

    def upd_c(ok, uv, ctx):
        return select(ok, vec(sample_col(0, uv).xyz, blur_amount(0, 6, uv)), const(0.0, 4))

    def upd_n(ok, uv, ctx):
        return select(ok, 1.0, 0.0)

    def result():
        n = cnt.get()
        return select(gt(n, 0.5), csum.get() / fmax(n, 1.0), const(0.0, 4))

    return _patch_loop(result, [(csum, upd_c), (cnt, upd_n)], (1, 2), 5, 3, 7)


def _patch_block(result, accs, read, isz_slot, rep_slot, area_slot, go=None, before=()):
    """_patch_loop that skips what isn't used (same result): a patch with no surface (and any
    patch where go() is 0) skips the loop, and each read is tested once; `accs` only run for
    the reads taken. accs: [(Var, update(uv, ctx) -> Ex)], added where the read is taken;
    before: statements run in every loop step before the accs (they can test ctx['ok']);
    go: callable -> float Ex."""
    S = ref.SUB
    i, cell, rep, r2, isz = Var('i', F1), Var('cell', F2), Var('rep', F4), Var('r2', F1), Var('isz', F2)
    uvv, ok, cnt, run = Var('uvv', F2), Var('ok', F1), Var('cnt', F1), Var('run', F1)
    ctx = dict(i=i, cell=cell, rep=rep, r2=r2, isz=isz, cnt=cnt, ok=ok, uv=uvv)

    def run_init():
        has = gt(rep.get().w, 0.5)
        return flag(has if go is None else land(has, gt(go(), 0.5)))

    stmts = [(isz, lambda: input_size(isz_slot)), (cell, B._cell),
             (rep, lambda: sample_col(rep_slot, (B._cell() + 0.5) / float(ref.PMAX))),
             (r2, lambda: sample_col(area_slot, pos()).z), (run, run_init), (cnt, lambda: const(0.0))]
    stmts += [(var, (lambda v=var: const(0.0, B.DIM[v.typ]))) for var, _ in accs]
    taken = [(var, (lambda u=upd, v=var: v.get() + select(on(ok.get()), u(uvv.get(), ctx),
                                                          const(0.0, B.DIM[v.typ])))) for var, upd in accs]
    stmts.append(Loop([(i, lambda: const(0.0))], lambda: lor(ge(i.get(), float(S * S)), lt(run.get(), 0.5)),
                      [(uvv, lambda: snap(B._sub_uv(cell.get(), i.get()), isz.get())),
                       (ok, lambda: flag(land(B._taken(uvv.get(), rep.get(), r2.get(), *read), on(run.get()))))]
                      + list(before) +
                      [When(lambda: ok.get(), taken),
                       (cnt, lambda: cnt.get() + select(on(ok.get()), 1.0, 0.0)),
                       (i, lambda: i.get() + 1.0)], i, S * S + 8))
    return Block(stmts, lambda: result(ctx))


def pass_pack_normal_skip():
    """WSNB_PackNormal doing only the work that is used (same result as pass_pack_normal)."""
    nsum = Var('nsum', F3)
    return _patch_block(lambda ctx: vec(nsum.get(), 1.0),
                        [(nsum, lambda uv, ctx: texel_cross(uv, ctx['isz'].get(), B._r2(3), frame_a(3).w))],
                        (0, 1), 4, 2, 5)


def pass_pack_vals_skip():
    """WSNB_PackVals doing only the work that is used (same result as pass_pack_vals): the
    Blur Map input is only read when it is used, and nothing is read at Intensity 0 (the
    result is then not used: the final lookup keeps the input)."""
    csum, ml, uvv, ok = Var('csum', F4), Var('ml', F1), Var('uvv', F2), Var('ok', F1)

    def upd_c(uv, ctx):
        c = sample_col(0, uv)
        return vec(c.xyz, blur_amount(0, 6, uv, c, ml.get()))

    def result(ctx):
        n = ctx['cnt'].get()
        return select(gt(n, 0.5), csum.get() / fmax(n, 1.0), const(0.0, 4))

    before = [(ml, lambda: const(0.0)),
              When(lambda: flag(land(on(ok.get()), on(map_used()))), [(ml, lambda: sample_lum(6, uvv.get()))])]
    return _patch_block(result, [(csum, upd_c)], (1, 2), 5, 3, 7, go=lambda: flag(gt(param('intensity'), 0.0)),
                        before=before)


def pass_pack_area_skip():
    """WSB_PackArea doing only the work that is used (same result as B.pass_pack_area, whose
    every loop step does both rounds' reads): the first round only looks for the texel spacing,
    the second tests each read once and only measures the area of the reads taken; patches
    with no surface skip both. Slots: 0 position, 1 uv mask, 2 PackPos, 3 Coarse, 4 Size."""
    S = ref.SUB
    n = float(S * S)
    i, cell, rep, isz, a = Var('i', F1), Var('cell', F2), Var('rep', F4), Var('isz', F2), Var('a', F1)
    cnt, smax, uvv, ok = Var('cnt', F1), Var('smax', F1), Var('uvv', F2), Var('ok', F1)

    def tol2():
        t = fmax(fmax(isz.get().x, isz.get().y) / float(ref.PMAX), 1.0)
        d = smax.get() * t * ref.REJECT_SPACING
        return fmax(B._r2(3), d * d)

    def upd_smax():
        uv = uvv.get()
        dx, dy = B.texel_diffs(uv, isz.get(), B._r2(3))
        sp = fsqrt(dot(dx, dx)) + fsqrt(dot(dy, dy))
        ok1 = land(lt(i.get(), n), land(gt(sample_lum(1, uv), ref.UV_INSIDE), gt(rep.get().w, 0.5)))
        return select(ok1, fmax(smax.get(), sp), smax.get())

    body = [(uvv, lambda: snap(B._sub_uv(cell.get(), select(lt(i.get(), n), i.get(), i.get() - n)), isz.get())),
            (ok, lambda: const(0.0)),
            When(lambda: flag(lt(i.get(), n)), [(smax, upd_smax)]),
            When(lambda: flag(ge(i.get(), n)),
                 [(ok, lambda: flag(land(ge(i.get(), n), B._taken(uvv.get(), rep.get(), tol2(), 0, 1))))]),
            When(lambda: ok.get(),
                 [(a, lambda: a.get() + select(on(ok.get()), B.texel_area(uvv.get(), isz.get(), B._r2(3)), 0.0))]),
            (cnt, lambda: cnt.get() + select(on(ok.get()), 1.0, 0.0)),
            (i, lambda: i.get() + 1.0)]
    return Block([(isz, lambda: input_size(4)), (cell, B._cell),
                  (rep, lambda: sample_col(2, (B._cell() + 0.5) / float(ref.PMAX))),
                  (smax, lambda: const(0.0)), (a, lambda: const(0.0)), (cnt, lambda: const(0.0)),
                  Loop([(i, lambda: const(0.0))], lambda: lor(ge(i.get(), 2.0 * n), le(rep.get().w, 0.5)), body, i,
                       2 * S * S + 8)],
                 lambda: vec(a.get() / n, cnt.get() / n, tol2(), 1.0))


def _search_step(s, target):
    """one step of B.pass_cells' binary search for the first sorted key >= target (slot 0)"""
    lo, hi = s.get().x, s.get().y
    mid = floor((lo + hi) * 0.5)
    key = sample_col(0, index_uv(mid)).x
    nxt = select(lt(key, target), vec(mid + 1.0, hi), vec(lo, mid))
    return select(lt(lo, hi), nxt, s.get())


def pass_cell_starts(W, H):
    """WSB_Starts0 / WSB_Starts1 (linear W x H, mesh only) -> (start, 0, 0, 1): where each cell's
    samples start in the sorted list: B.pass_cells' first binary search. Slot 0: sorted keys."""
    L = float(ref.PMAX ** 2)
    s1, cid, i = Var('s1', F2), Var('cid', F1), Var('i', F1)
    return Program(lambda: vec(s1.get().x, 0.0, 0.0, 1.0),
                   init=[(cid, lambda: my_index(W, H)), (s1, lambda: vec(0.0, L)), (i, lambda: const(0.0))],
                   cond=lambda: ge(s1.get().x, s1.get().y),
                   body=[(s1, lambda: _search_step(s1, cid.get())), (i, lambda: i.get() + 1.0)],
                   counter=i, max_iter=int(L).bit_length() + 9)


def pass_cells_joined(W, H):
    """WSB_Cells0 / WSB_Cells1 with half the searches (same result as B.pass_cells, which does
    two binary searches per cell): a cell's end is the first key >= its id + 1, which is the
    next cell's start. Only the last cell searches for its end. Slots: 0 sorted keys, 1 Starts."""
    L = float(ref.PMAX ** 2)
    last = float(W * H - 1)
    s2, cid, i, e = Var('s2', F2), Var('cid', F1), Var('i', F1), Var('e', F1)

    def is_last():
        return ge(cid.get(), last - 0.5)

    return Block([(cid, lambda: my_index(W, H)),
                  (e, lambda: sample_col(1, lin_uv(fmin(cid.get() + 1.0, last), W, H)).x),
                  When(lambda: flag(is_last()),
                       [Loop([(s2, lambda: vec(0.0, L)), (i, lambda: const(0.0))],
                             lambda: ge(s2.get().x, s2.get().y),
                             [(s2, lambda: _search_step(s2, cid.get() + 1.0)), (i, lambda: i.get() + 1.0)],
                             i, int(L).bit_length() + 9),
                        (e, lambda: select(is_last(), s2.get().x, e.get()))])],
                 lambda: vec(sample_col(1, pos()).x, e.get(), 0.0, 1.0))


# ----------------------------------------------------------------------------
# splats (AW x AH voxel tables): each voxel gathers the samples of its box next to
# it through the sorted list, like WSB_Splat.
#   slots: 0 sorted keys (Sort1), 1 Cells1, 2 PackPos, 3 PackArea, 4 VoxFit (h), 5 VoxMap,
#          6 VoxBoxes, 7 SortBoxes, 8 SortFit, 9 AreaTotal, 10 PackNormal / PackVals, 11 Normals,
#   and with `fast`: 12 SortedPos, 13 SortedNormal / SortedVals, 14 NearZ (Normals only)

def unit3(n):
    return n / fmax(fsqrt(dot(n, n)), 1e-30)


def splat_program(accs, result, active=None, extra=lambda v: v, fast=False, near=False, skip=False):
    """accs: [(Var, update(value, weight) -> Ex)]: value is the sample's slot-10 value after
    `extra`, weight the tent weight times the sample's area relative to the mean sample area.

    `fast` gives the same result with less work: samples are read from copies in sorted
    order (SortedPos: position and relative area, slot 13: the value after `extra`), 2 reads
    next to each other instead of 5 scattered ones; with `near`, a voxel whose sort cell has
    no occupied cell within NearZ's reach (more than the walk reaches) skips the walk, whose
    sum would be 0.

    `skip` (with `fast`) also gives the same result: a loop step that reads a sample doesn't
    read the cell table and a step that moves to the next cell reads no sample; the sample is
    read once for all the accumulators; with `active`, voxels where it is false skip the setup."""
    skip = skip and fast
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

    def next_cell():
        d = dims.get()
        tt = t.get()
        row = floor(tt / d.x)
        c = lo.get() + vec(tt - d.x * row, row - d.y * floor(row / d.y), floor(row / d.y))
        return sample_col(1, atlas_uv(so.get().w + lin3(c, sd.get()))).xy

    setup = [(so, lambda: sample_col(7, box_uv(fmax(vm().x, 0.0), 0.0))),
             (sd, lambda: sample_col(7, box_uv(fmax(vm().x, 0.0), 1.0)).xyz),
             (C, c_init), (lo, lambda: cell_range(-1.0)),
             (dims, lambda: cell_range(1.0) - lo.get() + 1.0), (nc, nc_init)]
    if skip:
        sp, sv, wv, nx = Var('sp', F4), Var('sv', F4), Var('wv', F1), Var('nx', F2)

        def weight():
            q = fmax(1.0 - fabs(sp.get().xyz - C.get()) / frame_a(4).x, 0.0)
            return select(gt(hv.get(), 0.5), q.x * q.y * q.z * sp.get().w, 0.0)

        stmts = [(nc, lambda: const(0.0)), (se, lambda: const(0.0, 2)), (hv, lambda: const(0.0)),
                 (nx, lambda: const(0.0, 2))]
        stmts += [(var, (lambda v=var: const(0.0, B.DIM[v.typ]))) for var, _ in accs]
        stmts += setup if active is None else [When(lambda: flag(active()), setup)]
        sample = [(sp, lambda: sample_col(12, index_uv(se.get().x))), (sv, lambda: sample_col(13, index_uv(se.get().x))),
                  (wv, weight)]
        sample += [(var, (lambda u=upd, v=var: v.get() + u(sv.get(), wv.get()))) for var, upd in accs]
        # hv first: the accumulators, se and t all decide on the state before this step
        body = [(hv, lambda: select(lt(se.get().x, se.get().y), 1.0, 0.0)),
                When(lambda: hv.get(), sample),
                When(lambda: 1.0 - hv.get(), [(nx, next_cell)]),
                (se, lambda: select(gt(hv.get(), 0.5), se.get() + vec(1.0, 0.0), nx.get())),
                (t, lambda: select(gt(hv.get(), 0.5), t.get(), t.get() + 1.0))]
        stmts.append(Loop([(t, lambda: const(0.0))], lambda: land(ge(se.get().x, se.get().y), ge(t.get(), nc.get())),
                          body, t, 65536))
        return Block(stmts, result)

    init = setup + [(t, lambda: const(0.0)), (se, lambda: const(0.0, 2)), (hv, lambda: const(0.0))]
    # hv first: the accumulators, se and t all decide on the state before this step
    body = [(hv, lambda: select(lt(se.get().x, se.get().y), 1.0, 0.0))]
    for var, upd in accs:
        init.append((var, (lambda v=var: const(0.0, B.DIM[v.typ]))))
        body.append((var, (lambda u=upd, v=var: v.get() + u(*value_and_weight()))))
    body += [(se, upd_se), (t, lambda: select(gt(hv.get(), 0.5), t.get(), t.get() + 1.0))]
    return Program(result, init=init, cond=lambda: land(ge(se.get().x, se.get().y), ge(t.get(), nc.get())),
                   body=body, counter=t, max_iter=65536)


def pass_normals(fast=False, skip=False, near=True):
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
    return splat_program(accs, result, extra=lambda v: unit3(v.xyz), fast=fast, near=near, skip=skip)


def pass_splat_colour(fast=False, skip=False):
    """WSNB_Splat (AW x AH) -> (sum colour * w, sum blur amount * w); sum w is Normals.w.
    Voxels with no surface nearby (Normals.w = 0) skip the loop; with `skip` they also skip its
    setup, and so does every voxel at Intensity 0 (the result is then not used)."""
    acc = Var('acc', F4)

    def active():
        a = gt(sample_col(11, pos()).w, 0.0)
        return land(a, gt(param('intensity'), 0.0)) if skip and fast else a

    return splat_program([(acc, lambda v, w: v * w)], lambda: acc.get(), active=active, fast=fast, skip=skip)


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


def pass_radius_of(k, e):
    """radius of pass k in voxels, for a blur amount e (slot 4: Radius)"""
    return frame_a(4).x * e * fexp(-(2.0 * k * C_DECAY) / fmax(pass_count(), 1.0))


def tap_level(r):
    """the mip level (0..4, fractional) of a fat tap of radius r voxels"""
    return approx_log2_clamped(fmax(param('softness'), 0.0) * 0.4 * r / 0.64)


def pass_pyramid(l, k, skip=False):
    """WSNB_Mip<l> (packed mip table): level l of every box = [1 3 3 1]/8 filter, stride 2, of
    level l - 1 of the same box (taps outside the box are empty); lower levels are copied. Only
    computed when pass k uses fat taps. Slots: 0 voxels (l = 1) or the previous mip table,
    1 MipMap, 2 MipBoxes, 3 VoxBoxes, and with `skip` 4 Radius.

    `skip`: only what pass k reads is computed, which leaves the pass's result unchanged. A
    tap at level lev reads level floor(lev) and, if lev has a fraction, the next one; lev grows
    with the radius, so no voxel's tap reads past the level of the pass's largest radius (blur
    amount 1). Level l is computed when that largest level is above l - 1; otherwise its entries
    stay 0. An inactive pass slot, Softness 0, or a radius too small for level 1 (all its taps
    at level 0) reads nothing at all."""
    if skip:
        return pass_pyramid_skip(l, k)
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


def pass_pyramid_skip(l, k):
    bases, QW, QH = mip_layout()
    i, acc, act, ijk = Var('i', F1), Var('acc', F4), Var('act', F1), Var('ijk', F3)
    dp, po, nn, low, mm = Var('dp', F3), Var('po', F1), Var('nn', F2), Var('low', F4), Var('mm', F4)

    def t():
        return my_index(QW, QH)

    def bx():
        return fmax(mm.get().x, 0.0)

    def uses_mips():
        return land(pass_on(k), gt(param('softness'), 0.0))

    def nn_init():
        # (this level is read, level 1 is read) by pass k
        top = tap_level(pass_radius_of(k, 1.0))
        return vec(flag(land(uses_mips(), gt(top, l - 1.0))), flag(land(uses_mips(), gt(top, 0.0))))

    def inreg():
        return land(ge(t(), float(bases[l])), lt(t(), float(bases[l + 1])))

    def lower():
        return land(on(nn.get().y), lt(t(), float(bases[l])))

    def parent_dims():
        n0 = box_dims(3, bx()).xyz
        return n0 if l == 1 else fceil(n0 * (1.0 / 2 ** (l - 1)))

    def parent_off():
        return sample_col(3, box_uv(bx(), 0.0)).w if l == 1 else chan(box_table2(2, bx()), float(l - 1))

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

    res = Var('res', F4)
    inner = [(nn, nn_init), (act, lambda: const(0.0)), (acc, lambda: const(0.0, 4)), (low, lambda: const(0.0, 4))]
    if l > 1:
        # the lower levels, from the previous table (only when the pass reads mips at all)
        inner.append(When(lambda: nn.get().y, [(low, lambda: select(lower(), sample_col(0, pos()), const(0.0, 4)))]))
    inner += [When(lambda: nn.get().x,
                   [(mm, lambda: sample_col(1, pos())),
                    (act, lambda: flag(land(land(on(nn.get().x), inreg()), gt(mm.get().x, -0.5))))]),
              When(lambda: act.get(), [(ijk, lambda: mm.get()[1, 2, 3]), (dp, parent_dims), (po, parent_off)]),
              Loop([(i, lambda: const(0.0))], lambda: lor(ge(i.get(), 64.0), lt(act.get(), 0.5)),
                   [(acc, upd), (i, lambda: i.get() + 1.0)], i, 72),
              (res, lambda: select(on(act.get()), acc.get(), low.get()))]
    # an idle slot (or Softness 0) tests one condition and writes 0
    return Block([(res, lambda: const(0.0, 4)), When(lambda: flag(uses_mips()), inner)], lambda: res.get())


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


def tap_voxels(g, n0, o0):
    """trilinear read of the voxel grid (slot 0) at voxel coords g of a box (n0 voxels, at o0)"""
    return corners(g, n0, lambda c: sample_col(0, atlas_uv(o0 + lin3(c, n0))))


def tap_mip(g, l, n0, mo):
    """trilinear read of mip level l (1..4, slot 1) at level-0 voxel coords g"""
    _, QW, QH = mip_layout()
    s = pow2(l)
    d = fceil(n0 / s)
    ol = chan(mo, l)
    return corners((g + 0.5) / s - 0.5, d, lambda c: sample_col(1, lin_uv(ol + lin3(c, d), QW, QH)))


def fat_tap(g, lev, n0, o0, mo):
    """(sum c*w, sum w) at level-0 voxel coords g of the voxel's box, blurred to mip level lev
    (0..4): two trilinear reads. Coarser levels are rescaled by 2^l so a tap on a flat surface
    weighs the same at any level."""
    l0 = fmin(floor(lev), 3.0)
    f = lev - l0
    a = select(lt(l0, 0.5), tap_voxels(g, n0, o0), tap_mip(g, fmax(l0, 1.0), n0, mo) * pow2(l0))
    b = tap_mip(g, l0 + 1.0, n0, mo) * pow2(l0 + 1.0)
    return a * (1.0 - f) + b * f


def approx_log2_clamped(x):
    """piecewise-linear log2(x), clamped to [0, LEVELS - 1]"""
    x = clamp(x, 1.0, 2.0 ** (LEVELS - 1))
    o = flag(ge(x, 2.0)) + flag(ge(x, 4.0)) + flag(ge(x, 8.0)) + flag(ge(x, 16.0))
    base = select(ge(x, 16.0), 16.0, select(ge(x, 8.0), 8.0, select(ge(x, 4.0), 4.0, select(ge(x, 2.0), 2.0, 1.0))))
    return o + (x / base - 1.0)


def pass_blur(k, skip=False, mips=True):
    """WSNB_Pass<k + 1> (AW x AH): one pass of the Non-Uniform Blur on the voxels.

    `skip` gives the same result with less work:
      * an inactive pass slot (k >= the number of passes, or Intensity 0) reads its input and
        nothing else; so does a voxel with no surface;
      * a voxel that isn't blurred (blur amount 0, or not in a box) skips the tangent frame;
      * a tap only reads the mip levels it uses: level floor(lev), and the next one if lev has
        a fraction (the plain graph reads the voxels, level max(l0, 1) and level l0 + 1 for
        every tap, and multiplies what it doesn't use by 0).
    `mips=False` (Lite, needs skip): no Softness, every tap is one trilinear read of the voxels."""
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
        return pass_radius_of(k, e)

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

    def tap_pos():
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
        return vm.get()[1, 2, 3] + off

    def tap():
        return acc.get() + fat_tap(tap_pos(), lev.get(), n0.get(), o0.get(), mo.get())

    def result():
        return select(on(act.get()), acc.get() / (clamp(iparam('blades'), 1.0, 9.0) + 1.0), vv.get())

    def stop():
        # j = 0 is the centre tap, j = 1..Blades the polygon corners
        return lor(gt(j.get(), clamp(iparam('blades'), 1.0, 9.0)), lt(act.get(), 0.5))

    if skip:
        l0, ff, gv = Var('l0', F1), Var('ff', F1), Var('gv', F3)
        a0, aq, bq = Var('a0', F4), Var('aq', F4), Var('bq', F4)
        frame = [(n0, lambda: box_dims(6, bx()).xyz), (o0, lambda: sample_col(6, box_uv(bx(), 0.0)).w),
                 (t1, t1_init), (t2, lambda: cross3(nrm(), t1.get()))]
        if mips:
            frame += [(mo, lambda: box_table2(7, bx())), (lev, lambda: tap_level(r.get())),
                      (l0, lambda: fmin(floor(lev.get()), 3.0)), (ff, lambda: lev.get() - l0.get())]
            # each tap reads the one or two levels it uses; a0 / aq / bq keep 0 where theirs isn't read
            taps = [(gv, tap_pos),
                    When(lambda: flag(lt(l0.get(), 0.5)), [(a0, lambda: tap_voxels(gv.get(), n0.get(), o0.get()))]),
                    When(lambda: flag(ge(l0.get(), 0.5)),
                         [(aq, lambda: tap_mip(gv.get(), fmax(l0.get(), 1.0), n0.get(), mo.get()) * pow2(l0.get()))]),
                    When(lambda: flag(gt(ff.get(), 0.0)),
                         [(bq, lambda: tap_mip(gv.get(), l0.get() + 1.0, n0.get(), mo.get())
                           * pow2(l0.get() + 1.0))]),
                    (acc, lambda: acc.get() + (select(lt(l0.get(), 0.5), a0.get(), aq.get()) * (1.0 - ff.get())
                                               + bq.get() * ff.get()))]
            frame += [(a0, lambda: const(0.0, 4)), (aq, lambda: const(0.0, 4)), (bq, lambda: const(0.0, 4))]
        else:
            taps = [(acc, lambda: acc.get() + tap_voxels(tap_pos(), n0.get(), o0.get()))]
        return Block(
            [(vv, lambda: sample_col(0, pos())), (act, lambda: const(0.0)), (acc, lambda: const(0.0, 4))] +
            [When(lambda: flag(land(pass_on(k), gt(vv.get().w, 0.0))),
                  [(vm, lambda: sample_col(5, pos())), (r, r_init), (act, act_init)]),
             When(lambda: act.get(), frame),
             Loop([(j, lambda: const(0.0))], stop, taps + [(j, lambda: j.get() + 1.0)], j, 12)],
            result)

    return Program(
        result,
        init=[(vm, lambda: sample_col(5, pos())), (vv, lambda: sample_col(0, pos())),
              (n0, lambda: box_dims(6, bx()).xyz), (o0, lambda: sample_col(6, box_uv(bx(), 0.0)).w),
              (mo, lambda: box_table2(7, bx())), (r, r_init), (act, act_init), (t1, t1_init),
              (t2, lambda: cross3(nrm(), t1.get())),
              (lev, lambda: tap_level(r.get())),
              (acc, lambda: const(0.0, 4)), (j, lambda: const(0.0))],
        cond=stop,
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
    """Triquadratic B-spline lookup of the texel's box grid around the nearest voxel (27 reads;
    v2.0 read 64 with a tricubic, as World Space Mask Blur v2.1), blended with the input by the
    local blur amount, dithered; written as 16 bit. Texels with nothing to blur (outside the UV
    islands, in no box, or with a black blur map) skip the reads and keep the input.

    `fast` gives the same result with less work: 3 loop steps of 9 reads instead of 27 of 1,
    added in the same order (z, then y, then x) with the same weights, so the sum is
    bit-identical."""
    p, bx, fnd, ta, n = Var('p', F3), Var('bx', F1), Var('fnd', F1), Var('ta', F4), Var('n', F3)
    i0, tf, al, S, i = Var('i0', F3), Var('tf', F3), Var('al', F1), Var('S', F4), Var('i', F1)

    def g():
        return clamp((p.get() - ta.get().xyz) / frame_a(4).x - 0.5, 1.0, n.get() - 2.0)

    def quad(t):
        a, b = 0.5 - t, 0.5 + t
        return [a * a * 0.5, 0.75 - t * t, b * b * 0.5]

    def al_init():
        rv = frame_a(10).x
        ok = land(gt(sample_lum(2, pos()), 0.5), on(fnd.get()))
        return select(ok, smoothstep01(rv * blur_amount(0, 9, pos()) * 0.5), 0.0)

    init = [(p, lambda: sample_col(1, pos()).xyz),
            (bx, lambda: box_of(p.get(), 5, 6, 7)[0]), (fnd, lambda: flag(box_of(p.get(), 5, 6, 7)[1])),
            (ta, lambda: sample_col(8, box_uv(bx.get(), 0.0))), (n, lambda: box_dims(8, bx.get()).xyz),
            (i0, lambda: floor(g() + 0.5)), (tf, lambda: g() - floor(g() + 0.5)),
            (al, al_init), (S, lambda: const(0.0, 4)), (i, lambda: const(0.0))]

    def pick(w3, x):
        return select(lt(x, 0.5), w3[0], select(lt(x, 1.5), w3[1], w3[2]))

    def read(a, b, c, wx, wy, wz):
        cell = i0.get() + vec(a - 1.0, b - 1.0, c - 1.0)
        return sample_col(3, atlas_uv(ta.get().w + lin3(cell, n.get()))) * (wx * wy * wz)

    def upd():
        ii = i.get()
        c = floor(ii / 9.0)
        b = floor(ii / 3.0) - c * 3.0
        a = ii - floor(ii / 3.0) * 3.0
        t = tf.get()
        return S.get() + read(a, b, c, pick(quad(t.x), a), pick(quad(t.y), b), pick(quad(t.z), c))

    def upd9():
        c = i.get()
        t = tf.get()
        wx, wy, wz = quad(t.x), quad(t.y), pick(quad(t.z), c)
        acc = S.get()
        for b in range(3):
            for a in range(3):
                acc = acc + read(float(a), float(b), c, wx[a], wy[b], wz)
        return acc

    def result():
        s = S.get()
        src = sample_col(0, pos())
        res = s.xyz / fmax(s.w, 1e-12)
        out = src.xyz + (res - src.xyz) * al.get()
        ok = land(gt(al.get(), 0.0), gt(s.w, 1e-6))          # s.w is relative to the mean sample area
        return select(ok, vec(dither3(out), src.w), src)

    steps = 3.0 if fast else 27.0
    return Program(result, init=init, cond=lambda: lor(ge(i.get(), steps), le(al.get(), 0.0)),
                   body=[(S, upd9 if fast else upd), (i, lambda: i.get() + 1.0)], counter=i, max_iter=int(steps) + 8)


def pass_resolve_skip():
    """pass_resolve(fast=True) doing only the work that is used (same result):
      * a texel that keeps the input (outside the UV islands, a black blur map, Intensity 0)
        reads the input, the UV mask and the radius, and nothing else: its box isn't looked up;
      * the box is looked up once (the plain graph does it twice);
      * the Blur Map input is only read when it is used;
      * the dither noise is only made where the result is blurred and Dither isn't 0."""
    p, bf, ta, n, src = Var('p', F3), Var('bf', F2), Var('ta', F4), Var('n', F3), Var('src', F4)
    i0, tf, al, S, i = Var('i0', F3), Var('tf', F3), Var('al', F1), Var('S', F4), Var('i', F1)
    al0, ml, gq, res, o3, dz = Var('al0', F1), Var('ml', F1), Var('gq', F3), Var('res', F4), Var('o3', F3), Var('dz', F3)

    def quad(t):
        a, b = 0.5 - t, 0.5 + t
        return [a * a * 0.5, 0.75 - t * t, b * b * 0.5]

    def al0_init():
        rv = frame_a(10).x
        return select(gt(sample_lum(2, pos()), 0.5),
                      smoothstep01(rv * blur_amount(0, 9, pos(), src.get(), ml.get()) * 0.5), 0.0)

    def bf_init():
        b, found = box_of(p.get(), 5, 6, 7)
        return vec(b, flag(found))

    def pick(w3, x):
        return select(lt(x, 0.5), w3[0], select(lt(x, 1.5), w3[1], w3[2]))

    def read(a, b, c, wx, wy, wz):
        cell = i0.get() + vec(a - 1.0, b - 1.0, c - 1.0)
        return sample_col(3, atlas_uv(ta.get().w + lin3(cell, n.get()))) * (wx * wy * wz)

    def upd9():
        c = i.get()
        t = tf.get()
        wx, wy, wz = quad(t.x), quad(t.y), pick(quad(t.z), c)
        acc = S.get()
        for b in range(3):
            for a in range(3):
                acc = acc + read(float(a), float(b), c, wx[a], wy[b], wz)
        return acc

    def blurred():
        return land(gt(al.get(), 0.0), gt(S.get().w, 1e-6))      # S.w is relative to the mean sample area

    def o3_init():
        s = S.get()
        return src.get().xyz + (s.xyz / fmax(s.w, 1e-12) - src.get().xyz) * al.get()

    def noise():
        # dither3's offset: triangular noise of +-Dither 8-bit steps, faded out near 0 and 1
        c = o3.get()
        px = floor(pos() * get('$size', F2))
        nz = hash12(px) + hash12(px + const((37.0, 91.0))) - 1.0
        k = clamp(fmin(c, 1.0 - c) * 255.0, 0.0, 1.0)
        return k * (nz * param('dither') * (1.0 / 255.0))

    def gq_init():
        return clamp((p.get() - ta.get().xyz) / frame_a(4).x - 0.5, 1.0, n.get() - 2.0)

    lookup_setup = [(p, lambda: sample_col(1, pos()).xyz), (bf, bf_init),
                    (ta, lambda: sample_col(8, box_uv(bf.get().x, 0.0))),
                    (n, lambda: box_dims(8, bf.get().x).xyz), (gq, gq_init)]
    lookup_setup += [(i0, lambda: floor(gq.get() + 0.5)), (tf, lambda: gq.get() - floor(gq.get() + 0.5)),
                     (al, lambda: select(on(bf.get().y), al0.get(), 0.0))]
    reads = Loop([(i, lambda: const(0.0))], lambda: lor(ge(i.get(), 3.0), le(al.get(), 0.0)),
                 [(S, upd9), (i, lambda: i.get() + 1.0)], i, 11)
    stmts = [(src, lambda: sample_col(0, pos())), (res, lambda: src.get()), (al, lambda: const(0.0)),
             (S, lambda: const(0.0, 4)), (ml, lambda: const(0.0)),
             When(map_used, [(ml, lambda: sample_lum(9, pos()))]),
             (al0, al0_init),
             When(lambda: flag(gt(al0.get(), 0.0)), lookup_setup + [reads])]
    stmts.append(When(lambda: flag(blurred()),
                      [(o3, o3_init), (dz, lambda: const(0.0, 3)),
                       When(lambda: flag(gt(fabs(param('dither')), 0.0)), [(dz, noise)]),
                       (res, lambda: select(blurred(), vec(clamp(o3.get() + dz.get(), 0.0, 1.0), src.get().w),
                                            src.get()))]))
    return Block(stmts, lambda: res.get())


def pass_resolve_cubic(fast=False):
    """v2.0's final lookup (build(lookup='cubic'), for comparison): tricubic B-spline lookup of the
    texel's box grid (64 reads), blended with the input by the
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


def params_xml(lite=False):
    BLUR = 'Blur'
    WS = 'World Space'
    axis_desc = 'Object-space direction that Angle 0 points to (projected on the surface).'
    levels = ', '.join('%d = %d^3' % (k, n) for k, n in sorted(ref.QUALITY.items()))
    items = [
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
    ]
    if lite:
        items = [x for x in items if '<identifier v="softness"/>' not in x]
    return ''.join(items)


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


def build_colour(fast=True, lookup='quadratic', skip=None, lite=False):
    skip = fast if skip is None else skip
    assert not lite or (fast and skip), 'Lite is built from the fast, skipping passes'
    g = B.Graph(LITE_GRAPH_ID if lite else GRAPH_ID, 'World Space Non Uniform Blur' + (' Lite' if lite else ''))
    if os.path.exists(icon_path()):
        g.set_icon(icon_path())
    I = inputs(g, True)
    g.paraminputs.append(params_xml(lite))
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

    # --- mesh only: World Space Mask Blur v2.1, as is (PackArea: the same result with less work) ---
    size = g.pixel_processor('WSB_Size', B.pass_size(), [position], None, x=-1400, y=600,
                             rel_log2=(B.SIZE_LOG2, B.SIZE_LOG2))
    bounds = pp('WSB_Bounds', global_table(B.pass_bounds(), skip), [position, uvmask, size], one, -1250, 500)
    coarse = pp('WSB_Coarse', global_table(B.pass_coarse(), skip), [bounds], one, -1150, 500)
    pack_pos = pp('WSB_PackPos', B.pass_pack_pos(), [position, uvmask, size], lp, -1250, 300)
    pack_area = pp('WSB_PackArea', pass_pack_area_skip() if skip else B.pass_pack_area(),
                   [position, uvmask, pack_pos, coarse, size], lp, -1100, 400)
    area_blocks = pp('WSB_AreaBlocks', B.pass_area_blocks(), [pack_area], (log2(ref.AREA_GRID),) * 2, -1000, 450)
    area_total = pp('WSB_AreaTotal', global_table(B.pass_area_total(), skip), [area_blocks], one, -900, 450)
    keys = pp('WSB_Keys0', B.pass_keys0(), [pack_pos, coarse], lp, -1100, 850)
    for n, (k, j) in enumerate(ref.sort_steps()):
        keys = pp('WSB_Sort0 %d/%d' % (k, j), B.pass_sort_step(k, j), [keys], lp, -950 + 30 * (n % 40),
                  850 + 120 * (n // 40))
    def cells(name, W, H, x, y):
        if not skip:
            return pp('WSB_Cells' + name, B.pass_cells(W, H), [keys], (log2(W), log2(H)), x, y)
        starts = pp('WSB_Starts' + name, pass_cell_starts(W, H), [keys], (log2(W), log2(H)), x - 50, y + 50)
        return pp('WSB_Cells' + name, pass_cells_joined(W, H), [keys, starts], (log2(W), log2(H)), x, y)

    cells0 = cells('0', ref.CW, ref.CH, -650, 1600)
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
    sortfit = pp('WSB_SortFit', global_table(B.pass_fit(lambda: const(float(ref.AW * ref.AH))), skip),
                 [boxes, bounds], one, 600, 1500)
    sortboxes = pp('WSB_SortBoxes', B.pass_grid_table(lambda: frame_a(1).x), [boxes, sortfit], boxt, 700, 1500)
    keys = pp('WSB_Keys1', B.pass_keys1(), [pack_pos, coarse, slabx, runtab, sortboxes, sortfit], lp, 800, 1500)
    for n, (k, j) in enumerate(ref.sort_steps()):
        keys = pp('WSB_Sort1 %d/%d' % (k, j), B.pass_sort_step(k, j), [keys], lp, 900 + 30 * (n % 40),
                  1500 + 120 * (n // 40))
    cells1 = cells('1', ref.AW, ref.AH, -650, 600)

    # --- Quality: the voxel grids of the boxes (the finest that fits the budget) and their mips ---
    voxfit = pp('WSB_VoxFit', global_table(B.pass_fit(B.q_budget), skip), [boxes, bounds], one, -1000, 700)
    voxboxes = pp('WSB_VoxBoxes', B.pass_grid_table(lambda: frame_a(1).x), [boxes, voxfit], boxt, -800, 650)
    voxmap = pp('WSB_VoxMap', B.pass_vox_map(), [voxboxes], atl, -700, 650)
    if not lite:
        mipboxes = pp('WSNB_MipBoxes', pass_mip_boxes(), [voxboxes], (log2(R), log2(S)), -700, 750)
        mipmap = pp('WSNB_MipMap', pass_mip_map(), [mipboxes, voxboxes], qt, -600, 750)

    # --- surface normals (mesh + Quality) and the input (colour, blur amount) ---
    pack_normal = pp('WSNB_PackNormal', pass_pack_normal_skip() if skip else pass_pack_normal(),
                     [position, uvmask, pack_pos, coarse, size, pack_area], lp, -950, 250)
    splat_in = [keys, cells1, pack_pos, pack_area, voxfit, voxmap, voxboxes, sortboxes, sortfit, area_total]
    pack_vals = pp('WSNB_PackVals', pass_pack_vals_skip() if skip else pass_pack_vals(),
                   [src, position, uvmask, pack_pos, coarse, size, bmap, pack_area], lp, -950, 150)
    if fast:
        # same result, less work: samples in sorted order, and voxels with no surface in reach skip the walk
        sorted_pos = pp('WSNB_SortedPos', pass_sorted_copy(lambda slot, uv: vec(sample_col(slot, uv).xyz,
                                                                                norm_area(2, 3, uv))),
                        [keys, pack_pos, pack_area, area_total], lp, -650, 450)
        sorted_nrm = pp('WSNB_SortedNormal', pass_sorted_copy(lambda slot, uv: vec(unit3(sample_col(slot, uv).xyz),
                                                                                   1.0)),
                        [keys, pack_normal], lp, -650, 500)
        sorted_vals = pp('WSNB_SortedVals', pass_sorted_copy(), [keys, pack_vals], lp, -650, 150)
        # NearX/Y/Z let voxels with no surface in reach skip the walk. With `skip` a walk step over an
        # empty cell costs one read, so the walk is cheaper than the four tables that avoid it
        # (sphere, 2048^2: 0.84 -> 0.75 G reads in total): they are left out, the result is the same.
        near = cells1
        if not skip:
            sortmap = pp('WSB_SortMap', B.pass_vox_map(), [sortboxes], atl, -650, 750)
            for axis, name in enumerate('XYZ'):
                near = pp('WSB_Near%s' % name, B.pass_near(axis, axis == 0),
                          [near, sortmap, sortboxes, voxfit, sortfit], atl, -620 + 40 * axis, 750)
        normals = pp('WSNB_Normals', pass_normals(True, skip, near=not skip),
                     splat_in + [pack_normal, pack_normal, sorted_pos, sorted_nrm, near], atl, -500, 450, fmt=FMT_16F)
        splat = pp('WSNB_Splat', pass_splat_colour(True, skip),
                   splat_in + [pack_vals, normals, sorted_pos, sorted_vals], atl, -500, 300, fmt=FMT_16F)
    else:
        normals = pp('WSNB_Normals', pass_normals(), splat_in + [pack_normal], atl, -500, 450, fmt=FMT_16F)
        splat = pp('WSNB_Splat', pass_splat_colour(), splat_in + [pack_vals, normals], atl, -500, 300, fmt=FMT_16F)
    V = pp('WSNB_Voxels', pass_voxels(), [splat, normals], atl, -400, 300, fmt=FMT_16F)
    radius = pp('WSNB_Radius', global_table(pass_radius(), skip), [area_total, voxfit, size], one, -400, 700)

    # --- Non-Uniform Blur passes ---
    for k in range(MAX_PASSES):
        x0 = -300 + 200 * k
        if lite:
            # no Softness: the taps read the voxels themselves (slot 1, the mip table, isn't read)
            V = pp('WSNB_Pass%d' % (k + 1), pass_blur(k, True, mips=False),
                   [V, V, normals, splat, radius, voxmap, voxboxes], atl, x0 + 100, 300, fmt=FMT_16F)
            continue
        q = V
        for l in range(1, LEVELS):
            q = pp('WSNB_Mip%d' % l, pass_pyramid(l, k, skip), [q, mipmap, mipboxes, voxboxes] + [radius] * skip,
                   qt, x0, 400 + 120 * l, fmt=FMT_16F)
        V = pp('WSNB_Pass%d' % (k + 1), pass_blur(k, skip), [V, q, normals, splat, radius, voxmap, voxboxes, mipboxes],
               atl, x0 + 100, 300, fmt=FMT_16F)
    if lookup == 'cubic':
        resolve = pass_resolve_cubic(fast)
    else:
        resolve = pass_resolve_skip() if skip else pass_resolve(fast)
    out = g.pixel_processor('WSNB_Resolve', resolve,
                            [src, position, uvmask, V, voxfit, coarse, slabx, runtab, voxboxes, bmap, radius],
                            None, color=True, fmt=FMT_16, x=-300 + 200 * MAX_PASSES + 200, y=0)
    output_color(g, 'output', 'Output', out, x=-300 + 200 * MAX_PASSES + 400, y=0)
    if lite:
        desc = ('v%s - Lite version of World Space Non Uniform Blur: the same non-uniform (bokeh) blur done in 3D '
                'using the mesh position, continuous across UV seams, with a smaller voxel budget and no Softness, '
                'for much less work per evaluation. Pieces laid out apart get their own voxel grids.' % LITE_VERSION)
    else:
        desc = ('v%s - Non-uniform (bokeh) blur done in 3D using the mesh position, so it is continuous across UV '
                'seams. The Blur Map (or the input itself) scales the blur per texel like the 2D Non-Uniform Blur. '
                "Pieces laid out apart get their own voxel grids, so spreading them out doesn't coarsen the blur, "
                "and they don't blur into each other." % VERSION)
    return graph_xml(g, desc), len(g.nodes)


LITE_FILE_UID = '{9c1e5d37-2b84-4f6a-a0d9-3e7b58c4f215}'


def build_xml(fast=True, lookup=None, skip=None, lite=False):
    """The package XML. fast=False builds the plain graph the speed-ups are tested against
    (same output, more work); skip (default: as fast) only does the work that is used: inactive
    pass slots, unused mip levels, unblurred texels ... (same output again). lite=True builds
    World Space Non Uniform Blur Lite at the table sizes ref is configured to (see build())."""
    lookup = lookup or 'quadratic'
    pkg_uid = new_uid()
    colour, n = build_colour(fast, lookup, skip, lite)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><package><identifier v="%s"/><formatVersion v="1.1.0.202302"/>'
           '<updaterVersion v="1.1.0.202302"/><fileUID v="%s"/>'
           '<versionUID v="0"/><dependencies><dependency><filename v="?himself"/><uid v="%d"/><type v="package"/>'
           '<fileUID v="0"/><versionUID v="0"/></dependency></dependencies><content>%s</content></package>'
           % (LITE_GRAPH_ID if lite else GRAPH_ID,
              LITE_FILE_UID if lite else '{4f6b2a1e-93c7-4d0a-b8e5-7c2d91a3f604}', pkg_uid, colour))
    return xml, n


def ref_config():
    """ref's current sizes, as configure() arguments"""
    return (ref.PMAX, ref.AW, ref.AH, ref.CW, ref.CH, ref.DMAX, ref.S_MAX, ref.R_MAX, dict(ref.QUALITY),
            ref.FRAME_GRID, ref.SUB, ref.AREA_GRID)


def build(path, fast=True, lookup=None, skip=None, lite=False, config=None):
    """Writes the package. lite=True: World Space Non Uniform Blur Lite, at LITE_CONFIG's table
    sizes (config: other ref.configure() arguments, for the reduced-size tests); ref's sizes
    are put back afterwards."""
    before = ref_config()
    if lite or config is not None:
        ref.configure(*(config if config is not None else LITE_CONFIG))
    try:
        xml, n = build_xml(fast, lookup, skip, lite)
    finally:
        ref.configure(*before)
    with open(path, 'w') as f:
        f.write(xml)
    return n


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--lite']
    lite = '--lite' in sys.argv
    out = args[0] if args else os.path.join(os.path.dirname(HERE), LITE_OUTPUT_NAME if lite else OUTPUT_NAME)
    n = build(out, lite=lite)
    from check_types import check
    probs = check(out)
    if probs:
        raise SystemExit('type check failed:\n' + '\n'.join(probs))
    print('wrote', out, n, 'nodes', os.path.getsize(out) // 1024, 'KB')
