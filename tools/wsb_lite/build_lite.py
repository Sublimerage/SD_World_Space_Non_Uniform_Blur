"""
Generate world_space_mask_blur_lite_v<VERSION>.sbs: World Space Mask Blur Lite,
the projection version (see reference/world_space_blur_lite.py, its spec).

    python designer/build_lite.py            -> designer/world_space_mask_blur_lite_v1.0.sbs

Reuses the v2.0 generator (designer/build_sbs.py) for the expression DSL and
the mesh-only passes up to v2.0's boxes (samples, coarse sort, slabs, runs,
boxes) and its bitonic sort. The Lite passes are prefixed WSB_L.

Reusable entry points for other filters (see docs/lite-core.md):
  build_mesh_core(g, mask, position, uvmask) -> dict of the mesh-only nodes
                                                (boxes, views, column sort, lookup)
  build_blur(g, core, image, position, uvmask) -> the blurred image (per stroke)
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "reference"))

import build_sbs as B  # noqa: E402
from build_sbs import (F1, F2, F3, F4, Program, Var, box_of, box_uv, clamp, const, dims_ex, dot,  # noqa: E402
                       fceil, floor, flag, fmax, fmin, fexp, fsqrt, frame_a, frame_b, ge, gt, half_uv,
                       index_uv, input_size, land, le, lift, lin_dims, lin_uv, lor, lt, my_half, my_index,
                       on, param, pos, sample_col, sample_lum, select, snap, vec, fabs)
import world_space_blur as ref  # noqa: E402
import world_space_blur_lite as lref  # noqa: E402

VERSION = "1.0"
OUTPUT_NAME = f"world_space_mask_blur_lite_v{VERSION}.sbs"
SLOTS = 8                     # view table entries per box: 6 views, 2 unused
RADIUS_DEFAULT = 0.01


def NE():
    """View table entries."""
    return ref.S_MAX * ref.R_MAX * SLOTS


def vt_dims():
    return lin_dims(NE())


def vt_uv(e, half):
    W, H = vt_dims()
    return half_uv(e, half, W, H)


def vw_uv(i):
    """Texel-centre UV of entry i of the VW x VH view images."""
    return lin_uv(i, lref.VW, lref.VH)


def comp(v, i):
    """Component i (0, 1, 2; a float expression) of a float3."""
    return select(lt(i, 0.5), v.x, select(lt(i, 1.5), v.y, v.z))


def axes_of(ai):
    """(u axis, v axis) of the view along axis ai: 0 -> (1, 2), 1 -> (2, 0), 2 -> (0, 1)."""
    return select(lt(ai, 1.5), ai + 1.0, 0.0), select(lt(ai, 0.5), 2.0, select(lt(ai, 1.5), 0.0, 1.0))


def from_axes(ai, za, zu, zv):
    """The float3 whose axis-ai component is za, its u component zu and v component zv."""
    a0, a1 = lt(ai, 0.5), lt(ai, 1.5)
    x = select(a0, za, select(a1, zv, zu))
    y = select(a0, zu, select(a1, za, zv))
    z = select(a0, zv, select(a1, zu, za))
    return vec(x, y, z)


def entry_parts(e):
    """(box, axis, sign) of view table entry e."""
    b = floor(lift(e) / float(SLOTS))
    slot = lift(e) - b * float(SLOTS)
    ai = floor(slot / 2.0)
    k = slot - ai * 2.0
    return b, ai, 1.0 - k * 2.0, slot


# ==========================================================================
# Mesh only
# ==========================================================================

def pass_frame_samples():
    """WSB_LFrame (half table 2 PMAX x PMAX, mesh only): left (Pu.xyz, rep u or
    -1 if unusable), right (Pv.xyz, rep v). Slots: 0 position, 1 uv mask,
    2 PackPos, 3 Coarse, 4 Size."""
    S = ref.SUB
    P = float(ref.PMAX)
    i, cell, bestD, bestUV, isz, half = (Var("i", F1), Var("cell", F2), Var("bestD", F1), Var("bestUV", F2),
                                          Var("isz", F2), Var("half", F1))

    def cell_init():
        px = floor(pos() * const((2.0 * P, P)))
        h = flag(ge(px.x, P))
        return vec(px.x - h * P, px.y)

    def better():
        uv = B._sub_uv(cell.get(), i.get())
        du = uv - (cell.get() + 0.5) / P
        d = dot(du, du)
        return land(gt(sample_lum(1, snap(uv, isz.get())), ref.UV_INSIDE), lt(d, bestD.get())), uv, d

    def result():
        S2 = isz.get()
        r = snap(bestUV.get(), S2)
        p = sample_col(0, r).xyz
        rr = B._r2(3)

        def inside(q):
            return land(gt(fmin(q.x, q.y), 0.0), lt(fmax(q.x, q.y), 1.0))

        def diff(step):
            # as B.texel_diffs (v2.1): forward, else backward within rr, else the shorter in-island
            # neighbour (a stretched island), else none
            f, b = r + step, r - step
            fwd = sample_col(0, f).xyz - p
            bwd = p - sample_col(0, b).xyz
            fin = land(gt(sample_lum(1, f), ref.UV_INSIDE), inside(f))
            bin_ = land(gt(sample_lum(1, b), ref.UV_INSIDE), inside(b))
            f2, b2 = dot(fwd, fwd), dot(bwd, bwd)
            fok, bok = land(fin, lt(f2, rr)), land(bin_, lt(b2, rr))
            short = select(land(fin, B.lnot(land(bin_, lt(b2, f2)))), fwd, select(bin_, bwd, const(0.0, 3)))
            return select(fok, fwd, select(bok, bwd, short)), lor(fin, bin_)

        dx, okx = diff(vec(1.0 / S2.x, 0.0))
        dy, oky = diff(vec(0.0, 1.0 / S2.y))
        Pu, Pv = dx * S2.x, dy * S2.y
        n = Pu[1, 2, 0] * Pv[2, 0, 1] - Pu[2, 0, 1] * Pv[1, 2, 0]
        ok = land(land(gt(sample_col(2, (cell.get() + 0.5) / P).w, 0.5), land(okx, oky)), gt(dot(n, n), 0.0))
        return select(on(half.get()), vec(Pv, r.y), vec(Pu, select(ok, r.x, -1.0)))

    return Program(
        result,
        init=[(isz, lambda: input_size(4)), (half, lambda: flag(ge(floor(pos().x * 2.0 * P), P))),
              (cell, cell_init), (bestUV, lambda: (cell_init() + 0.5) / P), (bestD, lambda: const(1e20)),
              (i, lambda: const(0.0))],
        cond=lambda: ge(i.get(), float(S * S)),
        body=[(bestUV, lambda: select(better()[0], better()[1], bestUV.get())),
              (bestD, lambda: select(better()[0], better()[2], bestD.get())), (i, lambda: i.get() + 1.0)],
        counter=i, max_iter=S * S + 8)


def frame_of(slot, idx):
    """(Pu, ru), (Pv, rv) of sample idx from WSB_LFrame in `slot`."""
    P = ref.PMAX
    return sample_col(slot, half_uv(idx, 0.0, P, P)), sample_col(slot, half_uv(idx, 1.0, P, P))


def patch_box(idx, ru, rv, S):
    """Grown patch square of sample idx relative to its rep: (a0, a1, b0, b1)
    (lref.patch_box; S: texture size)."""
    P = float(ref.PMAX)
    cy = floor(lift(idx) / P)
    cx = lift(idx) - cy * P
    a0, a1 = cx / P - ru, (cx + 1.0) / P - ru
    b0, b1 = cy / P - rv, (cy + 1.0) / P - rv
    ga = (a1 - a0) * lref.PATCH_GROW + 0.5 / S.x
    gb = (b1 - b0) * lref.PATCH_GROW + 0.5 / S.y
    return a0 - ga, a1 + ga, b0 - gb, b1 + gb


def patch_centre(idx, ru, rv, S):
    """Offset (a, b) in UV from the rep of sample idx to its grown patch's centre."""
    a0, a1, b0, b1 = patch_box(idx, ru, rv, S)
    return (a0 + a1) * 0.5, (b0 + b1) * 0.5


def normal_of(Pu, Pv):
    return Pu[1, 2, 0] * Pv[2, 0, 1] - Pu[2, 0, 1] * Pv[1, 2, 0]


def pass_reach_blocks():
    """WSB_LReachBlocks (AREA_GRID^2, mesh only): max over its block of samples of
    the squared distance, along either image axis of the sample's own view,
    from its grown patch's centre to a corner. Slots: 0 LFrame, 1 Size."""
    G, P = ref.AREA_GRID, ref.PMAX
    Bk = P // G
    k, acc = Var("k", F1), Var("acc", F1)

    def upd():
        blk = floor(pos() * float(G))
        c = blk * float(Bk) + B.cell_of_counter(k.get(), float(Bk))
        idx = c.y * float(P) + c.x
        fa, fb = frame_of(0, idx)
        a0, a1, b0, b1 = patch_box(idx, fa.w, fb.w, input_size(1))
        ac, bc = (a0 + a1) * 0.5, (b0 + b1) * 0.5
        ai = dominant(normal_of(fa.xyz, fb.xyz))
        ua, va = axes_of(ai)
        r = const(0.0)
        for a in (a0, a1):
            for b in (b0, b1):
                q = fa.xyz * (a - ac) + fb.xyz * (b - bc)
                du, dv = comp(q, ua), comp(q, va)
                r = fmax(r, fmax(du * du, dv * dv))
        return fmax(acc.get(), select(ge(fa.w, 0.0), r, 0.0))

    return Program(
        lambda: vec(acc.get(), 0.0, 0.0, 1.0),
        init=[(acc, lambda: const(0.0)), (k, lambda: const(0.0))],
        cond=lambda: ge(k.get(), float(Bk * Bk)),
        body=[(acc, upd), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=Bk * Bk + 8)


def pass_reach():
    """WSB_LReach (16 x 16, mesh only): the largest reach. Slot 0: LReachBlocks."""
    G = ref.AREA_GRID
    k, acc = Var("k", F1), Var("acc", F1)
    return Program(
        lambda: vec(fsqrt(acc.get()), 0.0, 0.0, 1.0),
        init=[(acc, lambda: const(0.0)), (k, lambda: const(0.0))],
        cond=lambda: ge(k.get(), float(G * G)),
        body=[(acc, lambda: fmax(acc.get(), sample_col(0, lin_uv(k.get(), G, G)).x)), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=G * G + 8)


def view_wh(e3, ai, p):
    """Fine dims (W, H) of the view along axis ai of a box with extents e3."""
    ua, va = axes_of(ai)
    return dims_ex(comp(e3, ua), p), dims_ex(comp(e3, va), p)


def pass_view_fit():
    """WSB_LViewFit (16 x 16, mesh + Quality): the smallest view pixel p0 whose
    views over all boxes fit the budget (bisection). Slots: 0 Boxes, 1 Bounds."""
    nb = float(ref.S_MAX * ref.R_MAX)
    a, hb, m, dn, ok = Var("a", F1), Var("hb", F1), Var("m", F1), Var("dn", F1), Var("ok", F1)
    acc, k, it = Var("acc", F1), Var("k", F1), Var("it", F1)

    def budget():
        q = B.level()
        total = float(lref.VW * lref.VH)
        out = const(total)
        for lv, f in ((3, 0.5), (2, 0.25), (1, 0.125)):     # innermost choice last
            out = select(lt(q, lv + 0.5), total * f, out)
        return out

    def vol():
        lo, hi = sample_col(0, box_uv(k.get(), 0.0)), sample_col(0, box_uv(k.get(), 1.0))
        e3 = fmax(hi.xyz - lo.xyz, 0.0)
        s = None
        for ai in range(3):
            W, H = view_wh(e3, const(float(ai)), m.get())
            t = (W * H) * 2.0
            s = t if s is None else s + t
        return select(gt(lo.w, 0.5), s, 0.0)

    done = lambda: on(dn.get())                            # noqa: E731
    return Program(
        lambda: vec(hb.get(), 0.0, 0.0, 1.0),
        init=[(a, lambda: const(0.0)), (hb, lambda: frame_a(1).w * 2.0), (m, lambda: const(0.0)),
              (dn, lambda: const(0.0)), (ok, lambda: const(0.0)), (acc, lambda: const(0.0)),
              (k, lambda: const(0.0)), (it, lambda: const(0.0))],
        cond=lambda: ge(it.get(), float(ref.FIT_STEPS)),
        body=[(m, lambda: (a.get() + hb.get()) * 0.5), (dn, lambda: flag(ge(k.get(), nb))),
              (ok, lambda: flag(le(acc.get(), budget()))),
              (a, lambda: select(land(done(), lt(ok.get(), 0.5)), m.get(), a.get())),
              (hb, lambda: select(land(done(), on(ok.get())), m.get(), hb.get())),
              (acc, lambda: select(done(), 0.0, acc.get() + vol())),
              (k, lambda: select(done(), 0.0, k.get() + 1.0)),
              (it, lambda: select(done(), it.get() + 1.0, it.get()))],
        counter=it, max_iter=ref.FIT_STEPS * (ref.S_MAX * ref.R_MAX + 1) + 8)


def pass_view_table(level_slot):
    """WSB_LViewTable0 (fine) / WSB_LViewTable (at the stroke's level): half table
    of S_MAX R_MAX 8 entries. Left (offset, Wl, Hl, valid), right (ou, ov, W, H).
    Slots: 0 Boxes, 1 ViewFit, and the level image (WSB_LLevel) in `level_slot`
    (None: level 0). Offsets: boxes in order, then a box's views in order."""
    W_, H_ = vt_dims()
    e, half, kb, acc = Var("e", F1), Var("half", F1), Var("kb", F1), Var("acc", F1)

    def Bsize():
        return const(1.0) if level_slot is None else frame_b(level_slot).y

    def box_ext(b):
        lo, hi = sample_col(0, box_uv(b, 0.0)), sample_col(0, box_uv(b, 1.0))
        return lo, hi

    def size_of(b, ai):
        lo, hi = box_ext(b)
        W, H = view_wh(fmax(hi.xyz - lo.xyz, 0.0), ai, frame_a(1).x)
        Bs = Bsize()
        return select(gt(lo.w, 0.5), fceil(W / Bs) * fceil(H / Bs), 0.0)

    def box_total(b):
        s = None
        for ai in range(3):
            t = size_of(b, const(float(ai))) * 2.0
            s = t if s is None else s + t
        return s

    def result():
        b, ai, sgn, slot = entry_parts(e.get())
        lo, hi = box_ext(b)
        e3 = fmax(hi.xyz - lo.xyz, 0.0)
        ctr = (lo.xyz + hi.xyz) * 0.5
        p0 = frame_a(1).x
        aic = fmin(ai, 2.0)
        W, H = view_wh(e3, aic, p0)
        Bs = Bsize()
        Wl, Hl = fceil(W / Bs), fceil(H / Bs)
        valid = land(gt(lo.w, 0.5), lt(slot, 5.5))
        # views of this box before this one
        within = None
        for s in range(6):
            t = select(lt(float(s), slot), size_of(b, const(float(s // 2))), 0.0)
            within = t if within is None else within + t
        ua, va = axes_of(aic)
        ou = comp(ctr, ua) - p0 * W * 0.5
        ov = comp(ctr, va) - p0 * H * 0.5
        return select(on(half.get()), vec(ou, ov, W, H), vec(acc.get() + within, Wl, Hl, flag(valid)))

    def kb_end():
        b, _, _, _ = entry_parts(e.get())
        return b

    return Program(
        result,
        init=[(e, lambda: my_half(W_, H_)[0]), (half, lambda: my_half(W_, H_)[1]), (acc, lambda: const(0.0)),
              (kb, lambda: const(0.0))],
        cond=lambda: ge(kb.get(), kb_end()),
        body=[(acc, lambda: acc.get() + box_total(kb.get())), (kb, lambda: kb.get() + 1.0)],
        counter=kb, max_iter=ref.S_MAX * ref.R_MAX + 8)


def pass_view_map():
    """WSB_LViewMap0 / WSB_LViewMap (VW x VH): this pixel's view entry (binary
    search for the first entry whose end is past it) and (x, y) in the view at
    that table's level -> (entry or -1, x, y, 0). Slot 0: the view table."""
    ne = float(NE())
    lo, hi, v, md, past = Var("lo", F1), Var("hi", F1), Var("v", F1), Var("md", F1), Var("past", F1)

    def end(i):
        t = sample_col(0, vt_uv(i, 0.0))
        return t.x + select(gt(t.w, 0.5), t.y * t.z, 0.0)

    def result():
        i = lo.get()
        t = sample_col(0, vt_uv(i, 0.0))
        l = v.get() - t.x                                    # noqa: E741
        y = floor(l / t.y)
        return vec(select(lt(i, ne), i, -1.0), l - y * t.y, y, 0.0)

    return Program(
        result,
        init=[(v, lambda: my_index(lref.VW, lref.VH)), (lo, lambda: const(0.0)), (hi, lambda: const(ne)),
              (md, lambda: const(0.0)), (past, lambda: const(0.0))],
        cond=lambda: ge(lo.get(), hi.get()),
        body=[(md, lambda: floor((lo.get() + hi.get()) * 0.5)), (past, lambda: flag(gt(end(md.get()), v.get()))),
              (hi, lambda: select(on(past.get()), md.get(), hi.get())),
              (lo, lambda: select(on(past.get()), lo.get(), md.get() + 1.0))],
        counter=lo, max_iter=NE().bit_length() + 8)


def pass_col_level():
    """WSB_LColLevel (16 x 16, mesh + Quality): the lookup's column grid, as
    lref.col_level. A (r, L, B, kc), B (0, B, 0, 1): read like WSB_LLevel by
    the view table pass. Slots: 0 ViewFit, 1 LReach."""
    def result():
        r = frame_a(1).x / frame_a(0).x + 1.0
        L, Bs = const(0.0), const(1.0)
        for _ in range(lref.MAX_LEVEL):
            up = gt(r, Bs * float(lref.MAX_REACH_CELLS))
            L, Bs = select(up, L + 1.0, L), select(up, Bs * 2.0, Bs)
        kc = fmin(fceil(r / Bs), float(lref.MAX_REACH_CELLS))
        return select(lt(pos().x, 0.5), vec(r, L, Bs, kc), vec(0.0, Bs, 0.0, 1.0))
    return Program(result)


def dominant(n):
    """Axis (0, 1, 2) of the largest |component| of float3 n, ties to the lower axis."""
    an = fmax(n, -n)
    return select(land(ge(an.x, an.y), ge(an.x, an.z)), 0.0, select(ge(an.y, an.z), 1.0, 2.0))


def pass_col_keys():
    """WSB_LColKeys (PMAX^2, mesh only) -> (column id, sample index, 0, 1): the
    column of the sample's patch centre in the view of its box along its
    dominant axis.
    Slots: 0 PackPos, 1 Coarse, 2 SlabX, 3 RunTab, 4 LFrame, 5 ColTable,
    6 ColLevel, 7 ViewFit, 8 Size."""
    def result():
        pp = sample_col(0, pos())
        b, found = box_of(pp.xyz, 1, 2, 3)
        px = B._cell()
        idx = px.y * float(ref.PMAX) + px.x
        fa, fb = frame_of(4, idx)
        Pu, Pv = fa.xyz, fb.xyz
        ai = dominant(normal_of(Pu, Pv))
        e0 = b * float(SLOTS) + ai * 2.0
        tl, tr = sample_col(5, vt_uv(e0, 0.0)), sample_col(5, vt_uv(e0, 1.0))
        ua, va = axes_of(ai)
        p0, Bs = frame_a(7).x, frame_b(6).y
        ac, bc = patch_centre(idx, fa.w, fb.w, input_size(8))
        cp = pp.xyz + Pu * ac + Pv * bc
        xs = clamp(floor((comp(cp, ua) - tr.x) / p0), 0.0, tr.z - 1.0)
        ys = clamp(floor((comp(cp, va) - tr.y) / p0), 0.0, tr.w - 1.0)
        key = tl.x + floor(ys / Bs) * tl.y + floor(xs / Bs)
        ok = land(land(gt(pp.w, 0.5), found), land(ge(fa.w, 0.0), on(tl.w)))
        return vec(select(ok, key, ref.KEY_NONE), idx, 0.0, 1.0)
    return Program(result)


def pass_sorted_samples():
    """WSB_LSorted (PMAX^2, mesh only): (rep position, sample index) in sorted
    order. Slots: 0 sorted keys (Sort1), 1 PackPos."""
    def result():
        idx = sample_col(0, pos()).y
        return vec(sample_col(1, index_uv(idx)).xyz, idx)
    return Program(result)


def pass_lookup():
    """WSB_LLookup (VW x VH, mesh only): the front-most hit of the fine view
    pixel's ray among the samples of its box and axis: the samples of the
    columns within kc of the pixel's own (WSB_LColLevel), in sorted order.
    -> (hit u, hit v, depth, a), a = 0: nothing.
    Slots: 0 ViewMap0, 1 ViewTable0, 2 ViewFit, 3 ColTable, 4 ColLevel,
    5 ColCells, 6 LSorted, 7 LFrame, 8 uv mask, 9 position, 10 Size."""
    P = float(ref.PMAX)
    e, U, V, ai, sgn, kc, pc, ct, ncol = (Var("e", F1), Var("U", F1), Var("V", F1), Var("ai", F1), Var("sgn", F1),
                                          Var("kc", F1), Var("pc", F2), Var("ct", F4), Var("ncol", F1))
    t, se, hv, bz, bid, buv, ba, stop = (Var("t", F1), Var("se", F2), Var("hv", F1), Var("bz", F1), Var("bid", F1),
                                         Var("buv", F2), Var("ba", F1), Var("stop", F1))
    bt, cd, ci = Var("bt", F1), Var("cd", F4), Var("ci", F1)

    def vm():
        return sample_col(0, pos())

    def tab(half):
        return sample_col(1, vt_uv(fmax(vm().x, 0.0), half))

    def candidate():
        """(hit, depth, hit uv, a, idx) of the sample at se.x."""
        s = sample_col(6, index_uv(se.get().x))
        idx = s.w
        fa, fb = frame_of(7, idx)
        Pu, Pv, ru, rv = fa.xyz, fb.xyz, fa.w, fb.w
        n = Pu[1, 2, 0] * Pv[2, 0, 1] - Pu[2, 0, 1] * Pv[1, 2, 0]
        an = fmax(n, -n)
        ua, va = axes_of(ai.get())
        m11, m12 = comp(Pu, ua), comp(Pv, ua)
        m21, m22 = comp(Pu, va), comp(Pv, va)
        det = m11 * m22 - m12 * m21
        du_, dv_ = U.get() - comp(s.xyz, ua), V.get() - comp(s.xyz, va)
        al = (du_ * m22 - m12 * dv_) / select(gt(fabs(det), 0.0), det, 1.0)
        be = (m11 * dv_ - du_ * m21) / select(gt(fabs(det), 0.0), det, 1.0)
        S = input_size(10)
        a0, a1, b0, b1 = patch_box(idx, ru, rv, S)
        h0 = vec(ru + al, rv + be)
        # outside the islands: read half a texel back towards the rep (lref.edge_uv)
        inside = gt(sample_lum(8, snap(h0, S)), ref.UV_INSIDE)
        half = const(0.5) / S
        huv = select(inside, h0, h0 - clamp(h0 - vec(ru, rv), -half, half))
        X = s.xyz + Pu * al + Pv * be
        tex = snap(huv, S)
        q = sample_col(9, tex).xyz - X
        diag2 = (dot(Pu, Pu) + dot(Pv, Pv)) / (P * P)
        hit = land(land(ge(ru, 0.0), gt(fabs(det), 0.0)),
                   land(land(ge(al, a0), lt(al, a1)), land(ge(be, b0), lt(be, b1))))
        hit = land(hit, land(gt(sample_lum(8, tex), ref.UV_INSIDE), lt(dot(q, q), diag2)))
        z = sgn.get() * comp(X, ai.get())
        a = fsqrt(dot(n, n)) / fmax(comp(an, ai.get()), 1e-30)
        return hit, z, huv, a, idx

    def cand():
        """(depth or -BIG if no hit, a, hit uv) of the sample at se.x."""
        hit, z, huv, a, _ = candidate()
        return vec(select(land(on(hv.get()), hit), z, -ref.BIG), a, huv.x, huv.y)

    def better():
        z, idx = cd.get().x, ci.get()
        return land(gt(z, -ref.BIG * 0.5), lor(gt(z, bz.get()), land(le(z, bz.get()), land(ge(z, bz.get()),
                                                                                           lt(idx, bid.get())))))

    def cell_range():
        """(start, end) of the samples of column t of the neighbourhood, or
        (0, 0) if it's off the view."""
        w2 = kc.get() * 2.0 + 1.0
        cv_ = floor(t.get() / w2)
        cu = pc.get().x + (t.get() - cv_ * w2) - kc.get()
        cv = pc.get().y + cv_ - kc.get()
        okc = land(land(ge(cu, 0.0), lt(cu, ct.get().y)), land(ge(cv, 0.0), lt(cv, ct.get().z)))
        r = sample_col(5, vw_uv(ct.get().x + cv * ct.get().y + cu)).xy
        return select(okc, r, const(0.0, 2))

    def upd_se():
        return select(on(hv.get()), se.get() + vec(1.0, 0.0), cell_range())

    def e0():
        b, a_, _, _ = entry_parts(fmax(e.get(), 0.0))
        return b * float(SLOTS) + a_ * 2.0

    def result():
        found = gt(bid.get(), -0.5)
        return select(found, vec(buv.get(), bz.get(), ba.get()), const(0.0, 4))

    return Program(
        result,
        init=[(e, lambda: vm().x),
              (ai, lambda: entry_parts(fmax(e.get(), 0.0))[1]), (sgn, lambda: entry_parts(fmax(e.get(), 0.0))[2]),
              (U, lambda: tab(1.0).x + (vm().y + 0.5) * frame_a(2).x),
              (V, lambda: tab(1.0).y + (vm().z + 0.5) * frame_a(2).x),
              (ct, lambda: sample_col(3, vt_uv(e0(), 0.0))),
              (kc, lambda: frame_a(4).w),
              (pc, lambda: floor(vm()[1, 2] / frame_b(4).y)),
              (ncol, lambda: select(land(ge(e.get(), 0.0), land(on(tab(0.0).w), on(ct.get().w))),
                                    (kc.get() * 2.0 + 1.0) * (kc.get() * 2.0 + 1.0), 0.0)),
              (t, lambda: const(0.0)), (se, lambda: const(0.0, 2)), (hv, lambda: const(0.0)),
              (bz, lambda: const(-ref.BIG)), (bid, lambda: const(-1.0)), (buv, lambda: const(0.0, 2)),
              (ba, lambda: const(0.0)), (bt, lambda: const(0.0)), (cd, lambda: const(0.0, 4)),
              (ci, lambda: const(0.0)), (stop, lambda: flag(lt(ncol.get(), 0.5)))],
        cond=lambda: land(ge(se.get().x, se.get().y), on(stop.get())),
        # hv first: the updates below decide on the state before this step;
        # the candidate is evaluated once (cd, ci) and then only read
        body=[(hv, lambda: flag(lt(se.get().x, se.get().y))), (cd, cand),
              (ci, lambda: sample_col(6, index_uv(se.get().x)).w), (bt, lambda: flag(better())),
              (bz, lambda: select(on(bt.get()), cd.get().x, bz.get())),
              (bid, lambda: select(on(bt.get()), ci.get(), bid.get())),
              (buv, lambda: select(on(bt.get()), cd.get()[2, 3], buv.get())),
              (ba, lambda: select(on(bt.get()), cd.get().y, ba.get())),
              (se, upd_se),
              (t, lambda: select(on(hv.get()), t.get(), t.get() + 1.0)),
              (stop, lambda: flag(ge(t.get(), ncol.get())))],
        counter=t, max_iter=1 << 20)


# ==========================================================================
# Per stroke
# ==========================================================================

def pass_level():
    """WSB_LLevel (16 x 16, Radius + mesh): A (sigma, L, p, sg), B (K, 2^L, 0, 1).
    The level: the first L with sigma <= 2.5 p, by doubling. Slots: 0 AreaTotal,
    1 ViewFit, 2 Size."""
    def result():
        tot = frame_a(0)
        S = input_size(2)
        luv = fsqrt(tot.x * S.x * S.y / fmax(tot.y, 1e-12))
        p0 = frame_a(1).x
        sigma = fmax(param("radius") * luv * 0.5, p0 * lref.MIN_SIGMA_PIXELS)
        L, p, Bs = const(0.0), p0, const(1.0)
        for _ in range(lref.MAX_LEVEL):
            up = gt(sigma, p * lref.PIXELS_PER_SIGMA)
            L, p, Bs = select(up, L + 1.0, L), select(up, p * 2.0, p), select(up, Bs * 2.0, Bs)
        sb2 = fmax(sigma * sigma - p * p * 0.25, (p * 0.3) * (p * 0.3))
        sg = fsqrt(sb2) / p
        return select(lt(pos().x, 0.5), vec(sigma, L, p, sg), vec(fceil(sg * 3.0), Bs, 0.0, 1.0))
    return Program(result)


def _px(slot_map, slot_tab):
    """This pixel's (entry, x, y) and its view table row (offset, Wl, Hl, valid)."""
    m = sample_col(slot_map, pos())
    return m, sample_col(slot_tab, vt_uv(fmax(m.x, 0.0), 0.0))


def pass_fill():
    """WSB_LFill (VW x VH): per level pixel, the front-most surface of its B x B
    fine pixels (others more than sigma + 2 p behind it dropped), as
    (sum m a, sum a, mean depth, has) / B^2. Slots: 0 ViewMap, 1 ViewTable,
    2 ViewTable0, 3 LLookup, 4 mask, 5 LLevel, 6 Size."""
    t, zf, acc = Var("t", F1), Var("zf", F1), Var("acc", F3)
    m_, tb, t0a, t0b, Bs = Var("m", F4), Var("tb", F4), Var("t0a", F4), Var("t0b", F4), Var("Bs", F1)

    def fine():
        """(value, ok) of fine pixel t (phase 1 and 2 use t mod B^2)."""
        Bv = Bs.get()
        tt = t.get() - select(ge(t.get(), Bv * Bv), Bv * Bv, 0.0)
        j = floor(tt / Bv)
        i = tt - j * Bv
        fx = m_.get().y * Bv + i
        fy = m_.get().z * Bv + j
        W, H = t0b.get().z, t0b.get().w
        ok = land(lt(fx, W), lt(fy, H))
        v = sample_col(3, vw_uv(t0a.get().x + fy * W + fx))
        return v, land(ok, gt(v.w, 0.0))

    def upd_zf():
        v, ok = fine()
        return select(land(ok, lt(t.get(), Bs.get() * Bs.get())), fmax(zf.get(), v.z), zf.get())

    def upd_acc():
        v, ok = fine()
        lev = frame_a(5)
        keep = land(land(ok, ge(t.get(), Bs.get() * Bs.get())), ge(v.z, zf.get() - (lev.x + lev.z * 2.0)))
        m = sample_lum(4, snap(v.xy, input_size(6)))
        return acc.get() + select(keep, vec(m * v.w, v.w, v.w * v.z), const(0.0, 3))

    def result():
        n = Bs.get() * Bs.get()
        s0, s1, s2 = acc.get().x / n, acc.get().y / n, acc.get().z / n
        has = gt(s1, 0.0)
        active = land(gt(m_.get().x, -0.5), on(tb.get().w))
        return select(active, vec(s0, s1, select(has, s2 / fmax(s1, 1e-30), 0.0), flag(has)), const(0.0, 4))

    return Program(
        result,
        init=[(m_, lambda: sample_col(0, pos())), (tb, lambda: sample_col(1, vt_uv(fmax(m_.get().x, 0.0), 0.0))),
              (t0a, lambda: sample_col(2, vt_uv(fmax(m_.get().x, 0.0), 0.0))),
              (t0b, lambda: sample_col(2, vt_uv(fmax(m_.get().x, 0.0), 1.0))),
              (Bs, lambda: frame_b(5).y), (zf, lambda: const(-ref.BIG)), (acc, lambda: const(0.0, 3)),
              (t, lambda: const(0.0))],
        cond=lambda: lor(ge(t.get(), Bs.get() * Bs.get() * 2.0), lt(m_.get().x, -0.5)),
        body=[(zf, upd_zf), (acc, upd_acc), (t, lambda: t.get() + 1.0)],
        counter=t, max_iter=2 * (1 << (2 * lref.MAX_LEVEL)) + 8)


def pass_depth_fill():
    """WSB_LDepthFill (VW x VH): an empty pixel next to pixels with a depth takes
    the mean depth of those 4-neighbours (+x, -x, +y, -y). Slots: 0 previous,
    1 ViewMap, 2 ViewTable."""
    def result():
        m, tb = _px(1, 2)
        own = sample_col(0, pos())
        acc, cnt = const(0.0), const(0.0)
        for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
            X, Y = m.y + dx, m.z + dy
            ok = land(land(ge(X, 0.0), lt(X, tb.y)), land(ge(Y, 0.0), lt(Y, tb.z)))
            q = sample_col(0, vw_uv(tb.x + Y * tb.y + X))
            h = land(ok, gt(q.w, 0.5))
            acc = acc + select(h, q.z, 0.0)
            cnt = cnt + select(h, 1.0, 0.0)
        grow = land(lt(own.w, 0.5), gt(cnt, 0.5))
        out = vec(own.x, own.y, select(grow, acc / fmax(cnt, 1.0), own.z), select(grow, 1.0, own.w))
        return select(gt(m.x, -0.5), out, const(0.0, 4))
    return Program(result)


def pass_blur(axis):
    """WSB_LBlurX / Y (VW x VH): depth-aware Gaussian along `axis` of the view:
    weight exp(-t^2 / 2 sg^2) exp(-(z_q - z_c)^2 / 2 sigma^2); taps outside the
    view add nothing; depth and has pass through. Slots: 0 previous,
    1 ViewMap, 2 ViewTable, 3 LLevel."""
    k, acc, m_, tb, c = Var("k", F1), Var("acc", F2), Var("m", F4), Var("tb", F4), Var("c", F4)

    def upd():
        lev = frame_a(3)
        tk = k.get()
        X = m_.get().y + (tk if axis == 0 else 0.0)
        Y = m_.get().z + (tk if axis == 1 else 0.0)
        ok = land(land(ge(X, 0.0), lt(X, tb.get().y)), land(ge(Y, 0.0), lt(Y, tb.get().z)))
        q = sample_col(0, vw_uv(tb.get().x + Y * tb.get().y + X))
        dz = q.z - c.get().z
        wt = fexp(-(tk * tk / (lev.w * lev.w * 2.0))) * fexp(-(dz * dz / (lev.x * lev.x * 2.0)))
        return acc.get() + select(ok, q.xy * wt, const(0.0, 2))

    return Program(
        lambda: select(gt(m_.get().x, -0.5), vec(acc.get(), c.get().z, c.get().w), const(0.0, 4)),
        init=[(m_, lambda: sample_col(1, pos())), (tb, lambda: sample_col(2, vt_uv(fmax(m_.get().x, 0.0), 0.0))),
              (c, lambda: sample_col(0, pos())), (acc, lambda: const(0.0, 2)),
              (k, lambda: -frame_b(3).x)],
        cond=lambda: lor(gt(k.get(), frame_b(3).x), lt(m_.get().x, -0.5)),
        body=[(acc, upd), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=4096)


def pass_resolve():
    """WSB_LResolve (input size, gray, 16 bit): each of the 6 views of the
    texel's box adds its blurred (num, den) at the texel's projection, each of
    the 4 bilinear pixels times exp(-(z_pixel - z_texel)^2 / 2 sigma^2); texels
    outside the islands or in no box skip it. Then Shape and Dither.
    Slots: 0 mask, 1 position, 2 uv mask, 3 blurred views, 4 LLevel, 5 Coarse,
    6 SlabX, 7 RunTab, 8 ViewTable."""
    k, s, go, p, bf = Var("k", F1), Var("s", F2), Var("go", F1), Var("p", F3), Var("bf", F2)

    def box_init():
        b, found = box_of(p.get(), 5, 6, 7)
        return vec(b, flag(found))

    def upd():
        lev = frame_a(4)
        pp, sigma = lev.z, lev.x
        kk = k.get()
        e = bf.get().x * float(SLOTS) + kk
        ta, tb2 = sample_col(8, vt_uv(e, 0.0)), sample_col(8, vt_uv(e, 1.0))
        ai = floor(kk / 2.0)
        sgn = 1.0 - (kk - ai * 2.0) * 2.0
        ua, va = axes_of(ai)
        x = (comp(p.get(), ua) - tb2.x) / pp - 0.5
        y = (comp(p.get(), va) - tb2.y) / pp - 0.5
        x0, y0 = floor(x), floor(y)
        fx, fy = x - x0, y - y0
        z = sgn * comp(p.get(), ai)
        acc = s.get()
        for oy in (0, 1):
            for ox in (0, 1):
                X, Y = x0 + float(ox), y0 + float(oy)
                ok = land(on(ta.w), land(land(ge(X, 0.0), ge(Y, 0.0)), land(lt(X, ta.y), lt(Y, ta.z))))
                q = sample_col(3, vw_uv(ta.x + Y * ta.y + X))
                dz = q.z - z
                f = fexp(-(dz * dz / (sigma * sigma * 2.0)))
                bil = (fx if ox else 1.0 - fx) * (fy if oy else 1.0 - fy)
                acc = acc + select(ok, q.xy * (bil * f), const(0.0, 2))
        return acc

    def result():
        num, den = s.get().x, s.get().y
        ok = land(on(go.get()), gt(den, 1e-6))
        bl = clamp(num / fmax(den, 1e-12), 0.0, 1.0)
        blurred = B.dither(select(B.get_bool("shape_enable"), B.levels(bl), bl))
        return select(ok, blurred, sample_lum(0, pos()))

    return Program(
        result,
        init=[(p, lambda: sample_col(1, pos()).xyz), (bf, box_init),
              (go, lambda: flag(land(gt(sample_lum(2, pos()), 0.5), on(bf.get().y)))),
              (s, lambda: const(0.0, 2)), (k, lambda: const(0.0))],
        cond=lambda: lor(ge(k.get(), 6.0), lt(go.get(), 0.5)),
        body=[(s, upd), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=6 + 8)


# ==========================================================================
# Package
# ==========================================================================

def build_mesh_core(g, mask, position, uvmask):
    """The mesh-only part, shared by every filter built on the Lite core:
    v2.0's samples, boxes and sorted samples, then the views and their lookup.
    Returns the nodes by name."""
    def log2(n):
        return n.bit_length() - 1

    def pp(name, prog, inputs, size, x, y, **kw):
        return g.pixel_processor(name, prog, inputs, size, x=x, y=y, **kw)

    def lin(n):
        w_, h_ = lin_dims(n)
        return log2(w_), log2(h_)

    P, S, R, Dn = ref.PMAX, ref.S_MAX, ref.R_MAX, ref.DMAX
    lp = (log2(P), log2(P))
    vws = (log2(lref.VW), log2(lref.VH))
    one = (4, 4)
    boxt = (log2(2 * R), log2(S))
    bw, bh = lin_dims(S * R * Dn)
    vtw, vth = vt_dims()
    vtab = (log2(2 * vtw), log2(vth))
    n = {}
    n["size"] = g.pixel_processor("WSB_Size", B.pass_size(), [position], None, x=-1400, y=600,
                                  rel_log2=(B.SIZE_LOG2, B.SIZE_LOG2))
    n["bounds"] = pp("WSB_Bounds", B.pass_bounds(), [position, uvmask, n["size"]], one, -1250, 500)
    n["coarse"] = pp("WSB_Coarse", B.pass_coarse(), [n["bounds"]], one, -1150, 500)
    n["pack_pos"] = pp("WSB_PackPos", B.pass_pack_pos(), [position, uvmask, n["size"]], lp, -1250, 300)
    n["pack_area"] = pp("WSB_PackArea", B.pass_pack_area(), [position, uvmask, n["pack_pos"], n["coarse"], n["size"]],
                        lp, -1100, 400)
    n["area_blocks"] = pp("WSB_AreaBlocks", B.pass_area_blocks(), [n["pack_area"]], (log2(ref.AREA_GRID),) * 2,
                          -1000, 450)
    n["area_total"] = pp("WSB_AreaTotal", B.pass_area_total(), [n["area_blocks"]], one, -900, 450)
    keys = pp("WSB_Keys0", B.pass_keys0(), [n["pack_pos"], n["coarse"]], lp, -1100, 850)
    for i, (k, j) in enumerate(ref.sort_steps()):
        keys = pp(f"WSB_Sort0 {k}/{j}", B.pass_sort_step(k, j), [keys], lp, -950 + 30 * (i % 40), 850 + 120 * (i // 40))
    cells0 = pp("WSB_Cells0", B.pass_cells(ref.CW, ref.CH), [keys], (log2(ref.CW), log2(ref.CH)), -650, 1600)
    cellb = pp("WSB_CellBounds", B.pass_cell_bounds(), [keys, cells0, n["pack_pos"]],
               (log2(2 * ref.CW), log2(ref.CH)), -500, 1600)
    colxz = pp("WSB_ColXZ", B.pass_columns(True), [cellb, n["coarse"]], (log2(2 * Dn), log2(Dn)), -350, 1550)
    colxy = pp("WSB_ColXY", B.pass_columns(False), [cellb, n["coarse"]], (log2(2 * Dn), log2(Dn)), -350, 1650)
    occx = pp("WSB_OccX", B.pass_occ_x(), [colxz, n["coarse"]], lin(Dn), -200, 1500)
    n["slabx"] = pp("WSB_SlabX", B.pass_slab_x(), [occx], lin(Dn), -100, 1500)
    slabr = pp("WSB_SlabRange", B.pass_slab_range(), [n["slabx"], n["coarse"]], lin(S), 0, 1500)
    occzy = pp("WSB_OccZY", B.pass_occ_zy(), [colxz, colxy, slabr, n["coarse"]], (log2(Dn), log2(S)), 100, 1600)
    n["runtab"] = pp("WSB_RunTab", B.pass_run_tab(), [occzy], (log2(Dn), log2(S)), 200, 1600)
    runr = pp("WSB_RunRange", B.pass_run_range(), [n["runtab"]], (log2(R), log2(S)), 300, 1600)
    boxrow = pp("WSB_BoxRow", B.pass_box_row(), [colxz, colxy, slabr, runr], (log2(2 * bw), log2(bh)), 400, 1600)
    n["boxes"] = pp("WSB_Boxes", B.pass_boxes(), [boxrow, slabr], boxt, 500, 1600)
    # Lite
    n["lframe"] = pp("WSB_LFrame", pass_frame_samples(), [position, uvmask, n["pack_pos"], n["coarse"], n["size"]],
                     (log2(2 * P), log2(P)), -600, 300)
    rb = pp("WSB_LReachBlocks", pass_reach_blocks(), [n["lframe"], n["size"]], (log2(ref.AREA_GRID),) * 2, -550, 350)
    n["reach"] = pp("WSB_LReach", pass_reach(), [rb], one, -500, 350)
    n["viewfit"] = pp("WSB_LViewFit", pass_view_fit(), [n["boxes"], n["bounds"]], one, 600, 1500)
    n["vt0"] = pp("WSB_LViewTable0", pass_view_table(None), [n["boxes"], n["viewfit"]], vtab, 700, 1450)
    n["vmap0"] = pp("WSB_LViewMap0", pass_view_map(), [n["vt0"]], vws, 800, 1450)
    n["collevel"] = pp("WSB_LColLevel", pass_col_level(), [n["viewfit"], n["reach"]], one, 700, 1550)
    n["coltable"] = pp("WSB_LColTable", pass_view_table(2), [n["boxes"], n["viewfit"], n["collevel"]], vtab,
                       800, 1550)
    keys = pp("WSB_LColKeys", pass_col_keys(), [n["pack_pos"], n["coarse"], n["slabx"], n["runtab"], n["lframe"],
                                                n["coltable"], n["collevel"], n["viewfit"], n["size"]], lp, 900, 1500)
    for i, (k, j) in enumerate(ref.sort_steps()):
        keys = pp(f"WSB_LSort {k}/{j}", B.pass_sort_step(k, j), [keys], lp, 1000 + 30 * (i % 40),
                  1500 + 120 * (i // 40))
    n["sorted"] = keys
    n["colcells"] = pp("WSB_LColCells", B.pass_cells(lref.VW, lref.VH), [keys], vws, -650, 600)
    lsorted = pp("WSB_LSorted", pass_sorted_samples(), [keys, n["pack_pos"]], lp, -450, 550)
    n["lookup"] = pp("WSB_LLookup", pass_lookup(),
                     [n["vmap0"], n["vt0"], n["viewfit"], n["coltable"], n["collevel"], n["colcells"], lsorted,
                      n["lframe"], uvmask, position, n["size"]], vws, -400, 450)
    return n


def build_blur(g, core, mask, position, uvmask):
    """The per-stroke part: blur grayscale image `mask` over the surface with
    the views of `core` (build_mesh_core). Reads the graph parameters radius,
    dither and the Shape group (shape_enable, grow, contrast). Returns the
    WSB_LResolve node (texture size, 16 bit)."""
    def log2(n):
        return n.bit_length() - 1
    vws = (log2(lref.VW), log2(lref.VH))
    vtw, vth = vt_dims()
    vtab = (log2(2 * vtw), log2(vth))

    def pp(name, prog, inputs, size, x, y, **kw):
        return g.pixel_processor(name, prog, inputs, size, x=x, y=y, **kw)

    level = pp("WSB_LLevel", pass_level(), [core["area_total"], core["viewfit"], core["size"]], (4, 4), -350, 650)
    vt = pp("WSB_LViewTable", pass_view_table(2), [core["boxes"], core["viewfit"], level], vtab, -300, 650)
    vmap = pp("WSB_LViewMap", pass_view_map(), [vt], vws, -250, 650)
    img = pp("WSB_LFill", pass_fill(), [vmap, vt, core["vt0"], core["lookup"], mask, level, core["size"]], vws,
             -200, 300)
    for i in range(lref.DEPTH_FILL):
        img = pp(f"WSB_LDepthFill {i + 1}", pass_depth_fill(), [img, vmap, vt], vws, -150 + 30 * i, 300)
    for axis, name in enumerate("XY"):
        img = pp(f"WSB_LBlur{name}", pass_blur(axis), [img, vmap, vt, level], vws, -50 + 60 * axis, 300)
    resolve = g.pixel_processor("WSB_LResolve", pass_resolve(),
                                [mask, position, uvmask, img, level, core["coarse"], core["slabx"], core["runtab"], vt],
                                None, color=False, fmt=B.FMT_16, x=100, y=0)
    return resolve


def build():
    g = B.Graph("world_space_mask_blur_lite", "World Space Mask Blur Lite")
    icon = os.path.join(HERE, "icon", "world_space_mask_blur.png")
    if os.path.exists(icon):
        g.set_icon(icon)
    mask = g.image_input("input", "Input", color=False, usage="input", primary=True, x=-1600, y=0)
    position = g.image_input("mesh_position", "Mesh Position", color=True, usage="meshPosition",
                             fmt32=True, x=-1600, y=200)
    uvmask = g.image_input("mesh_uv_mask", "Mesh UV Mask", color=False, usage="meshUVMask", x=-1600, y=400)
    g.float_param("radius", "Radius", RADIUS_DEFAULT, 0.0002, 0.25,
                  "Blur radius in UV units (0.01 = 1% of the texture width, about 20 px at 2K), at the mesh's "
                  "average texel density; same meaning as in World Space Mask Blur.", step=0.0002)
    g.float_param("dither", "Dither", ref.DITHER, 0, 2,
                  "Fine noise (in 8-bit steps) that hides banding in smooth gradients if the mask is stored in "
                  "8 bits. 0 = off.", step=0.05)
    g.float_param("quality", "Quality", 4, 1, 4,
                  "View resolution, shared by all pieces: sets the smallest Radius. 4 = finest; each step down "
                  "halves the pixels (faster when the mesh changes, coarser smallest Radius).", step=1)
    g.bool_param("shape_enable", "Enable", False,
                 "Levels after the blur: grow / shrink the mask in 3D and sharpen its edge.", group="Shape")
    g.float_param("grow", "Grow / Shrink", 0.0, -1, 1,
                  "Moves the edge outwards (+) or inwards (-), in 3D, across seams. Works best with some Contrast.",
                  step=0.01, group="Shape", visible_if="shape_enable")
    g.float_param("contrast", "Contrast", 0.5, 0, 1,
                  "0 = the soft blur, 1 = a hard edge. In between: rounded, clean edges.",
                  step=0.01, group="Shape", visible_if="shape_enable")

    core = build_mesh_core(g, mask, position, uvmask)
    resolve = build_blur(g, core, mask, position, uvmask)
    g.output("output", "Output", resolve, x=300, y=0)
    desc = (f"v{VERSION} (experimental) - Lite version of World Space Mask Blur: a 3D blur that can go to a much "
            "smaller Radius (1-2 texels). Each piece is seen from 6 directions and blurred in 2D with depth awareness, "
            "so the blur still crosses UV seams, edges and corners. Surface hidden from its views by another part "
            "of the same piece (e.g. the inner faces of a door frame) is not blurred; approximate on curved "
            "surfaces at large Radius.")
    return ('<?xml version="1.0" encoding="UTF-8"?><package><identifier v="Unsaved Package"/>'
            '<formatVersion v="1.1.0.202302"/><updaterVersion v="1.1.0.202302"/>'
            '<fileUID v="{2c8e5b1a-7d4f-4e93-a0b6-5f1c9d3e7a24}"/><versionUID v="0"/>'
            f'<dependencies><dependency><filename v="?himself"/><uid v="{B.new_uid()}"/><type v="package"/>'
            '<fileUID v="0"/><versionUID v="0"/></dependency></dependencies>'
            f'<content>{g.xml(desc)}</content></package>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=os.path.join(HERE, OUTPUT_NAME))
    args = ap.parse_args()
    xml = build()
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {os.path.normpath(args.output)} ({len(xml) // 1024} KB)")


if __name__ == "__main__":
    main()
