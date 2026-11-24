"""Generates world_space_non_uniform_blur_lite_v<VERSION>.sbs: World Space Non-Uniform Blur Lite.

Bump VERSION below for each release; it goes in the file name and the graph descriptions.
The NumPy spec is tools/lite_ref.py (tools/test_lite.py checks this graph against it).

The mesh part is the shared Lite core (tools/wsb_lite/build_lite.py, vendored from
sublimerage/my-project 015857f, unmodified): each piece seen from the 6 axis directions, every
surface sample owned by one view, and per fine view pixel the front-most hit. Per stroke:

  WSNBL_Level          view level from Intensity (about 16 level pixels per largest radius)
  WSNBL_ViewTable/Map  the views at that level (the core's table pass; a VW x VH/2 map)
  WSNBL_FillDepth      the core's fill: blur map and depth of the front surface per level pixel
  WSNBL_FillColour     the same, for the input colour
  WSNBL_Geometry       (depth, blur amount, normal along the view's image axes) per level pixel
  WSNBL_Start          (colour, depth) - the image the passes work on
  WSNBL_Pass1..16      the Non-Uniform Blur passes, ellipse in the tangent plane, depth-tested taps
  WSNBL_Resolve        each texel reads the 6 views of its piece, depth weighted, at texture size

Run:  python3 tools/build_wsnub_lite.py  [output.sbs]
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'wsb_lite'))
import world_space_blur as ref  # noqa: E402
import world_space_blur_lite as lref  # noqa: E402
import build_sbs as B  # noqa: E402
import build_lite as BL  # noqa: E402
from build_sbs import (Ex, Var, Program, const, lift, vec, fmin, fmax, clamp, floor, fexp, dot, lt, le, gt,  # noqa: E402
                       ge, land, lor, fsqrt, select, get, pos, param, get_bool, sample_col, sample_lum, frame_a,
                       fabs, flag, on, input_size, lin_dims, lin_uv, box_of, hash12, my_index, F1, F2, F3, F4, BOOL,
                       FMT_16, new_uid)
from build_lite import vt_uv, axes_of, from_axes, comp, entry_parts, SLOTS  # noqa: E402
import lite_ref as R  # noqa: E402

assert ref.__file__.startswith(os.path.join(HERE, 'wsb_lite')), 'tools/wsb_lite must come first on sys.path'

VERSION = '1.0'
GRAPH_ID = 'world_space_non_uniform_blur_lite'
OUTPUT_NAME = '%s_v%s.sbs' % (GRAPH_ID, VERSION)
TAGS = 'filter;blur;non uniform;bokeh;world space;lite'
T_INT = 16
LUM = (0.2126, 0.7152, 0.0722)


class DedupEmitter(B.FunctionEmitter):
    """B.FunctionEmitter that also merges structurally identical expression nodes inside one
    emit() call (the same class as in tools/build_wsnub.py)."""

    def emit(self, ex, cache, depth=0):
        keys = cache.setdefault('__keys__', {})
        intern = cache.setdefault('__intern__', {})
        uids = cache.setdefault('__uids__', {})

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


B.FunctionEmitter = DedupEmitter


def log2(n):
    return n.bit_length() - 1


def pass_dims():
    """the per-stroke tables: every view at level >= MIN_LEVEL fits in VW x VH / 2"""
    return lref.VW, lref.VH // 2


def pw_uv(i):
    W, H = pass_dims()
    return lin_uv(i, W, H)


# ----------------------------------------------------------------------------
# small DSL additions (as in tools/build_wsnub.py)

def fsin(v): return Ex('sin', F1, {'a': lift(v)})
def fcos(v): return Ex('cos', F1, {'a': lift(v)})


def iparam(name):
    return Ex('tofloat', F1, {'value': Ex('get_integer1', T_INT, data=('constantValueString', name))})


def cross3(a, b):
    return a[1, 2, 0] * b[2, 0, 1] - a[2, 0, 1] * b[1, 2, 0]


def smoothstep01(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - x * 2.0)


def blur_scale(colour, bm):
    """blur amount from a colour (Use Input As Blur Map) or a blur map value: Invert, clamp, Use"""
    e = select(get_bool('blur_map_from_input'), dot(colour, const(LUM)), bm)
    e = clamp(select(get_bool('invert_blur_map'), 1.0 - e, e), 0.0, 1.0)
    return select(get_bool('use_blur_map'), e, 1.0)


def pass_count():
    """same as the 2D graph: min(Samples, ceil(Intensity * pi))"""
    return fmin(fmax(iparam('samples'), 1.0), B.fceil(fmax(param('intensity'), 0.0) * math.pi))


def pass_on(k):
    return land(gt(pass_count(), k + 0.5), gt(param('intensity'), 0.0))


def dither3(c):
    """triangular noise of +-Dither 8-bit steps, faded out near 0 and 1 (per channel)"""
    px = floor(pos() * get('$size', F2))
    n = hash12(px) + hash12(px + const((37.0, 91.0))) - 1.0
    k = clamp(fmin(c, 1.0 - c) * 255.0, 0.0, 1.0)
    return clamp(c + k * (n * param('dither') * (1.0 / 255.0)), 0.0, 1.0)


# ----------------------------------------------------------------------------
# per stroke

def pass_level():
    """WSNBL_Level (16 x 16): A (fill depth tolerance, L, p, largest radius), B (0, 2^L, 0, 1).
    Slots: 0 AreaTotal, 1 ViewFit, 2 Size."""
    def result():
        tot = frame_a(0)
        S = input_size(2)
        luv = fsqrt(tot.x * S.x * S.y / fmax(tot.y, 1e-12))
        p0 = frame_a(1).x
        rmax = fmax(param('intensity'), 0.0) * (1.0 / 256.0) * luv
        L, p, Bs = const(float(R.MIN_LEVEL)), p0 * float(2 ** R.MIN_LEVEL), const(float(2 ** R.MIN_LEVEL))
        for _ in range(lref.MAX_LEVEL - R.MIN_LEVEL):
            up = gt(rmax, p * R.PIXELS_PER_RADIUS)
            L, p, Bs = select(up, L + 1.0, L), select(up, p * 2.0, p), select(up, Bs * 2.0, Bs)
        return select(lt(pos().x, 0.5), vec(rmax * 0.5, L, p, rmax), vec(0.0, Bs, 0.0, 1.0))
    return Program(result)


def pass_view_map():
    """WSNBL_ViewMap (VW x VH / 2): the core's view map at this table size."""
    ne = float(BL.NE())
    W_, H_ = pass_dims()
    lo, hi, v, md, past = Var('lo', F1), Var('hi', F1), Var('v', F1), Var('md', F1), Var('past', F1)

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
        init=[(v, lambda: my_index(W_, H_)), (lo, lambda: const(0.0)), (hi, lambda: const(ne)),
              (md, lambda: const(0.0)), (past, lambda: const(0.0))],
        cond=lambda: ge(lo.get(), hi.get()),
        body=[(md, lambda: floor((lo.get() + hi.get()) * 0.5)), (past, lambda: flag(gt(end(md.get()), v.get()))),
              (hi, lambda: select(on(past.get()), md.get(), hi.get())),
              (lo, lambda: select(on(past.get()), lo.get(), md.get() + 1.0))],
        counter=lo, max_iter=BL.NE().bit_length() + 8)


def pass_fill_colour():
    """WSNBL_FillColour: like the core's WSB_LFill (same front-surface rule), for the input
    colour: (sum c a) / B^2, 0. Slots: 0 ViewMap, 1 ViewTable, 2 ViewTable0, 3 LLookup,
    4 input, 5 Level, 6 Size."""
    t, zf, acc = Var('t', F1), Var('zf', F1), Var('acc', F3)
    m_, tb, t0a, t0b, Bs = Var('m', F4), Var('tb', F4), Var('t0a', F4), Var('t0b', F4), Var('Bs', F1)

    def fine():
        Bv = Bs.get()
        tt = t.get() - select(ge(t.get(), Bv * Bv), Bv * Bv, 0.0)
        j = floor(tt / Bv)
        i = tt - j * Bv
        fx = m_.get().y * Bv + i
        fy = m_.get().z * Bv + j
        W, H = t0b.get().z, t0b.get().w
        ok = land(lt(fx, W), lt(fy, H))
        v = sample_col(3, BL.vw_uv(t0a.get().x + fy * W + fx))
        return v, land(ok, gt(v.w, 0.0))

    def upd_zf():
        v, ok = fine()
        return select(land(ok, lt(t.get(), Bs.get() * Bs.get())), fmax(zf.get(), v.z), zf.get())

    def upd_acc():
        v, ok = fine()
        lev = frame_a(5)
        keep = land(land(ok, ge(t.get(), Bs.get() * Bs.get())), ge(v.z, zf.get() - (lev.x + lev.z * 2.0)))
        c = sample_col(4, B.snap(v.xy, input_size(6))).xyz
        return acc.get() + select(keep, c * v.w, const(0.0, 3))

    def result():
        n = Bs.get() * Bs.get()
        active = land(gt(m_.get().x, -0.5), on(tb.get().w))
        return select(active, vec(acc.get() / n, 0.0), const(0.0, 4))

    return Program(
        result,
        init=[(m_, lambda: sample_col(0, pos())), (tb, lambda: sample_col(1, vt_uv(fmax(m_.get().x, 0.0), 0.0))),
              (t0a, lambda: sample_col(2, vt_uv(fmax(m_.get().x, 0.0), 0.0))),
              (t0b, lambda: sample_col(2, vt_uv(fmax(m_.get().x, 0.0), 1.0))),
              (Bs, lambda: B.frame_b(5).y), (zf, lambda: const(-ref.BIG)), (acc, lambda: const(0.0, 3)),
              (t, lambda: const(0.0))],
        cond=lambda: lor(ge(t.get(), Bs.get() * Bs.get() * 2.0), lt(m_.get().x, -0.5)),
        body=[(zf, upd_zf), (acc, upd_acc), (t, lambda: t.get() + 1.0)],
        counter=t, max_iter=2 * (1 << (2 * lref.MAX_LEVEL)) + 8)


def pass_geometry():
    """WSNBL_Geometry: (depth, blur amount, n_u, n_v), or (0, -1, 0, 0) where empty. The normal
    comes from the depth slope (one-sided at edges, the smaller slope where both sides exist)
    and faces the view. Slots: 0 FillDepth, 1 FillColour, 2 ViewMap, 3 ViewTable, 4 Level."""
    def result():
        m = sample_col(2, pos())
        tb = sample_col(3, vt_uv(fmax(m.x, 0.0), 0.0))
        own = sample_col(0, pos())
        p = frame_a(4).z
        has = land(gt(m.x, -0.5), gt(own.w, 0.5))
        g = []
        for dx, dy in ((1.0, 0.0), (0.0, 1.0)):
            side = []
            for s in (1.0, -1.0):
                X, Y = m.y + dx * s, m.z + dy * s
                ok = land(land(ge(X, 0.0), lt(X, tb.y)), land(ge(Y, 0.0), lt(Y, tb.z)))
                q = sample_col(0, pw_uv(tb.x + Y * tb.y + X))
                side.append((land(ok, gt(q.w, 0.5)), (q.z - own.z) * s))
            (hp, dp), (hm, dm) = side
            gg = select(land(hp, hm), select(le(fabs(dp), fabs(dm)), dp, dm), select(hp, dp, select(hm, dm, 0.0)))
            g.append(gg / p)
        gu, gv = g
        ln = fsqrt(1.0 + gu * gu + gv * gv)
        a = fmax(own.y, 1e-30)
        colour = sample_col(1, pos()).xyz / a
        eff = blur_scale(colour, own.x / a)
        return select(has, vec(own.z, eff, -gu / ln, -gv / ln), const((0.0, -1.0, 0.0, 0.0)))
    return Program(result)


def pass_start():
    """WSNBL_Start: (colour, depth), or (0, 0, 0, EMPTY) where empty. Slots: 0 FillDepth,
    1 FillColour, 2 ViewMap."""
    def result():
        own = sample_col(0, pos())
        has = land(gt(sample_col(2, pos()).x, -0.5), gt(own.w, 0.5))
        return select(has, vec(sample_col(1, pos()).xyz / fmax(own.y, 1e-30), own.z), const((0.0, 0.0, 0.0, R.EMPTY)))
    return Program(result)


def tangent_frame(n):
    """v2.0's frame: the Reference Axis projected on the surface (another axis near its pole)"""
    ax = vec(param('axis_x'), param('axis_y'), param('axis_z'))
    l2 = dot(ax, ax)
    D = select(gt(l2, 1e-8), ax / fmax(fsqrt(l2), 1e-6), const((0.0, 1.0, 0.0)))
    t = D - n * dot(D, n)
    alt = select(lt(fabs(D.x), 0.9), const((1.0, 0.0, 0.0)), const((0.0, 0.0, 1.0)))
    tb = alt - n * dot(alt, n)
    t = select(gt(dot(t, t), 0.04), t, tb)
    t1 = t / fmax(fsqrt(dot(t, t)), 1e-6)
    return t1, cross3(n, t1)


def pass_bokeh(k):
    """WSNBL_Pass<k+1>: one Non-Uniform Blur pass on the views. Only the view map entry and the
    pixel's own value are read up front; covered pixels of an active pass then run the loop:
    step 0 reads the geometry and the view row, steps 1..Blades add the corner taps (4 pixels
    each, bilinear, depth tested). Slots: 0 previous pass, 1 Geometry, 2 ViewMap, 3 ViewTable,
    4 Level."""
    TWO_PI = 2.0 * math.pi
    m, c, d, tb, act, aw, j = (Var('m', F4), Var('c', F4), Var('d', F4), Var('tb', F4), Var('act', F1),
                               Var('aw', F4), Var('j', F1))

    def nbl():
        return clamp(iparam('blades'), 1.0, 9.0)

    def radius(dd):
        return frame_a(4).w * dd.y * fexp(-(2.0 * k * R.C_DECAY) / fmax(pass_count(), 1.0))

    def go():
        return land(land(pass_on(k), gt(m.get().x, -0.5)), gt(c.get().w, R.EMPTY * 0.5))

    def setup(new, old):
        return select(lt(j.get(), 0.5), new, old)

    def upd_act():
        return setup(flag(gt(radius(d.get()), frame_a(4).z * 0.05)), act.get())

    def tap():
        p = frame_a(4).z
        dd, row = d.get(), tb.get()
        _, ai, sgn, _ = entry_parts(m.get().x)
        ua, va = axes_of(ai)
        nu, nv = dd.z, dd.w
        n = from_axes(ai, sgn * fsqrt(fmax(1.0 - nu * nu - nv * nv, 0.0)), nu, nv)
        t1, t2 = tangent_frame(n)
        r = radius(dd)
        sm = fmax(iparam('samples'), 1.0)
        th = (1.0 / sm) * (TWO_PI / (k + 1.0)) + j.get() * (TWO_PI / nbl())
        an = clamp(param('anisotropy'), 0.0, 1.0)
        phi = param('angle') * -TWO_PI
        s1 = (1.0 - an) * fsin(th)
        c1 = fcos(th)
        X = s1 * fsin(phi) + c1 * fcos(phi)
        Y = s1 * fcos(phi) - fsin(phi) * c1
        off = (t1 * X + t2 * Y) * r
        want = c.get().w + sgn * comp(off, ai)
        tol = r * 0.5 + p * 2.0
        x = m.get().y + comp(off, ua) / p
        y = m.get().z + comp(off, va) / p
        x0, y0 = floor(x), floor(y)
        fx, fy = x - x0, y - y0
        acc = aw.get()
        for oy in (0, 1):
            for ox in (0, 1):
                Xc, Yc = x0 + float(ox), y0 + float(oy)
                inb = land(land(ge(Xc, 0.0), ge(Yc, 0.0)), land(lt(Xc, row.y), lt(Yc, row.z)))
                q = sample_col(0, pw_uv(row.x + Yc * row.y + Xc))
                ok = land(inb, lt(fabs(q.w - want), tol))
                bil = (fx if ox else 1.0 - fx) * (fy if oy else 1.0 - fy)
                acc = acc + select(ok, vec(q.xyz, 1.0) * bil, const(0.0, 4))
        return select(land(gt(j.get(), 0.5), on(act.get())), acc, aw.get())

    def result():
        a = aw.get()
        return select(land(go(), on(act.get())), vec(a.xyz / a.w, c.get().w), c.get())

    return Program(
        result,
        init=[(m, lambda: sample_col(2, pos())), (c, lambda: sample_col(0, pos())), (act, lambda: flag(go())),
              (aw, lambda: vec(c.get().xyz, 1.0)), (j, lambda: const(0.0)),
              (d, lambda: const(0.0, 4)), (tb, lambda: const(0.0, 4))],
        cond=lambda: lor(gt(j.get(), nbl()), lt(act.get(), 0.5)),
        body=[(d, lambda: setup(sample_col(1, pos()), d.get())),
              (tb, lambda: setup(sample_col(3, vt_uv(fmax(m.get().x, 0.0), 0.0)), tb.get())),
              (act, upd_act), (aw, tap), (j, lambda: j.get() + 1.0)],
        counter=j, max_iter=12)


def pass_resolve():
    """WSNBL_Resolve (texture size, colour, 16 bit): each of the 6 views of the texel's box adds
    its result at the texel's projection, each of the 4 bilinear pixels times
    exp(-dz^2 / 2 (DEPTH_SIGMA p)^2); the blur fades in by coverage and by the radius in level
    pixels; dithered. Texels outside the islands or in no box keep the input.
    Slots: 0 input, 1 position, 2 uv mask, 3 last pass, 4 Level, 5 Coarse, 6 SlabX, 7 RunTab,
    8 ViewTable, 9 blur map."""
    k, s, go, p_, bf = Var('k', F1), Var('s', F4), Var('go', F1), Var('p', F3), Var('bf', F2)

    def box_init():
        b, found = box_of(p_.get(), 5, 6, 7)
        return vec(b, flag(found))

    def upd():
        lev = frame_a(4)
        pp = lev.z
        sz = pp * R.DEPTH_SIGMA
        kk = k.get()
        e = bf.get().x * float(SLOTS) + kk
        ta, tb2 = sample_col(8, vt_uv(e, 0.0)), sample_col(8, vt_uv(e, 1.0))
        ai = floor(kk / 2.0)
        sgn = 1.0 - (kk - ai * 2.0) * 2.0
        ua, va = axes_of(ai)
        x = (comp(p_.get(), ua) - tb2.x) / pp - 0.5
        y = (comp(p_.get(), va) - tb2.y) / pp - 0.5
        x0, y0 = floor(x), floor(y)
        fx, fy = x - x0, y - y0
        z = sgn * comp(p_.get(), ai)
        acc = s.get()
        for oy in (0, 1):
            for ox in (0, 1):
                X, Y = x0 + float(ox), y0 + float(oy)
                ok = land(on(ta.w), land(land(ge(X, 0.0), ge(Y, 0.0)), land(lt(X, ta.y), lt(Y, ta.z))))
                q = sample_col(3, pw_uv(ta.x + Y * ta.y + X))
                dz = q.w - z
                f = fexp(-(dz * dz / (sz * sz * 2.0)))
                bil = (fx if ox else 1.0 - fx) * (fy if oy else 1.0 - fy)
                acc = acc + select(ok, vec(q.xyz, 1.0) * (bil * f), const(0.0, 4))
        return acc

    def result():
        lev = frame_a(4)
        num, den = s.get().xyz, s.get().w
        src = sample_col(0, pos())
        e_t = blur_scale(src.xyz, sample_lum(9, pos()))
        rp = select(land(gt(pass_count(), 0.5), gt(param('intensity'), 0.0)), lev.w * e_t / lev.z * 0.5, 0.0)
        al = smoothstep01(den / R.COVER_FULL) * smoothstep01(rp)
        ok = land(land(on(go.get()), gt(den, 1e-6)), gt(al, 0.0))
        res = num / fmax(den, 1e-12)
        return select(ok, vec(dither3(src.xyz + (res - src.xyz) * al), src.w), src)

    return Program(
        result,
        init=[(p_, lambda: sample_col(1, pos()).xyz), (bf, box_init),
              (go, lambda: flag(land(gt(sample_lum(2, pos()), 0.5), on(bf.get().y)))),
              (s, lambda: const(0.0, 4)), (k, lambda: const(0.0))],
        cond=lambda: lor(ge(k.get(), 6.0), lt(go.get(), 0.5)),
        body=[(s, upd), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=6 + 8)


# ----------------------------------------------------------------------------
# parameters, inputs, outputs (as in tools/build_wsnub.py)

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


def params_xml():
    BLUR = 'Blur'
    WS = 'World Space'
    axis_desc = 'Object-space direction that Angle 0 points to (projected on the surface).'
    return ''.join([
        param_int('samples', 'Samples',
                  'Number of blur passes. More passes fill the bokeh shape more smoothly (and cost more).',
                  4, 1, 16, BLUR),
        param_float('intensity', 'Intensity',
                    'Blur radius, in the same units as the 2D Non-Uniform Blur (1 = 1/256 of the UV space), '
                    "at the mesh's average texel density.", 10, 0, 50, 0.01, BLUR),
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
        param_float('quality', 'Quality',
                    'View resolution, shared by all pieces (1 = 1/8, 2 = 1/4, 3 = 1/2, 4 = all of a 2048 x 2048 '
                    'budget). Higher keeps finer detail at low Intensity; lower is faster when the mesh changes.',
                    4, 1, 4, 1, WS),
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
    p = os.path.join(os.path.dirname(HERE), 'docs', 'icon', GRAPH_ID + '.png')
    return p if os.path.exists(p) else os.path.join(os.path.dirname(HERE), 'docs', 'icon',
                                                    'world_space_non_uniform_blur.png')


DESCRIPTION = (
    'v%s (Lite) - a cheaper World Space Non Uniform Blur: the same bokeh blur, continuous across UV seams, done on '
    '6 axis views of each piece instead of voxels. Faces hidden from every view by another part of the same '
    'piece (e.g. the inner faces of a door frame, between stacked parts) are not blurred; use World Space Non '
    'Uniform Blur for those meshes.')


def build_colour():
    g = B.Graph(GRAPH_ID, 'World Space Non Uniform Blur Lite')
    if os.path.exists(icon_path()):
        g.set_icon(icon_path())
    I = inputs(g, True)
    g.paraminputs.append(params_xml())
    src, bmap, position, uvmask = I['input'], I['blur_map'], I['mesh_position'], I['mesh_uv_mask']

    def pp(name, prog, ins, size, x, y, **kw):
        return g.pixel_processor(name, prog, ins, size, x=x, y=y, **kw)

    core = BL.build_mesh_core(g, src, position, uvmask)
    PW, PH = pass_dims()
    pws = (log2(PW), log2(PH))
    vtw, vth = BL.vt_dims()
    vtab = (log2(2 * vtw), log2(vth))
    level = pp('WSNBL_Level', pass_level(), [core['area_total'], core['viewfit'], core['size']], (4, 4), -350, 650)
    vt = pp('WSNBL_ViewTable', BL.pass_view_table(2), [core['boxes'], core['viewfit'], level], vtab, -300, 650)
    vmap = pp('WSNBL_ViewMap', pass_view_map(), [vt], pws, -250, 650)
    fz = pp('WSNBL_FillDepth', BL.pass_fill(), [vmap, vt, core['vt0'], core['lookup'], bmap, level, core['size']],
            pws, -200, 400)
    fc = pp('WSNBL_FillColour', pass_fill_colour(), [vmap, vt, core['vt0'], core['lookup'], src, level,
                                                      core['size']], pws, -200, 250)
    geo = pp('WSNBL_Geometry', pass_geometry(), [fz, fc, vmap, vt, level], pws, -100, 400)
    P = pp('WSNBL_Start', pass_start(), [fz, fc, vmap], pws, -100, 250)
    for k in range(R.MAX_PASSES):
        P = pp('WSNBL_Pass%d' % (k + 1), pass_bokeh(k), [P, geo, vmap, vt, level], pws, 0 + 100 * k, 250)
    out = g.pixel_processor('WSNBL_Resolve', pass_resolve(),
                            [src, position, uvmask, P, level, core['coarse'], core['slabx'], core['runtab'], vt, bmap],
                            None, color=True, fmt=FMT_16, x=100 * R.MAX_PASSES + 100, y=0)
    output_color(g, 'output', 'Output', out, x=100 * R.MAX_PASSES + 300, y=0)
    return graph_xml(g, DESCRIPTION % VERSION), len(g.nodes)


def build_xml():
    pkg_uid = new_uid()
    colour, n = build_colour()
    xml = ('<?xml version="1.0" encoding="UTF-8"?><package><identifier v="%s"/><formatVersion v="1.1.0.202302"/>'
           '<updaterVersion v="1.1.0.202302"/><fileUID v="{8d1e5c3a-27b4-4f96-a0c1-3e9b7d52f418}"/>'
           '<versionUID v="0"/><dependencies><dependency><filename v="?himself"/><uid v="%d"/><type v="package"/>'
           '<fileUID v="0"/><versionUID v="0"/></dependency></dependencies><content>%s</content></package>'
           % (GRAPH_ID, pkg_uid, colour))
    return xml, n


def build(path):
    xml, n = build_xml()
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
