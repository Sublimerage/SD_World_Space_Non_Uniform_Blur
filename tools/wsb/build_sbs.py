"""
Generate world_space_mask_blur_v<VERSION>.sbs, the Substance 3D Designer source of the
World Space Mask Blur filter, without needing Designer or the Automation
Toolkit.

    python designer/build_sbs.py              -> designer/world_space_mask_blur_v<VERSION>.sbs

Bump VERSION below for each release; it goes in the file name and the graph description.

The graph is the one described in docs/designer-build-guide.md and implemented
in reference/world_space_blur.py: Pixel Processors that pack the surface into
samples, sort them into a 3D grid (a bitonic sorting network, one node per
step), splat them into voxels, blur the voxels with a separable Gaussian and
read the result back at every texel's 3D position. Each per-pixel function is written below in a
tiny expression DSL and emitted as Designer function-graph XML.

Rules followed when emitting function graphs (from field-tested Designer
behaviour, see README "How the .sbs was made"):
  * binary operators always get operands of the same dimension; scalars are
    broadcast with vector2/3/4 nodes (mixed dimensions silently return 0)
  * While loop: Init / Exit Cond. / Loop Body never share nodes with each
    other or with the rest of the graph; Exit Cond. == True stops the loop
  * each Set statement builds its own fresh Get nodes
  * the root is a plain operator (add 0), never a Sequence / While / If node
"""

import argparse
import itertools
import os
import sys
from xml.sax.saxutils import quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference"))
import world_space_blur as ref  # noqa: E402  (constants shared with the reference)

# Designer type ids
BOOL, F1, F2, F3, F4 = 4, 256, 512, 1024, 2048
DIM = {F1: 1, F2: 2, F3: 3, F4: 4}
FTYPE = {1: F1, 2: F2, 3: F3, 4: F4}

# Shown in Designer / Painter for every generated graph
VERSION = "2.0"
RADIUS_DEFAULT = 0.02
OUTPUT_NAME = f"world_space_mask_blur_v{VERSION}.sbs"
AUTHOR = "Doru Bogdan"
AUTHOR_URL = "https://www.artstation.com/sublime"

# Output formats (Base Parameters > Output Format)
FMT_8, FMT_16, FMT_16F, FMT_32F = 0, 1, 2, 3

# Sample node filtering modes (Image Input, Filtering)
NEAREST, BILINEAR = 0, 1


# ==========================================================================
# Expression DSL
# ==========================================================================

class Ex:
    """A function-graph expression node (not yet emitted)."""

    def __init__(self, op, typ, inputs=None, data=None):
        self.op = op            # Designer function id, e.g. "add"
        self.typ = typ          # output type id
        self.inputs = inputs or {}   # port name -> Ex
        self.data = data        # (xml value tag, value string) for funcData

    # arithmetic with automatic scalar broadcast -------------------------
    def __add__(self, o): return _arith("add", self, o)
    def __radd__(self, o): return _arith("add", o, self)
    def __sub__(self, o): return _arith("sub", self, o)
    def __rsub__(self, o): return _arith("sub", o, self)
    def __mul__(self, o): return _arith("mul", self, o)
    def __rmul__(self, o): return _arith("mul", o, self)
    def __truediv__(self, o): return _arith("div", self, o)
    def __rtruediv__(self, o): return _arith("div", o, self)
    def __neg__(self): return _arith("sub", const(0.0, self.dim), self)

    @property
    def dim(self):
        return DIM[self.typ]

    def __getitem__(self, comps):
        """Swizzle: v[0], v[0, 1], v[0, 1, 2]."""
        if isinstance(comps, int):
            return Ex("swizzle1", F1, {"vector": self}, ("constantValueInt1", str(comps)))
        n = len(comps)
        return Ex(f"swizzle{n}", FTYPE[n], {"vector": self},
                  (f"constantValueInt{n}", " ".join(map(str, comps))))

    @property
    def x(self): return self[0]
    @property
    def y(self): return self[1]
    @property
    def z(self): return self[2]
    @property
    def w(self): return self[3]
    @property
    def xy(self): return self[0, 1]
    @property
    def xyz(self): return self[0, 1, 2]


def _fmt(v):
    """Plain decimal, the way Designer writes floats (no 1e+20 notation)."""
    import numpy as np
    return np.format_float_positional(np.float32(v), trim="-")


def const(v, n=1):
    """Float constant; `v` is a number (broadcast to n) or a sequence."""
    vals = list(v) if isinstance(v, (list, tuple)) else [v] * n
    n = len(vals)
    return Ex(f"const_float{n}", FTYPE[n], data=(f"constantValueFloat{n}" if n > 1 else "constantValueFloat1",
                                                 " ".join(_fmt(x) for x in vals)))


def lift(v):
    return v if isinstance(v, Ex) else const(v)


def splat(s, n):
    """Broadcast a float1 expression to floatN."""
    if n == 1:
        return s
    if n == 2:
        return vec(s, s)
    return vec(splat(s, n - 1), s)


def vec(*parts):
    """vector2/3/4 from components: vec(x, y), vec(float2, z), vec(float3, w), vec(x, y, z)…"""
    parts = [lift(p) for p in parts]
    while len(parts) > 2:
        parts = [vec(parts[0], parts[1])] + parts[2:]
    a, b = parts
    assert b.typ == F1, "last component must be float1"
    n = a.dim + 1
    return Ex(f"vector{n}", FTYPE[n], {"componentsin": a, "componentslast": b})


def _arith(op, a, b):
    a, b = lift(a), lift(b)
    if a.typ != b.typ:
        if a.typ == F1:
            a = splat(a, b.dim)
        elif b.typ == F1:
            b = splat(b, a.dim)
        else:
            raise TypeError(f"{op}: mismatched types {a.typ} {b.typ}")
    return Ex(op, a.typ, {"a": a, "b": b})


def fmin(a, b): return _arith("min", a, b)
def fmax(a, b): return _arith("max", a, b)
def clamp(v, lo, hi): return fmin(fmax(v, lo), hi)


def lerp(a, b, t):
    """a + (b - a) * t, built from same-dimension add/sub/mul (avoids relying on
    how the Lerp node treats a scalar weight with vector operands)."""
    a, b = lift(a), lift(b)
    return a + (b - a) * t


def floor(v): return Ex("floor", lift(v).typ, {"a": lift(v)})
def fexp(v): return Ex("exp", F1, {"a": lift(v)})


def dot(a, b):
    assert a.typ == b.typ
    return Ex("dot", F1, {"a": a, "b": b})


def _cmp(op, a, b):
    a, b = lift(a), lift(b)
    assert a.typ == b.typ == F1
    return Ex(op, BOOL, {"a": a, "b": b})


def lt(a, b): return _cmp("lr", a, b)
def le(a, b): return _cmp("lreq", a, b)
def gt(a, b): return _cmp("gt", a, b)
def ge(a, b): return _cmp("gteq", a, b)
def land(a, b): return Ex("and", BOOL, {"a": a, "b": b})
def lor(a, b): return Ex("or", BOOL, {"a": a, "b": b})
def lnot(a): return Ex("not", BOOL, {"a": a})
def fsqrt(v): return Ex("sqrt", F1, {"a": lift(v)})


def select(c, a, b):
    a, b = lift(a), lift(b)
    if a.typ != b.typ:
        a, b = (splat(a, b.dim), b) if a.typ == F1 else (a, splat(b, a.dim))
    return Ex("ifelse", a.typ, {"condition": c, "ifpath": a, "elsepath": b})


def get(name, typ):
    n = DIM[typ]
    return Ex(f"get_float{n}", typ, data=("constantValueString", name))


def pos():
    return get("$pos", F2)


def param(name):
    return get(name, F1)


def get_bool(name):
    return Ex("get_bool", BOOL, data=("constantValueString", name))


def sample_col(slot, uv, mode=NEAREST):
    return Ex("samplecol", F4, {"pos": uv}, ("constantValueInt2", f"{slot} {mode}"))


def sample_lum(slot, uv, mode=NEAREST):
    return Ex("samplelum", F1, {"pos": uv}, ("constantValueInt2", f"{slot} {mode}"))


# ==========================================================================
# Program = optional While loop + final expression
# ==========================================================================

class Var:
    """A function-graph variable: `v.get()` builds a fresh Get node each call."""

    def __init__(self, name, typ):
        self.name, self.typ = name, typ

    def get(self):
        return get(self.name, self.typ)


class Program:
    def __init__(self, result, init=(), cond=None, body=(), counter=None, max_iter=None):
        self.result = result            # callable -> Ex (built after the loop)
        self.init = list(init)          # [(Var, callable -> Ex)]
        self.cond = cond                # callable -> Ex(bool), True = stop
        self.body = list(body)          # [(Var, callable -> Ex)]
        self.counter = counter          # Var read as the loop's tail value
        self.max_iter = max_iter


# ==========================================================================
# Emission
# ==========================================================================

_uid = itertools.count(1500000001)


def new_uid():
    return next(_uid)


class FunctionEmitter:
    def __init__(self):
        self.nodes = []      # (uid, op, typ, connections[(port, uid)], data, depth)

    def emit(self, ex, cache, depth=0):
        key = id(ex)
        if key in cache:
            return cache[key]
        conns = [(port, self.emit(src, cache, depth + 1)) for port, src in ex.inputs.items()]
        uid = new_uid()
        self.nodes.append([uid, ex.op, ex.typ, conns, ex.data, depth])
        cache[key] = uid
        return uid

    def raw(self, op, typ, conns, data=None, depth=0):
        uid = new_uid()
        self.nodes.append([uid, op, typ, conns, data, depth])
        return uid

    def statement(self, var, build, depth):
        """Set node with its own private expression nodes."""
        value = build()
        assert value.typ == var.typ, f"{var.name}: {value.typ} != {var.typ}"
        vuid = self.emit(value, {}, depth + 1)
        return self.raw("set", var.typ, [("value", vuid)], ("constantValueString", var.name), depth), var.typ

    def chain(self, stmts, depth):
        """Sequence(s1, Sequence(s2, …)) so statements run in order."""
        if len(stmts) == 1:
            return stmts[0]
        first, rest = stmts[0], self.chain(stmts[1:], depth)
        return self.raw("sequence", rest[1], [("seqin", first[0]), ("seqlast", rest[0])], depth=depth), rest[1]

    def program(self, prog):
        if prog.cond is None:
            result = prog.result()
            body_uid = self.emit(result, {}, 2)
            zero = self.emit(const(0.0, result.dim), {}, 2)
            return self.raw("add", result.typ, [("a", body_uid), ("b", zero)], depth=0)

        init = self.chain([self.statement(v, b, 4) for v, b in prog.init], 3)
        cond = self.emit(prog.cond(), {}, 3)
        body_stmts = self.chain([self.statement(v, b, 5) for v, b in prog.body], 4)
        tail = self.emit(prog.counter.get(), {}, 4)
        body = self.raw("sequence", prog.counter.typ, [("seqin", body_stmts[0]), ("seqlast", tail)], depth=3)
        loop = self.raw("while", prog.counter.typ, [("init", init[0]), ("cond", cond), ("loop", body)],
                        ("constantValueInt1", str(prog.max_iter)), depth=2)
        result = prog.result()
        res_uid = self.emit(result, {}, 2)
        seq = self.raw("sequence", result.typ, [("seqin", loop), ("seqlast", res_uid)], depth=1)
        zero = self.emit(const(0.0, result.dim), {}, 1)
        return self.raw("add", result.typ, [("a", seq), ("b", zero)], depth=0)

    def xml(self, root):
        rows = {}
        out = [f'<dynamicValue><rootnode v="{root}"/><paramNodes>']
        for uid, op, typ, conns, data, depth in self.nodes:
            row = rows.get(depth, 0)
            rows[depth] = row + 1
            x, y = -depth * 180, row * 70
            s = f'<paramNode><uid v="{uid}"/><function v="{op}"/><type v="{typ}"/>'
            if conns:
                s += "<connections>" + "".join(
                    f'<connection><identifier v="{p}"/><connRef v="{u}"/></connection>' for p, u in conns
                ) + "</connections>"
            if data is not None:
                tag, val = data
                s += (f'<funcDatas><funcData><name v="{op}"/><constantValue><{tag} v={quoteattr(val)}/>'
                      f'</constantValue></funcData></funcDatas>')
            s += f'<GUILayout><gpos v="{x} {y} 0"/></GUILayout></paramNode>'
            out.append(s)
        out.append("</paramNodes></dynamicValue>")
        return "".join(out)


# ==========================================================================
# The passes (mirror reference/world_space_blur.py)
# ==========================================================================

FRAME_A_UV = (4.5 / 16, 8.5 / 16)    # texel centers inside the 16x16 global images
FRAME_B_UV = (12.5 / 16, 8.5 / 16)
SIZE_LOG2 = -4                        # WSB_Size is 1/16 of the input size


def frame_a(slot):
    """Left half of a 16x16 global image: Bounds (lo, size), Coarse (origin, c),
    Fit (cell, 0, 0, 1), AreaTotal (area, uv, 0, 1), Frame (h, sg, K, sv)."""
    return sample_col(slot, const(FRAME_A_UV))


def frame_b(slot):
    """Right half: Bounds (hi, 1), Coarse (dims, 1), Frame (sigma, length per UV, 0, 1)."""
    return sample_col(slot, const(FRAME_B_UV))


def fceil(v): return -floor(-lift(v))          # (no Ceil node: only nodes already seen working)
def fabs(v): return fmax(v, -v)
def flag(c): return select(c, 1.0, 0.0)
def on(f): return gt(f, 0.5)
def eq(a, b): return lt(fabs(lift(a) - lift(b)), 0.5)


def pass_size():
    """Every pixel holds the input resolution (W, H). The image is uniform, so any
    UV and either filtering mode read the same value."""
    return Program(lambda: vec(get("$size", F2) * float(2 ** -SIZE_LOG2), 0.0, 1.0))


def input_size(slot):
    return sample_col(slot, const((0.5, 0.5))).xy


def snap(uv, S):
    """Centre of the input texel containing uv (S = input size). Reads at texel
    centres return that texel whether the Sample node filters or not."""
    return (floor(uv * S) + 0.5) / S


def level():
    """Quality slider -> 1, 2, … (rounded, clamped)."""
    return clamp(floor(param("quality") + 0.5), 1.0, float(max(ref.QUALITY)))


def q_budget():
    """Voxel budget at the current Quality (fresh nodes each call)."""
    keys = sorted(ref.QUALITY)
    out = const(float(ref.QUALITY[keys[-1]]) ** 3)
    for k in reversed(keys[:-1]):
        out = select(lt(level(), k + 0.5), float(ref.QUALITY[k]) ** 3, out)
    return out


def cell_of_counter(i, n):
    """Grid cell (x, y) of counter i in a row-major n-wide grid."""
    row = floor(i / n)
    return vec(i - n * row, row)


def index_uv(i):
    """Texel-centre UV of entry i of a PMAX x PMAX image (row-major)."""
    return (cell_of_counter(i, float(ref.PMAX)) + 0.5) / float(ref.PMAX)


# table layouts --------------------------------------------------------------
#   linear W x H table: entry i at pixel (i % W, i // W)
#   half table, 2W x H: the same, left half (0) and right half (1) side by side
#   2D tables: entry (x, y) at pixel (x, y); 2D half tables: (x + half * X, y)

def lin_dims(n):
    """(W, H) of a linear table of n (a power of two) entries, as square as possible."""
    k = n.bit_length() - 1
    w = 1 << ((k + 1) // 2)
    return w, n // w


def lin_uv(i, W, H):
    row = floor(i / float(W))
    return (vec(i - row * float(W), row) + 0.5) / const((float(W), float(H)))


def half_uv(i, half, W, H):
    """Entry i of a 2W x H half table (half: 0 left, 1 right; float or Ex)."""
    row = floor(i / float(W))
    x = i - row * float(W) + lift(half) * float(W)
    return (vec(x, row) + 0.5) / const((2.0 * W, float(H)))


def grid2_uv(x, y, W, H, half=None):
    """Entry (x, y) of a W x H 2D table (or of a 2W x H half table)."""
    if half is None:
        return (vec(x, y) + 0.5) / const((float(W), float(H)))
    return (vec(lift(x) + lift(half) * float(W), y) + 0.5) / const((2.0 * W, float(H)))


def my_index(W, H):
    """This pixel's entry in a linear W x H table."""
    px = floor(pos() * const((float(W), float(H))))
    return px.y * float(W) + px.x


def my_half(W, H):
    """This pixel's (entry, half) in a 2W x H half table."""
    px = floor(pos() * const((2.0 * W, float(H))))
    half = flag(ge(px.x, float(W)))
    return px.y * float(W) + px.x - half * float(W), half


def my_grid2(W, H, halves=False):
    """This pixel's (x, y[, half]) in a 2D table."""
    if not halves:
        px = floor(pos() * const((float(W), float(H))))
        return px.x, px.y
    px = floor(pos() * const((2.0 * W, float(H))))
    half = flag(ge(px.x, float(W)))
    return px.x - half * float(W), px.y, half


def minmax(acc, v, half):
    """min for the left half (lo), max for the right half (hi)."""
    return select(on(half), fmax(acc, v), fmin(acc, v))


def big3(half):
    """Start value of a lo (left) / hi (right) reduction."""
    return select(on(half), const(-ref.BIG, 3), const(ref.BIG, 3))


DM = lambda: ref.DMAX                                      # noqa: E731
NB = lambda: ref.S_MAX * ref.R_MAX                         # noqa: E731


def box_uv(b, half):
    """Entry of box b (= slab * R_MAX + run) in a (2 R_MAX) x S_MAX half table."""
    s = floor(lift(b) / float(ref.R_MAX))
    r = lift(b) - s * float(ref.R_MAX)
    return grid2_uv(r, s, ref.R_MAX, ref.S_MAX, half)


def dims_ex(e, c):
    """Cells per axis covering extent e with cells of size c, plus margins."""
    return fceil(e / c - ref.CEIL_EPS) + 2.0 * ref.MARGIN


def lin3(c, d):
    """Linear index of cell c in a d.x x d.y x d.z grid."""
    return (c.z * d.y + c.y) * d.x + c.x


# global passes --------------------------------------------------------------

def pass_bounds():
    """WSB_Bounds (16 x 16, mesh only): left (lo.xyz, size), right (hi.xyz, 1)."""
    G = float(ref.FRAME_GRID)
    i, mn, mx, isz = Var("i", F1), Var("mn", F3), Var("mx", F3), Var("isz", F2)
    big = ref.BIG

    def p_and_valid():
        uv = snap((cell_of_counter(i.get(), G) + ref.SUB_OFFSET) / G, isz.get())
        return sample_col(0, uv).xyz, gt(sample_lum(1, uv), ref.UV_INSIDE)

    def upd_min():
        p, ok = p_and_valid()
        return fmin(mn.get(), select(ok, p, const(big, 3)))

    def upd_max():
        p, ok = p_and_valid()
        return fmax(mx.get(), select(ok, p, const(-big, 3)))

    def result():
        lo, hi = mn.get(), mx.get()
        e = hi - lo
        size = fmax(fmax(fmax(e.x, e.y), e.z), 1e-8)
        return select(lt(pos().x, 0.5), vec(lo, size), vec(hi, 1.0))

    n = ref.FRAME_GRID ** 2
    return Program(
        result,
        init=[(isz, lambda: input_size(2)), (mn, lambda: const(big, 3)), (mx, lambda: const(-big, 3)),
              (i, lambda: const(0.0))],
        cond=lambda: ge(i.get(), float(n)),
        body=[(mn, upd_min), (mx, upd_max), (i, lambda: i.get() + 1.0)],
        counter=i, max_iter=n + 8)


def pass_coarse():
    """WSB_Coarse (16 x 16, mesh only): the coarse grid over the whole texture
    set, the finest that fits CW * CH cells (and DMAX per axis), by bisection.
    Left (origin.xyz, c), right (dims.xyz, 1). Slot 0: Bounds."""
    a, hb, m, ok, it = Var("a", F1), Var("hb", F1), Var("m", F1), Var("ok", F1), Var("it", F1)

    def ext():
        return fmax(frame_b(0).xyz - frame_a(0).xyz, 0.0)

    def fits():
        d = dims_ex(ext(), m.get())
        return flag(land(le(d.x * d.y * d.z, float(ref.CW * ref.CH)),
                         le(fmax(fmax(d.x, d.y), d.z), float(ref.DMAX))))

    def result():
        lo, hi = frame_a(0).xyz, frame_b(0).xyz
        c = hb.get()
        d = dims_ex(ext(), c)
        origin = (lo + hi) * 0.5 - d * c * 0.5
        return select(lt(pos().x, 0.5), vec(origin, c), vec(d, 1.0))

    return Program(
        result,
        init=[(a, lambda: const(0.0)), (hb, lambda: frame_a(0).w), (m, lambda: const(0.0)),
              (ok, lambda: const(0.0)), (it, lambda: const(0.0))],
        cond=lambda: ge(it.get(), float(ref.FIT_STEPS)),
        body=[(m, lambda: (a.get() + hb.get()) * 0.5), (ok, fits),
              (a, lambda: select(on(ok.get()), a.get(), m.get())),
              (hb, lambda: select(on(ok.get()), m.get(), hb.get())),
              (it, lambda: it.get() + 1.0)],
        counter=it, max_iter=ref.FIT_STEPS + 8)


def pass_fit(budget):
    """WSB_SortFit / WSB_VoxFit (16 x 16): the smallest cell size whose grids
    over all boxes hold at most budget() cells, by bisection; each step sums
    the boxes in order. -> (cell, 0, 0, 1). Slots: 0 Boxes, 1 Bounds."""
    nb = float(NB())
    a, hb, m, dn, ok = Var("a", F1), Var("hb", F1), Var("m", F1), Var("dn", F1), Var("ok", F1)
    acc, k, it = Var("acc", F1), Var("k", F1), Var("it", F1)

    def vol():
        lo, hi = sample_col(0, box_uv(k.get(), 0.0)), sample_col(0, box_uv(k.get(), 1.0))
        d = dims_ex(fmax(hi.xyz - lo.xyz, 0.0), m.get())
        return select(gt(lo.w, 0.5), d.x * d.y * d.z, 0.0)

    done = lambda: on(dn.get())                            # noqa: E731
    return Program(
        lambda: vec(hb.get(), 0.0, 0.0, 1.0),
        init=[(a, lambda: const(0.0)), (hb, lambda: frame_a(1).w * 2.0), (m, lambda: const(0.0)),
              (dn, lambda: const(0.0)), (ok, lambda: const(0.0)), (acc, lambda: const(0.0)),
              (k, lambda: const(0.0)), (it, lambda: const(0.0))],
        cond=lambda: ge(it.get(), float(ref.FIT_STEPS)),
        # m, dn and ok first: a, hb, acc, k and it all decide on the state before this step
        body=[(m, lambda: (a.get() + hb.get()) * 0.5), (dn, lambda: flag(ge(k.get(), nb))),
              (ok, lambda: flag(le(acc.get(), budget()))),
              (a, lambda: select(land(done(), lt(ok.get(), 0.5)), m.get(), a.get())),
              (hb, lambda: select(land(done(), on(ok.get())), m.get(), hb.get())),
              (acc, lambda: select(done(), 0.0, acc.get() + vol())),
              (k, lambda: select(done(), 0.0, k.get() + 1.0)),
              (it, lambda: select(done(), it.get() + 1.0, it.get()))],
        counter=it, max_iter=ref.FIT_STEPS * (NB() + 1) + 8)


def pass_frame():
    """WSB_Frame (16 x 16, Radius / Quality): left (h, sigmaBlur/h, K, sigma/h),
    right (sigma, length per UV, 0, 1). Slots: 0 AreaTotal, 1 VoxFit, 2 SortFit, 3 Size."""
    def result():
        tot = frame_a(0)
        S = input_size(3)
        luv = fsqrt(tot.x * S.x * S.y / fmax(tot.y, 1e-12))
        hq, hs = frame_a(1).x, frame_a(2).x
        sigma = fmax(param("radius") * luv * 0.5, hq * ref.MIN_SIGMA_VOXELS)
        h = fmin(fmax(sigma / ref.VOXELS_PER_SIGMA, hq), fmax(hq, hs * ref.MAX_VOXEL_CELLS))
        sb2 = fmax(sigma * sigma - h * h * 0.5, h * h * (ref.MIN_BLUR_VOXELS ** 2))
        sg = fsqrt(sb2) / h
        return select(lt(pos().x, 0.5), vec(h, sg, fceil(sg * 3.0), sigma / h), vec(sigma, luv, 0.0, 1.0))
    return Program(result)


# surface samples --------------------------------------------------------------

def _cell():
    return floor(pos() * float(ref.PMAX))


def _sub_uv(cell, i):
    S, P = float(ref.SUB), float(ref.PMAX)
    return (cell + (cell_of_counter(i, S) + ref.SUB_OFFSET) / S) / P


def pass_pack_pos():
    """WSB_PackPos (PMAX^2, mesh only) -> (rep.xyz, found)"""
    S = ref.SUB
    i, cell, bestD, bestUV = Var("i", F1), Var("cell", F2), Var("bestD", F1), Var("bestUV", F2)
    isz = Var("isz", F2)
    P = float(ref.PMAX)

    def better():
        uv = _sub_uv(cell.get(), i.get())
        du = uv - (cell.get() + 0.5) / P
        d = dot(du, du)
        ok = land(gt(sample_lum(1, snap(uv, isz.get())), ref.UV_INSIDE), lt(d, bestD.get()))
        return ok, uv, d

    def upd_uv():
        ok, uv, _ = better()
        return select(ok, uv, bestUV.get())

    def upd_d():
        ok, _, d = better()
        return select(ok, d, bestD.get())

    def result():
        found = lt(bestD.get(), 1e19)
        p = sample_col(0, snap(bestUV.get(), isz.get())).xyz
        return vec(select(found, p, const(0.0, 3)), select(found, 1.0, 0.0))

    return Program(
        result,
        init=[(isz, lambda: input_size(2)), (cell, _cell),
              (bestUV, lambda: (_cell() + 0.5) / P),
              (bestD, lambda: const(1e20)),
              (i, lambda: const(0.0))],
        cond=lambda: ge(i.get(), float(S * S)),
        # bestUV must update before bestD: both test against the old bestD
        body=[(bestUV, upd_uv), (bestD, upd_d), (i, lambda: i.get() + 1.0)],
        counter=i, max_iter=S * S + 8)


def _taken(uv, rep, r2, slot_pos, slot_uvm):
    """Does read `uv` (texel centre) count: inside an island and near rep?"""
    dp = sample_col(slot_pos, uv).xyz - rep.xyz
    return land(land(gt(sample_lum(slot_uvm, uv), ref.UV_INSIDE), lt(dot(dp, dp), r2)), gt(rep.w, 0.5))


def _r2(slot_coarse):
    rr = frame_a(slot_coarse).w * ref.REJECT_CELLS
    return rr * rr


def texel_area(uv, S, r2):
    """3D area of the texel at uv (texel centre): |dP/dx x dP/dy| from neighbours,
    forward difference, else backward, else 0 (slots: 0 position, 1 uv mask)."""
    p = sample_col(0, uv).xyz

    def inside(q):
        return land(gt(fmin(q.x, q.y), 0.0), lt(fmax(q.x, q.y), 1.0))

    def diff(step):
        f, b = uv + step, uv - step
        fwd = sample_col(0, f).xyz - p
        bwd = p - sample_col(0, b).xyz
        fok = land(land(gt(sample_lum(1, f), ref.UV_INSIDE), lt(dot(fwd, fwd), r2)), inside(f))
        bok = land(land(gt(sample_lum(1, b), ref.UV_INSIDE), lt(dot(bwd, bwd), r2)), inside(b))
        return select(fok, fwd, select(bok, bwd, const(0.0, 3)))

    dx = diff(vec(1.0 / S.x, 0.0))
    dy = diff(vec(0.0, 1.0 / S.y))
    c = dx[1, 2, 0] * dy[2, 0, 1] - dx[2, 0, 1] * dy[1, 2, 0]       # cross product
    return fsqrt(dot(c, c))


def pass_pack_area():
    """WSB_PackArea (PMAX^2, mesh only) -> (area, uv share, 0, 1).
    Slots: 0 position, 1 uv mask, 2 PackPos, 3 Coarse, 4 Size."""
    S = ref.SUB
    i, cell, rep, r2, isz, a = Var("i", F1), Var("cell", F2), Var("rep", F4), Var("r2", F1), Var("isz", F2), \
        Var("a", F1)
    cnt = Var("cnt", F1)

    def take_and_uv():
        uv = snap(_sub_uv(cell.get(), i.get()), isz.get())
        return _taken(uv, rep.get(), r2.get(), 0, 1), uv

    def upd_a():
        ok, uv = take_and_uv()
        return a.get() + select(ok, texel_area(uv, isz.get(), r2.get()), 0.0)

    def upd_cnt():
        ok, _ = take_and_uv()
        return cnt.get() + select(ok, 1.0, 0.0)

    n = float(S * S)
    return Program(
        lambda: vec(a.get() / n, cnt.get() / n, 0.0, 1.0),
        init=[(isz, lambda: input_size(4)), (cell, _cell),
              (rep, lambda: sample_col(2, (_cell() + 0.5) / float(ref.PMAX))),
              (r2, lambda: _r2(3)), (a, lambda: const(0.0)), (cnt, lambda: const(0.0)), (i, lambda: const(0.0))],
        cond=lambda: ge(i.get(), n),
        body=[(a, upd_a), (cnt, upd_cnt), (i, lambda: i.get() + 1.0)],
        counter=i, max_iter=S * S + 8)


def pass_pack_mask():
    """WSB_PackMask (PMAX^2) -> (mean mask * area, area, 0, 1), area in units of
    the mean sample area (so nothing downstream depends on the mesh's size).
    Slots: 0 mask, 1 position, 2 uv mask, 3 PackPos, 4 Coarse, 5 PackArea, 6 Size, 7 AreaTotal."""
    S = ref.SUB
    i, cell, rep, r2 = Var("i", F1), Var("cell", F2), Var("rep", F4), Var("r2", F1)
    msum, cnt, isz = Var("msum", F1), Var("cnt", F1), Var("isz", F2)

    def take_and_uv():
        uv = snap(_sub_uv(cell.get(), i.get()), isz.get())
        return _taken(uv, rep.get(), r2.get(), 1, 2), uv

    def upd_msum():
        ok, uv = take_and_uv()
        return msum.get() + select(ok, sample_lum(0, uv), 0.0)

    def upd_cnt():
        ok, _ = take_and_uv()
        return cnt.get() + select(ok, 1.0, 0.0)

    def result():
        a = sample_col(5, pos()).x / fmax(frame_a(7).x, 1e-30) * float(ref.PMAX * ref.PMAX)
        has = gt(cnt.get(), 0.5)
        return select(has, vec(msum.get() / fmax(cnt.get(), 1.0) * a, a, 0.0, 1.0), const((0.0, 0.0, 0.0, 1.0)))

    n = float(S * S)
    return Program(
        result,
        init=[(isz, lambda: input_size(6)), (cell, _cell),
              (rep, lambda: sample_col(3, (_cell() + 0.5) / float(ref.PMAX))),
              (r2, lambda: _r2(4)),
              (msum, lambda: const(0.0)), (cnt, lambda: const(0.0)), (i, lambda: const(0.0))],
        cond=lambda: ge(i.get(), n),
        body=[(msum, upd_msum), (cnt, upd_cnt), (i, lambda: i.get() + 1.0)],
        counter=i, max_iter=S * S + 8)


def pass_area_blocks():
    """WSB_AreaBlocks (AREA_GRID^2, mesh only): each pixel sums its block of
    samples row by row -> (area, uv, 0, 1). Slot 0: PackArea."""
    B = ref.PMAX // ref.AREA_GRID
    k, acc = Var("k", F1), Var("acc", F2)

    def upd():
        blk = floor(pos() * float(ref.AREA_GRID))
        uv = (blk * float(B) + cell_of_counter(k.get(), float(B)) + 0.5) / float(ref.PMAX)
        return acc.get() + sample_col(0, uv).xy

    return Program(
        lambda: vec(acc.get(), 0.0, 1.0),
        init=[(acc, lambda: const(0.0, 2)), (k, lambda: const(0.0))],
        cond=lambda: ge(k.get(), float(B * B)),
        body=[(acc, upd), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=B * B + 8)


def pass_area_total():
    """WSB_AreaTotal (16 x 16, mesh only): sum of the blocks -> (area, uv, 0, 1)."""
    G = ref.AREA_GRID
    k, acc = Var("k", F1), Var("acc", F2)
    return Program(
        lambda: vec(acc.get(), 0.0, 1.0),
        init=[(acc, lambda: const(0.0, 2)), (k, lambda: const(0.0))],
        cond=lambda: ge(k.get(), float(G * G)),
        body=[(acc, lambda: acc.get() + sample_col(0, lin_uv(k.get(), G, G)).xy), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=G * G + 8)


# sorting ----------------------------------------------------------------------

def pass_keys0():
    """WSB_Keys0 (PMAX^2, mesh only) -> (coarse cell id, sample index, 0, 1).
    Slots: 0 PackPos, 1 Coarse."""
    def result():
        pp = sample_col(0, pos())
        ca, cb = frame_a(1), frame_b(1)
        c = clamp(floor((pp.xyz - ca.xyz) / ca.w), 0.0, cb.xyz - 1.0)
        px = _cell()
        idx = px.y * float(ref.PMAX) + px.x
        return vec(select(gt(pp.w, 0.5), lin3(c, cb.xyz), ref.KEY_NONE), idx, 0.0, 1.0)
    return Program(result)


def _bit(i, b):
    """Bit b (a power of two) of integer-valued float i, as 0.0 / 1.0."""
    return floor(i / b) - floor(i / (2.0 * b)) * 2.0


def pass_sort_step(k, j):
    """One bitonic compare-exchange step: entry i pairs with i xor j and keeps
    the smaller or larger (key, index) of the two."""
    def result():
        px = _cell()
        i = px.y * float(ref.PMAX) + px.x
        bj = _bit(i, float(j))
        partner = i + (1.0 - bj * 2.0) * float(j)
        a = sample_col(0, pos())
        b = sample_col(0, index_uv(partner))
        a_first = lor(lt(a.x, b.x), land(le(a.x, b.x), lt(a.y, b.y)))
        d = _bit(i, float(k)) - _bit(i, float(j))
        keep_min = lt(d * d, 0.5)                 # ascending block and lower element, or neither
        take_a = lor(land(keep_min, a_first), land(lnot(keep_min), lnot(a_first)))
        return select(take_a, a, b)
    return Program(result)


def pass_cells(W, H):
    """WSB_Cells0 / WSB_Cells1 (linear W x H, mesh only) -> (start, end, 0, 1) of
    every cell's samples in the sorted list (two binary searches). Slot 0: sorted keys."""
    L = float(ref.PMAX ** 2)
    s1, s2, cid, i = Var("s1", F2), Var("s2", F2), Var("cid", F1), Var("i", F1)

    def step(s, target):
        lo, hi = s.get().x, s.get().y
        mid = floor((lo + hi) * 0.5)
        key = sample_col(0, index_uv(mid)).x
        nxt = select(lt(key, target), vec(mid + 1.0, hi), vec(lo, mid))
        return select(lt(lo, hi), nxt, s.get())

    iters = int(L).bit_length() + 1
    return Program(
        lambda: vec(s1.get().x, s2.get().x, 0.0, 1.0),
        init=[(cid, lambda: my_index(W, H)), (s1, lambda: vec(0.0, L)), (s2, lambda: vec(0.0, L)),
              (i, lambda: const(0.0))],
        cond=lambda: land(ge(s1.get().x, s1.get().y), ge(s2.get().x, s2.get().y)),
        body=[(s1, lambda: step(s1, cid.get())), (s2, lambda: step(s2, cid.get() + 1.0)),
              (i, lambda: i.get() + 1.0)],
        counter=i, max_iter=iters + 8)


# boxes ------------------------------------------------------------------------

def pass_cell_bounds():
    """WSB_CellBounds (2 CW x CH half table, mesh only): exact lo (left) / hi
    (right) of each coarse cell's samples, w = occupied.
    Slots: 0 sorted keys, 1 Cells0, 2 PackPos."""
    W, H = ref.CW, ref.CH
    ptr, end, acc, half, cid = Var("ptr", F1), Var("end", F1), Var("acc", F3), Var("half", F1), Var("cid", F1)

    def upd():
        idx = sample_col(0, index_uv(ptr.get())).y
        return minmax(acc.get(), sample_col(2, index_uv(idx)).xyz, half.get())

    return Program(
        lambda: vec(acc.get(), flag(gt(end.get(), sample_col(1, lin_uv(cid.get(), W, H)).x))),
        init=[(half, lambda: my_half(W, H)[1]), (cid, lambda: my_half(W, H)[0]),
              (ptr, lambda: sample_col(1, lin_uv(cid.get(), W, H)).x),
              (end, lambda: sample_col(1, lin_uv(cid.get(), W, H)).y),
              (acc, lambda: big3(half.get()))],
        cond=lambda: ge(ptr.get(), end.get()),
        body=[(acc, upd), (ptr, lambda: ptr.get() + 1.0)],
        counter=ptr, max_iter=ref.PMAX ** 2 + 8)


def pass_columns(over_y):
    """WSB_ColXZ (over_y) / WSB_ColXY (2 DMAX x DMAX half table, mesh only):
    exact lo / hi of the coarse column (x, z) over y, or (x, y) over z, w = occupied.
    Slots: 0 CellBounds, 1 Coarse."""
    Dn = DM()
    j, n, acc, occ = Var("j", F1), Var("n", F1), Var("acc", F3), Var("occ", F1)
    x, y, half = Var("x", F1), Var("y", F1), Var("half", F1)

    def cell():
        d = frame_b(1).xyz
        c = vec(x.get(), j.get(), y.get()) if over_y else vec(x.get(), y.get(), j.get())
        return lin3(c, d)

    def count():
        d = frame_b(1).xyz
        inside = land(lt(x.get(), d.x), lt(y.get(), d.z if over_y else d.y))
        return select(inside, d.y if over_y else d.z, 0.0)

    def read():
        return sample_col(0, half_uv(cell(), half.get(), ref.CW, ref.CH))

    return Program(
        lambda: vec(acc.get(), occ.get()),
        init=[(x, lambda: my_grid2(Dn, Dn, True)[0]), (y, lambda: my_grid2(Dn, Dn, True)[1]),
              (half, lambda: my_grid2(Dn, Dn, True)[2]), (n, count), (acc, lambda: big3(half.get())),
              (occ, lambda: const(0.0)), (j, lambda: const(0.0))],
        cond=lambda: ge(j.get(), n.get()),
        body=[(acc, lambda: minmax(acc.get(), read().xyz, half.get())),
              (occ, lambda: fmax(occ.get(), read().w)), (j, lambda: j.get() + 1.0)],
        counter=j, max_iter=Dn + 8)


def _col(slot, x, b, half=0.0):
    """Column table entry (x, b) (a 2 DMAX x DMAX half table)."""
    return sample_col(slot, grid2_uv(x, b, DM(), DM(), half))


def pass_occ_x():
    """WSB_OccX (DMAX linear, mesh only): is column x or a neighbour occupied
    (over all z) -> (occ, 0, 0, 1). Slots: 0 ColXZ, 1 Coarse."""
    Dn = DM()
    W, H = lin_dims(Dn)
    z, n, occ = Var("z", F1), Var("n", F1), Var("occ", F1)

    def x():
        return my_index(W, H)

    def upd():
        dx = frame_b(1).x
        v = _col(0, x(), z.get()).w
        vl = select(gt(x(), 0.5), _col(0, x() - 1.0, z.get()).w, 0.0)
        vr = select(lt(x() + 1.0, dx), _col(0, x() + 1.0, z.get()).w, 0.0)
        return fmax(occ.get(), fmax(v, fmax(vl, vr)))

    return Program(
        lambda: vec(occ.get(), 0.0, 0.0, 1.0),
        init=[(n, lambda: select(lt(x(), frame_b(1).x), frame_b(1).z, 0.0)), (occ, lambda: const(0.0)),
              (z, lambda: const(0.0))],
        cond=lambda: ge(z.get(), n.get()),
        body=[(occ, upd), (z, lambda: z.get() + 1.0)],
        counter=z, max_iter=Dn + 8)


def pass_runs_1d(W, H, read, at, cap, count):
    """Shared scan: run id of bin `at()` (a new run starts at every occupied bin
    after an empty one), capped; -1 if empty. read(j) -> occupied 0/1."""
    j, rid, prev = Var("j", F1), Var("rid", F1), Var("prev", F1)

    def upd_id():
        o = read(j.get())
        return rid.get() + select(land(on(o), lt(prev.get(), 0.5)), 1.0, 0.0)

    return Program(
        lambda: vec(select(on(read(at())), fmin(rid.get(), float(cap - 1)), -1.0), 0.0, 0.0, 1.0),
        init=[(rid, lambda: const(-1.0)), (prev, lambda: const(0.0)), (j, lambda: const(0.0))],
        cond=lambda: gt(j.get(), at()),
        body=[(rid, upd_id), (prev, lambda: read(j.get())), (j, lambda: j.get() + 1.0)],
        counter=j, max_iter=count + 8)


def pass_slab_x():
    """WSB_SlabX (DMAX linear, mesh only): slab id of column x, -1 = empty. Slot 0: OccX."""
    W, H = lin_dims(DM())
    return pass_runs_1d(W, H, lambda j: sample_col(0, lin_uv(j, W, H)).x, lambda: my_index(W, H),
                        ref.S_MAX, DM())


def pass_slab_range():
    """WSB_SlabRange (S_MAX linear, mesh only): (first x, last x, valid, 1) of each slab.
    Slots: 0 SlabX, 1 Coarse."""
    W, H = lin_dims(ref.S_MAX)
    WD, HD = lin_dims(DM())
    x, x0, x1 = Var("x", F1), Var("x0", F1), Var("x1", F1)

    def mine():
        return eq(sample_col(0, lin_uv(x.get(), WD, HD)).x, my_index(W, H))

    return Program(
        lambda: vec(x0.get(), x1.get(), flag(ge(x1.get(), x0.get())), 1.0),
        init=[(x0, lambda: const(ref.BIG)), (x1, lambda: const(-1.0)), (x, lambda: const(0.0))],
        cond=lambda: ge(x.get(), frame_b(1).x),
        body=[(x0, lambda: select(mine(), fmin(x0.get(), x.get()), x0.get())),
              (x1, lambda: select(mine(), fmax(x1.get(), x.get()), x1.get())),
              (x, lambda: x.get() + 1.0)],
        counter=x, max_iter=DM() + 8)


def _slab_range(slot, s):
    W, H = lin_dims(ref.S_MAX)
    return sample_col(slot, lin_uv(s, W, H))


def pass_occ_zy():
    """WSB_OccZY (DMAX x S_MAX, pixel (bin, slab), mesh only): is bin b (or a
    neighbour) occupied along Z / along Y by the slab's columns -> (z, y, 0, 1).
    Slots: 0 ColXZ, 1 ColXY, 2 SlabRange, 3 Coarse."""
    Dn = DM()
    x, oz, oy = Var("x", F1), Var("oz", F1), Var("oy", F1)

    def b(): return my_grid2(Dn, ref.S_MAX)[0]
    def s(): return my_grid2(Dn, ref.S_MAX)[1]

    def dil(slot, lim):
        def at(bb):
            return select(land(ge(bb, 0.0), lt(bb, lim)), _col(slot, x.get(), bb).w, 0.0)
        return fmax(at(b() - 1.0), fmax(at(b()), at(b() + 1.0)))

    def stop():
        sr = _slab_range(2, s())
        return lor(gt(x.get(), sr.y), lt(sr.z, 0.5))

    return Program(
        lambda: vec(oz.get(), oy.get(), 0.0, 1.0),
        init=[(oz, lambda: const(0.0)), (oy, lambda: const(0.0)), (x, lambda: _slab_range(2, s()).x)],
        cond=stop,
        body=[(oz, lambda: fmax(oz.get(), dil(0, frame_b(3).z))),
              (oy, lambda: fmax(oy.get(), dil(1, frame_b(3).y))), (x, lambda: x.get() + 1.0)],
        counter=x, max_iter=Dn + 8)


def pass_run_tab():
    """WSB_RunTab (DMAX x S_MAX, pixel (bin, slab), mesh only) ->
    (run id along Z, run id along Y, runs so far - 1 along Z, along Y); -1 = empty.
    Slot 0: OccZY."""
    Dn, S = DM(), ref.S_MAX
    j, iz, iy, pz, py = Var("j", F1), Var("iz", F1), Var("iy", F1), Var("pz", F1), Var("py", F1)

    def b(): return my_grid2(Dn, S)[0]
    def s(): return my_grid2(Dn, S)[1]
    def occ(jj): return sample_col(0, grid2_uv(jj, s(), Dn, S))

    def run(idv, comp):
        return select(on(occ(b())[comp]), fmin(idv, float(ref.R_MAX - 1)), -1.0)

    return Program(
        lambda: vec(run(iz.get(), 0), run(iy.get(), 1), iz.get(), iy.get()),
        init=[(iz, lambda: const(-1.0)), (iy, lambda: const(-1.0)), (pz, lambda: const(0.0)),
              (py, lambda: const(0.0)), (j, lambda: const(0.0))],
        cond=lambda: gt(j.get(), b()),
        body=[(iz, lambda: iz.get() + select(land(on(occ(j.get()).x), lt(pz.get(), 0.5)), 1.0, 0.0)),
              (iy, lambda: iy.get() + select(land(on(occ(j.get()).y), lt(py.get(), 0.5)), 1.0, 0.0)),
              (pz, lambda: occ(j.get()).x), (py, lambda: occ(j.get()).y), (j, lambda: j.get() + 1.0)],
        counter=j, max_iter=Dn + 8)


def _use_y(slot_run, s):
    last = sample_col(slot_run, grid2_uv(float(DM() - 1), s, DM(), ref.S_MAX))
    return gt(last.w, last.z)


def pass_run_range():
    """WSB_RunRange (R_MAX x S_MAX, pixel (run, slab), mesh only) ->
    (first bin, last bin, valid, split along Y). Slot 0: RunTab."""
    Dn, S, R = DM(), ref.S_MAX, ref.R_MAX
    j, b0, b1 = Var("j", F1), Var("b0", F1), Var("b1", F1)

    def r(): return my_grid2(R, S)[0]
    def s(): return my_grid2(R, S)[1]

    def mine():
        rt = sample_col(0, grid2_uv(j.get(), s(), Dn, S))
        return eq(select(_use_y(0, s()), rt.y, rt.x), r())

    return Program(
        lambda: vec(b0.get(), b1.get(), flag(ge(b1.get(), b0.get())), flag(_use_y(0, s()))),
        init=[(b0, lambda: const(ref.BIG)), (b1, lambda: const(-1.0)), (j, lambda: const(0.0))],
        cond=lambda: ge(j.get(), float(Dn)),
        body=[(b0, lambda: select(mine(), fmin(b0.get(), j.get()), b0.get())),
              (b1, lambda: select(mine(), fmax(b1.get(), j.get()), b1.get())),
              (j, lambda: j.get() + 1.0)],
        counter=j, max_iter=Dn + 8)


def pass_box_row():
    """WSB_BoxRow (half table of S_MAX R_MAX DMAX entries, mesh only): lo / hi of
    box (slab, run) in column x, over the run's bins. Entry = box * DMAX + x.
    Slots: 0 ColXZ, 1 ColXY, 2 SlabRange, 3 RunRange."""
    Dn, S, R = DM(), ref.S_MAX, ref.R_MAX
    W, H = lin_dims(NB() * Dn)
    bb, acc, occ, half = Var("bb", F1), Var("acc", F3), Var("occ", F1), Var("half", F1)
    ent, b1 = Var("ent", F1), Var("b1", F1)

    def parts():
        box = floor(ent.get() / float(Dn))
        x = ent.get() - box * float(Dn)
        s = floor(box / float(R))
        return x, s, box - s * float(R)

    def rr():
        x, s, r = parts()
        return sample_col(3, grid2_uv(r, s, R, S))

    def active():
        x, s, _ = parts()
        sr = _slab_range(2, s)
        return land(land(on(sr.z), on(rr().z)), land(ge(x, sr.x), le(x, sr.y)))

    def read():
        x, _, _ = parts()
        uy = on(rr().w)
        return select(uy, _col(1, x, bb.get(), half.get()), _col(0, x, bb.get(), half.get()))

    return Program(
        lambda: vec(acc.get(), occ.get()),
        init=[(ent, lambda: my_half(W, H)[0]), (half, lambda: my_half(W, H)[1]),
              (acc, lambda: big3(half.get())), (occ, lambda: const(0.0)),
              (bb, lambda: rr().x), (b1, lambda: select(active(), rr().y, -1.0))],
        cond=lambda: gt(bb.get(), b1.get()),
        body=[(acc, lambda: minmax(acc.get(), read().xyz, half.get())),
              (occ, lambda: fmax(occ.get(), read().w)), (bb, lambda: bb.get() + 1.0)],
        counter=bb, max_iter=Dn + 8)


def pass_boxes():
    """WSB_Boxes ((2 R_MAX) x S_MAX half table, mesh only): exact lo (left) / hi
    (right) of each box's samples, w = valid. Slots: 0 BoxRow, 1 SlabRange."""
    Dn, S, R = DM(), ref.S_MAX, ref.R_MAX
    W, H = lin_dims(NB() * Dn)
    x, acc, occ, half, box = Var("x", F1), Var("acc", F3), Var("occ", F1), Var("half", F1), Var("box", F1)

    def sr():
        return _slab_range(1, floor(box.get() / float(R)))

    def read():
        return sample_col(0, half_uv(box.get() * float(Dn) + x.get(), half.get(), W, H))

    def box_init():
        r, s, _ = my_grid2(R, S, True)
        return s * float(R) + r

    return Program(
        lambda: vec(acc.get(), occ.get()),
        init=[(half, lambda: my_grid2(R, S, True)[2]), (box, box_init), (acc, lambda: big3(half.get())),
              (occ, lambda: const(0.0)), (x, lambda: sr().x)],
        cond=lambda: lor(gt(x.get(), sr().y), lt(sr().z, 0.5)),
        body=[(acc, lambda: minmax(acc.get(), read().xyz, half.get())),
              (occ, lambda: fmax(occ.get(), read().w)), (x, lambda: x.get() + 1.0)],
        counter=x, max_iter=Dn + 8)


def pass_grid_table(cell):
    """WSB_SortBoxes / WSB_VoxBoxes ((2 R_MAX) x S_MAX half table): per box, its
    grid with cells of size cell(): left (origin.xyz, offset), right (dims.xyz,
    valid). Boxes are packed in id order in the linear tables. Slot 0: Boxes."""
    S, R = ref.S_MAX, ref.R_MAX
    k, acc, box = Var("k", F1), Var("acc", F1), Var("box", F1)

    def dims_k(kk):
        lo, hi = sample_col(0, box_uv(kk, 0.0)), sample_col(0, box_uv(kk, 1.0))
        return dims_ex(fmax(hi.xyz - lo.xyz, 0.0), cell()), gt(lo.w, 0.5)

    def upd():
        d, ok = dims_k(k.get())
        return acc.get() + select(ok, d.x * d.y * d.z, 0.0)

    def box_init():
        r, s = my_grid2(R, S, True)[:2]
        return s * float(R) + r

    def result():
        lo, hi = sample_col(0, box_uv(box.get(), 0.0)), sample_col(0, box_uv(box.get(), 1.0))
        c = cell()
        d, _ = dims_k(box.get())
        origin = (lo.xyz + hi.xyz) * 0.5 - d * c * 0.5
        right = my_grid2(R, S, True)[2]
        return select(on(right), vec(d, lo.w), vec(origin, acc.get()))

    return Program(
        result,
        init=[(box, box_init), (acc, lambda: const(0.0)), (k, lambda: const(0.0))],
        cond=lambda: ge(k.get(), box.get()),
        body=[(acc, upd), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=NB() + 8)


def box_of(p, s_coarse, s_slab, s_run):
    """(box id, found) of 3D point p, from its coarse column: slab of x, then
    run of z or y."""
    Dn, S, R = DM(), ref.S_MAX, ref.R_MAX
    ca, cb = frame_a(s_coarse), frame_b(s_coarse)
    c = clamp(floor((p - ca.xyz) / ca.w), 0.0, cb.xyz - 1.0)
    s = sample_col(s_slab, lin_uv(c.x, *lin_dims(Dn))).x
    sc = fmax(s, 0.0)
    uy = _use_y(s_run, sc)
    rr = sample_col(s_run, grid2_uv(select(uy, c.y, c.z), sc, Dn, S))
    run = select(uy, rr.y, rr.x)
    return sc * float(R) + fmax(run, 0.0), land(gt(s, -0.5), gt(run, -0.5))


def pass_keys1():
    """WSB_Keys1 (PMAX^2, mesh only) -> (fine cell id, sample index, 0, 1): the
    cell of the sample in its box's sort grid, offset by the box.
    Slots: 0 PackPos, 1 Coarse, 2 SlabX, 3 RunTab, 4 SortBoxes, 5 SortFit."""
    def result():
        pp = sample_col(0, pos())
        b, found = box_of(pp.xyz, 1, 2, 3)
        ta, tb = sample_col(4, box_uv(b, 0.0)), sample_col(4, box_uv(b, 1.0))
        hs = frame_a(5).x
        c = clamp(floor((pp.xyz - ta.xyz) / hs), 0.0, tb.xyz - 1.0)
        key = ta.w + lin3(c, tb.xyz)
        ok = land(land(gt(pp.w, 0.5), found), gt(tb.w, 0.5))
        px = _cell()
        return vec(select(ok, key, ref.KEY_NONE), px.y * float(ref.PMAX) + px.x, 0.0, 1.0)
    return Program(result)


# per stroke -------------------------------------------------------------------

def atlas_uv(i):
    """Texel-centre UV of entry i of the AW x AH linear tables (sort cells, voxels)."""
    return lin_uv(i, ref.AW, ref.AH)


def pass_vox_map():
    """WSB_VoxMap (AW x AH): this voxel's box (binary search for the first box
    whose end is past it) and (i, j, k) in it -> (box or -1, i, j, k).
    Slot 0: VoxBoxes."""
    nb = float(NB())
    lo, hi, v, md, past = Var("lo", F1), Var("hi", F1), Var("v", F1), Var("md", F1), Var("past", F1)

    def end(b):
        ta, tb = sample_col(0, box_uv(b, 0.0)), sample_col(0, box_uv(b, 1.0))
        return ta.w + select(gt(tb.w, 0.5), tb.x * tb.y * tb.z, 0.0)

    def result():
        b = lo.get()
        ta, tb = sample_col(0, box_uv(b, 0.0)), sample_col(0, box_uv(b, 1.0))
        l = v.get() - ta.w                                   # noqa: E741
        row = floor(l / tb.x)
        k = floor(row / tb.y)
        return vec(select(lt(b, nb), b, -1.0), l - row * tb.x, row - k * tb.y, k)

    return Program(
        result,
        init=[(v, lambda: my_index(ref.AW, ref.AH)), (lo, lambda: const(0.0)), (hi, lambda: const(nb)),
              (md, lambda: const(0.0)), (past, lambda: const(0.0))],
        cond=lambda: ge(lo.get(), hi.get()),
        # md and past first: lo and hi both decide on the same middle box
        body=[(md, lambda: floor((lo.get() + hi.get()) * 0.5)), (past, lambda: flag(gt(end(md.get()), v.get()))),
              (hi, lambda: select(on(past.get()), md.get(), hi.get())),
              (lo, lambda: select(on(past.get()), lo.get(), md.get() + 1.0))],
        counter=lo, max_iter=NB().bit_length() + 8)


def pass_splat():
    """WSB_Splat (AW x AH) -> (sum w*area*mask, sum w*area, same, same), tent
    weights. Each voxel walks the cells of its box's sort grid overlapping
    [C - h, C + h] and their samples. Slots: 0 sorted keys, 1 Cells1, 2 PackPos,
    3 PackMask, 4 Frame, 5 VoxMap, 6 VoxBoxes, 7 SortBoxes, 8 SortFit."""
    C, lo, dims, nc = Var("C", F3), Var("lo", F3), Var("dims", F3), Var("nc", F1)
    t, se, hv, acc = Var("t", F1), Var("se", F2), Var("hv", F1), Var("acc", F2)
    sd, so = Var("sd", F3), Var("so", F4)

    def vm():
        return sample_col(5, pos())

    def c_init():
        m = vm()
        return sample_col(6, box_uv(fmax(m.x, 0.0), 0.0)).xyz + (m[1, 2, 3] + 0.5) * frame_a(4).x

    def cell_range(sign):
        o = so.get()
        return clamp(floor((C.get() + frame_a(4).x * sign - o.xyz) / frame_a(8).x), 0.0, sd.get() - 1.0)

    def nc_init():
        d = dims.get()
        return select(gt(vm().x, -0.5), d.x * d.y * d.z, 0.0)

    def upd_acc():
        ptr = se.get().x
        idx = sample_col(0, index_uv(ptr)).y
        uv = index_uv(idx)
        h = frame_a(4).x
        q = fmax(1.0 - fabs(sample_col(2, uv).xyz - C.get()) / h, 0.0)
        w = q.x * q.y * q.z
        return acc.get() + select(gt(hv.get(), 0.5), sample_col(3, uv).xy * w, const(0.0, 2))

    def upd_se():
        d = dims.get()
        tt = t.get()
        row = floor(tt / d.x)
        c = lo.get() + vec(tt - d.x * row, row - d.y * floor(row / d.y), floor(row / d.y))
        nxt = sample_col(1, atlas_uv(so.get().w + lin3(c, sd.get()))).xy
        return select(gt(hv.get(), 0.5), se.get() + vec(1.0, 0.0), nxt)

    return Program(
        # (num, den) twice: zw carries the plain (unsmeared) blur for Smear Strength
        lambda: vec(acc.get(), acc.get().x, acc.get().y),
        init=[(so, lambda: sample_col(7, box_uv(fmax(vm().x, 0.0), 0.0))),
              (sd, lambda: sample_col(7, box_uv(fmax(vm().x, 0.0), 1.0)).xyz),
              (C, c_init), (lo, lambda: cell_range(-1.0)),
              (dims, lambda: cell_range(1.0) - lo.get() + 1.0), (nc, nc_init),
              (t, lambda: const(0.0)), (se, lambda: const(0.0, 2)), (hv, lambda: const(0.0)),
              (acc, lambda: const(0.0, 2))],
        cond=lambda: land(ge(se.get().x, se.get().y), ge(t.get(), nc.get())),
        # hv first: acc, se and t all decide on the state before this step
        body=[(hv, lambda: select(lt(se.get().x, se.get().y), 1.0, 0.0)), (acc, upd_acc), (se, upd_se),
              (t, lambda: select(gt(hv.get(), 0.5), t.get(), t.get() + 1.0))],
        counter=t, max_iter=65536)


def pass_blur(axis):
    """WSB_Blur{X,Y,Z} (AW x AH): 1D Gaussian along `axis` inside the voxel's
    box grid, out to 3 sigma on each side. xy: the smeared blur (with Smear on,
    the side the mask is dragged from gets a longer sigma, reference:
    smear_sides); zw: the plain blur, carried in the same reads for Smear
    Strength. Slots: 0 previous grid, 1 Frame, 2 VoxMap, 3 VoxBoxes."""
    e = [0.0, 0.0, 0.0]
    e[axis] = 1.0
    k, acc, act, ijk = Var("k", F1), Var("acc", F4), Var("act", F1), Var("ijk", F3)
    sn, sp, kmin, kmax = Var("sn", F1), Var("sp", F1), Var("kmin", F1), Var("kmax", F1)
    n, off = Var("n", F3), Var("off", F1)

    def bx():
        return fmax(sample_col(2, pos()).x, 0.0)

    def side(sign):
        fa = frame_a(1)
        sg = fa.y
        d = vec(param("smear_x"), param("smear_y"), param("smear_z"))
        ln = fsqrt(dot(d, d))
        da = d[axis] / fmax(ln, 1e-6)
        enabled = land(get_bool("smear_enable"), gt(ln, 1e-6))
        toward = land(enabled, gt(da, 0.0) if sign < 0 else lt(da, 0.0))
        return select(toward, sg + param("smear_length") * fabs(da) * fa.w, sg)

    def upd_acc():
        fa = frame_a(1)
        tk = k.get()
        q = ijk.get() + const(e) * tk
        qa = q[axis]
        ok = land(ge(qa, 0.0), lt(qa, n.get()[axis]))
        sd = select(lt(tk, 0.0), sn.get(), sp.get())
        in_smear = land(ge(tk, -fceil(sn.get() * 3.0)), le(tk, fceil(sp.get() * 3.0)))
        ws = select(in_smear, fexp(-(tk * tk / (sd * sd * 2.0))), 0.0)
        wp = select(le(fmax(tk, -tk), fa.z), fexp(-(tk * tk / (fa.y * fa.y * 2.0))), 0.0)
        v = sample_col(0, atlas_uv(off.get() + lin3(clamp(q, 0.0, n.get() - 1.0), n.get())))
        return acc.get() + select(ok, v * vec(ws, ws, wp, wp), const(0.0, 4))

    def limit(sigma):
        # at least the plain blur's K taps; taps n or more voxels away are outside the grid
        return fmin(fmax(fceil(sigma * 3.0), frame_a(1).z), n.get()[axis])

    return Program(
        lambda: acc.get(),
        init=[(ijk, lambda: sample_col(2, pos())[1, 2, 3]),
              (act, lambda: flag(gt(sample_col(2, pos()).x, -0.5))),
              (n, lambda: sample_col(3, box_uv(bx(), 1.0)).xyz),
              (off, lambda: sample_col(3, box_uv(bx(), 0.0)).w),
              (sn, lambda: side(-1)), (sp, lambda: side(1)),
              (kmin, lambda: -limit(sn.get())), (kmax, lambda: limit(sp.get())), (acc, lambda: const(0.0, 4)),
              (k, lambda: kmin.get())],
        cond=lambda: lor(gt(k.get(), kmax.get()), lt(act.get(), 0.5)),
        body=[(acc, upd_acc), (k, lambda: k.get() + 1.0)],
        counter=k, max_iter=65536)


def bspline_weights(t):
    """Cubic B-spline weights for voxel offsets -1, 0, +1, +2 at fraction t."""
    t2 = t * t
    t3 = t2 * t
    u = 1.0 - t
    return [u * u * u * (1.0 / 6.0),
            (t3 * 3.0 - t2 * 6.0 + 4.0) * (1.0 / 6.0),
            (t2 * 3.0 + t * 3.0 + 1.0 - t3 * 3.0) * (1.0 / 6.0),
            t3 * (1.0 / 6.0)]


def pass_resolve():
    """Tricubic B-spline lookup of the texel's box grid (64 texel-center
    samples), dithered against 8-bit banding; written as 16 bit.
    Slots: 0 mask, 1 position, 2 uv mask, 3 blurred grid, 4 Frame, 5 Coarse,
    6 SlabX, 7 RunTab, 8 VoxBoxes."""
    def result():
        m = sample_lum(0, pos())
        h = frame_a(4).x
        p = sample_col(1, pos()).xyz
        b, found = box_of(p, 5, 6, 7)
        ta, tb = sample_col(8, box_uv(b, 0.0)), sample_col(8, box_uv(b, 1.0))
        n = tb.xyz
        g = clamp((p - ta.xyz) / h - 0.5, 1.0, n - 3.0)
        i0 = floor(g)
        t = g - i0
        wx, wy, wz = bspline_weights(t.x), bspline_weights(t.y), bspline_weights(t.z)
        s = None
        for c in range(4):
            for bb in range(4):
                for a in range(4):
                    cell = i0 + const([a - 1.0, bb - 1.0, c - 1.0])
                    term = sample_col(3, atlas_uv(ta.w + lin3(cell, n))) * (wx[a] * wy[bb] * wz[c])
                    s = term if s is None else s + term
        ok = land(land(gt(sample_lum(2, pos()), 0.5), gt(s.y, 1e-6)), found)
        bl = clamp(s.x / fmax(s.y, 1e-12), 0.0, 1.0)
        # Smear Strength: towards smeared mass / plain density (adds, never averages away)
        st = select(get_bool("smear_enable"), param("smear_strength"), 1.0)
        undiluted = clamp(s.x / fmax(s.w, 1e-12), 0.0, 1.0)
        bl = bl + (undiluted - bl) * ((st - 1.0) * (1.0 / 3.0))
        blurred = dither(select(get_bool("shape_enable"), levels(bl), bl))
        return select(ok, blurred, m)

    return Program(result)


def levels(b):
    """Shape: Grow (-1..1) moves the 0.5 threshold, Contrast (0..1) narrows the
    ramp around it (smoothstep). Contrast 0, Grow 0 = unchanged."""
    c, g = param("contrast"), param("grow")
    t = 0.5 - g * 0.5
    u = 1.0 - c
    w = u * u * 0.5 + c * 0.002
    lin_ = clamp((b - t) / (w * 2.0) + 0.5, 0.0, 1.0)
    return lin_ + (lin_ * lin_ * (3.0 - lin_ * 2.0) - lin_) * c


def frac(v): return v - floor(v)


def hash12(p):
    """Uniform 0..1 from a texel coordinate p (float2), arithmetic only."""
    q = frac(vec(p, p.x) * 0.1031)
    q = q + dot(q, q[1, 2, 0] + 33.33)
    return frac((q.x + q.y) * q.z)


def dither(b):
    """Triangular noise of +-`dither` 8-bit steps; fades out within one step of
    0 and 1 so fully black / white areas stay clean."""
    px = floor(pos() * get("$size", F2))
    n = hash12(px) + hash12(px + const((37.0, 91.0))) - 1.0
    k = clamp(fmin(b, 1.0 - b) * 255.0, 0.0, 1.0)
    return clamp(b + n * (param("dither") * (1.0 / 255.0)) * k, 0.0, 1.0)


# ==========================================================================
# Compositing graph
# ==========================================================================

def fmt_param(name, tag, value, relative=0):
    return (f'<parameter><name v="{name}"/><relativeTo v="{relative}"/>'
            f'<paramValue><{tag} v="{value}"/></paramValue></parameter>')


class Graph:
    def __init__(self, ident, label):
        self.ident, self.label = ident, label
        self.paraminputs, self.nodes, self.outputs = [], [], []
        self.primary = None
        self.comments = []
        self.icon = ""

    def set_icon(self, png_path):
        """Graph icon (shelf thumbnail): a PNG stored the way Designer does,
        base64 of (4-byte big-endian length + zlib data), i.e. Qt's qCompress."""
        import base64
        import zlib
        with open(png_path, "rb") as f:
            png = f.read()
        data = base64.b64encode(len(png).to_bytes(4, "big") + zlib.compress(png, 9)).decode()
        self.icon = f'<icon><datalength v="{len(png)}"/><format v="png"/><strdata v="{data}"/></icon>'

    def image_input(self, ident, label, color, usage, fmt32=False, primary=False, x=0, y=0):
        # Painter binds inputs by usage *or* identifier. The identifiers below are the
        # ones Painter expects; `usage` is not written because no reference .sbs shows
        # where Designer stores input usages, and an unknown element breaks loading.
        del usage
        uid, node_uid, out_uid = new_uid(), new_uid(), new_uid()
        default = '<constantValueFloat4 v="0 0 0 0"/>' if color else '<constantValueFloat1 v="0"/>'
        self.paraminputs.append(
            f'<paraminput><identifier v="{ident}"/><uid v="{uid}"/><attributes><label v="{label}"/>'
            f'</attributes>'
            f'<isConnectable v="1"/><type v="{1 if color else 2}"/><defaultValue>{default}</defaultValue>'
            f'<defaultWidget><name v=""/><options/></defaultWidget></paraminput>')
        if primary:
            self.primary = uid
        params = fmt_param("format", "constantValueInt1", 3) if fmt32 else ""
        self.nodes.append(
            f'<compNode><uid v="{node_uid}"/><GUILayout><gpos v="{x} {y} 0"/></GUILayout>'
            f'<compOutputs><compOutput><uid v="{out_uid}"/><comptype v="{1 if color else 2}"/></compOutput>'
            f'</compOutputs><compImplementation><compInputBridge><entry v="{uid}"/>'
            f'<parameters>{params}</parameters></compInputBridge></compImplementation></compNode>')
        return node_uid, out_uid

    def bool_param(self, ident, label, default, description, group=None):
        extra = f'<group v="{group}"/>' if group else ""
        self.paraminputs.append(
            f'<paraminput><identifier v="{ident}"/><uid v="{new_uid()}"/><attributes><label v="{label}"/>'
            f'<description v={quoteattr(description)}/></attributes><type v="4"/>'
            f'<defaultValue><constantValueBool v="{int(default)}"/></defaultValue><defaultWidget>'
            f'<name v="buttons"/><options><option><name v="default"/><value v="{int(default)}"/></option>'
            f'</options></defaultWidget>{extra}</paraminput>')

    def float_param(self, ident, label, default, lo, hi, description, step=0.005, group=None, visible_if=None):
        extra = (f'<group v="{group}"/>' if group else "") + \
            (f'<visibleIf v="input[&quot;{visible_if}&quot;]"/>' if visible_if else "")
        self.paraminputs.append(
            f'<paraminput><identifier v="{ident}"/><uid v="{new_uid()}"/><attributes><label v="{label}"/>'
            f'<description v={quoteattr(description)}/></attributes><type v="256"/>'
            f'<defaultValue><constantValueFloat1 v="{default}"/></defaultValue><defaultWidget>'
            f'<name v="slider"/><options><option><name v="clamp"/><value v="1"/></option>'
            f'<option><name v="default"/><value v="{default}"/></option>'
            f'<option><name v="max"/><value v="{hi}"/></option><option><name v="min"/><value v="{lo}"/></option>'
            f'<option><name v="step"/><value v="{step}"/></option></options></defaultWidget>{extra}</paraminput>')

    def pixel_processor(self, name, program, inputs, size_log2=None, color=True, fmt=FMT_32F, x=0, y=0,
                        rel_log2=(0, 0)):
        """inputs: list of (node_uid, out_uid) in slot order. size_log2=None -> relative to
        input 0, scaled by 2^rel_log2 (e.g. (-3, -3) = 1/8 of the input size).
        fmt: output format code, or None to inherit from input 0."""
        node_uid, out_uid = new_uid(), new_uid()
        em = FunctionEmitter()
        root = em.program(program)
        conns = "".join(
            f'<connection><identifier v="{"input" if k == 0 else f"input:{k}"}"/><connRef v="{n}"/>'
            f'<connRefOutput v="{o}"/></connection>' for k, (n, o) in enumerate(inputs))
        params = ""
        if size_log2 is None:
            params += fmt_param("outputsize", "constantValueInt2", f"{rel_log2[0]} {rel_log2[1]}", relative=1)
        else:
            params += fmt_param("outputsize", "constantValueInt2", f"{size_log2[0]} {size_log2[1]}")
        if fmt is not None:
            params += fmt_param("format", "constantValueInt32", fmt)
        params += (f'<parameter><name v="colorswitch"/><relativeTo v="0"/><paramValue>'
                   f'<constantValueBool v="{1 if color else 0}"/></paramValue></parameter>')
        params += (f'<parameter><name v="perpixel"/><relativeTo v="0"/><paramValue>{em.xml(root)}'
                   f'</paramValue></parameter>')
        self.nodes.append(
            f'<compNode><uid v="{node_uid}"/><connections>{conns}</connections>'
            f'<GUILayout><gpos v="{x} {y} 0"/></GUILayout>'
            f'<compOutputs><compOutput><uid v="{out_uid}"/><comptype v="{1 if color else 2}"/></compOutput>'
            f'</compOutputs><compImplementation><compFilter><filter v="pixelprocessor"/>'
            f'<parameters>{params}</parameters></compFilter></compImplementation></compNode>')
        self.comments.append(
            f'<GUIObject><type v="COMMENT"/><GUIDependency v="NODE?{node_uid}"/><GUILayout>'
            f'<gpos v="0 60 -100"/><size v="140 40"/></GUILayout><GUIName v={quoteattr(name)}/>'
            f'<uid v="{new_uid()}"/><frameColor v="0.196 0.196 0.51 0.196"/></GUIObject>')
        return node_uid, out_uid

    def output(self, ident, label, src, x=0, y=0):
        uid, node_uid = new_uid(), new_uid()
        self.outputs.append(
            f'<graphoutput><identifier v="{ident}"/><uid v="{uid}"/><attributes><label v="{label}"/>'
            f'</attributes><channels v="2"/></graphoutput>')
        n, o = src
        self.nodes.append(
            f'<compNode><uid v="{node_uid}"/><connections><connection><identifier v="inputNodeOutput"/>'
            f'<connRef v="{n}"/><connRefOutput v="{o}"/></connection></connections>'
            f'<GUILayout><gpos v="{x} {y} 0"/></GUILayout><compImplementation><compOutputBridge>'
            f'<output v="{uid}"/></compOutputBridge></compImplementation></compNode>')
        self.root_output = uid

    def xml(self, description):
        nodes = "".join(self.nodes)
        return (
            # Graph Type "Filter" (Designer: graph attributes > Type), stored where real 2023 files
            # put <graphtype v="material"/>, so Painter can import it as a filter
            f'<graph><identifier v="{self.ident}"/><uid v="{new_uid()}"/><graphtype v="filter"/>'
            f'<attributes><label v="{self.label}"/><author v={quoteattr(AUTHOR)}/>'
            f'<authorURL v={quoteattr(AUTHOR_URL)}/><tags v="filter;mask;blur;world space"/>'
            f'<description v={quoteattr(description)}/>{self.icon}</attributes>'
            f'<paraminputs>{"".join(self.paraminputs)}</paraminputs><primaryInput v="{self.primary}"/>'
            f'<graphOutputs>{"".join(self.outputs)}</graphOutputs>'
            f'<compNodes>{nodes}</compNodes><baseParameters/>'
            f'<GUIObjects>{"".join(self.comments)}</GUIObjects>'
            f'<options><option><name v="defaultParentSize"/><value v="11x11"/></option></options>'
            f'<root><rootOutputs><rootOutput><output v="{self.root_output}"/><format v="0"/>'
            f'<usertag v=""/></rootOutput></rootOutputs></root></graph>')


def build():
    g = Graph("world_space_mask_blur", "World Space Mask Blur")
    icon = os.path.join(HERE, "icon", "world_space_mask_blur.png")
    if os.path.exists(icon):
        g.set_icon(icon)
    mask = g.image_input("input", "Input", color=False, usage="input", primary=True, x=-1600, y=0)
    position = g.image_input("mesh_position", "Mesh Position", color=True, usage="meshPosition",
                             fmt32=True, x=-1600, y=200)
    uvmask = g.image_input("mesh_uv_mask", "Mesh UV Mask", color=False, usage="meshUVMask", x=-1600, y=400)
    g.float_param("radius", "Radius", RADIUS_DEFAULT, 0.001, 0.25,
                  "Blur radius in UV units (0.01 = 1% of the texture width, about 20 px at 2K), at the "
                  "mesh's average texel density. Doesn't depend on how far apart the pieces are.",
                  step=0.001)
    qmax = max(ref.QUALITY)
    levels = ", ".join(f"{k} = {n}^3" for k, n in sorted(ref.QUALITY.items()))
    g.float_param("dither", "Dither", ref.DITHER, 0, 2,
                  "Fine noise (in 8-bit steps) that hides banding in smooth gradients if the mask is "
                  "stored in 8 bits. 0 = off.", step=0.05)
    g.float_param("quality", "Quality", qmax, 1, qmax,
                  "Voxel budget, shared by all pieces: sets the smallest Radius that still gets smaller "
                  f"({levels} voxels). Lower is faster at small Radius.", step=1)

    # extras, in their own groups, off by default so the plain blur comes first
    g.bool_param("shape_enable", "Enable", False,
                 "Levels after the blur: grow / shrink the mask in 3D and sharpen its edge.", group="Shape")
    g.float_param("grow", "Grow / Shrink", 0.0, -1, 1,
                  "Moves the edge outwards (+) or inwards (-), in 3D, across seams. Works best with some "
                  "Contrast.", step=0.01, group="Shape", visible_if="shape_enable")
    g.float_param("contrast", "Contrast", 0.5, 0, 1,
                  "0 = the soft blur, 1 = a hard edge. In between: rounded, clean edges.",
                  step=0.01, group="Shape", visible_if="shape_enable")
    g.bool_param("smear_enable", "Enable", False,
                 "One-sided blur: the mask is dragged in a direction, e.g. downwards for drips and grime.",
                 group="Smear")
    g.float_param("smear_length", "Length", 1.0, 0, 16,
                  "How much farther the blur reaches in the smear direction, in multiples of Radius.",
                  step=0.05, group="Smear", visible_if="smear_enable")
    g.float_param("smear_strength", "Strength", 1.0, 1, 4,
                  "1 = a soft smear that fades as it gets longer; up to 4 = what is dragged along keeps "
                  "its brightness.",
                  step=0.05, group="Smear", visible_if="smear_enable")
    for axis, default in zip("xyz", (0.0, -1.0, 0.0)):
        g.float_param(f"smear_{axis}", f"Direction {axis.upper()}", default, -1, 1,
                      f"{axis.upper()} component of the smear direction, in the mesh's object axes "
                      "(default: down, -Y).", step=0.05, group="Smear", visible_if="smear_enable")

    def log2(n):
        return n.bit_length() - 1

    def pp(name, prog, inputs, size, x, y, **kw):
        return g.pixel_processor(name, prog, inputs, size, x=x, y=y, **kw)

    def lin(n):
        w, h = lin_dims(n)
        return log2(w), log2(h)

    P, S, R, Dn = ref.PMAX, ref.S_MAX, ref.R_MAX, ref.DMAX
    lp = (log2(P), log2(P))
    atl = (log2(ref.AW), log2(ref.AH))
    one = (4, 4)
    boxt = (log2(2 * R), log2(S))
    bw, bh = lin_dims(S * R * Dn)

    # mesh only: samples, coarse grid, boxes, fine sort
    size = g.pixel_processor("WSB_Size", pass_size(), [position], None, x=-1400, y=600,
                             rel_log2=(SIZE_LOG2, SIZE_LOG2))
    bounds = pp("WSB_Bounds", pass_bounds(), [position, uvmask, size], one, -1250, 500)
    coarse = pp("WSB_Coarse", pass_coarse(), [bounds], one, -1150, 500)
    pack_pos = pp("WSB_PackPos", pass_pack_pos(), [position, uvmask, size], lp, -1250, 300)
    pack_area = pp("WSB_PackArea", pass_pack_area(), [position, uvmask, pack_pos, coarse, size], lp, -1100, 400)
    area_blocks = pp("WSB_AreaBlocks", pass_area_blocks(), [pack_area], (log2(ref.AREA_GRID),) * 2, -1000, 450)
    area_total = pp("WSB_AreaTotal", pass_area_total(), [area_blocks], one, -900, 450)
    keys = pp("WSB_Keys0", pass_keys0(), [pack_pos, coarse], lp, -1100, 850)
    for n, (k, j) in enumerate(ref.sort_steps()):
        keys = pp(f"WSB_Sort0 {k}/{j}", pass_sort_step(k, j), [keys], lp, -950 + 30 * (n % 40), 850 + 120 * (n // 40))
    cells0 = pp("WSB_Cells0", pass_cells(ref.CW, ref.CH), [keys], (log2(ref.CW), log2(ref.CH)), -650, 1600)
    cellb = pp("WSB_CellBounds", pass_cell_bounds(), [keys, cells0, pack_pos],
               (log2(2 * ref.CW), log2(ref.CH)), -500, 1600)
    colxz = pp("WSB_ColXZ", pass_columns(True), [cellb, coarse], (log2(2 * Dn), log2(Dn)), -350, 1550)
    colxy = pp("WSB_ColXY", pass_columns(False), [cellb, coarse], (log2(2 * Dn), log2(Dn)), -350, 1650)
    occx = pp("WSB_OccX", pass_occ_x(), [colxz, coarse], lin(Dn), -200, 1500)
    slabx = pp("WSB_SlabX", pass_slab_x(), [occx], lin(Dn), -100, 1500)
    slabr = pp("WSB_SlabRange", pass_slab_range(), [slabx, coarse], lin(S), 0, 1500)
    occzy = pp("WSB_OccZY", pass_occ_zy(), [colxz, colxy, slabr, coarse], (log2(Dn), log2(S)), 100, 1600)
    runtab = pp("WSB_RunTab", pass_run_tab(), [occzy], (log2(Dn), log2(S)), 200, 1600)
    runr = pp("WSB_RunRange", pass_run_range(), [runtab], (log2(R), log2(S)), 300, 1600)
    boxrow = pp("WSB_BoxRow", pass_box_row(), [colxz, colxy, slabr, runr], (log2(2 * bw), log2(bh)), 400, 1600)
    boxes = pp("WSB_Boxes", pass_boxes(), [boxrow, slabr], boxt, 500, 1600)
    sortfit = pp("WSB_SortFit", pass_fit(lambda: const(float(ref.AW * ref.AH))), [boxes, bounds], one, 600, 1500)
    sortboxes = pp("WSB_SortBoxes", pass_grid_table(lambda: frame_a(1).x), [boxes, sortfit], boxt, 700, 1500)
    keys = pp("WSB_Keys1", pass_keys1(), [pack_pos, coarse, slabx, runtab, sortboxes, sortfit], lp, 800, 1500)
    for n, (k, j) in enumerate(ref.sort_steps()):
        keys = pp(f"WSB_Sort1 {k}/{j}", pass_sort_step(k, j), [keys], lp, 900 + 30 * (n % 40), 1500 + 120 * (n // 40))
    cells1 = pp("WSB_Cells1", pass_cells(ref.AW, ref.AH), [keys], atl, -650, 600)

    # per stroke (Radius / Quality / mask)
    voxfit = pp("WSB_VoxFit", pass_fit(q_budget), [boxes, bounds], one, -1000, 700)
    frame = pp("WSB_Frame", pass_frame(), [area_total, voxfit, sortfit, size], one, -900, 650)
    voxboxes = pp("WSB_VoxBoxes", pass_grid_table(lambda: frame_a(1).x), [boxes, frame], boxt, -800, 650)
    voxmap = pp("WSB_VoxMap", pass_vox_map(), [voxboxes], atl, -700, 650)
    pack_mask = pp("WSB_PackMask", pass_pack_mask(), [mask, position, uvmask, pack_pos, coarse, pack_area, size, area_total],
                   lp, -950, 150)
    grid = pp("WSB_Splat", pass_splat(),
              [keys, cells1, pack_pos, pack_mask, frame, voxmap, voxboxes, sortboxes, sortfit], atl, -500, 300)
    for axis, name in enumerate("XYZ"):
        grid = pp(f"WSB_Blur{name}", pass_blur(axis), [grid, frame, voxmap, voxboxes], atl, -380 + 120 * axis, 300)
    resolve = g.pixel_processor("WSB_Resolve", pass_resolve(),
                                [mask, position, uvmask, grid, frame, coarse, slabx, runtab, voxboxes],
                                None, color=False, fmt=FMT_16, x=0, y=0)
    g.output("output", "Output", resolve, x=200, y=0)

    desc = (f"v{VERSION} - Blurs a mask in 3D using Painter's mesh position, so the blur is continuous across UV "
            "seams. Pieces laid out apart get their own voxel grids, so spreading them out doesn't coarsen the "
            "blur, and they don't blur into each other. Radius is in UV units.")
    return ('<?xml version="1.0" encoding="UTF-8"?><package><identifier v="Unsaved Package"/>'
            '<formatVersion v="1.1.0.202302"/><updaterVersion v="1.1.0.202302"/>'
            '<fileUID v="{7d3c1f0e-5a8b-4c61-9f2e-3b7a1d9c4e52}"/><versionUID v="0"/>'
            # Designer requires a <dependencies> member (it reports "invalid member package"
            # without one); a package using only atomic nodes just depends on itself.
            f'<dependencies><dependency><filename v="?himself"/><uid v="{new_uid()}"/><type v="package"/>'
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
