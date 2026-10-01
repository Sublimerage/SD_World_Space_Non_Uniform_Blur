"""Tiny DSL for Substance Designer function graphs (pixel processor bodies)
and an .sbs XML writer.

Expressions are built with Python operators; statements (set / while) are
collected by a `Program`.  The emitted XML mirrors the conventions used by
world_space_mask_blur.sbs:

  * a statement list is right-nested `sequence(seqin=stmt, seqlast=rest)`;
  * `while` takes `init` (evaluated once), `cond` (the loop stops as soon as
    it is true) and `loop` (the body, whose value is the loop variable);
  * vectors are built with vector2(f,f) / vector3(f2,f) / vector4(f3,f) and
    scalars are broadcast explicitly (no implicit scalar*vector).

Every statement gets its own expression nodes; nodes are only shared inside
one statement, so a `get` never observes a later `set` of the same variable.
"""
from xml.sax.saxutils import quoteattr

T_BOOL, T_INT, T_F1, T_F2, T_F3, T_F4 = 4, 16, 256, 512, 1024, 2048
FTYPES = {1: T_F1, 2: T_F2, 3: T_F3, 4: T_F4}
DIM = {T_F1: 1, T_F2: 2, T_F3: 3, T_F4: 4, T_BOOL: 1, T_INT: 1}


def fmt(x):
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    r = repr(x)
    if 'e' in r:
        # plain decimals only (like the original files); refuse to round a value to 0
        r = ('%.12f' % x).rstrip('0').rstrip('.')
        if float(r) == 0.0:
            raise ValueError('constant %r too small for 12 decimals' % x)
    return r


class E:
    """Expression node."""
    __slots__ = ('fn', 'typ', 'ins', 'data', '_key')

    def __init__(self, fn, typ, ins=None, data=None):
        self.fn, self.typ, self.ins, self.data = fn, typ, dict(ins or {}), data
        self._key = None

    @property
    def dim(self):
        return DIM[self.typ]

    def key(self):
        if self._key is None:
            self._key = (self.fn, self.typ, self.data if not isinstance(self.data, list) else tuple(self.data),
                         tuple(sorted((k, v.key()) for k, v in self.ins.items())))
        return self._key

    # arithmetic -----------------------------------------------------------
    def _bin(self, fn, other, rev=False):
        a, b = (lift(other), self) if rev else (self, lift(other))
        a, b = bcast(a, b)
        return E(fn, a.typ, {'a': a, 'b': b})

    def __add__(self, o): return self._bin('add', o)
    def __radd__(self, o): return self._bin('add', o, True)
    def __sub__(self, o): return self._bin('sub', o)
    def __rsub__(self, o): return self._bin('sub', o, True)
    def __mul__(self, o): return self._bin('mul', o)
    def __rmul__(self, o): return self._bin('mul', o, True)
    def __truediv__(self, o): return self._bin('div', o)
    def __rtruediv__(self, o): return self._bin('div', o, True)
    def __neg__(self): return lift(0.0 if self.dim == 1 else [0.0] * self.dim) - self

    def _cmp(self, fn, o):
        a, b = bcast(self, lift(o))
        return E(fn, T_BOOL, {'a': a, 'b': b})

    def __lt__(self, o): return self._cmp('lr', o)
    def __le__(self, o): return self._cmp('lreq', o)
    def __gt__(self, o): return self._cmp('gt', o)
    def __ge__(self, o): return self._cmp('gteq', o)

    def __and__(self, o): return E('and', T_BOOL, {'a': self, 'b': lift(o)})
    def __or__(self, o): return E('or', T_BOOL, {'a': self, 'b': lift(o)})
    def __invert__(self): return E('not', T_BOOL, {'a': self})

    def __getattr__(self, name):
        if name and all(c in 'xyzw' for c in name) and len(name) <= 4:
            return swz(self, ['xyzw'.index(c) for c in name])
        if name and all(c in 'rgba' for c in name) and len(name) <= 4:
            return swz(self, ['rgba'.index(c) for c in name])
        raise AttributeError(name)

    def __getitem__(self, i):
        return swz(self, [i])

    def __hash__(self):
        return id(self)


def lift(v):
    if isinstance(v, E):
        return v
    if isinstance(v, bool):
        raise TypeError('use bool constants explicitly')
    if isinstance(v, (int, float)):
        return E('const_float1', T_F1, data=[float(v)])
    v = [float(x) for x in v]
    return E('const_float%d' % len(v), FTYPES[len(v)], data=v)


def bcast(a, b):
    if a.typ == b.typ:
        return a, b
    if a.dim == 1 and a.typ == T_F1 and b.typ in (T_F2, T_F3, T_F4):
        return splat(a, b.dim), b
    if b.dim == 1 and b.typ == T_F1 and a.typ in (T_F2, T_F3, T_F4):
        return a, splat(b, a.dim)
    raise TypeError('type mismatch %s %s (%s, %s)' % (a.typ, b.typ, a.fn, b.fn))


def splat(s, n):
    if s.fn == 'const_float1':
        return lift([s.data[0]] * n)
    return vec(*([s] * n))


def vec(*cs):
    cs = [lift(c) for c in cs]
    # flatten to a left-nested chain: vector2(f,f) -> vector3(f2,f) -> vector4(f3,f)
    if all(c.fn.startswith('const_float') for c in cs):
        flat = []
        for c in cs:
            flat += c.data
        return lift(flat)
    out = cs[0]
    for c in cs[1:]:
        if c.dim != 1:
            raise TypeError('vec: only the first component may be a vector')
        n = out.dim + 1
        out = E('vector%d' % n, FTYPES[n], {'componentsin': out, 'componentslast': c})
    return out


def swz(e, idx):
    n = len(idx)
    if e.dim == n and idx == list(range(n)):
        return e
    if e.fn.startswith('const_float'):
        return lift([e.data[i] for i in idx] if n > 1 else e.data[idx[0]])
    return E('swizzle%d' % n, FTYPES[n], {'vector': e}, data=list(idx))


def f1(fn, a, typ=None):
    a = lift(a)
    return E(fn, typ or a.typ, {'a': a})


def floor(a): return f1('floor', a)
def ceil(a): return f1('ceil', a)
def sqrt(a): return f1('sqrt', a)
def exp(a): return f1('exp', a)
def sin(a): return f1('sin', a)
def cos(a): return f1('cos', a)
def absv(a): return f1('abs', a)


def fmin(a, b):
    a, b = bcast(lift(a), lift(b))
    return E('min', a.typ, {'a': a, 'b': b})


def fmax(a, b):
    a, b = bcast(lift(a), lift(b))
    return E('max', a.typ, {'a': a, 'b': b})


def clamp(a, lo, hi): return fmin(fmax(a, lo), hi)


def dot(a, b):
    a, b = bcast(lift(a), lift(b))
    return E('dot', T_F1, {'a': a, 'b': b})


def lerp(a, b, x):
    a, b = bcast(lift(a), lift(b))
    return E('lerp', a.typ, {'a': a, 'b': b, 'x': lift(x)})


def cross(a, b):
    return a.yzx * b.zxy - a.zxy * b.yzx


def length(a): return sqrt(dot(a, a))


def ifelse(c, a, b):
    a, b = lift(a), lift(b)
    if a.typ != b.typ:
        a, b = bcast(a, b)
    return E('ifelse', a.typ, {'condition': c, 'ifpath': a, 'elsepath': b})


def sel(c, a, b): return ifelse(c, a, b)


def tofloat(a): return E('tofloat', T_F1, {'value': a})


def get(name, typ=T_F1):
    fn = {T_F1: 'get_float1', T_F2: 'get_float2', T_F3: 'get_float3', T_F4: 'get_float4',
          T_BOOL: 'get_bool', T_INT: 'get_integer1'}[typ]
    return E(fn, typ, data=name)


def pf(name): return get(name, T_F1)            # float graph parameter
def pi(name): return tofloat(get(name, T_INT))  # integer graph parameter as float
def pb(name): return get(name, T_BOOL)          # bool graph parameter


POS = E('get_float2', T_F2, data='$pos')
SIZE = E('get_float2', T_F2, data='$size')


def samplecol(idx, pos):
    return E('samplecol', T_F4, {'pos': lift(pos)}, data=[idx, 0])


def samplelum(idx, pos):
    return E('samplelum', T_F1, {'pos': lift(pos)}, data=[idx, 0])


def fmod(a, b):
    return a - b * floor(a / b)


# statements ----------------------------------------------------------------
class Set:
    def __init__(self, name, expr):
        self.name, self.expr = name, lift(expr)


class While:
    def __init__(self, var, init, cond, body, maxiter):
        self.var, self.init, self.cond, self.body, self.maxiter = var, lift(init), cond, body, maxiter


class Program:
    """Collects statements.  Variables are typed by their first assignment."""

    def __init__(self):
        self.stmts = []
        self.vtypes = {}
        self._stack = [self.stmts]

    def set(self, name, expr):
        expr = lift(expr)
        t = self.vtypes.setdefault(name, expr.typ)
        if t != expr.typ:
            raise TypeError('variable %s retyped %s -> %s' % (name, t, expr.typ))
        self._stack[-1].append(Set(name, expr))
        return get(name, t)

    def v(self, name):
        return get(name, self.vtypes[name])

    def loop(self, var, init, maxiter):
        """with p.loop('i', 0, 64) as L: L.until(cond) ; body ..."""
        return _LoopCtx(self, var, init, maxiter)


class _LoopCtx:
    def __init__(self, p, var, init, maxiter):
        self.p, self.var, self.init, self.maxiter = p, var, lift(init), maxiter
        self.cond = None
        self.body = []

    def until(self, cond):
        self.cond = cond

    def __enter__(self):
        self.p.vtypes.setdefault(self.var, self.init.typ)
        self.p._stack.append(self.body)
        return self

    def __exit__(self, *a):
        self.p._stack.pop()
        if self.cond is None:
            raise ValueError('loop without until()')
        self.p._stack[-1].append(While(self.var, self.init, self.cond, self.body, self.maxiter))


# emission ------------------------------------------------------------------
class UID:
    def __init__(self, start):
        self.n = start

    def __call__(self):
        self.n += 1
        return self.n


class FnEmitter:
    """Turns a Program + return expression into <paramNode> XML."""

    def __init__(self, uid, vtypes):
        self.uid = uid
        self.vtypes = vtypes
        self.nodes = []      # xml strings
        self.count = 0

    def node(self, fn, typ, conns, data):
        u = self.uid()
        x = ['<paramNode><uid v="%d"/><function v="%s"/><type v="%d"/>' % (u, fn, typ)]
        if conns:
            x.append('<connections>')
            for k, v in conns:
                x.append('<connection><identifier v="%s"/><connRef v="%d"/></connection>' % (k, v))
            x.append('</connections>')
        if data is not None:
            x.append('<funcDatas><funcData><name v="%s"/><constantValue>%s</constantValue></funcData></funcDatas>'
                     % (fn, data))
        x.append('<GUILayout><gpos v="%d %d 0"/></GUILayout></paramNode>' % (-200 * (self.count % 40), 70 * (self.count // 40)))
        self.count += 1
        self.nodes.append(''.join(x))
        return u

    def expr(self, e, memo):
        k = e.key()
        if k in memo:
            return memo[k]
        conns = [(name, self.expr(sub, memo)) for name, sub in e.ins.items()]
        data = None
        fn = e.fn
        if fn.startswith('const_float'):
            n = int(fn[-1])
            data = '<constantValueFloat%d v="%s"/>' % (n, ' '.join(fmt(x) for x in e.data))
        elif fn == 'const_int1':
            data = '<constantValueInt1 v="%d"/>' % e.data
        elif fn.startswith('get_'):
            data = '<constantValueString v=%s/>' % quoteattr(e.data)
        elif fn.startswith('swizzle'):
            n = int(fn[-1])
            data = '<constantValueInt%d v="%s"/>' % (n, ' '.join(str(i) for i in e.data)) if n > 1 else \
                '<constantValueInt1 v="%d"/>' % e.data[0]
        elif fn in ('samplecol', 'samplelum'):
            data = '<constantValueInt2 v="%d %d"/>' % tuple(e.data)
        u = self.node(fn, e.typ, conns, data)
        memo[k] = u
        return u

    def stmt(self, s):
        memo = {}
        if isinstance(s, Set):
            v = self.expr(s.expr, memo)
            return self.node('set', s.expr.typ, [('value', v)],
                             '<constantValueString v=%s/>' % quoteattr(s.name)), s.expr.typ
        # While
        t = self.vtypes[s.var]
        init = self.node('set', t, [('value', self.expr(s.init, memo))],
                         '<constantValueString v=%s/>' % quoteattr(s.var))
        cond = self.expr(s.cond, {})
        body = self.seq(s.body, get(s.var, t))
        w = self.node('while', t, [('init', init), ('cond', cond), ('loop', body[0])],
                      '<constantValueInt1 v="%d"/>' % s.maxiter)
        return w, t

    def seq(self, stmts, ret):
        """right-nested sequence ending in expression `ret`; returns (uid, type)."""
        r = self.expr(ret, {})
        rt = ret.typ
        for s in reversed(stmts):
            su, _ = self.stmt(s)
            r = self.node('sequence', rt, [('seqin', su), ('seqlast', r)], None)
        return r, rt

    def program(self, prog, ret):
        ret = lift(ret)
        # final "+0" mirrors the original files (root is an add node)
        zero = lift(0.0 if ret.dim == 1 else [0.0] * ret.dim)
        body, rt = self.seq(prog.stmts, ret)
        z = self.expr(zero, {})
        root = self.node('add', rt, [('a', body), ('b', z)], None)
        return root


def dynamic_value(uid, prog, ret):
    em = FnEmitter(uid, prog.vtypes)
    root = em.program(prog, ret)
    return '<dynamicValue><rootnode v="%d"/><paramNodes>%s</paramNodes></dynamicValue>' % (root, ''.join(em.nodes)), em.count


def dynamic_expr(uid, e):
    p = Program()
    em = FnEmitter(uid, p.vtypes)
    e = lift(e)
    root = em.expr(e, {})
    return '<dynamicValue><rootnode v="%d"/><paramNodes>%s</paramNodes></dynamicValue>' % (root, ''.join(em.nodes))
