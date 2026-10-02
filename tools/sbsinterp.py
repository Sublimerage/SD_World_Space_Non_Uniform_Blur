"""Minimal numpy interpreter for .sbs graphs made of pixel processors.

Used to test generated packages without Substance Designer.  Supports the
function nodes used by the world space blur graphs (arithmetic, comparisons,
vectors, swizzles, samplers, variables, sequence, while).

Sampling is nearest-texel with wrapping; every sampler in the graphs reads
texel centres, so this matches the GPU result.  `while` stops as soon as
`cond` is true (checked before each iteration).
"""
import numpy as np
import xml.etree.ElementTree as ET

np.seterr(all='ignore')


class FnGraph:
    def __init__(self, dv):
        self.nodes = {}
        for pn in dv.findall('paramNodes/paramNode'):
            uid = pn.find('uid').get('v')
            fn = pn.find('function').get('v')
            conns = {c.find('identifier').get('v'): c.find('connRef').get('v')
                     for c in pn.findall('connections/connection')}
            cv = pn.find('funcDatas/funcData/constantValue')
            data = None
            if cv is not None:
                c = cv[0]
                v = c.get('v')
                if c.tag.startswith('constantValueFloat'):
                    data = np.array([float(x) for x in v.split()], np.float32)
                elif c.tag.startswith('constantValueInt'):
                    data = [int(x) for x in v.split()]
                elif c.tag == 'constantValueString':
                    data = v
                elif c.tag == 'constantValueBool':
                    data = v == '1'
            self.nodes[uid] = (fn, conns, data)
        self.root = dv.find('rootnode').get('v')


class Ctx:
    def __init__(self, n, pos, size, inputs, params):
        self.n, self.pos, self.size, self.inputs, self.params = n, pos, size, inputs, params
        self.vars = {}
        self.cache = {}


def _bc(a, n):
    a = np.asarray(a)
    if a.ndim == 0 or (a.ndim == 1 and a.shape[0] != n):
        return np.broadcast_to(a, (n,) + a.shape).copy() if a.ndim else np.full(n, a)
    return a


def sample(img, pos):
    H, W = img.shape[:2]
    x = np.floor(pos[:, 0] * W).astype(np.int64) % W
    y = np.floor(pos[:, 1] * H).astype(np.int64) % H
    v = img[y, x]
    if img.ndim == 2:
        v = np.stack([v, v, v, np.ones_like(v)], -1)
    return v.astype(np.float32)


class Evaluator:
    def __init__(self, g):
        self.g = g

    def ev(self, uid, c):
        if uid in c.cache:
            return c.cache[uid]
        fn, ins, data = self.g.nodes[uid]
        r = self.op(fn, ins, data, c)
        if fn not in ('set', 'sequence', 'while'):
            c.cache[uid] = r
        return r

    def op(self, fn, ins, d, c):
        A = lambda k='a': self.ev(ins[k], c)
        n = c.n
        if fn.startswith('const_float'):
            return np.broadcast_to(d if len(d) > 1 else d[0], (n,) + ((len(d),) if len(d) > 1 else ()))
        if fn == 'const_int1':
            return np.full(n, d[0], np.float32)
        if fn == 'const_bool':
            return np.full(n, bool(d))
        if fn.startswith('get_'):
            if d == '$pos':
                return c.pos
            if d == '$size':
                return np.broadcast_to(c.size, (n, 2))
            if d in c.vars:
                return c.vars[d]
            if d in c.params:
                v = c.params[d]
                if isinstance(v, bool):
                    return np.full(n, v)
                v = np.asarray(v, np.float32)
                return np.broadcast_to(v, (n,) + v.shape)
            raise KeyError('unknown variable %s' % d)
        if fn in ('add', 'sub', 'mul', 'div', 'min', 'max'):
            a, b = A('a'), A('b')
            return {'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': np.divide,
                    'min': np.minimum, 'max': np.maximum}[fn](a, b).astype(np.float32)
        if fn in ('lr', 'lreq', 'gt', 'gteq', 'eq', 'noteq'):
            a, b = A('a'), A('b')
            return {'lr': np.less, 'lreq': np.less_equal, 'gt': np.greater, 'gteq': np.greater_equal,
                    'eq': np.equal, 'noteq': np.not_equal}[fn](a, b)
        if fn == 'and':
            return np.logical_and(A('a'), A('b'))
        if fn == 'or':
            return np.logical_or(A('a'), A('b'))
        if fn == 'not':
            return np.logical_not(A('a'))
        if fn in ('floor', 'ceil', 'sqrt', 'exp', 'sin', 'cos', 'abs', 'neg'):
            f = {'floor': np.floor, 'ceil': np.ceil, 'sqrt': np.sqrt, 'exp': np.exp, 'sin': np.sin,
                 'cos': np.cos, 'abs': np.abs, 'neg': np.negative}[fn]
            return f(A('a')).astype(np.float32)
        if fn == 'dot':
            a, b = A('a'), A('b')
            return np.sum(a * b, -1).astype(np.float32) if np.ndim(a) > 1 else (a * b).astype(np.float32)
        if fn == 'lerp':
            a, b, x = A('a'), A('b'), A('x')
            if np.ndim(a) > 1 and np.ndim(x) == 1:
                x = x[:, None]
            return (a + (b - a) * x).astype(np.float32)
        if fn == 'tofloat':
            return np.asarray(self.ev(ins['value'], c), np.float32)
        if fn.startswith('swizzle'):
            v = self.ev(ins['vector'], c)
            idx = d if d is not None else [0]
            if v.ndim == 1:
                v = v[:, None]
            r = v[:, idx]
            return r[:, 0] if len(idx) == 1 else r
        if fn.startswith('vector'):
            a = self.ev(ins['componentsin'], c)
            b = self.ev(ins['componentslast'], c)
            a = a[:, None] if np.ndim(a) == 1 else a
            return np.concatenate([a, np.asarray(b)[:, None]], 1).astype(np.float32)
        if fn == 'ifelse':
            cnd = self.ev(ins['condition'], c)
            a, b = self.ev(ins['ifpath'], c), self.ev(ins['elsepath'], c)
            if np.ndim(a) > 1:
                cnd = cnd[:, None]
            return np.where(cnd, a, b)
        if fn in ('samplecol', 'samplelum'):
            p = self.ev(ins['pos'], c)
            img = c.inputs[d[0]]
            if img is None:
                v = np.zeros((n, 4), np.float32)
            else:
                v = sample(img, p)
            if fn == 'samplelum':
                return v[:, 0]
            return v
        if fn == 'set':
            v = self.ev(ins['value'], c)
            c.vars[d] = np.array(v)
            c.cache = {}
            return v
        if fn == 'sequence':
            self.ev(ins['seqin'], c)
            return self.ev(ins['seqlast'], c)
        if fn == 'while':
            return self.do_while(ins, d, c)
        raise NotImplementedError(fn)

    def do_while(self, ins, d, c):
        self.ev(ins['init'], c)
        c.cache = {}
        maxit = d[0]
        active = np.ones(c.n, bool)
        idx = np.arange(c.n)
        res = None
        for it in range(maxit):
            sub = self._sub(c, idx)
            cond = np.asarray(self.ev(ins['cond'], sub))
            cond = np.broadcast_to(cond, (len(idx),))
            keep = ~cond
            idx = idx[keep]
            if len(idx) == 0:
                break
            sub = self._sub(c, idx)
            r = self.ev(ins['loop'], sub)
            self._writeback(c, sub, idx)
            c.cache = {}
        return self.ev(ins['loop'], c) if False else None

    def _sub(self, c, idx):
        s = Ctx(len(idx), c.pos[idx], c.size, c.inputs, c.params)
        s.vars = {k: v[idx] for k, v in c.vars.items()}
        return s

    def _writeback(self, c, s, idx):
        for k, v in s.vars.items():
            if k not in c.vars:
                shp = (c.n,) + np.shape(v)[1:]
                c.vars[k] = np.zeros(shp, np.asarray(v).dtype)
            c.vars[k][idx] = v


FORMATS = {0: 8, 1: 16, 2: None, 3: None}


def eval_pp(dv, out_w, out_h, inputs, params, color=True, chunk=1 << 16):
    """Pixels are independent, so the image is evaluated in chunks to bound memory."""
    g = FnGraph(dv)
    ev = Evaluator(g)
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    pos_all = np.stack([(xs.ravel() + 0.5) / out_w, (ys.ravel() + 0.5) / out_h], -1).astype(np.float32)
    parts = []
    for s in range(0, out_w * out_h, chunk):
        pos = pos_all[s:s + chunk]
        c = Ctx(len(pos), pos, np.array([out_w, out_h], np.float32), inputs, params)
        r = _bc(np.asarray(ev.ev(g.root, c), np.float32), c.n)
        if color and r.ndim == 1:
            r = np.stack([r, r, r, np.ones_like(r)], -1)
        if not color and r.ndim > 1:
            r = r[:, 0]
        parts.append(r)
    r = np.concatenate(parts, 0)
    return r.reshape((out_h, out_w) + r.shape[1:])


def param_defaults(graph):
    out = {}
    for p in graph.findall('paraminputs/paraminput'):
        t = int(p.find('type').get('v'))
        if t in (1, 2):
            continue
        dv = p.find('defaultValue')[0]
        v = dv.get('v')
        if t == 4:
            out[p.find('identifier').get('v')] = v == '1'
        else:
            vals = [float(x) for x in v.split()]
            out[p.find('identifier').get('v')] = vals[0] if len(vals) == 1 else vals
    return out


def run_graph(path, graph_id, images, params=None, targets=None, keep=None, verbose=False):
    """Evaluate graph `graph_id` of package `path`.

    images: dict input identifier -> numpy image (H,W) or (H,W,4)
    returns dict of outputs (by output identifier) and intermediate results for uids in `keep`.
    """
    import time
    root = ET.parse(path).getroot()
    graph = [g for g in root.iter('graph') if g.find('identifier').get('v') == graph_id][0]
    prm = param_defaults(graph)
    prm.update(params or {})
    entries = {p.find('uid').get('v'): p.find('identifier').get('v') for p in graph.findall('paraminputs/paraminput')}
    outs_by_uid = {o.find('uid').get('v'): o.find('identifier').get('v') for o in graph.findall('graphOutputs/graphoutput')}
    nodes = {n.find('uid').get('v'): n for n in graph.find('compNodes')}
    res = {}
    results = {}

    def conns(n):
        return {c.find('identifier').get('v'): c.find('connRef').get('v') for c in n.findall('connections/connection')}

    def evaln(uid):
        if uid in res:
            return res[uid]
        n = nodes[uid]
        impl = n.find('compImplementation')[0]
        if impl.tag == 'compInputBridge':
            ident = entries[impl.find('entry').get('v')]
            r = images.get(ident)
            res[uid] = r
            return r
        if impl.tag == 'compOutputBridge':
            r = evaln(conns(n)['inputNodeOutput'])
            results[outs_by_uid[impl.find('output').get('v')]] = r
            res[uid] = r
            return r
        if impl.tag == 'compInstance':
            sub_id = impl.find('path').get('v').split('pkg:///')[1].split('?')[0]
            sub_imgs = {k: evaln(v) for k, v in conns(n).items()}
            sub_prm = {}
            for p in impl.findall('parameters/parameter'):
                v = eval_pp(p.find('paramValue/dynamicValue'), 1, 1, [], prm, color=False)
                v = v.ravel()[0]
                if p.find('.//function').get('v') == 'get_bool':
                    v = bool(v)
                sub_prm[p.find('name').get('v')] = float(v) if not isinstance(v, bool) else v
            sub = run_graph(path, sub_id, sub_imgs, sub_prm, verbose=verbose)
            ob = impl.find('outputBridgings/outputBridging/identifier').get('v')
            res[uid] = sub[ob]
            return res[uid]
        if impl.tag != 'compFilter' or impl.find('filter').get('v') != 'pixelprocessor':
            raise NotImplementedError(impl.tag)
        cs = conns(n)
        ins = []
        for k in range(16):
            key = 'input' if k == 0 else 'input:%d' % k
            ins.append(evaln(cs[key]) if key in cs else None)
        P = {p.find('name').get('v'): p for p in impl.findall('parameters/parameter')}
        osz = P['outputsize']
        rel = int(osz.find('relativeTo').get('v'))
        sv = [int(x) for x in osz.find('paramValue')[0].get('v').split()]
        if rel == 0:
            W, H = 2 ** sv[0], 2 ** sv[1]
        else:
            ref = ins[0]
            W, H = int(ref.shape[1] * 2.0 ** sv[0]), int(ref.shape[0] * 2.0 ** sv[1])
        color = P['colorswitch'].find('paramValue')[0].get('v') == '1'
        fmtv = int(P['format'].find('paramValue')[0].get('v'))
        t0 = time.time()
        img = eval_pp(P['perpixel'].find('paramValue/dynamicValue'), W, H, ins, prm, color)
        if fmtv in (0, 1):
            q = 255.0 if fmtv == 0 else 65535.0
            img = np.round(np.clip(img, 0, 1) * q) / q
        elif fmtv == 2:
            img = img.astype(np.float16).astype(np.float32)
        if verbose:
            print('node', uid, W, H, '%.2fs' % (time.time() - t0))
        res[uid] = img.astype(np.float32)
        return res[uid]

    if targets:
        for uid in targets:
            evaln(uid)
    else:
        for uid, n in nodes.items():
            if n.find('compImplementation')[0].tag == 'compOutputBridge':
                evaln(uid)
    if keep:
        results['_nodes'] = {k: res.get(k) for k in keep}
    results['_all'] = res
    return results
