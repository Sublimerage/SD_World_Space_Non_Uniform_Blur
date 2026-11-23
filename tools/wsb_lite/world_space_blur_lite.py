"""
World Space Mask Blur Lite: a cheaper companion to world_space_blur.py that
can also go to a much smaller Radius. The spec for designer/build_lite.py,
checked against it by designer/test_lite.py.

Idea: a 3D Gaussian is a 2D Gaussian in a plane times a 1D Gaussian across it.
Each piece (v2.0's boxes) is seen from the 6 axis directions; each surface
sample belongs to the one view that faces it most squarely (its dominant
normal axis). Each view is a 2D image of the surface it owns (front-most
point per pixel: depth, the UV there, and its area per projected area).

Per stroke each view is filled with the mask at those UVs and blurred in 2D
with a depth-aware ("bilateral") Gaussian: a neighbour counts by
exp(-|d_uv|^2 / 2 s^2) exp(-(z_q - z_c)^2 / 2 s^2), which is the 3D Gaussian
for a surface that is a height field over the view. Every texel then adds each
view's result at its projection, times exp(-(z_view - z_texel)^2 / 2 s^2), so a
view contributes only where its surface is near the texel in 3D. On flat faces
(and across their edges and corners) this is the 3D blur; on curved surfaces
it is close (the 2-pass depth-aware blur drops a cross term on diagonal
slopes).

The views are 2D, so they can be much finer than v2.0's voxels for the same
cost: the Radius can go much lower. For bigger blurs, coarser levels
(2x, 4x, ... pixels) keep the cost flat.

Known limits:
  * Surface hidden from its own view direction by another part of the same
    piece (deep pockets) is missing from the blur.
  * Curved surfaces at a large Radius: up to a few % different from v2.0.
  * No Smear.

How a view pixel finds its surface point: every surface sample stands for a
small UV patch; its tangent plane (from the position map's derivatives) maps
the patch to 3D. The pixel's ray hits the patch if it meets the plane inside
the (slightly grown) patch square, the UV there is inside an island and the
position map there agrees with the plane. The front-most hit wins (ties: the
lower sample index). The UV of the hit is kept, so the fill reads the mask
texel itself and detail down to a texel survives.
"""

import numpy as np

import world_space_blur as w

VW, VH = 2048, 2048          # view tables (linear): fine pixels, all boxes and views
MAX_LEVEL = 7                # coarsest per-stroke view: 2^7 x the fine pixel
PIXELS_PER_SIGMA = 2.5       # per stroke: the level where sigma / pixel <= this
MIN_SIGMA_PIXELS = 0.8       # smallest sigma, in fine view pixels
PATCH_GROW = 0.05            # patches overlap by this fraction of their size: no cracks
MAX_REACH_CELLS = 2          # the lookup's columns: a patch reaches at most this many columns sideways
DEPTH_FILL = 3               # empty pixels this close to a surface get its depth
AXES = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]   # view axis, image u axis, image v axis
NVIEWS = 6                   # per box: axis 0 +, axis 0 -, axis 1 +, ...


def configure(vw, vh, max_level=None):
    global VW, VH, MAX_LEVEL
    VW, VH = vw, vh
    MAX_LEVEL = MAX_LEVEL if max_level is None else max_level


# --------------------------------------------------------------------------
# WSB_LFrame (half table 2 PMAX x PMAX, mesh only): per sample, the texel
# centre of its rep read and dP/du, dP/dv per UV unit there.
#   left (Pu.xyz, rep u), right (Pv.xyz, rep v); unusable: rep u = -1
# --------------------------------------------------------------------------

def pass_frame_samples(position, uv_mask, pack_pos, coarse):
    cu, cv, su, sv = w._patches()
    valid = w.sample_nearest(uv_mask, su, sv) > w.UV_INSIDE
    d = (su - cu[..., None]) ** 2 + (sv - cv[..., None]) ** 2
    d = np.where(valid, d, np.inf)
    best = d.argmin(axis=-1)
    ru = np.take_along_axis(su, best[..., None], -1)[..., 0]
    rv = np.take_along_axis(sv, best[..., None], -1)[..., 0]
    hgt, wid = uv_mask.shape
    ru = (np.floor(ru * wid) + 0.5) / wid
    rv = (np.floor(rv * hgt) + 0.5) / hgt
    p = w.sample_nearest(position, ru, rv)
    r2 = w._r2(coarse)

    def inside(x, y):
        return (np.minimum(x, y) > 0.0) & (np.maximum(x, y) < 1.0)

    def diff(du, dv):
        # as w.texel_diffs (v2.1): forward, else backward within r2, else the shorter in-island
        # neighbour (a stretched island), else none
        fwd = w.sample_nearest(position, ru + du, rv + dv) - p
        bwd = p - w.sample_nearest(position, ru - du, rv - dv)
        fin = (w.sample_nearest(uv_mask, ru + du, rv + dv) > w.UV_INSIDE) & inside(ru + du, rv + dv)
        bin_ = (w.sample_nearest(uv_mask, ru - du, rv - dv) > w.UV_INSIDE) & inside(ru - du, rv - dv)
        f2, b2 = np.sum(fwd * fwd, -1), np.sum(bwd * bwd, -1)
        fok, bok = fin & (f2 < r2), bin_ & (b2 < r2)
        short = np.where(fin & ~(bin_ & (b2 < f2)), 1, np.where(bin_, 2, 0))
        pick = np.where(fok, 1, np.where(bok, 2, short))
        return np.where((pick == 1)[..., None], fwd, np.where((pick == 2)[..., None], bwd, 0.0)), fin | bin_

    dx, okx = diff(1.0 / wid, 0.0)
    dy, oky = diff(0.0, 1.0 / hgt)
    Pu, Pv = dx * float(wid), dy * float(hgt)
    n = np.cross(Pu, Pv)
    ok = (pack_pos[..., 3] > 0.5) & okx & oky & (np.sum(n * n, -1) > 0.0)
    return dict(ru=np.where(ok, ru, -1.0), rv=rv, Pu=Pu, Pv=Pv, ok=ok, texel=(1.0 / wid, 1.0 / hgt))


def dominant_axis(n):
    """0, 1 or 2: the largest |component|, ties to the lower axis."""
    a = np.abs(n)
    return np.where((a[..., 0] >= a[..., 1]) & (a[..., 0] >= a[..., 2]), 0,
                    np.where(a[..., 1] >= a[..., 2], 1, 2))


def patch_box(P, cx, cy, ru, rv, texel):
    """Grown patch square relative to the rep: (a0, a1, b0, b1) in UV. Grown
    by PATCH_GROW of its size (no cracks between patches) plus half a texel
    (`texel`: (1 / width, 1 / height)), so with edge_uv a patch at an island's
    border reaches the face's 3D edge."""
    a0, a1 = cx / P - ru, (cx + 1.0) / P - ru
    b0, b1 = cy / P - rv, (cy + 1.0) / P - rv
    ga = (a1 - a0) * PATCH_GROW + 0.5 * texel[0]
    gb = (b1 - b0) * PATCH_GROW + 0.5 * texel[1]
    return a0 - ga, a1 + ga, b0 - gb, b1 + gb


def patch_centre(P, fs):
    """Offset (a, b) in UV from the rep to the centre of the grown patch."""
    cy, cx = np.mgrid[0:P, 0:P].astype(float)
    a0, a1, b0, b1 = patch_box(P, cx, cy, fs["ru"], fs["rv"], fs["texel"])
    return (a0 + a1) * 0.5, (b0 + b1) * 0.5, (a0, a1, b0, b1)


def pass_reach(fs):
    """WSB_LReach (mesh only): the largest distance, along either image axis of
    the sample's own view (its dominant axis), from the centre of its grown
    patch to a corner, over all usable samples. A view pixel can only hit the
    patches of samples whose centre is this close along both image axes."""
    P = w.PMAX
    ac, bc, (a0, a1, b0, b1) = patch_centre(P, fs)
    dom = dominant_axis(np.cross(fs["Pu"], fs["Pv"]))
    r = 0.0
    for a in (a0, a1):
        for b in (b0, b1):
            c = fs["Pu"] * (a - ac)[..., None] + fs["Pv"] * (b - bc)[..., None]
            for ax, ua, va in AXES:
                d = np.maximum(np.abs(c[..., ua]), np.abs(c[..., va]))
                r = np.maximum(r, np.where(dom == ax, d, 0.0))
    return float(np.max(np.where(fs["ok"], r, 0.0)))


# --------------------------------------------------------------------------
# Views (mesh only). View pixel size p0: the smallest whose views over all
# boxes fit VW x VH pixels (bisection, like SortFit). View e = box * 6 +
# 2 * axis + (0: seen from +axis, 1: from -axis).
# --------------------------------------------------------------------------

def view_dims(e, p):
    return [(w.dims_of(e[..., ua], p), w.dims_of(e[..., va], p)) for _, ua, va in AXES]


QUALITY_SHARE = {1: 0.125, 2: 0.25, 3: 0.5, 4: 1.0}   # Quality -> share of VW x VH


def fit_view_pixel(boxes, c_hi, quality=4):
    e = np.maximum(boxes["hi"] - boxes["lo"], 0.0)
    a, b = 0.0, c_hi
    for _ in range(w.FIT_STEPS):
        mid = (a + b) * 0.5
        acc = 0.0
        for k in range(len(e)):
            if boxes["ok"][k]:
                for W, H in view_dims(e[k], mid):
                    acc = acc + 2.0 * (W * H)
        a, b = (a, mid) if acc <= float(VW * VH) * QUALITY_SHARE[w.level_of(quality)] else (mid, b)
    return b


def view_table(boxes, p0, level=0):
    """Per view e: fine origin (ou, ov), fine dims (W, H), dims at the level
    (Wl, Hl), offset in the linear table at the level, valid."""
    NB = len(boxes["ok"])
    e = np.maximum(boxes["hi"] - boxes["lo"], 0.0)
    ctr = (boxes["lo"] + boxes["hi"]) * 0.5
    B = 2.0 ** level
    t = {k: np.zeros(NB * NVIEWS) for k in ("ou", "ov", "W", "H", "Wl", "Hl", "size", "offset", "ok", "ax", "sgn")}
    for b in range(NB):
        for ai, (ax, ua, va) in enumerate(AXES):
            W, H = w.dims_of(e[b, ua], p0), w.dims_of(e[b, va], p0)
            for k, sgn in enumerate((1.0, -1.0)):
                i = b * NVIEWS + 2 * ai + k
                t["ou"][i] = ctr[b, ua] - p0 * W * 0.5
                t["ov"][i] = ctr[b, va] - p0 * H * 0.5
                t["W"][i], t["H"][i] = W, H
                t["Wl"][i], t["Hl"][i] = w.ceil_(W / B), w.ceil_(H / B)
                t["ok"][i] = boxes["ok"][b]
                t["ax"][i], t["sgn"][i] = ax, sgn
                t["size"][i] = t["Wl"][i] * t["Hl"][i] if boxes["ok"][b] else 0.0
    t["offset"] = np.concatenate([[0.0], np.cumsum(t["size"])[:-1]])
    return t


def col_level(reach, p0):
    """WSB_LColLevel: the lookup's column grid. Each view is cut into columns of
    B x B fine pixels (B = 2^L, the smallest with MAX_REACH_CELLS columns
    spanning a patch's reach plus a pixel); the samples are sorted by the
    column of their patch centre in their view (WSB_LColKeys) and a pixel
    tests the samples of the columns within kc of its own. -> (L, B, kc)."""
    r = reach / p0 + 1.0
    L, Bs = 0, 1.0
    for _ in range(MAX_LEVEL):
        if r > Bs * MAX_REACH_CELLS:
            L, Bs = L + 1, Bs * 2.0
    return L, Bs, min(w.ceil_(r / Bs), MAX_REACH_CELLS)


def edge_uv(uv_mask, hu, hv, ru, rv):
    """The UV a hit reads (mask, position): the hit's own UV, or, if that is
    outside the islands, the UV half a texel back towards the sample's rep.
    An island's last texel centre can be up to a texel short of the face's 3D
    edge; this lets the patch reach the edge instead of leaving a gap the
    surface behind shows through."""
    hgt, wid = uv_mask.shape
    inside = w.sample_nearest(uv_mask, hu, hv) > w.UV_INSIDE
    du, dv = 0.5 / wid, 0.5 / hgt
    bu = hu - np.clip(hu - ru, -du, du)
    bv = hv - np.clip(hv - rv, -dv, dv)
    return np.where(inside, hu, bu), np.where(inside, hv, bv)


def pass_lookup(position, uv_mask, mesh, fs, p0, reach):
    """Front-most hit per fine view pixel, as the graph's WSB_LLookup gathers
    it. -> per fine table entry (VW * VH): (hit u, hit v, depth, a), a = 0
    where nothing covers the pixel."""
    boxes = mesh["boxes"]
    pos = mesh["pack_pos"][..., :3].reshape(-1, 3)
    sbox = mesh["sample_box"].reshape(-1)
    P = w.PMAX
    tab = view_table(boxes, p0, 0)
    out = np.zeros((VW * VH, 4))
    cy, cx = np.mgrid[0:P, 0:P].astype(float)
    Pu, Pv = fs["Pu"].reshape(-1, 3), fs["Pv"].reshape(-1, 3)
    n = np.cross(Pu, Pv)
    dom = dominant_axis(n)
    a0, a1, b0, b1 = (x.reshape(-1) for x in patch_box(P, cx, cy, fs["ru"], fs["rv"], fs["texel"]))
    ac, bc = (a0 + a1) * 0.5, (b0 + b1) * 0.5
    ru, rv = fs["ru"].reshape(-1), fs["rv"].reshape(-1)
    diag2 = (np.sum(Pu * Pu, -1) + np.sum(Pv * Pv, -1)) / float(P * P)
    _, Bs, kc = col_level(reach, p0)
    usable = fs["ok"].reshape(-1) & (sbox >= 0)
    for e in range(len(tab["ok"])):
        if not tab["ok"][e]:
            continue
        b, ax, sgn = e // NVIEWS, int(tab["ax"][e]), tab["sgn"][e]
        _, ua, va = AXES[ax]
        W, H, ou, ov = int(tab["W"][e]), int(tab["H"][e]), tab["ou"][e], tab["ov"][e]
        ids = np.nonzero(usable & (sbox == b) & (dom == ax))[0]
        if len(ids) == 0:
            continue
        rp = pos[ids]
        cp = rp + Pu[ids] * ac[ids, None] + Pv[ids] * bc[ids, None]
        scu = np.floor(np.clip(np.floor((cp[:, ua] - ou) / p0), 0, W - 1) / Bs)
        scv = np.floor(np.clip(np.floor((cp[:, va] - ov) / p0), 0, H - 1) / Bs)
        cs = [rp + Pu[ids] * a[ids, None] + Pv[ids] * bb[ids, None] for a in (a0, a1) for bb in (b0, b1)]
        uu = np.stack([c[:, ua] for c in cs], 1)
        vv = np.stack([c[:, va] for c in cs], 1)
        x0 = np.floor((uu.min(1) - ou) / p0 - 0.5).astype(int)
        x1 = np.ceil((uu.max(1) - ou) / p0 - 0.5).astype(int)
        y0 = np.floor((vv.min(1) - ov) / p0 - 0.5).astype(int)
        y1 = np.ceil((vv.max(1) - ov) / p0 - 0.5).astype(int)
        K = int(max((x1 - x0).max(initial=0), (y1 - y0).max(initial=0))) + 1
        best_z = np.full(H * W, -np.inf)
        best_id = np.full(H * W, -1)
        best_uv = np.zeros((H * W, 2))
        m11, m12, m21, m22 = Pu[ids, ua], Pv[ids, ua], Pu[ids, va], Pv[ids, va]
        det = m11 * m22 - m12 * m21
        sdet = np.where(det == 0.0, 1.0, det)
        for oy in range(K):
            for ox in range(K):
                px, py = x0 + ox, y0 + oy
                live = (px <= x1) & (py <= y1) & (px >= 0) & (py >= 0) & (px < W) & (py < H)
                if not live.any():
                    continue
                U = ou + (px + 0.5) * p0
                V = ov + (py + 0.5) * p0
                live &= (np.abs(scu - np.floor(px / Bs)) <= kc) & (np.abs(scv - np.floor(py / Bs)) <= kc)
                du_, dv_ = U - rp[:, ua], V - rp[:, va]
                al = (du_ * m22 - m12 * dv_) / sdet
                be = (m11 * dv_ - du_ * m21) / sdet
                hit = live & (det != 0.0) & (al >= a0[ids]) & (al < a1[ids]) & (be >= b0[ids]) & (be < b1[ids])
                hu, hv = edge_uv(uv_mask, ru[ids] + al, rv[ids] + be, ru[ids], rv[ids])
                hit &= w.sample_nearest(uv_mask, hu, hv) > w.UV_INSIDE
                X = rp + Pu[ids] * al[:, None] + Pv[ids] * be[:, None]
                q = w.sample_nearest(position, hu, hv)
                hit &= np.sum((q - X) ** 2, -1) < diag2[ids]
                z = sgn * X[:, ax]
                sel = np.nonzero(hit)[0]
                if len(sel) == 0:
                    continue
                pix, zz, ii, hu_, hv_ = py[sel] * W + px[sel], z[sel], ids[sel], hu[sel], hv[sel]
                order = np.lexsort((-ii, zz, pix))          # per pixel: max depth, then lowest id last
                pix, zz, ii, hu_, hv_ = pix[order], zz[order], ii[order], hu_[order], hv_[order]
                last = np.r_[pix[1:] != pix[:-1], True]
                pix, zz, ii, hu_, hv_ = pix[last], zz[last], ii[last], hu_[last], hv_[last]
                better = (zz > best_z[pix]) | ((zz == best_z[pix]) & (ii < best_id[pix]))
                pix, zz, ii, hu_, hv_ = pix[better], zz[better], ii[better], hu_[better], hv_[better]
                best_z[pix], best_id[pix] = zz, ii
                best_uv[pix, 0], best_uv[pix, 1] = hu_, hv_
        off = int(tab["offset"][e])
        cov = best_id >= 0
        nn = n[np.maximum(best_id, 0)]
        a = np.sqrt(np.sum(nn * nn, -1)) / np.maximum(np.abs(nn[:, ax]), 1e-30)
        blk = np.zeros((H * W, 4))
        blk[:, 0:2] = best_uv
        blk[:, 2] = np.where(cov, best_z, 0.0)
        blk[:, 3] = np.where(cov, a, 0.0)
        out[off:off + H * W] = blk
    return out, tab


# --------------------------------------------------------------------------
# Per stroke
# --------------------------------------------------------------------------

def pass_level(radius, luv, p0):
    """WSB_LFrame: sigma, the level L (the first with sigma / pixel <= 2.5, by
    doubling), pixel p, the 2D Gaussian's sigma in pixels."""
    sigma = max(radius * luv * 0.5, MIN_SIGMA_PIXELS * p0)
    L, p = 0, p0
    for _ in range(MAX_LEVEL):
        if sigma > PIXELS_PER_SIGMA * p:
            L, p = L + 1, p * 2.0
    sb2 = max(sigma * sigma - p * p * 0.25, (0.3 * p) ** 2)
    sg = np.sqrt(sb2) / p
    return dict(sigma=sigma, L=L, p=p, sg=sg, K=w.ceil_(3.0 * sg))


def pass_fill(mask, fine, tab0, tabL, lev):
    """WSB_LFill: per level pixel, the front-most surface of its B x B fine
    pixels (fine pixels more than sigma + 2 p behind it are dropped), as
    (sum m a, sum a, mean depth, has) / B^2."""
    B = 2 ** lev["L"]
    out = np.zeros((VW * VH, 4))
    for e in range(len(tab0["ok"])):
        if not tab0["ok"][e]:
            continue
        W, H = int(tab0["W"][e]), int(tab0["H"][e])
        Wl, Hl = int(tabL["Wl"][e]), int(tabL["Hl"][e])
        f = fine[int(tab0["offset"][e]):int(tab0["offset"][e]) + W * H].reshape(H, W, 4)
        f = np.pad(f, ((0, Hl * B - H), (0, Wl * B - W), (0, 0)))
        cov = f[..., 3] > 0
        m = np.where(cov, w.sample_nearest(mask, f[..., 0], f[..., 1]), 0.0)
        z = np.where(cov, f[..., 2], -np.inf).reshape(Hl, B, Wl, B)
        zf = z.max((1, 3))
        keep = (z >= (zf - (lev["sigma"] + 2.0 * lev["p"]))[:, None, :, None]).reshape(Hl * B, Wl * B) & cov
        a = np.where(keep, f[..., 3], 0.0)
        s = np.stack([m * a, a, a * np.where(keep, f[..., 2], 0.0)], -1).reshape(Hl, B, Wl, B, 3).sum((1, 3))
        s = s / float(B * B)
        blk = np.zeros((Hl, Wl, 4))
        has = s[..., 1] > 0
        blk[..., 0], blk[..., 1] = s[..., 0], s[..., 1]
        blk[..., 2] = np.where(has, s[..., 2] / np.maximum(s[..., 1], 1e-30), 0.0)
        blk[..., 3] = has
        o = int(tabL["offset"][e])
        out[o:o + Hl * Wl] = blk.reshape(-1, 4)
    return out


def _views(img, tab):
    for e in range(len(tab["ok"])):
        if tab["ok"][e]:
            Wl, Hl, o = int(tab["Wl"][e]), int(tab["Hl"][e]), int(tab["offset"][e])
            yield e, img[o:o + Hl * Wl].reshape(Hl, Wl, 4), o


def pass_depth_fill(img, tab):
    """WSB_LDepthFill (one ring): empty pixels next to a pixel with a depth take
    the mean depth of those 4-neighbours."""
    out = img.copy()
    for e, v, o in _views(img, tab):
        has = v[..., 3] > 0.5
        acc = np.zeros(v.shape[:2])
        cnt = np.zeros(v.shape[:2])
        Hl, Wl = v.shape[:2]
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.zeros((Hl, Wl))
            hh = np.zeros((Hl, Wl), bool)
            ys, xs = slice(max(dy, 0), Hl + min(dy, 0)), slice(max(dx, 0), Wl + min(dx, 0))
            yd, xd = slice(max(-dy, 0), Hl + min(-dy, 0)), slice(max(-dx, 0), Wl + min(-dx, 0))
            sh[yd, xd], hh[yd, xd] = v[ys, xs, 2], has[ys, xs]
            acc += np.where(hh, sh, 0.0)
            cnt += hh
        grow = ~has & (cnt > 0)
        nv = v.copy()
        nv[..., 2] = np.where(grow, acc / np.maximum(cnt, 1), v[..., 2])
        nv[..., 3] = has | grow
        out[o:o + Hl * Wl] = nv.reshape(-1, 4)
    return out


def pass_blur(img, tab, lev, axis):
    """WSB_LBlurX / Y: depth-aware Gaussian along `axis` (0: x, 1: y) of each
    view; taps outside the view add nothing; depth and has pass through."""
    sg, K, sz = lev["sg"], int(lev["K"]), lev["sigma"]
    out = img.copy()
    for e, v, o in _views(img, tab):
        nv = v.copy()
        nv[..., 0:2] = 0.0
        n = v.shape[1 - axis]
        zc = v[..., 2]
        for t in range(-K, K + 1):
            lo, hi = max(0, -t), min(n, n - t)
            if lo >= hi:
                continue
            g = np.exp(-(t * t) / (2.0 * sg * sg))
            if axis == 0:
                q, c = v[:, lo + t:hi + t], (slice(None), slice(lo, hi))
            else:
                q, c = v[lo + t:hi + t, :], (slice(lo, hi), slice(None))
            wz = g * np.exp(-(q[..., 2] - zc[c]) ** 2 / (2.0 * sz * sz))
            nv[c + (slice(0, 2),)] += wz[..., None] * q[..., 0:2]
        out[o:o + v.shape[0] * v.shape[1]] = nv.reshape(-1, 4)
    return out


def pass_resolve(mask, position, uv_mask, mesh, img, tab, lev, amount=w.DITHER, shape_params=None, stats=None):
    """WSB_LResolve: every view of the texel's box adds its blurred (num, den)
    at the texel's projection, each of the 4 bilinear pixels times
    exp(-(z_pixel - z_texel)^2 / 2 sigma^2)."""
    hgt, wid = mask.shape
    yy, xx = np.mgrid[0:hgt, 0:wid]
    u, v = (xx + 0.5) / wid, (yy + 0.5) / hgt
    pos = w.sample_nearest(position, u, v)
    inside = w.sample_nearest(uv_mask, u, v) > 0.5
    b = w.box_of(pos, mesh["coarse"], mesh["boxes"])
    p, sigma = lev["p"], lev["sigma"]
    num = np.zeros((hgt, wid))
    den = np.zeros((hgt, wid))
    for e, vw, o in _views(img, tab):
        ax, sgn = int(tab["ax"][e]), tab["sgn"][e]
        _, ua, va = AXES[ax]
        sel = inside & (b == e // NVIEWS)
        if not sel.any():
            continue
        x = (pos[..., ua] - tab["ou"][e]) / p - 0.5
        y = (pos[..., va] - tab["ov"][e]) / p - 0.5
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        z = sgn * pos[..., ax]
        Wl, Hl = int(tab["Wl"][e]), int(tab["Hl"][e])
        for oy in (0, 1):
            for ox in (0, 1):
                X, Y = (x0 + ox).astype(int), (y0 + oy).astype(int)
                ok = sel & (X >= 0) & (Y >= 0) & (X < Wl) & (Y < Hl)
                q = vw[np.clip(Y, 0, Hl - 1), np.clip(X, 0, Wl - 1)]
                f = np.exp(-(q[..., 2] - z) ** 2 / (2.0 * sigma * sigma))
                wt = np.where(ok, (fx if ox else 1.0 - fx) * (fy if oy else 1.0 - fy) * f, 0.0)
                num += wt * q[..., 0]
                den += wt * q[..., 1]
    ok = inside & (den > 1e-6) & (b >= 0)
    if stats is not None:
        stats["uncovered"] = float((inside & ~ok).sum() / max(inside.sum(), 1))
    bl = np.clip(num / np.maximum(den, 1e-12), 0.0, 1.0)
    if shape_params is not None:
        bl = w.levels(bl, *shape_params)
    bl = w.dither(bl, xx.astype(float), yy.astype(float), amount)
    return np.where(ok, bl, w.sample_nearest(mask, u, v))


def prepare(position, uv_mask, mesh=None, quality=4):
    """Everything that depends only on the mesh and Quality (cached by Painter)."""
    mesh = mesh or w.prepare(position, uv_mask)
    fs = pass_frame_samples(position, uv_mask, mesh["pack_pos"], mesh["coarse"])
    reach = pass_reach(fs)
    p0 = fit_view_pixel(mesh["boxes"], mesh["c_hi"], quality)
    fine, tab0 = pass_lookup(position, uv_mask, mesh, fs, p0, reach)
    return dict(mesh=mesh, fs=fs, reach=reach, p0=p0, fine=fine, tab0=tab0)


def world_space_blur_lite(mask, position, uv_mask, radius=0.01, stats=None, lite=None, dither=w.DITHER,
                          shape=None, quality=4):
    lite = lite or prepare(position, uv_mask, quality=quality)
    mesh = lite["mesh"]
    lev = pass_level(radius, mesh["luv"], lite["p0"])
    tabL = view_table(mesh["boxes"], lite["p0"], lev["L"])
    img = pass_fill(mask, lite["fine"], lite["tab0"], tabL, lev)
    for _ in range(DEPTH_FILL):
        img = pass_depth_fill(img, tabL)
    img = pass_blur(img, tabL, lev, 0)
    img = pass_blur(img, tabL, lev, 1)
    out = pass_resolve(mask, position, uv_mask, mesh, img, tabL, lev, dither, shape, stats)
    if stats is not None:
        stats.update(lev)
        stats.update(p0=lite["p0"], reach=lite["reach"], views=int(tabL["ok"].sum()),
                     view_pixels=float(lite["tab0"]["size"].sum()), level_pixels=float(tabL["size"].sum()))
    return out
