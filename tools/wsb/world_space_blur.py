"""
Reference implementation of the World Space Mask Blur filter.

Each pass_* function below corresponds to one Pixel Processor (or a group of
them) in the Substance 3D Designer graph described in
docs/designer-build-guide.md, with the same constants, sample positions, loop
order and math. It is the spec for the Designer build: designer/test_sbs.py
checks the generated .sbs against it.

How it works
  1. The surface is reduced to PMAX x PMAX samples (position + mask average).
  2. Pieces separated by empty space get their own box: the samples are
     sorted into a coarse grid over the whole texture set, the occupied cells
     are projected onto X (slabs separated by empty columns) and, inside each
     slab, onto Z or Y (runs). Each (slab, run) is a box, cut to the exact
     bounds of its samples (mesh only: cached).
  3. Every box gets its own sort grid and voxel grid, all with the same cell /
     voxel size, packed one after the other in linear tables. So the voxel
     size depends on how much space the pieces themselves take, not on how far
     apart they are laid out.
  4. The samples are sorted by their cell in their box's sort grid, and every
     cell gets its range in the sorted list (mesh only).
  5. Each voxel gathers the samples next to it (tent weights), reading only
     the cells it overlaps; a separable 3D Gaussian (x, y, z passes) blurs
     each box's grid.
  6. Every texel reads its box's grid at its own 3D position (cubic B-spline).
Samples are weighted by the 3D surface area they stand for. The grid stores
(sum w*area*mask, sum w*area); their ratio is the blurred mask, so empty space
and UV padding don't darken it. Pieces in different boxes never blur into
each other.

Radius is in UV units: sigma = Radius / 2 x the mesh's average 3D length per
UV unit (from its total 3D area and UV area), so it doesn't depend on how the
pieces are laid out or on the texture resolution.

Conventions (same as Designer's Pixel Processor):
  - Images are numpy arrays indexed [row, col] == [y, x].
  - UV (0,0) is the top-left corner; texel centers are at (i + 0.5) / size.

Inputs (what Painter feeds the filter):
  mask      : (H, W)    float, the mask being filtered          -> "input"
  position  : (H, W, 3) float, object-space position per texel  -> meshPosition
  uv_mask   : (H, W)    float, 1 inside UV islands, 0 outside   -> meshUVMask
"""

import numpy as np

# Fixed sizes
PMAX = 512         # surface samples: PMAX x PMAX
SUB = 4            # reads per sample patch, per axis (4 x 4)
SUB_OFFSET = 0.47  # read k sits at (k + SUB_OFFSET) / SUB of its patch. Not 0.5:
                   # with power-of-two sizes that lands exactly on texel edges
FRAME_GRID = 128   # the bounds pass looks at FRAME_GRID^2 points (same offset rule)
AREA_GRID = 32     # total area: AREA_GRID^2 block sums, then their sum
CW, CH = 1024, 1024    # coarse cell table: at most CW * CH coarse cells
DMAX = 256             # ... and at most DMAX per axis
S_MAX, R_MAX = 32, 32  # slabs along X, runs per slab (along Z or Y): S_MAX * R_MAX boxes
AW, AH = 2048, 1024    # linear tables for the fine sort cells and the voxels (AW * AH each)

# Quality level -> voxel budget n^3, shared by all boxes
QUALITY = {1: 48, 2: 64, 3: 96, 4: 128}

MARGIN = 2.0           # empty cells / voxels kept around each box on each side
FIT_STEPS = 24         # bisection steps when fitting a cell size to a budget
CEIL_EPS = 0.001       # cells per axis = ceil(extent / cell - CEIL_EPS) + 2 MARGIN
MIN_SIGMA_VOXELS = 0.8   # smallest total blur sigma, in voxels of the finest grid
MIN_BLUR_VOXELS = 0.3    # smallest Gaussian sigma of the separable blur, in voxels
VOXELS_PER_SIGMA = 2.5   # bigger blurs use bigger voxels: sigma / h stays <= this ...
MAX_VOXEL_CELLS = 4.0    # ... until a voxel is this many sort cells wide
REJECT_CELLS = 2.0     # pack: reads farther than this (in coarse cells) from the patch's
                       # representative point belong to another surface
UV_INSIDE = 0.99   # uv-mask threshold for point samples
KEY_NONE = 8388608.0   # sort key of empty patches (2^23, after every cell id)
BIG = 1e20
DITHER = 1.0           # default dither amplitude, in 8-bit steps (0 = off)


def configure(pmax, aw, ah, cw, ch, dmax, s_max, r_max, quality, frame_grid=None, sub=None, area_grid=None):
    """Change the fixed sizes (for small automated tests)."""
    global PMAX, AW, AH, CW, CH, DMAX, S_MAX, R_MAX, QUALITY, FRAME_GRID, SUB, AREA_GRID
    PMAX, AW, AH, CW, CH, DMAX, S_MAX, R_MAX = pmax, aw, ah, cw, ch, dmax, s_max, r_max
    QUALITY = dict(quality)
    FRAME_GRID = frame_grid or min(pmax, 128)
    SUB = sub or SUB
    AREA_GRID = area_grid or min(pmax, 32)


def n_boxes():
    return S_MAX * R_MAX


def level_of(quality):
    return int(min(max(np.floor(quality + 0.5), 1), max(QUALITY)))


def budget_of(quality):
    return float(QUALITY[level_of(quality)]) ** 3


def sample_nearest(img, u, v):
    h, w = img.shape[:2]
    x = np.clip(np.floor(u * w).astype(int), 0, w - 1)
    y = np.clip(np.floor(v * h).astype(int), 0, h - 1)
    return img[y, x]


def ceil_(x):
    return -np.floor(-x)


def dims_of(e, c):
    """Cells per axis covering extent e with cells of size c, plus margins."""
    return ceil_(e / c - CEIL_EPS) + 2.0 * MARGIN


def fit_cell(lo, hi, valid, budget, c_hi, dmax=None):
    """Smallest cell size (bisection, FIT_STEPS steps from (0, c_hi]) whose grids
    over all valid boxes [lo, hi] hold at most `budget` cells in total (and at
    most `dmax` per axis). Boxes are summed in order, like the graph's loop."""
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    e = np.maximum(hi - lo, 0.0)
    a, b = 0.0, c_hi
    for _ in range(FIT_STEPS):
        mid = (a + b) * 0.5
        acc, worst = 0.0, 0.0
        for k in range(len(e)):
            if valid[k]:
                d = dims_of(e[k], mid)
                acc = acc + d[0] * d[1] * d[2]
                worst = max(worst, d.max())
        ok = acc <= budget and (dmax is None or worst <= dmax)
        a, b = (a, mid) if ok else (mid, b)
    return b


# --------------------------------------------------------------------------
# WSB_Bounds (16 x 16, mesh only): bounding box of FRAME_GRID^2 points
#   left half = (lo.xyz, size), right half = (hi.xyz, 1)
# WSB_Coarse (16 x 16, mesh only): the coarse grid over the whole texture set
#   left half = (origin.xyz, c), right half = (dims.xyz, 1)
# --------------------------------------------------------------------------

def pass_bounds(position, uv_mask):
    c = (np.arange(FRAME_GRID) + SUB_OFFSET) / FRAME_GRID
    u, v = np.meshgrid(c, c)                           # loop i -> (i % G, i // G)
    valid = sample_nearest(uv_mask, u, v) > UV_INSIDE
    p = sample_nearest(position, u, v)[valid]
    if len(p):
        lo, hi = p.min(axis=0), p.max(axis=0)
    else:
        lo, hi = np.full(3, BIG), np.full(3, -BIG)
    size = max(float(np.max(hi - lo)), 1e-8)
    return dict(lo=lo, hi=hi, size=size)


def pass_coarse(bounds):
    lo, hi = bounds["lo"], bounds["hi"]
    c = fit_cell(lo, hi, [True], float(CW * CH), bounds["size"], dmax=float(DMAX))
    dims = dims_of(np.maximum(hi - lo, 0.0), c)
    origin = (lo + hi) * 0.5 - c * dims * 0.5
    return dict(origin=origin, c=c, dims=dims)


# --------------------------------------------------------------------------
# WSB_PackPos / WSB_PackArea / WSB_PackMask (PMAX x PMAX)
#   Sample (cx, cy) covers [cx, cx+1) / PMAX x [cy, cy+1) / PMAX of the UV square.
#   PackPos  -> (rep.xyz, found): valid read closest to the patch centre (mesh only)
#   PackArea -> (area, uv, 0, 1): mean 3D texel area of the reads taken times
#               their share, and that share (mesh only)
#   PackMask -> (mean mask * area, area, 0, 1): mask averaged over the reads
#               near rep (a patch straddling two far-apart islands only
#               averages one of them)
# --------------------------------------------------------------------------

def _patches():
    cy, cx = np.mgrid[0:PMAX, 0:PMAX].astype(float)
    i = np.arange(SUB * SUB)
    sx, sy = i % SUB, i // SUB
    su = (cx[..., None] + (sx + SUB_OFFSET) / SUB) / PMAX
    sv = (cy[..., None] + (sy + SUB_OFFSET) / SUB) / PMAX
    return (cx + 0.5) / PMAX, (cy + 0.5) / PMAX, su, sv


def pass_pack_pos(position, uv_mask):
    cu, cv, su, sv = _patches()
    valid = sample_nearest(uv_mask, su, sv) > UV_INSIDE
    d = (su - cu[..., None]) ** 2 + (sv - cv[..., None]) ** 2
    d = np.where(valid, d, np.inf)
    best = d.argmin(axis=-1)                           # first minimum, loop order
    found = np.isfinite(d.min(axis=-1))
    bu = np.take_along_axis(su, best[..., None], -1)[..., 0]
    bv = np.take_along_axis(sv, best[..., None], -1)[..., 0]
    rep = sample_nearest(position, bu, bv)
    out = np.zeros((PMAX, PMAX, 4))
    out[..., :3] = np.where(found[..., None], rep, 0.0)
    out[..., 3] = found
    return out


def _r2(coarse):
    return (REJECT_CELLS * coarse["c"]) ** 2


def _taken(position, uv_mask, pack_pos, coarse):
    """Reads of each patch that count: inside a UV island and near the patch's
    representative point."""
    _, _, su, sv = _patches()
    rep = pack_pos[..., :3]
    valid = sample_nearest(uv_mask, su, sv) > UV_INSIDE
    pos = sample_nearest(position, su, sv)
    take = valid & (np.sum((pos - rep[:, :, None, :]) ** 2, axis=-1) < _r2(coarse))
    return take & (pack_pos[..., 3:4] > 0.5), su, sv


def texel_area(position, uv_mask, u, v, r2):
    """3D area of the texel at (u, v): |dP/dx x dP/dy| from neighbouring texels.
    Each axis uses the forward difference, or the backward one when the forward
    neighbour is outside the island (or on another surface); 0 if neither works."""
    hgt, wid = uv_mask.shape
    p = sample_nearest(position, u, v)

    def inside(x, y):
        return (np.minimum(x, y) > 0.0) & (np.maximum(x, y) < 1.0)

    def diff(du, dv):
        fwd = sample_nearest(position, u + du, v + dv) - p
        bwd = p - sample_nearest(position, u - du, v - dv)
        fok = (sample_nearest(uv_mask, u + du, v + dv) > UV_INSIDE) & (np.sum(fwd * fwd, -1) < r2)
        bok = (sample_nearest(uv_mask, u - du, v - dv) > UV_INSIDE) & (np.sum(bwd * bwd, -1) < r2)
        fok &= inside(u + du, v + dv)                      # off the texture: no neighbour
        bok &= inside(u - du, v - dv)
        return np.where(fok[..., None], fwd, np.where(bok[..., None], bwd, 0.0))
    return np.linalg.norm(np.cross(diff(1.0 / wid, 0.0), diff(0.0, 1.0 / hgt)), axis=-1)


def pass_pack_area(position, uv_mask, pack_pos, coarse):
    take, su, sv = _taken(position, uv_mask, pack_pos, coarse)
    a = texel_area(position, uv_mask, su, sv, _r2(coarse))
    out = np.zeros((PMAX, PMAX, 4))
    out[..., 0] = np.where(take, a, 0.0).sum(-1) / (SUB * SUB)
    out[..., 1] = take.sum(-1) / (SUB * SUB)
    out[..., 3] = 1.0
    return out


def pass_pack_mask(mask, position, uv_mask, pack_pos, pack_area, coarse, area_total):
    """-> (mean mask * area, area, 0, 1). Weighting by 3D area (not UV area)
    keeps islands with different texel density from pulling unevenly. The area
    is in units of the mean sample area (sum of areas / PMAX^2), so the voxel
    sums, and the "is anything here" test in Resolve, don't depend on the
    mesh's size or units."""
    take, su, sv = _taken(position, uv_mask, pack_pos, coarse)
    m = sample_nearest(mask, su, sv)
    msum = np.where(take, m, 0.0).sum(-1)
    cnt = take.sum(-1)
    a = pack_area[..., 0] / max(area_total[0], 1e-30) * float(PMAX * PMAX)
    out = np.zeros((PMAX, PMAX, 4))
    out[..., 0] = np.where(cnt > 0, msum / np.maximum(cnt, 1) * a, 0.0)
    out[..., 1] = np.where(cnt > 0, a, 0.0)
    out[..., 3] = 1.0
    return out


def pass_area_total(pack_area):
    """WSB_AreaBlocks (AREA_GRID^2: each sums its block of samples, row by row)
    then WSB_AreaTotal (sums the blocks, row by row) -> (sum area, sum uv)."""
    G, B = AREA_GRID, PMAX // AREA_GRID
    blocks = pack_area[..., :2].reshape(G, B, G, B, 2).transpose(0, 2, 1, 3, 4).reshape(G * G, B * B, 2)
    acc = np.zeros((G * G, 2))
    for k in range(B * B):
        acc = acc + blocks[:, k]
    tot = np.zeros(2)
    for k in range(G * G):
        tot = tot + acc[k]
    return tot


def length_per_uv(area_total, tex_w, tex_h):
    """Average 3D length per UV unit: sqrt(3D area / UV area). Sum of `area`
    is the 3D area in units of (texel count / PMAX^2)."""
    return np.sqrt(area_total[0] * tex_w * tex_h / max(area_total[1], 1e-12))


# --------------------------------------------------------------------------
# Coarse sort (mesh only): WSB_Keys0, WSB_Sort0 k/j, WSB_Cells0, WSB_CellBounds
#   Keys: (cell id, sample index). Sort: bitonic sorting network, one Pixel
#   Processor per step, ordered by (key, index). Cells: (start, end) of each
#   cell's samples. CellBounds: exact (lo, hi) of each cell's samples.
# --------------------------------------------------------------------------

def coarse_cell(p, coarse):
    c = np.floor((p - coarse["origin"]) / coarse["c"])
    return np.clip(c, 0, coarse["dims"] - 1)


def pass_keys0(pack_pos, coarse):
    c = coarse_cell(pack_pos[..., :3], coarse)
    d = coarse["dims"]
    key = (c[..., 2] * d[1] + c[..., 1]) * d[0] + c[..., 0]
    out = np.zeros((PMAX, PMAX, 4))
    out[..., 0] = np.where(pack_pos[..., 3] > 0.5, key, KEY_NONE)
    out[..., 1] = np.arange(PMAX * PMAX).reshape(PMAX, PMAX)
    out[..., 3] = 1.0
    return out


def sort_steps():
    """(k, j) of every bitonic step, in order."""
    L = PMAX * PMAX
    k = 2
    while k <= L:
        j = k // 2
        while j >= 1:
            yield k, j
            j //= 2
        k *= 2


def pass_sort(keys):
    flat = keys.reshape(-1, 4)
    order = np.lexsort((flat[:, 1], flat[:, 0]))        # what the network produces
    return flat[order].reshape(keys.shape)


def cell_ranges(sorted_keys, n_cells):
    keys = sorted_keys[..., 0].reshape(-1)
    cid = np.arange(n_cells)
    return np.searchsorted(keys, cid, side="left"), np.searchsorted(keys, cid + 1, side="left")


def cell_bounds(sorted_keys, pack_pos, coarse):
    """Exact (lo, hi) of the samples in each coarse cell, as [z, y, x] arrays."""
    d = coarse["dims"].astype(int)
    n = d[0] * d[1] * d[2]
    keys = sorted_keys[..., 0].reshape(-1)
    idx = sorted_keys[..., 1].reshape(-1).astype(int)
    keep = keys < n
    p = pack_pos[..., :3].reshape(-1, 3)[idx[keep]]
    k = keys[keep].astype(int)
    lo = np.full((n, 3), BIG)
    hi = np.full((n, 3), -BIG)
    np.minimum.at(lo, k, p)
    np.maximum.at(hi, k, p)
    occ = np.zeros(n, bool)
    occ[k] = True
    sh = (d[2], d[1], d[0])
    return lo.reshape(sh + (3,)), hi.reshape(sh + (3,)), occ.reshape(sh)


# --------------------------------------------------------------------------
# Boxes (mesh only)
#   WSB_ColXZ / WSB_ColXY: (lo, hi, occupied) of each coarse column, over y / z
#   WSB_OccX: is column x (or a neighbour) occupied; WSB_SlabX: slab id of x
#             (a new slab after every empty bin), capped at S_MAX - 1, -1 = empty
#   WSB_SlabRange: first / last x of each slab
#   WSB_OccZY: per slab and bin, occupied along Z / along Y (dilated by 1 bin)
#   WSB_RunTab: run id of each bin along Z and Y (capped at R_MAX - 1, -1 = empty),
#               and the runs counted so far. A slab is split along Y if that
#               gives more runs, else along Z.
#   WSB_RunRange: first / last bin of each run; WSB_BoxRow, WSB_Boxes: exact
#               (lo, hi) of each box's samples (box id = slab * R_MAX + run)
# --------------------------------------------------------------------------

def _dilate(occ):
    out = occ.copy()
    out[1:] |= occ[:-1]
    out[:-1] |= occ[1:]
    return out


def _runs(occ, cap):
    """Run id of each True bin (a new run after every False bin), capped; -1 for False.
    Also the number of runs started so far minus 1, per bin."""
    start = occ & ~np.concatenate([[False], occ[:-1]])
    count = np.cumsum(start) - 1
    return np.where(occ, np.minimum(count, cap - 1), -1), count


def pass_boxes(cb, coarse):
    lo, hi, occ = cb                                   # [z, y, x]
    dx, dy, dz = coarse["dims"].astype(int)
    D = DMAX
    xz_lo, xz_hi, xz_occ = lo.min(1), hi.max(1), occ.any(1)        # [z, x]
    xy_lo, xy_hi, xy_occ = lo.min(0), hi.max(0), occ.any(0)        # [y, x]
    occx = _dilate(xz_occ.any(0))                                  # [x]
    slab, _ = _runs(occx, S_MAX)
    slab_x0 = np.full(S_MAX, BIG)
    slab_x1 = np.full(S_MAX, -1.0)
    for x in range(dx):
        if slab[x] >= 0:
            s = slab[x]
            slab_x0[s] = min(slab_x0[s], x)
            slab_x1[s] = max(slab_x1[s], x)
    run_z = np.full((S_MAX, D), -1)
    run_y = np.full((S_MAX, D), -1)
    use_y = np.zeros(S_MAX, bool)
    for s in range(S_MAX):
        if slab_x1[s] < slab_x0[s]:
            continue
        xs = slice(int(slab_x0[s]), int(slab_x1[s]) + 1)
        oz = np.zeros(D, bool)
        oy = np.zeros(D, bool)
        oz[:dz] = _dilate(xz_occ[:, xs].any(1))
        oy[:dy] = _dilate(xy_occ[:, xs].any(1))
        run_z[s], cz = _runs(oz, R_MAX)
        run_y[s], cy = _runs(oy, R_MAX)
        use_y[s] = cy[-1] > cz[-1]
    NB = n_boxes()
    box_lo = np.full((NB, 3), BIG)
    box_hi = np.full((NB, 3), -BIG)
    box_ok = np.zeros(NB, bool)
    for s in range(S_MAX):
        if slab_x1[s] < slab_x0[s]:
            continue
        xs = slice(int(slab_x0[s]), int(slab_x1[s]) + 1)
        runs = run_y[s] if use_y[s] else run_z[s]
        clo, chi, cocc = (xy_lo, xy_hi, xy_occ) if use_y[s] else (xz_lo, xz_hi, xz_occ)
        for r in range(R_MAX):
            bins = np.nonzero(runs == r)[0]
            if len(bins) == 0:
                continue
            bs = slice(bins.min(), bins.max() + 1)
            b = s * R_MAX + r
            box_lo[b] = clo[bs, xs].reshape(-1, 3).min(0)
            box_hi[b] = chi[bs, xs].reshape(-1, 3).max(0)
            box_ok[b] = cocc[bs, xs].any()
    return dict(slab=slab, run_z=run_z, run_y=run_y, use_y=use_y, lo=box_lo, hi=box_hi, ok=box_ok)


def box_of(p, coarse, boxes):
    """Box id of points p (..., 3) from their coarse column, -1 if none."""
    c = coarse_cell(p, coarse).astype(int)
    s = boxes["slab"][c[..., 0]]
    sc = np.maximum(s, 0)
    uy = boxes["use_y"][sc]
    run = np.where(uy, boxes["run_y"][sc, c[..., 1]], boxes["run_z"][sc, c[..., 2]])
    return np.where((s >= 0) & (run >= 0), sc * R_MAX + run, -1)


def grid_table(boxes, cell):
    """Per box: grid origin, cells per axis and offset in a linear table
    (boxes packed in id order). WSB_SortBoxes / WSB_VoxBoxes."""
    e = np.maximum(boxes["hi"] - boxes["lo"], 0.0)
    dims = dims_of(e, cell)
    origin = (boxes["lo"] + boxes["hi"]) * 0.5 - cell * dims * 0.5
    vol = np.where(boxes["ok"], dims[:, 0] * dims[:, 1] * dims[:, 2], 0.0)
    offset = np.concatenate([[0.0], np.cumsum(vol)[:-1]])
    return dict(origin=origin, dims=dims, offset=offset, vol=vol, cell=cell)


def lin(c, dims):
    return (c[..., 2] * dims[..., 1] + c[..., 1]) * dims[..., 0] + c[..., 0]


# --------------------------------------------------------------------------
# Fine sort (mesh only): WSB_SortFit (cell size hs), WSB_SortBoxes, WSB_Keys1,
# WSB_Sort1 k/j, WSB_Cells1 (AW x AH linear: (start, end) of each cell)
# --------------------------------------------------------------------------

def pass_keys1(pack_pos, coarse, boxes, st):
    p = pack_pos[..., :3]
    b = box_of(p, coarse, boxes)
    bb = np.maximum(b, 0)
    c = np.clip(np.floor((p - st["origin"][bb]) / st["cell"]), 0, st["dims"][bb] - 1)
    key = st["offset"][bb] + lin(c, st["dims"][bb])
    ok = (pack_pos[..., 3] > 0.5) & (b >= 0) & boxes["ok"][bb]
    out = np.zeros((PMAX, PMAX, 4))
    out[..., 0] = np.where(ok, key, KEY_NONE)
    out[..., 1] = np.arange(PMAX * PMAX).reshape(PMAX, PMAX)
    out[..., 3] = 1.0
    return out, b


# --------------------------------------------------------------------------
# WSB_VoxFit (16 x 16, Quality): finest voxel hq
# WSB_Frame (16 x 16, Radius / Quality)
#   left half  = (h, sigmaBlur / h, K, sigma / h)
#   right half = (sigma, length per UV, 0, 1)
#
# Total blur = tent splat (variance h^2/6) + Gaussian (sigmaBlur) + cubic
# B-spline lookup (h^2/3), so sigmaBlur^2 = sigma^2 - h^2/2.
# --------------------------------------------------------------------------

def pass_frame(radius, quality, hq, hs, luv):
    sigma = max(radius * luv * 0.5, MIN_SIGMA_VOXELS * hq)
    h = min(max(sigma / VOXELS_PER_SIGMA, hq), max(hq, MAX_VOXEL_CELLS * hs))
    sb2 = max(sigma * sigma - h * h * 0.5, (MIN_BLUR_VOXELS * h) ** 2)
    sg = np.sqrt(sb2) / h
    return dict(h=h, sg=sg, K=ceil_(3.0 * sg), sigma=sigma, sv=sigma / h, luv=luv, hq=hq)


# --------------------------------------------------------------------------
# WSB_VoxBoxes (per box: voxel grid origin, n, offset), WSB_VoxMap (AW x AH:
# each voxel's box and (i, j, k)), WSB_Splat: each voxel sums the samples of
# its box next to it with tent weights  w = prod(max(0, 1 - |p - c| / h)),
# reading only the sort cells that overlap [c - h, c + h].
#   -> (sum w*area*mask, sum w*area, same, same)
# The reference scatters instead: every sample adds to the voxels of its box.
# --------------------------------------------------------------------------

def pass_splat(pack_pos, pack_mask, sample_box, vt, h):
    vox = np.zeros((AW * AH, 2))
    pos = pack_pos[..., :3].reshape(-1, 3)
    b = pack_mask[..., :2].reshape(-1, 2)
    bx = sample_box.reshape(-1)
    keep = (pack_pos[..., 3].reshape(-1) > 0.5) & (b[:, 1] > 0) & (bx >= 0)
    pos, b, bx = pos[keep], b[keep], bx[keep]
    g = (pos - vt["origin"][bx]) / h - 0.5               # voxel-centre coordinates
    i0 = np.floor(g).astype(int)
    t = g - i0
    n = vt["dims"][bx].astype(int)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (np.where(dx, t[:, 0], 1 - t[:, 0]) * np.where(dy, t[:, 1], 1 - t[:, 1])
                     * np.where(dz, t[:, 2], 1 - t[:, 2]))
                q = i0 + np.array([dx, dy, dz])
                ok = (q.min(1) >= 0) & np.all(q < n, axis=1)
                idx = (vt["offset"][bx] + lin(q, n)).astype(int)
                np.add.at(vox, idx[ok], b[ok] * w[ok, None])
    return np.concatenate([vox, vox], 1)                 # smeared (0-1) and plain (2-3) blur


# --------------------------------------------------------------------------
# WSB_BlurX / Y / Z (AW x AH): 1D Gaussian inside each box's grid, sigma sg
# voxels, taps out to 3 sigma. With Smear on, one side of each axis gets a
# longer tail (see smear_sides). Taps outside the box's grid count as empty.
# --------------------------------------------------------------------------

def smear_sides(frame, axis, smear):
    """Gaussian sigma (voxels) for taps behind / ahead of a voxel along `axis`.
    Smear (length in multiples of the blur sigma, direction vector) gives the
    side the mask is dragged *from* a longer tail: content spreads towards
    +direction. Without smear both sides are sigmaBlur."""
    sg = frame["sg"]
    if smear is None:
        return sg, sg
    length, d = smear[0], np.asarray(smear[1], float)
    ln = np.sqrt(np.dot(d, d))
    if ln < 1e-6:
        return sg, sg
    da = d[axis] / ln
    tail = sg + length * abs(da) * frame["sv"]
    return (tail if da > 0 else sg), (tail if da < 0 else sg)   # (k < 0 side, k > 0 side)


def pass_blur(vox, vt, ok, frame, axis, smear=None):
    """Channels 0-1: the smeared blur (num, den). Channels 2-3: the plain blur,
    carried along in the same passes for Smear Strength (no extra reads)."""
    s_neg, s_pos = smear_sides(frame, axis, smear)
    sg, K = frame["sg"], int(frame["K"])
    out = np.zeros_like(vox)
    for b in np.nonzero(ok)[0]:
        n = vt["dims"][b].astype(int)
        o = int(vt["offset"][b])
        grid = vox[o:o + n[0] * n[1] * n[2]].reshape(n[2], n[1], n[0], 4)
        res = np.zeros_like(grid)
        na = n[axis]
        ax = 2 - axis                                   # grid is [z, y, x]
        # taps n or more voxels away are outside the grid for every voxel
        lo_t = -min(max(int(ceil_(3.0 * s_neg)), K), na)
        hi_t = min(max(int(ceil_(3.0 * s_pos)), K), na)
        for t in range(lo_t, hi_t + 1):
            sd = s_neg if t < 0 else s_pos
            w = np.zeros(4)
            if -ceil_(3.0 * s_neg) <= t <= ceil_(3.0 * s_pos):
                w[:2] = np.exp(-(t * t) / (2.0 * sd * sd))
            if abs(t) <= K:
                w[2:] = np.exp(-(t * t) / (2.0 * sg * sg))
            lo, hi = max(0, -t), min(na, na - t)
            if lo >= hi:                                # tap reaches past the whole grid
                continue
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            dst[ax], src[ax] = slice(lo, hi), slice(lo + t, hi + t)
            res[tuple(dst)] += w * grid[tuple(src)]
        out[o:o + n[0] * n[1] * n[2]] = res.reshape(-1, 4)
    return out


# --------------------------------------------------------------------------
# WSB_Resolve (input size, grayscale, 16 bit)
#   Tricubic B-spline lookup of the texel's box grid at its 3D position, plus
#   optional dithering against 8-bit banding.
# --------------------------------------------------------------------------

def bspline_weights(t):
    """Cubic B-spline weights for offsets -1, 0, +1, +2 at fraction t."""
    t2, t3 = t * t, t * t * t
    return [(1 - t) ** 3 / 6.0,
            (3 * t3 - 6 * t2 + 4) / 6.0,
            (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0,
            t3 / 6.0]


def _frac(x):
    return x - np.floor(x)


def hash12(x, y):
    """Uniform 0..1 per texel from arithmetic only ("hash without sine")."""
    p = _frac(np.stack([x, y, x], -1) * 0.1031)
    p = p + np.sum(p * (p[..., [1, 2, 0]] + 33.33), -1, keepdims=True)
    return _frac((p[..., 0] + p[..., 1]) * p[..., 2])


def dither(b, xx, yy, amount):
    """Add triangular noise of +-amount 8-bit steps, so that storing the mask
    in 8 bits gives fine grain instead of bands. Fades out within one step
    of 0 and 1, so fully black / white areas stay clean."""
    n = hash12(xx, yy) + hash12(xx + 37.0, yy + 91.0) - 1.0
    k = np.clip(np.minimum(b, 1.0 - b) * 255.0, 0.0, 1.0)
    return np.clip(b + n * (amount / 255.0) * k, 0.0, 1.0)


def levels(b, contrast, grow):
    """Levels after the blur. Grow (-1..1) moves the 0.5 threshold, so the mask
    grows / shrinks in 3D; Contrast (0..1) narrows the ramp around it, towards
    a hard edge, with a smoothstep. Contrast 0, Grow 0 = unchanged."""
    t = 0.5 - 0.5 * grow
    w = 0.5 * (1.0 - contrast) ** 2 + 0.002 * contrast
    lin_ = np.clip((b - t) / (2.0 * w) + 0.5, 0.0, 1.0)
    return lin_ + (lin_ * lin_ * (3.0 - 2.0 * lin_) - lin_) * contrast


def pass_resolve(mask, position, uv_mask, vox, vt, frame, coarse, boxes, amount=DITHER, shape_params=None,
                 strength=1.0):
    hgt, wid = mask.shape
    yy, xx = np.mgrid[0:hgt, 0:wid]
    u, v = (xx + 0.5) / wid, (yy + 0.5) / hgt
    p = sample_nearest(position, u, v)
    b = box_of(p, coarse, boxes)
    bb = np.maximum(b, 0)
    n = vt["dims"][bb]
    g = np.clip((p - vt["origin"][bb]) / frame["h"] - 0.5, 1.0, n - 3.0)
    i0 = np.floor(g)
    t = g - i0
    w = [bspline_weights(t[..., k]) for k in range(3)]
    s = 0.0
    for c in range(4):
        for bi in range(4):
            for a in range(4):
                wt = (w[0][a] * w[1][bi] * w[2][c])[..., None]
                q = i0 + np.array([a - 1.0, bi - 1.0, c - 1.0])
                idx = np.clip(vt["offset"][bb] + lin(q, n), 0, AW * AH - 1).astype(int)
                s = s + wt * vox[idx]
    blurred = np.clip(s[..., 0] / np.maximum(s[..., 1], 1e-12), 0.0, 1.0)
    # Smear Strength: towards smeared mass / *plain* density, so what the smear
    # drags in adds up instead of being averaged away (no change upstream of
    # the mask). Strength 1 = the plain smear.
    undiluted = np.clip(s[..., 0] / np.maximum(s[..., 3], 1e-12), 0.0, 1.0)
    blurred = blurred + (undiluted - blurred) * ((strength - 1.0) * (1.0 / 3.0))
    if shape_params is not None:
        blurred = levels(blurred, *shape_params)
    blurred = dither(blurred, xx.astype(float), yy.astype(float), amount)
    ok = (sample_nearest(uv_mask, u, v) > 0.5) & (s[..., 1] > 1e-6) & (b >= 0)
    return np.where(ok, blurred, sample_nearest(mask, u, v))


# --------------------------------------------------------------------------

def prepare(position, uv_mask):
    """Everything that depends only on the mesh (cached by Painter)."""
    bounds = pass_bounds(position, uv_mask)
    coarse = pass_coarse(bounds)
    pack_pos = pass_pack_pos(position, uv_mask)
    pack_area = pass_pack_area(position, uv_mask, pack_pos, coarse)
    hgt, wid = uv_mask.shape
    area_total = pass_area_total(pack_area)
    luv = length_per_uv(area_total, wid, hgt)
    sorted0 = pass_sort(pass_keys0(pack_pos, coarse))
    boxes = pass_boxes(cell_bounds(sorted0, pack_pos, coarse), coarse)
    c_hi = 2.0 * bounds["size"]
    hs = fit_cell(boxes["lo"], boxes["hi"], boxes["ok"], float(AW * AH), c_hi)
    st = grid_table(boxes, hs)
    _, sample_box = pass_keys1(pack_pos, coarse, boxes, st)
    return dict(bounds=bounds, coarse=coarse, pack_pos=pack_pos, pack_area=pack_area, area_total=area_total, luv=luv,
                boxes=boxes, hs=hs, sort_table=st, sample_box=sample_box, c_hi=c_hi)


def world_space_blur(mask, position, uv_mask, radius=0.02, quality=4, stats=None, dither=DITHER,
                     shape=None, smear=None, mesh=None):
    """Full filter. `radius`: blur radius in UV units (2 sigma, at the mesh's
    average texel density). `quality`: 1..4, the voxel budget (QUALITY).
    `dither`: anti-banding noise, in 8-bit steps. `shape`: None or
    (contrast, grow). `smear`: None or (length, (dx, dy, dz)[, strength]).
    `mesh`: prepare() result, to reuse it across calls."""
    m = mesh or prepare(position, uv_mask)
    boxes = m["boxes"]
    hq = fit_cell(boxes["lo"], boxes["hi"], boxes["ok"], budget_of(quality), m["c_hi"])
    frame = pass_frame(radius, quality, hq, m["hs"], m["luv"])
    vt = grid_table(boxes, frame["h"])
    pack_mask = pass_pack_mask(mask, position, uv_mask, m["pack_pos"], m["pack_area"], m["coarse"], m["area_total"])
    vox = pass_splat(m["pack_pos"], pack_mask, m["sample_box"], vt, frame["h"])
    for axis in range(3):
        vox = pass_blur(vox, vt, boxes["ok"], frame, axis, smear)
    if stats is not None:
        stats.update(frame)
        stats.update(boxes=int(boxes["ok"].sum()), voxels=float(vt["vol"].sum()), hs=m["hs"],
                     coarse=m["coarse"]["c"])
    strength = smear[2] if (smear is not None and len(smear) > 2) else 1.0
    return pass_resolve(mask, position, uv_mask, vox, vt, frame, m["coarse"], boxes, dither, shape, strength)
