# World Space Non-Uniform Blur Lite: design

A cheaper companion to World Space Non-Uniform Blur v2.0. It runs the same 2D Non-Uniform Blur passes on
six axis projections of the mesh (+X, −X, +Y, −Y, +Z, −Z), not in voxels, then blends them back onto each
texel by its normal (triplanar). The projection lookups come from the shared Lite core (World Space Mask
Blur session). This note covers the part specific to Non-Uniform Blur: how the bokeh passes, the blur
map and the Reference Axis carry over to the projections.

`tools/lite_proto.py` is a numpy prototype of the design. Its projection lookups are a stand-in for
the core. `tools/lite_compare.py` compares it with v2.0, which runs through the interpreter on the
same scenes as `tools/bench_v2.py`.

## The passes, per projection pixel

Each pixel of a projection is the front-most surface point facing that direction. Each pass does what
the 2D graph and v2.0 do:

* Passes: N = min(Samples, ⌈Intensity·π⌉). Pass k has radius r·exp(−2k·√(−ln 0.001)/N) and an inner
  rotation of (1/Samples)/(k+1) turns. It takes one centre tap plus one tap at each Blades corner, then
  averages them.
* Radius: Intensity in the same units as v2.0 (1 = 1/256 of the UV space at the mesh's average
  length per UV), times the blur amount read at the centre pixel.

**Ellipse: built in 3D, then projected (this is how Anisotropy, Angle and Reference Axis carry over).**
Each pixel's tangent frame comes from its own world normal, which is gathered once per mesh, exactly as
in v2.0:

* t1 is the Reference Axis projected on the surface. When the axis is close to the normal, t1 falls
  back to X (or to Z).
* t2 = n × t1.
* The Anisotropy / Angle ellipse is (t1·X + t2·Y)·r in world space.

That world offset is then projected into the view. The 2D tap offset is the offset's two in-plane
components divided by the pixel size; the third component is the depth change a flat surface would have
there.

On a flat area this lands on the same surface points as v2.0's world-space taps. So every projection
sees the same bokeh, and where the triplanar blend mixes two views, their shapes agree.

The alternative is one fixed 2D frame per view: the Reference Axis projected on the view plane, which
is what Designer's own node would allow. It is right only on faces square to the view. On the sphere it
is further from v2.0: a mean difference of 0.038 (0.042 with the depth test) against 0.026 for the
3D frame, where the blur itself changes the texture by 0.246 on average.

On the plane the prototype reproduces the 2D graph's orientation: Angle 0.15 with Anisotropy 0.9 gives
the same ellipse as the 2D Non-Uniform Blur, and the mirrored or rotated variants do not.

**Depth test.** A tap counts only when the depth at its pixel is within 0.5·r + 2 pixels of what a flat
surface would have there. This stops the blur jumping across a silhouette onto a farther surface, such
as a stacked piece or the far wall of a cavity. A rejected tap drops out of the average, which is
renormalised, the same way v2.0 handles empty voxels.

**Blur map.** It is gathered into the projections together with the input and read at the centre pixel
in every pass, as the 2D graph reads its Effect map. "Use Input As Blur Map" uses the luminance of the
gathered input, so it needs no extra gather.

**Resolve, per texel.**

* For each axis, take the view on the side the normal faces, weighted by |n_axis|⁴.
* Use a view only where the texel is front-most in it: its depth is within 2.5 pixels of the view's
  depth there.
* Blend from the input to the blurred result by coverage (a smoothstep from 0.3 to 0.8 of the total
  weight). Also fade the blur in over the first two pixels of radius, as v2.0 does, so a tiny Intensity
  returns the input exactly. Without that fade, projecting into the views and back softens hard edges
  by about 0.01 on its own.

## Textures and reads (what the .sbs will do)

| Texture | Made | Size | Contents |
|---|---|---|---|
| geometry | once per mesh | per view S×S | (depth·cov, cov, n_b, n_c). n_a is rebuilt from the sign of the view. |
| colour (colour graph) | per stroke and per pass | per view S×S | (R·cov, G·cov, B·cov, blur amount) |
| value (grayscale graph) | per stroke and per pass | per view S×S | (v·cov, cov, depth·cov, blur amount): one read per tap |

* Colour graph: one pass costs 2 + 2·Blades reads per covered pixel and 1 per empty pixel, which reads
  only coverage and stops.
* Taps are bilinear. Colour and depth are premultiplied by coverage, so a tap that straddles an edge is
  weighted correctly.
* S defaults to half the input size; a Projection Size parameter will offer 1/4, 1/2 and 1×.
* If the core packs the views in one atlas, each pass is a single pixel processor. Taps that leave a
  tile land on its empty one-pixel border, so they are rejected.
* Pass slots: 16, as in v2.0, skipped when inactive.

## Designer's own Non-Uniform Blur node: not used

A compiled filter can include the library Non-Uniform Blur node; library nodes are embedded when the
.sbsar is cooked. I'd still not use it, for three reasons:

* Its Angle is one value for the whole image, but each projection needs the ellipse per pixel (the
  surface tilts).
* It can't do the depth test, so blur bleeds across silhouettes and between stacked pieces.
* It wouldn't match v2.0 and the 2D graph tap for tap. The user's own 2D filter
  (`non_uniform_blur_sample` chained per pass) is what v2.0 reproduces, and what Lite reproduces.

Needs a Designer check if we want it as a "fastest" mode: that the node cooks into a Painter filter
and what it costs.

## Parameters

Same as v2.0: Samples, Intensity, Anisotropy, Blades, Angle, Use Blur Map, Use Input As Blur Map, Invert
Blur Map, Reference Axis X/Y/Z, Dither.

* Dropped: Softness (no mip pyramid; taps are bilinear like the 2D graph) and Quality (no voxels).
* New: Projection Size (1/4, 1/2 or 1× the input size, default 1/2).

## What I need from the core

1. Six front-most lookups, the per-pixel depth, the pixel size h and the origin, so world offsets
   can be converted to pixels.
2. A gather entry point that takes any texture: the input, the blur map and the world normal.
3. The resolve's visibility test and coverage, so the blurred views can be handed back.
4. Pieces laid apart: if the views cover the whole bounding box, a spread-out kit layout gets few
   pixels per piece. This is v1.0's old problem (see the layout row below). Per-piece views, or
   tiling them by piece, would fix it.

## Prototype results against v2.0

The scenes and settings are those of `tools/bench_v2.py`: 512² maps, Intensity 10, Samples 4, v2.0 at
Quality 4. Reads are scaled to a 2048² texture.

* v2.0's reads are measured by the interpreter.
* Lite's reads are an estimate from the design above (the prototype counts the reads each pixel
  processor would make). They don't include the core's per-stroke gather lookups beyond one read per
  view pixel, or node evaluations. The real numbers come once the .sbs exists.

| Scene | v2.0 reads / stroke | Lite ½ size | Lite ¼ size | Lite 1× | Mean diff to v2.0 (½) | Blur change by v2.0 |
|---|---|---|---|---|---|---|
| sphere | 1.18 G | 0.31 G (3.8× less) | 0.09 G (13×) | 1.09 G (1.1×) | 0.026 | 0.246 |
| kit layout | 1.35 G | 0.07 G (20×) | 0.03 G (43×) | 0.21 G (6.4×) | 0.012 | 0.055 |
| plane | 1.99 G | 0.14 G (15×) | 0.06 G (32×) | 0.43 G (4.6×) | 0.011 | 0.050 |
| thin plank, Intensity 10 | 1.24 G | 0.08 G (16×) | 0.03 G (37×) | 0.24 G (5.1×) | 0.007 | 0.020 |

* The cost tracks how much of each view the mesh fills. A sphere fills most of every view, so it
  gains least. Spread-out pieces fill little, so they gain most.
* Once per mesh, Lite's own part is small: it gathers depth and normal into the six views, about 0.02 G
  reads at ½ size (6 views × 1024² × 3 reads). Building the lookups is the core's cost. v2.0's once-per-mesh work is 0.86–1.36 G
  reads.

**Thin plank (2.5 × 0.12 × 0.25), dots painted on the top face only.** Neither version leaks to the
bottom face at Intensity 10 (both 0.0000). At Intensity 30, v2.0 leaks a trace (mean 0.0003, max
0.0013) and Lite leaks none: each texel is only ever read from the views on the side it faces. The
top face changes about the same amount in both (0.094 for Lite, 0.090 for v2.0).

**Depth test.** For a cube floating 0.5 above a slab, without the test the cube's top bleeds onto the
slab around it (mean 0.06–0.13, max 0.34). With it, the bleed is at most 0.0007.

`docs/images/lite_vs_v2.png` shows, per scene, the input, v2.0, Lite at 1×, ½ and ¼, and |Lite ½ − v2.0| × 4.

### Found on the way: v2.0's Angle mirrors on some UV islands

v2.0 gets its normals from the position map and orients them by UV winding. On islands whose winding
points inward, the tangent frame flips, so Anisotropy with an Angle gives a mirrored ellipse
(Angle → −Angle). On the test sphere, 3 of the 6 islands are like this. On every face, Lite matches v2.0
with one sign of the Angle or the other (difference 0.015–0.03).

Lite takes the outward normal, so it doesn't have this issue. In v2.0 it shows up on meshes with
mirrored UV islands. It's not fixed here.

## Known failure cases (for the product description)

1. **Hidden surfaces aren't blurred.** A surface no axis view can see is left as it is: the faces
   between stacked or touching pieces, the inside of cavities, and the hidden parts of overlapping
   pieces. In a stack of four slabs with gaps, the top faces of the lower three and the bottom faces
   of the upper three keep the input exactly. Laying pieces out apart avoids it. (v2.0 blurs these.)
2. **No blur around hard edges.** Each face blurs within itself, so the blur doesn't wrap around a
   corner onto the next face. On faces narrower than the blur (plank ends, thin bevels) Lite keeps the
   face's own colours where v2.0 mixes in the neighbouring faces. In the kit layout, the largest
   differences (about 0.3) are on such faces.
3. **Fine detail at small Intensity.** The views span the whole bounding box. Detail smaller than one
   view pixel (½ size: about 1/1000 of the object's largest dimension at 2K) is softened in the
   blurred areas. Spread-out kit layouts lose most, as v1.0 did, unless the core lays the views out
   per piece.
4. **45° areas.** Where two views blend, the bokeh can look slightly doubled on strongly curved
   surfaces (both views agree only on flat areas).
5. **Concave detail facing away from all six axes**, such as deep grooves at an angle, gets little
   coverage and stays close to the input.
