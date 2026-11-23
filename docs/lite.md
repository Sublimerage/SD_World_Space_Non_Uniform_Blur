# World Space Non-Uniform Blur Lite v1.0

A cheaper companion to World Space Non-Uniform Blur: the same bokeh blur, in 3D so it's
continuous across UV seams, but done on **6 axis views of each piece** instead of voxels.
`world_space_non_uniform_blur_lite_v1.0.sbs` holds two graphs, like v2:

| Graph | Use |
|---|---|
| `world_space_non_uniform_blur_lite` | colour input / output |
| `world_space_non_uniform_blur_lite_grayscale` | grayscale input / output (masks) |

Same inputs as v2 (Input, Blur Map, Mesh Position, Mesh UV Mask). The icon is v2's for now.

**Use it when** the mesh is made of separate pieces whose faces can all be seen from outside
along some axis, and painting with v2 feels slow. **Use v2 when** a piece hides some of its own
visible faces from all 6 directions (the inner faces of a door frame, an L-bracket's inside
corner, between legs and seat of a chair) or when you change the mesh or Quality often: see
[When Lite fails](#when-lite-fails) and the costs below.

## Parameters

Same as v2: Samples, Intensity, Anisotropy, Blades, Angle, Use Blur Map, Use Input As Blur
Map, Invert Blur Map, Reference Axis X / Y / Z, Dither, plus:

* **Quality** (1–4): the view resolution, shared by all pieces (⅛, ¼, ½ or all of a 2048 ×
  2048 budget). Lower is faster when the mesh changes and loses fine detail at small Intensity.

v2's Softness isn't there: taps are plain bilinear reads, like the 2D Non-Uniform Blur's.

## How it works

The mesh part is the **shared Lite core** of World Space Mask Blur Lite, vendored in
`tools/wsb_lite/` (see its README; with the v2.1 stretched-island fix applied). Each piece (v2's
boxes) is seen from the 6 axis directions; every surface sample belongs to the view along its
dominant normal axis; per fine view pixel the core finds the front-most surface point (its UV
and depth). Per stroke (`tools/build_wsnub_lite.py`, spec `tools/lite_ref.py`):

1. **Level**: the view level L ≥ 1 (pixel = 2^L fine pixels) where the largest radius is at most
   16 pixels, so big blurs work on few pixels. Intensity is in the 2D filter's units (1 = 1/256
   of the UV space at the mesh's average texel density), as in v2.
2. **Fill**: per level pixel, the input colour, the Blur Map and the depth of the front surface
   of its 2^L × 2^L fine pixels (area weighted; the core's fill rule).
3. **Geometry**: the surface normal from the depth's slope between neighbouring pixels. It faces
   the view by construction, so it doesn't depend on UV winding.
4. **Passes** (up to 16): the 2D Non-Uniform Blur's passes, per level pixel: N = min(Samples,
   ⌈Intensity·π⌉) passes, radius decaying by exp(−2k·√(−ln 0.001)/N), inner rotation
   (1/Samples)/(k+1) turns, a centre tap and one tap per Blades corner. The ellipse (Anisotropy
   across the Angle direction, from the Reference Axis projected on the surface) is v2's, built
   in the surface's tangent plane and projected into the view, so every view taps the same
   surface points. A tap reads the 4 pixels around it; a pixel counts only if its depth is within
   0.5·r + 2 pixels of where a flat surface would be, so the blur doesn't jump across a silhouette
   onto a farther surface.
5. **Resolve** (texture size): each texel adds the 6 views of its piece at its projection, each
   pixel weighted by exp(−Δdepth² / 2p²), so only views where its surface is are used, and blends
   them back over the input by coverage and by the radius in pixels (tiny blurs keep the input).

Inactive pass slots cost one read per table pixel; the rest of a pass's setup happens inside its
tap loop, which only covered pixels enter.

## Cost at 2048²

`tools/bench_lite.py` (Lite) and `tools/bench_v2.py` (v2.1), the same scenes and settings: Quality 4,
Intensity 10, Samples 4. Texture reads and node evaluations as executed (loops at their real
length, both sides of every If/Else): upper bounds, not timings.

| Scene | Per stroke: reads, v2.1 → Lite | node evals | Once per mesh or Quality: reads | node evals |
|---|---|---|---|---|
| Sphere | 1.13 → **0.37 G** (3.1× less) | 34.7 → **13.5 G** (2.6× less) | 1.02 → 4.3 G (4.2× more) | 29 → 171 G |
| Kit layout, 4 pieces | 1.27 → **0.40 G** (3.2× less) | 41.2 → **14.7 G** (2.8× less) | 1.16 → 13.2 G (11× more) | 33 → 544 G |
| Flat plane | 1.84 → **0.52 G** (3.5× less) | 66.1 → **19.5 G** (3.4× less) | 1.52 → 2.3 G (1.5× more) | 41 → 86 G |
| Thin plank | 1.24 → **0.40 G** (3.1× less, v2.0) | – → 14.6 G | 1.00 → 14.0 G (14× more, v2.0) | – → 576 G |

* **Per stroke** = what reruns when you paint under the filter or move a Blur slider. Most of
  Lite's is the resolve (6 views × 4 pixels per texel) and the two fills; all 16 pass slots
  together are about a fifth. The level, view table and map (about 0.06 G) only rerun when
  Intensity changes, not while painting.
* **Once per mesh or Quality** is **1.5–14× v2.1's**, and 65–94 % of it is the core's lookup
  (`WSB_LLookup`: every fine view pixel tests the surface patches around it). It's highest with
  thin or spread-out pieces (12–13 G reads in the lookup on the kit layout and the plank). Painter
  caches it, so it doesn't slow painting down, but changing the mesh, the bake or Quality takes
  noticeably longer than with v2. Quality 3 uses half the view pixels, so it should roughly halve
  this (not measured).

## Quality against v2.1

Mean difference over the UV islands (colour 0–1), same scenes. Lite's own numbers come from the
graph, except the thin plank at Intensity 30 (from the spec). The plank is 2.5 × 0.12 × 0.25, with dots
painted on its top face only; v2.1 wasn't run on it, so its row compares with v2.0.

| Scene | Lite vs v2.1 | How much the blur changes the input (v2.1) |
|---|---|---|
| Sphere | 0.029 | 0.245 |
| Kit layout, 4 pieces | 0.012 | 0.055 |
| Flat plane | 0.004 | 0.049 |
| Thin plank, Intensity 10 | 0.007 (v2.0) | 0.020 |
| Thin plank, Intensity 30 | 0.011 (v2.0) | 0.020 |

* On the plane Lite is 0.005 from the 2D Non-Uniform Blur (v2.1: 0.0013): the views are resampled
  twice (fill, resolve).
* **Thin plank**: no leak to the bottom face at Intensity 10 or 30 (max 0.0001; v2.0 at 30: max
  0.0013). The blur wraps a little onto the side faces, like v2's.
* **Separate pieces** never mix: a cube floating 0.5 above a slab doesn't bleed onto it (max
  0.0000), and slabs stacked with gaps are each blurred on every face.
* **Stretched UV islands** (a door frame with every face in an equal square island): every post
  texel is blurred, 0.011 from the proportionally unwrapped door.
* **Bokeh orientation**: Lite turns Angle the same way on every island. v2 (like the 2D filter,
  which works in UV) mirrors it on mirrored UV islands. With Anisotropy and an Angle the two can
  differ there.

## When Lite fails

1. **Faces a piece hides from itself aren't blurred.** Each view keeps only the front surface. A
   face that no axis view of its own piece can see keeps the input: the inner faces of a door
   frame's posts (5–12 % of its texels), the inside corner of an L-bracket, between a chair's
   legs. Pieces that touch share a piece's views (the faces pressed together keep the input,
   which is fine since they're hidden). Separate pieces, even stacked, are fine.
2. **Edges and corners**: the blur crosses an edge onto the next face only within about one view
   pixel of depth, so on faces narrower than the blur (plank ends, bevels) Lite keeps more of the
   face's own colour than v2. These are the largest differences in the kit layout.
3. **Curved surfaces at a large Intensity**: where two views hand over, the bokeh is a little
   different from v2's (the sphere's mean difference is 0.03).
4. **Very small Intensity**: the finest view level is 2 fine pixels; below about 2 level pixels of
   radius the blur fades out and the input shows through.
5. **Thin walls** thinner than about one view pixel at the stroke's level can blur through, like
   any 3D blur.

## Testing

```
python3 tools/test_lite.py       # the graph against tools/lite_ref.py: 31 cases, view levels 1-4
python3 tools/bench_lite.py world_space_non_uniform_blur_lite_v1.0.sbs sphere out.npz   # cost
python3 tools/lite_compare.py --v2 <v2.sbs> sphere <dir>   # v2's reference output for a scene
```

`tools/test_lite.py` builds the graph at reduced table sizes (views 64², 256² and 1024²) and runs it
in `tools/sbsinterp.py` on a box, a sphere, a kit layout, stacked slabs and a door frame with
stretched UV islands, with every option: every case matches the spec to 1.3 × 10⁻⁵. At full size
the graph matches the spec to 5 × 10⁻⁵ on the plane, the kit layout and the plank; on the sphere
60 of 86,400 texels differ by more than 0.001 (max 0.017): float32 against float64 on either side
of a depth test.

Not tested inside Substance Designer or Painter yet. Worth checking there: that the per-stroke
view tables (2048 × 1024, 32-bit) don't use too much memory, and the real timings.

## History

Before the core existed, `tools/lite_proto.py` tested the idea with stand-in projections covering
the whole bounding box (no per-piece views): `tools/lite_compare.py` and
`docs/images/lite_vs_v2.png` are from that prototype. It settled the ellipse question: a fixed 2D
angle per view (all Designer's own Non-Uniform Blur node would allow) was further from v2 on the
sphere (0.038) than the ellipse built in 3D (0.026), and without the depth test a cube floating
above a slab bled onto it (max 0.34).
