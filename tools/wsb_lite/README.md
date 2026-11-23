# The Lite core, vendored from World Space Mask Blur Lite

World Space Non-Uniform Blur Lite (`tools/build_wsnub_lite.py`) is built on the shared Lite
core: each piece seen from the 6 axis directions, and per fine view pixel the front-most
surface point (see `docs/lite-core.md` in the source repository). These files come from
`Sublimerage/my-project`:

| File | Source |
|---|---|
| `build_lite.py` | `designer/build_lite.py` at `015857f` (branch `claude/world-space-mask-blur-lite`) |
| `world_space_blur_lite.py` | `reference/world_space_blur_lite.py` at `015857f` (its NumPy spec) |
| `build_sbs.py`, `world_space_blur.py` | `designer/build_sbs.py`, `reference/world_space_blur.py` at `015857f` |
| `layout_mesh.py`, `demo_box.py` | `reference/` at `744426a` (adds the door-frame scenes) |

Two changes on top of `015857f`, both the World Space Mask Blur v2.1 stretched-island fix
(a UV island much longer in 3D than in UV, like a door-frame post in a square island):

1. `build_sbs.py` and `world_space_blur.py`: the fix itself, `git diff ef8f9ed 615af99` of those
   two files, applied as is (texel differences fall back to the nearer in-island neighbour;
   per-sample reject distance in `WSB_PackArea.z`).
2. `build_lite.py` and `world_space_blur_lite.py`, `pass_frame_samples` (`WSB_LFrame`): the same
   fallback for the samples' tangent frames. Without it a sample whose neighbours are all farther
   than the fixed reject distance is unusable, so it never reaches a view: on the door frame at
   512², half the texels had no view and the posts stayed unblurred (after: 5 %, the faces the
   posts hide from each other).

`tools/test_lite.py` checks the graph against the spec with these changes, a door frame with
stretched islands included.
