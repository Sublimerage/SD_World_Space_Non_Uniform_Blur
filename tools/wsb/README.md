# Vendored from World Space Mask Blur v2.0

Unmodified copies from `Sublimerage/my-project`, branch
`claude/substance-painter-advanced-filters-d7wbo5`, commit `ef8f9ed` ("Speed-ups that change no pixel"; the v2.0 passes are unchanged from `b0a3582`):

| File | Source |
|---|---|
| `build_sbs.py` | `designer/build_sbs.py`: expression DSL, `.sbs` emitter, and the v2.0 passes |
| `world_space_blur.py` | `reference/world_space_blur.py`: constants and reference algorithm |
| `layout_mesh.py`, `demo_box.py` | `reference/`: kit-layout and box test scenes |

`tools/build_wsnub.py` reuses the mesh-only voxelization of v2.0 from here as is
(bounds, coarse grid, samples, area totals, both sorts, slabs / runs / boxes,
cell-size fits, voxel map), so it stays identical to the field-tested mask blur.
Update by copying the files again; don't edit them here.
