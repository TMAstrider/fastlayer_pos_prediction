# Part 10013: stored Magics pose versus support-aware search

## Scope

This case compares three things for the same part:

1. The pose stored in the Magics project and the support mesh saved in that project.
2. A simplified support estimate at the same saved `+Z` build direction.
3. The lowest-support-volume direction among 264 query-free candidates on an orientation sphere.

The source project is `M300-400mm-2X2-Y.mpf_AI-0706-铝6_260910_0_V8.pjf`.
The part entity is `10013`, named `13.stl`, and its support child is `10080`,
named `s_13.stl`.

- Part mesh: `outputs/meshes_stl/parts/project_0558e01b77/part_10013__13.stl`
- Stored support mesh: `outputs/meshes_stl/supports/project_0558e01b77/part_10080__s_13.stl`
- Part mesh bounds: `[4.9183, 14.0530, 6.0000]` to `[21.6696, 140.6390, 93.2490]`
- Stored support bounds: `[-1.9978, 10.1738, 0.0000]` to `[23.8823, 146.3626, 85.3500]`

The extracted part mesh is already in the saved Magics build coordinates. Its
stored build-up direction is therefore `+Z`; this is the historical pose used
below. It is not the original CAD-frame orientation.

## What is actually in the project

The SQLite-backed PJF contains one slice plan,
`m300-alsi10mg-ai-40-20260822@alsi10mg&40`. Its stored up-surface and
down-surface angle fields are both `45`. The visible geometry has part `10013`
and support `10080` as its child. No separate user-defined support-blocking or
avoidance entity was found in the entity graph.

The absence of such an entity does not prove the shop had no verbal or external
requirement. It only means this project file does not contain an explicit
no-support object for this part.

The saved Magics support mesh has:

- 61,203 raw vertices and 20,401 triangles.
- 12,451.87 mm2 of triangle area.
- 625 disconnected components.
- No watertight boundary or consistent winding. A closed mesh volume therefore
  cannot be computed reliably and is not reported.

The mesh is a finished surface representation, not the proprietary Magics
generation process. Neither its settings nor the saved result determine how
Magics would support each alternative candidate orientation.

## Recognized holes and observed stored support

Six internal cylinders are recognized, all approximately along the input `X`
axis and all at least 7.98 mm in diameter. Under the current process rule, none
is a small-hole exemption because every diameter is greater than 6 mm.

| Feature | Diameter (mm) | Length (mm) | Support vertices inside finite cylinder | Support triangle centroids inside |
| --- | ---: | ---: | ---: | ---: |
| `internal_cylinder_0001` | 10.989 | 7.194 | 364 | 143 |
| `internal_cylinder_0002` | 10.991 | 7.196 | 378 | 147 |
| `internal_cylinder_0003` | 7.983 | 3.000 | 240 | 120 |
| `internal_cylinder_0004` | 10.996 | 3.000 | 391 | 157 |
| `internal_cylinder_0005` | 10.996 | 3.000 | 332 | 146 |
| `internal_cylinder_0006` | 10.996 | 3.000 | 360 | 176 |

This is direct geometric evidence that the stored Magics support occupies all
six recognized hole volumes. It does not establish whether that was acceptable:
a large hole can be support-allowed by default, while a user query can turn a
selected hole into a hard constraint.

## Query-free direction search

The search contains 14 axis/diagonal seeds plus 250 Fibonacci-sphere directions.
All duplicates are removed, leaving 264 candidates. Supports use a 4 mm initial
lift, 1.5 mm overhang sampling, and 0.4 mm-radius vertical-column proxies.

| Metric | Historical `+Z` | Searched `S047` |
| --- | ---: | ---: |
| Direction in input frame | `[0, 0, 1]` | `[-0.7032, -0.3333, 0.6280]` |
| Query-free feasible | Yes | Yes |
| Support columns | 253 | 84 |
| Support-volume proxy | 2,848.93 mm3 | 1,219.05 mm3 |
| Candidate overhang area | 567.77 mm2 | 188.35 mm2 |
| Build-plate landings | 133 | 82 |
| Part-surface landings | 120 | 2 |
| Mean / maximum column height | 22.40 / 58.05 mm | 28.87 / 66.99 mm |
| Rank by proxy volume | 36 / 264 | 1 / 264 |

The searched direction is 51.10 degrees away from historical `+Z`. Relative to
the historical pose, the proxy volume falls by 57.2%, the column count by 66.8%,
and the candidate overhang area by 66.8%. The lower support amount comes with a
flatter build posture and taller mean columns.

The reduction in part-surface landings, from 120 to 2, is operationally useful:
most predicted columns would reach the build plate instead of resting on the
part. This metric is only a proxy and still needs validation against Magics and
actual removability.

## When every recognized hole is forbidden

A second search treats all six recognized internal cylinders as hard no-support
regions. All 264 directions are rejected:

- Feasible directions: `0 / 264`.
- Historical `+Z`: 154 violations, split into 139 overhangs on forbidden faces
  and 15 support shafts intersecting a protected cylinder.
- Least-violating direction: `S163` with 6 violations, all
  `overhang_on_no_support_face`.
- Next least-violating direction: `S213` with 7 violations.

The current search does not have a support strategy that can form every required
overhang while keeping all six holes completely clear. A production response
would need one of:

1. Permit support in selected large holes.
2. Add process evidence that a particular hole is self-supporting.
3. Add a different support topology, such as bridging or a removable insert.
4. Report the least-violating direction with an explicit exception request.

## Deliverables

- `part_10013_comparison.png`: historical Magics support, historical simplified
  estimate, and searched best simplified estimate in aligned build views.
- `part_10013_orientation_sphere.png`: all sampled directions colored by
  support-volume proxy, with historical and best directions marked.
- `comparison_metrics.json`: machine-readable metrics, hole overlaps, and limits.
- `historical_magics/deliverables/`: extracted part, stored Magics support,
  matching plate, combined STL, and editable OBJ/MTL scene.
- `historical_simplified/support_report.json`: simplified support report at
  historical `+Z`.
- `historical_simplified/deliverables/`: simplified support and matching plate.
- `search_best/simplified/support_report.json`: simplified support report at
  `S047`.
- `search_best/simplified/deliverables/`: simplified support and matching plate.
- `../part_10013_search_sphere250_no_query/orientation_search.json`: full
  query-free candidate table.
- `../part_10013_search_sphere250_no_hole_support/orientation_search.json`: full
  all-holes-forbidden candidate table.

The combined deliverables preserve the extracted input coordinate frame. The
plate in a nonzero-build-direction scene is therefore tilted in XYZ coordinates
while remaining perpendicular to its corresponding build direction. The
comparison image transforms each panel into its own build view for visual
inspection.

## Reproduction

The query-free sphere search was run with:

```bash
python tools/search_support_orientations.py \
  outputs/meshes_stl/parts/project_0558e01b77/part_10013__13.stl \
  --material AlSi10Mg \
  --mode standard \
  --support-query reports/feature_recognition_v0/part_10013_query_none.json \
  --sphere-samples 250 \
  --lift-mm 4 \
  --output-dir reports/feature_recognition_v0/part_10013_search_sphere250_no_query
```

The same command with
`part_10013_query_all_holes.json` produces the all-holes-forbidden result.
`tools/generate_simplified_supports.py` regenerates a detailed support report
for one chosen direction, and
`tools/render_support_orientation_comparison.py` creates the two comparison
images and metrics from the saved reports. The alignment transform maps the
selected build-up vector to `+Z`; support rays are then cast along that build
axis rather than always along the input world `Z`.

## Limits

- The simplified support is a geometric vertical-column proxy. It is not Magics
  SG+, e-Stage, or validated production support.
- The proxy has no branches, lattice, contact tips, curling, thermal stress,
  recoater interaction, or removal simulation.
- Magics support could not be regenerated for `S047`; the actual-support and
  simplified-search sides of this report are different algorithms.
- The 45 degree threshold and 4 mm lift are experiment settings. They still need
  calibration against the actual machine, material build mode, layer thickness,
  and Magics support profile.
- Hole recognition covers stable cylindrical walls. Non-cylindrical channels,
  slots, sealing faces, mating surfaces, and other functional regions still
  require explicit query or annotation.
