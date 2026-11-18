# Deterministic internal-hole recognition v0

This experiment adds a geometric feature-recognition stage before support generation. The first version recognizes internal cylindrical holes and channels. It does not infer bridges, bosses, sealing faces, mating faces, or other customer-specific functional surfaces; those still require query or face annotations.

## Pipeline

1. Merge duplicated STL vertices and build face adjacency from shared mesh edges.
2. Group adjacent triangles whose normals differ by at most 50 degrees.
3. Fit a cylinder axis from each region's normal distribution.
4. Fit the cylinder center and radius from projected triangle centroids.
5. Keep inward-facing cylindrical fits, which represent hole or channel walls.
6. Reject unstable fits, external cylinders, partial arcs covering fewer than 8 of 12 angular bins, and cylinders shorter than 0.25 mm.
7. Treat recognition and support permission as separate concepts. Recognizing a hole does not automatically make it forbidden.
8. Mark horizontal internal cylinders strictly below 6 mm diameter as process exemptions, so the support generator does not add support there.
9. Resolve a structured user query into selected feature IDs or face indices. Only those selected regions become hard no-support constraints.
10. During support generation, reject a support column if its centerline intersects or enters the configured clearance volume of a query-selected cylinder.

The default horizontal tolerance is 10 degrees from the build plane. The default protected clearance beyond the hole radius plus the support radius is 0.2 mm. All dimensions are assumed to be millimetres because the source PJF unit metadata remains unconfirmed.

## Batch result

The recognizer processed all 193 extracted meshes:

- 162 parts contain at least one recognized internal cylinder.
- 1,948 internal cylinders were recognized.
- 1,114 are approximately horizontal to the current build direction.
- 1,046 satisfy the prototype rule `horizontal and diameter < 6 mm`.
- 31 parts have no accepted internal-cylinder feature. They may have no such geometry, or the feature may not be a full cylindrical surface and need annotation.

Representative checks:

- Part 10001: one horizontal internal cylinder, approximately 3.19 mm diameter and 4.82 mm long.
- Part 10040: 216 approximately 4.30 mm horizontal holes across the perforated cylindrical part.
- A 238 x 200 x 34 part in project 66: 99 accepted internal-cylinder fits, including repeated approximately 1.99 mm holes and large internal circular geometry.

The 10040 and 10009 previews were inspected visually and agree with visible internal cylindrical geometry.

An explicit large-hole query check was run on part 10013:

- Without a user query, its recognized approximately 11 mm hole remains support-allowed and the pose receives 253 support columns.
- Selecting `internal_cylinder_0001` adds its 328 faces to the hard no-support set and removes 20 columns.
- Those same 20 source locations become `overhang_on_no_support_face` violations, so the current pose must be rejected unless process rules make that hole self-supporting or another support strategy is supplied.

This demonstrates the intended contract: a large hole can receive support by default, while a user query can prohibit it and force the orientation search to find a feasible alternative.

## Part 10013 stored-pose comparison

Part 10013 was traced back to the historical Magics project
`M300-400mm-2X2-Y.mpf_AI-0706-铝6_260910_0_V8.pjf`. Part entity 10013 has
support child 10080, and the saved support mesh is present in the extracted
assets. The stored part is already in build coordinates with build-up direction
`+Z`.

Six approximately horizontal internal cylinders were recognized. Their
diameters are 7.98 to 11.00 mm, so none is removed by the current `<6 mm`
small-hole exemption. The saved Magics support geometrically occupies all six
recognized hole volumes. The project file contains no separate explicit
support-blocking entity, although that does not rule out an external process
requirement.

A 264-direction query-free sphere search found 264 feasible directions. The
historical `+Z` pose ranked 36th by simplified support-volume proxy with 253
columns and 2,848.93 mm3 proxy volume. The best direction, `S047`, is 51.10
degrees away from historical `+Z` and reduces this to 84 columns and
1,219.05 mm3, a 57.2% volume reduction. It also reduces part-surface landings
from 120 to 2, but has a taller mean column height.

A second run treated all six recognized holes as hard no-support regions. It
found no feasible direction in 264 candidates. The least-violating direction,
`S163`, still has six overhang-face violations. This case therefore requires
either selective support permission, evidence that a selected hole is
self-supporting, or a different support strategy.

The full comparison and deliverables are in
`part_10013_comparison/README.md`. The main images are:

- `part_10013_comparison/part_10013_comparison.png`
- `part_10013_comparison/part_10013_orientation_sphere.png`

## Support integration check

For part 10001 with AlSi10Mg standard rules and 1.5 mm overhang sampling:

- 27 overhang-candidate faces are removed by the small-hole exemption.
- Candidate faces fall from 216 to 189.
- Support columns fall from 2,104 to 2,099.
- Without a user query, no recognized-hole face is placed in the hard no-support set.
- The example query selects `internal_cylinder_0001`, adding its 108 faces to the hard no-support set without changing the outcome of this small-hole case.
- No source, landing, or support-shaft intersection violation remains in either run.

## Outputs

- `all_parts/batch_summary.json`: aggregate and per-part file index.
- `all_parts/part_summary.csv`: one row per part.
- `all_parts/feature_summary.csv`: one row per recognized feature.
- `all_parts/*.features.json`: face indices, axis, diameter, length, confidence, fit diagnostics, and suggested support annotations for each accepted feature.
- `part_10001.features.png`: recognition close-up for part 10001.
- `part_10040.features.png`: dense small-hole example.
- `part_10009_project66.features.png`: large and small internal-cylinder example.
- `part_10013_query.json`: example query selecting one approximately 11 mm recognized hole.
- `part_10013_support_no_query.png` and `part_10013_support_with_query.png`: default-allowed versus user-forbidden comparison.
- `part_10001_support/support_report.json`: feature-aware support diagnostics.
- `part_10001_support/support_report_no_query.json`: recognition and small-hole exemption without a user prohibition.
- `part_10001_support/support_report_with_query.json`: recognition plus an explicit no-support query for the recognized hole.
- `part_10001_support/support_preview.png`: overhang and support preview.
- `part_10001_support/deliverables/`: regenerated part, plate, and support scene files.

## Commands

Single part:

```bash
python tools/recognize_part_features.py path/to/part.stl \
  --output part.features.json \
  --preview part.features.png
```

Batch:

```bash
python tools/recognize_part_features.py \
  --input-root outputs/meshes_stl/parts \
  --output-dir reports/feature_recognition_v0/all_parts \
  --jobs 16 \
  --no-batch-previews
```

Feature-aware support generation:

```bash
python tools/generate_simplified_supports.py path/to/part.stl \
  --material AlSi10Mg \
  --mode standard \
  --auto-features \
  --support-query path/to/support_query.json \
  --feature-output part.features.json \
  --output part.support.json \
  --support-obj part.supports.obj
```

## Limitations

- Recognition is based on triangle winding. Meshes with inward winding or inconsistent normals can produce incorrect internal/external classification.
- A cylindrical CAD fillet can have the same local geometry as a partial bore. The angular-coverage and inward-facing checks reduce these cases but do not eliminate them.
- Non-cylindrical channels, threaded holes, slots, bridges, bosses, and customer-defined functional surfaces need explicit annotations.
- Natural-language text is preserved as query metadata, but this version requires the UI or an upstream parser to resolve the request into feature IDs, face indices, or supported filters.
- Protected support-shaft checks model recognized holes as finite cylinders plus clearance. Cone countersinks and non-cylindrical transitions are not yet modelled.
- The generated supports remain simplified vertical-column proxies. They are not validated production supports or Materialise Magics output.
