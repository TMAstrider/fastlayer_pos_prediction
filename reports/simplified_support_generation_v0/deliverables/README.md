# Part and simplified support deliverables

This sample uses historical part 10001. The part is kept at its source orientation, with build direction `+Z`. The build plate top is at `Z=0` and the plate is 2 mm thick. Source units are assumed to be millimetres; the original PJF unit metadata has not been confirmed.

## Files

- `part_original.stl`: original part mesh, unchanged and without a plate.
- `part_original_with_platform.obj`: original part and matching build plate as separate OBJ objects.
- `part_with_simplified_supports.stl`: original part with generated support-column proxies, without a plate.
- `build_plate.stl`: matching plate as a separate mesh.
- `part_with_simplified_supports_and_platform.stl`: part, support proxies, and plate in one STL.
- `part_with_simplified_supports_and_platform.obj`: editable scene with separate `Part`, `SimplifiedSupportColumns`, and `BuildPlate` objects; the accompanying MTL assigns display colours.
- `original_vs_supported.png`: side-by-side preview.
- `artifact_manifest.json`: dimensions, assumptions, and filenames.

The simplified generator produced 2,104 vertical support-column proxies with an assumed 0.4 mm radius. These are geometric proxies from the current approximation, not Materialise Magics supports and not validated manufacturing supports. The combined STL is convenient for viewing; use the OBJ scene to retain separate objects, or the part-only and plate-only STLs when importing components separately.
