# Feature-aware support deliverable for part 10001

This scene uses the same part, build direction, platform, support sampling, and 0.4 mm column-radius assumption as the earlier simplified-support deliverable. It also includes automatic recognition of one internal cylindrical hole.

- Recognized hole diameter: approximately 3.19 mm.
- Recognized hole axis length: approximately 4.82 mm.
- Candidate overhang faces removed by the small-hole exemption: 27.
- Remaining support columns: 2,099.
- The accompanying query explicitly selects the recognized hole and adds its 108 faces to the hard no-support set.
- Hard constraint violations: 0.

The platform top is at `Z=0`; the plate is 2 mm thick. Source units are assumed to be millimetres. Support meshes are geometric vertical-column proxies, not validated manufacturing supports and not Materialise Magics output.
