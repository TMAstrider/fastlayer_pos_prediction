"""Describe the human chosen printing poses and compare objective weightings.

Reads the historical placements (each shape is stored in its printing pose, so the
human up direction is +Z) and reports, per shape, the quantities a placement rule
would care about: remaining overhang, build height, plate contact, whether a large
flat face is parallel to the plate, and how the human choice ranks under several
objective weightings.

No training, no data modification.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import score_build_direction_prior as core

ROOT = core.ROOT
REPORT = ROOT / 'reports/build_direction_physics_prior_2026-11-10'
FACET_AREA = .01
FLAT_DEG = 5.
PERTURB_RADII = (3., 5., 10., 15., 30.)
PERTURB_AZIMUTHS = 8
WEIGHT_SETS = {
    'support_only': [0., 1., 0., 0.],
    'overhang_only': [1., 0., 0., 0.],
    'overhang_support': [1., 1., 0., 0.],
    'overhang_support_contact': [1., 1., 0., 1.],
    'overhang_support_height': [1., 1., 1., 0.],
    'contact_only': [0., 0., 0., 1.],
}


def facet_tilts(points, normals, directions):
    labels = np.argmax(normals @ directions.T, 1)
    counts = np.bincount(labels, minlength=len(directions))
    tilts = []
    for label in np.nonzero(counts / len(labels) >= FACET_AREA)[0]:
        mean_normal = normals[labels == label].mean(0)
        length = np.linalg.norm(mean_normal)
        if length > 1e-6:
            tilts.append(float(np.degrees(np.arccos(np.clip(mean_normal @ -core.HUMAN / length, -1, 1)))))
    return len(tilts), sorted(tilts)


def perturb(points, normals, weight, centre, scale, radius_deg, overhang_deg, azimuths=PERTURB_AZIMUTHS):
    """Cost of directions tilted by radius_deg away from the human direction."""
    reference = np.cross(core.HUMAN, np.array([1., 0., 0.], dtype=np.float32))
    if np.linalg.norm(reference) < 1e-6:
        reference = np.cross(core.HUMAN, np.array([0., 1., 0.], dtype=np.float32))
    reference = reference / np.linalg.norm(reference)
    second = np.cross(core.HUMAN, reference)
    radius = np.deg2rad(radius_deg)
    costs, ranks = [], []
    for azimuth in np.linspace(0, 2*np.pi, azimuths, endpoint=False):
        direction = (np.cos(radius)*core.HUMAN
                     + np.sin(radius)*(np.cos(azimuth)*reference + np.sin(azimuth)*second)).astype(np.float32)
        value = (core.raw_terms_dir(points, normals, direction, overhang_deg) - centre) / scale
        costs.append(float(value @ weight))
    return np.array(costs)


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p10': float(np.percentile(values, 10)), 'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)), 'p90': float(np.percentile(values, 90)),
            'min': float(values.min()), 'max': float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directions', type=int, default=512)
    parser.add_argument('--overhang', type=float, default=40.)
    parser.add_argument('--report', type=Path, default=REPORT / 'report.json')
    arguments = parser.parse_args()
    reference = json.loads(arguments.report.read_text(encoding='utf8'))['results'][str(arguments.overhang)]
    weight = np.array([reference['weights'][name] for name in core.NAMES], dtype=np.float32)
    directions = core.sphere_directions(arguments.directions)
    shapes, split, flipped = core.load(0)
    validation = np.array(split['validation_indices'])
    grids, humans, rows = [], [], []
    for points, normals in shapes:
        raw = core.raw_terms(points, normals, directions, arguments.overhang)
        human_raw = core.raw_terms_dir(points, normals, core.HUMAN, arguments.overhang)
        centre, scale = raw.mean(0), np.maximum(raw.std(0), 1e-6)
        grid, human = (raw - centre) / scale, (human_raw - centre) / scale
        grids.append(grid)
        humans.append(human)
        facets, tilts = facet_tilts(points, normals, directions)
        height = float(points.max(0) @ core.HUMAN - points.min(0) @ core.HUMAN)
        diagonal = max(float(np.linalg.norm(points.max(0) - points.min(0))), 1e-6)
        ordered = np.argsort(grid @ weight)
        picks = {}
        for k in (16, 32, 51, 64, 128):
            picks[f'random_pick_within_top{k}_deg'] = float(np.degrees(np.arccos(
                np.clip(directions[ordered[:k]] @ core.HUMAN, -1, 1))).mean())
        human_cost = float(human @ weight)
        grid_cost = grid @ weight
        spread = max(float(grid_cost.std()), 1e-6)
        perturbed = {}
        for radius in PERTURB_RADII:
            costs = perturb(points, normals, weight, centre, scale, radius, arguments.overhang)
            perturbed[f'{radius:g}deg'] = {
                'cost_delta_sigma_median': float(np.median(np.abs(costs - human_cost)) / spread),
                'percentile_median': float(np.median([(grid_cost < c).mean() for c in costs])),
                'stay_top10_fraction': float(np.mean([(grid_cost < c).mean() <= .1 for c in costs])),
            }
        best = int(np.argmin(grid @ weight))
        rows.append({
            'cost_percentile': float((grid @ weight < human @ weight).mean()),
            'contact_fraction': float(-human_raw[3]),
            'contact_fraction_argmin': float(-raw[best, 3]),
            'overhang_fraction': float(human_raw[0]),
            'overhang_best': float(raw[:, 0].min()),
            'overhang_regret': float(human_raw[0] - raw[:, 0].min()),
            'support_fraction': float(human_raw[1]),
            'support_best': float(raw[:, 1].min()),
            'height_mm': height, 'height_ratio': height / diagonal,
            'large_facet_count': facets,
            'flat_face_tilt_deg': tilts[0] if tilts else 180.,
            'has_flat_contact_face': bool(tilts and tilts[0] <= FLAT_DEG),
            **picks,
            'perturbation': perturbed,
        })
    grids, humans = np.stack(grids), np.stack(humans)
    comparison = {}
    for name, vector in WEIGHT_SETS.items():
        vector = np.array(vector, dtype=np.float32)
        values = (grids @ vector < (humans @ vector)[:, None]).mean(1)
        comparison[name] = {'all_shapes': stats(values), 'validation': stats(values[validation]),
                            'all_top10_fraction': float((values <= .1).mean()),
                            'validation_top10_fraction': float((values[validation] <= .1).mean())}
    keys = ['cost_percentile', 'contact_fraction', 'contact_fraction_argmin', 'overhang_fraction',
            'overhang_best', 'overhang_regret', 'support_fraction', 'support_best', 'height_mm',
            'height_ratio', 'large_facet_count', 'flat_face_tilt_deg',
            'random_pick_within_top16_deg', 'random_pick_within_top32_deg',
            'random_pick_within_top51_deg', 'random_pick_within_top64_deg',
            'random_pick_within_top128_deg']
    summary = {key: stats([row[key] for row in rows]) for key in keys}
    summary['has_flat_contact_face_fraction'] = float(np.mean([r['has_flat_contact_face'] for r in rows]))
    summary['argmin_contact_above_5pct_fraction'] = float(np.mean([r['contact_fraction_argmin'] > .05 for r in rows]))
    summary['human_contact_above_5pct_fraction'] = float(np.mean([r['contact_fraction'] > .05 for r in rows]))
    summary['human_overhang_above_5pct_fraction'] = float(np.mean([r['overhang_fraction'] > .05 for r in rows]))
    summary['human_overhang_above_10pct_fraction'] = float(np.mean([r['overhang_fraction'] > .10 for r in rows]))
    summary['overhang_regret_above_5pct_fraction'] = float(np.mean([r['overhang_regret'] > .05 for r in rows]))
    summary['perturbation'] = {key: {measure: float(np.median([r['perturbation'][key][measure] for r in rows]))
                                     if measure != 'stay_top10_fraction' else
                                     float(np.mean([r['perturbation'][key][measure] for r in rows]))
                                     for measure in rows[0]['perturbation'][key]}
                               for key in rows[0]['perturbation']}
    document = {
        'purpose': 'Characterise the human printing poses and compare objective weightings.',
        'overhang_deg': arguments.overhang,
        'reference_weights': dict(zip(core.NAMES, [float(v) for v in weight])),
        'weight_set_comparison': comparison,
        'facet_area_threshold': FACET_AREA, 'flat_face_threshold_deg': FLAT_DEG,
        'shapes': len(rows), 'validation_shapes': len(validation),
        'normals_flipped_by_signed_volume': flipped,
        'summary_all_shapes': summary,
        'per_shape': rows,
        'limitations': [
            'Flat facet detection clusters sampled normals; it is not mesh face grouping.',
            'Overhang uses the 40 deg rule on area fractions, not sliced support volumes.',
            'First layer contact is measured against the lowest 2 mm of the part, which in these '
            'projects sits about 4 mm above the plate on supports, so contact is a weak criterion.',
            'Narrow holes or channels are not measured; the 4096 point pool cannot resolve them.',
            'Height is the extent along the chosen direction, not build time.',
        ],
    }
    (REPORT / 'placement_characteristics.json').write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=1))
    print(json.dumps({name: {k: round(v, 3) for k, v in values.items() if not isinstance(v, dict)}
                      for name, values in comparison.items()}, indent=1))
    print(f'wrote {REPORT / "placement_characteristics.json"}')


if __name__ == '__main__':
    main()
