"""W&B metrics only: credentials live in the user's WSL login, outside the repo."""

import argparse
import json
import os
from pathlib import Path


def start_run(output, config, *, name=None, mode=None, tags=()):
    mode = mode or os.environ.get('WANDB_MODE', 'disabled')
    if mode == 'disabled':
        return None
    import wandb

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'fastlayer-pos-prediction'),
        entity=os.environ.get('WANDB_ENTITY') or None,
        name=name or output.name,
        config=config,
        dir=str(output),
        mode=mode,
        tags=list(tags),
        save_code=False,
        settings=wandb.Settings(disable_git=True, x_disable_stats=True),
    )
    run.define_metric('epoch')
    run.define_metric('*', step_metric='epoch')
    (output / 'wandb_run.json').write_text(json.dumps({
        'id': run.id, 'url': run.url, 'project': run.project, 'entity': run.entity,
    }, indent=2), encoding='utf8')
    return run


def flatten(values, prefix=''):
    result = {}
    for key, value in values.items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = value
    return result


def log_epoch(run, row):
    if run is None:
        return
    names = {
        'online_train_loss': 'train/cosine_loss',
        'train_cosine_loss': 'train/cosine_loss',
        'fixed_probe': 'same_shape_probe',
        'recent_training_views_eval': 'recent_train_eval',
        'train_eval': 'fixed_train_eval',
        'validation': 'unseen_parts_validation',
    }
    run.log(flatten({names.get(key, key): value for key, value in row.items()}))


def finish_run(run, result):
    if run is None:
        return
    run.summary.update(flatten({key: value for key, value in result.items()
                                if key not in ('history', 'per_part_independent_rotations')}))
    history = result.get('history', [])
    if history:
        for source in ('online_train_loss', 'train_cosine_loss'):
            rows = [row for row in history if source in row]
            if rows:
                best = min(rows, key=lambda row: row[source])
                run.summary['min_recorded_train_loss'] = best[source]
                run.summary['min_recorded_train_loss_epoch'] = best['epoch']
    per_part = result.get('per_part_independent_rotations', [])
    if per_part:
        import wandb
        run.log({'independent_same_shape/per_part': wandb.Table(
            columns=['part_index', 'shape_group_id', 'mean_angle_deg', 'within_10deg_pct'],
            data=[[row['part_index'], row['shape_group_id'],
                   row['metrics']['mean_angle_deg'], row['metrics']['within_10deg_pct']]
                  for row in per_part],
        )})
    run.finish()


def main():
    parser = argparse.ArgumentParser(description='Backfill recorded metrics without retraining or uploading meshes/weights.')
    parser.add_argument('folders', nargs='+', type=Path)
    args = parser.parse_args()
    if os.environ.get('WANDB_MODE', 'disabled') == 'disabled':
        parser.error('W&B is disabled; explicitly set WANDB_MODE=online or offline to import')
    for folder in args.folders:
        config = json.loads((folder / 'config.json').read_text(encoding='utf8'))
        result = json.loads((folder / 'metrics.json').read_text(encoding='utf8'))
        root = Path(__file__).resolve().parents[1]
        output = root / 'runs/wandb_backfill' / folder.name
        marker = output / 'wandb_run.json'
        if marker.exists() and json.loads(marker.read_text(encoding='utf8')).get('backfill_complete'):
            print(f'Already imported: {folder.name}; see {output / "wandb_run.json"}', flush=True)
            continue
        run = start_run(output, {**config, 'historical_backfill': True,
                                 'history_source': str(folder),
                                 'history_sampling': 'Only original recorded rows; no interpolated epochs'},
                        name=folder.name, tags=['historical-backfill'])
        for row in result.get('history', []):
            log_epoch(run, row)
        finish_run(run, result)
        metadata = json.loads(marker.read_text(encoding='utf8'))
        metadata['backfill_complete'] = True
        marker.write_text(json.dumps(metadata, indent=2), encoding='utf8')
        print(f'Imported {folder.name}: {run.url}', flush=True)


if __name__ == '__main__':
    main()
