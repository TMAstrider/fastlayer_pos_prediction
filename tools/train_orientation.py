import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from feature_extraction import extract_features_fast
from orientation_model import (
    ROOT, UTONIA_REVISION, FEATURE_PIPELINE_VERSION, DirectionHead, direction_metrics,
    file_sha256, load_encoder,
)


def make_split(records, validation_count, seed):
    project_records = defaultdict(list)
    for index, record in enumerate(records):
        project_records[record["source_project"]].append(index)
    parent = {project: project for project in project_records}

    def find(project):
        while parent[project] != project:
            project = parent[project]
        return project

    identifiers = defaultdict(set)
    for record in records:
        match = re.search(r"(?<!\d)(\d{9})(?!\d)", record["source_part_name"])
        if match:
            identifiers[match.group(1)].add(record["source_project"])
    for projects in identifiers.values():
        projects = sorted(projects)
        for project in projects[1:]:
            parent[find(project)] = find(projects[0])
    groups = defaultdict(list)
    for project, indices in project_records.items():
        groups[find(project)].extend(indices)
    candidates = list(groups.values())
    random.Random(seed).shuffle(candidates)
    choices = {0: []}
    for group in candidates:
        for count, indices in list(choices.items()):
            choices.setdefault(count + len(group), indices + group)
    viable = [count for count in choices if 0 < count < len(records)]
    chosen_count = min(viable, key=lambda count: (abs(count - validation_count), count))
    validation = sorted(choices[chosen_count])
    training = sorted(set(range(len(records))) - set(validation))
    return {
        "train_indices": training,
        "validation_indices": validation,
        "method": "Hold out whole projects; merge projects sharing a nine-digit part identifier before splitting.",
        "limitation": "Preliminary split. Geometry duplicates, families and symmetric equivalent poses are not yet reviewed. This is not an independent final test set.",
    }


def main():
    parser = argparse.ArgumentParser(description="Train a direction head on frozen pretrained Utonia Tiny PTv3 features")
    parser.add_argument("--pointclouds", type=Path, default=ROOT / "outputs/pointclouds")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/utonia_tiny.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/utonia_tiny_direction")
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--validation-parts", type=int, default=20)
    parser.add_argument("--train-rotations", type=int, default=16)
    parser.add_argument("--head-rotations", type=int, default=0,
                        help="Number of leading cached training rotations per part used for fitting; defaults to all.")
    parser.add_argument("--validation-rotations", type=int, default=8)
    parser.add_argument("--use-normals", action="store_true",
                        help="Use mesh-derived sampled normals; default keeps the previous zero-normal baseline.")
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--extraction-chunk-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=1)
    arguments = parser.parse_args()
    if min(arguments.train_rotations, arguments.validation_rotations, arguments.points, arguments.epochs, arguments.batch_size) <= 0 or arguments.grid_size <= 0:
        parser.error("Counts and grid size must be positive")
    head_rotations = arguments.head_rotations or arguments.train_rotations
    if not 0 < head_rotations <= arguments.train_rotations:
        parser.error("--head-rotations must be positive and no greater than --train-rotations")
    arguments.head_rotations = head_rotations
    if min(arguments.workers, arguments.encoder_batch_size, arguments.extraction_chunk_size, arguments.torch_threads) < 1:
        parser.error("Worker, batch, chunk and thread counts must be positive")
    if not torch.cuda.is_available():
        parser.error("This training run requires CUDA")
    torch.set_num_threads(arguments.torch_threads)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = torch.device("cuda")
    records = json.loads((arguments.pointclouds / "manifest.json").read_text(encoding="utf-8"))["records"]
    if not 0 < arguments.validation_parts < len(records):
        parser.error("Validation count must leave at least one training part")
    split = make_split(records, arguments.validation_parts, arguments.seed)
    arguments.output.mkdir(parents=True, exist_ok=True)
    split["records"] = [{key: record[key] for key in ("source_project", "source_part_id", "source_part_name", "pointcloud_file")} for record in records]
    (arguments.output / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    configuration.update(
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
        weights_sha256=file_sha256(arguments.weights),
        utonia_revision=UTONIA_REVISION,
        feature_pipeline_version=FEATURE_PIPELINE_VERSION,
        coordinate_assumption="Historical raw mesh +Z is the accepted build-up direction; pending FastLayer round-trip verification.",
        training_mode="Frozen pretrained encoder, cached random-rotation features, direction head training only.",
        normal_input=(
            "Sampled mesh face normals from the source triangle winding; orientation/watertightness not yet independently audited."
            if arguments.use_normals else "Zero normals. Color is also zero."
        ),
        feature_pooling="Mean and max at every encoder stage, deepest stage first.",
        metric_scope="Signed direction error against one historical pose, without symmetry correction or printability verification.",
    )
    encoder_configuration = dict(torch.load(arguments.weights, map_location="cpu", weights_only=True)["config"])
    encoder_configuration.update(enable_flash=False, enc_patch_size=[arguments.patch_size] * len(encoder_configuration["enc_depths"]), shuffle_orders=False)
    configuration["encoder_config"] = encoder_configuration
    (arguments.output / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(f"CUDA: {configuration['gpu']}; train={len(split['train_indices'])}, validation={len(split['validation_indices'])}", flush=True)
    cache_path = arguments.feature_cache or (arguments.output / "features.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_metadata = {
        key: configuration[key] for key in (
            "seed", "points", "grid_size", "patch_size", "train_rotations", "validation_rotations",
            "use_normals", "weights_sha256", "utonia_revision", "feature_pipeline_version",
        )
    }
    cache_metadata["manifest_sha256"] = file_sha256(arguments.pointclouds / "manifest.json")
    cache_metadata["train_indices"] = split["train_indices"]
    cache_metadata["validation_indices"] = split["validation_indices"]
    cache_metadata["extraction_chunk_size"] = arguments.extraction_chunk_size
    metadata_path = (
        arguments.output / "features_config.json"
        if arguments.feature_cache is None
        else cache_path.with_suffix(".config.json")
    )
    if cache_path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == cache_metadata:
        with np.load(cache_path) as cache:
            data = {key: cache[key] for key in cache.files}
        print("Loaded matching cached encoder features", flush=True)
    else:
        encoder, _ = load_encoder(arguments.weights, device, arguments.patch_size)
        data = extract_features_fast(arguments, records, split, encoder, device, cache_path, cache_metadata)
        temporary_cache = cache_path.with_suffix(".tmp.npz")
        np.savez(temporary_cache, **data)
        temporary_cache.replace(cache_path)
        metadata_path.write_text(json.dumps(cache_metadata, indent=2), encoding="utf-8")
        del encoder
    torch.cuda.empty_cache()
    training_features = torch.from_numpy(data["train_features"]).to(device)
    training_targets = torch.from_numpy(data["train_targets"]).to(device)
    validation_features = torch.from_numpy(data["validation_features"]).to(device)
    num_training_parts = len(split["train_indices"])
    training_features = training_features.reshape(
        num_training_parts, arguments.train_rotations, -1
    )[:, :head_rotations].reshape(-1, training_features.shape[-1])
    training_targets = training_targets.reshape(
        num_training_parts, arguments.train_rotations, 3
    )[:, :head_rotations].reshape(-1, 3)
    torch.manual_seed(arguments.seed)
    head = DirectionHead(training_features.shape[1]).to(device)
    head.feature_mean.copy_(training_features.mean(dim=0))
    head.feature_std.copy_(training_features.std(dim=0).clamp_min(0.05))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arguments.epochs)
    history = []
    best = float("inf")
    fixed_up = np.tile([0.0, 0.0, 1.0], (len(data["validation_targets"]), 1))
    baseline = direction_metrics(fixed_up, data["validation_targets"])
    training_targets_cpu = training_targets.detach().cpu().numpy()
    training_started = time.monotonic()
    for epoch in range(1, arguments.epochs + 1):
        head.train()
        permutation = torch.randperm(len(training_features), device=device)
        accumulated_loss = torch.zeros((), device=device)
        for indices in permutation.split(arguments.batch_size):
            prediction = head(training_features[indices])
            loss = (1 - (prediction * training_targets[indices]).sum(dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            accumulated_loss += loss.detach() * len(indices)
        scheduler.step()
        head.eval()
        with torch.no_grad():
            predicted = head(validation_features).cpu().numpy()
            training_predicted = head(training_features).cpu().numpy()
        metrics = direction_metrics(predicted, data["validation_targets"])
        training_metrics = direction_metrics(
            training_predicted, training_targets_cpu
        )
        validation_loss = float((1 - (predicted * data["validation_targets"]).sum(axis=-1)).mean())
        history.append({"epoch": epoch, "train_loss": float(accumulated_loss) / len(training_features), "validation_loss": validation_loss, "training": training_metrics, "validation": metrics})
        if metrics["mean_angle_deg"] < best:
            best = metrics["mean_angle_deg"]
            torch.save({
                "head_state_dict": head.state_dict(),
                "feature_dimension": training_features.shape[1],
                "epoch": epoch,
                "validation_metrics": metrics,
                "training_metrics": training_metrics,
                "configuration": configuration,
            }, arguments.output / "best_head.pt")
            np.savez_compressed(arguments.output / "validation_predictions.npz", predicted=predicted, target=data["validation_targets"], record_ids=data["validation_record_ids"])
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch}: train={training_metrics['mean_angle_deg']:.2f} deg, validation={metrics['mean_angle_deg']:.2f} deg, best={best:.2f} deg", flush=True)
    report = {
        "training_parts": len(split["train_indices"]),
        "validation_parts": len(split["validation_indices"]),
        "training_rotations_per_part": head_rotations,
        "use_normals": arguments.use_normals,
        "training_examples": len(training_features),
        "training_rotations": len(training_features),
        "training_seconds": time.monotonic() - training_started,
        "validation_rotations": len(validation_features),
        "fixed_up_baseline": baseline,
        "best_checkpoint": "best_head.pt",
        "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "history": history,
        "limitations": [split["limitation"], configuration["coordinate_assumption"], configuration["metric_scope"]],
    }
    (arguments.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training complete. Best validation mean angle: {best:.2f} deg; fixed-up baseline: {baseline['mean_angle_deg']:.2f} deg", flush=True)


if __name__ == "__main__":
    main()
