#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
training_environment="${FASTLAYER_TRAIN_ENV:-$HOME/envs/fastlayer-ptv3}"
if [[ -f "$training_environment/bin/activate" ]]; then
    source "$training_environment/bin/activate"
elif [[ -n "${FASTLAYER_TRAIN_ENV:-}" ]]; then
    printf 'Requested environment does not exist: %s\n' "$training_environment" >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled

if [[ "${1:-}" != "--inside-memory-limit" ]]; then
    exec systemd-run --user --scope --quiet \
        --unit=fastlayer-fresh-points-audit \
        --property=MemoryMax=4G --property=MemorySwapMax=256M \
        bash tools/run_fresh_points_audit.sh --inside-memory-limit
fi

group="$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup)"
limit="$(cat "/sys/fs/cgroup${group}/memory.max")"
if [[ "$limit" == "max" ]] || (( limit > 4294967296 )); then
    printf 'Refusing to run: effective process memory limit is not <=4GiB\n' >&2
    exit 1
fi
printf 'Verified memory.max=%s bytes; one sampling process; 32768-face file blocks\n' "$limit"
python -u tools/prepare_fresh_mesh_samples.py --workers 1 --resume
python -u tools/evaluate_full_equivariant_fresh_points.py
