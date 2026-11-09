set -euo pipefail

cd "$(dirname "$0")/.."
training_environment="${FASTLAYER_TRAIN_ENV:-$HOME/envs/fastlayer-ptv3}"
point_count=2048
rotations=64
validation_rotations=8
epochs=120
batch_size=256
seed=20261108
cache_root=runs/utonia_tiny_direction_ablation/features

for normals in no_normals with_normals; do
    normal_argument=()
    if [ "$normals" = with_normals ]; then
        normal_argument=(--use-normals)
    fi
    feature_cache="$cache_root/${normals}_r${rotations}_p${point_count}.npz"
    for head_rotations in 16 64; do
        run="runs/utonia_tiny_direction_ablation/${normals}_r${head_rotations}"
        "$training_environment/bin/python" -u tools/train_orientation.py \
            --output "$run" \
            --feature-cache "$feature_cache" \
            --train-rotations "$rotations" \
            --head-rotations "$head_rotations" \
            --validation-rotations "$validation_rotations" \
            --batch-size "$batch_size" \
            --points "$point_count" \
            --epochs "$epochs" \
            --seed "$seed" \
            "${normal_argument[@]}"
    done
done
