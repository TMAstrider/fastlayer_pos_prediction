set -euo pipefail

cd "$(dirname "$0")/.."
training_environment="${FASTLAYER_TRAIN_ENV:-$HOME/envs/fastlayer-ptv3}"
if [ ! -x "$training_environment/bin/python" ]; then
    uv venv --python python3.12 "$training_environment"
fi
uv pip install --python "$training_environment/bin/python" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$training_environment/bin/python" -r requirements-training.txt --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html --only-binary torch-scatter
if [ ! -d external/Utonia/.git ]; then
    git clone https://github.com/Pointcept/Utonia.git external/Utonia
fi
git -C external/Utonia checkout 5fce241d3fd943bdef2db14fdd41fff8b090d98e
mkdir -p checkpoints
if [ ! -f checkpoints/utonia_tiny.pth ]; then
    curl --fail --location --retry 3 --output checkpoints/utonia_tiny.pth https://huggingface.co/Pointcept/Utonia/resolve/9457f025e36c7001ee962a542f0d27a5b600e1e7/utonia_tiny.pth
fi
printf '%s  %s\n' e49a1944403d4ec3df459537fb7d5b132326736d0ebdb7c5a74868f3fa50d137 checkpoints/utonia_tiny.pth | sha256sum --check
"$training_environment/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); tensor = torch.randn(128, 128, device="cuda"); print((tensor @ tensor.T).mean().item())'
