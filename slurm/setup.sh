#!/bin/bash
#SBATCH --job-name=lewm-setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs

echo "=== Setting up LeWM-VLA environment ==="

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr

if ! command -v uv &> /dev/null; then
    pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

rm -rf .venv
uv venv --python=3.11
source .venv/bin/activate

echo "=== Installing dependencies ==="
uv pip install stable-worldmodel[train,env]
uv pip install tensorflow tensorflow-datasets opencv-python-headless transformers
# stable-worldmodel pins datasets==1.1.1 but stable_pretraining needs >=2.x
uv pip install "datasets>=2.14.0" --upgrade

echo ""
echo "=== Verifying ==="
python -c "import stable_worldmodel as swm; print('stable-worldmodel OK')"
python -c "import stable_pretraining as spt; print('stable-pretraining OK')"
python -c "import tensorflow_datasets as tfds; print('tensorflow-datasets OK')"
python -c "import transformers; print('transformers OK')"
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup complete ==="
