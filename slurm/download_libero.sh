#!/bin/bash
#SBATCH --job-name=lewm-libero
#SBATCH --output=logs/download_libero_%j.out
#SBATCH --error=logs/download_libero_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs

export HF_TOKEN="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"

module load python/3.11.11-5e66
export PATH="$HOME/.local/bin:$PATH"
source .venv-tfds/bin/activate

uv pip install -q opencv-python-headless transformers torch h5py tqdm \
    huggingface_hub pandas av

echo "=== Converting LIBERO to HDF5 ==="
python scripts/augment_libero_language.py \
    --input annotations/libero_task_map_canonical.json \
    --output annotations/libero_task_map_augmented.json \
    --variants_per_task 8 \
    --seed 0

rm -f ~/.stable-wm/libero.h5

python scripts/convert_libero.py \
    --out_dir ~/.stable-wm \
    --img_size 96 \
    --require_language \
    --task_map_json annotations/libero_task_map_augmented.json \
    --lang_variant_policy random \
    --lang_seed 0

echo "=== Done ==="
