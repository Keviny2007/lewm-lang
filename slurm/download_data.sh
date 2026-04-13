#!/bin/bash
#SBATCH --job-name=lewm-data
#SBATCH --output=logs/download_data_%j.out
#SBATCH --error=logs/download_data_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs

# ── HuggingFace token (for CLIP download) ──
export HF_TOKEN="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"

module load python/3.11.11-5e66
export PATH="$HOME/.local/bin:$PATH"
source .venv-tfds/bin/activate

uv pip install -q opencv-python-headless transformers torch h5py tqdm

echo "=== Downloading CALVIN debug dataset ==="
mkdir -p ~/.stable-wm
cd ~/.stable-wm
if [ ! -d "calvin_debug_dataset" ]; then
    wget -q --show-progress http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
    unzip -q calvin_debug_dataset.zip
    rm calvin_debug_dataset.zip
    echo "Download complete: $(du -sh calvin_debug_dataset)"
else
    echo "Already exists, skipping download."
fi

cd ~/scratch/lewm

echo "=== Converting CALVIN to HDF5 ==="
python scripts/convert_calvin.py \
    --calvin_dir ~/.stable-wm/calvin_debug_dataset \
    --annotations annotations/calvin_task_annotations.json \
    --out_dir ~/.stable-wm \
    --split training \
    --img_size 96

echo "=== Done ==="
