#!/bin/bash
#SBATCH --job-name=lewm-calvin-full
#SBATCH --output=logs/download_full_calvin_%j.out
#SBATCH --error=logs/download_full_calvin_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs

export HF_TOKEN="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"

module load python/3.11.11-5e66
export PATH="$HOME/.local/bin:$PATH"
source .venv-tfds/bin/activate

uv pip install -q opencv-python-headless transformers torch h5py tqdm

echo "=== Downloading CALVIN task_D_D ==="
mkdir -p ~/.stable-wm
cd ~/.stable-wm
if [ ! -d "task_D_D" ]; then
    wget -q --show-progress http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip -q -o task_D_D.zip
    rm task_D_D.zip
    echo "Download complete: $(du -sh task_D_D)"
else
    echo "Already exists, skipping download."
fi

cd ~/scratch/lewm

echo "=== Converting CALVIN task_D_D to HDF5 ==="
python scripts/convert_calvin.py \
    --calvin_dir ~/.stable-wm/task_D_D \
    --annotations annotations/calvin_task_annotations.json \
    --out_dir ~/.stable-wm \
    --split training \
    --img_size 96

echo "=== Done ==="
