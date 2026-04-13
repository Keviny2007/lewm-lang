#!/bin/bash
#SBATCH --job-name=lewm-collapse
#SBATCH --output=logs/eval_collapse_baseline_%j.out
#SBATCH --error=logs/eval_collapse_baseline_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

export STABLEWM_HOME="$HOME/.stable-wm"
export PYTHONPATH="$HOME/scratch/lewm:${PYTHONPATH:-}"

echo "=== Checking dimensional collapse on baseline epoch 50 ==="
python scripts/check_collapse.py \
    --ckpt "$STABLEWM_HOME/lewm_libero_baseline_epoch_39_object.ckpt" \
    --dataset libero \
    --img_size 96 \
    --num_samples 2000
