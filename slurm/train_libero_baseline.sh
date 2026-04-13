#!/bin/bash
#SBATCH --job-name=lewm-libero-baseline
#SBATCH --output=logs/train_libero_baseline_%j.out
#SBATCH --error=logs/train_libero_baseline_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

# PyTorch cu130 doesn't work with Oscar's CUDA 12.9 driver — use cu124
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

export STABLEWM_HOME="$HOME/.stable-wm"
export WANDB_API_KEY="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"

echo "=== Training vision-only baseline on LIBERO ==="
python train.py \
    data=libero_vision_only \
    output_model_name=lewm_libero_baseline \
    img_size=96 \
    wandb.config.entity=kevin_c_yang-brown-university \
    wandb.config.project=dl-project \
    wandb.config.name=lewm_libero_baseline
