#!/bin/bash
#SBATCH --job-name=lewm-early-fusion
#SBATCH --output=logs/train_libero_early_fusion_%j.out
#SBATCH --error=logs/train_libero_early_fusion_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
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
export WANDB_API_KEY="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"

echo "=== Training early fusion on LIBERO ==="
python train.py \
    data=libero \
    output_model_name=lewm_libero_early_fusion \
    img_size=96 \
    wm.fusion_type=early \
    wandb.config.entity=kevin_c_yang-brown-university \
    wandb.config.project=dl-project \
    wandb.config.name=lewm_libero_early_fusion
