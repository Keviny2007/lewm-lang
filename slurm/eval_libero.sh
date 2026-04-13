#!/bin/bash
#SBATCH --job-name=lewm-eval-libero
#SBATCH --output=logs/eval_libero_%j.out
#SBATCH --error=logs/eval_libero_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs results

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

export STABLEWM_HOME="$HOME/.stable-wm"
export PYTHONUNBUFFERED=1

# ── configuration ──
# Override these via: sbatch --export=ALL,CKPT=...,FUSION=... slurm/eval_libero.sh
CKPT="${CKPT:-$STABLEWM_HOME/lewm_libero_baseline_epoch_39_object.ckpt}"
FUSION="${FUSION:-none}"
TAG="${TAG:-baseline}"

echo "=== LIBERO Evaluation ==="
echo "  checkpoint: $CKPT"
echo "  fusion:     $FUSION"
echo "  tag:        $TAG"

python eval_libero.py \
    --ckpt "$CKPT" \
    --suites libero_spatial libero_object libero_goal \
    --fusion_type "$FUSION" \
    --dataset libero \
    --img_size 96 \
    --num_episodes 20 \
    --horizon 10 \
    --cem_samples 200 \
    --cem_elites 20 \
    --cem_iters 5 \
    --max_steps 300 \
    --output "results/libero_eval_${TAG}.json"
