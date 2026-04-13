#!/bin/bash
#SBATCH --job-name=lewm-eval-smoke
#SBATCH --output=logs/eval_libero_smoke_%j.out
#SBATCH --error=logs/eval_libero_smoke_%j.err
#SBATCH --time=00:30:00
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

CKPT="$STABLEWM_HOME/lewm_libero_baseline_epoch_30_object.ckpt"

echo "=== Smoke test: 1 suite, 1 episode, 3 max steps ==="

python eval_libero.py \
    --ckpt "$CKPT" \
    --suites libero_spatial \
    --fusion_type none \
    --dataset libero \
    --img_size 96 \
    --num_episodes 1 \
    --horizon 3 \
    --cem_samples 16 \
    --cem_elites 4 \
    --cem_iters 2 \
    --max_steps 6 \
    --output "results/libero_eval_smoke.json"

echo "=== Smoke test passed ==="
