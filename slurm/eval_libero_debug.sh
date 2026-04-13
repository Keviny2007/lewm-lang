#!/bin/bash
#SBATCH --job-name=lewm-eval-debug
#SBATCH --output=logs/eval_libero_debug_%j.out
#SBATCH --error=logs/eval_libero_debug_%j.err
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
export PYTHONUNBUFFERED=1

echo "=== Debug: epoch 10 vs epoch 30, horizon=10 ==="

for EP in 10 30; do
    CKPT="$STABLEWM_HOME/lewm_libero_baseline_epoch_${EP}_object.ckpt"
    echo ""
    echo "--- epoch $EP ---"
    python eval_libero.py \
        --ckpt "$CKPT" \
        --suites libero_spatial \
        --fusion_type none \
        --dataset libero \
        --img_size 96 \
        --num_episodes 1 \
        --horizon 10 \
        --cem_samples 200 \
        --cem_elites 20 \
        --cem_iters 5 \
        --max_steps 20 \
        --debug \
        --output "results/libero_eval_debug_ep${EP}.json"
done

echo "=== Debug run done ==="
