#!/bin/bash
#SBATCH --job-name=lewm-eval-pusht
#SBATCH --output=logs/eval_pusht_%j.out
#SBATCH --error=logs/eval_pusht_%j.err
#SBATCH --time=02:00:00
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
export MUJOCO_GL=egl

echo "=== PushT LeWM Eval (sanity check vs published checkpoint) ==="
ls -lh "$STABLEWM_HOME/checkpoints/pusht/lewm_object.ckpt"
ls -lh "$STABLEWM_HOME/datasets/pusht_expert_train.h5"

python eval.py --config-name=pusht policy=pusht/lewm
