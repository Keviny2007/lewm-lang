#!/bin/bash
#SBATCH --job-name=lewm-setup-pusht
#SBATCH --output=logs/setup_pusht_%j.out
#SBATCH --error=logs/setup_pusht_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

set -euo pipefail

cd ~/scratch/lewm
mkdir -p logs

module load python/3.11.11-5e66
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

export STABLEWM_HOME="$HOME/.stable-wm"
export PYTHONUNBUFFERED=1

echo "=== Downloading PushT dataset from HuggingFace ==="
mkdir -p "$STABLEWM_HOME/datasets"
# swm.data.HDF5Dataset expects $STABLEWM_HOME/datasets/<name>.h5
if [ -f "$STABLEWM_HOME/pusht_expert_train.h5" ] && [ ! -f "$STABLEWM_HOME/datasets/pusht_expert_train.h5" ]; then
    mv "$STABLEWM_HOME/pusht_expert_train.h5" "$STABLEWM_HOME/datasets/"
fi
if [ ! -f "$STABLEWM_HOME/datasets/pusht_expert_train.h5" ]; then
    uv pip install -U "huggingface_hub>=0.23" zstandard
    python - <<'PY'
import os
from huggingface_hub import hf_hub_download
out = hf_hub_download(
    repo_id="quentinll/lewm-pusht",
    filename="pusht_expert_train.h5.zst",
    repo_type="dataset",
    local_dir=os.environ["STABLEWM_HOME"],
)
print("downloaded:", out)
PY
    echo "Decompressing..."
    if command -v zstd &>/dev/null; then
        zstd -d "$STABLEWM_HOME/pusht_expert_train.h5.zst" -o "$STABLEWM_HOME/pusht_expert_train.h5"
    else
        python - <<'PY'
import os, zstandard
src = os.path.join(os.environ["STABLEWM_HOME"], "pusht_expert_train.h5.zst")
dst = os.path.join(os.environ["STABLEWM_HOME"], "pusht_expert_train.h5")
with open(src, "rb") as fin, open(dst, "wb") as fout:
    dctx = zstandard.ZstdDecompressor()
    dctx.copy_stream(fin, fout)
print("decompressed:", dst)
PY
    fi
    ls -lh "$STABLEWM_HOME/pusht_expert_train.h5"
else
    echo "Dataset already present."
fi

echo ""
echo "=== Upgrading stable-worldmodel from git (PyPI 0.0.6 lacks LeWM) ==="
# Use --no-deps to avoid torch/torchvision getting bumped to versions that break
# cluster NCCL (previous attempt upgraded torch 2.6.0+cu124 -> 2.11.0 and broke env).
python -c "import stable_worldmodel.wm.lewm" 2>/dev/null || {
    # repair torch if a previous run nuked it (import fails on libtorch_cuda symbol)
    python -c "import torch; print(torch.__version__)" 2>/dev/null || \
        uv pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install --no-deps --force-reinstall "git+https://github.com/galilai-group/stable-worldmodel.git"
}
python -c "import torch; print('torch:', torch.__version__)"
python -c "from stable_worldmodel.wm.lewm import LeWM; print('LeWM OK:', LeWM)"

echo ""
echo "=== Downloading LeWM PushT checkpoint from HuggingFace ==="
# The HF model repo quentinll/lewm-pusht holds config.json + weights.pt (state_dict),
# but swm.policy.AutoCostModel expects a pickled nn.Module at {run_name}_object.ckpt.
# So: download weights + config, instantiate LeWM via hydra, load state_dict, pickle it.
if [ ! -f "$STABLEWM_HOME/checkpoints/pusht/lewm_object.ckpt" ]; then
    mkdir -p "$STABLEWM_HOME/checkpoints/pusht"
    python - <<'PY'
import os, json, torch, hydra
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

home = os.environ["STABLEWM_HOME"]
repo = "quentinll/lewm-pusht"
cfg_path = hf_hub_download(repo_id=repo, filename="config.json", repo_type="model")
w_path = hf_hub_download(repo_id=repo, filename="weights.pt", repo_type="model")
print("config:", cfg_path)
print("weights:", w_path)

cfg = OmegaConf.create(json.load(open(cfg_path)))
model = hydra.utils.instantiate(cfg)
sd = torch.load(w_path, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing keys:", len(missing), "unexpected keys:", len(unexpected))
if missing: print("  missing[:5]:", missing[:5])
if unexpected: print("  unexpected[:5]:", unexpected[:5])
model.eval()

out = os.path.join(home, "checkpoints", "pusht", "lewm_object.ckpt")
torch.save(model, out)
print("saved module to:", out)
PY
else
    echo "Checkpoint already present."
fi

echo ""
echo "=== Verifying ==="
python -c "import stable_worldmodel as swm; import h5py; f = h5py.File('$STABLEWM_HOME/pusht_expert_train.h5'); print('pusht keys:', list(f.keys())[:10])"
ls -lh "$STABLEWM_HOME/pusht/" 2>/dev/null || echo "pusht/ dir not found"

echo "=== setup_pusht done ==="
