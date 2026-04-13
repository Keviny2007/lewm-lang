"""Pre-flight checks for PushT LeWM eval. Run on a login node before sbatch."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import stable_worldmodel as swm
from hydra import initialize, compose
from stable_worldmodel.data import HDF5Dataset

# 1. hydra config resolves with the same overrides the slurm script uses
with initialize(version_base=None, config_path="../config/eval"):
    cfg = compose(config_name="pusht", overrides=["policy=pusht/lewm"])
print("[1/3] hydra config OK — policy:", cfg.policy)
print("      solver target:", cfg.solver.get("_target_", "?"))

# 2. checkpoint loads as a cost model
model = swm.policy.AutoCostModel(cfg.policy)
print("[2/3] ckpt OK — type:", type(model).__name__, "get_cost:", hasattr(model, "get_cost"))

# 3. dataset opens
ds = HDF5Dataset("pusht_expert_train", keys_to_cache=["action", "proprio", "state"])
print("[3/3] dataset OK — cols:", list(ds.column_names)[:8], "len:", len(ds))

print("PREFLIGHT OK")
