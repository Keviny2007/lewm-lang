"""Preflight for slurm/train_eval_baselines.sh on Oscar."""

import pathlib

import stable_worldmodel as swm


def describe(path: pathlib.Path) -> str:
    size_gb = path.stat().st_size / 1_000_000_000
    return f"{path} ({size_gb:.2f} GB)"


cache = pathlib.Path(swm.data.utils.get_cache_dir())
ckpt_cache = pathlib.Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))

required_datasets = {
    "pusht": cache / "datasets" / "pusht_expert_train.h5",
    "tworoom": cache / "datasets" / "tworoom.h5",
}

required_released_ckpts = {
    "pusht": ckpt_cache / "pusht" / "lewm_object.ckpt",
    "tworoom": ckpt_cache / "tworoom" / "lewm_object.ckpt",
}

for name, path in required_datasets.items():
    assert path.exists(), f"{name} dataset missing: {path}"
    print(f"[dataset] {name}: {describe(path)}")

for name, path in required_released_ckpts.items():
    if path.exists():
        print(f"[released-ckpt] {name}: {path}")
    else:
        print(f"[released-ckpt] {name}: missing at {path} (comparison eval will be skipped)")

print("[train-out] pusht:", cache / "pusht_baseline")
print("[train-out] tworoom:", cache / "tworoom_baseline")
print("[eval-in] pusht:", ckpt_cache / "pusht_baseline")
print("[eval-in] tworoom:", ckpt_cache / "tworoom_baseline")
print("PREFLIGHT OK")
