"""Inspect an LeWM benchmark's HF dataset+model repos without installing anything.

Usage:
    python scripts/inspect_hf_bench.py tworoom
    python scripts/inspect_hf_bench.py cube

Reports:
    - model repo file sizes
    - dataset tar contents (first N entries, without extracting)
    - config.json target class
"""
import sys
import json
import tarfile
import io
from huggingface_hub import hf_hub_download, HfApi

BENCHES = {
    "tworoom": {
        "model_repo": "quentinll/lewm-tworooms",
        "dataset_repo": "quentinll/lewm-tworooms",
        "tar_name": "tworoom.tar.zst",
    },
    "cube": {
        "model_repo": "quentinll/lewm-cube",
        "dataset_repo": "quentinll/lewm-cube",
        "tar_name": "cube_single_expert.tar.zst",
    },
    "pusht": {
        "model_repo": "quentinll/lewm-pusht",
        "dataset_repo": "quentinll/lewm-pusht",
        "tar_name": "pusht_expert_train.h5.zst",  # single file, not a tar
    },
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "tworoom"
    spec = BENCHES[name]
    api = HfApi()

    print(f"=== {name}: model repo {spec['model_repo']} ===")
    m = api.model_info(spec["model_repo"], files_metadata=True)
    for s in m.siblings:
        sz = getattr(s, "size", None) or "?"
        print(f"  {s.rfilename}  ({sz} bytes)")

    print(f"\n=== {name}: dataset repo {spec['dataset_repo']} ===")
    d = api.dataset_info(spec["dataset_repo"], files_metadata=True)
    for s in d.siblings:
        sz = getattr(s, "size", None) or "?"
        print(f"  {s.rfilename}  ({sz} bytes)")

    print(f"\n=== {name}: fetch config.json ===")
    cfg_path = hf_hub_download(
        repo_id=spec["model_repo"], filename="config.json", repo_type="model"
    )
    cfg = json.load(open(cfg_path))
    print(f"  target: {cfg.get('_target_')}")
    print(f"  keys: {list(cfg.keys())}")


if __name__ == "__main__":
    main()
