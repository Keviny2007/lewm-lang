"""Add language embeddings to an existing HDF5 dataset.

This is intended for simple single-task datasets such as PushT, Reacher, and
TwoRoom where each episode can be assigned a canonical instruction plus a small
bank of paraphrases.

Example:
    python scripts/annotate_hdf5_language.py \
        --input ~/.stable-wm/datasets/pusht_expert_train.h5 \
        --output ~/.stable-wm/datasets/pusht_expert_train_language.h5 \
        --task_key pusht_expert_train \
        --annotations annotations/task_language_bank.json \
        --variant_policy random \
        --seed 0

    python scripts/annotate_hdf5_language.py \
        --input ~/.stable-wm/datasets/pusht_expert_train.h5 \
        --task_key pusht_expert_train \
        --annotations annotations/task_language_bank.json \
        --variant_policy canonical \
        --inplace
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer


def load_clip(device="cpu"):
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    return model(**inputs).pooler_output.cpu().float().numpy()


def episode_ranges(h5_file):
    if "ep_offset" in h5_file and "ep_len" in h5_file:
        offsets = h5_file["ep_offset"][:].astype(np.int64)
        lengths = h5_file["ep_len"][:].astype(np.int64)
        return [(int(off), int(ln)) for off, ln in zip(offsets, lengths)]

    if "ep_idx" in h5_file:
        ep_idx = h5_file["ep_idx"][:]
        if len(ep_idx) == 0:
            return []
        boundaries = np.flatnonzero(np.diff(ep_idx)) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(ep_idx)]])
        return [(int(s), int(e - s)) for s, e in zip(starts, ends)]

    total_rows = len(next(iter(h5_file.values())))
    return [(0, int(total_rows))]


def main():
    parser = argparse.ArgumentParser(description="Attach CLIP language embeddings to an HDF5 dataset")
    parser.add_argument("--input", type=Path, required=True, help="Source HDF5 path")
    parser.add_argument("--output", type=Path, help="Destination HDF5 path")
    parser.add_argument("--task_key", required=True, help="Key in annotations JSON")
    parser.add_argument("--annotations", type=Path, required=True, help="Task language bank JSON")
    parser.add_argument(
        "--variant_policy",
        choices=["canonical", "random", "cycle"],
        default="random",
        help="How to choose one language variant per episode",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write language datasets directly into the input HDF5 instead of copying to a new file",
    )
    args = parser.parse_args()

    if args.inplace and args.output is not None:
        raise ValueError("Use either --inplace or --output, not both")
    if not args.inplace and args.output is None:
        raise ValueError("--output is required unless --inplace is set")

    with open(args.annotations, encoding="utf-8") as f:
        bank = json.load(f)

    if args.task_key not in bank:
        raise KeyError(f"task_key={args.task_key!r} missing from {args.annotations}")

    entry = bank[args.task_key]
    canonical = str(entry["canonical"]).strip()
    variants = [str(v).strip() for v in entry.get("variants", []) if str(v).strip()]
    if canonical and canonical not in variants:
        variants.insert(0, canonical)
    if not variants:
        raise ValueError(f"No usable language variants found for task_key={args.task_key}")

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, clip_model = load_clip(device)
    variant_embs = encode_texts(variants, tokenizer, clip_model, device).astype(np.float32)

    with h5py.File(args.input, "r") as src:
        ep_ranges = episode_ranges(src)
        total_rows = len(src["pixels"])
        lang_emb = np.zeros((total_rows, variant_embs.shape[1]), dtype=np.float32)
        lang_variant_idx = np.zeros((total_rows,), dtype=np.int16)

        for ep_id, (start, length) in enumerate(ep_ranges):
            if args.variant_policy == "canonical":
                variant_idx = 0
            elif args.variant_policy == "cycle":
                variant_idx = ep_id % len(variants)
            else:
                variant_idx = int(rng.integers(len(variants)))

            lang_emb[start:start + length] = variant_embs[variant_idx]
            lang_variant_idx[start:start + length] = variant_idx

    if args.inplace:
        with h5py.File(args.input, "r+") as dst:
            if "language_emb" in dst:
                del dst["language_emb"]
            if "language_variant_idx" in dst:
                del dst["language_variant_idx"]
            dst.create_dataset("language_emb", data=lang_emb)
            dst.create_dataset("language_variant_idx", data=lang_variant_idx)
            dst.attrs["language_task_key"] = args.task_key
            dst.attrs["language_canonical"] = canonical
            dst.attrs["language_variants_json"] = json.dumps(variants)
        out_path = args.input
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.input, "r") as src, h5py.File(args.output, "w") as dst:
            for key in src.keys():
                src.copy(key, dst, name=key)

            dst.create_dataset("language_emb", data=lang_emb)
            dst.create_dataset("language_variant_idx", data=lang_variant_idx)
            dst.attrs["language_task_key"] = args.task_key
            dst.attrs["language_canonical"] = canonical
            dst.attrs["language_variants_json"] = json.dumps(variants)
        out_path = args.output

    print(f"Wrote {out_path}")
    print(f"  task_key: {args.task_key}")
    print(f"  variants: {len(variants)}")
    print(f"  episodes: {len(ep_ranges)}")
    print(f"  rows:     {total_rows}")


if __name__ == "__main__":
    main()
