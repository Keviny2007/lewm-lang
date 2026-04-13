"""
Convert LIBERO (HuggingFace LeRobot format) to HDF5 for swm.data.HDF5Dataset.

Source: lerobot/libero (LeRobot v3 format)
  - data/*.parquet     : per-frame rows (action, state, episode_index, task_index)
  - videos/image/      : MP4 files per episode (workspace camera)
  - meta/tasks.jsonl   : task_index -> language instruction
  - meta/episodes/     : per-episode metadata

HDF5 output (flat, concatenated):
  ep_offset    (N_eps,)               int64
  ep_len       (N_eps,)               int32
  ep_idx       (total_steps,)         int32
  pixels       (total_steps, H, W, 3) uint8
  action       (total_steps, 7)       float32
  language_emb (total_steps, 512)     float32  -- CLIP ViT-B/32

Usage:
    python scripts/convert_libero.py --out_dir ~/.stable-wm --img_size 96
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import av
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def log(msg):
    print(msg, flush=True)


def _get_first(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _coerce_task_texts(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        texts = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                texts.append(text)
        return texts
    text = str(value).strip()
    return [text] if text else []


def _merge_task_maps(base: Dict[int, List[str]], update: Dict[int, List[str]]) -> Dict[int, List[str]]:
    merged = {k: list(v) for k, v in base.items()}
    for idx, values in update.items():
        existing = merged.get(idx, [])
        seen = set(existing)
        for text in values:
            if text not in seen:
                existing.append(text)
                seen.add(text)
        merged[idx] = existing
    return merged


def _parse_task_map_file(path: Path) -> Dict[int, List[str]]:
    task_map = {}
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                task_idx = _get_first(entry, ["task_index", "task_id", "id", "index"])
                task_text = _get_first(entry, ["task", "instruction", "language_instruction", "description"])
                if task_idx is None or task_text is None:
                    continue
                task_map[int(task_idx)] = _coerce_task_texts(task_text)
        return task_map

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            if payload and all(str(k).isdigit() for k in payload.keys()):
                return {int(k): _coerce_task_texts(v) for k, v in payload.items()}
            if "tasks" in payload and isinstance(payload["tasks"], list):
                payload = payload["tasks"]
            else:
                payload = [{"task_index": k, "task": v} for k, v in payload.items()]

        if isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                task_idx = _get_first(entry, ["task_index", "task_id", "id", "index"])
                task_text = _get_first(entry, ["task", "instruction", "language_instruction", "description"])
                if task_idx is None or task_text is None:
                    continue
                task_map[int(task_idx)] = _coerce_task_texts(task_text)

        return task_map

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        idx_col = next((c for c in ["task_index", "task_id", "id", "index"] if c in df.columns), None)
        txt_col = next((c for c in ["task", "instruction", "language_instruction", "description"] if c in df.columns), None)
        if idx_col is None or txt_col is None:
            return {}
        parsed = {}
        for k, v in df[[idx_col, txt_col]].drop_duplicates().values:
            texts = _coerce_task_texts(v)
            if texts:
                parsed[int(k)] = texts
        return parsed

    return {}


def load_task_map(repo_dir: Path, repo_id: str, ep_meta: Optional[pd.DataFrame] = None) -> Dict[int, List[str]]:
    local_candidates = []
    meta_dir = repo_dir / "meta"
    if meta_dir.exists():
        local_candidates.extend(sorted(meta_dir.rglob("tasks.jsonl")))
        local_candidates.extend(sorted(meta_dir.rglob("*task*.jsonl")))
        local_candidates.extend(sorted(meta_dir.rglob("*task*.json")))
        local_candidates.extend(sorted(meta_dir.rglob("*task*.parquet")))

    for path in local_candidates:
        try:
            task_map = _parse_task_map_file(path)
            if task_map:
                log(f"Loaded {len(task_map)} tasks from {path}")
                return task_map
        except Exception as exc:
            log(f"  warning: failed parsing {path}: {exc}")

    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        remote_candidates = [
            p for p in repo_files
            if p.startswith("meta/")
            and "task" in p.lower()
            and (p.endswith(".jsonl") or p.endswith(".json") or p.endswith(".parquet"))
        ]
        for rel_path in sorted(remote_candidates):
            try:
                file_path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_path))
                task_map = _parse_task_map_file(file_path)
                if task_map:
                    log(f"Loaded {len(task_map)} tasks from Hub file {rel_path}")
                    return task_map
            except Exception as exc:
                log(f"  warning: failed loading Hub file {rel_path}: {exc}")
    except Exception as exc:
        log(f"  warning: could not list remote metadata files: {exc}")

    if ep_meta is not None and not ep_meta.empty and "task_index" in ep_meta.columns:
        txt_col = next((c for c in ["task", "instruction", "language_instruction", "description"] if c in ep_meta.columns), None)
        if txt_col is not None:
            task_map = {
                int(row["task_index"]): _coerce_task_texts(row[txt_col])
                for _, row in ep_meta[["task_index", txt_col]].drop_duplicates().iterrows()
            }
            task_map = {idx: vals for idx, vals in task_map.items() if vals}
            if task_map:
                log(f"Loaded {len(task_map)} tasks from episode metadata ({txt_col})")
                return task_map

    return {}


# ─── CLIP ─────────────────────────────────────────────────────────────────────

def load_clip(device="cpu"):
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    return model(**inputs).pooler_output.cpu().float().numpy()


# ─── Video ────────────────────────────────────────────────────────────────────

def decode_video(video_path: Path, img_size: int) -> np.ndarray:
    """Decode all frames from an MP4, resize, return (T, H, W, 3) uint8."""
    frames = []
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(img)
    container.close()
    return np.stack(frames, axis=0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def convert(
    out_dir: Path,
    img_size: int,
    max_episodes: int,
    require_language: bool = False,
    task_map_json: Optional[Path] = None,
    lang_variant_policy: str = "first",
    lang_seed: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "libero.h5"

    repo_id = "lerobot/libero"

    log("Downloading LIBERO dataset from HuggingFace...")
    repo_dir = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["*.md", "*.txt"],
    ))
    log(f"Dataset at: {repo_dir}")

    def find_parquet_files(base_dir: Path):
        if not base_dir.exists():
            return []
        parquet_files = []
        for root, _, files in os.walk(base_dir, followlinks=True):
            for name in files:
                if name.endswith(".parquet"):
                    parquet_files.append(Path(root) / name)
        return sorted(parquet_files)

    # ── load episode metadata ──
    ep_meta_root = repo_dir / "meta" / "episodes"
    ep_meta_files = find_parquet_files(ep_meta_root)
    if not ep_meta_files:
        raise RuntimeError(
            f"No episode metadata parquet files found under {ep_meta_root}. "
            "The downloaded LIBERO dataset layout may have changed or download is incomplete."
        )
    ep_meta = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True)
    ep_meta = ep_meta.sort_values("episode_index").reset_index(drop=True)
    n_eps = min(max_episodes, len(ep_meta))
    ep_meta = ep_meta.iloc[:n_eps]
    log(f"Episodes to convert: {n_eps}")

    # ── load frame-level data ──
    data_root = repo_dir / "data"
    data_files = find_parquet_files(data_root)
    if not data_files:
        raise RuntimeError(
            f"No frame parquet files found under {data_root}. "
            "The downloaded LIBERO dataset layout may have changed or download is incomplete."
        )
    frames_df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    frames_df = frames_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # ── load task descriptions ──
    discovered_task_map = load_task_map(repo_dir=repo_dir, repo_id=repo_id, ep_meta=ep_meta)
    task_map = {k: list(v) for k, v in discovered_task_map.items()}

    if task_map_json is not None:
        provided_map_path = task_map_json.expanduser()
        if not provided_map_path.exists():
            raise RuntimeError(f"Provided task map file does not exist: {provided_map_path}")
        provided_task_map = _parse_task_map_file(provided_map_path)
        if not provided_task_map:
            raise RuntimeError(
                f"Could not parse any task annotations from {provided_map_path}. "
                "Expected .json/.jsonl/.parquet with task_index -> instruction mapping."
            )
        task_map = _merge_task_maps(task_map, provided_task_map)
        log(f"Loaded {len(provided_task_map)} task mappings from override: {provided_map_path}")

    has_real_language = bool(task_map)

    if not task_map:
        if require_language:
            raise RuntimeError(
                "No language annotations found in LIBERO metadata. "
                "Provide a mapping file via --task_map_json or disable --require_language."
            )
        if "task_index" in frames_df.columns and len(frames_df):
            unique_task_idx = sorted(int(i) for i in frames_df["task_index"].dropna().unique())
            task_map = {idx: [f"Task {idx}"] for idx in unique_task_idx}
            log(
                "  warning: no task text metadata found; "
                f"falling back to synthetic labels for {len(task_map)} task indices"
            )
        elif "task_index" in ep_meta.columns and len(ep_meta):
            unique_task_idx = sorted(int(i) for i in ep_meta["task_index"].dropna().unique())
            task_map = {idx: [f"Task {idx}"] for idx in unique_task_idx}
            log(
                "  warning: no task text metadata found; "
                f"falling back to synthetic labels for {len(task_map)} task indices"
            )
        else:
            task_map = {0: ["Task 0"]}
            log("  warning: no task metadata or task_index column found; using single fallback label")

    if lang_variant_policy not in {"first", "random"}:
        raise RuntimeError("--lang_variant_policy must be one of: first, random")

    task_text_map = {}
    rng = np.random.default_rng(lang_seed)
    for task_idx, variants in task_map.items():
        valid_variants = _coerce_task_texts(variants)
        if not valid_variants:
            continue
        if lang_variant_policy == "random":
            task_text_map[task_idx] = str(rng.choice(valid_variants))
        else:
            task_text_map[task_idx] = valid_variants[0]

    if require_language and not task_text_map:
        raise RuntimeError(
            "No usable task language strings were found after parsing task maps. "
            "Check --task_map_json contents."
        )

    # ── CLIP encode task descriptions ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading CLIP on {device}...")
    tokenizer, clip_model = load_clip(device)
    emb_dim = 512

    unique_tasks = list(task_text_map.values())
    unique_indices = list(task_text_map.keys())
    log(f"Encoding {len(unique_tasks)} task descriptions...")
    embs = encode_texts(unique_tasks, tokenizer, clip_model, device)
    task_emb = {idx: embs[i] for i, idx in enumerate(unique_indices)}

    # ── find video files ──
    # LeRobot v3 often stores videos by chunk/file and maps episodes via meta/episodes parquet.
    video_root = repo_dir / "videos"
    video_stream = next(
        (d.name for d in video_root.iterdir() if d.is_dir() and "observation.images.image" in d.name),
        None,
    )
    if video_stream is None:
        video_stream = "image" if (video_root / "image").exists() else next(d.name for d in video_root.iterdir() if d.is_dir())

    ep_video_chunk_col = f"videos/{video_stream}/chunk_index"
    ep_video_file_col = f"videos/{video_stream}/file_index"
    ep_video_from_col = f"videos/{video_stream}/from_timestamp"
    ep_video_to_col = f"videos/{video_stream}/to_timestamp"
    has_ep_video_index = all(c in ep_meta.columns for c in [ep_video_chunk_col, ep_video_file_col])
    has_ep_video_timestamps = all(c in ep_meta.columns for c in [ep_video_from_col, ep_video_to_col])

    fps = 10.0
    info_path = repo_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        fps = float(_get_first(info, ["fps"]) or fps)

    video_cache = {}

    def get_video_frames(video_path: Path) -> np.ndarray:
        key = str(video_path)
        if key not in video_cache:
            video_cache.clear()
            video_cache[key] = decode_video(video_path, img_size)
        return video_cache[key]

    def find_video_by_episode_name(ep_idx: int) -> Path:
        ep_str = f"episode_{ep_idx:06d}.mp4"
        for chunk_dir in sorted((video_root / video_stream).iterdir()):
            p = chunk_dir / ep_str
            if p.exists():
                return p
        raise FileNotFoundError(f"No video for episode {ep_idx}")

    # ── convert episodes ──
    action_cols = [c for c in frames_df.columns if c.startswith("action")]
    has_vector_action_col = "action" in frames_df.columns
    all_pixels, all_actions, all_lang_emb = [], [], []
    ep_lengths = []
    skipped = 0

    for i in tqdm(range(n_eps), desc="Converting"):
        ep_idx = int(ep_meta.iloc[i]["episode_index"])
        ep_meta_row = ep_meta.iloc[i]

        ep_frames = frames_df[frames_df["episode_index"] == ep_idx]
        T = len(ep_frames)
        if T == 0:
            skipped += 1
            continue

        if "task_index" in ep_frames.columns and len(ep_frames):
            task_idx = int(ep_frames.iloc[0]["task_index"])
        elif "task_index" in ep_meta.columns:
            task_idx = int(ep_meta_row["task_index"])
        else:
            task_idx = 0

        try:
            if has_ep_video_index:
                chunk_idx = int(ep_meta_row[ep_video_chunk_col])
                file_idx = int(ep_meta_row[ep_video_file_col])
                video_path = video_root / video_stream / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                frames_all = get_video_frames(video_path)
                if has_ep_video_timestamps:
                    from_ts = float(ep_meta_row[ep_video_from_col])
                    to_ts = float(ep_meta_row[ep_video_to_col])
                    start = max(0, int(round(from_ts * fps)))
                    end = max(start + 1, int(round(to_ts * fps)))
                    pixels = frames_all[start:end]
                else:
                    pixels = frames_all
            else:
                video_path = find_video_by_episode_name(ep_idx)
                pixels = get_video_frames(video_path)  # (T_vid, H, W, 3)
        except (FileNotFoundError, av.error.InvalidDataError) as e:
            log(f"  skip ep {ep_idx}: {e}")
            skipped += 1
            continue

        # video frames and parquet rows should match
        T = min(T, len(pixels))
        pixels = pixels[:T]
        if has_vector_action_col:
            actions = np.stack(ep_frames["action"].values[:T]).astype(np.float32)
        elif action_cols:
            actions = ep_frames[action_cols].values[:T].astype(np.float32)
        else:
            raise RuntimeError("No action columns found in frame data")
        if task_idx not in task_emb:
            if require_language and has_real_language:
                raise RuntimeError(
                    f"Missing language annotation for task_index={task_idx}. "
                    "Update --task_map_json to include all task indices in this dataset."
                )
            fallback_text = f"Task {task_idx}"
            fallback_emb = encode_texts([fallback_text], tokenizer, clip_model, device)[0]
            task_emb[task_idx] = fallback_emb
            log(f"  warning: missing task_index={task_idx}, using fallback text '{fallback_text}'")
        lang_rep = np.tile(task_emb[task_idx][None], (T, 1)).astype(np.float32)

        all_pixels.append(pixels)
        all_actions.append(actions)
        all_lang_emb.append(lang_rep)
        ep_lengths.append(T)

    log(f"Converted {len(ep_lengths)} episodes ({skipped} skipped)")

    # ── build HDF5 ──
    ep_lengths_arr = np.array(ep_lengths, dtype=np.int32)
    ep_offsets_arr = np.concatenate([[0], np.cumsum(ep_lengths_arr[:-1])]).astype(np.int64)
    ep_idx_arr = np.repeat(np.arange(len(ep_lengths), dtype=np.int32), ep_lengths_arr)
    total = int(ep_lengths_arr.sum())

    pixels_all = np.concatenate(all_pixels, axis=0)
    actions_all = np.concatenate(all_actions, axis=0)
    lang_emb_all = np.concatenate(all_lang_emb, axis=0)

    pix_chunk = min(100, total)
    act_chunk = min(1000, total)

    log(f"Writing {out_path} ...")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("ep_offset", data=ep_offsets_arr)
        f.create_dataset("ep_len", data=ep_lengths_arr)
        f.create_dataset("ep_idx", data=ep_idx_arr)
        f.create_dataset("pixels", data=pixels_all,
                         chunks=(pix_chunk, img_size, img_size, 3), compression="lzf")
        f.create_dataset("action", data=actions_all,
                         chunks=(act_chunk, actions_all.shape[-1]))
        f.create_dataset("language_emb", data=lang_emb_all,
                         chunks=(act_chunk, emb_dim))

    log(f"Done. {out_path.stat().st_size / 1e9:.2f} GB")
    log(f"  episodes:    {len(ep_lengths)}")
    log(f"  total steps: {total}")
    log(f"  action dim:  {actions_all.shape[-1]}")
    log(f"  img size:    {img_size}x{img_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.expanduser("~/.stable-wm"))
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--max_episodes", type=int, default=10000)
    parser.add_argument(
        "--require_language",
        action="store_true",
        help="Fail if real task language annotations are unavailable (disables synthetic fallback labels).",
    )
    parser.add_argument(
        "--task_map_json",
        type=Path,
        default=None,
        help="Optional path to task mapping file (.json/.jsonl/.parquet) with task_index to instruction text.",
    )
    parser.add_argument(
        "--lang_variant_policy",
        choices=["first", "random"],
        default="first",
        help="How to choose text when a task has multiple instruction variants.",
    )
    parser.add_argument(
        "--lang_seed",
        type=int,
        default=0,
        help="Random seed used when --lang_variant_policy=random.",
    )
    args = parser.parse_args()
    convert(
        Path(args.out_dir),
        args.img_size,
        args.max_episodes,
        require_language=args.require_language,
        task_map_json=args.task_map_json,
        lang_variant_policy=args.lang_variant_policy,
        lang_seed=args.lang_seed,
    )
