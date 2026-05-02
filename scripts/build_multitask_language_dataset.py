"""Build a joint multi-task HDF5 dataset from annotated single-task datasets.

The merged dataset keeps only the columns needed by the current language model:

- pixels
- action
- language_emb
- step_idx
- ep_idx
- ep_offset
- ep_len
- task_id

Each source dataset is expected to already contain `language_emb`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  Ensures external HDF5 compression plugins are registered.
import numpy as np


def get_episode_meta(h5_file: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    if "ep_offset" in h5_file and "ep_len" in h5_file:
        return (
            h5_file["ep_offset"][:].astype(np.int64),
            h5_file["ep_len"][:].astype(np.int64),
        )

    ep_key = "episode_idx" if "episode_idx" in h5_file else "ep_idx"
    ep_idx = h5_file[ep_key][:].astype(np.int64)
    if len(ep_idx) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    boundaries = np.flatnonzero(np.diff(ep_idx)) + 1
    starts = np.concatenate([[0], boundaries]).astype(np.int64)
    ends = np.concatenate([boundaries, [len(ep_idx)]]).astype(np.int64)
    lengths = ends - starts
    return starts, lengths


def copy_dataset_chunked(
    src_ds: h5py.Dataset,
    dst_ds: h5py.Dataset,
    dst_start: int,
    *,
    cast_dtype=None,
    chunk_rows: int = 2048,
) -> None:
    n = len(src_ds)
    for src_start in range(0, n, chunk_rows):
        src_end = min(src_start + chunk_rows, n)
        data = src_ds[src_start:src_end]
        if cast_dtype is not None:
            data = np.asarray(data, dtype=cast_dtype)
        dst_ds[dst_start + src_start : dst_start + src_end] = data


def copy_dataset_chunked_with_offset(
    src_ds: h5py.Dataset,
    dst_ds: h5py.Dataset,
    dst_start: int,
    *,
    cast_dtype,
    offset: int,
    chunk_rows: int = 2048,
) -> None:
    n = len(src_ds)
    for src_start in range(0, n, chunk_rows):
        src_end = min(src_start + chunk_rows, n)
        data = np.asarray(src_ds[src_start:src_end], dtype=cast_dtype) + offset
        dst_ds[dst_start + src_start : dst_start + src_end] = data


def write_step_idx_chunked(
    dst_ds: h5py.Dataset,
    dst_start: int,
    lengths: np.ndarray,
    *,
    chunk_rows: int = 65536,
) -> None:
    cursor = dst_start
    buffer = []
    buffer_size = 0
    for length in lengths:
        steps = np.arange(int(length), dtype=np.int64)
        buffer.append(steps)
        buffer_size += len(steps)
        if buffer_size >= chunk_rows:
            merged = np.concatenate(buffer)
            dst_ds[cursor : cursor + len(merged)] = merged
            cursor += len(merged)
            buffer = []
            buffer_size = 0
    if buffer:
        merged = np.concatenate(buffer)
        dst_ds[cursor : cursor + len(merged)] = merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a merged multi-task language HDF5")
    parser.add_argument("--output", type=Path, required=True, help="Destination .h5 path")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="TASK=PATH",
        help="Input datasets as task_key=/path/to/file.h5",
    )
    args = parser.parse_args()

    sources: list[tuple[str, Path]] = []
    for item in args.inputs:
        if "=" not in item:
            raise ValueError(f"Expected TASK=PATH, got: {item}")
        task_key, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        sources.append((task_key, path))

    total_rows = 0
    total_eps = 0
    pixels_shape = None
    action_dim = None
    lang_dim = None
    metadata: list[dict] = []

    for task_id, (task_key, path) in enumerate(sources):
        with h5py.File(path, "r") as src:
            if "language_emb" not in src:
                raise KeyError(f"{path} missing language_emb")

            pixels = src["pixels"]
            action = src["action"]
            lang = src["language_emb"]
            offsets, lengths = get_episode_meta(src)

            if pixels_shape is None:
                pixels_shape = tuple(pixels.shape[1:])
                action_dim = int(action.shape[1])
                lang_dim = int(lang.shape[1])
            else:
                if tuple(pixels.shape[1:]) != pixels_shape:
                    raise ValueError(f"{path} pixels shape mismatch: {pixels.shape[1:]} vs {pixels_shape}")
                if int(action.shape[1]) != action_dim:
                    raise ValueError(f"{path} action dim mismatch: {action.shape[1]} vs {action_dim}")
                if int(lang.shape[1]) != lang_dim:
                    raise ValueError(f"{path} language dim mismatch: {lang.shape[1]} vs {lang_dim}")

            total_rows += len(pixels)
            total_eps += len(offsets)
            metadata.append(
                {
                    "task_id": task_id,
                    "task_key": task_key,
                    "path": str(path),
                    "rows": int(len(pixels)),
                    "episodes": int(len(offsets)),
                }
            )

    assert pixels_shape is not None
    assert action_dim is not None
    assert lang_dim is not None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    with h5py.File(tmp_output, "w") as dst:
        dst.create_dataset("pixels", shape=(total_rows, *pixels_shape), dtype=np.uint8, chunks=True)
        dst.create_dataset("action", shape=(total_rows, action_dim), dtype=np.float32, chunks=True)
        dst.create_dataset("language_emb", shape=(total_rows, lang_dim), dtype=np.float32, chunks=True)
        dst.create_dataset("step_idx", shape=(total_rows,), dtype=np.int64, chunks=True)
        dst.create_dataset("ep_idx", shape=(total_rows,), dtype=np.int64, chunks=True)
        dst.create_dataset("task_id", shape=(total_rows,), dtype=np.int16, chunks=True)
        dst.create_dataset("ep_offset", shape=(total_eps,), dtype=np.int64)
        dst.create_dataset("ep_len", shape=(total_eps,), dtype=np.int64)

        row_cursor = 0
        ep_cursor = 0

        for task_id, (task_key, path) in enumerate(sources):
            with h5py.File(path, "r") as src:
                n_rows = len(src["pixels"])
                offsets, lengths = get_episode_meta(src)
                n_eps = len(offsets)

                copy_dataset_chunked(src["pixels"], dst["pixels"], row_cursor)
                copy_dataset_chunked(
                    src["action"], dst["action"], row_cursor, cast_dtype=np.float32
                )
                copy_dataset_chunked(
                    src["language_emb"],
                    dst["language_emb"],
                    row_cursor,
                    cast_dtype=np.float32,
                )

                if "step_idx" in src:
                    copy_dataset_chunked(src["step_idx"], dst["step_idx"], row_cursor)
                else:
                    write_step_idx_chunked(dst["step_idx"], row_cursor, lengths)

                src_ep_idx_key = "episode_idx" if "episode_idx" in src else "ep_idx"
                copy_dataset_chunked_with_offset(
                    src[src_ep_idx_key],
                    dst["ep_idx"],
                    row_cursor,
                    cast_dtype=np.int64,
                    offset=ep_cursor,
                )
                dst["task_id"][row_cursor : row_cursor + n_rows] = task_id
                dst["ep_offset"][ep_cursor : ep_cursor + n_eps] = offsets + row_cursor
                dst["ep_len"][ep_cursor : ep_cursor + n_eps] = lengths

                row_cursor += n_rows
                ep_cursor += n_eps

        dst.attrs["task_keys_json"] = json.dumps([task_key for task_key, _ in sources])
        dst.attrs["task_sources_json"] = json.dumps(metadata)

    tmp_output.replace(args.output)

    print(f"Wrote {args.output}")
    print(f"  rows:     {total_rows}")
    print(f"  episodes: {total_eps}")
    print(f"  tasks:    {[task_key for task_key, _ in sources]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
