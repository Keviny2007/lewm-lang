from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np


class CompositeHDF5Dataset:
    """Sample fixed-length sequences from multiple HDF5 datasets without merging them."""

    def __init__(
        self,
        sources,
        num_steps: int,
        frameskip: int,
        keys_to_load,
        keys_to_cache=None,
        name=None,
        transform=None,
        **_,
    ):
        self.sources = list(sources)
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.keys_to_load = list(keys_to_load)
        self.keys_to_cache = list(keys_to_cache or [])
        self.name = name or "composite_hdf5"
        self.transform = transform
        self.column_names = list(self.keys_to_load)

        self._file_handles = {}
        self._column_cache = {}
        self._source_meta = []
        self._source_lengths = []
        self._cumulative_lengths = []

        running_total = 0
        for source_idx, source in enumerate(self.sources):
            path = Path(source["path"]).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Composite dataset source missing: {path}")

            with h5py.File(path, "r") as h5_file:
                ep_key = "episode_idx" if "episode_idx" in h5_file else "ep_idx"
                step_idx = np.asarray(h5_file["step_idx"][:], dtype=np.int64)
                ep_idx = np.asarray(h5_file[ep_key][:], dtype=np.int64)
                if len(step_idx) != len(ep_idx):
                    raise ValueError(f"{path}: step_idx and {ep_key} length mismatch")

                if "ep_len" in h5_file:
                    ep_len = np.asarray(h5_file["ep_len"][:], dtype=np.int64)
                    if ep_idx.min() < 0 or ep_idx.max() >= len(ep_len):
                        raise ValueError(f"{path}: episode ids out of bounds for ep_len")
                    ep_lengths_per_row = ep_len[ep_idx]
                else:
                    _, counts = np.unique(ep_idx, return_counts=True)
                    ep_lengths_per_row = counts[ep_idx]

                max_start = ep_lengths_per_row - self.num_steps * self.frameskip
                valid_start_rows = np.flatnonzero(step_idx <= max_start).astype(np.int64)
                if len(valid_start_rows) == 0:
                    raise ValueError(
                        f"{path}: no valid start rows for num_steps={self.num_steps}, "
                        f"frameskip={self.frameskip}"
                    )

                key_dims = {}
                for key in self.keys_to_load:
                    if key not in h5_file:
                        raise KeyError(f"{path}: missing required key '{key}'")
                    key_dims[key] = tuple(h5_file[key].shape[1:])

                raw_action_dim = key_dims.get("action")
                lang_dim = key_dims.get("language_emb")

            self._source_meta.append(
                {
                    "path": path,
                    "task_key": source.get("task_key", f"source_{source_idx}"),
                    "valid_start_rows": valid_start_rows,
                    "ep_key": ep_key,
                    "step_idx": step_idx,
                    "ep_lengths_per_row": ep_lengths_per_row,
                    "key_dims": key_dims,
                    "raw_action_dim": raw_action_dim,
                    "lang_dim": lang_dim,
                }
            )

            source_len = len(valid_start_rows)
            self._source_lengths.append(source_len)
            running_total += source_len
            self._cumulative_lengths.append(running_total)

        if not self._source_meta:
            raise ValueError("CompositeHDF5Dataset requires at least one source")

        self._validate_shared_dimensions()

    def _validate_shared_dimensions(self):
        ref_action_dim = self._source_meta[0]["raw_action_dim"]
        ref_lang_dim = self._source_meta[0]["lang_dim"]
        for meta in self._source_meta[1:]:
            if meta["raw_action_dim"] != ref_action_dim:
                raise ValueError(
                    f"Action dim mismatch: {meta['path']} has {meta['raw_action_dim']}, "
                    f"expected {ref_action_dim}"
                )
            if meta["lang_dim"] != ref_lang_dim:
                raise ValueError(
                    f"Language dim mismatch: {meta['path']} has {meta['lang_dim']}, "
                    f"expected {ref_lang_dim}"
                )

    def _get_file(self, source_idx: int):
        handle = self._file_handles.get(source_idx)
        if handle is None:
            handle = h5py.File(self._source_meta[source_idx]["path"], "r")
            self._file_handles[source_idx] = handle
        return handle

    def _resolve_index(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        source_idx = bisect_right(self._cumulative_lengths, idx)
        prev_total = 0 if source_idx == 0 else self._cumulative_lengths[source_idx - 1]
        local_idx = idx - prev_total
        return source_idx, local_idx

    def _cast_array(self, key: str, array):
        if key in {"action", "language_emb"}:
            return np.asarray(array, dtype=np.float32)
        return np.asarray(array)

    def _get_action_chunks(self, h5_file, start_rows):
        start_rows = np.asarray(start_rows, dtype=np.int64)
        offsets = np.arange(self.frameskip, dtype=np.int64)
        flat_indices = (start_rows[:, None] + offsets[None, :]).reshape(-1)
        action = self._cast_array("action", h5_file["action"][flat_indices])
        return action.reshape(len(start_rows), -1)

    def __len__(self):
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int):
        source_idx, local_idx = self._resolve_index(idx)
        meta = self._source_meta[source_idx]
        h5_file = self._get_file(source_idx)

        start_row = int(meta["valid_start_rows"][local_idx])
        row_indices = start_row + np.arange(self.num_steps, dtype=np.int64) * self.frameskip

        sample = {}
        for key in self.keys_to_load:
            if key == "action":
                sample[key] = self._get_action_chunks(h5_file, row_indices)
            else:
                sample[key] = self._cast_array(key, h5_file[key][row_indices])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_col_data(self, key: str):
        if key in self._column_cache:
            return self._column_cache[key]

        if key not in self.keys_to_load:
            raise KeyError(key)

        pieces = []
        for source_idx, _meta in enumerate(self._source_meta):
            h5_file = self._get_file(source_idx)
            if key == "action":
                meta = self._source_meta[source_idx]
                valid_rows = np.flatnonzero(
                    meta["step_idx"] <= (meta["ep_lengths_per_row"] - self.frameskip)
                ).astype(np.int64)
                pieces.append(self._get_action_chunks(h5_file, valid_rows))
            else:
                pieces.append(self._cast_array(key, h5_file[key][:]))

        data = np.concatenate(pieces, axis=0)
        if key in self.keys_to_cache:
            self._column_cache[key] = data
        return data

    def get_dim(self, key: str):
        if key not in self.keys_to_load:
            raise KeyError(key)
        if key == "action":
            raw_dim = self._source_meta[0]["raw_action_dim"][-1]
            return self.frameskip * raw_dim
        return self._source_meta[0]["key_dims"][key][-1]

    def __del__(self):
        for handle in self._file_handles.values():
            try:
                handle.close()
            except Exception:
                pass
