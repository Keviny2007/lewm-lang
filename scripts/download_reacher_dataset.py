"""Download and extract the official LeWM Reacher dataset from Hugging Face.

This script extracts the HDF5 onto scratch storage and can be paired with a
symlink from ~/.stable-wm/datasets/reacher.h5 to avoid home quota pressure.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import zstandard
from huggingface_hub import hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract reacher.tar.zst")
    parser.add_argument(
        "--scratch_dir",
        type=Path,
        default=Path("/oscar/scratch/kyang128/reacher_dataset"),
        help="Scratch directory where the archive and extracted HDF5 will live",
    )
    args = parser.parse_args()

    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    archive = Path(
        hf_hub_download(
            repo_id="quentinll/lewm-reacher",
            filename="reacher.tar.zst",
            repo_type="dataset",
            local_dir=str(args.scratch_dir),
        )
    )
    print(f"downloaded: {archive}")

    out_h5 = args.scratch_dir / "reacher.h5"
    if out_h5.exists():
        print(f"already extracted: {out_h5}")
        return 0

    with open(archive, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                members = 0
                for member in tf:
                    if not member.isfile():
                        continue
                    name = Path(member.name).name
                    if name.endswith(".h5"):
                        member.name = name
                        tf.extract(member, path=args.scratch_dir)
                        print(f"extracted: {args.scratch_dir / name}")
                    members += 1
                print(f"members scanned: {members}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
