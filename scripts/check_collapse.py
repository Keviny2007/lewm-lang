"""Check latent space dimensional collapse by analyzing the covariance matrix rank.

Loads a checkpoint, encodes validation data, and reports:
  - Singular value spectrum of the embedding covariance matrix
  - Effective rank (number of singular values > 1% of max)
  - Condition number
"""

import argparse
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import stable_pretraining as spt
import torch
from torchvision.transforms import v2 as transforms


def img_transform(img_size):
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to _object.ckpt file")
    parser.add_argument("--dataset", default="libero", help="Dataset name")
    parser.add_argument("--cache_dir", default=None, help="Dataset cache dir")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of frames to encode")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = model.cuda().eval()

    # Load dataset
    cache_dir = Path(args.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        args.dataset,
        num_steps=1,
        frameskip=1,
        keys_to_load=["pixels"],
        cache_dir=cache_dir,
    )

    transform = img_transform(args.img_size)

    # Collect embeddings
    num_samples = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    all_embs = []
    for start in range(0, len(indices), args.batch_size):
        batch_idx = indices[start:start + args.batch_size]
        frames = []
        for i in batch_idx:
            sample = dataset[int(i)]
            pix = sample["pixels"]  # (1, C, H, W) uint8
            pix = transform(pix[0])  # (C, H, W) float32 normalized
            frames.append(pix)

        pixels = torch.stack(frames).unsqueeze(1).cuda()  # (B, 1, C, H, W)
        info = {"pixels": pixels}
        output = model.encode(info)
        emb = output["emb"][:, 0]  # (B, D)
        all_embs.append(emb.cpu())

    all_embs = torch.cat(all_embs, dim=0).numpy()  # (N, D)
    print(f"\nEncoded {all_embs.shape[0]} frames -> embeddings shape: {all_embs.shape}")

    # Compute covariance and SVD
    emb_centered = all_embs - all_embs.mean(axis=0)
    cov = np.cov(emb_centered, rowvar=False)  # (D, D)
    singular_values = np.linalg.svdvals(cov)

    # Metrics
    sv_normalized = singular_values / singular_values[0]
    effective_rank_01 = np.sum(sv_normalized > 0.01)  # > 1% of max
    effective_rank_001 = np.sum(sv_normalized > 0.001)  # > 0.1% of max
    condition_number = singular_values[0] / max(singular_values[-1], 1e-12)
    embed_dim = all_embs.shape[1]

    print(f"\n{'='*50}")
    print(f"Dimensional Collapse Analysis")
    print(f"{'='*50}")
    print(f"Embedding dim:          {embed_dim}")
    print(f"Effective rank (>1%):   {effective_rank_01} / {embed_dim}")
    print(f"Effective rank (>0.1%): {effective_rank_001} / {embed_dim}")
    print(f"Condition number:       {condition_number:.2f}")
    print(f"Top-10 singular values: {singular_values[:10].round(4)}")
    print(f"Bottom-5 singular vals: {singular_values[-5:].round(6)}")
    print(f"\nCollapse ratio (eff_rank/dim): {effective_rank_01/embed_dim:.2%}")

    if effective_rank_01 / embed_dim > 0.5:
        print("-> Healthy: SIGReg is preventing collapse")
    elif effective_rank_01 / embed_dim > 0.2:
        print("-> Partial collapse detected")
    else:
        print("-> Severe collapse detected")


if __name__ == "__main__":
    main()
