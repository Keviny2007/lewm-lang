# Progress Log (LeWM + LIBERO Language)

This file tracks custom work added on top of the upstream `le-wm` repo.

## Current Status

- LIBERO conversion pipeline on Oscar is completed successfully.
- Output artifact exists: `~/.stable-wm/libero.h5` (job `1366421`, ~5.86 GB reported).
- Dataset was generated with strict language mode enabled and offline augmented language mappings.

## What Was Implemented

### 1) Robust LIBERO conversion support

Updated `scripts/convert_libero.py` to support current `lerobot/libero` layout:

- Recursive parquet discovery with symlink-following
- LeRobot v3-style video lookup (`chunk/file` + episode timestamps)
- Vector action column handling (`action` array column)
- Better metadata fallbacks and error messages

### 2) Strict language controls

Added language safety controls to `scripts/convert_libero.py`:

- `--require_language`: fail if real language mapping is unavailable
- `--task_map_json`: inject canonical / augmented task mappings

### 3) Language variant support in conversion

`convert_libero.py` now supports task maps where values are lists:

- `task_index -> [variant_1, variant_2, ...]`
- `--lang_variant_policy {first,random}`
- `--lang_seed` for deterministic random selection

### 4) Offline augmentation script

Added `scripts/augment_libero_language.py`:

- Input: canonical map JSON (`task_index -> instruction`)
- Output: augmented map JSON (`task_index -> [variants...]`)
- Configurable `--variants_per_task` and `--seed`

### 5) Canonical task map file

Added `annotations/libero_task_map_canonical.json`:

- Contains 40 canonical instructions (task indices `0..39`)
- Derived from official LIBERO benchmark suite task definitions
- Used as seed input for augmentation

### 6) Oscar SLURM pipeline update

Updated `slurm/download_libero.sh` to:

1. Run offline augmentation
2. Remove stale `~/.stable-wm/libero.h5`
3. Run strict conversion with:
   - `--require_language`
   - `--task_map_json annotations/libero_task_map_augmented.json`
   - `--lang_variant_policy random`
   - `--lang_seed 0`

## Key Output Verification

From latest logs (`download_libero_1366421`):

- `Wrote augmented map for 40 tasks`
- `Loaded 40 task mappings from override`
- `Converted 1693 episodes (0 skipped)`
- `Done. 5.86 GB`

## Phase 2: Vision-Only Baseline (Completed)

### Training

- Trained standard LeWM (no language) on LIBERO for **78 epochs** (~8 hours, 1 GPU)
- Config: `data=libero_vision_only`, `img_size=96`, batch_size=128
- WandB run: `kevin_c_yang-brown-university/dl-project/lewm_libero_baseline`
- Job 1373748 on Oscar (hit 8h wall time, 78 epochs completed)

### Baseline Results (epoch 39)

| Metric | Value |
|--------|-------|
| Val pred_loss | ~0.003 |
| Val sigreg_loss | ~1.3 |
| Val total loss | ~0.12 |

### Dimensional Collapse Analysis (epoch 39)

| Metric | Value |
|--------|-------|
| Embedding dim | 192 |
| Effective rank (>1%) | 154 / 192 (80.2%) |
| Effective rank (>0.1%) | 188 / 192 |
| Condition number | 2744.89 |
| Status | **Healthy — no collapse** |

Checkpoints: `~/.stable-wm/lewm_libero_baseline_epoch_{1..39}_object.ckpt`

## Phase 3: Language-Conditioned Training (Completed)

### A) Model/data plumbing

- `language_emb` loaded and normalized via existing HDF5 pipeline (`config/train/data/libero.yaml`)
- Added `language_encoder` (MLP: 512 → 192) to JEPA
- `fusion_type` config switch: `none`, `early`, `cross_attn`
- Forward pass threads `lang_emb` through encode → predict

### B) Architecture integration

Two fusion variants implemented:

1. **Early fusion** (`wm.fusion_type=early`): Language embedding added to action conditioning signal (AdaLN-zero modulation). Language acts as a bias on the action conditioning across all timesteps.

2. **Cross-attention fusion** (`wm.fusion_type=cross_attn`): Added `CrossAttnConditionalBlock` with cross-attention layers where visual state (Q) attends to language embedding (K/V). Language acts as a contextual filter deeper in the predictor network.

Training completed on Oscar (8h, 1 GPU each, 6 CPUs):
- Early fusion: job 1407033, ~75 epochs, 36 checkpoints saved
- Cross-attention: job 1407035, ~69 epochs, 30 checkpoints saved

### C) Validation results

#### Loss comparison (WandB overlay, all 3 runs)

**Val pred_loss**: Early fusion tracks identically with baseline (~0.002). Cross-attention converges slower but reaches the same ballpark by ~60k steps.

**Val sigreg_loss**: All three converge to ~1.3. No degradation from language injection.

#### Dimensional Collapse Analysis (3-way comparison)

| Metric | Baseline (ep39) | Early Fusion (ep36) | Cross-Attn (ep30) |
|--------|----------------|--------------------|--------------------|
| Effective rank (>1%) | 154/192 (80.2%) | 151/192 (78.6%) | 142/192 (74.0%) |
| Effective rank (>0.1%) | 188/192 | 187/192 | 185/192 |
| Condition number | 2744.89 | 2409.28 | 2103.40 |
| Status | Healthy | Healthy | Healthy |

**Key findings:**
- **SIGReg prevents collapse in all variants** — the core proposal hypothesis is confirmed
- **Early fusion** has minimal impact on effective rank (80.2% → 78.6%), tracks baseline loss curves exactly
- **Cross-attention** has slightly lower effective rank (74.0%) but the best condition number (2103), suggesting more uniform spread across dimensions
- Language embeddings do not destabilize the latent space under SIGReg regularization

## Repro Commands (Oscar)

```bash
# Data conversion
cd ~/scratch/lewm
source .venv-tfds/bin/activate
python scripts/augment_libero_language.py \
  --input annotations/libero_task_map_canonical.json \
  --output annotations/libero_task_map_augmented.json \
  --variants_per_task 8 \
  --seed 0

python scripts/convert_libero.py \
  --out_dir ~/.stable-wm \
  --img_size 96 \
  --require_language \
  --task_map_json annotations/libero_task_map_augmented.json \
  --lang_variant_policy random \
  --lang_seed 0

# Training (activate .venv, not .venv-tfds)
source .venv/bin/activate
python train.py data=libero_vision_only img_size=96 output_model_name=lewm_libero_baseline
python train.py data=libero img_size=96 wm.fusion_type=early output_model_name=lewm_libero_early_fusion
python train.py data=libero img_size=96 wm.fusion_type=cross_attn output_model_name=lewm_libero_cross_attn

# Collapse analysis
python scripts/check_collapse.py --ckpt <path_to_ckpt> --dataset libero --img_size 96
```
