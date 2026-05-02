# Progress Log (LeWM Language Conditioning Pivot)

This file tracks project-specific changes and results after pivoting away from
LIBERO toward `tworoom`, `pusht`, and `reacher`.

## Current Status

- Baseline reproduction is complete for `tworoom` and `pusht`.
- Language-annotated training/eval is complete for `tworoom`, `pusht`, and `reacher`.
- Training remains stable with language conditioning: no SIGReg collapse or NaN
  issue remains in the final pipeline.
- Current evidence suggests the model is **not meaningfully using language** in
  the present single-task, canonical-prompt setup.

## Repo / Pipeline Changes

### 1) Dataset pivot and cleanup

- Removed LIBERO / CALVIN-specific train, eval, conversion, and SLURM pipeline
  pieces from the active workflow.
- Kept the repo focused on:
  - `tworoom`
  - `pusht`
  - `reacher`

### 2) Language annotation pipeline

Added a lightweight offline annotation flow:

- `scripts/generate_task_language_bank.py`
  - Generates canonical instructions / variants using OpenRouter.
- `scripts/annotate_hdf5_language.py`
  - Writes `language_emb` and `language_variant_idx` into HDF5 datasets.
  - Supports `--inplace` to avoid duplicating very large files.
- `annotations/task_language_bank.json`
  - Static task language source of truth.

Added language-aware dataset configs:

- `config/train/data/tworoom_language.yaml`
- `config/train/data/pusht_language.yaml`
- `config/train/data/reacher_language.yaml`

### 3) Language-conditioned model path

Integrated language conditioning into LeWM:

- `train.py`
  - Builds a language encoder when `wm.fusion_type != none`
  - Skips normalization for `language_emb`
- `jepa.py`
  - Threads `language_emb -> lang_emb -> predictor`
- `module.py`
  - Supports `cross_attn` fusion through `CrossAttnConditionalBlock`

### 4) Training / eval infrastructure

Added scratch-backed Oscar workflows:

- `slurm/train_eval_tworoom_language.sh`
- `slurm/train_eval_pusht_language.sh`
- `slurm/train_eval_reacher_language.sh`
- `slurm/eval_tworoom_language.sh`

These scripts:

- request A5000 GPUs
- keep checkpoints/results on scratch
- use the language-annotated datasets
- run train and downstream eval in the same structure as the baseline runs

## Important Fixes

### 1) Hydra config fix

- Added `wm.language_emb_dim: null` to `config/train/lewm.yaml`
- This allows language-conditioned overrides like `wm.language_emb_dim=512`

### 2) NaN fix in language training

Initial language runs produced `NaN` losses because `language_emb` was being
normalized like a regular feature despite being constant within a dataset under
canonical-only prompting.

Fixes:

- `train.py`
  - do not normalize `language_emb`
- `utils.py`
  - clamp zero-variance normalizer denominators to `1`

### 3) Eval dataset-name fixes

Language eval initially failed because eval configs still pointed to baseline
dataset names like `tworoom` instead of `tworoom_language`.

Fixed by overriding eval dataset names in the language SLURM scripts.

## Oscar Dataset State

### Tworoom

- Language dataset:
  `/oscar/scratch/kyang128/tworoom_language_run/datasets/tworoom_language.h5`

### PushT

- Annotated in place due to home quota pressure:
  `/users/kyang128/.stable-wm/datasets/pusht_expert_train.h5`
- Symlink used for language name:
  `/users/kyang128/.stable-wm/datasets/pusht_expert_train_language.h5`

### Reacher

- Extracted and annotated on scratch:
  `/oscar/scratch/kyang128/reacher_dataset/reacher.h5`

## Results

### Baseline vs language

| Task | Baseline | Language | Delta |
|------|----------|----------|-------|
| `tworoom` | `0.84` | `0.88` | `+0.04` |
| `pusht` | `0.98` | `0.92` | `-0.06` |
| `reacher` | not recovered from current logs | `0.94` | pending |

### Main takeaway

- Language conditioning does **not** inherently destabilize LeWM.
- The effect on downstream success is task-dependent.
- Current evidence does **not** support the claim that the model is using
  language semantically in this setup.

## Language-Use Ablation

To test whether the `tworoom` language model actually uses text, eval-time
language ablations were added to `eval.py`:

- `normal`
- `zero`
- `random`
- `permute` (not informative for constant-prompt single-task runs)

Also added cross-attention residual logging:

- `module.py`
- `jepa.py`

### Tworoom ablation results

| Condition | Success |
|-----------|---------|
| normal | `88.0` |
| zero | `90.0` |
| random | `88.0` |

### Cross-attention debug

Approximate `cross_attn_ratio_mean`:

- normal: `~0.0075`
- zero: `~0.07`
- random: `~0.05`

### Ablation conclusion

- Zeroing or randomizing language does not hurt performance.
- In this setup, the language branch contributes only a tiny residual under the
  real prompt.
- The current single-task, canonical-prompt design is therefore consistent with
  the model largely ignoring language.

## Next Step

Move to a **joint multi-task language-conditioned model** across:

- `tworoom`
- `pusht`
- `reacher`

Then evaluate with:

- correct task text
- swapped task text
- ablated text

That will create an identifiable test of whether the model actually uses
language rather than merely tolerating it.
