import os

os.environ["MUJOCO_GL"] = "egl"

import time
from copy import deepcopy
import json
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from transformers import CLIPTextModel, CLIPTokenizer


class LanguageAblationDataset:
    """Wrap a dataset and optionally perturb language embeddings at eval time."""

    def __init__(
        self,
        dataset,
        mode="normal",
        seed=0,
        override_language_emb=None,
    ):
        self.dataset = dataset
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.override_language_emb = override_language_emb

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def load_chunk(self, episodes_idx, start, end):
        chunk = self.dataset.load_chunk(episodes_idx, start, end)
        if self.override_language_emb is not None:
            chunk = [deepcopy(ep) for ep in chunk]
            for ep in chunk:
                lang = ep.get("language_emb")
                if lang is None:
                    continue
                if isinstance(lang, torch.Tensor):
                    override = torch.from_numpy(self.override_language_emb).to(
                        device=lang.device, dtype=lang.dtype
                    )
                    ep["language_emb"] = override.unsqueeze(0).expand_as(lang).clone()
                else:
                    ep["language_emb"] = np.broadcast_to(
                        self.override_language_emb[None, :], lang.shape
                    ).astype(lang.dtype, copy=True)
            return chunk

        if self.mode == "normal":
            return chunk

        if len(chunk) == 0 or "language_emb" not in chunk[0]:
            return chunk

        chunk = [deepcopy(ep) for ep in chunk]
        if self.mode == "zero":
            for ep in chunk:
                ep["language_emb"] = np.zeros_like(ep["language_emb"])
            return chunk

        if self.mode == "random":
            for ep in chunk:
                lang = ep["language_emb"]
                if isinstance(lang, torch.Tensor):
                    ep["language_emb"] = torch.randn_like(lang)
                else:
                    ep["language_emb"] = self.rng.standard_normal(
                        size=lang.shape
                    ).astype(lang.dtype, copy=False)
            return chunk

        if self.mode == "permute":
            perm = self.rng.permutation(len(chunk))
            lang_bank = [chunk[i]["language_emb"].copy() for i in perm]
            for ep, lang in zip(chunk, lang_bank):
                ep["language_emb"] = lang
            return chunk

        raise ValueError(f"Unknown lang_eval.mode={self.mode!r}")

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def load_clip(device="cpu"):
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode_text(text, tokenizer, model, device):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
    return model(**inputs).pooler_output[0].detach().cpu().float().numpy()


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        model.debug_language_stats = bool(
            cfg.get("lang_eval", {}).get("debug_cross_attn", False)
        )
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    lang_eval_cfg = cfg.get("lang_eval", {})
    override_language_emb = None
    if lang_eval_cfg.get("override_task_key"):
        annotations_path = Path(lang_eval_cfg.get("annotations_path", "annotations/task_language_bank.json"))
        with annotations_path.open(encoding="utf-8") as f:
            bank = json.load(f)
        task_key = lang_eval_cfg.override_task_key
        if task_key not in bank:
            raise KeyError(f"override_task_key={task_key!r} missing from {annotations_path}")
        text = str(bank[task_key]["canonical"]).strip()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, clip_model = load_clip(device)
        override_language_emb = encode_text(text, tokenizer, clip_model, device).astype(np.float32)
        print(
            f"[lang eval] overriding language with canonical prompt for task_key={task_key}",
            flush=True,
        )

    dataset_for_eval = LanguageAblationDataset(
        dataset,
        mode=lang_eval_cfg.get("mode", "normal"),
        seed=lang_eval_cfg.get("seed", cfg.seed),
        override_language_emb=override_language_emb,
    )

    if lang_eval_cfg.get("mode", "normal") != "normal":
        print(
            f"[lang eval] applying language ablation mode={lang_eval_cfg.get('mode')}",
            flush=True,
        )

    metrics = world.evaluate_from_dataset(
        dataset_for_eval,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
