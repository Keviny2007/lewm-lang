"""
Evaluate LeWM on LIBERO benchmark tasks via CEM planning.

For each task: reset LIBERO env, plan with the world model using CEM,
execute actions, and report success rates per task and suite.

Usage:
    python eval_libero.py \
        --ckpt ~/.stable-wm/lewm_libero_early_fusion_epoch_36_object.ckpt \
        --suites libero_spatial libero_object libero_goal libero_long \
        --fusion_type early \
        --img_size 96 \
        --num_episodes 20
"""

import argparse
import json
import os
import time

os.environ["MUJOCO_GL"] = "egl"

import torch
# LIBERO's init state files contain numpy arrays; PyTorch 2.6+ rejects them
# with weights_only=True (the new default). Monkey-patch torch.load so LIBERO's
# internal calls use weights_only=False.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from pathlib import Path

import h5py
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ── constants ────────────────────────────────────────────────────────────────

SUITE_OFFSET = {
    "libero_spatial": 0,
    "libero_object": 10,
    "libero_goal": 20,
    "libero_90": 30,   # long-horizon tasks (90/10 split in LIBERO benchmark)
    "libero_10": 30,
}
FRAMESKIP = 2
HISTORY_SIZE = 3
ACTION_DIM = 7
EFFECTIVE_ACT_DIM = FRAMESKIP * ACTION_DIM  # 14


# ── image transform ─────────────────────────────────────────────────────────

def make_img_transform(img_size):
    return transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


# ── CLIP ─────────────────────────────────────────────────────────────────────

def load_clip(device="cpu"):
    name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(name)
    model = CLIPTextModel.from_pretrained(name).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def clip_encode(text, tokenizer, model, device):
    tok = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
    return model(**tok).pooler_output.cpu().float().squeeze(0)  # (512,)


# ── normalization ────────────────────────────────────────────────────────────

def load_norm_stats(dataset_name, columns, cache_dir=None):
    """Compute mean/std for columns from the training HDF5."""
    ds = swm.data.HDF5Dataset(
        dataset_name,
        num_steps=1,
        frameskip=1,
        keys_to_load=columns,
        cache_dir=cache_dir,
    )
    stats = {}
    for col in columns:
        data = ds.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        stats[col] = {
            "mean": torch.tensor(data.mean(axis=0), dtype=torch.float32),
            "std": torch.tensor(data.std(axis=0), dtype=torch.float32),
        }
    return stats


def norm(x, s):
    return (x - s["mean"].to(x.device)) / s["std"].to(x.device)


def denorm(x, s):
    return x * s["std"].to(x.device) + s["mean"].to(x.device)


# ── goal images from training HDF5 ──────────────────────────────────────────

def extract_goal_images(h5_path, task_instructions, clip_tokenizer, clip_model,
                        img_transform, device):
    """For each task, find a matching episode in the HDF5 and return its last frame."""
    goals = {}
    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lens = f["ep_len"][:]
        lang_embs_ds = f["language_emb"]
        pixels_ds = f["pixels"]

        n_eps = len(ep_offsets)

        # get first-frame language embedding per episode
        ep_first_embs = np.stack(
            [lang_embs_ds[int(ep_offsets[i])] for i in range(n_eps)]
        )  # (n_eps, 512)
        ep_embs_t = torch.tensor(ep_first_embs, dtype=torch.float32)

        for task_key, instruction in task_instructions.items():
            query = clip_encode(instruction, clip_tokenizer, clip_model, device)
            sims = F.cosine_similarity(query.unsqueeze(0), ep_embs_t, dim=1)
            best_ep = sims.argmax().item()

            offset = int(ep_offsets[best_ep])
            length = int(ep_lens[best_ep])
            last_frame = pixels_ds[offset + length - 1]  # (H, W, 3) uint8

            goal_t = img_transform(
                torch.from_numpy(last_frame).permute(2, 0, 1)
            )
            goals[int(task_key)] = goal_t.to(device)

    return goals


# ── observation helpers ──────────────────────────────────────────────────────

def obs_to_tensor(obs, img_transform, device):
    img = obs["agentview_image"][::-1].copy()  # flip from OpenGL convention
    return img_transform(torch.from_numpy(img).permute(2, 0, 1)).to(device)


def check_success(env):
    try:
        result = env._check_success()
        if isinstance(result, dict):
            return bool(result.get("task", False))
        return bool(result)
    except Exception:
        return False


# ── CEM planner ──────────────────────────────────────────────────────────────

class CEMPlanner:
    def __init__(self, model, horizon, n_samples=200, n_elites=20, n_iters=5,
                 debug=False):
        self.model = model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self._debug = debug

    @torch.no_grad()
    def plan(self, pixel_history, action_history, goal_image, lang_emb=None):
        """
        pixel_history:  (H, C, h, w)  last H transformed frames
        action_history: (H, act_dim)  last H normalized actions
        goal_image:     (C, h, w)     transformed goal frame
        lang_emb:       (lang_dim,)   normalized language embedding or None

        Returns: (horizon, act_dim) best normalized action plan
        """
        device = next(self.model.parameters()).device
        H = pixel_history.size(0)
        S = self.n_samples

        # expand static tensors for all samples:  (1, S, ...)
        pix = pixel_history[None, None].expand(1, S, -1, -1, -1, -1)
        goal = goal_image[None, None, None].expand(1, S, 1, -1, -1, -1)
        past_act = action_history[None, None].expand(1, S, -1, -1)

        lang_tensor = None
        if lang_emb is not None:
            lang_tensor = lang_emb[None, None, None].expand(1, S, H, -1)

        mean = torch.zeros(self.horizon, EFFECTIVE_ACT_DIM, device=device)
        std = torch.ones(self.horizon, EFFECTIVE_ACT_DIM, device=device)
        best_actions = mean.clone()
        best_cost = float("inf")

        for it in range(self.n_iters):
            # sample future actions  (S, horizon, act_dim)
            noise = torch.randn(S, self.horizon, EFFECTIVE_ACT_DIM, device=device)
            future = (mean[None] + std[None] * noise).clamp(-3, 3)

            # full candidate sequences  (1, S, H + horizon, act_dim)
            candidates = torch.cat([past_act, future[None]], dim=2)

            # build info dict (fresh copy each iteration)
            info = {
                "pixels": pix.clone(),
                "goal": goal.clone(),
                "action": past_act.clone(),
            }
            if lang_tensor is not None:
                info["language_emb"] = lang_tensor.clone()

            costs = self.model.get_cost(info, candidates).squeeze(0)  # (S,)

            if self._debug:
                print(f"        CEM iter {it}: cost min={costs.min():.4f} "
                      f"max={costs.max():.4f} mean={costs.mean():.4f} "
                      f"std={costs.std():.4f}", flush=True)

            # elite selection
            elite_idx = torch.argsort(costs)[: self.n_elites]
            elites = future[elite_idx]

            if costs[elite_idx[0]] < best_cost:
                best_cost = costs[elite_idx[0]].item()
                best_actions = elites[0].clone()

            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=0.05)

        return best_actions  # (horizon, act_dim)


# ── single-task evaluation ───────────────────────────────────────────────────

def evaluate_task(
    model, planner, env_args, init_states, goal_image, lang_emb,
    img_transform, norm_stats, num_episodes, max_steps, device,
):
    successes = []
    n_eps = min(num_episodes, len(init_states))

    for ep in range(n_eps):
        ep_start = time.time()
        env = OffScreenRenderEnv(**env_args)
        env.seed(ep)
        obs = env.reset()
        env.set_init_state(init_states[ep])
        obs = env.step(np.zeros(ACTION_DIM))[0]  # apply init state

        # initialise history buffers
        frame = obs_to_tensor(obs, img_transform, device)
        pixel_hist = frame[None].repeat(HISTORY_SIZE, 1, 1, 1)
        action_hist = torch.zeros(HISTORY_SIZE, EFFECTIVE_ACT_DIM, device=device)

        success = False
        step = 0
        is_first_step = (ep == 0)

        while step < max_steps and not success:
            if is_first_step:
                planner._debug = True
            plan = planner.plan(pixel_hist, action_hist, goal_image, lang_emb=lang_emb)
            planner._debug = False

            # execute first planned action (one world-model step = FRAMESKIP env steps)
            act_norm = plan[0]  # (EFFECTIVE_ACT_DIM,)
            act_raw = denorm(act_norm[None], norm_stats["action"]).squeeze(0)
            act_raw = act_raw.clamp(-1, 1).cpu().numpy()

            if is_first_step:
                print(f"      [debug] norm action: {act_norm.cpu().numpy().round(3)}", flush=True)
                print(f"      [debug] raw  action: {act_raw.round(4)}", flush=True)
                is_first_step = False

            for fs in range(FRAMESKIP):
                sub = act_raw[fs * ACTION_DIM : (fs + 1) * ACTION_DIM]
                obs, _, done, info = env.step(sub)
                step += 1
                if check_success(env):
                    success = True
                    break

            if not success:
                frame = obs_to_tensor(obs, img_transform, device)
                pixel_hist = torch.cat([pixel_hist[1:], frame[None]], dim=0)
                action_hist = torch.cat([action_hist[1:], act_norm[None]], dim=0)

        env.close()
        ep_time = time.time() - ep_start
        tag = "OK" if success else "--"
        print(f"    ep {ep:2d}/{n_eps}: {tag}  (steps {step}, {ep_time:.1f}s)", flush=True)
        successes.append(success)

    return successes


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeWM LIBERO evaluation")
    parser.add_argument("--ckpt", required=True, help="Path to _object.ckpt")
    parser.add_argument(
        "--suites", nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--fusion_type", default="none",
                        choices=["none", "early", "cross_attn"])
    parser.add_argument("--dataset", default="libero")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cem_samples", type=int, default=200)
    parser.add_argument("--cem_elites", type=int, default=20)
    parser.add_argument("--cem_iters", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/libero_eval.json")
    parser.add_argument("--debug", action="store_true",
                        help="Print CEM diagnostics for first step of first episode per task")
    args = parser.parse_args()

    device = args.device
    cache_dir = args.cache_dir or swm.data.utils.get_cache_dir()

    # ── load model ──
    print(f"Loading model: {args.ckpt}", flush=True)
    model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print("  model loaded", flush=True)

    # ── image transform ──
    img_transform = make_img_transform(args.img_size)

    # ── normalization stats from training data ──
    norm_cols = ["action"]
    if args.fusion_type != "none":
        norm_cols.append("language_emb")
    print("Computing normalization stats...", flush=True)
    norm_stats = load_norm_stats(args.dataset, norm_cols, cache_dir)

    # action stats are 7-dim (single step) but model uses 14-dim (frameskip=2)
    # tile the stats to match EFFECTIVE_ACT_DIM
    act_stats = norm_stats["action"]
    norm_stats["action"] = {
        "mean": act_stats["mean"].repeat(FRAMESKIP),
        "std": act_stats["std"].repeat(FRAMESKIP),
    }
    print(f"  action mean: {norm_stats['action']['mean'].numpy().round(4)}", flush=True)
    print(f"  action std:  {norm_stats['action']['std'].numpy().round(4)}", flush=True)

    # ── CLIP (for language encoding + goal image matching) ──
    print("Loading CLIP...", flush=True)
    clip_tok, clip_model = load_clip(device)

    # ── canonical task instructions ──
    task_map_path = Path(__file__).parent / "annotations" / "libero_task_map_canonical.json"
    with open(task_map_path) as f:
        canonical_tasks = json.load(f)

    # ── goal images from training HDF5 ──
    h5_path = Path(cache_dir) / f"{args.dataset}.h5"
    print(f"Extracting goal images from {h5_path}...", flush=True)
    goal_images = extract_goal_images(
        h5_path, canonical_tasks, clip_tok, clip_model, img_transform, device,
    )
    print(f"  got goal images for {len(goal_images)} tasks")

    # ── planner ──
    planner = CEMPlanner(
        model, args.horizon,
        n_samples=args.cem_samples, n_elites=args.cem_elites, n_iters=args.cem_iters,
        debug=args.debug,
    )

    # ── evaluate ──
    benchmark_dict = benchmark.get_benchmark_dict()
    all_results = {}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_start = time.time()
    total_episodes_done = 0

    # count total episodes for ETA
    total_episodes_planned = 0
    for suite_name in args.suites:
        suite_tmp = benchmark_dict[suite_name](0)
        total_episodes_planned += suite_tmp.n_tasks * args.num_episodes
    print(f"\nTotal episodes planned: {total_episodes_planned}", flush=True)

    for si, suite_name in enumerate(args.suites):
        suite = benchmark_dict[suite_name](0)  # instantiate with task_order_index=0
        offset = SUITE_OFFSET.get(suite_name, 0)
        suite_results = {}

        print(f"\n{'='*60}", flush=True)
        print(f"  Suite {si+1}/{len(args.suites)}: {suite_name}  ({suite.n_tasks} tasks)", flush=True)
        print(f"{'='*60}", flush=True)

        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            global_idx = offset + task_id

            task_start = time.time()
            print(f"\n  [{task_id}/{suite.n_tasks}] {task.language}", flush=True)

            env_args = {
                "bddl_file_name": suite.get_task_bddl_file_path(task_id),
                "camera_heights": args.img_size,
                "camera_widths": args.img_size,
            }

            init_states = suite.get_task_init_states(task_id)
            goal = goal_images.get(global_idx)
            if goal is None:
                print(f"    WARNING: no goal image for task {global_idx}, skipping", flush=True)
                continue

            # language embedding (normalised to match training)
            lang_emb = None
            if args.fusion_type != "none":
                raw = clip_encode(task.language, clip_tok, clip_model, device).to(device)
                lang_emb = norm(raw, norm_stats["language_emb"])

            successes = evaluate_task(
                model, planner, env_args, init_states, goal, lang_emb,
                img_transform, norm_stats, args.num_episodes, args.max_steps,
                device,
            )

            task_time = time.time() - task_start
            total_episodes_done += len(successes)
            rate = sum(successes) / len(successes) if successes else 0.0
            suite_results[task_id] = {
                "task_name": task.name,
                "instruction": task.language,
                "success_rate": rate,
                "num_success": sum(successes),
                "num_episodes": len(successes),
                "task_time_s": round(task_time, 1),
            }
            print(f"    success: {sum(successes)}/{len(successes)} ({rate:.0%})  [{task_time:.0f}s]", flush=True)

            # ETA
            elapsed = time.time() - eval_start
            eps_per_sec = total_episodes_done / elapsed if elapsed > 0 else 0
            remaining_eps = total_episodes_planned - total_episodes_done
            eta_s = remaining_eps / eps_per_sec if eps_per_sec > 0 else float("inf")
            eta_h = eta_s / 3600
            print(f"    progress: {total_episodes_done}/{total_episodes_planned} eps, "
                  f"elapsed {elapsed/3600:.1f}h, ETA {eta_h:.1f}h", flush=True)

        suite_rate = np.mean([r["success_rate"] for r in suite_results.values()])
        all_results[suite_name] = {
            "tasks": suite_results,
            "suite_success_rate": float(suite_rate),
        }
        print(f"\n  {suite_name} overall: {suite_rate:.1%}", flush=True)

        # save intermediate results after each suite
        interim = {"args": vars(args), "results": all_results}
        with open(out_path, "w") as f:
            json.dump(interim, f, indent=2, default=str)
        print(f"  (intermediate results saved to {out_path})", flush=True)

    # ── save final results ──
    total_time = time.time() - eval_start
    results_payload = {
        "args": vars(args),
        "total_time_s": round(total_time, 1),
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(results_payload, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print(f"Total eval time: {total_time/3600:.2f}h ({total_time:.0f}s)", flush=True)

    # summary table
    print(f"\n{'='*40}")
    print(f"  Suite                 Success Rate")
    print(f"{'='*40}")
    for suite_name, data in all_results.items():
        print(f"  {suite_name:<22} {data['suite_success_rate']:.1%}")
    overall = np.mean([d["suite_success_rate"] for d in all_results.values()])
    print(f"{'─'*40}")
    print(f"  {'Overall':<22} {overall:.1%}")


if __name__ == "__main__":
    main()
