"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        language_encoder=None,
        fusion_type="none",
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.language_encoder = language_encoder
        self.fusion_type = fusion_type

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        if "language_emb" in info and self.language_encoder is not None:
            lang = info["language_emb"][:, 0]  # (B, lang_dim) — same across timesteps
            info["lang_emb"] = self.language_encoder(lang)  # (B, embed_dim)

        return info

    def predict(self, emb, act_emb, lang_emb=None):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        lang_emb: optional (B, D) language embedding
        """
        lang_ctx = None
        if lang_emb is not None and self.fusion_type == "early":
            # Add language to action conditioning (broadcast over time)
            act_emb = act_emb + lang_emb.unsqueeze(1)
        elif lang_emb is not None and self.fusion_type == "cross_attn":
            # Pass as cross-attention context
            lang_ctx = lang_emb.unsqueeze(1)  # (B, 1, D)

        preds = self.predictor(emb, act_emb, lang_ctx=lang_ctx)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        lang_emb = _init.get("lang_emb")  # (B, D) or None
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")
        if lang_emb is not None:
            lang_emb = lang_emb.repeat_interleave(S, dim=0)  # (B, D) -> (BS, D)

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc, lang_emb=lang_emb)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc, lang_emb=lang_emb)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        if getattr(self, "_criterion_debug_count", 0) < 3:
            print(f"      [criterion debug] pred_emb: {pred_emb.shape}, "
                  f"goal_emb: {goal_emb.shape}", flush=True)
            # how much do predictions vary across samples?
            pred_last = pred_emb[0, :, -1, :]  # (S, D) — last-step predictions
            pred_std = pred_last.std(dim=0).mean().item()
            pred_mean_norm = pred_last.norm(dim=1).mean().item()
            goal_norm = goal_emb.norm().item()
            print(f"      [criterion debug] pred last-step: mean_norm={pred_mean_norm:.2f}, "
                  f"cross-sample std={pred_std:.4f}", flush=True)
            print(f"      [criterion debug] goal_emb norm={goal_norm:.2f}", flush=True)
            # cosine sim between pred and goal
            cos_sim = F.cosine_similarity(pred_last, goal_emb.squeeze().unsqueeze(0).expand_as(pred_last), dim=1)
            print(f"      [criterion debug] cos_sim(pred, goal): mean={cos_sim.mean():.4f}, "
                  f"std={cos_sim.std():.4f}", flush=True)
            self._criterion_debug_count = getattr(self, "_criterion_debug_count", 0) + 1

        # ensure goal_emb has same ndim as pred_emb (may be missing S dim)
        while goal_emb.ndim < pred_emb.ndim:
            goal_emb = goal_emb.unsqueeze(1)
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
