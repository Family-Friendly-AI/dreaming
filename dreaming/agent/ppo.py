"""CleanRL-style GAE + clipped PPO; the Q-ensemble is regressed to the GAE returns on wake data."""

from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from dreaming.config import PPOConfig


def autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def compute_gae(rewards, values, dones, next_value, next_done, gamma: float, lam: float):
    """Tensors shaped [T, N]; `dones[t]` marks that obs t is the first of a new episode (CleanRL)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal, nextv = 1.0 - next_done, next_value
        else:
            nonterminal, nextv = 1.0 - dones[t + 1], values[t + 1]
        delta = rewards[t] + gamma * nextv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def q_regression_loss(
    q_all: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """q_all [B, K, A], actions [B], targets [B], mask [B, K] (bootstrap mask) -> scalar."""
    q_a = q_all.gather(2, actions.view(-1, 1, 1).expand(-1, q_all.shape[1], 1)).squeeze(-1)
    err = (q_a - targets.unsqueeze(1)) ** 2
    if mask is None:
        return err.mean()
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def ppo_update(
    net: nn.Module, opt: torch.optim.Optimizer, batch: dict, cfg: PPOConfig, device: torch.device, amp: bool = False
) -> dict:
    obs, actions = batch["obs"], batch["actions"]
    old_logp, adv, ret, old_v = batch["logprobs"], batch["advantages"], batch["returns"], batch["values"]
    n = obs.shape[0]
    mb = max(1, n // cfg.num_minibatches)
    k = net.n_q_heads
    q_mask = (torch.rand(n, k, device=device) < cfg.q_bootstrap_p).float()
    stats = {"pg_loss": [], "v_loss": [], "q_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, mb):
            idx = perm[s : s + mb]
            with autocast(device, amp):
                logits, v, q = net(obs[idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])
            ratio = (logp - old_logp[idx]).exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - (logp - old_logp[idx])).mean()
                clipfrac = ((ratio - 1).abs() > cfg.clip_coef).float().mean()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            v_clipped = old_v[idx] + (v - old_v[idx]).clamp(-cfg.clip_coef, cfg.clip_coef)
            v_loss = 0.5 * torch.max((v - ret[idx]) ** 2, (v_clipped - ret[idx]) ** 2).mean()
            q_loss = q_regression_loss(q, actions[idx], ret[idx], q_mask[idx])
            ent = dist.entropy().mean()
            loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss + cfg.q_coef * q_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            for key, val in zip(stats, (pg_loss, v_loss, q_loss, ent, approx_kl, clipfrac), strict=True):
                stats[key].append(val.item())
    with torch.no_grad():
        var_y = ret.var()
        ev = (1 - (ret - old_v).var() / var_y).item() if var_y > 0 else float("nan")
    out = {f"ppo/{k}": float(np.mean(v)) for k, v in stats.items()}
    out["ppo/explained_variance"] = ev
    return out
