"""Fast-learner update from dreams: Q-ensemble regression + advantage-weighted regression (AWR).

- Q_k(s, a) regresses onto discounted (V-bootstrapped) dream returns for every branch/control transition.
- The actor gets an AWR loss on branch roots, weight exp(clip(A / tau, max=awr_clip)) for A(s, a_rare) > 0
  (A <= 0 -> weight 0: only rare actions that beat the matched control are reinforced; with all-positive
  weights on rare-only roots, pi would be pushed towards every rare action regardless of outcome).
- KL(pi_before || pi) on the roots keeps the update local (pi_f is consolidated and gated afterwards).
No importance ratios are used anywhere.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreaming.agent.ppo import autocast, q_regression_loss
from dreaming.config import DreamConfig


def awr_weights(adv: torch.Tensor, tau: float, clip: float) -> torch.Tensor:
    w = torch.exp((adv / tau).clamp(max=clip))
    return torch.where(adv > 0, w, torch.zeros_like(w))


def dream_update(net: nn.Module, opt: torch.optim.Optimizer, trans: dict, roots: dict,
                 cfg: DreamConfig, device: torch.device, amp: bool = False,
                 max_grad_norm: float = 0.5) -> dict:
    """trans: obs [N,...] uint8, actions [N], returns [N]; roots: obs [R,...], actions [R], adv [R]."""
    t_obs = torch.as_tensor(trans["obs"], device=device)
    t_act = torch.as_tensor(trans["actions"], device=device, dtype=torch.long)
    t_ret = torch.as_tensor(trans["returns"], device=device, dtype=torch.float32)
    r_obs = torch.as_tensor(roots["obs"], device=device)
    r_act = torch.as_tensor(roots["actions"], device=device, dtype=torch.long)
    r_adv = torch.as_tensor(roots["adv"], device=device, dtype=torch.float32)
    w = awr_weights(r_adv, cfg.tau, cfg.awr_clip)

    with torch.no_grad(), autocast(device, amp):
        logp_before = F.log_softmax(net(r_obs)[0].float(), -1)

    n = t_obs.shape[0]
    k = net.n_q_heads
    mask = (torch.rand(n, k, device=device) < 0.8).float()
    stats = {"q_loss": [], "awr_loss": [], "anchor_kl": []}
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, cfg.minibatch_size):
            idx = perm[s:s + cfg.minibatch_size]
            with autocast(device, amp):
                _, _, q = net(t_obs[idx])
                logits_r = net(r_obs)[0]
            q_loss = q_regression_loss(q.float(), t_act[idx], t_ret[idx], mask[idx])
            logp = F.log_softmax(logits_r.float(), -1)
            awr = -(w * logp.gather(1, r_act[:, None]).squeeze(1)).mean()
            anchor = (logp_before.exp() * (logp_before - logp)).sum(-1).mean()
            loss = cfg.q_coef * q_loss + cfg.awr_coef * awr + cfg.anchor_coef * anchor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            stats["q_loss"].append(q_loss.item())
            stats["awr_loss"].append(awr.item())
            stats["anchor_kl"].append(anchor.item())
    out = {f"dream/{k}": float(np.mean(v)) for k, v in stats.items() if v}
    out["dream/awr_pos_frac"] = float((r_adv > 0).float().mean().item()) if len(r_adv) else 0.0
    return out
