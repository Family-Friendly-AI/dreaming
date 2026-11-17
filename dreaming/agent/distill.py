"""Consolidation: distil the fast learner into a candidate slow learner.

L = KL(pi_f || pi_new) + beta_tr * KL(pi_s_old || pi_new) + value_coef * (|V_new - V_f|^2 + |Q_new - Q_f|^2)

The candidate starts from the old slow weights. Because pi, V and Q share one trunk, the critic and
Q-ensemble cannot be copied from the fast network without also overwriting the distilled policy trunk;
they are therefore regressed onto the fast outputs in the same pass (functionally a copy on the
cycle's observation distribution).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreaming.agent.ppo import autocast
from dreaming.config import ConsolidateConfig


def kl_categorical(logp_p: torch.Tensor, logp_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row from log-probabilities."""
    return (logp_p.exp() * (logp_p - logp_q)).sum(-1)


def distil(fast: nn.Module, slow_old: nn.Module, obs: np.ndarray, cfg: ConsolidateConfig,
           device: torch.device, amp: bool = False) -> tuple[nn.Module, dict]:
    cand = copy.deepcopy(slow_old)
    if obs is None or len(obs) == 0 or cfg.distil_epochs == 0:
        return cand, {"distil/kl_fast": float("nan")}
    o = torch.as_tensor(obs, device=device)
    n = o.shape[0]
    opt = torch.optim.Adam(cand.parameters(), lr=cfg.lr, eps=1e-5)
    stats = {"kl_fast": [], "kl_old": [], "value": []}
    for _ in range(cfg.distil_epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, cfg.minibatch_size):
            idx = perm[s:s + cfg.minibatch_size]
            with torch.no_grad(), autocast(device, amp):
                lf, vf, qf = fast(o[idx])
                lo = slow_old(o[idx])[0]
            with autocast(device, amp):
                ln, vn, qn = cand(o[idx])
            logp_n = F.log_softmax(ln.float(), -1)
            kl_f = kl_categorical(F.log_softmax(lf.float(), -1), logp_n).mean()
            kl_o = kl_categorical(F.log_softmax(lo.float(), -1), logp_n).mean()
            v_loss = F.mse_loss(vn.float(), vf.float()) + F.mse_loss(qn.float(), qf.float())
            loss = kl_f + cfg.beta_tr * kl_o + cfg.value_coef * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(cand.parameters(), 1.0)
            opt.step()
            stats["kl_fast"].append(kl_f.item())
            stats["kl_old"].append(kl_o.item())
            stats["value"].append(v_loss.item())
    return cand, {f"distil/{k}": float(np.mean(v)) for k, v in stats.items()} | {"distil/n_obs": n}
