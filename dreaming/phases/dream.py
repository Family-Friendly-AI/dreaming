"""Dream phase: targeted deviate-then-follow branches from real snapshots, with matched controls.

Per selected snapshot s with rare action a (repeated k ~ U{1..k_max} times):
  for m in 1..M:  restore(s); a x k, then follow pi_f for H steps; R_branch = sum gamma^t r + gamma^L V(s_L)
                  restore(s); follow pi_f for k + H steps;          R_ctrl   = ...
  A(s, a) = mean(R_branch) - mean(R_ctrl)
Branch m and control m share the restored env state (incl. its RNG) and the same Gumbel noise for
pi_f's action sampling (common random numbers), so A isolates the effect of the deviation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from dreaming.agent.dream_update import dream_update
from dreaming.agent.ppo import autocast
from dreaming.config import Config, DreamConfig
from dreaming.envs import EnvPool
from dreaming.phases.snapshots import ObsReservoir, SnapshotBuffer


@dataclass
class RolloutSpec:
    state: bytes
    prefix: list[int]
    length: int
    noise: np.ndarray  # [length, A] Gumbel noise for pi_f sampling


@dataclass
class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    done: bool
    bootstrap: float
    returns: np.ndarray = field(default=None)


def discounted_returns(rewards: np.ndarray, bootstrap: float, gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    g = bootstrap
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def gumbel(rng: np.random.Generator, shape) -> np.ndarray:
    u = rng.random(shape).clip(1e-12, 1 - 1e-12)
    return -np.log(-np.log(u)).astype(np.float32)


def score_actions(
    logits_f: torch.Tensor, logits_s: torch.Tensor, q: torch.Tensor, prio: torch.Tensor, cfg: DreamConfig
) -> torch.Tensor:
    """P(s, a) = prio(s) * rare(s, a) * (1 + eta * std_k Q_k(s, a)); returns [S, A] (non-negative)."""
    logp_f = F.log_softmax(logits_f, -1)
    logp_s = F.log_softmax(logits_s, -1)
    surprisal = -logp_f
    kl = (logp_f.exp() * (logp_f - logp_s)).sum(-1, keepdim=True).clamp_min(0.0)
    if cfg.rarity == "both":
        rare = surprisal + cfg.lam * kl
    elif cfg.rarity == "surprisal":
        rare = surprisal
    else:
        rare = kl.expand_as(surprisal) + 1e-6
    bonus = 1.0 + cfg.eta * q.std(dim=1)
    return prio[:, None] * rare * bonus


@torch.no_grad()
def select_branches(
    buffer: SnapshotBuffer, fast, slow, device, cfg: Config, n: int, rng: np.random.Generator
) -> tuple[list[tuple[int, int, int]], dict]:
    """Returns [(snapshot index, rare action, k)] and diagnostics."""
    dc = cfg.dream
    td = buffer.recompute_priorities(fast, device, cfg.ppo.gamma)
    state_prio = buffer.visit_weights() if dc.selection == "go_explore" else td
    obs = torch.as_tensor(np.stack([s.obs for s in buffer.items]), device=device)
    logits_f, _, q = fast(obs)
    logits_s = slow(obs)[0]
    P = (
        score_actions(
            logits_f.float(),
            logits_s.float(),
            q.float(),
            torch.as_tensor(state_prio, device=device, dtype=torch.float32),
            dc,
        )
        .cpu()
        .numpy()
    )
    P = np.nan_to_num(P, nan=0.0, posinf=0.0).clip(min=0.0) + 1e-12
    if dc.selection == "random":
        w_state = td + 1e-8
    else:
        w_state = P.sum(1)
    n = min(n, len(buffer))
    chosen = rng.choice(len(buffer), size=n, replace=False, p=w_state / w_state.sum())
    out = []
    n_actions = P.shape[1]
    for j in chosen:
        if dc.selection == "random":
            a = int(rng.integers(n_actions))
        else:
            a = int(rng.choice(n_actions, p=P[j] / P[j].sum()))
        out.append((int(j), a, int(rng.integers(1, dc.k_max + 1))))
    diag = {"dream/td_prio_mean": float(td.mean()), "dream/buffer_size": len(buffer)}
    return out, diag


@torch.no_grad()
def run_rollouts(
    pool: EnvPool, net, device, specs: list[RolloutSpec], gamma: float, amp: bool = False, phase: str = "dream"
) -> list[Rollout]:
    """Execute rollouts in waves of `pool.num_envs`, each starting from `spec.state`."""
    results: list[Rollout] = []
    N = pool.num_envs
    for w0 in range(0, len(specs), N):
        wave = specs[w0 : w0 + N]
        idx = list(range(len(wave)))
        obs = pool.set_state(idx, [s.state for s in wave])
        L = max(s.length for s in wave)
        o_buf = [[] for _ in wave]
        a_buf = [[] for _ in wave]
        r_buf = [[] for _ in wave]
        done = np.zeros(len(wave), dtype=bool)
        for t in range(L):
            active = [j for j, s in enumerate(wave) if t < s.length and not done[j]]
            if not active:
                break
            with autocast(device, amp):
                logits = net(torch.as_tensor(obs[active], device=device))[0].float().cpu().numpy()
            acts = []
            for row, j in enumerate(active):
                s = wave[j]
                a = s.prefix[t] if t < len(s.prefix) else int(np.argmax(logits[row] + s.noise[t]))
                acts.append(a)
                o_buf[j].append(obs[j].copy())
                a_buf[j].append(a)
            nobs, r, d, _, _ = pool.step(acts, idx=active, phase=phase, auto_reset=False)
            for row, j in enumerate(active):
                obs[j] = nobs[row]
                r_buf[j].append(float(r[row]))
                done[j] = bool(d[row])
        alive = [j for j in range(len(wave)) if not done[j]]
        boot = np.zeros(len(wave), dtype=np.float32)
        if alive:
            with autocast(device, amp):
                boot[alive] = net(torch.as_tensor(obs[alive], device=device))[1].float().cpu().numpy()
        for j in range(len(wave)):
            rew = np.array(r_buf[j], dtype=np.float32)
            ro = Rollout(
                obs=np.stack(o_buf[j]),
                actions=np.array(a_buf[j]),
                rewards=rew,
                done=bool(done[j]),
                bootstrap=float(boot[j]),
            )
            ro.returns = discounted_returns(rew, ro.bootstrap, gamma)
            results.append(ro)
    return results


def build_specs(
    buffer: SnapshotBuffer, branches, cfg: DreamConfig, n_actions: int, rng: np.random.Generator
) -> list[RolloutSpec]:
    """Order: for each branch, M (branch, control) pairs -> [b0, c0, b1, c1, ...]."""
    specs = []
    for j, a, k in branches:
        state = buffer.items[j].state
        L = k + cfg.horizon
        for _ in range(cfg.m_rollouts):
            noise = gumbel(rng, (L, n_actions))
            specs.append(RolloutSpec(state=state, prefix=[a] * k, length=L, noise=noise))
            specs.append(RolloutSpec(state=state, prefix=[], length=L, noise=noise))
    return specs


def dream_phase(
    cfg: Config,
    pool: EnvPool,
    fast,
    slow,
    buffer: SnapshotBuffer,
    device,
    wake_steps_this_cycle: int,
    rng: np.random.Generator,
    reservoir: ObsReservoir | None = None,
    opt: torch.optim.Optimizer | None = None,
) -> dict:
    dc = cfg.dream
    cost = 2 * dc.m_rollouts * (dc.k_max + dc.horizon)
    n = int(dc.budget_frac * wake_steps_this_cycle) // cost
    if dc.max_snapshots is not None:
        n = min(n, dc.max_snapshots)
    if len(buffer) == 0 or n == 0:
        warnings.warn("dream phase skipped: empty snapshot buffer or zero budget", stacklevel=2)
        return {"dream/snapshots": 0}

    start = pool.counters["dream"]
    branches, diag = select_branches(buffer, fast, slow, device, cfg, n, rng)
    specs = build_specs(buffer, branches, dc, pool.n_actions, rng)
    rolls = run_rollouts(pool, fast, device, specs, cfg.ppo.gamma, cfg.amp)

    M = dc.m_rollouts
    adv, root_obs, root_act = [], [], []
    for b, (j, a, _) in enumerate(branches):
        group = rolls[b * 2 * M : (b + 1) * 2 * M]
        r_branch = np.mean([g.returns[0] for g in group[0::2]])
        r_ctrl = np.mean([g.returns[0] for g in group[1::2]])
        adv.append(r_branch - r_ctrl)
        root_obs.append(buffer.items[j].obs)
        root_act.append(a)
    adv = np.array(adv, dtype=np.float32)
    root_obs = np.stack(root_obs)
    root_act = np.array(root_act)
    trans = {
        "obs": np.concatenate([r.obs for r in rolls]),
        "actions": np.concatenate([r.actions for r in rolls]),
        "returns": np.concatenate([r.returns for r in rolls]),
    }

    def pi_rare():
        with torch.no_grad(), autocast(device, cfg.amp):
            lg = fast(torch.as_tensor(root_obs, device=device))[0].float()
        return F.softmax(lg, -1).gather(1, torch.as_tensor(root_act, device=device)[:, None]).squeeze(1)

    p_before = pi_rare()
    if opt is None:
        opt = torch.optim.Adam(fast.parameters(), lr=dc.lr, eps=1e-5)
    stats = dream_update(fast, opt, trans, {"obs": root_obs, "actions": root_act, "adv": adv}, dc, device, cfg.amp)
    p_after = pi_rare()
    pos = torch.as_tensor(adv > 0, device=device)

    if reservoir is not None:
        reservoir.add(trans["obs"][rng.random(len(trans["obs"])) < 0.1])

    stats.update(diag)
    stats.update(
        {
            "dream/snapshots": len(branches),
            "dream/env_steps": pool.counters["dream"] - start,
            "dream/A_mean": float(adv.mean()),
            "dream/A_std": float(adv.std()),
            "dream/A_max": float(adv.max()),
            "dream/pi_rare_before": float(p_before.mean().item()),
            "dream/pi_rare_after": float(p_after.mean().item()),
            "dream/pi_rare_after_posA": float(p_after[pos].mean().item()) if pos.any() else float("nan"),
            "dream/pi_rare_before_posA": float(p_before[pos].mean().item()) if pos.any() else float("nan"),
        }
    )
    return stats
