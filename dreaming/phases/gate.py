"""Gate: paired held-out evaluation of candidate vs. old slow policy.

Both policies play the same held-out seeds (>= GATE_SEED_BASE, never seen in wake/dream) with the same
per-episode Gumbel noise for action sampling (common random numbers), capped at `max_steps` per episode.
With `rotate_seeds` (default) every cycle draws a fresh disjoint seed block and noise, and both policies
are re-evaluated on it, so the reference is never a stale (possibly lucky) score. Without it, the set is
fixed for the run and the old slow policy's results are cached (only the candidate is evaluated).
Accept if
  - non-inferiority: paired-bootstrap lower bound of mean(R_cand - R_old) >= -max(eps * |mean(R_old)|, eps_abs), or
  - new-skill override: the candidate unlocks an achievement the old slow policy has never unlocked, or
  - forced: `max_reject_streak` consecutive rejections already happened (escape from a frozen policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dreaming.agent.ppo import autocast
from dreaming.config import GateConfig
from dreaming.envs import GATE_SEED_BASE, EnvPool


@dataclass
class GateState:
    history: set = field(default_factory=set)
    """Achievements ever unlocked by the (accepted) slow policy in gate evaluations."""
    n_decisions: int = 0
    n_rejections: int = 0
    n_forced: int = 0
    reject_streak: int = 0
    noise_seed: int | None = None
    cached_old: tuple | None = None
    """(returns, achievements) of the current slow policy on the fixed held-out set."""

    @property
    def rejection_rate(self) -> float:
        return self.n_rejections / max(self.n_decisions, 1)


def gate_seeds(n: int, block: int = 0) -> list[int]:
    """Held-out seeds of block `block` (disjoint across blocks)."""
    return [GATE_SEED_BASE + block * n + i for i in range(n)]


@torch.no_grad()
def evaluate(pool: EnvPool, net, seeds: list[int], max_steps: int, device, amp: bool = False,
             noise_seed: int = 0) -> tuple[np.ndarray, list[set]]:
    """Run one capped episode per seed; returns (returns [n], achievements per episode)."""
    rets = np.zeros(len(seeds), dtype=np.float64)
    achs: list[set] = [set() for _ in seeds]
    N = pool.num_envs
    for w0 in range(0, len(seeds), N):
        ws = seeds[w0:w0 + N]
        idx = list(range(len(ws)))
        obs = pool.reset(idx, ws)
        rngs = [np.random.default_rng([noise_seed, s]) for s in ws]
        done = np.zeros(len(ws), dtype=bool)
        for _ in range(max_steps):
            active = [j for j in idx if not done[j]]
            if not active:
                break
            with autocast(device, amp):
                logits = net(torch.as_tensor(obs[active], device=device))[0].float().cpu().numpy()
            acts = []
            for row, j in enumerate(active):
                u = rngs[j].random(logits.shape[1]).clip(1e-12, 1 - 1e-12)
                acts.append(int(np.argmax(logits[row] - np.log(-np.log(u)))))
            nobs, r, d, infos, _ = pool.step(acts, idx=active, phase="gate", auto_reset=False)
            for row, j in enumerate(active):
                obs[j] = nobs[row]
                rets[w0 + j] += r[row]
                achs[w0 + j].update(infos[row].get("unlocked", ()))
                done[j] = d[row]
    return rets, achs


def paired_bootstrap_lower(delta: np.ndarray, n_boot: int, confidence: float,
                           rng: np.random.Generator) -> float:
    if len(delta) == 0:
        return 0.0
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(1)
    return float(np.quantile(means, 1.0 - confidence))


def gate_decision(ret_cand: np.ndarray, ret_old: np.ndarray, ach_cand: set, history: set,
                  cfg: GateConfig, rng: np.random.Generator) -> tuple[bool, dict]:
    delta = np.asarray(ret_cand, dtype=np.float64) - np.asarray(ret_old, dtype=np.float64)
    lower = paired_bootstrap_lower(delta, cfg.n_boot, cfg.confidence, rng)
    threshold = -max(cfg.eps * abs(float(np.mean(ret_old))), cfg.eps_abs)
    non_inferior = lower >= threshold
    new_skills = sorted(set(ach_cand) - set(history))
    accept = bool(non_inferior or new_skills)
    return accept, {
        "gate/accepted": float(accept),
        "gate/non_inferior": float(non_inferior),
        "gate/new_skill_override": float(bool(new_skills) and not non_inferior),
        "gate/new_skills": ",".join(new_skills),
        "gate/delta_mean": float(delta.mean()) if len(delta) else 0.0,
        "gate/delta_lower": lower,
        "gate/threshold": threshold,
        "gate/return_cand": float(np.mean(ret_cand)),
        "gate/return_old": float(np.mean(ret_old)),
    }


def run_gate(pool: EnvPool, cand, old, state: GateState, cfg: GateConfig, device, amp: bool,
             rng: np.random.Generator, cycle: int) -> tuple[bool, dict]:
    n = cfg.n_seeds or pool.default_gate_seeds
    start = pool.counters["gate"]
    if state.noise_seed is None:
        state.noise_seed = int(rng.integers(0, 2**31))
    if cfg.rotate_seeds:
        seeds = gate_seeds(n, block=cycle)
        noise = int(np.random.default_rng([state.noise_seed, cycle]).integers(0, 2**31))
        r_old, a_old = evaluate(pool, old, seeds, cfg.max_steps, device, amp, noise)
    else:
        seeds, noise = gate_seeds(n), state.noise_seed
        if state.cached_old is None:
            state.cached_old = evaluate(pool, old, seeds, cfg.max_steps, device, amp, noise)
        r_old, a_old = state.cached_old
    r_cand, a_cand = evaluate(pool, cand, seeds, cfg.max_steps, device, amp, noise)
    old_union = set().union(*a_old)
    cand_union = set().union(*a_cand)
    state.history |= old_union
    accept, info = gate_decision(r_cand, r_old, cand_union, state.history, cfg, rng)
    forced = (not accept and cfg.max_reject_streak > 0
              and state.reject_streak >= cfg.max_reject_streak)
    accept = accept or forced
    state.n_decisions += 1
    if accept:
        state.history |= cand_union
        state.cached_old = (r_cand, a_cand)
        state.reject_streak = 0
        state.n_forced += int(forced)
    else:
        state.n_rejections += 1
        state.reject_streak += 1
    info.update({
        "gate/accepted": float(accept),
        "gate/forced": float(forced),
        "gate/reject_streak": state.reject_streak,
        "gate/rejection_rate": state.rejection_rate,
        "gate/env_steps": pool.counters["gate"] - start,
        "gate/n_ach_cand": len(cand_union),
        "gate/n_ach_old": len(old_union),
    })
    return accept, info
