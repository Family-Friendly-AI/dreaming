"""Gate: paired held-out evaluation of candidate vs. old slow policy.

Both policies play the same held-out worlds (gate key stream, never seen in wake/dream) with the same
per-episode action noise (common random numbers), capped at `max_steps` per episode.
With `rotate_seeds` (default) every cycle draws a fresh disjoint world block and noise, and both policies
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

from dreaming.config import GateConfig


@dataclass
class GateState:
    history: set = field(default_factory=set)
    """Achievements ever unlocked by the (accepted) slow policy in gate evaluations."""
    n_decisions: int = 0
    n_rejections: int = 0
    n_forced: int = 0
    reject_streak: int = 0
    cached_old: tuple | None = None
    """(returns, achievements) of the current slow policy on the fixed held-out set."""

    @property
    def rejection_rate(self) -> float:
        return self.n_rejections / max(self.n_decisions, 1)


def paired_bootstrap_lower(delta: np.ndarray, n_boot: int, confidence: float, rng: np.random.Generator) -> float:
    if len(delta) == 0:
        return 0.0
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(1)
    return float(np.quantile(means, 1.0 - confidence))


def gate_decision(
    ret_cand: np.ndarray, ret_old: np.ndarray, ach_cand: set, history: set, cfg: GateConfig, rng: np.random.Generator
) -> tuple[bool, dict]:
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
