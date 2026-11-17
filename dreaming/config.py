"""Configuration dataclasses (parsed with tyro in `dreaming.train`)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

Method = Literal["ppo", "dream", "random_dream"]


@dataclass
class EnvConfig:
    name: Literal["Craftax-Classic-Symbolic-v1"] = "Craftax-Classic-Symbolic-v1"
    num_envs: int = 64
    """Parallel envs, vmapped on one device."""
    max_episode_steps: int | None = None
    """Episode cap (`EnvParams.max_timesteps`); `None` keeps the Craftax default of 10000."""


@dataclass
class PPOConfig:
    num_steps: int = 128
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    q_coef: float = 0.5
    max_grad_norm: float = 0.5
    q_bootstrap_p: float = 0.8
    """Per-head Bernoulli mask for Q-ensemble regression (keeps heads diverse)."""


@dataclass
class DreamConfig:
    selection: Literal["targeted", "random"] = "targeted"
    rarity: Literal["both", "surprisal", "kl"] = "both"
    budget_frac: float = 0.3
    """Dream env steps per cycle as a fraction of `wake_steps_per_cycle` (fixed per run: one compiled shape)."""
    k_max: int = 3
    """Rare action is repeated k ~ U{1..k_max} times."""
    horizon: int = 64
    """Follow-pi_f steps after the deviation; then bootstrap with V."""
    m_rollouts: int = 2
    """Branch rollouts per snapshot (plus the same number of matched controls)."""
    lam: float = 0.5
    eta: float = 1.0
    tau: float = 1.0
    awr_clip: float = 5.0
    awr_coef: float = 1.0
    q_coef: float = 0.5
    anchor_coef: float = 1.0
    """KL(pi_before || pi) on dream roots; keeps the AWR push local."""
    update_epochs: int = 4
    minibatch_size: int = 256
    lr: float = 2.5e-4
    buffer_size: int = 2000
    snapshot_prob: float = 0.05
    """Fraction of envs captured as snapshot candidates per wake step (ceil(snapshot_prob * num_envs) slots)."""
    max_snapshots: int | None = None
    """Optional hard cap on snapshots dreamed per cycle."""


@dataclass
class GateConfig:
    n_seeds: int = 0
    """Held-out seeds; 0 means env default (Crafter 32, MiniGrid 50)."""
    max_steps: int = 1000
    eps: float = 0.05
    """Relative non-inferiority margin (fraction of the old policy's mean return)."""
    eps_abs: float = 0.02
    """Absolute margin floor; avoids a zero margin when returns are ~0 early in training."""
    n_boot: int = 2000
    confidence: float = 0.95
    rotate_seeds: bool = True
    """Fresh held-out seeds + action noise every cycle, with *both* policies re-evaluated on them. Avoids
    a single lucky score of the old slow policy on a fixed seed set blocking all later candidates.
    `False` uses one fixed set for the run and caches the old policy's results (cheaper)."""
    max_reject_streak: int = 3
    """After this many consecutive rejections the next candidate is accepted (`gate/forced`); 0 disables."""


@dataclass
class Config:
    method: Method = "ppo"
    seed: int = 0
    total_wake_steps: int = 1_000_000
    """Env-step budget, counted on `budget_axis`."""
    budget_axis: Literal["wake", "total"] = "wake"
    """`total` counts wake + dream + gate steps against the budget (Crafter paper protocol: 1M env
    steps in total, success rates over all training episodes within the budget)."""
    wake_steps_per_cycle: int = 50_000
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dream: DreamConfig = field(default_factory=DreamConfig)
    n_q_heads: int = 5
    device: Literal["auto", "gpu", "cpu"] = "auto"
    """`gpu` fails fast when JAX sees no GPU."""
    amp: bool = True
    """bf16 matmuls on GPU."""
    track: bool = False
    """Log to wandb (CSV is always written)."""
    wandb_project: str = "dreaming"
    run_dir: str = "runs"
    run_name: str | None = None
    log_every: int = 1
    """Log every n PPO iterations."""
    video_every: int = 0
    """Record one pi_f episode as `videos/step_<steps/total>.gif` (wandb `video`) each time `steps/total` passes a
    multiple of n; 0 disables."""
    video_steps: int = 1000
    """Frame cap per video (fixed rollout length: one compiled shape)."""

    @property
    def dreams(self) -> bool:
        return self.method != "ppo"


def apply_method(cfg: Config) -> Config:
    """Translate `cfg.method` into component settings. Returns a modified copy."""
    cfg = dataclasses.replace(cfg, dream=dataclasses.replace(cfg.dream))
    if cfg.method == "random_dream":
        cfg.dream.selection = "random"
    return cfg
