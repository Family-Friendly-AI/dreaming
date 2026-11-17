"""Entry point: `python -m dreaming.train --method full --env.name crafter ...`.

Cycle: wake -> dream -> consolidate -> gate -> reset (method `ppo` runs wake only).
"""

from __future__ import annotations

import dataclasses
import os
import random
import time

import numpy as np
import torch
import tyro

from dreaming.agent.distill import distil
from dreaming.agent.networks import ActorCritic
from dreaming.config import Config, apply_method
from dreaming.envs import EnvPool
from dreaming.metrics import EpisodeTracker, RunLogger
from dreaming.phases.dream import dream_phase
from dreaming.phases.gate import GateState, run_gate
from dreaming.phases.snapshots import ObsReservoir, SnapshotBuffer
from dreaming.phases.wake import WakeRunner


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def env_achievements(name: str) -> list[str] | None:
    if name == "crafter":
        from dreaming.envs.crafter_env import ACHIEVEMENTS
        return ACHIEVEMENTS
    return ["success"]


def step_row(pool: EnvPool) -> dict:
    c = pool.counters
    return {"steps/wake": c["wake"], "steps/dream": c["dream"], "steps/gate": c["gate"],
            "steps/total": c["wake"] + c["dream"] + c["gate"]}


def train(cfg: Config) -> dict:
    cfg = apply_method(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    env_tag = cfg.env.name.split(":")[-1]
    run_name = cfg.run_name or f"{env_tag}_{cfg.method}_s{cfg.seed}_{int(time.time())}"
    logger = RunLogger(os.path.join(cfg.run_dir, run_name), dataclasses.asdict(cfg), cfg.track,
                       cfg.wandb_project, run_name)

    pool = EnvPool(cfg.env.name, cfg.env.num_envs, seed=cfg.seed,
                   max_episode_steps=cfg.env.max_episode_steps, mode=cfg.env.pool_mode)
    fast = ActorCritic(pool.obs_shape, pool.n_actions, cfg.n_q_heads).to(device)
    slow = ActorCritic(pool.obs_shape, pool.n_actions, cfg.n_q_heads).to(device)
    slow.load_state_dict(fast.state_dict())
    wake = WakeRunner(cfg, pool, device, seed=cfg.seed)
    buffer = SnapshotBuffer(cfg.dream.buffer_size)
    reservoir = ObsReservoir(cfg.consolidate.obs_buffer, seed=cfg.seed)
    tracker = EpisodeTracker(env_achievements(cfg.env.name))
    ctx = dict(cfg=cfg, pool=pool, fast=fast, slow=slow, opt=new_optimizer(fast, cfg),
               dream_opt=new_optimizer(fast, cfg, cfg.dream.lr), device=device,
               buffer=buffer, reservoir=reservoir, tracker=tracker, logger=logger, wake=wake,
               gate_state=GateState(), rng=np.random.default_rng(cfg.seed + 777))

    t0 = time.time()
    it = 0
    n_cycles = max(1, cfg.total_wake_steps // cfg.wake_steps_per_cycle)

    def used() -> int:
        return pool.total_steps if cfg.budget_axis == "total" else pool.counters["wake"]

    try:
        cycle = 0
        while used() < cfg.total_wake_steps and (cfg.budget_axis == "total" or cycle < n_cycles):
            cycle_start = pool.counters["wake"]
            while (pool.counters["wake"] - cycle_start < cfg.wake_steps_per_cycle
                   and used() < cfg.total_wake_steps):
                stats, episodes = wake.iteration(fast, ctx["opt"], buffer if cfg.dream.enabled else None,
                                                 reservoir if cfg.cycles_enabled else None)
                tracker.add(episodes, pool.counters["wake"])
                it += 1
                if it % cfg.log_every == 0:
                    row = {**step_row(pool), **stats, **tracker.summary("train"), "cycle": cycle,
                           "sps": pool.total_steps / (time.time() - t0)}
                    logger.log(row)
            if cfg.cycles_enabled and used() < cfg.total_wake_steps:
                cycle_stats = sleep_cycle(ctx, cycle, pool.counters["wake"] - cycle_start)
                logger.log({**step_row(pool), **cycle_stats, "cycle": cycle})
            print(f"[cycle {cycle + 1}] {step_row(pool)} "
                  f"return={tracker.summary().get('train/return', float('nan')):.3f} "
                  f"sps={pool.total_steps / (time.time() - t0):.0f}", flush=True)
            cycle += 1
        summary = {**step_row(pool), **tracker.summary("final"), "budget_axis": cfg.budget_axis,
                   "budget": cfg.total_wake_steps,
                   "final/return_last100k": tracker.return_since(pool.counters["wake"] - 100_000),
                   "unlock_step": tracker.unlock_step, "wall_time_s": time.time() - t0}
    finally:
        pool.close()
    logger.close(summary)
    return summary


def new_optimizer(net: torch.nn.Module, cfg: Config, lr: float | None = None) -> torch.optim.Optimizer:
    return torch.optim.Adam(net.parameters(), lr=cfg.ppo.lr if lr is None else lr, eps=1e-5)


def sleep_cycle(ctx: dict, cycle: int, wake_steps: int) -> dict:
    """dream -> consolidate -> gate -> reset.

    Dream and gate reuse the wake workers, so in-progress wake episodes are saved and restored around
    the sleep phase (wake continues exactly where it stopped).
    """
    cfg, pool, fast, slow, device = ctx["cfg"], ctx["pool"], ctx["fast"], ctx["slow"], ctx["device"]
    t0 = time.time()
    wake_states = pool.get_state(None)
    stats: dict = {}

    if cfg.dream.enabled:
        stats |= dream_phase(cfg, pool, fast, slow, ctx["buffer"], device, wake_steps, ctx["rng"],
                             reservoir=ctx["reservoir"], opt=ctx["dream_opt"])

    obs = ctx["reservoir"].array() if len(ctx["reservoir"]) else None
    cand, dstats = distil(fast, slow, obs, cfg.consolidate, device, cfg.amp)
    stats |= dstats

    if cfg.gate.enabled:
        accept, gstats = run_gate(pool, cand, slow, ctx["gate_state"], cfg.gate, device, cfg.amp,
                                  ctx["rng"], cycle)
        stats |= gstats
    else:
        accept = True
        stats["gate/accepted"] = 1.0

    # Reset: pi_f := pi_s (accepted candidate, or the old slow policy -> true rollback).
    # load_state_dict copies in place, so the optimizers keep pointing at fast's parameters.
    if accept:
        slow.load_state_dict(cand.state_dict())
    if cfg.reset_fast:
        fast.load_state_dict(slow.state_dict())
    stats["reset/fast"] = float(cfg.reset_fast)
    reset_opt = cfg.opt_reset == "always" or (cfg.opt_reset == "on_reject" and not accept)
    if reset_opt:
        ctx["opt"] = new_optimizer(fast, cfg)
        ctx["dream_opt"] = new_optimizer(fast, cfg, cfg.dream.lr)
    stats["opt/reset"] = float(reset_opt)
    ctx["reservoir"].clear()

    obs = pool.set_state(None, wake_states)
    ctx["wake"].obs = obs
    stats["sleep/wall_s"] = time.time() - t0
    return stats


def main():
    train(tyro.cli(Config))


if __name__ == "__main__":
    main()
