"""Entry point: `python -m dreaming.train --method ppo ...` on Craftax-Classic-Symbolic.

Cycle: wake -> dream -> consolidate -> gate -> reset (method `ppo` runs wake only).
"""

from __future__ import annotations

import dataclasses
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from dreaming.agent.networks import ActorCritic, apply_fn, init_params
from dreaming.agent.ppo import make_optimizer
from dreaming.config import Config, apply_method
from dreaming.envs import craftax as E
from dreaming.metrics import EpisodeTracker, RunLogger
from dreaming.phases.wake import Episodes, init_wake, make_wake_iteration
from dreaming.state import StepCounters, TrainState


def resolve_device(name: str):
    if name == "auto":
        return jax.devices()[0]
    try:
        return jax.devices(name)[0]
    except RuntimeError as e:
        raise RuntimeError(
            f"--device {name} requested but JAX sees only {jax.devices()}. On Snellius run `uv sync` on Linux so "
            "jax[cuda12] is installed (and drop `module load CUDA` if it shadows the pip CUDA wheels), "
            "or pass --device cpu."
        ) from e


def episode_records(ep: Episodes) -> list[dict]:
    done = np.asarray(ep.done)
    if not done.any():
        return []
    t_idx, n_idx = np.nonzero(done)
    order = np.lexsort((n_idx, t_idx))
    ret, length, ach = np.asarray(ep.ret), np.asarray(ep.length), np.asarray(ep.achievements)
    return [
        {
            "r": float(ret[t, n]),
            "l": int(length[t, n]),
            "achievements": [a for a, hit in zip(E.ACHIEVEMENTS, ach[t, n], strict=True) if hit],
        }
        for t, n in zip(t_idx[order], n_idx[order], strict=True)
    ]


def train(cfg: Config) -> dict:
    cfg = apply_method(cfg)
    device = resolve_device(cfg.device)
    with jax.default_device(device):
        return _train(cfg, device.platform)


def _train(cfg: Config, platform: str) -> dict:
    run_name = cfg.run_name or f"craftax_{cfg.method}_s{cfg.seed}_{int(time.time())}"
    logger = RunLogger(
        os.path.join(cfg.run_dir, run_name), dataclasses.asdict(cfg), cfg.track, cfg.wandb_project, run_name
    )
    print(f"platform={platform} devices={jax.devices()}", flush=True)

    env_params = E.env_params(cfg.env.max_episode_steps)
    compute_dtype = jnp.bfloat16 if cfg.amp and platform == "gpu" else jnp.float32
    net = ActorCritic(E.NUM_ACTIONS, cfg.n_q_heads, compute_dtype=compute_dtype)
    tx = make_optimizer(cfg.ppo.lr, cfg.ppo.max_grad_norm)
    root = jax.random.key(cfg.seed)
    fast = init_params(net, E.stream(root, E.KeyStream.INIT), E.OBS_DIM)
    wake_key = E.stream(root, E.KeyStream.WAKE)
    k_env, wake_key = jax.random.split(wake_key)
    ts = TrainState(
        fast=fast,
        opt=tx.init(fast),
        wake=init_wake(k_env, cfg.env.num_envs, env_params),
        key=wake_key,
        counters=StepCounters(),
    )
    wake_iteration = make_wake_iteration(cfg, apply_fn(net), tx, env_params)
    tracker = EpisodeTracker(list(E.ACHIEVEMENTS))
    steps_per_iter = cfg.ppo.num_steps * cfg.env.num_envs

    def used() -> int:
        return ts.counters.total if cfg.budget_axis == "total" else ts.counters.wake

    t0 = time.time()
    it = 0
    n_cycles = max(1, cfg.total_wake_steps // cfg.wake_steps_per_cycle)
    cycle = 0
    while used() < cfg.total_wake_steps and (cfg.budget_axis == "total" or cycle < n_cycles):
        cycle_start = ts.counters.wake
        while ts.counters.wake - cycle_start < cfg.wake_steps_per_cycle and used() < cfg.total_wake_steps:
            ts.key, k = jax.random.split(ts.key)
            ts.fast, ts.opt, ts.wake, out = wake_iteration(ts.fast, ts.opt, ts.wake, k)
            ts.counters.wake += steps_per_iter
            tracker.add(episode_records(out.episodes), ts.counters.wake)
            it += 1
            if it % cfg.log_every == 0:
                row = {
                    **ts.counters.row(),
                    **{k: float(v) for k, v in out.stats.items()},
                    **tracker.summary("train"),
                    "cycle": cycle,
                    "sps": ts.counters.total / (time.time() - t0),
                }
                logger.log(row)
        print(
            f"[cycle {cycle + 1}] {ts.counters.row()} "
            f"return={tracker.summary().get('train/return', float('nan')):.3f} "
            f"sps={ts.counters.total / (time.time() - t0):.0f}",
            flush=True,
        )
        cycle += 1
    summary = {
        **ts.counters.row(),
        **tracker.summary("final"),
        "budget_axis": cfg.budget_axis,
        "budget": cfg.total_wake_steps,
        "final/return_last100k": tracker.return_since(ts.counters.wake - 100_000),
        "unlock_step": tracker.unlock_step,
        "wall_time_s": time.time() - t0,
        "sps": ts.counters.total / (time.time() - t0),
    }
    logger.close(summary)
    return summary


def main():
    train(tyro.cli(Config))


if __name__ == "__main__":
    main()
