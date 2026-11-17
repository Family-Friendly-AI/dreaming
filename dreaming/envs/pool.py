"""Environment pool shared by wake, dream and gate.

Every env index can be stepped, reset or restored independently, so dream branches and gate
episodes can run on any subset of workers. Env steps are counted per phase name.

Seeds: automatic wake resets draw from [0, 2**30); held-out gate seeds live at >= GATE_SEED_BASE,
so gate worlds are never seen in wake or dreams.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from typing import Sequence

import numpy as np

GATE_SEED_BASE = 2**30


def make_env(name: str, seed: int = 0, max_episode_steps: int | None = None):
    if name == "crafter":
        from dreaming.envs.crafter_env import CrafterEnv
        return CrafterEnv(seed=seed, max_episode_steps=max_episode_steps)
    if name.startswith("minigrid:"):
        from dreaming.envs.minigrid_env import MiniGridEnv
        return MiniGridEnv(name.split(":", 1)[1], seed=seed, max_episode_steps=max_episode_steps)
    raise ValueError(f"unknown env {name!r}")


class _Worker:
    def __init__(self, name: str, seed: int, idx: int, max_episode_steps: int | None):
        self.env = make_env(name, seed, max_episode_steps)
        self.rng = np.random.default_rng([seed, idx])

    def _wake_seed(self) -> int:
        return int(self.rng.integers(0, GATE_SEED_BASE))

    def handle(self, cmd: str, data):
        if cmd == "reset":
            return self.env.reset(self._wake_seed() if data is None else int(data))
        if cmd == "step":
            action, auto_reset, want_state = data
            state = self.env.get_state() if want_state else None
            obs, r, done, info = self.env.step(action)
            if done and auto_reset:
                obs = self.env.reset(self._wake_seed())
            return obs, r, done, info, state
        if cmd == "get_state":
            return self.env.get_state()
        if cmd == "set_state":
            return self.env.set_state(data)
        if cmd == "spec":
            return self.env.n_actions, self.env.obs_shape, self.env.default_gate_seeds
        raise ValueError(cmd)


def _worker_loop(remote, name, seed, idx, max_episode_steps):
    worker = _Worker(name, seed, idx, max_episode_steps)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "close":
                break
            remote.send(worker.handle(cmd, data))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        remote.close()


class EnvPool:
    def __init__(self, name: str, num_envs: int, seed: int = 0,
                 max_episode_steps: int | None = None, mode: str = "process"):
        self.num_envs = num_envs
        self.mode = mode
        self.counters: dict[str, int] = defaultdict(int)
        if mode == "serial":
            self._workers = [_Worker(name, seed, i, max_episode_steps) for i in range(num_envs)]
        elif mode == "process":
            ctx = mp.get_context("spawn")
            self._remotes, self._procs = [], []
            for i in range(num_envs):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker_loop, args=(child, name, seed, i, max_episode_steps),
                                daemon=True)
                p.start()
                child.close()
                self._remotes.append(parent)
                self._procs.append(p)
        else:
            raise ValueError(mode)
        self.n_actions, self.obs_shape, self.default_gate_seeds = self._call([0], "spec", [None])[0]
        self._closed = False

    # ------------------------------------------------------------------ internals
    def _call(self, idx: Sequence[int], cmd: str, data: Sequence) -> list:
        if self.mode == "serial":
            return [self._workers[i].handle(cmd, d) for i, d in zip(idx, data)]
        for i, d in zip(idx, data):
            self._remotes[i].send((cmd, d))
        return [self._remotes[i].recv() for i in idx]

    def _idx(self, idx) -> list[int]:
        return list(range(self.num_envs)) if idx is None else [int(i) for i in idx]

    # ------------------------------------------------------------------ API
    def reset(self, idx: Sequence[int] | None = None, seeds: Sequence[int] | None = None) -> np.ndarray:
        idx = self._idx(idx)
        seeds = [None] * len(idx) if seeds is None else list(seeds)
        return np.stack(self._call(idx, "reset", seeds))

    def step(self, actions, idx: Sequence[int] | None = None, phase: str = "wake",
             auto_reset: bool = True, snapshot_idx: Sequence[int] | None = None):
        """Step envs `idx` with `actions`.

        Returns `(obs, rew, done, infos, states)`; `states[j]` is the pickled state *before* the
        action for positions j whose env index is in `snapshot_idx`, else None.
        """
        idx = self._idx(idx)
        want = set() if snapshot_idx is None else {int(i) for i in snapshot_idx}
        data = [(int(a), auto_reset, i in want) for i, a in zip(idx, actions)]
        out = self._call(idx, "step", data)
        self.counters[phase] += len(idx)
        obs = np.stack([o[0] for o in out])
        rew = np.array([o[1] for o in out], dtype=np.float32)
        done = np.array([o[2] for o in out], dtype=bool)
        infos = [o[3] for o in out]
        states = [o[4] for o in out]
        return obs, rew, done, infos, states

    def get_state(self, idx: Sequence[int]) -> list[bytes]:
        idx = self._idx(idx)
        return self._call(idx, "get_state", [None] * len(idx))

    def set_state(self, idx: Sequence[int], states: Sequence[bytes]) -> np.ndarray:
        idx = self._idx(idx)
        return np.stack(self._call(idx, "set_state", list(states)))

    @property
    def total_steps(self) -> int:
        return sum(self.counters.values())

    def close(self):
        if self._closed or self.mode == "serial":
            self._closed = True
            return
        for r in self._remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
