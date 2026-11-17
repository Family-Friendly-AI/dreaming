"""MiniGrid (gymnasium) wrapper: partial image observation, success as an 'achievement'."""

from __future__ import annotations

import zlib

import gymnasium as gym
import minigrid  # noqa: F401  (registers envs)

from dreaming.envs.base import EpisodeTrackingEnv


class MiniGridEnv(EpisodeTrackingEnv):
    n_actions = 7
    obs_shape = (7, 7, 3)
    default_gate_seeds = 50

    def __init__(self, env_id: str, seed: int = 0, max_episode_steps: int | None = None):
        super().__init__()
        kwargs = {} if max_episode_steps is None else {"max_steps": max_episode_steps}
        self.env = gym.make(env_id, **kwargs)
        self._seed = seed

    def _reset(self, seed):
        obs, _ = self.env.reset(seed=seed if seed is not None else self._seed)
        self._seed = None
        return obs["image"]

    def _step(self, a):
        obs, r, term, trunc, _ = self.env.step(a)
        u = self.env.unwrapped
        ach = {"success"} if (term and r > 0) else set()
        carrying = None if u.carrying is None else u.carrying.type
        visit_key = (
            tuple(int(x) for x in u.agent_pos),
            int(u.agent_dir),
            carrying,
            zlib.crc32(u.grid.encode().tobytes()),
        )
        return obs["image"], r, term or trunc, ach, visit_key
