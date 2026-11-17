"""Crafter with bit-exact snapshot restores.

`crafter.Env._balance_object` picks a creature to despawn by indexing into a list built from
`World._chunks[chunk]`, which is a `set` of objects. Set order follows `id()`, which changes
after unpickling, so restored copies diverge. Sorting by the object's index in the world's
object table makes the order a pure function of the (pickled) world state.
"""

from __future__ import annotations

import zlib

import crafter
import numpy as np

from dreaming.envs.base import EpisodeTrackingEnv

ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling",
    "collect_stone", "collect_wood", "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant", "place_stone",
    "place_table", "wake_up",
]


class DeterministicCrafter(crafter.Env):
    def _balance_object(self, chunk, objs, *args, **kwargs):
        obj_map = self._world._obj_map
        objs = sorted(objs, key=lambda o: int(obj_map[int(o.pos[0]), int(o.pos[1])]))
        return super()._balance_object(chunk, objs, *args, **kwargs)


class CrafterEnv(EpisodeTrackingEnv):
    n_actions = 17
    obs_shape = (64, 64, 3)
    default_gate_seeds = 32

    def __init__(self, seed: int = 0, max_episode_steps: int | None = None):
        super().__init__()
        self.env = DeterministicCrafter(
            area=(64, 64), size=(64, 64), length=max_episode_steps or 10000, seed=seed
        )

    def _reset(self, seed):
        if seed is not None:
            # Crafter seeds each world with hash((seed, episode)); pin both for reproducible layouts.
            self.env._seed = int(seed)
            self.env._episode = 0
        return self.env.reset()

    def _step(self, a):
        obs, r, done, info = self.env.step(a)
        ach = {k for k, v in info["achievements"].items() if v > 0}
        pos = info["player_pos"]
        inv = bytes(min(int(v), 2) for _, v in sorted(info["inventory"].items()))
        visit_key = (int(pos[0]) // 8, int(pos[1]) // 8, zlib.crc32(inv))
        return obs, r, done, ach, visit_key
