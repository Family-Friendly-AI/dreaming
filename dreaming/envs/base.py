"""Common snapshot-able environment interface used by wake, dream and gate."""

from __future__ import annotations

import pickle
from typing import Any, Protocol

import numpy as np


class SnapshotEnv(Protocol):
    n_actions: int
    obs_shape: tuple[int, ...]

    def reset(self, seed: int | None = None) -> np.ndarray: ...

    def step(self, a: int) -> tuple[np.ndarray, float, bool, dict]: ...

    def get_state(self) -> bytes: ...

    def set_state(self, s: bytes) -> np.ndarray: ...


class EpisodeTrackingEnv:
    """Base class: episode statistics + pickle-based snapshots of the whole wrapper.

    Subclasses implement `_reset(seed)`, `_step(a) -> (obs, r, done, achievements, visit_key)`.
    The pickled state contains the underlying env (incl. its RNG) and the episode counters,
    so a restored env continues exactly as the original would.
    """

    n_actions: int
    obs_shape: tuple[int, ...]

    def __init__(self):
        self._obs: np.ndarray | None = None
        self._ret = 0.0
        self._len = 0
        self._ach: set[str] = set()

    def _reset(self, seed: int | None) -> np.ndarray:
        raise NotImplementedError

    def _step(self, a: int) -> tuple[np.ndarray, float, bool, set[str], Any]:
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._obs = self._reset(seed)
        self._ret, self._len, self._ach = 0.0, 0, set()
        return self._obs

    def step(self, a: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, r, done, ach, visit_key = self._step(int(a))
        self._ret += float(r)
        self._len += 1
        new = ach - self._ach
        self._ach |= ach
        self._obs = obs
        info: dict = {"visit_key": visit_key}
        if new:
            info["unlocked"] = sorted(new)
        if done:
            info["episode"] = {"r": self._ret, "l": self._len, "achievements": sorted(self._ach)}
        return obs, float(r), bool(done), info

    def get_state(self) -> bytes:
        return pickle.dumps(self.__dict__, protocol=pickle.HIGHEST_PROTOCOL)

    def set_state(self, s: bytes) -> np.ndarray:
        self.__dict__.update(pickle.loads(s))
        return self._obs
