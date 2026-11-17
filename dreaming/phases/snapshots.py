"""Snapshot buffer (wake -> dream) and an observation reservoir (for consolidation)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import torch


@dataclass
class Snapshot:
    state: bytes
    obs: np.ndarray
    next_obs: np.ndarray
    reward: float
    done: bool
    """The wake transition taken from this state; used to recompute |TD| with the current critic."""
    wake_step: int
    td_error: float
    visit_key: Hashable


class SnapshotBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: list[Snapshot] = []
        self.visits: Counter = Counter()

    def __len__(self):
        return len(self.items)

    def add(self, snaps: list[Snapshot]):
        self.items.extend(snaps)
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda s: s.td_error, reverse=True)
            del self.items[self.capacity:]

    def count_visits(self, keys):
        self.visits.update(keys)

    @torch.no_grad()
    def recompute_priorities(self, net, device, gamma: float, batch_size: int = 512) -> np.ndarray:
        """Re-score every snapshot with the current critic: |r + gamma V(s') (1-d) - V(s)|."""
        if not self.items:
            return np.zeros(0)
        out = []
        for s in range(0, len(self.items), batch_size):
            chunk = self.items[s:s + batch_size]
            obs = torch.as_tensor(np.stack([c.obs for c in chunk]), device=device)
            nxt = torch.as_tensor(np.stack([c.next_obs for c in chunk]), device=device)
            r = torch.tensor([c.reward for c in chunk], device=device)
            d = torch.tensor([float(c.done) for c in chunk], device=device)
            v = net(obs)[1]
            v2 = net(nxt)[1]
            out.append((r + gamma * v2 * (1 - d) - v).abs().cpu().numpy())
        prio = np.concatenate(out)
        for c, p in zip(self.items, prio):
            c.td_error = float(p)
        return prio

    def visit_weights(self) -> np.ndarray:
        """Go-Explore-style weight 1/sqrt(visits + 1)."""
        return np.array([1.0 / np.sqrt(self.visits[s.visit_key] + 1.0) for s in self.items])


class ObsReservoir:
    """Uniform reservoir sample of observations seen during the current cycle."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.buf: list[np.ndarray] = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.buf)

    def add(self, obs: np.ndarray):
        for o in obs:
            self.seen += 1
            if len(self.buf) < self.capacity:
                self.buf.append(o.copy())
            else:
                j = self.rng.integers(0, self.seen)
                if j < self.capacity:
                    self.buf[j] = o.copy()

    def array(self) -> np.ndarray:
        return np.stack(self.buf)

    def clear(self):
        self.buf, self.seen = [], 0
