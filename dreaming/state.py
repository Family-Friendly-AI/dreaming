"""Typed host-side training state carried across wake iterations and sleep cycles."""

from __future__ import annotations

from dataclasses import dataclass

import optax
from jax import Array

from dreaming.agent.networks import Params
from dreaming.phases.dream import DreamPhase
from dreaming.phases.snapshots import SnapshotBuffer
from dreaming.phases.wake import WakeState


@dataclass
class StepCounters:
    """Active env steps per phase (padded or masked steps are never counted)."""

    wake: int = 0
    dream: int = 0
    gate: int = 0

    @property
    def total(self) -> int:
        return self.wake + self.dream + self.gate

    def row(self) -> dict[str, int]:
        return {"steps/wake": self.wake, "steps/dream": self.dream, "steps/gate": self.gate, "steps/total": self.total}


@dataclass
class Dreams:
    """Dream-only state; `ref` is pi_f at the start of the cycle (reference policy of the KL rarity term)."""

    phase: DreamPhase
    ref: Params
    opt: optax.OptState
    buffer: SnapshotBuffer
    key: Array


@dataclass
class TrainState:
    fast: Params
    opt: optax.OptState
    wake: WakeState
    key: Array
    counters: StepCounters
    dreams: Dreams | None
    """`None` for `ppo`."""
