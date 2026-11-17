"""Device snapshot buffer (wake -> dream): stacked `EnvState`s kept top-C by |TD|."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.envs.craftax_state import EnvState
from flax import struct
from jax import Array

from dreaming.agent.networks import ApplyFn, Params


class SnapshotBuffer(struct.PyTreeNode):
    """Capacity-C pytree; `reward`/`done`/`next_obs` are the wake transition taken from `state` (for |TD|)."""

    state: EnvState
    obs: Array
    next_obs: Array
    reward: Array
    done: Array
    td: Array
    valid: Array

    @property
    def capacity(self) -> int:
        return self.td.shape[0]


def empty(template: EnvState, obs_dim: int, capacity: int) -> SnapshotBuffer:
    """All-invalid buffer shaped like `capacity` copies of one unbatched `template` state."""
    state = jax.tree_util.tree_map(lambda x: jnp.zeros((capacity, *x.shape), x.dtype), template)
    return SnapshotBuffer(
        state=state,
        obs=jnp.zeros((capacity, obs_dim), jnp.float32),
        next_obs=jnp.zeros((capacity, obs_dim), jnp.float32),
        reward=jnp.zeros(capacity, jnp.float32),
        done=jnp.zeros(capacity, bool),
        td=jnp.zeros(capacity, jnp.float32),
        valid=jnp.zeros(capacity, bool),
    )


def insert(buf: SnapshotBuffer, cand: SnapshotBuffer) -> SnapshotBuffer:
    """Keep the `buf.capacity` valid entries with the largest |TD| among `buf` and `cand`."""
    both = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), buf, cand)
    prio = jnp.where(both.valid, both.td, -jnp.inf)
    _, idx = jax.lax.top_k(prio, buf.capacity)
    return jax.tree_util.tree_map(lambda x: x[idx], both)


def td_errors(apply_fn: ApplyFn, params: Params, buf: SnapshotBuffer, gamma: float) -> Array:
    """|r + gamma V(s') (1 - d) - V(s)| under `params`; 0 for invalid slots."""
    _, v, _ = apply_fn(params, buf.obs)
    _, v_next, _ = apply_fn(params, buf.next_obs)
    td = jnp.abs(buf.reward + gamma * v_next * (1.0 - buf.done) - v)
    return jnp.where(buf.valid, td, 0.0)
