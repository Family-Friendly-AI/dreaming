"""Shared-trunk actor-critic with a value head and a K-head Q-ensemble (Flax)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

Params = Any
ApplyFn = Callable[[Params, Array], tuple[Array, Array, Array]]


class ActorCritic(nn.Module):
    n_actions: int
    n_q_heads: int
    hidden: int = 512
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array, Array]:
        def dense(features: int, scale: float) -> nn.Dense:
            return nn.Dense(
                features,
                kernel_init=nn.initializers.orthogonal(scale),
                bias_init=nn.initializers.zeros,
                dtype=self.compute_dtype,
            )

        h = obs.astype(self.compute_dtype)
        for _ in range(2):
            h = nn.relu(dense(self.hidden, float(np.sqrt(2)))(h))
        logits = dense(self.n_actions, 0.01)(h).astype(jnp.float32)
        value = dense(1, 1.0)(h)[..., 0].astype(jnp.float32)
        q = dense(self.n_q_heads * self.n_actions, 1.0)(h).astype(jnp.float32)
        return logits, value, q.reshape(*q.shape[:-1], self.n_q_heads, self.n_actions)


def apply_fn(net: ActorCritic) -> ApplyFn:
    """`net.apply` typed as (params, obs) -> (logits, value, q)."""

    def apply(params: Params, obs: Array) -> tuple[Array, Array, Array]:
        return cast(tuple[Array, Array, Array], net.apply(params, obs))

    return apply


def init_params(net: ActorCritic, key: Array, obs_dim: int) -> Params:
    return net.init(key, jnp.zeros((1, obs_dim), jnp.float32))


def log_softmax(logits: Array) -> Array:
    return jax.nn.log_softmax(logits, axis=-1)


def entropy(logits: Array) -> Array:
    logp = log_softmax(logits)
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)


def take(x: Array, actions: Array) -> Array:
    """x[..., a] for integer actions with matching leading dims."""
    return jnp.take_along_axis(x, actions[..., None], axis=-1)[..., 0]
