"""Craftax-Classic-Symbolic, vmapped over a batch of N envs.

Every step's randomness comes from the explicit per-env key, so an `EnvState` plus the same keys
replays a trajectory exactly; a snapshot is just a stored `EnvState`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from enum import IntEnum

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_IMG, Achievement
from craftax.craftax_classic.envs.craftax_state import EnvParams, EnvState
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnvNoAutoReset
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
from flax import struct
from jax import Array

ENV_NAME = "Craftax-Classic-Symbolic-v1"
ACHIEVEMENTS: tuple[str, ...] = tuple(a.name.lower() for a in Achievement)
NUM_ACTIONS = 17
OBS_DIM = 1345
VISIT_CELL = 8
_INV_LEVELS = 3

_ENV = CraftaxClassicSymbolicEnvNoAutoReset()


class KeyStream(IntEnum):
    """Independent random streams derived from the run seed; gate worlds never overlap wake/dream worlds."""

    INIT = 0
    WAKE = 1
    DREAM = 2
    GATE_WORLD = 3
    GATE_NOISE = 4
    VIDEO = 5


def stream(root: Array, s: KeyStream) -> Array:
    return jax.random.fold_in(root, int(s))


def gate_world_keys(root: Array, block: int, n: int) -> Array:
    """World keys for gate block `block`: world i is `fold_in(fold_in(GATE_WORLD, block), i)`."""
    base = jax.random.fold_in(stream(root, KeyStream.GATE_WORLD), block)
    return jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.arange(n))


def env_params(max_episode_steps: int | None) -> EnvParams:
    params = _ENV.default_params
    return params if max_episode_steps is None else dataclasses.replace(params, max_timesteps=max_episode_steps)


class EnvCarry(struct.PyTreeNode):
    """Batched env state plus the running episode statistics."""

    state: EnvState
    obs: Array
    ep_return: Array
    ep_length: Array


class StepOut(struct.PyTreeNode):
    """Per-step outputs; `ep_*` hold the finished episode's values where `done`."""

    reward: Array
    done: Array
    next_obs: Array
    unlocked: Array
    ep_return: Array
    ep_length: Array
    ep_achievements: Array
    visit_key: Array


def visit_key(state: EnvState) -> Array:
    """int32 count-based exploration key: 8x8 map cell x base-3 code of min(inventory, 2)."""
    pos = state.player_position // VISIT_CELL
    cells_per_row = jnp.int32(64 // VISIT_CELL)
    cell = pos[..., 0].astype(jnp.int32) * cells_per_row + pos[..., 1].astype(jnp.int32)
    inv = jnp.stack([jnp.asarray(x) for x in jax.tree_util.tree_leaves(state.inventory)], axis=-1)
    digits = jnp.minimum(inv, _INV_LEVELS - 1).astype(jnp.int32)
    code = jnp.sum(digits * _INV_LEVELS ** jnp.arange(digits.shape[-1], dtype=jnp.int32), axis=-1)
    return cell * jnp.int32(_INV_LEVELS ** digits.shape[-1]) + code


def pixel_renderer() -> Callable[[EnvState], Array]:
    """Unbatched state -> uint8 [144, 144, 3] frame. Loads the Craftax textures (~30 s once, then cached)."""
    render = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_IMG)
    return lambda state: jnp.clip(render(state), 0, 255).astype(jnp.uint8)


def restore(states: EnvState) -> EnvCarry:
    """Fresh episode counters around stored states (dream roots)."""
    n = states.achievements.shape[0]
    obs = jax.vmap(_ENV.get_obs)(states)
    return EnvCarry(states, obs, jnp.zeros(n, jnp.float32), jnp.zeros(n, jnp.int32))


def reset(keys: Array, params: EnvParams) -> EnvCarry:
    _, states = jax.vmap(_ENV.reset, in_axes=(0, None))(keys, params)
    return restore(states)


def step(carry: EnvCarry, actions: Array, keys: Array, params: EnvParams) -> tuple[EnvCarry, StepOut]:
    """Step without reset; callers mask everything after `done` themselves."""
    prev_ach = carry.state.achievements
    obs, states, reward, done, _ = jax.vmap(_ENV.step, in_axes=(0, 0, 0, None))(keys, carry.state, actions, params)
    reward = reward.astype(jnp.float32)
    ep_return = carry.ep_return + reward
    ep_length = carry.ep_length + 1
    out = StepOut(
        reward=reward,
        done=done,
        next_obs=obs,
        unlocked=states.achievements & ~prev_ach,
        ep_return=ep_return,
        ep_length=ep_length,
        ep_achievements=states.achievements,
        visit_key=visit_key(states),
    )
    return EnvCarry(states, obs, ep_return, ep_length), out


def step_auto_reset(
    carry: EnvCarry, actions: Array, key: Array, params: EnvParams, reset_ratio: int = 16
) -> tuple[EnvCarry, StepOut]:
    """Step and reset finished envs (Craftax optimistic reset: N/reset_ratio fresh worlds per step)."""
    n = actions.shape[0]
    k_step, k_reset, k_pick = jax.random.split(key, 3)
    carry, out = step(carry, actions, jax.random.split(k_step, n), params)
    num_resets = max(1, n // max(1, min(reset_ratio, n)))
    fresh = reset(jax.random.split(k_reset, num_resets), params)
    chosen = jax.random.choice(k_pick, n, shape=(num_resets,), replace=False, p=out.done + 1e-12)
    slot = jnp.zeros(n, jnp.int32).at[chosen].set(jnp.arange(num_resets, dtype=jnp.int32))
    fresh = jax.tree_util.tree_map(lambda x: x[slot], fresh)
    done = out.done

    def pick(new, old):
        mask = done.reshape(done.shape + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)

    carry = jax.tree_util.tree_map(pick, fresh, carry)
    return carry, out
