"""Video: one pi_f episode in a fresh world from the VIDEO key stream, rendered to frames on device."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from craftax.craftax_classic.envs.craftax_state import EnvParams
from jax import Array

from dreaming.agent.networks import ApplyFn, Params
from dreaming.envs import craftax as E

VideoRollout = Callable[[Params, Array], tuple[Array, Array]]


def make_video_rollout(apply_fn: ApplyFn, env_params: EnvParams, max_steps: int) -> VideoRollout:
    """Jitted `(params, key) -> (frames [max_steps, H, W, 3] uint8, n_frames)`; actions are sampled as in wake.

    Frames from `n_frames` on are past the episode end and carry no information.
    """
    render = E.pixel_renderer()

    def run(params: Params, key: Array):
        k_reset, k_steps = jax.random.split(key)

        def step(carry, k):
            env, alive = carry
            k_act, k_env = jax.random.split(k)
            logits, _, _ = apply_fn(params, env.obs)
            frame = render(jax.tree_util.tree_map(lambda x: x[0], env.state))
            env, out = E.step(env, jax.random.categorical(k_act, logits), k_env[None], env_params)
            return (env, alive & ~out.done[0]), (frame, alive)

        carry = (E.reset(k_reset[None], env_params), jnp.bool_(True))
        _, (frames, alive) = jax.lax.scan(step, carry, jax.random.split(k_steps, max_steps))
        return frames, alive.sum()

    return jax.jit(run)
