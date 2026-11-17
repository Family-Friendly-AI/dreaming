"""Wake phase: one jitted PPO iteration (rollout of T steps on N envs, GAE, clipped update)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from craftax.craftax_classic.envs.craftax_state import EnvParams
from flax import struct
from jax import Array

from dreaming.agent.networks import ApplyFn, Params, log_softmax, take
from dreaming.agent.ppo import PPOBatch, compute_gae, ppo_update
from dreaming.config import Config
from dreaming.envs import craftax as E


class WakeState(struct.PyTreeNode):
    env: E.EnvCarry
    done: Array
    """CleanRL convention: `done[i]` marks that the current obs of env i starts a new episode."""


class Episodes(struct.PyTreeNode):
    """Per-step episode records [T, N]; valid where `done`."""

    done: Array
    ret: Array
    length: Array
    achievements: Array


class WakeOut(struct.PyTreeNode):
    stats: dict[str, Array]
    episodes: Episodes


def init_wake(key: Array, num_envs: int, params: EnvParams) -> WakeState:
    return WakeState(E.reset(jax.random.split(key, num_envs), params), jnp.zeros(num_envs, bool))


WakeIteration = Callable[[Params, optax.OptState, WakeState, Array], tuple[Params, optax.OptState, WakeState, WakeOut]]


def make_wake_iteration(
    cfg: Config, apply_fn: ApplyFn, tx: optax.GradientTransformation, env_params: EnvParams
) -> WakeIteration:
    pc = cfg.ppo

    def iteration(params: Params, opt_state: optax.OptState, wake: WakeState, key: Array):
        k_roll, k_update = jax.random.split(key)

        def step(carry: WakeState, k):
            k_act, k_env = jax.random.split(k)
            logits, value, _ = apply_fn(params, carry.env.obs)
            action = jax.random.categorical(k_act, logits)
            logp = take(log_softmax(logits), action)
            env, out = E.step_auto_reset(carry.env, action, k_env, env_params)
            rec = (carry.env.obs, carry.done, action, logp, value, out)
            return WakeState(env, out.done), rec

        wake, (obs, dones, actions, logps, values, out) = jax.lax.scan(
            step, wake, jax.random.split(k_roll, pc.num_steps)
        )
        _, next_value, _ = apply_fn(params, wake.env.obs)
        adv, ret = compute_gae(out.reward, values, dones, next_value, wake.done, pc.gamma, pc.gae_lambda)

        def flat(x: Array) -> Array:
            return x.reshape(-1, *x.shape[2:])

        batch = PPOBatch(flat(obs), flat(actions), flat(logps), flat(adv), flat(ret), flat(values))
        params, opt_state, stats = ppo_update(apply_fn, tx, params, opt_state, batch, k_update, pc, cfg.n_q_heads)
        stats["wake/td_abs_mean"] = jnp.abs(ret - values).mean()
        episodes = Episodes(out.done, out.ep_return, out.ep_length, out.ep_achievements)
        return params, opt_state, wake, WakeOut(stats, episodes)

    return jax.jit(iteration)
