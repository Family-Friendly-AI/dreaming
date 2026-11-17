"""Wake phase: one jitted PPO iteration (rollout of T steps on N envs, GAE, clipped update).

With a snapshot buffer, `ceil(snapshot_prob * N)` random env slots per step are captured with the wake
transition taken from them, scored by |TD| under the rollout's critic, and merged into the buffer top-C by |TD|.
"""

from __future__ import annotations

import math
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
from dreaming.phases.snapshots import SnapshotBuffer, insert


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


WakeIteration = Callable[
    [Params, optax.OptState, WakeState, SnapshotBuffer | None, Array],
    tuple[Params, optax.OptState, WakeState, SnapshotBuffer | None, WakeOut],
]


def snapshot_slots(snapshot_prob: float, num_envs: int) -> int:
    return min(num_envs, max(1, math.ceil(snapshot_prob * num_envs)))


def make_wake_iteration(
    cfg: Config, apply_fn: ApplyFn, tx: optax.GradientTransformation, env_params: EnvParams
) -> WakeIteration:
    pc = cfg.ppo
    n_slots = snapshot_slots(cfg.dream.snapshot_prob, cfg.env.num_envs)

    def iteration(
        params: Params, opt_state: optax.OptState, wake: WakeState, buffer: SnapshotBuffer | None, key: Array
    ):
        k_roll, k_update = jax.random.split(key)

        def step(carry: WakeState, k):
            k_act, k_env, k_snap = jax.random.split(k, 3)
            logits, value, _ = apply_fn(params, carry.env.obs)
            action = jax.random.categorical(k_act, logits)
            logp = take(log_softmax(logits), action)
            env, out = E.step_auto_reset(carry.env, action, k_env, env_params)
            rec = (carry.env.obs, carry.done, action, logp, value, out)
            if buffer is None:
                return WakeState(env, out.done), (rec, None)
            idx = jax.random.choice(k_snap, carry.done.shape[0], (n_slots,), replace=False)
            snap = SnapshotBuffer(
                state=jax.tree_util.tree_map(lambda x: x[idx], carry.env.state),
                obs=carry.env.obs[idx],
                next_obs=out.next_obs[idx],
                reward=out.reward[idx],
                done=out.done[idx],
                td=jnp.zeros(n_slots, jnp.float32),
                valid=jnp.ones(n_slots, bool),
            )
            return WakeState(env, out.done), (rec, (snap, value[idx]))

        wake, ((obs, dones, actions, logps, values, out), snaps) = jax.lax.scan(
            step, wake, jax.random.split(k_roll, pc.num_steps)
        )
        if buffer is not None:
            snaps, v = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), snaps)
            _, v_next, _ = apply_fn(params, snaps.next_obs)
            td = jnp.abs(snaps.reward + pc.gamma * v_next * (1.0 - snaps.done) - v)
            buffer = insert(buffer, snaps.replace(td=td))
        _, next_value, _ = apply_fn(params, wake.env.obs)
        adv, ret = compute_gae(out.reward, values, dones, next_value, wake.done, pc.gamma, pc.gae_lambda)

        def flat(x: Array) -> Array:
            return x.reshape(-1, *x.shape[2:])

        batch = PPOBatch(flat(obs), flat(actions), flat(logps), flat(adv), flat(ret), flat(values))
        params, opt_state, stats = ppo_update(apply_fn, tx, params, opt_state, batch, k_update, pc, cfg.n_q_heads)
        stats["wake/td_abs_mean"] = jnp.abs(ret - values).mean()
        episodes = Episodes(out.done, out.ep_return, out.ep_length, out.ep_achievements)
        return params, opt_state, wake, buffer, WakeOut(stats, episodes)

    return jax.jit(iteration)
