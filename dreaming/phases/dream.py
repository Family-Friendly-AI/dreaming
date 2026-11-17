"""Dream phase: deviate-then-follow branches from real snapshots, with matched controls.

Per selected snapshot s with action a (repeated k ~ U{1..k_max} times), for m in 1..M:
  branch m:  restore(s); a x k, then follow pi_f;  length k + H, R = sum gamma^t r + gamma^L V(s_L)
  control m: restore(s); follow pi_f;              length k + H
  A(s, a) = mean_m(R_branch - R_ctrl)
Branch m and control m share the restored `EnvState`, the per-step env keys and the Gumbel noise used to sample
pi_f (common random numbers), so A isolates the effect of the deviation. All n x M x 2 rollouts run as one
batched scan; steps after an episode end or past the rollout length are masked and never counted.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from craftax.craftax_classic.envs.craftax_state import EnvParams, EnvState
from flax import struct
from jax import Array

from dreaming.agent.dream_update import Roots, Transitions, dream_update
from dreaming.agent.networks import ApplyFn, Params, log_softmax, take
from dreaming.config import Config, DreamConfig
from dreaming.envs import craftax as E
from dreaming.phases.snapshots import SnapshotBuffer, td_errors


def branch_count(cfg: Config) -> int:
    """Snapshots dreamed per cycle: `budget_frac * wake_steps_per_cycle` env steps at 2*M*(k_max + H) each."""
    dc = cfg.dream
    n = int(dc.budget_frac * cfg.wake_steps_per_cycle) // (2 * dc.m_rollouts * (dc.k_max + dc.horizon))
    return n if dc.max_snapshots is None else min(n, dc.max_snapshots)


def score_actions(logits_f: Array, logits_ref: Array, q: Array, prio: Array, cfg: DreamConfig) -> Array:
    """P(s, a) = prio(s) * rare(s, a) * (1 + eta * std_k Q_k(s, a)); [S, A], non-negative."""
    logp_f = log_softmax(logits_f)
    logp_ref = log_softmax(logits_ref)
    surprisal = -logp_f
    kl = jnp.maximum(jnp.sum(jnp.exp(logp_f) * (logp_f - logp_ref), axis=-1, keepdims=True), 0.0)
    if cfg.rarity == "both":
        rare = surprisal + cfg.lam * kl
    elif cfg.rarity == "surprisal":
        rare = surprisal
    else:
        rare = jnp.broadcast_to(kl, surprisal.shape) + 1e-6
    bonus = 1.0 + cfg.eta * jnp.std(q, axis=-2)
    return prio[:, None] * rare * bonus


class Branches(struct.PyTreeNode):
    """n selected (snapshot, action, k); `valid` is False where the buffer had fewer than n snapshots."""

    idx: Array
    action: Array
    k: Array
    valid: Array


def select_branches(
    apply_fn: ApplyFn, fast: Params, ref: Params, buf: SnapshotBuffer, td: Array, key: Array, cfg: DreamConfig, n: int
) -> Branches:
    """Snapshots without replacement (Gumbel top-k) with weight sum_a P(s, a), then a ~ P(s, .).

    `random` selection weights snapshots by |TD| and draws the action uniformly (no rarity, no bonus).
    """
    k_state, k_action, k_rep = jax.random.split(key, 3)
    logits_f, _, q = apply_fn(fast, buf.obs)
    if cfg.selection == "random":
        w_state = td + 1e-8
        log_p_action = jnp.zeros_like(logits_f)
    else:
        logits_ref = apply_fn(ref, buf.obs)[0]
        p = jnp.clip(jnp.nan_to_num(score_actions(logits_f, logits_ref, q, td, cfg), nan=0.0, posinf=0.0), 0.0)
        p = p + 1e-12
        w_state = p.sum(-1)
        log_p_action = jnp.log(p)
    log_w = jnp.where(buf.valid, jnp.log(w_state), -jnp.inf)
    score, idx = jax.lax.top_k(log_w + jax.random.gumbel(k_state, log_w.shape), n)
    action = jax.random.categorical(k_action, log_p_action[idx])
    k = jax.random.randint(k_rep, (n,), 1, cfg.k_max + 1)
    return Branches(idx=idx, action=action, k=k, valid=jnp.isfinite(score))


class Rollouts(struct.PyTreeNode):
    """Time-major [L, B] records; `returns` are masked discounted returns bootstrapped with V."""

    obs: Array
    actions: Array
    rewards: Array
    active: Array
    returns: Array


def rollout(
    apply_fn: ApplyFn,
    params: Params,
    states: EnvState,
    prefix_action: Array,
    prefix_len: Array,
    length: Array,
    env_keys: Array,
    noise: Array,
    env_params: EnvParams,
    gamma: float,
) -> Rollouts:
    """B rollouts from `states`: `prefix_action` for t < prefix_len, else argmax(logits + noise[t]).

    `env_keys` [L, B] and `noise` [L, B, A] fix all randomness; a rollout is active while t < length and its
    episode has not ended.
    """

    def step(carry, x):
        env, alive, t = carry
        keys, g = x
        logits, value, _ = apply_fn(params, env.obs)
        action = jnp.where(t < prefix_len, prefix_action, jnp.argmax(logits + g, axis=-1))
        active = alive & (t < length)
        env_next, out = E.step(env, action, keys, env_params)
        rec = (env.obs, action, out.reward * active, active, value, alive)
        return (env_next, alive & ~(active & out.done), t + 1), rec

    alive0 = jnp.ones(prefix_action.shape, bool)
    (env, alive, _), (obs, actions, rewards, active, values, alives) = jax.lax.scan(
        step, (E.restore(states), alive0, jnp.int32(0)), (env_keys, noise)
    )
    v_last = apply_fn(params, env.obs)[1]

    def back(g, x):
        r, act, v, al = x
        g = jnp.where(act, r + gamma * g, jnp.where(al, v, 0.0))
        return g, g

    _, returns = jax.lax.scan(back, jnp.where(alive, v_last, 0.0), (rewards, active, values, alives), reverse=True)
    return Rollouts(obs, actions, rewards, active, returns)


DreamPhase = Callable[
    [Params, Params, optax.OptState, SnapshotBuffer, Array],
    tuple[Params, optax.OptState, dict[str, Array], Array],
]


def make_dream_phase(
    cfg: Config, apply_fn: ApplyFn, tx: optax.GradientTransformation, env_params: EnvParams
) -> DreamPhase:
    """Jitted `(fast, ref, opt, buffer, key) -> (fast, opt, stats, active env steps)` for a fixed branch count."""
    dc = cfg.dream
    n, m = branch_count(cfg), dc.m_rollouts
    steps = dc.k_max + dc.horizon
    if not 1 <= n <= dc.buffer_size:
        raise ValueError(
            f"{n} dream snapshots per cycle (budget_frac * wake_steps_per_cycle / (2 * m_rollouts * (k_max + horizon)))"
            f" must be in [1, dream.buffer_size={dc.buffer_size}]: adjust dream.budget_frac or dream.buffer_size."
        )
    if m < 2:
        raise ValueError("dream.m_rollouts must be >= 2: the paired standard error of A needs two pairs.")

    def phase(fast: Params, ref: Params, opt_state: optax.OptState, buf: SnapshotBuffer, key: Array):
        k_sel, k_env, k_noise, k_update = jax.random.split(key, 4)
        td = td_errors(apply_fn, fast, buf, cfg.ppo.gamma)
        br = select_branches(apply_fn, fast, ref, buf, td, k_sel, dc, n)

        def per_pair(x: Array) -> Array:
            return jnp.broadcast_to(x[:, :, None], (n, m, 2, *x.shape[2:])).reshape(n * m * 2, *x.shape[2:])

        def per_rollout(x: Array) -> Array:
            return jnp.broadcast_to(x[:, None, None], (n, m, 2, *x.shape[1:])).reshape(n * m * 2, *x.shape[1:])

        env_keys = jax.random.split(k_env, (n, m, steps))
        noise = jax.random.gumbel(k_noise, (n, m, steps, E.NUM_ACTIONS))
        is_branch = jnp.broadcast_to(jnp.array([True, False]), (n, m, 2)).reshape(-1)
        states = jax.tree_util.tree_map(lambda x: per_rollout(x[br.idx]), buf.state)
        ro = rollout(
            apply_fn,
            fast,
            states,
            prefix_action=per_rollout(br.action),
            prefix_len=jnp.where(is_branch, per_rollout(br.k), 0),
            length=jnp.where(per_rollout(br.valid), per_rollout(br.k) + dc.horizon, 0),
            env_keys=jnp.moveaxis(per_pair(env_keys), 1, 0),
            noise=jnp.moveaxis(per_pair(noise), 1, 0),
            env_params=env_params,
            gamma=cfg.ppo.gamma,
        )
        g0 = ro.returns[0].reshape(n, m, 2)
        diff = g0[..., 0] - g0[..., 1]
        adv = diff.mean(-1)
        se = diff.std(-1, ddof=1) / jnp.sqrt(m)
        roots = Roots(buf.obs[br.idx], br.action, adv, br.valid)
        trans = Transitions(
            obs=ro.obs.reshape(-1, ro.obs.shape[-1]),
            actions=ro.actions.reshape(-1),
            returns=ro.returns.reshape(-1),
            mask=ro.active.reshape(-1),
        )

        def pi_rare(p: Params) -> Array:
            return jnp.exp(take(log_softmax(apply_fn(p, roots.obs)[0]), roots.actions))

        before = pi_rare(fast)
        fast, opt_state, stats = dream_update(
            apply_fn, tx, fast, opt_state, trans, roots, k_update, dc, cfg.n_q_heads, cfg.ppo.q_bootstrap_p
        )
        after = pi_rare(fast)
        v = br.valid
        n_valid = jnp.maximum(v.sum(), 1)
        pos = v & (adv > 0)
        n_pos = jnp.maximum(pos.sum(), 1)

        def mean(x: Array, mask: Array, count: Array) -> Array:
            return jnp.where(mask, x, 0.0).sum() / count

        stats |= {
            "dream/snapshots": v.sum(),
            "dream/buffer_size": buf.valid.sum(),
            "dream/td_prio_mean": mean(td, buf.valid, jnp.maximum(buf.valid.sum(), 1)),
            "dream/A_mean": mean(adv, v, n_valid),
            "dream/A_std": jnp.sqrt(mean((adv - mean(adv, v, n_valid)) ** 2, v, n_valid)),
            "dream/A_max": jnp.where(v, adv, -jnp.inf).max(),
            "dream/A_se_mean": mean(se, v, n_valid),
            "dream/A_sig_pos_frac": mean((adv > 2 * se).astype(jnp.float32), v, n_valid),
            "dream/pi_rare_before": mean(before, v, n_valid),
            "dream/pi_rare_after": mean(after, v, n_valid),
            "dream/pi_rare_before_posA": mean(before, pos, n_pos),
            "dream/pi_rare_after_posA": mean(after, pos, n_pos),
        }
        return fast, opt_state, stats, ro.active.sum()

    return jax.jit(phase)
