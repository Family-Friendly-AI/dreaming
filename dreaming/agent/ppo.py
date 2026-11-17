"""CleanRL-style GAE + clipped PPO; the Q-ensemble is regressed to the GAE returns on wake data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax import Array

from dreaming.agent.networks import ApplyFn, Params, entropy, log_softmax, take
from dreaming.config import PPOConfig


def make_optimizer(lr: float, max_grad_norm: float) -> optax.GradientTransformation:
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr, eps=1e-5))


def compute_gae(
    rewards: Array, values: Array, dones: Array, next_value: Array, next_done: Array, gamma: float, lam: float
) -> tuple[Array, Array]:
    """Arrays shaped [T, N]; `dones[t]` marks that obs t is the first of a new episode (CleanRL)."""
    next_nonterminal = 1.0 - jnp.concatenate([dones[1:], next_done[None]], axis=0).astype(jnp.float32)
    next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)
    deltas = rewards + gamma * next_values * next_nonterminal - values

    def body(last, x):
        delta, nonterminal = x
        last = delta + gamma * lam * nonterminal * last
        return last, last

    _, adv = jax.lax.scan(body, jnp.zeros_like(next_value), (deltas, next_nonterminal), reverse=True)
    return adv, adv + values


def q_regression_loss(q_all: Array, actions: Array, targets: Array, mask: Array | None = None) -> Array:
    """q_all [B, K, A], actions [B], targets [B], mask [B, K] (bootstrap mask) -> scalar."""
    q_a = take(q_all, jnp.broadcast_to(actions[:, None], q_all.shape[:2]))
    err = (q_a - targets[:, None]) ** 2
    if mask is None:
        return err.mean()
    return (err * mask).sum() / jnp.maximum(mask.sum(), 1.0)


class PPOBatch(struct.PyTreeNode):
    obs: Array
    actions: Array
    logprobs: Array
    advantages: Array
    returns: Array
    values: Array


def ppo_update(
    apply_fn: ApplyFn,
    tx: optax.GradientTransformation,
    params: Params,
    opt_state: optax.OptState,
    batch: PPOBatch,
    key: Array,
    cfg: PPOConfig,
    n_q_heads: int,
) -> tuple[Params, optax.OptState, dict[str, Array]]:
    n = batch.obs.shape[0]
    n_mb = max(1, cfg.num_minibatches)
    mb = n // n_mb
    k_mask, k_perm = jax.random.split(key)
    q_mask = (jax.random.uniform(k_mask, (n, n_q_heads)) < cfg.q_bootstrap_p).astype(jnp.float32)

    def loss_fn(p, b: PPOBatch, qm):
        logits, v, q = apply_fn(p, b.obs)
        logp = take(log_softmax(logits), b.actions)
        log_ratio = logp - b.logprobs
        ratio = jnp.exp(log_ratio)
        a = (b.advantages - b.advantages.mean()) / (b.advantages.std(ddof=1) + 1e-8)
        pg_loss = jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
        v_clipped = b.values + jnp.clip(v - b.values, -cfg.clip_coef, cfg.clip_coef)
        v_loss = 0.5 * jnp.maximum((v - b.returns) ** 2, (v_clipped - b.returns) ** 2).mean()
        q_loss = q_regression_loss(q, b.actions, b.returns, qm)
        ent = entropy(logits).mean()
        loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss + cfg.q_coef * q_loss
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfrac = (jnp.abs(ratio - 1) > cfg.clip_coef).astype(jnp.float32).mean()
        stats = {"pg_loss": pg_loss, "v_loss": v_loss, "q_loss": q_loss, "entropy": ent}
        return loss, stats | {"approx_kl": approx_kl, "clipfrac": clipfrac}

    grad_fn = jax.grad(loss_fn, has_aux=True)

    def minibatch(carry, idx):
        p, o = carry
        b = jax.tree_util.tree_map(lambda x: x[idx], batch)
        grads, stats = grad_fn(p, b, q_mask[idx])
        updates, o = tx.update(grads, o, p)
        return (optax.apply_updates(p, updates), o), stats

    def epoch(carry, k):
        perm = jax.random.permutation(k, n)[: n_mb * mb].reshape(n_mb, mb)
        return jax.lax.scan(minibatch, carry, perm)

    (params, opt_state), stats = jax.lax.scan(epoch, (params, opt_state), jax.random.split(k_perm, cfg.update_epochs))
    out = {f"ppo/{k}": v.mean() for k, v in stats.items()}
    var_y = batch.returns.var()
    out["ppo/explained_variance"] = jnp.where(
        var_y > 0, 1 - (batch.returns - batch.values).var() / jnp.where(var_y > 0, var_y, 1.0), jnp.nan
    )
    return params, opt_state, out
