"""Fast-learner update from dreams: Q-ensemble regression + advantage-weighted regression (AWR).

- Q_k(s, a) regresses onto discounted (V-bootstrapped) dream returns for every active branch/control step.
- The actor gets an AWR loss on branch roots, weight exp(min(A / tau, awr_clip)) for A(s, a_rare) > 0 and 0
  otherwise: only rare actions that beat the matched control are reinforced.
- KL(pi_before || pi) on the roots keeps the update local.
No importance ratios are used anywhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax import Array

from dreaming.agent.networks import ApplyFn, Params, log_softmax, take
from dreaming.agent.ppo import q_regression_loss
from dreaming.config import DreamConfig


class Transitions(struct.PyTreeNode):
    """Flattened dream steps; `mask` is False for steps after an episode end or past the rollout length."""

    obs: Array
    actions: Array
    returns: Array
    mask: Array


class Roots(struct.PyTreeNode):
    """One entry per selected snapshot; `valid` is False for padding when the buffer had too few snapshots."""

    obs: Array
    actions: Array
    adv: Array
    valid: Array


def awr_weights(adv: Array, tau: float, clip: float) -> Array:
    w = jnp.exp(jnp.minimum(adv / tau, clip))
    return jnp.where(adv > 0, w, 0.0)


def dream_update(
    apply_fn: ApplyFn,
    tx: optax.GradientTransformation,
    params: Params,
    opt_state: optax.OptState,
    trans: Transitions,
    roots: Roots,
    key: Array,
    cfg: DreamConfig,
    n_q_heads: int,
    q_bootstrap_p: float,
) -> tuple[Params, optax.OptState, dict[str, Array]]:
    n = trans.obs.shape[0]
    mb = min(cfg.minibatch_size, n)
    n_mb = n // mb
    k_mask, k_perm = jax.random.split(key)
    q_mask = (jax.random.uniform(k_mask, (n, n_q_heads)) < q_bootstrap_p) & trans.mask[:, None]
    q_mask = q_mask.astype(jnp.float32)
    root_w = roots.valid.astype(jnp.float32)
    n_roots = jnp.maximum(root_w.sum(), 1.0)
    w = awr_weights(roots.adv, cfg.tau, cfg.awr_clip) * root_w
    logp_before = jax.lax.stop_gradient(log_softmax(apply_fn(params, roots.obs)[0]))

    def loss_fn(p, idx):
        _, _, q = apply_fn(p, trans.obs[idx])
        q_loss = q_regression_loss(q, trans.actions[idx], trans.returns[idx], q_mask[idx])
        logp = log_softmax(apply_fn(p, roots.obs)[0])
        awr = -(w * take(logp, roots.actions)).sum() / n_roots
        anchor = (jnp.sum(jnp.exp(logp_before) * (logp_before - logp), axis=-1) * root_w).sum() / n_roots
        loss = cfg.q_coef * q_loss + cfg.awr_coef * awr + cfg.anchor_coef * anchor
        return loss, {"q_loss": q_loss, "awr_loss": awr, "anchor_kl": anchor}

    grad_fn = jax.grad(loss_fn, has_aux=True)

    def minibatch(carry, idx):
        p, o = carry
        grads, stats = grad_fn(p, idx)
        updates, o = tx.update(grads, o, p)
        return (optax.apply_updates(p, updates), o), stats

    def epoch(carry, k):
        perm = jax.random.permutation(k, n)[: n_mb * mb].reshape(n_mb, mb)
        return jax.lax.scan(minibatch, carry, perm)

    (params, opt_state), stats = jax.lax.scan(epoch, (params, opt_state), jax.random.split(k_perm, cfg.update_epochs))
    out = {f"dream/{k}": v.mean() for k, v in stats.items()}
    out["dream/awr_pos_frac"] = ((roots.adv > 0) & roots.valid).sum() / n_roots
    return params, opt_state, out
