import jax
import jax.numpy as jnp
import numpy as np

from dreaming.agent.networks import ActorCritic, apply_fn, init_params
from dreaming.agent.ppo import PPOBatch, compute_gae, make_optimizer, ppo_update, q_regression_loss
from dreaming.config import PPOConfig


def reference_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """CleanRL `compute_gae` loop, verbatim semantics."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal, nextv = 1.0 - next_done, next_value
        else:
            nonterminal, nextv = 1.0 - dones[t + 1], values[t + 1]
        delta = rewards[t] + gamma * nextv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def test_gae_matches_cleanrl_reference():
    rng = np.random.default_rng(0)
    T, N = 17, 6
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = (rng.random((T, N)) < 0.2).astype(np.float32)
    next_value = rng.normal(size=N).astype(np.float32)
    next_done = (rng.random(N) < 0.2).astype(np.float32)
    ref_adv, ref_ret = reference_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)
    adv, ret = compute_gae(
        jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(dones), jnp.asarray(next_value),
        jnp.asarray(next_done), 0.99, 0.95,
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(adv), ref_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ret), ref_ret, rtol=1e-5, atol=1e-5)


def test_q_regression_masks_heads():
    q = jnp.zeros((2, 3, 4)).at[:, 0, 1].set(1.0)
    actions = jnp.array([1, 1])
    targets = jnp.zeros(2)
    only_head0 = jnp.array([[1.0, 0, 0], [1.0, 0, 0]])
    assert float(q_regression_loss(q, actions, targets, only_head0)) == 1.0
    assert float(q_regression_loss(q, actions, targets, 1.0 - only_head0)) == 0.0


def test_ppo_update_follows_advantage_sign():
    net = ActorCritic(n_actions=4, n_q_heads=2, hidden=32)
    apply = apply_fn(net)
    params = init_params(net, jax.random.key(0), 8)
    cfg = PPOConfig(num_minibatches=2, update_epochs=4, ent_coef=0.0, lr=1e-2)
    tx = make_optimizer(cfg.lr, cfg.max_grad_norm)
    n = 64
    obs = jax.random.normal(jax.random.key(1), (n, 8))
    actions = jnp.arange(n, dtype=jnp.int32) % 2
    logits, v, _ = apply(params, obs)
    logp = jax.nn.log_softmax(logits)
    adv = jnp.where(actions == 0, 1.0, -1.0)
    batch = PPOBatch(obs, actions, jnp.take_along_axis(logp, actions[:, None], -1)[:, 0], adv, v + adv, v)
    new, _, stats = jax.jit(lambda p, o, b, k: ppo_update(apply, tx, p, o, b, k, cfg, 2))(
        params, tx.init(params), batch, jax.random.key(2)
    )
    new_logp = jax.nn.log_softmax(apply(new, obs)[0])
    assert float(new_logp[:, 0].mean()) > float(logp[:, 0].mean())
    assert float(new_logp[:, 1].mean()) < float(logp[:, 1].mean())
    assert set(stats) >= {"ppo/pg_loss", "ppo/v_loss", "ppo/q_loss", "ppo/approx_kl", "ppo/explained_variance"}
