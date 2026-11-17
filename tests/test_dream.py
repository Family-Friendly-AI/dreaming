import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreaming.agent.dream_update import Roots, Transitions, awr_weights, dream_update
from dreaming.agent.networks import ActorCritic, apply_fn, init_params
from dreaming.agent.ppo import make_optimizer
from dreaming.config import Config, DreamConfig, EnvConfig, apply_method
from dreaming.envs import craftax as E
from dreaming.phases import snapshots
from dreaming.phases.dream import branch_count, make_dream_phase, rollout, score_actions, select_branches

PARAMS = E.env_params(max_episode_steps=200)
NET = ActorCritic(E.NUM_ACTIONS, n_q_heads=3, hidden=32)
APPLY = apply_fn(NET)
FAST = init_params(NET, jax.random.key(0), E.OBS_DIM)
REF = init_params(NET, jax.random.key(1), E.OBS_DIM)
GAMMA = 0.9


def _filled_buffer(capacity: int, n_valid: int) -> snapshots.SnapshotBuffer:
    carry = E.reset(jax.random.split(jax.random.key(2), n_valid), PARAMS)
    _, out = E.step(carry, jnp.zeros(n_valid, jnp.int32), jax.random.split(jax.random.key(3), n_valid), PARAMS)
    cand = snapshots.SnapshotBuffer(
        state=carry.state,
        obs=carry.obs,
        next_obs=out.next_obs,
        reward=out.reward,
        done=out.done,
        td=jnp.arange(n_valid, dtype=jnp.float32),
        valid=jnp.ones(n_valid, bool),
    )
    template = jax.tree_util.tree_map(lambda x: x[0], carry.state)
    return snapshots.insert(snapshots.empty(template, E.OBS_DIM, capacity), cand)


def _rollout_inputs(b: int, length: int, seed: int):
    k = jax.random.key(seed)
    keys = jax.random.split(k, (length, b))
    noise = jax.random.gumbel(jax.random.fold_in(k, 1), (length, b, E.NUM_ACTIONS))
    return keys, noise


def _same_root(buf, j: int, b: int):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[j], (b, *x.shape[1:])), buf.state)


def test_insert_keeps_top_capacity_by_td():
    buf = _filled_buffer(capacity=4, n_valid=6)
    assert sorted(np.asarray(buf.td).tolist()) == [2.0, 3.0, 4.0, 5.0]
    assert bool(buf.valid.all())


def test_insert_into_larger_buffer_keeps_invalid_slots_invalid():
    buf = _filled_buffer(capacity=8, n_valid=3)
    assert int(buf.valid.sum()) == 3


def test_empty_prefix_branch_equals_control():
    buf = _filled_buffer(capacity=4, n_valid=4)
    keys, noise = _rollout_inputs(1, 12, seed=4)
    keys, noise = jnp.repeat(keys, 2, axis=1), jnp.repeat(noise, 2, axis=1)
    ro = rollout(
        APPLY, FAST, _same_root(buf, 1, 2), jnp.array([5, 5]), jnp.array([0, 0]), jnp.array([12, 12]),
        keys, noise, PARAMS, GAMMA,
    )  # fmt: skip
    for x in (ro.obs, ro.actions, ro.rewards, ro.returns):
        np.testing.assert_array_equal(np.asarray(x[:, 0]), np.asarray(x[:, 1]))


def test_prefix_forces_the_deviation_then_follows_policy():
    buf = _filled_buffer(capacity=4, n_valid=4)
    keys, noise = _rollout_inputs(1, 10, seed=5)
    ro = rollout(
        APPLY, FAST, _same_root(buf, 0, 1), jnp.array([7]), jnp.array([3]), jnp.array([10]), keys, noise, PARAMS, GAMMA
    )
    acts = np.asarray(ro.actions[:, 0])
    assert acts[:3].tolist() == [7, 7, 7]
    np.testing.assert_array_equal(np.asarray(ro.obs[0, 0]), np.asarray(buf.obs[0]))


def test_rollout_is_deterministic():
    buf = _filled_buffer(capacity=4, n_valid=4)
    keys, noise = _rollout_inputs(3, 15, seed=6)
    args = (_same_root(buf, 2, 3), jnp.array([1, 2, 3]), jnp.array([2, 0, 1]), jnp.array([15, 9, 4]))
    a = rollout(APPLY, FAST, *args, keys, noise, PARAMS, GAMMA)
    b = rollout(APPLY, FAST, *args, keys, noise, PARAMS, GAMMA)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_masked_returns_bootstrap_at_rollout_length():
    buf = _filled_buffer(capacity=4, n_valid=4)
    length, steps = 5, 8
    keys, noise = _rollout_inputs(2, steps, seed=7)
    ro = rollout(
        APPLY, FAST, _same_root(buf, 0, 2), jnp.array([0, 0]), jnp.array([0, 0]), jnp.array([length, 0]),
        keys, noise, PARAMS, GAMMA,
    )  # fmt: skip
    active = np.asarray(ro.active)
    assert active[:, 0].tolist() == [True] * length + [False] * (steps - length)
    assert not active[:, 1].any()
    r = np.asarray(ro.rewards[:, 0])
    v_boot = float(APPLY(FAST, ro.obs[length, :1])[1][0])
    expected = sum(GAMMA**t * r[t] for t in range(length)) + GAMMA**length * v_boot
    np.testing.assert_allclose(float(ro.returns[0, 0]), expected, rtol=1e-5)
    np.testing.assert_allclose(float(ro.returns[0, 1]), float(APPLY(FAST, ro.obs[0, 1:2])[1][0]), rtol=1e-5)


def test_awr_weights_clipped_and_positive_only():
    w = np.asarray(awr_weights(jnp.array([-1.0, 0.0, 0.5, 100.0]), tau=1.0, clip=5.0))
    assert w[0] == 0 and w[1] == 0
    np.testing.assert_allclose(w[2:], np.exp([0.5, 5.0]), rtol=1e-6)


def test_score_actions_rarity_modes():
    s, a, k = 3, 7, 5
    keys = jax.random.split(jax.random.key(8), 4)
    lf, lr = jax.random.normal(keys[0], (s, a)), jax.random.normal(keys[1], (s, a))
    q, prio = jax.random.normal(keys[2], (s, k, a)), jax.random.uniform(keys[3], (s,))
    for rarity in ("both", "surprisal", "kl"):
        p = score_actions(lf, lr, q, prio, DreamConfig(rarity=rarity))
        assert p.shape == (s, a) and bool((p >= 0).all())
    p = score_actions(lf, lr, q, jnp.ones(s), DreamConfig(rarity="surprisal", eta=0.0))
    np.testing.assert_array_equal(np.asarray(p.argmax(1)), np.asarray(lf.argmin(1)))
    both = score_actions(lf, lf, q, prio, DreamConfig(rarity="both"))
    sur = score_actions(lf, lf, q, prio, DreamConfig(rarity="surprisal"))
    np.testing.assert_allclose(np.asarray(both), np.asarray(sur), atol=1e-5)


@pytest.mark.parametrize("selection", ["targeted", "random"])
def test_selection_skips_invalid_slots(selection):
    buf = _filled_buffer(capacity=8, n_valid=3)
    td = snapshots.td_errors(APPLY, FAST, buf, GAMMA)
    br = select_branches(APPLY, FAST, REF, buf, td, jax.random.key(9), DreamConfig(selection=selection), 5)
    valid, idx = np.asarray(br.valid), np.asarray(br.idx)
    assert valid.sum() == 3
    assert set(idx[valid].tolist()) == set(np.flatnonzero(np.asarray(buf.valid)).tolist())
    assert bool(((br.k >= 1) & (br.k <= 3)).all())


def test_awr_raises_prob_of_positive_advantage_action():
    obs = jax.random.normal(jax.random.key(10), (8, E.OBS_DIM))
    acts = jnp.full(8, 4)
    roots = Roots(obs, acts, jnp.ones(8), jnp.ones(8, bool))
    trans = Transitions(obs, acts, jnp.zeros(8), jnp.ones(8, bool))
    cfg = DreamConfig(update_epochs=20, anchor_coef=0.0, minibatch_size=8, lr=1e-3)
    tx = make_optimizer(cfg.lr, 0.5)
    new, _, _ = dream_update(APPLY, tx, FAST, tx.init(FAST), trans, roots, jax.random.key(11), cfg, 3, 0.8)

    def p4(p):
        return float(jax.nn.softmax(APPLY(p, obs)[0])[:, 4].mean())

    assert p4(new) > p4(FAST)


def _tiny_cfg(**dream) -> Config:
    cfg = apply_method(
        Config(method="dream", wake_steps_per_cycle=1024, env=EnvConfig(num_envs=8, max_episode_steps=200))
    )
    return dataclasses.replace(
        cfg,
        dream=dataclasses.replace(cfg.dream, horizon=8, m_rollouts=2, update_epochs=1, buffer_size=16, **dream),
        n_q_heads=3,
    )


def test_dream_phase_counts_active_steps_and_compiles_once():
    cfg = _tiny_cfg()
    n = branch_count(cfg)
    assert n == int(0.3 * 1024) // (2 * 2 * (3 + 8))
    tx = make_optimizer(cfg.dream.lr, 0.5)
    phase = make_dream_phase(cfg, APPLY, tx, PARAMS)
    buf = _filled_buffer(capacity=16, n_valid=10)
    fast, opt, stats, steps = phase(FAST, REF, tx.init(FAST), buf, jax.random.key(12))
    assert int(stats["dream/snapshots"]) == n
    assert 0 < int(steps) <= n * 2 * 2 * (3 + 8)
    assert np.isfinite(float(stats["dream/A_mean"])) and np.isfinite(float(stats["dream/A_se_mean"]))
    _, _, stats2, _ = phase(fast, REF, opt, _filled_buffer(capacity=16, n_valid=2), jax.random.key(13))
    assert int(stats2["dream/snapshots"]) == 2
    assert phase._cache_size() == 1  # type: ignore[attr-defined]


def test_dream_phase_rejects_budget_without_snapshots():
    with pytest.raises(ValueError, match="budget_frac"):
        make_dream_phase(_tiny_cfg(budget_frac=0.001), APPLY, make_optimizer(1e-3, 0.5), PARAMS)
