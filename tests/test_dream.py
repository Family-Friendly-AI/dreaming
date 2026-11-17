import numpy as np
import pytest
import torch

from dreaming.agent.dream_update import awr_weights, dream_update
from dreaming.agent.networks import ActorCritic
from dreaming.config import Config, DreamConfig, apply_method
from dreaming.envs import EnvPool
from dreaming.phases.dream import (RolloutSpec, build_specs, discounted_returns, dream_phase, gumbel,
                                   run_rollouts, score_actions, select_branches)
from dreaming.phases.snapshots import Snapshot, SnapshotBuffer

ENV = "minigrid:MiniGrid-DoorKey-5x5-v0"
DEV = torch.device("cpu")


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    pool = EnvPool(ENV, num_envs=4, seed=0, mode="serial")
    net = ActorCritic(pool.obs_shape, pool.n_actions, 5)
    slow = ActorCritic(pool.obs_shape, pool.n_actions, 5)
    buf = SnapshotBuffer(64)
    obs = pool.reset(seeds=[0, 1, 2, 3])
    rng = np.random.default_rng(0)
    for t in range(12):
        a = rng.integers(0, 3, size=4)
        nobs, r, d, infos, states = pool.step(a, snapshot_idx=[0, 1, 2, 3])
        buf.add([Snapshot(states[i], obs[i], nobs[i], float(r[i]), bool(d[i]), t, float(rng.random()),
                          infos[i]["visit_key"]) for i in range(4)])
        buf.count_visits(i["visit_key"] for i in infos)
        obs = nobs
    return pool, net, slow, buf


def test_discounted_returns_bootstrap_and_truncation():
    r = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    g = 0.5
    out = discounted_returns(r, bootstrap=4.0, gamma=g)
    assert np.allclose(out, [1 + g * (0 + g * (2 + g * 4)), g * (2 + g * 4), 2 + g * 4])
    # Terminal episode -> bootstrap 0
    assert np.allclose(discounted_returns(r, 0.0, g)[-1], 2.0)


def test_awr_weights_clipped_and_positive_only():
    adv = torch.tensor([-1.0, 0.0, 0.5, 100.0])
    w = awr_weights(adv, tau=1.0, clip=5.0)
    assert w[0] == 0 and w[1] == 0
    assert torch.isclose(w[2], torch.exp(torch.tensor(0.5)))
    assert torch.isclose(w[3], torch.exp(torch.tensor(5.0)))


def test_score_actions_shapes_and_rarity_modes():
    S, A, K = 3, 7, 5
    lf, ls, q = torch.randn(S, A), torch.randn(S, A), torch.randn(S, K, A)
    prio = torch.rand(S)
    for rarity in ["both", "surprisal", "kl"]:
        P = score_actions(lf, ls, q, prio, DreamConfig(rarity=rarity))
        assert P.shape == (S, A) and (P >= 0).all()
    # surprisal: rarer actions score higher when bonus is off
    P = score_actions(lf, ls, q, torch.ones(S), DreamConfig(rarity="surprisal", eta=0.0))
    probs = torch.softmax(lf, -1)
    assert (P.argmax(1) == probs.argmin(1)).all()
    # identical fast/slow -> KL term vanishes
    P_both = score_actions(lf, lf, q, prio, DreamConfig(rarity="both"))
    P_sur = score_actions(lf, lf, q, prio, DreamConfig(rarity="surprisal"))
    assert torch.allclose(P_both, P_sur, atol=1e-5)


def test_branch_and_control_share_root_and_noise(setup):
    pool, net, _, buf = setup
    rng = np.random.default_rng(1)
    specs = build_specs(buf, [(5, 3, 2)], DreamConfig(m_rollouts=2, horizon=6), pool.n_actions, rng)
    assert len(specs) == 4
    b0, c0, b1, c1 = specs
    assert b0.state == c0.state == b1.state == c1.state
    assert np.array_equal(b0.noise, c0.noise) and not np.array_equal(b0.noise, b1.noise)
    assert b0.prefix == [3, 3] and c0.prefix == [] and b0.length == c0.length == 8
    rolls = run_rollouts(pool, net, DEV, specs, 0.99)
    assert all(np.array_equal(r.obs[0], buf.items[5].obs) for r in rolls)
    assert list(rolls[0].actions[:2]) == [3, 3]
    assert all(len(r.actions) == len(r.rewards) == len(r.obs) <= 8 for r in rolls)


def test_same_spec_twice_is_identical(setup):
    pool, net, _, buf = setup
    noise = gumbel(np.random.default_rng(2), (10, pool.n_actions))
    spec = RolloutSpec(state=buf.items[3].state, prefix=[], length=10, noise=noise)
    r1, r2 = run_rollouts(pool, net, DEV, [spec, spec], 0.99)
    assert np.array_equal(r1.actions, r2.actions) and np.array_equal(r1.obs, r2.obs)
    assert np.array_equal(r1.returns, r2.returns)


@pytest.mark.parametrize("selection", ["targeted", "random", "go_explore"])
def test_selection_modes_valid(setup, selection):
    pool, net, slow, buf = setup
    cfg = Config()
    cfg.dream.selection = selection
    branches, diag = select_branches(buf, net, slow, DEV, cfg, 10, np.random.default_rng(0))
    assert len(branches) == 10 and len({b[0] for b in branches}) == 10
    for j, a, k in branches:
        assert 0 <= j < len(buf) and 0 <= a < pool.n_actions and 1 <= k <= cfg.dream.k_max


def test_dream_phase_runs_and_counts_steps(setup):
    pool, net, slow, buf = setup
    cfg = apply_method(Config(method="full"))
    cfg.dream.horizon, cfg.dream.update_epochs = 8, 1
    cfg.amp = False
    before = pool.counters["dream"]
    stats = dream_phase(cfg, pool, net, slow, buf, DEV, wake_steps_this_cycle=2000,
                        rng=np.random.default_rng(0))
    assert stats["dream/snapshots"] > 0
    assert pool.counters["dream"] - before == stats["dream/env_steps"] > 0
    assert stats["dream/env_steps"] <= 0.3 * 2000 + 1
    for key in ["dream/A_mean", "dream/pi_rare_before", "dream/pi_rare_after", "dream/q_loss"]:
        assert key in stats


def test_dream_phase_empty_buffer_is_noop(setup):
    pool, net, slow, _ = setup
    with pytest.warns(UserWarning):
        stats = dream_phase(Config(), pool, net, slow, SnapshotBuffer(10), DEV, 50000, np.random.default_rng(0))
    assert stats == {"dream/snapshots": 0}


def test_awr_increases_prob_of_positive_advantage_action():
    torch.manual_seed(0)
    net = ActorCritic((7, 7, 3), 7, 5)
    obs = np.random.default_rng(0).integers(0, 6, size=(8, 7, 7, 3)).astype(np.uint8)
    acts = np.full(8, 4)
    adv = np.ones(8, dtype=np.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    p0 = torch.softmax(net(torch.as_tensor(obs))[0], -1)[:, 4].mean().item()
    cfg = DreamConfig(update_epochs=20, anchor_coef=0.0)
    dream_update(net, opt, {"obs": obs, "actions": acts, "returns": np.zeros(8, np.float32)},
                 {"obs": obs, "actions": acts, "adv": adv}, cfg, DEV)
    p1 = torch.softmax(net(torch.as_tensor(obs))[0], -1)[:, 4].mean().item()
    assert p1 > p0
