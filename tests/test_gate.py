import copy

import numpy as np
import pandas as pd
import pytest
import torch

from dreaming.agent.distill import distil
from dreaming.agent.networks import ActorCritic
from dreaming.config import Config, ConsolidateConfig, EnvConfig, GateConfig
from dreaming.train import train
from dreaming.envs import GATE_SEED_BASE, EnvPool
from dreaming.phases.gate import GateState, evaluate, gate_decision, gate_seeds, run_gate

CFG = GateConfig(eps=0.05, n_boot=2000)
DEV = torch.device("cpu")


def rng():
    return np.random.default_rng(0)


def test_accept_on_equal_performance():
    r = np.random.default_rng(1).normal(10, 2, size=32)
    ok, info = gate_decision(r, r.copy(), set(), set(), CFG, rng())
    assert ok and info["gate/non_inferior"] == 1.0 and info["gate/delta_mean"] == 0.0


def test_accept_on_zero_returns():
    z = np.zeros(32)
    assert gate_decision(z, z, set(), set(), CFG, rng())[0]


def test_reject_on_clear_drop():
    old = np.random.default_rng(2).normal(10, 1, size=32)
    cand = old - 3.0
    ok, info = gate_decision(cand, old, {"collect_wood"}, {"collect_wood"}, CFG, rng())
    assert not ok and info["gate/accepted"] == 0.0


def test_new_skill_override_despite_drop():
    old = np.random.default_rng(3).normal(10, 1, size=32)
    cand = old - 3.0
    ok, info = gate_decision(cand, old, {"collect_wood", "place_table"}, {"collect_wood"}, CFG, rng())
    assert ok and info["gate/new_skill_override"] == 1.0 and info["gate/new_skills"] == "place_table"


def test_small_dip_within_eps_is_accepted():
    old = np.full(32, 10.0)
    cand = old - 0.1  # 1% drop, eps = 5%
    assert gate_decision(cand, old, set(), set(), CFG, rng())[0]


def test_rejection_rate_tracking():
    s = GateState()
    s.n_decisions, s.n_rejections = 4, 1
    assert s.rejection_rate == 0.25


def test_gate_seeds_are_held_out():
    assert all(s >= GATE_SEED_BASE for s in gate_seeds(50))


def test_identical_policies_accepted_and_paired():
    torch.manual_seed(0)
    pool = EnvPool("minigrid:MiniGrid-DoorKey-5x5-v0", num_envs=4, seed=0, mode="serial")
    net = ActorCritic(pool.obs_shape, pool.n_actions, 5)
    twin = copy.deepcopy(net)
    seeds = gate_seeds(6)
    r1, _ = evaluate(pool, net, seeds, 50, DEV, noise_seed=7)
    r2, _ = evaluate(pool, twin, seeds, 50, DEV, noise_seed=7)
    assert np.array_equal(r1, r2)
    state = GateState()
    ok, info = run_gate(pool, twin, net, state, GateConfig(n_seeds=6, max_steps=50), DEV, False, rng(), 0)
    assert ok and state.rejection_rate == 0.0
    assert 0 < info["gate/env_steps"] <= 2 * 6 * 50
    assert pool.counters["gate"] > 0


def test_gate_seed_blocks_are_disjoint():
    assert not set(gate_seeds(16, block=0)) & set(gate_seeds(16, block=1))


def _always_reject_gate(monkeypatch, **kw):
    """The paired test always fails, so only the reject-streak escape can accept."""
    import dreaming.phases.gate as gate_mod
    monkeypatch.setattr(gate_mod, "gate_decision", lambda *a, **k: (False, {}))
    torch.manual_seed(0)
    pool = EnvPool("minigrid:MiniGrid-DoorKey-5x5-v0", num_envs=4, seed=0, mode="serial")
    net = ActorCritic(pool.obs_shape, pool.n_actions, 5)
    cfg = GateConfig(n_seeds=4, max_steps=20, **kw)
    return pool, net, cfg


def test_rotating_gate_reevaluates_old_each_cycle(monkeypatch):
    pool, net, cfg = _always_reject_gate(monkeypatch, max_reject_streak=0)
    state = GateState()
    before = pool.counters["gate"]
    run_gate(pool, net, copy.deepcopy(net), state, cfg, DEV, False, rng(), cycle=0)
    first = pool.counters["gate"] - before
    run_gate(pool, net, copy.deepcopy(net), state, cfg, DEV, False, rng(), cycle=1)
    # Both policies are evaluated on fresh seeds every cycle (no cached old score).
    assert pool.counters["gate"] - before - first == first


def test_forced_accept_after_reject_streak(monkeypatch):
    pool, net, cfg = _always_reject_gate(monkeypatch, max_reject_streak=2)
    state = GateState()
    decisions = [run_gate(pool, net, net, state, cfg, DEV, False, rng(), cycle=c) for c in range(6)]
    accepted = [ok for ok, _ in decisions]
    forced = [info["gate/forced"] for _, info in decisions]
    assert accepted == [False, False, True, False, False, True]
    assert forced == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert state.n_forced == 2 and state.rejection_rate == 4 / 6


def test_no_forced_accept_when_disabled(monkeypatch):
    pool, net, cfg = _always_reject_gate(monkeypatch, max_reject_streak=0)
    state = GateState()
    assert not any(run_gate(pool, net, net, state, cfg, DEV, False, rng(), cycle=c)[0] for c in range(4))


def test_distil_moves_slow_towards_fast():
    torch.manual_seed(0)
    fast = ActorCritic((7, 7, 3), 7, 5)
    slow = ActorCritic((7, 7, 3), 7, 5)
    obs = np.random.default_rng(0).integers(0, 6, size=(512, 7, 7, 3)).astype(np.uint8)
    o = torch.as_tensor(obs)

    def kl(a, b):
        la, lb = torch.log_softmax(a(o)[0], -1), torch.log_softmax(b(o)[0], -1)
        return (la.exp() * (la - lb)).sum(-1).mean().item()

    before = kl(fast, slow)
    cand, stats = distil(fast, slow, obs, ConsolidateConfig(distil_epochs=20, minibatch_size=128, beta_tr=0.0), DEV)
    assert kl(fast, cand) < 0.5 * before
    assert "distil/kl_fast" in stats


@pytest.mark.parametrize("mode", ["never", "on_reject", "always"])
def test_opt_reset_modes(tmp_path, mode):
    cfg = Config(method="no_dream", total_wake_steps=1024, wake_steps_per_cycle=256, opt_reset=mode,
                 env=EnvConfig(name="minigrid:MiniGrid-DoorKey-5x5-v0", num_envs=4, pool_mode="serial"),
                 run_dir=str(tmp_path), run_name="o", device="cpu")
    cfg.ppo.num_steps = 64
    cfg.gate.n_seeds, cfg.gate.max_steps = 4, 20
    train(cfg)
    df = pd.read_csv(tmp_path / "o" / "progress.csv")
    c = df[df["opt/reset"].notna()]
    assert len(c) >= 3
    acc = np.asarray(c["gate/accepted"])
    expected = {"never": 0.0 * acc, "always": 0.0 * acc + 1.0, "on_reject": 1.0 - acc}[mode]
    assert (np.asarray(c["opt/reset"]) == expected).all()


@pytest.mark.parametrize("reset_fast", [True, False])
def test_reset_fast_flag(tmp_path, monkeypatch, reset_fast):
    import dreaming.train as train_mod
    snaps = {}
    orig_sleep = train_mod.sleep_cycle

    def clone(net):
        return {k: v.detach().clone() for k, v in net.state_dict().items()}

    def sleep(ctx, cycle, wake_steps):
        before = clone(ctx["fast"])
        out = orig_sleep(ctx, cycle, wake_steps)
        snaps[cycle] = (before, clone(ctx["fast"]), clone(ctx["slow"]))
        return out
    monkeypatch.setattr(train_mod, "sleep_cycle", sleep)
    cfg = Config(method="no_gate", total_wake_steps=512, wake_steps_per_cycle=256, reset_fast=reset_fast,
                 env=EnvConfig(name="minigrid:MiniGrid-DoorKey-5x5-v0", num_envs=4, pool_mode="serial"),
                 run_dir=str(tmp_path), run_name="r", device="cpu")
    cfg.ppo.num_steps = 64
    train_mod.train(cfg)
    before, fast_sd, slow_sd = snaps[0]
    assert all(torch.equal(before[k], fast_sd[k]) for k in before) == (not reset_fast)
    assert all(torch.equal(slow_sd[k], fast_sd[k]) for k in slow_sd) == reset_fast
