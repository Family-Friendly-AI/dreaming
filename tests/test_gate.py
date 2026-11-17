import numpy as np

from dreaming.config import GateConfig
from dreaming.phases.gate import GateState, gate_decision

CFG = GateConfig(eps=0.05, n_boot=2000)


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
