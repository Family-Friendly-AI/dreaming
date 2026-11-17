import numpy as np
import pytest

from dreaming.envs import EnvPool

ENV = "minigrid:MiniGrid-DoorKey-8x8-v0"


@pytest.fixture(params=["serial", "process"])
def pool(request):
    p = EnvPool(ENV, num_envs=4, seed=0, mode=request.param)
    yield p
    p.close()


def test_reset_and_step_shapes(pool):
    obs = pool.reset()
    assert obs.shape == (4, 7, 7, 3)
    obs, rew, done, infos, states = pool.step(np.zeros(4, dtype=int))
    assert obs.shape == (4, 7, 7, 3) and rew.shape == (4,) and done.shape == (4,)
    assert len(infos) == 4 and all("visit_key" in i for i in infos)
    assert states == [None] * 4


def test_per_index_restore_round_trip(pool):
    pool.reset(seeds=[10, 11, 12, 13])
    pool.step([2, 2, 2, 2])
    (snap,) = pool.get_state([1])
    ref = [pool.step([a], idx=[1], auto_reset=False)[0][0] for a in [0, 2, 1, 2]]
    # Restore the same snapshot into two different workers: both must replay identically.
    restored = pool.set_state([0, 3], [snap, snap])
    assert np.array_equal(restored[0], restored[1])
    for t, a in enumerate([0, 2, 1, 2]):
        obs = pool.step([a, a], idx=[0, 3], auto_reset=False)[0]
        assert np.array_equal(obs[0], ref[t]) and np.array_equal(obs[1], ref[t])


def test_snapshot_capture_is_pre_action(pool):
    obs0 = pool.reset(seeds=[1, 2, 3, 4])
    _, _, _, _, states = pool.step([2, 2, 2, 2], snapshot_idx=[2])
    assert states[0] is None and states[2] is not None
    restored = pool.set_state([2], [states[2]])
    assert np.array_equal(restored[0], obs0[2])


def test_counters_per_phase(pool):
    pool.reset()
    pool.step([0] * 4, phase="wake")
    pool.step([0, 0], idx=[0, 1], phase="dream", auto_reset=False)
    pool.step([0], idx=[3], phase="gate", auto_reset=False)
    assert pool.counters["wake"] == 4
    assert pool.counters["dream"] == 2
    assert pool.counters["gate"] == 1
    assert pool.total_steps == 7


def test_crafter_pool_restore():
    with EnvPool("crafter", num_envs=2, seed=0, mode="process") as p:
        p.reset(seeds=[5, 5])
        for _ in range(30):
            p.step([1, 1])
        (snap,) = p.get_state([0])
        p.set_state([0, 1], [snap, snap])
        for a in np.random.RandomState(0).randint(17, size=50):
            obs = p.step([a, a], auto_reset=False)[0]
            assert np.array_equal(obs[0], obs[1])
