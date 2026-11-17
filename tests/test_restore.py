import numpy as np
import pytest

from dreaming.envs import make_env


def _rollout(env, actions):
    obs, rews, dones = [], [], []
    for a in actions:
        o, r, d, _ = env.step(int(a))
        obs.append(o)
        rews.append(r)
        dones.append(d)
        if d:
            break
    return np.stack(obs), np.array(rews), np.array(dones)


@pytest.mark.parametrize("seed", range(5))
def test_crafter_restore_is_bit_exact(seed):
    env = make_env("crafter", seed=seed)
    env.reset(seed=seed)
    rng = np.random.RandomState(seed)
    for _ in range(200):
        env.step(rng.randint(17))
    snap = env.get_state()
    actions = rng.randint(17, size=300)
    outs = []
    for _ in range(3):
        env.set_state(snap)
        outs.append(_rollout(env, actions))
    for o in outs[1:]:
        assert o[0].shape == outs[0][0].shape
        assert np.array_equal(o[0], outs[0][0])
        assert np.array_equal(o[1], outs[0][1])


@pytest.mark.parametrize("env_id", ["MiniGrid-DoorKey-8x8-v0", "MiniGrid-KeyCorridorS3R3-v0"])
def test_minigrid_restore_is_bit_exact(env_id):
    env = make_env(f"minigrid:{env_id}")
    env.reset(seed=3)
    rng = np.random.RandomState(0)
    for _ in range(20):
        env.step(rng.randint(7))
    snap = env.get_state()
    actions = rng.randint(7, size=100)
    outs = []
    for _ in range(3):
        restored_obs = env.set_state(snap)
        outs.append((restored_obs, *_rollout(env, actions)))
    for o in outs[1:]:
        assert np.array_equal(o[0], outs[0][0])
        assert np.array_equal(o[1], outs[0][1])
        assert np.array_equal(o[2], outs[0][2])


def test_seeded_reset_is_reproducible():
    for name in ["crafter", "minigrid:MiniGrid-DoorKey-8x8-v0"]:
        a, b = make_env(name, seed=1), make_env(name, seed=2)
        assert np.array_equal(a.reset(seed=123), b.reset(seed=123))


def test_episode_stats_restored():
    env = make_env("minigrid:MiniGrid-DoorKey-8x8-v0")
    env.reset(seed=0)
    for _ in range(5):
        env.step(2)
    snap = env.get_state()
    for _ in range(3):
        env.step(2)
    env.set_state(snap)
    assert env._len == 5
