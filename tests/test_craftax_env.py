import jax
import jax.numpy as jnp
import numpy as np

from dreaming.envs import craftax as E

PARAMS = E.env_params(max_episode_steps=60)
N, T = 5, 300


def _rollout(carry, keys, actions, auto_reset=False):
    def body(c, x):
        k, a = x
        if auto_reset:
            return E.step_auto_reset(c, a, k[0], PARAMS)
        return E.step(c, a, k, PARAMS)

    return jax.lax.scan(body, carry, (keys, actions))


def _inputs(seed):
    k = jax.random.key(seed)
    keys = jax.random.split(k, T * N).reshape(T, N)
    actions = jax.random.randint(jax.random.fold_in(k, 1), (T, N), 0, E.NUM_ACTIONS)
    return keys, actions


def test_achievement_names_match_crafter():
    assert len(E.ACHIEVEMENTS) == 22
    assert {"collect_wood", "place_table", "collect_diamond", "wake_up", "eat_cow"} <= set(E.ACHIEVEMENTS)


def test_restore_replays_exactly():
    keys, actions = _inputs(0)
    big = E.env_params(max_episode_steps=10_000)
    carry = E.reset(jax.random.split(jax.random.key(1), N), big)

    def run(c, ks, acts):
        return jax.lax.scan(lambda c, x: E.step(c, x[1], x[0], big), c, (ks, acts))

    mid, _ = run(carry, keys[:100], actions[:100])
    _, tail = run(mid, keys[100:], actions[100:])
    _, replay = run(E.restore(mid.state), keys[100:], actions[100:])
    for a, b in [(tail.next_obs, replay.next_obs), (tail.reward, replay.reward), (tail.done, replay.done)]:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_episode_bookkeeping_with_auto_reset():
    keys, actions = _inputs(2)
    carry = E.reset(jax.random.split(jax.random.key(3), N), PARAMS)
    _, out = jax.jit(lambda c, k, a: _rollout(c, k, a, auto_reset=True))(carry, keys, actions)
    done = np.asarray(out.done)
    assert done.sum() >= N * (T // 60) - N
    ep_len = np.asarray(out.ep_length)
    rew = np.asarray(out.reward)
    for i in range(N):
        ends = np.flatnonzero(done[:, i])
        start = 0
        for e in ends:
            assert ep_len[e, i] == e - start + 1
            np.testing.assert_allclose(np.asarray(out.ep_return)[e, i], rew[start : e + 1, i].sum(), rtol=1e-5)
            start = e + 1


def test_unlocked_is_achievement_diff():
    keys, actions = _inputs(4)
    carry = E.reset(jax.random.split(jax.random.key(5), N), E.env_params(None))
    _, out = _rollout(carry, keys, actions)
    ach = np.asarray(out.ep_achievements)
    prev = np.concatenate([np.zeros_like(ach[:1]), ach[:-1]])
    np.testing.assert_array_equal(np.asarray(out.unlocked), ach & ~prev)
    assert np.asarray(out.unlocked).sum(0).max() <= 1


def test_visit_key_is_int32_and_nonnegative():
    carry = E.reset(jax.random.split(jax.random.key(6), N), PARAMS)
    vk = E.visit_key(carry.state)
    assert vk.dtype == jnp.int32 and bool((vk >= 0).all())


def test_gate_blocks_are_disjoint():
    root = jax.random.key(0)
    a = jax.random.key_data(E.gate_world_keys(root, 0, 16))
    b = jax.random.key_data(E.gate_world_keys(root, 1, 16))
    wake = jax.random.key_data(E.stream(root, E.KeyStream.WAKE))
    rows_a = {tuple(r) for r in np.asarray(a)}
    assert len(rows_a) == 16
    assert rows_a.isdisjoint({tuple(r) for r in np.asarray(b)})
    assert tuple(np.asarray(wake)) not in rows_a
