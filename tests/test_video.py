import jax
import jax.numpy as jnp
import numpy as np

from dreaming.agent.networks import ActorCritic, apply_fn, init_params
from dreaming.envs import craftax as E
from dreaming.phases.video import make_video_rollout

NET = ActorCritic(E.NUM_ACTIONS, n_q_heads=3, hidden=32)
PARAMS = init_params(NET, jax.random.key(0), E.OBS_DIM)


def test_video_frames_stop_at_episode_end_and_replay():
    video = make_video_rollout(apply_fn(NET), E.env_params(max_episode_steps=12), max_steps=20)
    frames, n = video(PARAMS, jax.random.key(1))
    assert frames.dtype == jnp.uint8 and frames.shape == (20, 144, 144, 3)
    assert int(n) == 12
    assert int(frames[: int(n)].max()) > 0
    again, _ = video(PARAMS, jax.random.key(1))
    np.testing.assert_array_equal(np.asarray(frames), np.asarray(again))
