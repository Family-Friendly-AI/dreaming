import dataclasses
import json
from typing import get_args

import pytest

from dreaming.config import Config, EnvConfig, Method
from dreaming.train import train


def tiny(tmp_path, method, **kw) -> Config:
    cfg = Config(
        method=method,
        total_wake_steps=2048,
        budget_axis="total",
        wake_steps_per_cycle=1024,
        env=EnvConfig(num_envs=8, max_episode_steps=50),
        run_dir=str(tmp_path),
        run_name=method,
        device="cpu",
    )
    cfg = dataclasses.replace(cfg, **kw)
    cfg.ppo.num_steps = 16
    cfg.gate.n_seeds, cfg.gate.max_steps = 4, 20
    return cfg


@pytest.mark.parametrize("method", get_args(Method))
def test_total_budget_stops_training(tmp_path, method):
    s = train(tiny(tmp_path, method))
    per_iter = 16 * 8
    assert s["steps/total"] == s["steps/wake"] + s["steps/dream"] + s["steps/gate"]
    assert 2048 <= s["steps/total"] < 2048 + per_iter + s["steps/dream"] + s["steps/gate"]
    assert s["budget_axis"] == "total"
    assert s["final/episodes"] > 0
    cfg = json.loads((tmp_path / method / "config.json").read_text())
    assert "git_sha" in cfg


def test_same_seed_same_metrics(tmp_path):
    a = train(tiny(tmp_path / "a", "ppo"))
    b = train(tiny(tmp_path / "b", "ppo"))
    assert a["final/return"] == b["final/return"] and a["final/episodes"] == b["final/episodes"]


def test_gpu_without_gpu_names_the_fix(tmp_path):
    import jax

    if any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("a GPU is present")
    with pytest.raises(RuntimeError, match="--device cpu"):
        train(tiny(tmp_path, "ppo", device="gpu"))
