import json

from dreaming.metrics import RunLogger, git_sha


def test_config_records_git_sha_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_SHA", "abc123")
    RunLogger(str(tmp_path), {"seed": 0}).close()
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg == {"seed": 0, "git_sha": "abc123"}


def test_git_sha_falls_back_to_checkout(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    sha = git_sha()
    assert sha == "unknown" or len(sha.removesuffix("-dirty")) == 40
