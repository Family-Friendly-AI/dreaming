import json

import pytest

from analysis.crafter_paper import PAPER_SCORES, PAPER_SUCCESS, compare, crafter_score, load_summaries
from dreaming.metrics import EpisodeTracker


@pytest.mark.parametrize("method", list(PAPER_SUCCESS))
def test_score_formula_reproduces_paper(method):
    # The paper averages per-seed scores; scoring the seed-averaged rates (Appendix A/B) is >= that
    # by Jensen's inequality, and close to it.
    s, paper = crafter_score(PAPER_SUCCESS[method]), PAPER_SCORES[method][0]
    assert paper - 0.1 <= s <= paper * 1.1 + 0.1


def test_return_since():
    t = EpisodeTracker(["a"])
    t.add([{"r": 1.0, "l": 5, "achievements": set()}], step=10)
    t.add([{"r": 3.0, "l": 5, "achievements": {"a"}}], step=20)
    assert t.return_since(15) == 3.0
    assert t.return_since(0) == 2.0
    assert t.return_since(30) is None


def test_compare_includes_paper_and_runs(tmp_path):
    d = tmp_path / "r0"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"method": "ppo", "env": {"name": "crafter"}}))
    (d / "summary.json").write_text(
        json.dumps(
            {
                "budget_axis": "total",
                "budget": 1_000_000,
                "achievements/collect_wood": 80.0,
                "final/return_last100k": 3.0,
            }
        )
    )
    table, success = compare(load_summaries(str(tmp_path)))
    assert "ppo" in set(table["method"]) and "paper: PPO" in set(table["method"])
    assert success.loc["collect_wood", "ppo"] == 80.0
