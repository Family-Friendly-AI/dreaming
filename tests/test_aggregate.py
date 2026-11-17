import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from aggregate import aggregate, load_runs  # noqa: E402


def _write_run(root, name, method, slope, seed):
    d = os.path.join(root, name)
    os.makedirs(d)
    rng = np.random.default_rng(seed)
    wake = np.arange(1, 21) * 1000
    df = pd.DataFrame(
        {
            "steps/wake": wake,
            "steps/total": (wake * 1.5).astype(int),
            "train/score": slope * wake / 1000 + rng.normal(0, 0.1, len(wake)),
        }
    )
    df.to_csv(os.path.join(d, "progress.csv"), index=False)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"method": method}, f)


def test_aggregate_synthetic(tmp_path):
    # Synthetic per-seed CSVs only: no real multi-seed training is run.
    for s in range(5):
        _write_run(tmp_path, f"a{s}", "full", 1.0, s)
        _write_run(tmp_path, f"b{s}", "no_dream", 0.5, 100 + s)
    runs = load_runs(str(tmp_path), "train/score")
    assert set(runs) == {"full", "no_dream"} and len(runs["full"]) == 5
    curves, final = aggregate(runs, "train/score", n_points=10, reps=200)
    assert set(final["x_axis"]) == {"steps/wake", "steps/total"}
    f = final[final.x_axis == "steps/wake"].set_index("method")
    assert f.loc["full", "iqm"] > f.loc["no_dream", "iqm"]
    assert f.loc["full", "ci_low"] > f.loc["no_dream", "ci_high"]
    assert (curves.ci_low <= curves.iqm + 1e-9).all() and (curves.iqm <= curves.ci_high + 1e-9).all()
