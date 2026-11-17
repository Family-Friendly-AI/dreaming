"""Aggregate per-seed runs with rliable: IQM + stratified bootstrap 95% CIs on both x-axes.

Usage:
    python analysis/aggregate.py runs/ --metric train/score --out analysis/out
Run directories are grouped by method from their `config.json`. For each method and x-axis
(`steps/wake`, `steps/total`) the metric is interpolated onto a common grid, then IQM and CIs are
computed across seeds at every grid point; the final point is also reported in `final.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics

X_AXES = ("steps/wake", "steps/total")


def load_runs(root: str, metric: str) -> dict[str, list[pd.DataFrame]]:
    runs: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        csv, cfg = os.path.join(d, "progress.csv"), os.path.join(d, "config.json")
        if not (os.path.isfile(csv) and os.path.isfile(cfg)):
            continue
        df = pd.read_csv(csv)
        if metric not in df:
            continue
        with open(cfg) as f:
            method = json.load(f)["method"]
        runs[method].append(df)
    return runs


def on_grid(dfs: list[pd.DataFrame], x: str, metric: str, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate every seed onto a shared grid up to the shortest run; returns (grid, [seeds, points])."""
    curves = [df.loc[:, [x, metric]].dropna().sort_values(by=x) for df in dfs]
    curves = [c for c in curves if len(c)]
    x_max = min(c[x].iloc[-1] for c in curves)
    grid = np.linspace(0, x_max, n_points)
    ys = np.stack([np.interp(grid, c[x].to_numpy(), c[metric].to_numpy()) for c in curves])
    return grid, ys


def aggregate(runs: dict[str, list[pd.DataFrame]], metric: str, n_points: int = 50, reps: int = 2000,
              seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, final = [], []
    for x in X_AXES:
        grids = {m: on_grid(dfs, x, metric, n_points) for m, dfs in runs.items()}
        # score matrix per method: [seeds, points]; rliable expects [runs, tasks] -> one task per point
        scores = {m: ys for m, (_, ys) in grids.items()}
        iqm = lambda s: np.array([metrics.aggregate_iqm(s[:, [j]]) for j in range(s.shape[1])])  # noqa: E731
        point, ci = rly.get_interval_estimates(scores, iqm, reps=reps, random_state=np.random.RandomState(seed))
        for m, (grid, ys) in grids.items():
            for j, xv in enumerate(grid):
                rows.append({"x_axis": x, "method": m, "x": xv, "iqm": point[m][j],
                             "ci_low": ci[m][0, j], "ci_high": ci[m][1, j], "n_seeds": ys.shape[0]})
            final.append({"x_axis": x, "method": m, "x": grid[-1], "iqm": point[m][-1],
                          "ci_low": ci[m][0, -1], "ci_high": ci[m][1, -1], "n_seeds": ys.shape[0]})
    return pd.DataFrame(rows), pd.DataFrame(final)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--metric", default="train/score")
    p.add_argument("--out", default="analysis/out")
    p.add_argument("--points", type=int, default=50)
    p.add_argument("--reps", type=int, default=2000)
    a = p.parse_args()
    runs = load_runs(a.root, a.metric)
    if not runs:
        raise SystemExit(f"no runs with metric {a.metric!r} under {a.root}")
    curves, final = aggregate(runs, a.metric, a.points, a.reps)
    os.makedirs(a.out, exist_ok=True)
    tag = a.metric.replace("/", "_")
    curves.to_csv(os.path.join(a.out, f"curves_{tag}.csv"), index=False)
    final.to_csv(os.path.join(a.out, f"final_{tag}.csv"), index=False)
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
