"""Compare runs against the Crafter paper (Hafner 2021, arXiv:2109.06780, Table 1 + Appendix A-C).

Protocol of the paper: 1M env steps, success rate of an achievement = % of *training* episodes
within the budget in which it was unlocked at least once, score = exp(mean ln(1 + s_i)) - 1 over
the 22 achievements, return = mean over episodes in the last 1e5 steps. Paper numbers are mean
(± std) over 10 seeds. Our runs must use `--budget-axis total` to be comparable (dream and gate
steps count toward the 1M); `scripts/crafter_*.sh` do this by default.

Usage:
    python analysis/crafter_paper.py runs/ --out analysis/out
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# Table 1: (score %, score std, return, return std)
PAPER_SCORES = {
    "Human Experts": (50.5, 6.8, 14.3, 2.3),
    "DreamerV2": (10.0, 1.2, 9.0, 1.7),
    "PPO": (4.6, 0.3, 4.2, 1.2),
    "Rainbow": (4.3, 0.2, 5.0, 1.3),
    "Plan2Explore (no reward)": (2.1, 0.1, 2.1, 1.5),
    "RND (no reward)": (2.0, 0.1, 0.7, 1.3),
    "Random": (1.6, 0.0, 2.1, 1.3),
}

ACH_ORDER = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling",
    "collect_stone", "collect_wood", "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant", "place_stone",
    "place_table", "wake_up",
]

# Appendix A/B success rates (%), in ACH_ORDER.
PAPER_SUCCESS = {
    "Rainbow": [0.0, 0.0, 24.0, 0.0, 97.4, 0.2, 74.9, 0.7, 39.6, 26.1, 0.0,
                0.0, 0.0, 0.0, 0.0, 4.8, 9.8, 0.0, 94.2, 0.0, 52.3, 93.3],
    "PPO": [0.4, 0.0, 30.3, 0.0, 66.7, 3.0, 83.0, 0.2, 2.0, 12.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 21.1, 20.1, 0.1, 65.0, 1.7, 66.1, 92.5],
    "DreamerV2": [14.7, 0.0, 80.0, 0.0, 86.6, 42.7, 92.7, 2.6, 53.1, 17.1, 0.1,
                  0.0, 0.0, 0.2, 0.3, 59.6, 40.2, 1.8, 84.4, 29.0, 85.7, 92.8],
    "Random": [0.0, 0.0, 9.3, 0.0, 50.2, 0.0, 24.4, 0.0, 0.1, 0.4, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 44.6, 0.0, 4.4, 93.6],
}


def crafter_score(rates_percent) -> float:
    s = np.asarray(rates_percent, dtype=np.float64)
    return float(np.exp(np.mean(np.log(1.0 + s))) - 1.0)


def load_summaries(root: str, budget: int = 1_000_000) -> dict[str, list[dict]]:
    """Group `summary.json` of Crafter runs by method; warns about runs off the paper protocol."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        s_path, c_path = os.path.join(d, "summary.json"), os.path.join(d, "config.json")
        if not (os.path.isfile(s_path) and os.path.isfile(c_path)):
            continue
        with open(s_path) as f:
            s = json.load(f)
        with open(c_path) as f:
            cfg = json.load(f)
        if cfg.get("env", {}).get("name") != "crafter":
            continue
        if s.get("budget_axis") != "total" or s.get("budget") != budget:
            print(f"warning: {name} is not on the paper protocol (budget_axis={s.get('budget_axis')}, "
                  f"budget={s.get('budget')}); comparison is not like-for-like")
        s["_run"] = name
        runs[cfg["method"]].append(s)
    return runs


def compare(runs: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"method": f"paper: {m}", "score": v[0], "score_std": v[1], "return": v[2],
             "return_std": v[3], "n_seeds": 10} for m, v in PAPER_SCORES.items()]
    ach = {f"paper: {m}": v for m, v in PAPER_SUCCESS.items()}
    for m, ss in sorted(runs.items()):
        rates = np.array([[s.get(f"achievements/{a}", 0.0) for a in ACH_ORDER] for s in ss])
        scores = np.array([crafter_score(r) for r in rates])
        rets = np.array([s["final/return_last100k"] for s in ss
                         if s.get("final/return_last100k") is not None], dtype=float)
        rows.append({"method": m, "score": scores.mean(), "score_std": scores.std(),
                     "return": rets.mean() if rets.size else np.nan,
                     "return_std": rets.std() if rets.size else np.nan, "n_seeds": len(ss)})
        ach[m] = rates.mean(0).tolist()
    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    success = pd.DataFrame(ach, index=pd.Index(ACH_ORDER))
    # Score of the seed-averaged rates; >= the paper's score (mean of per-seed scores, Jensen).
    success.loc["score_of_mean_rates"] = [crafter_score(success[c].iloc[:22]) for c in success.columns]
    return table, success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--out", default="analysis/out")
    p.add_argument("--budget", type=int, default=1_000_000)
    a = p.parse_args()
    runs = load_summaries(a.root, a.budget)
    if not runs:
        raise SystemExit(f"no Crafter runs with summary.json under {a.root}")
    table, success = compare(runs)
    os.makedirs(a.out, exist_ok=True)
    table.to_csv(os.path.join(a.out, "crafter_paper_scores.csv"), index=False)
    success.to_csv(os.path.join(a.out, "crafter_paper_success_rates.csv"))
    print(table.to_string(index=False, float_format="%.2f"))
    print()
    print(success.to_string(float_format="%.1f"))


if __name__ == "__main__":
    main()
