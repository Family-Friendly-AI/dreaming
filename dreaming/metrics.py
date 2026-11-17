"""Episode metrics (returns, achievements, Crafter score, unlock times) and run logging."""

from __future__ import annotations

import json
import os
from collections import deque

import numpy as np


def crafter_score(success_rates_percent: np.ndarray) -> float:
    """Crafter score (Hafner 2021): geometric mean of per-achievement success rates (in %)."""
    s = np.asarray(success_rates_percent, dtype=np.float64)
    if s.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(1.0 + s))) - 1.0)


class EpisodeTracker:
    """Tracks wake episodes. Success rates follow the Crafter protocol (all episodes so far)."""

    def __init__(self, achievements: list[str] | None = None, window: int = 100):
        self.achievements = list(achievements or [])
        self.returns: deque = deque(maxlen=window)
        self.lengths: deque = deque(maxlen=window)
        self.n_episodes = 0
        self.ach_counts: dict[str, int] = {a: 0 for a in self.achievements}
        self.unlock_step: dict[str, int] = {}
        self.history: list[tuple[int, float]] = []
        """(wake step at episode end, return) for every episode."""

    def add(self, episodes: list[dict], step: int):
        for ep in episodes:
            self.n_episodes += 1
            self.returns.append(ep["r"])
            self.lengths.append(ep["l"])
            self.history.append((step, float(ep["r"])))
            for a in ep["achievements"]:
                self.ach_counts[a] = self.ach_counts.get(a, 0) + 1
                self.unlock_step.setdefault(a, step)

    def return_since(self, step: int) -> float | None:
        """Mean return of episodes that ended at or after `step` (Crafter paper: last 1e5 steps)."""
        rs = [r for s, r in self.history if s >= step]
        return float(np.mean(rs)) if rs else None

    def success_rates(self) -> dict[str, float]:
        n = max(self.n_episodes, 1)
        names = self.achievements or sorted(self.ach_counts)
        return {a: 100.0 * self.ach_counts.get(a, 0) / n for a in names}

    def summary(self, prefix: str = "train") -> dict:
        out: dict[str, float] = {f"{prefix}/episodes": self.n_episodes}
        if self.returns:
            out[f"{prefix}/return"] = float(np.mean(self.returns))
            out[f"{prefix}/length"] = float(np.mean(self.lengths))
        rates = self.success_rates()
        if rates:
            out[f"{prefix}/score"] = crafter_score(np.array(list(rates.values())))
            out[f"{prefix}/n_unlocked"] = sum(1 for v in self.ach_counts.values() if v > 0)
            for a, r in rates.items():
                out[f"achievements/{a}"] = r
        return out


class RunLogger:
    """Writes `metrics.jsonl` + `progress.csv` in `run_dir`; mirrors to wandb when enabled."""

    def __init__(
        self, run_dir: str, config: dict, track: bool = False, project: str = "dreaming", name: str | None = None
    ):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.rows: list[dict] = []
        self._jsonl = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
        self.wandb = None
        if track:
            import wandb

            self.wandb = wandb.init(project=project, name=name, config=config, dir=run_dir)

    def log(self, row: dict):
        row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
        self.rows.append(row)
        self._jsonl.write(json.dumps(row) + "\n")
        self._jsonl.flush()
        if self.wandb is not None:
            self.wandb.log(row, step=int(row.get("steps/total", len(self.rows))))
        if len(self.rows) % 20 == 0:
            self.write_csv()

    def write_csv(self):
        import pandas as pd

        pd.DataFrame(self.rows).to_csv(os.path.join(self.run_dir, "progress.csv"), index=False)

    def close(self, summary: dict | None = None):
        self.write_csv()
        if summary is not None:
            with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2, default=str)
        self._jsonl.close()
        if self.wandb is not None:
            if summary:
                self.wandb.summary.update(summary)
            self.wandb.finish()
