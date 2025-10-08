from __future__ import annotations

import numpy as np
import pytest

from learner.env import FPLSeasonEOEnv


def test_env_reset_and_step_smoke():
    env = FPLSeasonEOEnv(n_npc=10, n_per_pos={0: 2, 1: 6, 2: 6, 3: 4}, horizon=3, rng_seed=0)
    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1

    obs2, _, terminated, truncated, _ = env.step(0)
    assert isinstance(obs2, np.ndarray)
    assert not terminated
    assert not truncated


def test_low_eo_has_higher_rank_variance_than_high_eo():
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 10_000, size=12)

    def run_and_collect_weekly_percentiles(action: int) -> np.ndarray:
        env = FPLSeasonEOEnv(n_npc=400, horizon=38, rng_seed=13)
        per_week = []
        for s in seeds:
            env.reset(seed=int(s))
            weekly = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(action)
                weekly.append(info["weekly_percentile"])  # length up to horizon
            per_week.append(weekly)
        return np.array(per_week, dtype=float)  # shape: [n_seeds, horizon]

    high = run_and_collect_weekly_percentiles(0)  # follow-high-EO
    low = run_and_collect_weekly_percentiles(1)   # follow-low-EO

    # Compute variance across seeds for each week, then average across weeks
    var_high = np.var(high, axis=0, ddof=0).mean()
    var_low = np.var(low, axis=0, ddof=0).mean()

    assert var_low > var_high


