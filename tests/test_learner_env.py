from __future__ import annotations

import numpy as np
import pytest

from learner.env import FPLSeasonEOEnv


def test_env_reset_and_step_smoke():
    env = FPLSeasonEOEnv(n_npc=10, n_per_pos={0: 2, 1: 6, 2: 6, 3: 4}, horizon=3, rng_seed=0)
    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1

    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert isinstance(obs2, np.ndarray)
    assert reward in (0.0, pytest.approx(0.0))
    assert not terminated
    assert not truncated


