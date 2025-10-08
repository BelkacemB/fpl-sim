from __future__ import annotations

import numpy as np

from .pool import PlayerPool


POSITION_PARAMS = {
    0: {"distribution": "poisson", "mean": 3.5},
    1: {"distribution": "negbin", "mean": 4.2, "dispersion": 2.1},
    2: {"distribution": "negbin", "mean": 5.8, "dispersion": 1.8},
    3: {"distribution": "negbin", "mean": 6.1, "dispersion": 1.9},
}


def _poisson_points(rng: np.random.Generator, n: int, mean: float) -> np.ndarray:
    return rng.poisson(mean, n).astype(np.float32)


def _negative_binomial_points(
    rng: np.random.Generator, n: int, mean: float, dispersion: float
) -> np.ndarray:
    n_param = dispersion
    p_param = dispersion / (mean + dispersion)
    return rng.negative_binomial(n_param, p_param, n).astype(np.float32)


def sample_points(
    pool: PlayerPool, rng: np.random.Generator, beta_skill: float = 0.0
) -> np.ndarray:
    points = np.zeros(pool.num_players, dtype=np.float32)
    for position, params in POSITION_PARAMS.items():
        ids = pool.position_to_ids[position]
        if ids.size == 0:
            continue
        base_mean = params["mean"]
        if beta_skill == 0.0:
            scale = 1.0
        else:
            scale = np.exp(beta_skill * pool.skill[ids])
        if params["distribution"] == "poisson":
            if beta_skill == 0.0:
                vals = rng.poisson(base_mean, ids.size).astype(np.float32)
            else:
                lam = base_mean * scale
                vals = rng.poisson(lam).astype(np.float32)
        else:
            r = params["dispersion"]
            if beta_skill == 0.0:
                p = r / (base_mean + r)
                vals = rng.negative_binomial(r, p, ids.size).astype(np.float32)
            else:
                mean_i = base_mean * scale
                p = r / (mean_i + r)
                vals = rng.negative_binomial(r, p).astype(np.float32)
        points[ids] = vals
    return points


def sample_points_poisson(pool: PlayerPool, rng: np.random.Generator) -> np.ndarray:
    return sample_points(pool, rng, beta_skill=0.0)


def score_team(team: np.ndarray, points: np.ndarray) -> float:
    return float(points[team].sum())


