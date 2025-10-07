from __future__ import annotations

from typing import List

import numpy as np

from learner.constants import FORMATION
from learner.pool import PlayerPool


def agent_pick_from_eo(
    action: int,
    eo: np.ndarray,
    pool: PlayerPool,
    rng: np.random.Generator,
) -> np.ndarray:
    team: List[int] = []
    for position, required in FORMATION.items():
        candidates = pool.position_to_ids[position]
        if action == 2:
            perm = rng.permutation(candidates.shape[0])
            chosen = candidates[perm[: required]]
        else:
            order = np.argsort(eo[candidates])
            if action == 0:
                order = order[::-1]
            chosen = candidates[order[: required]]
        team.extend(chosen.tolist())
    return np.array(team, dtype=int)



def run_season_with_fixed_action(action: int, *, seed: int | None = 7, horizon: int = 38) -> dict:
    from learner.env import FPLSeasonEOEnv
    env = FPLSeasonEOEnv(rng_seed=seed, horizon=horizon)
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    last_info: dict = {}
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
    return {
        "action": action,
        "my_total": float(last_info.get("my_total", 0.0)),
        "percentile": float(last_info.get("percentile", 0.5)),
    }


