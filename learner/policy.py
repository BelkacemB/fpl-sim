from __future__ import annotations

from typing import List

import numpy as np

from .constants import FORMATION
from .pool import PlayerPool


def agent_pick_from_eo(
    action: int,
    eo: np.ndarray,
    pool: PlayerPool,
    rng: np.random.Generator,
) -> np.ndarray:
    team: List[int] = []
    for position, required in FORMATION.items():
        candidates = pool.position_to_ids[position]
        order = np.argsort(eo[candidates])
        if action == 0:
            order = order[::-1]
        chosen = candidates[order[: required]]
        team.extend(chosen.tolist())
    return np.array(team, dtype=int)


