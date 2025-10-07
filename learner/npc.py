from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .constants import FORMATION
from .pool import PlayerPool


def pick_team_by_score(
    scores: np.ndarray,
    pool: PlayerPool,
    formation: Dict[int, int],
    rng: np.random.Generator,
    tie_noise: float = 1e-6,
) -> np.ndarray:
    team: List[int] = []
    for position, required in formation.items():
        candidates = pool.position_to_ids[position]
        noisy = scores[candidates] + rng.normal(0.0, tie_noise, size=candidates.size)
        chosen = candidates[np.argsort(-noisy)[: required]]
        team.extend(chosen.tolist())
    return np.array(team, dtype=int)


def npc_pick_xi_by_skill(
    pool: PlayerPool,
    rng: np.random.Generator,
    beta_follow_skill: float = 1.0,
) -> np.ndarray:
    z_skill = (pool.skill - pool.skill.mean()) / (pool.skill.std() + 1e-8)
    noise = rng.normal(0.0, 0.5, size=pool.num_players)
    score = beta_follow_skill * z_skill + noise
    return pick_team_by_score(score, pool, FORMATION, rng)

def npc_pick_xi_random(
    pool: PlayerPool,
    rng: np.random.Generator,
) -> np.ndarray:
    return pick_team_by_score(rng.random(pool.num_players), pool, FORMATION, rng)

def compute_effective_ownership(field_squads: np.ndarray, num_players: int) -> np.ndarray:
    counts = np.zeros(num_players, dtype=np.int32)
    for squad in field_squads:
        counts[squad] += 1
    eo = counts / float(field_squads.shape[0])
    return eo.astype(np.float32)


