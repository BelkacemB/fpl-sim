from __future__ import annotations

from typing import Dict, List

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


_week_size_to_trend: dict[tuple[int, int], np.ndarray] = {}
_week_size_to_alpha: dict[tuple[int, int], float] = {}

def npc_pick_xi(
    pool: PlayerPool,
    rng: np.random.Generator,
    *,
    week: int,
    alpha: float | None = None,
    trend_scale: float = 1.0,
    tie_noise: float = 1e-6,
    alpha_low: float = 0.3,
    alpha_high: float = 0.9,
) -> np.ndarray:
    key = (int(week), int(pool.num_players))
    if key not in _week_size_to_trend:
        _week_size_to_trend[key] = rng.normal(0.0, trend_scale, size=pool.num_players)
    if key not in _week_size_to_alpha:
        sampled_alpha = float(rng.uniform(alpha_low, alpha_high))
        _week_size_to_alpha[key] = sampled_alpha
    trend = _week_size_to_trend[key]
    effective_alpha = float(_week_size_to_alpha[key] if alpha is None else alpha)
    individual = rng.normal(0.0, 1.0, size=pool.num_players)
    scores = effective_alpha * trend + (1.0 - effective_alpha) * individual
    return pick_team_by_score(scores, pool, FORMATION, rng, tie_noise=tie_noise)

def compute_effective_ownership(field_squads: np.ndarray, num_players: int) -> np.ndarray:
    counts = np.zeros(num_players, dtype=np.int32)
    for squad in field_squads:
        counts[squad] += 1
    eo = counts / float(field_squads.shape[0])
    return eo.astype(np.float32)



