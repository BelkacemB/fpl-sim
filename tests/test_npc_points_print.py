from __future__ import annotations

import numpy as np

from learner.pool import PlayerPool
from learner.npc import npc_pick_xi
from learner.points import sample_points_poisson, score_team


def test_npc_points_print_only():
    rng = np.random.default_rng(2)
    pool = PlayerPool({0: 4, 1: 20, 2: 20, 3: 12}, rng)

    team = npc_pick_xi(
        pool=pool,
        rng=rng,
        week=0,
        alpha=0.9,
        trend_scale=1.0,
    )
    points = sample_points_poisson(pool, rng)
    team_points = score_team(team, points)
    print(f"NPC team points (print-only): {team_points:.2f}")


