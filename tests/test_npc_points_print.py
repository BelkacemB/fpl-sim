from __future__ import annotations

import numpy as np

from learner.pool import PlayerPool
from learner.npc import npc_pick_xi
from learner.points import sample_points_poisson, score_team


def test_npc_points_print_only():
    rng = np.random.default_rng(32)
    pool = PlayerPool({0: 4, 1: 20, 2: 20, 3: 12}, rng)
    ownership_signal = rng.random(pool.num_players).astype(np.float32)

    team = npc_pick_xi(
        pool=pool,
        ownership_signal=ownership_signal,
        beta_follow_eo=1.0,
        beta_follow_skill=1.0,
        rng=rng,
    )
    points = sample_points_poisson(pool, rng)
    team_points = score_team(team, points)
    print(f"NPC team points (print-only): {team_points:.2f}")


