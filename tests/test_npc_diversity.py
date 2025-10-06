from __future__ import annotations

import numpy as np

from learner.pool import PlayerPool
from learner.npc import npc_pick_xi


def test_npc_team_diversity():
    rng = np.random.default_rng(123)
    pool = PlayerPool({0: 4, 1: 20, 2: 20, 3: 12}, rng)
    ownership_signal = rng.random(pool.num_players).astype(np.float32)

    teams = []
    for _ in range(10):
        team = npc_pick_xi(
            pool=pool,
            ownership_signal=ownership_signal,
            beta_follow_eo=1.0,
            beta_follow_skill=1.0,
            rng=rng,
        )
        teams.append(tuple(sorted(team.tolist())))

    unique_teams = set(teams)
    assert len(unique_teams) >= 5


