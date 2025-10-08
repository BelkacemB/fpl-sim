from __future__ import annotations

import numpy as np

from learner.pool import PlayerPool
from learner.npc import npc_pick_xi


def test_npc_team_diversity():
    rng = np.random.default_rng(123)
    pool = PlayerPool({0: 4, 1: 20, 2: 20, 3: 12}, rng)
    ownership_signal = np.zeros(pool.num_players, dtype=np.float32)

    teams = []
    team_sums = []
    for _ in range(100):
        team = npc_pick_xi(
            pool=pool,
            rng=rng,
            week=0,
            alpha=0.9,
            trend_scale=1.0,
        )
        teams.append(tuple(sorted(team.tolist())))
        team_sums.append(int(np.array(team).sum()))

    unique_teams = set(teams)
    sums = np.array(team_sums, dtype=np.float64)
    s_min, s_max, s_std = sums.min(), sums.max(), sums.std()
    print(f"NPC diversity: unique={len(unique_teams)}/100, sum_range=({s_min},{s_max}), std={s_std:.3f}")
    assert s_std > 0.0


