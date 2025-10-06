from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .constants import FORMATION
from .npc import compute_effective_ownership, npc_pick_xi
from .points import sample_points_poisson, score_team
from .policy import agent_pick_from_eo
from .pool import PlayerPool


class FPLSeasonEOEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_npc: int = 500,
        n_per_pos: Dict[int, int] | None = None,
        horizon: int = 38,
        beta_follow_eo: float = 1.0,
        beta_follow_skill: float = 1.0,
        rng_seed: Optional[int] = 7,
        include_week_in_obs: bool = True,
    ) -> None:
        super().__init__()
        if n_per_pos is None:
            n_per_pos = {0: 4, 1: 20, 2: 20, 3: 12}
        self.rng = np.random.default_rng(rng_seed)
        self.pool = PlayerPool(n_per_pos, self.rng)

        self.num_npc = n_npc
        self.horizon = horizon
        self.beta_eo = beta_follow_eo
        self.beta_skill = beta_follow_skill
        self.include_week = include_week_in_obs

        self.num_players = self.pool.num_players
        obs_dim = self.num_players + (1 if self.include_week else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.week = 0
        self.eo: np.ndarray | None = None
        self.my_total = 0.0
        self.opp_totals: np.ndarray | None = None
        self.last_points: np.ndarray | None = None

        self._reset_initial_eo()

    def _reset_initial_eo(self) -> None:
        logits = self.pool.skill + self.rng.normal(0.0, 0.5, size=self.num_players)
        logits = logits - logits.max()
        e = np.exp(logits)
        prior = e / e.sum()
        self.eo = prior.astype(np.float32)

    def _obs(self) -> np.ndarray:
        assert self.eo is not None
        if self.include_week:
            wk = np.array([self.week / max(1, self.horizon - 1)], dtype=np.float32)
            return np.concatenate([self.eo, wk], axis=0).astype(np.float32)
        return self.eo.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: Dict | None = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_initial_eo()
        self.week = 0
        self.my_total = 0.0
        self.opp_totals = np.zeros(self.num_npc, dtype=np.float32)
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        assert self.eo is not None
        assert self.opp_totals is not None
        info: Dict = {}

        field_squads = []
        for _ in range(self.num_npc):
            squad = npc_pick_xi(
                pool=self.pool,
                ownership_signal=self.eo,
                beta_follow_eo=self.beta_eo,
                beta_follow_skill=self.beta_skill,
                rng=self.rng,
            )
            field_squads.append(squad)
        field_squads = np.stack(field_squads, axis=0)

        eo_gw = compute_effective_ownership(field_squads, self.num_players)

        my_team = agent_pick_from_eo(action, eo_gw, self.pool, self.rng)

        points = sample_points_poisson(self.pool, self.rng)
        self.last_points = points

        gw_scores_field = points[field_squads].sum(axis=1).astype(np.float32)
        self.opp_totals += gw_scores_field

        my_gw = score_team(my_team, points)
        self.my_total += my_gw

        self.week += 1
        terminated = self.week >= self.horizon
        truncated = False

        # weekly percentile-centered reward shaping
        better_gw = (my_gw > gw_scores_field).sum()
        equal_gw = (my_gw == gw_scores_field).sum()
        reward_gw = (better_gw + 0.5 * equal_gw) / max(1, self.num_npc) - 0.5

        if terminated:
            better = (self.my_total > self.opp_totals).sum()
            equal = (self.my_total == self.opp_totals).sum()
            percentile = (better + 0.5 * equal) / max(1, self.num_npc)
            reward = float(reward_gw + (percentile - 0.5))
            obs = self._obs()
        else:
            reward = float(reward_gw)
            self.eo = eo_gw
            obs = self._obs()

        info.update(
            {
                "week": self.week,
                "my_gw": my_gw,
                "my_total": self.my_total,
                "field_mean_gw": float(gw_scores_field.mean()),
                "field_mean_total": float(self.opp_totals.mean()),
                "action": int(action),
            }
        )
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        assert self.eo is not None
        top = np.argsort(-self.eo)[:5]
        print(
            f"W{self.week}/{self.horizon} top EO: {[(int(i), float(self.eo[i])) for i in top]} | my_total={self.my_total:.1f}"
        )


