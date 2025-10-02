"""Simulation engine for FPL EO-Only Simulator.

Contains the main simulation logic and NPC picker implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from fpl_eo_sim.models import Manager, Player, Squad, validate_squad_constraints
from fpl_eo_sim.sampling import fpl_like_points


class PointsModel(ABC):
    """Abstract base class for points models."""

    @abstractmethod
    def sample_points(
        self, rng: np.random.Generator, players: list[Player]
    ) -> dict[int, float]:
        """Sample points for all players."""
        pass


class NormalPointsModel(PointsModel):
    """Normal distribution points model."""

    def __init__(self, mean: float = 0.0, sd: float = 1.0) -> None:
        self.mean = mean
        self.sd = sd

    def sample_points(
        self, rng: np.random.Generator, players: list[Player]
    ) -> dict[int, float]:
        """Sample points from normal distribution."""
        points = rng.normal(self.mean, self.sd, len(players))
        return {player.id: float(points[i]) for i, player in enumerate(players)}


class StudentTPointsModel(PointsModel):
    """Student's t-distribution points model."""

    def __init__(self, df: float = 5.0) -> None:
        self.df = df

    def sample_points(
        self, rng: np.random.Generator, players: list[Player]
    ) -> dict[int, float]:
        """Sample points from Student's t-distribution."""
        points = rng.standard_t(self.df, len(players))
        return {player.id: float(points[i]) for i, player in enumerate(players)}


class FplLikePointsModel(PointsModel):
    """FPL-like points model with realistic scoring distribution."""

    def __init__(
        self,
        base_probs: np.ndarray | None = None,
        base_max: int = 6,
        haul_prob: float = 0.05,
        haul_min: int = 8,
        haul_cap: int = 20,
    ) -> None:
        self.base_probs = base_probs
        self.base_max = base_max
        self.haul_prob = haul_prob
        self.haul_min = haul_min
        self.haul_cap = haul_cap

    def sample_points(
        self, rng: np.random.Generator, players: list[Player]
    ) -> dict[int, float]:
        """Sample points using FPL-like distribution."""
        points = fpl_like_points(
            rng,
            len(players),
            base_probs=self.base_probs,
            base_max=self.base_max,
            haul_prob=self.haul_prob,
            haul_min=self.haul_min,
            haul_cap=self.haul_cap,
        )
        return {player.id: float(points[i]) for i, player in enumerate(players)}


class RandomNpcPicker:
    """Random NPC picker that selects XI uniformly at random under budget."""

    def pick_xi(
        self, rng: np.random.Generator, players: list[Player], budget: float
    ) -> np.ndarray:
        """Pick XI randomly under budget constraint.

        Returns array of 11 unique player IDs.
        Retries with capped loop; if not feasible, removes most expensive player.
        """
        max_attempts = 1000
        player_ids = np.array([p.id for p in players])
        prices = np.array([p.price for p in players])

        for attempt in range(max_attempts):
            # Random selection without replacement
            selected_indices = rng.choice(len(players), size=11, replace=False)
            selected_ids = player_ids[selected_indices]
            total_price = prices[selected_indices].sum()

            if total_price <= budget:
                return selected_ids

        # If budget constraint too tight, remove most expensive players
        sorted_indices = np.argsort(prices)[::-1]  # Most expensive first
        for i in range(len(players) - 11):
            remaining_indices = sorted_indices[i:]
            if len(remaining_indices) >= 11:
                selected_indices = rng.choice(remaining_indices, size=11, replace=False)
                selected_ids = player_ids[selected_indices]
                total_price = prices[selected_indices].sum()
                if total_price <= budget:
                    return selected_ids

        # Fallback: just return first 11 players
        return player_ids[:11]


class ConcentratedNpcPicker:
    """NPC picker that concentrates ownership around star players (high-priced)."""

    def __init__(self, concentration: float = 0.7):
        """Initialize with concentration parameter.
        
        Args:
            concentration: Concentration level (0.0 = uniform, 1.0 = maximum concentration)
                          Higher values mean more focus on expensive players
        """
        self.concentration = max(0.0, min(1.0, concentration))

    def pick_xi(
        self, rng: np.random.Generator, players: list[Player], budget: float
    ) -> np.ndarray:
        """Pick XI with concentration around star players under budget constraint.

        Returns array of 11 unique player IDs.
        Uses weighted selection favoring expensive players based on concentration parameter.
        """
        max_attempts = 1000
        player_ids = np.array([p.id for p in players])
        prices = np.array([p.price for p in players])
        
        # Create selection weights based on price and concentration
        # Higher concentration = more weight on expensive players
        if self.concentration == 0.0:
            # Uniform selection
            weights = np.ones(len(players))
        else:
            # Weight by price raised to concentration power
            # This creates concentration around expensive players
            normalized_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
            weights = np.power(normalized_prices + 0.1, self.concentration * 3)
        
        weights = weights / weights.sum()

        for attempt in range(max_attempts):
            # Weighted selection without replacement
            selected_indices = rng.choice(
                len(players), size=11, replace=False, p=weights
            )
            selected_ids = player_ids[selected_indices]
            total_price = prices[selected_indices].sum()

            if total_price <= budget:
                return selected_ids

        # If budget constraint too tight, fall back to cheaper players
        # but still maintain some concentration
        sorted_indices = np.argsort(prices)
        for i in range(len(players) - 11):
            remaining_indices = sorted_indices[i:]
            if len(remaining_indices) >= 11:
                remaining_weights = weights[remaining_indices]
                remaining_weights = remaining_weights / remaining_weights.sum()
                
                selected_indices = rng.choice(
                    remaining_indices, size=11, replace=False, p=remaining_weights
                )
                selected_ids = player_ids[selected_indices]
                total_price = prices[selected_indices].sum()
                if total_price <= budget:
                    return selected_ids

        # Fallback: return cheapest 11 players
        cheapest_indices = np.argsort(prices)[:11]
        return player_ids[cheapest_indices]


class SimulationEngine:
    """Main simulation engine."""

    def simulate_once(
        self,
        rng: np.random.Generator,
        managers: list[Manager],
        players: list[Player],
        npc_picker: RandomNpcPicker,
        my_strategy_fn: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
        points_model: PointsModel,
        budget: float,
    ) -> dict[str, Any]:
        """Run a single simulation.

        Returns dict with eo, my_team, my_score, field_scores, my_rank.
        """
        # Step 1: NPCs pick XI for each manager
        field_squads = []
        for manager in managers:
            squad_ids = npc_picker.pick_xi(rng, players, manager.budget)
            field_squads.append(squad_ids)

        field_squads = np.array(field_squads)

        # Step 2: Compute EO
        from .strategies import compute_effective_ownership

        eo = compute_effective_ownership(field_squads, len(players))

        # Step 3: "My" team selection
        my_team = my_strategy_fn(eo, 11, rng)

        # Step 4: Sample points
        points = points_model.sample_points(rng, players)

        # Step 5: Calculate scores and rank
        field_scores = []
        for squad_ids in field_squads:
            score = sum(points[pid] for pid in squad_ids)
            field_scores.append(score)

        my_score = sum(points[pid] for pid in my_team)

        # Competition ranking: 1 + #strictly_greater + 0.5*#ties
        strictly_greater = sum(1 for score in field_scores if score > my_score)
        ties = sum(1 for score in field_scores if score == my_score)
        my_rank = 1 + strictly_greater + 0.5 * ties

        return {
            "eo": eo,
            "my_team": my_team,
            "my_score": my_score,
            "field_scores": field_scores,
            "my_rank": my_rank,
        }
