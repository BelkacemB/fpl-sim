"""Simulation engine for FPL EO-Only Simulator.

Contains the main simulation logic and NPC picker implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from fpl_eo_sim.models import Manager, Player, Squad, validate_squad_constraints
from fpl_eo_sim.sampling import generate_points_by_position


class PointsModel(ABC):
    """Abstract base class for points models."""

    @abstractmethod
    def sample_points(
        self, rng: np.random.Generator, players: list[Player]
    ) -> dict[int, float]:
        """Sample points for all players."""
        pass


class PositionBasedPointsModel(PointsModel):
    """Position-based points model using Poisson for GK and Negative Binomial for others."""

    def sample_points(
        self, rng: np.random.Generator, players: list[Player]
    ) -> dict[int, float]:
        """Sample points using position-specific distributions."""
        return generate_points_by_position(rng, players)


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
    """Optimized NPC picker that concentrates ownership around star players."""

    def __init__(self, concentration: float = 0.7):
        """Initialize with concentration parameter.
        
        Args:
            concentration: Concentration level (0.0 = uniform, 1.0 = maximum concentration)
        """
        self.concentration = max(0.0, min(1.0, concentration))

    def pick_xi(
        self, rng: np.random.Generator, players: list[Player], budget: float
    ) -> np.ndarray:
        """Pick XI with concentration around star players under budget constraint.

        Returns array of 11 unique player IDs.
        Uses optimized selection for better performance.
        """
        player_ids = np.array([p.id for p in players])
        prices = np.array([p.price for p in players])
        
        # Quick budget check - if cheapest 11 exceed budget, return them
        sorted_prices = np.sort(prices)
        if sorted_prices[:11].sum() > budget:
            return player_ids[np.argsort(prices)[:11]]
        
        # Create selection weights based on price and concentration
        if self.concentration == 0.0:
            # Uniform selection - use fast random selection
            max_attempts = min(20, len(players))
            for attempt in range(max_attempts):
                selected_indices = rng.choice(len(players), size=11, replace=False)
                selected_ids = player_ids[selected_indices]
                if prices[selected_indices].sum() <= budget:
                    return selected_ids
        else:
            # Weighted selection for concentration
            price_min, price_max = prices.min(), prices.max()
            if price_max > price_min:
                normalized_prices = (prices - price_min) / (price_max - price_min)
                weights = np.power(normalized_prices + 0.1, self.concentration * 3)
            else:
                weights = np.ones(len(players))
            
            weights = weights / weights.sum()
            
            # Try weighted selection with reduced attempts
            max_attempts = min(50, len(players))
            for attempt in range(max_attempts):
                selected_indices = rng.choice(
                    len(players), size=11, replace=False, p=weights
                )
                selected_ids = player_ids[selected_indices]
                if prices[selected_indices].sum() <= budget:
                    return selected_ids

        # Fast greedy fallback - much faster than weighted selection
        selected_indices = []
        remaining_budget = budget
        
        if self.concentration > 0.0:
            # Sort by weight (descending) for concentration
            weight_sorted_indices = np.argsort(weights)[::-1]
            for idx in weight_sorted_indices:
                if len(selected_indices) >= 11:
                    break
                if prices[idx] <= remaining_budget:
                    selected_indices.append(idx)
                    remaining_budget -= prices[idx]
        
        # Fill remaining slots with cheapest players
        if len(selected_indices) < 11:
            remaining_indices = [i for i in range(len(players)) if i not in selected_indices]
            remaining_prices = prices[remaining_indices]
            remaining_sorted = np.argsort(remaining_prices)
            
            for idx in remaining_sorted:
                if len(selected_indices) >= 11:
                    break
                if remaining_prices[idx] <= remaining_budget:
                    selected_indices.append(remaining_indices[idx])
                    remaining_budget -= remaining_prices[idx]

        # Final fallback: cheapest 11 players
        if len(selected_indices) < 11:
            return player_ids[np.argsort(prices)[:11]]

        return player_ids[selected_indices]

class SimulationEngine:
    """Main simulation engine."""

    def simulate_once(
        self,
        rng: np.random.Generator,
        managers: list[Manager],
        players: list[Player],
        npc_picker: RandomNpcPicker,
        my_strategy_fn: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
        points_model: PointsModel | None = None,
        budget: float = 100.0,
    ) -> dict[str, Any]:
        """Run a single simulation.

        Returns dict with eo, my_team, my_score, field_scores, my_rank.
        """
        # Use default position-based points model if none provided
        if points_model is None:
            points_model = PositionBasedPointsModel()
        
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
