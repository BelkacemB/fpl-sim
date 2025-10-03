"""Simulation engine for FPL EO-Only Simulator.

Contains the main simulation logic and NPC picker implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from fpl_eo_sim.models import (
    Manager,
    Player,
    Squad,
    validate_squad_constraints,
)
from fpl_eo_sim.models import DEFAULT_FORMATION
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
        """Pick XI with required formation under budget constraint."""
        max_attempts = 100
        prices = np.array([p.price for p in players])

        # Build index lists per position
        pos_to_indices = {
            "GK": np.array([i for i, p in enumerate(players) if p.position == "GK"]),
            "DEF": np.array([i for i, p in enumerate(players) if p.position == "DEF"]),
            "MID": np.array([i for i, p in enumerate(players) if p.position == "MID"]),
            "FWD": np.array([i for i, p in enumerate(players) if p.position == "FWD"]),
        }

        # Try random-by-position selections
        for _ in range(max_attempts):
            chosen_indices: list[int] = []
            for pos, count in DEFAULT_FORMATION.items():
                pool = pos_to_indices[pos]
                if len(pool) < count:
                    # Not enough players in this position; try another attempt
                    chosen_indices = []
                    break
                chosen = rng.choice(pool, size=count, replace=False)
                chosen_indices.extend(list(chosen))
            if len(chosen_indices) != 11:
                # Try again
                continue
            total_price = prices[np.array(chosen_indices)].sum()
            if total_price <= budget:
                return np.array([players[i].id for i in chosen_indices])

        # Greedy cheapest-by-position fallback
        chosen_indices = []
        for pos, count in DEFAULT_FORMATION.items():
            pool = pos_to_indices[pos]
            if len(pool) < count:
                continue
            pool_sorted = pool[np.argsort(prices[pool])]
            # Add randomness: pick required from top_k cheapest within the position
            top_k = min(len(pool_sorted), max(count * 3, count))
            candidates = pool_sorted[:top_k]
            chosen = rng.choice(candidates, size=count, replace=False)
            chosen_indices.extend(list(chosen))
        if len(chosen_indices) == 11:
            # If over budget, try to swap expensive with cheaper within same pos
            if prices[np.array(chosen_indices)].sum() <= budget:
                return np.array([players[i].id for i in chosen_indices])

        # Final fallback: return any valid first formation ignoring budget
        chosen_indices = []
        for pos, count in DEFAULT_FORMATION.items():
            pool = pos_to_indices[pos]
            chosen_indices.extend(list(pool[:count]))
        return np.array([players[i].id for i in chosen_indices])


class ConcentratedNpcPicker:
    """Optimized NPC picker that concentrates ownership around star players.

    Concentration is randomized per run between 0.0 and 1.0.
    """

    def pick_xi(
        self, rng: np.random.Generator, players: list[Player], budget: float
    ) -> np.ndarray:
        """Pick XI with concentration around star players under budget and formation."""
        concentration = float(rng.random())
        player_ids = np.array([p.id for p in players])
        prices = np.array([p.price for p in players])
        pos_to_indices = {
            "GK": np.array([i for i, p in enumerate(players) if p.position == "GK"]),
            "DEF": np.array([i for i, p in enumerate(players) if p.position == "DEF"]),
            "MID": np.array([i for i, p in enumerate(players) if p.position == "MID"]),
            "FWD": np.array([i for i, p in enumerate(players) if p.position == "FWD"]),
        }
        
        # Quick budget check - if cheapest 11 exceed budget, return them
        sorted_prices = np.sort(prices)
        if sorted_prices[:11].sum() > budget:
            return player_ids[np.argsort(prices)[:11]]
        
        # Create selection weights based on price and concentration
        # Build global weights for concentration
        price_min, price_max = prices.min(), prices.max()
        if price_max > price_min:
            normalized_prices = (prices - price_min) / (price_max - price_min)
            weights = np.power(normalized_prices + 0.1, max(0.0, concentration) * 3)
        else:
            weights = np.ones(len(players))
        weights = weights / weights.sum()

        # Try weighted selection within each position (with noise to diversify)
        max_attempts = min(50, len(players))
        for _ in range(max_attempts):
            chosen_indices: list[int] = []
            for pos, count in DEFAULT_FORMATION.items():
                pool = pos_to_indices[pos]
                if len(pool) < count:
                    break
                pos_weights = weights[pool]
                # Inject small randomness to avoid identical picks
                jitter = 0.05
                pos_weights = pos_weights + jitter * rng.random(len(pos_weights))
                pos_weights = pos_weights / pos_weights.sum()
                chosen = rng.choice(pool, size=count, replace=False, p=pos_weights)
                chosen_indices.extend(list(chosen))
            if len(chosen_indices) != 11:
                break
            if prices[np.array(chosen_indices)].sum() <= budget:
                return player_ids[np.array(chosen_indices)]

        # Fast greedy fallback - much faster than weighted selection
        selected_indices = []
        remaining_budget = budget
        
        if concentration > 0.0:
            # Sort by weight (descending) for concentration
            weight_sorted_indices = np.argsort(weights)[::-1]
            for idx in weight_sorted_indices:
                if len(selected_indices) >= 11:
                    break
                if prices[idx] <= remaining_budget:
                    selected_indices.append(idx)
                    remaining_budget -= prices[idx]
        
        # Fill by cheapest within remaining formation slots, randomizing among top options
        if len(selected_indices) < 11:
            counts = {k: 0 for k in DEFAULT_FORMATION.keys()}
            for idx in selected_indices:
                counts[players[idx].position] += 1
            for pos, required in DEFAULT_FORMATION.items():
                if counts[pos] >= required:
                    continue
                pool = [i for i in range(len(players)) if players[i].position == pos and i not in selected_indices]
                pool = np.array(pool)
                pool_sorted = pool[np.argsort(prices[pool])]
                # Choose randomly among top_k cheapest for diversity
                top_k = min(len(pool_sorted), max((required - counts[pos]) * 3, 1))
                for idx in rng.choice(pool_sorted[:top_k], size=len(pool_sorted[:top_k]), replace=False):
                    if counts[pos] >= required:
                        break
                    if prices[idx] <= remaining_budget:
                        selected_indices.append(idx)
                        counts[pos] += 1
                        remaining_budget -= prices[idx]

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
        points_model: PointsModel | None = None
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

        # Step 3: "My" team selection with formation
        from .strategies import select_team_with_formation
        my_team = select_team_with_formation(eo, players, my_strategy_fn, DEFAULT_FORMATION, rng)

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
