"""Sampling utilities for FPL EO-Only Simulator.

Centralized, reproducible randomness and sampling primitives.
"""

from __future__ import annotations

import numpy as np

from fpl_eo_sim.models import Player


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a seeded random number generator.

    Args:
        seed: Random seed. If None, uses system entropy.

    Returns:
        NumPy Generator instance
    """
    return np.random.default_rng(seed)


def weighted_choice_without_replacement(
    rng: np.random.Generator, population: np.ndarray, weights: np.ndarray, k: int
) -> np.ndarray:
    """Sample k items without replacement using weights.

    Args:
        rng: Random number generator
        population: Array of items to sample from
        weights: Weights for each item (must be non-negative)
        k: Number of items to sample

    Returns:
        Array of k sampled items
    """
    if len(population) != len(weights):
        raise ValueError("Population and weights must have same length")

    if k > len(population):
        raise ValueError("Cannot sample more items than available")

    if k < 0:
        raise ValueError("Cannot sample negative number of items")

    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")

    # Normalize weights
    weights = weights / weights.sum()

    # Sample without replacement using cumulative distribution
    selected_indices = []
    remaining_indices = np.arange(len(population))
    remaining_weights = weights.copy()

    for _ in range(k):
        # Normalize remaining weights
        remaining_weights = remaining_weights / remaining_weights.sum()

        # Sample one index
        selected_idx = rng.choice(remaining_indices, p=remaining_weights)
        selected_indices.append(selected_idx)

        # Remove selected index
        mask = remaining_indices != selected_idx
        remaining_indices = remaining_indices[mask]
        remaining_weights = remaining_weights[mask]

    return population[selected_indices]


# Position-specific point generation constants
POSITION_PARAMS = {
    "GK": {"distribution": "poisson", "mean": 3.5},
    "DEF": {"distribution": "negbin", "mean": 4.2, "dispersion": 2.1},
    "MID": {"distribution": "negbin", "mean": 5.8, "dispersion": 1.8},
    "FWD": {"distribution": "negbin", "mean": 6.1, "dispersion": 1.9},
}


def poisson_points(rng: np.random.Generator, n: int, mean: float) -> np.ndarray:
    """Generate n points from Poisson distribution.

    Args:
        rng: Random number generator
        n: Number of points to generate
        mean: Distribution mean (lambda parameter)

    Returns:
        Array of n points
    """
    if n < 0:
        raise ValueError("Cannot generate negative number of points")
    if mean < 0:
        raise ValueError("Mean must be non-negative")

    return rng.poisson(mean, n).astype(float)


def negative_binomial_points(
    rng: np.random.Generator, n: int, mean: float, dispersion: float
) -> np.ndarray:
    """Generate n points from Negative Binomial distribution.

    Args:
        rng: Random number generator
        n: Number of points to generate
        mean: Distribution mean
        dispersion: Dispersion parameter (r in scipy.stats.nbinom)

    Returns:
        Array of n points
    """
    if n < 0:
        raise ValueError("Cannot generate negative number of points")
    if mean < 0:
        raise ValueError("Mean must be non-negative")
    if dispersion <= 0:
        raise ValueError("Dispersion must be positive")

    # Convert mean and dispersion to nbinom parameters
    # For numpy nbinom: mean = n * (1-p) / p, var = n * (1-p) / p^2
    # We have: mean = mu, var = mu + mu^2 / dispersion
    # So: n = dispersion, p = dispersion / (mean + dispersion)
    n_param = dispersion
    p_param = dispersion / (mean + dispersion)
    
    return rng.negative_binomial(n_param, p_param, n).astype(float)


def generate_points_by_position(
    rng: np.random.Generator, players: list[Player]
) -> dict[int, float]:
    """Generate points for players based on their position using appropriate distributions.

    Args:
        rng: Random number generator
        players: List of Player objects

    Returns:
        Dictionary mapping player ID to points
    """
    points = {}
    
    # Group players by position
    position_groups = {}
    for player in players:
        if player.position not in position_groups:
            position_groups[player.position] = []
        position_groups[player.position].append(player)
    
    # Generate points for each position group
    for position, group_players in position_groups.items():
        if position not in POSITION_PARAMS:
            raise ValueError(f"Unknown position: {position}")
        
        params = POSITION_PARAMS[position]
        n = len(group_players)
        
        if params["distribution"] == "poisson":
            group_points = poisson_points(rng, n, params["mean"])
        elif params["distribution"] == "negbin":
            group_points = negative_binomial_points(
                rng, n, params["mean"], params["dispersion"]
            )
        else:
            raise ValueError(f"Unknown distribution: {params['distribution']}")
        
        # Assign points to players
        for i, player in enumerate(group_players):
            points[player.id] = group_points[i]
    
    return points

