"""Sampling utilities for FPL EO-Only Simulator.

Centralized, reproducible randomness and sampling primitives.
"""

from __future__ import annotations

import numpy as np


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


def normal_points(
    rng: np.random.Generator, n: int, mean: float = 0.0, sd: float = 1.0
) -> np.ndarray:
    """Generate n points from normal distribution.

    Args:
        rng: Random number generator
        n: Number of points to generate
        mean: Distribution mean
        sd: Distribution standard deviation

    Returns:
        Array of n points
    """
    if n < 0:
        raise ValueError("Cannot generate negative number of points")

    return rng.normal(mean, sd, n)


def student_t_points(rng: np.random.Generator, n: int, df: float = 5.0) -> np.ndarray:
    """Generate n points from Student's t-distribution.

    Args:
        rng: Random number generator
        n: Number of points to generate
        df: Degrees of freedom

    Returns:
        Array of n points
    """
    if n < 0:
        raise ValueError("Cannot generate negative number of points")

    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    return rng.standard_t(df, n)


def fpl_like_points(
    rng: np.random.Generator,
    n: int,
    base_probs: np.ndarray | None = None,
    base_max: int = 6,
    haul_prob: float = 0.15,  # Increased from 0.05 to 0.15 for more hauls
    haul_min: int = 8,
    haul_cap: int = 30,  # Increased from 20 to 30 for fatter tail
) -> np.ndarray:
    """Sample FPL-like points with common 0–6 scores and rare 8+ hauls.

    The distribution is a simple mixture:
    - With probability (1 - haul_prob): draw from a categorical over {0..base_max}
      skewed towards higher values with mode at 3 for realistic totals.
    - With probability haul_prob: draw a haul score in [haul_min, haul_cap]
      using a capped geometric tail with fatter distribution.

    Args:
        rng: Random number generator
        n: Number of samples to draw
        base_probs: Optional probabilities over {0..base_max}. If None, uses a
            default vector shaped for mode at 3 and higher variance.
        base_max: Maximum non-haul score (inclusive), default 6
        haul_prob: Probability of a haul event, default 0.12
        haul_min: Minimum haul points, default 8
        haul_cap: Maximum haul points (cap), default 30

    Returns:
        Array of shape (n,) with integer-like floats representing points.
    """
    if n < 0:
        raise ValueError("Cannot generate negative number of points")
    if base_max < 0:
        raise ValueError("base_max must be non-negative")
    if not (0.0 <= haul_prob <= 1.0):
        raise ValueError("haul_prob must be in [0, 1]")
    if haul_min <= base_max:
        # Hauls should exceed the common range
        raise ValueError("haul_min must be greater than base_max")
    if haul_cap < haul_min:
        raise ValueError("haul_cap must be >= haul_min")

    support = np.arange(base_max + 1)
    if base_probs is None:
        # Heuristic discrete profile for {0..6}: mode at 2, thin right tail
        # If base_max != 6, construct a decaying distribution centered near 2
        if base_max == 6:
            # Shifted distribution with higher mean for 40-60 point totals
            base_probs = np.array(
                [0.05, 0.08, 0.15, 0.25, 0.22, 0.15, 0.10], dtype=float
            )
        else:
            # Create a discrete Laplace-like shape peaked around 4 for higher totals
            center = 4.0
            decay = 0.5
            weights = np.power(decay, np.abs(support - center))
            base_probs = weights.astype(float)
        base_probs = base_probs / float(base_probs.sum())
    else:
        if len(base_probs) != len(support):
            raise ValueError("base_probs length must be base_max + 1")
        if np.any(base_probs < 0):
            raise ValueError("base_probs must be non-negative")
        total = float(np.sum(base_probs))
        if total <= 0.0:
            raise ValueError("base_probs must sum to a positive value")
        base_probs = base_probs / total

    base_draws = rng.choice(support, size=n, p=base_probs)

    if haul_prob == 0.0:
        return base_draws.astype(float)

    haul_mask = rng.random(n) < haul_prob
    num_hauls = int(haul_mask.sum())
    if num_hauls == 0:
        return base_draws.astype(float)

    # Geometric tail (p=0.3) shifted by haul_min and capped at haul_cap
    # Lower p creates fatter tail for more variance
    # Values: haul_min + Geometric(p) - 1, then clip to cap
    tail = haul_min + rng.geometric(p=0.3, size=num_hauls) - 1
    tail = np.clip(tail, haul_min, haul_cap)

    result = base_draws.astype(int)
    result[haul_mask] = tail
    return result.astype(float)
