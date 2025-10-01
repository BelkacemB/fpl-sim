"""Strategy implementations for FPL EO-Only Simulator.

Contains EO computation and strategy selection functions.
"""

from __future__ import annotations

import numpy as np


def compute_effective_ownership(
    field_squads: np.ndarray, num_players: int
) -> np.ndarray:
    """Compute effective ownership for each player.

    Args:
        field_squads: Array of shape (M, 11) containing player IDs for each manager
        num_players: Total number of players in the gameweek

    Returns:
        Array of shape (num_players,) with EO values in [0, 1]
    """
    eo = np.zeros(num_players)

    for squad in field_squads:
        for player_id in squad:
            if 0 <= player_id < num_players:
                eo[player_id] += 1

    return eo / len(field_squads)


def pick_lowest_eo(
    eo: np.ndarray, K: int = 11, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Pick K players with lowest effective ownership.

    Ties are broken randomly.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get indices sorted by EO (ascending)
    sorted_indices = np.argsort(eo)

    # Handle ties by shuffling within equal EO groups
    unique_eo, inverse_indices = np.unique(eo[sorted_indices], return_inverse=True)
    shuffled_indices = np.empty_like(sorted_indices)

    for i, eo_val in enumerate(unique_eo):
        mask = inverse_indices == i
        group_indices = sorted_indices[mask]
        rng.shuffle(group_indices)
        shuffled_indices[mask] = group_indices

    return shuffled_indices[:K]


def pick_highest_eo(
    eo: np.ndarray, K: int = 11, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Pick K players with highest effective ownership.

    Ties are broken randomly.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get indices sorted by EO (descending)
    sorted_indices = np.argsort(eo)[::-1]

    # Handle ties by shuffling within equal EO groups
    unique_eo, inverse_indices = np.unique(eo[sorted_indices], return_inverse=True)
    shuffled_indices = np.empty_like(sorted_indices)

    for i, eo_val in enumerate(unique_eo):
        mask = inverse_indices == i
        group_indices = sorted_indices[mask]
        rng.shuffle(group_indices)
        shuffled_indices[mask] = group_indices

    return shuffled_indices[:K]


def pick_barbell(
    eo: np.ndarray, K: int = 11, k_safe: int = 5, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Pick a barbell strategy: k_safe highest EO + (K-k_safe) lowest EO players.

    Args:
        eo: Effective ownership array
        K: Total players to pick
        k_safe: Number of safe (high EO) players
        rng: Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()

    if k_safe >= K:
        return pick_highest_eo(eo, K, rng)

    k_risky = K - k_safe

    # Get safe players (highest EO)
    safe_players = pick_highest_eo(eo, k_safe, rng)

    # Get risky players (lowest EO) from remaining players
    remaining_mask = np.ones(len(eo), dtype=bool)
    remaining_mask[safe_players] = False
    remaining_eo = eo[remaining_mask]
    remaining_indices = np.where(remaining_mask)[0]

    risky_selection = pick_lowest_eo(remaining_eo, k_risky, rng)
    risky_players = remaining_indices[risky_selection]

    return np.concatenate([safe_players, risky_players])


def pick_random(
    eo: np.ndarray, K: int = 11, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Pick K players randomly (ignoring EO).

    This strategy picks players uniformly at random, similar to NPC behavior.
    Useful as a baseline comparison against EO-based strategies.

    Args:
        eo: Effective ownership array (ignored)
        K: Total players to pick
        rng: Random number generator

    Returns:
        Array of K randomly selected player indices
    """
    if rng is None:
        rng = np.random.default_rng()

    if K > len(eo):
        raise ValueError("Cannot pick more players than available")

    return rng.choice(len(eo), size=K, replace=False)
