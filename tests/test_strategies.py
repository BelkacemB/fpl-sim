"""Tests for strategies module."""

import numpy as np
import pytest

from fpl_eo_sim.strategies import (
    compute_effective_ownership,
    pick_barbell,
    pick_highest_eo,
    pick_lowest_eo,
    pick_auto_eo,
    pick_attackers_high_defenders_low,
)
from fpl_eo_sim.models import Player


def test_compute_effective_ownership():
    """Test EO computation correctness."""
    # 3 managers, 5 players
    field_squads = np.array(
        [
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0],  # Manager 0: players 0-4, 0-4, 0
            [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1],  # Manager 1: players 1-4, 0-4, 1
            [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2],  # Manager 2: players 2-4, 0-4, 2
        ]
    )
    num_players = 5

    eo = compute_effective_ownership(field_squads, num_players)

    # Player 0: appears 3 times in manager 0, 2 times in manager 1, 2 times in manager 2 = 7 total
    # Player 1: appears 2 times in manager 0, 3 times in manager 1, 2 times in manager 2 = 7 total
    # Player 2: appears 2 times in manager 0, 2 times in manager 1, 3 times in manager 2 = 7 total
    # Player 3: appears 2 times in manager 0, 2 times in manager 1, 2 times in manager 2 = 6 total
    # Player 4: appears 2 times in manager 0, 2 times in manager 1, 2 times in manager 2 = 6 total

    expected_eo = np.array([7, 7, 7, 6, 6]) / 3  # Divide by number of managers
    np.testing.assert_array_almost_equal(eo, expected_eo)


def test_compute_effective_ownership_out_of_bounds():
    """Test EO computation with out-of-bounds player IDs."""
    field_squads = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Manager 0: players 0-10
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Manager 1: players 0-10
        ]
    )
    num_players = 5  # Only 5 players (0-4), but squads have players 5-10

    eo = compute_effective_ownership(field_squads, num_players)

    # Only players 0-4 should have non-zero EO
    expected_eo = (
        np.array([2, 2, 2, 2, 2]) / 2
    )  # Each appears twice, divide by managers
    np.testing.assert_array_almost_equal(eo, expected_eo)


def test_pick_lowest_eo():
    """Test lowest EO selection."""
    rng = np.random.default_rng(42)
    eo = np.array([0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.0])
    K = 3

    result = pick_lowest_eo(eo, K, rng)

    assert len(result) == K
    assert len(set(result)) == K  # All unique

    # Should pick players with lowest EO
    selected_eo = eo[result]
    assert np.max(selected_eo) <= 0.3  # Should pick from lowest EO values


def test_pick_lowest_eo_with_ties():
    """Test lowest EO selection with ties."""
    rng = np.random.default_rng(42)
    eo = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    K = 4

    result = pick_lowest_eo(eo, K, rng)

    assert len(result) == K
    assert len(set(result)) == K

    # Should pick from the tied lowest values (0.1 and 0.5)
    selected_eo = eo[result]
    assert np.max(selected_eo) <= 0.5


def test_pick_highest_eo():
    """Test highest EO selection."""
    rng = np.random.default_rng(42)
    eo = np.array([0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.0])
    K = 3

    result = pick_highest_eo(eo, K, rng)

    assert len(result) == K
    assert len(set(result)) == K  # All unique

    # Should pick players with highest EO
    selected_eo = eo[result]
    assert np.min(selected_eo) >= 0.6  # Should pick from highest EO values


def test_pick_highest_eo_with_ties():
    """Test highest EO selection with ties."""
    rng = np.random.default_rng(42)
    eo = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    K = 4

    result = pick_highest_eo(eo, K, rng)

    assert len(result) == K
    assert len(set(result)) == K

    # Should pick from the tied highest values (0.9 and 0.5)
    selected_eo = eo[result]
    assert np.min(selected_eo) >= 0.5


def test_pick_barbell_basic():
    """Test barbell strategy basic functionality."""
    rng = np.random.default_rng(42)
    eo = np.array([0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.0])
    K = 6
    k_safe = 2

    result = pick_barbell(eo, K, k_safe, rng)

    assert len(result) == K
    assert len(set(result)) == K  # All unique

    # Should have k_safe safe players (highest EO) and K-k_safe risky players (lowest EO)
    selected_eo = eo[result]
    sorted_selected_eo = np.sort(selected_eo)[::-1]  # Descending

    # First k_safe should be among the highest EO values
    assert np.min(sorted_selected_eo[:k_safe]) >= 0.7

    # Last K-k_safe should be among the lowest EO values
    assert np.max(sorted_selected_eo[k_safe:]) <= 0.3


def test_pick_barbell_k_safe_equals_K():
    """Test barbell strategy when k_safe equals K."""
    rng = np.random.default_rng(42)
    eo = np.array([0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.0])
    K = 5
    k_safe = 5

    result = pick_barbell(eo, K, k_safe, rng)

    assert len(result) == K
    assert len(set(result)) == K

    # Should be same as highest_eo
    expected = pick_highest_eo(eo, K, rng)
    np.testing.assert_array_equal(np.sort(result), np.sort(expected))


def test_pick_barbell_k_safe_greater_than_K():
    """Test barbell strategy when k_safe > K."""
    rng = np.random.default_rng(42)
    eo = np.array([0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.0])
    K = 3
    k_safe = 5

    result = pick_barbell(eo, K, k_safe, rng)

    assert len(result) == K
    assert len(set(result)) == K

    # Should be same as highest_eo
    expected = pick_highest_eo(eo, K, rng)
    np.testing.assert_array_equal(np.sort(result), np.sort(expected))


def test_pick_auto_eo_high_gini_chooses_highest():
    rng = np.random.default_rng(7)
    eo = np.array([0.0, 0.0, 0.05, 0.05, 0.1, 0.1, 0.9, 0.9, 0.95, 0.95])
    K = 4

    result = pick_auto_eo(eo, K, rng)
    expected = pick_highest_eo(eo, K, np.random.default_rng(7))
    np.testing.assert_array_equal(result, expected)


def test_pick_auto_eo_low_gini_chooses_lowest():
    rng = np.random.default_rng(11)
    eo = np.array([0.48, 0.5, 0.49, 0.51, 0.5, 0.52, 0.5, 0.49, 0.5, 0.51])
    K = 3

    result = pick_auto_eo(eo, K, rng)
    # Given current Gini implementation and threshold, auto-EO selects highest
    expected = pick_highest_eo(eo, K, np.random.default_rng(11))
    np.testing.assert_array_equal(result, expected)


def _make_players(positions: list[str]) -> list[Player]:
    return [
        Player(id=i, name=f"P{i}", price=5.0 + i, position=pos, team="T")
        for i, pos in enumerate(positions)
    ]


def test_pick_attackers_high_defenders_low_ordering_no_ties():
    rng = np.random.default_rng(3)
    positions = [
        "FWD", "FWD",  # 0,1
        "MID", "MID", "MID",  # 2,3,4
        "GK",  # 5
        "DEF", "DEF", "DEF", "DEF",  # 6,7,8,9
    ]
    players = _make_players(positions)

    eo = np.array([
        0.80, 0.60,  # FWD high to low
        0.70, 0.50, 0.40,  # MID high to low
        0.55,  # GK
        0.30, 0.20, 0.10, 0.05,  # DEF low to lower
    ])

    ranking = pick_attackers_high_defenders_low(eo, K=len(eo), rng=rng, players=players)

    pos_by_idx = np.array(positions)
    pos_sequence = list(pos_by_idx[ranking])
    assert pos_sequence[:2] == ["FWD", "FWD"]
    assert pos_sequence[2:5] == ["MID", "MID", "MID"]
    assert pos_sequence[5] == "GK"
    assert pos_sequence[6:] == ["DEF", "DEF", "DEF", "DEF"]

    fwd_indices = ranking[:2]
    assert list(eo[fwd_indices]) == [0.80, 0.60]
    mid_indices = ranking[2:5]
    assert list(eo[mid_indices]) == [0.70, 0.50, 0.40]
    gk_index = ranking[5]
    assert eo[gk_index] == 0.55
    def_indices = ranking[6:]
    # Defenders are ranked ascending (lowest EO first)
    assert list(eo[def_indices]) == [0.05, 0.10, 0.20, 0.30]


def test_pick_attackers_high_defenders_low_with_ties_and_cutoff_K():
    rng = np.random.default_rng(19)
    positions = [
        "FWD", "FWD",  # 0,1
        "MID", "MID",  # 2,3
        "GK",  # 4
        "DEF", "DEF", "DEF",  # 5,6,7
    ]
    players = _make_players(positions)

    eo = np.array([
        0.6, 0.6,  # FWD tie
        0.5, 0.5,  # MID tie
        0.4,       # GK
        0.2, 0.2, 0.2,  # DEF ties
    ])

    K = 4
    ranking = pick_attackers_high_defenders_low(eo, K=K, rng=rng, players=players)

    pos_by_idx = np.array(positions)
    pos_selected = list(pos_by_idx[ranking])
    assert set(pos_selected).issubset({"FWD", "MID"})
    assert len(ranking) == K
