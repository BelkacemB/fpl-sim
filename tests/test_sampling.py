"""Tests for sampling module."""

import numpy as np
import pytest

from fpl_eo_sim.sampling import (
    make_rng,
    poisson_points,
    negative_binomial_points,
    generate_points_by_position,
    weighted_choice_without_replacement,
)


def test_make_rng_with_seed():
    """Test RNG creation with fixed seed."""
    rng1 = make_rng(42)
    rng2 = make_rng(42)

    # Same seed should produce same sequence
    assert rng1.random() == rng2.random()
    assert rng1.random() == rng2.random()


def test_make_rng_without_seed():
    """Test RNG creation without seed."""
    rng1 = make_rng(None)
    rng2 = make_rng(None)

    # Different instances should produce different sequences
    assert rng1.random() != rng2.random()


def test_weighted_choice_without_replacement_basic():
    """Test basic weighted choice without replacement."""
    rng = make_rng(42)
    population = np.array([1, 2, 3, 4, 5])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

    result = weighted_choice_without_replacement(rng, population, weights, k=3)

    assert len(result) == 3
    assert len(set(result)) == 3  # All unique
    assert all(item in population for item in result)


def test_weighted_choice_without_replacement_weights_favor_higher():
    """Test that higher weights are favored over many trials."""
    rng = make_rng(42)
    population = np.array([1, 2, 3, 4, 5])
    weights = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # Item 5 has highest weight

    # Run many trials and count selections
    selections = []
    for _ in range(1000):
        result = weighted_choice_without_replacement(rng, population, weights, k=1)
        selections.extend(result)

    # Item 5 should be selected most often
    selections_array = np.array(selections)
    counts = np.bincount(selections_array - 1)  # Convert to 0-based indexing
    assert counts[4] > counts[0]  # Item 5 > Item 1
    assert counts[4] > counts[1]  # Item 5 > Item 2


def test_weighted_choice_without_replacement_invalid_inputs():
    """Test weighted choice with invalid inputs."""
    rng = make_rng(42)
    population = np.array([1, 2, 3, 4, 5])
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # k > population size
    with pytest.raises(ValueError, match="Cannot sample more items than available"):
        weighted_choice_without_replacement(rng, population, weights, k=10)

    # k < 0
    with pytest.raises(ValueError, match="Cannot sample negative number of items"):
        weighted_choice_without_replacement(rng, population, weights, k=-1)

    # Mismatched lengths
    with pytest.raises(
        ValueError, match="Population and weights must have same length"
    ):
        weighted_choice_without_replacement(rng, population, np.array([0.2, 0.2]), k=2)

    # Negative weights
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        weighted_choice_without_replacement(
            rng, population, np.array([0.2, -0.1, 0.2, 0.2, 0.2]), k=2
        )


def test_poisson_points():
    """Test Poisson points generation."""
    rng = make_rng(42)
    n = 100
    mean = 3.5

    points = poisson_points(rng, n, mean)

    assert len(points) == n
    assert isinstance(points, np.ndarray)
    # Check that mean is approximately correct (within 3 standard errors)
    expected_std = np.sqrt(mean)
    assert abs(np.mean(points) - mean) < 3 * expected_std / np.sqrt(n)
    # All points should be non-negative integers
    assert np.all(points >= 0)
    assert np.all(points == np.floor(points))


def test_poisson_points_invalid_inputs():
    """Test Poisson points with invalid inputs."""
    rng = make_rng(42)

    with pytest.raises(ValueError, match="Cannot generate negative number of points"):
        poisson_points(rng, -1, 3.5)

    with pytest.raises(ValueError, match="Mean must be non-negative"):
        poisson_points(rng, 10, -1.0)


def test_negative_binomial_points():
    """Test Negative Binomial points generation."""
    rng = make_rng(42)
    n = 100
    mean = 5.0
    dispersion = 2.0

    points = negative_binomial_points(rng, n, mean, dispersion)

    assert len(points) == n
    assert isinstance(points, np.ndarray)
    # Check that mean is approximately correct (within 3 standard errors)
    expected_var = mean + mean**2 / dispersion
    expected_std = np.sqrt(expected_var)
    assert abs(np.mean(points) - mean) < 3 * expected_std / np.sqrt(n)
    # All points should be non-negative integers
    assert np.all(points >= 0)
    assert np.all(points == np.floor(points))


def test_negative_binomial_points_invalid_inputs():
    """Test Negative Binomial points with invalid inputs."""
    rng = make_rng(42)

    with pytest.raises(ValueError, match="Cannot generate negative number of points"):
        negative_binomial_points(rng, -1, 5.0, 2.0)

    with pytest.raises(ValueError, match="Mean must be non-negative"):
        negative_binomial_points(rng, 10, -1.0, 2.0)

    with pytest.raises(ValueError, match="Dispersion must be positive"):
        negative_binomial_points(rng, 10, 5.0, 0.0)

    with pytest.raises(ValueError, match="Dispersion must be positive"):
        negative_binomial_points(rng, 10, 5.0, -1.0)


def test_generate_points_by_position():
    """Test position-based point generation."""
    from fpl_eo_sim.models import Player
    
    rng = make_rng(42)
    players = [
        Player(id=0, name="GK1", price=5.0, position="GK", team="A"),
        Player(id=1, name="GK2", price=5.5, position="GK", team="B"),
        Player(id=2, name="DEF1", price=6.0, position="DEF", team="A"),
        Player(id=3, name="MID1", price=8.0, position="MID", team="A"),
        Player(id=4, name="FWD1", price=10.0, position="FWD", team="A"),
    ]

    points = generate_points_by_position(rng, players)

    assert len(points) == 5
    assert all(pid in points for pid in [0, 1, 2, 3, 4])
    assert all(isinstance(p, float) for p in points.values())
    assert all(p >= 0 for p in points.values())


def test_generate_points_by_position_unknown_position():
    """Test position-based point generation with unknown position."""
    from fpl_eo_sim.models import Player
    
    rng = make_rng(42)
    # Create a player with invalid position by bypassing validation
    player = Player.__new__(Player)
    player.id = 0
    player.name = "Test"
    player.price = 5.0
    player.position = "INVALID"  # type: ignore
    player.team = "A"

    with pytest.raises(ValueError, match="Unknown position: INVALID"):
        generate_points_by_position(rng, [player])
