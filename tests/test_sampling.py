"""Tests for sampling module."""

import numpy as np
import pytest

from fpl_eo_sim.sampling import (
    make_rng,
    normal_points,
    student_t_points,
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


def test_normal_points():
    """Test normal points generation."""
    rng = make_rng(42)
    n = 100
    mean = 5.0
    sd = 2.0

    points = normal_points(rng, n, mean, sd)

    assert len(points) == n
    assert isinstance(points, np.ndarray)
    # Check that mean is approximately correct (within 3 standard errors)
    assert abs(np.mean(points) - mean) < 3 * sd / np.sqrt(n)


def test_normal_points_invalid_n():
    """Test normal points with invalid n."""
    rng = make_rng(42)

    with pytest.raises(ValueError, match="Cannot generate negative number of points"):
        normal_points(rng, -1)


def test_student_t_points():
    """Test Student's t points generation."""
    rng = make_rng(42)
    n = 100
    df = 5.0

    points = student_t_points(rng, n, df)

    assert len(points) == n
    assert isinstance(points, np.ndarray)
    # Student's t should have mean approximately 0
    assert abs(np.mean(points)) < 3 / np.sqrt(n)


def test_student_t_points_invalid_inputs():
    """Test Student's t points with invalid inputs."""
    rng = make_rng(42)

    with pytest.raises(ValueError, match="Cannot generate negative number of points"):
        student_t_points(rng, -1)

    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        student_t_points(rng, 10, df=0)

    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        student_t_points(rng, 10, df=-1)
