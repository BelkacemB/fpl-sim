"""Tests for simulator module."""

import numpy as np

from fpl_eo_sim.models import Manager, Player
from fpl_eo_sim.simulator import (
    PositionBasedPointsModel,
    RandomNpcPicker,
    SimulationEngine,
)
from fpl_eo_sim.strategies import pick_lowest_eo


def test_position_based_points_model():
    """Test PositionBasedPointsModel."""
    rng = np.random.default_rng(42)
    players = [
        Player(id=0, name="GK1", price=5.0, position="GK", team="A"),
        Player(id=1, name="DEF1", price=6.0, position="DEF", team="A"),
        Player(id=2, name="MID1", price=8.0, position="MID", team="A"),
        Player(id=3, name="FWD1", price=10.0, position="FWD", team="A"),
    ]

    model = PositionBasedPointsModel()
    points = model.sample_points(rng, players)

    assert len(points) == 4
    assert all(pid in points for pid in [0, 1, 2, 3])
    assert all(isinstance(p, float) for p in points.values())
    assert all(p >= 0 for p in points.values())


def test_random_npc_picker_basic():
    """Test RandomNpcPicker basic functionality."""
    rng = np.random.default_rng(42)
    # Ensure enough players per position to satisfy 1-3-4-3 formation
    positions = (
        ["GK"] * 3
        + ["DEF"] * 6
        + ["MID"] * 6
        + ["FWD"] * 5
    )
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position=positions[i], team="A")
        for i in range(20)
    ]
    budget = 100.0

    picker = RandomNpcPicker()
    squad_ids = picker.pick_xi(rng, players, budget)

    assert len(squad_ids) == 11
    assert len(set(squad_ids)) == 11  # All unique
    assert all(0 <= pid < 20 for pid in squad_ids)  # Valid player IDs


def test_random_npc_picker_budget_constraint():
    """Test RandomNpcPicker respects budget constraint."""
    rng = np.random.default_rng(42)
    positions = (
        ["GK"] * 3
        + ["DEF"] * 6
        + ["MID"] * 6
        + ["FWD"] * 5
    )
    players = [
        Player(id=i, name=f"P{i}", price=5.0, position=positions[i], team="A")
        for i in range(20)
    ]
    budget = 55.0  # 11 * 5.0 = 55.0, should be exactly on budget

    picker = RandomNpcPicker()
    squad_ids = picker.pick_xi(rng, players, budget)

    assert len(squad_ids) == 11
    assert len(set(squad_ids)) == 11

    # Check budget constraint
    total_price = sum(players[pid].price for pid in squad_ids)
    assert total_price <= budget


def test_simulation_engine_simulate_once():
    """Test SimulationEngine.simulate_once with fixed seed."""
    rng = np.random.default_rng(42)

    # Create test data
    positions = (
        ["GK"] * 3
        + ["DEF"] * 6
        + ["MID"] * 6
        + ["FWD"] * 5
    )
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position=positions[i], team="A")
        for i in range(20)
    ]
    managers = [Manager(id=i, name=f"M{i}", budget=100.0) for i in range(5)]

    # Create components
    npc_picker = RandomNpcPicker()
    points_model = PositionBasedPointsModel()
    my_strategy_fn = pick_lowest_eo
    engine = SimulationEngine()

    # Run simulation
    result = engine.simulate_once(
        rng, managers, players, npc_picker, my_strategy_fn, points_model, 100.0
    )

    # Check result structure
    assert "eo" in result
    assert "my_team" in result
    assert "my_score" in result
    assert "field_scores" in result
    assert "my_rank" in result

    # Check shapes and types
    assert isinstance(result["eo"], np.ndarray)
    assert len(result["eo"]) == 20  # Number of players

    assert isinstance(result["my_team"], np.ndarray)
    assert len(result["my_team"]) == 11
    assert len(set(result["my_team"])) == 11  # All unique

    assert isinstance(result["my_score"], (int, float))

    assert isinstance(result["field_scores"], list)
    assert len(result["field_scores"]) == 5  # Number of managers

    assert isinstance(result["my_rank"], (int, float))
    assert 1 <= result["my_rank"] <= 6  # Rank should be between 1 and M+1 (5+1)


def test_simulation_engine_deterministic():
    """Test that simulation is deterministic with same seed."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    # Create test data
    positions = (
        ["GK"] * 3
        + ["DEF"] * 6
        + ["MID"] * 6
        + ["FWD"] * 5
    )
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position=positions[i], team="A")
        for i in range(20)
    ]
    managers = [Manager(id=i, name=f"M{i}", budget=100.0) for i in range(3)]

    # Create components
    npc_picker = RandomNpcPicker()
    points_model = PositionBasedPointsModel()
    my_strategy_fn = pick_lowest_eo
    engine = SimulationEngine()

    # Run simulation twice with same seed
    result1 = engine.simulate_once(
        rng1, managers, players, npc_picker, my_strategy_fn, points_model, 100.0
    )
    result2 = engine.simulate_once(
        rng2, managers, players, npc_picker, my_strategy_fn, points_model, 100.0
    )

    # Results should be identical
    np.testing.assert_array_equal(result1["my_team"], result2["my_team"])
    assert result1["my_score"] == result2["my_score"]
    assert result1["my_rank"] == result2["my_rank"]
    np.testing.assert_array_almost_equal(
        result1["field_scores"], result2["field_scores"]
    )
