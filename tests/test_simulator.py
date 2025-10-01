"""Tests for simulator module."""

import numpy as np

from fpl_eo_sim.models import Manager, Player
from fpl_eo_sim.simulator import (
    NormalPointsModel,
    RandomNpcPicker,
    SimulationEngine,
    StudentTPointsModel,
)
from fpl_eo_sim.strategies import pick_lowest_eo


def test_normal_points_model():
    """Test NormalPointsModel."""
    rng = np.random.default_rng(42)
    players = [
        Player(id=0, name="P0", price=10.0, position="MID", team="A"),
        Player(id=1, name="P1", price=10.0, position="MID", team="A"),
    ]

    model = NormalPointsModel(mean=5.0, sd=2.0)
    points = model.sample_points(rng, players)

    assert len(points) == 2
    assert 0 in points
    assert 1 in points
    assert isinstance(points[0], float)
    assert isinstance(points[1], float)


def test_student_t_points_model():
    """Test StudentTPointsModel."""
    rng = np.random.default_rng(42)
    players = [
        Player(id=0, name="P0", price=10.0, position="MID", team="A"),
        Player(id=1, name="P1", price=10.0, position="MID", team="A"),
    ]

    model = StudentTPointsModel(df=5.0)
    points = model.sample_points(rng, players)

    assert len(points) == 2
    assert 0 in points
    assert 1 in points
    assert isinstance(points[0], float)
    assert isinstance(points[1], float)


def test_random_npc_picker_basic():
    """Test RandomNpcPicker basic functionality."""
    rng = np.random.default_rng(42)
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position="MID", team="A")
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
    players = [
        Player(id=0, name="P0", price=5.0, position="MID", team="A"),
        Player(id=1, name="P1", price=5.0, position="MID", team="A"),
        Player(id=2, name="P2", price=5.0, position="MID", team="A"),
        Player(id=3, name="P3", price=5.0, position="MID", team="A"),
        Player(id=4, name="P4", price=5.0, position="MID", team="A"),
        Player(id=5, name="P5", price=5.0, position="MID", team="A"),
        Player(id=6, name="P6", price=5.0, position="MID", team="A"),
        Player(id=7, name="P7", price=5.0, position="MID", team="A"),
        Player(id=8, name="P8", price=5.0, position="MID", team="A"),
        Player(id=9, name="P9", price=5.0, position="MID", team="A"),
        Player(id=10, name="P10", price=5.0, position="MID", team="A"),
        Player(id=11, name="P11", price=5.0, position="MID", team="A"),
        Player(id=12, name="P12", price=5.0, position="MID", team="A"),
        Player(id=13, name="P13", price=5.0, position="MID", team="A"),
        Player(id=14, name="P14", price=5.0, position="MID", team="A"),
        Player(id=15, name="P15", price=5.0, position="MID", team="A"),
        Player(id=16, name="P16", price=5.0, position="MID", team="A"),
        Player(id=17, name="P17", price=5.0, position="MID", team="A"),
        Player(id=18, name="P18", price=5.0, position="MID", team="A"),
        Player(id=19, name="P19", price=5.0, position="MID", team="A"),
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
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position="MID", team="A")
        for i in range(20)
    ]
    managers = [Manager(id=i, name=f"M{i}", budget=100.0) for i in range(5)]

    # Create components
    npc_picker = RandomNpcPicker()
    points_model = NormalPointsModel(mean=0.0, sd=1.0)
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
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position="MID", team="A")
        for i in range(20)
    ]
    managers = [Manager(id=i, name=f"M{i}", budget=100.0) for i in range(3)]

    # Create components
    npc_picker = RandomNpcPicker()
    points_model = NormalPointsModel(mean=0.0, sd=1.0)
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
