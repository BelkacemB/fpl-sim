"""Tests for models module."""

import numpy as np
import pytest

from fpl_eo_sim.models import (
    Gameweek,
    Manager,
    Player,
    Squad,
    players_by_id,
    validate_squad_constraints,
)


def test_player_creation():
    """Test Player creation and attributes."""
    player = Player(id=1, name="Test Player", price=10.5, position="MID", team="Team A")

    assert player.id == 1
    assert player.name == "Test Player"
    assert player.price == 10.5
    assert player.position == "MID"
    assert player.team == "Team A"


def test_squad_creation_valid():
    """Test Squad creation with valid data."""
    squad = Squad(player_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    assert len(squad.player_ids) == 11
    assert squad.player_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_squad_creation_invalid_size():
    """Test Squad creation with invalid size."""
    with pytest.raises(ValueError, match="Squad must have exactly 11 players"):
        Squad(player_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Only 10 players


def test_squad_creation_duplicate_players():
    """Test Squad creation with duplicate players."""
    with pytest.raises(ValueError, match="Squad must have unique players"):
        Squad(player_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1])  # Duplicate player 1


def test_squad_total_price():
    """Test Squad total_price calculation."""
    players = [
        Player(id=1, name="P1", price=5.0, position="GK", team="A"),
        Player(id=2, name="P2", price=7.5, position="DEF", team="A"),
        Player(id=3, name="P3", price=10.0, position="MID", team="A"),
        Player(id=4, name="P4", price=12.5, position="FWD", team="A"),
        Player(id=5, name="P5", price=8.0, position="GK", team="B"),
        Player(id=6, name="P6", price=6.5, position="DEF", team="B"),
        Player(id=7, name="P7", price=9.0, position="MID", team="B"),
        Player(id=8, name="P8", price=11.0, position="FWD", team="B"),
        Player(id=9, name="P9", price=4.5, position="GK", team="C"),
        Player(id=10, name="P10", price=8.5, position="DEF", team="C"),
        Player(id=11, name="P11", price=7.0, position="MID", team="C"),
    ]

    squad = Squad(player_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    players_by_id_dict = players_by_id(players)

    expected_total = sum(p.price for p in players)
    assert squad.total_price(players_by_id_dict) == expected_total


def test_manager_creation():
    """Test Manager creation."""
    manager = Manager(id=1, name="Test Manager", budget=100.0)

    assert manager.id == 1
    assert manager.name == "Test Manager"
    assert manager.budget == 100.0
    assert manager.squad is None


def test_manager_with_squad():
    """Test Manager creation with squad."""
    squad = Squad(player_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    manager = Manager(id=1, name="Test Manager", budget=100.0, squad=squad)

    assert manager.squad == squad


def test_gameweek_creation():
    """Test Gameweek creation."""
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position="MID", team="A")
        for i in range(5)
    ]
    managers = [Manager(id=i, name=f"M{i}", budget=100.0) for i in range(3)]

    gw = Gameweek(id=1, players=players, managers=managers)

    assert gw.id == 1
    assert len(gw.players) == 5
    assert len(gw.managers) == 3


def test_players_by_id():
    """Test players_by_id helper function."""
    players = [
        Player(id=1, name="P1", price=10.0, position="MID", team="A"),
        Player(id=5, name="P5", price=15.0, position="FWD", team="B"),
        Player(id=3, name="P3", price=8.0, position="DEF", team="C"),
    ]

    lookup = players_by_id(players)

    assert len(lookup) == 3
    assert lookup[1].name == "P1"
    assert lookup[5].name == "P5"
    assert lookup[3].name == "P3"


def test_validate_squad_constraints_valid():
    """Test squad validation with valid constraints."""
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position="MID", team="A")
        for i in range(11)
    ]
    squad = Squad(player_ids=list(range(11)))
    players_by_id_dict = players_by_id(players)

    assert validate_squad_constraints(squad, players_by_id_dict, 110.0) is True


def test_validate_squad_constraints_budget_exceeded():
    """Test squad validation with budget exceeded."""
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position="MID", team="A")
        for i in range(11)
    ]
    squad = Squad(player_ids=list(range(11)))
    players_by_id_dict = players_by_id(players)

    assert validate_squad_constraints(squad, players_by_id_dict, 50.0) is False


def test_validate_squad_constraints_wrong_size():
    """Test squad validation with wrong size."""
    players = [
        Player(id=i, name=f"P{i}", price=10.0, position="MID", team="A")
        for i in range(11)
    ]
    # Create a squad with wrong size by manually creating the Squad object
    # We need to bypass the __post_init__ validation to test the validate_squad_constraints function
    squad = Squad.__new__(Squad)
    squad.player_ids = list(range(10))  # Only 10 players instead of 11
    players_by_id_dict = players_by_id(players)

    assert validate_squad_constraints(squad, players_by_id_dict, 100.0) is False
