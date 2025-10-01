"""Data models for FPL EO-Only Simulator.

Contains Player, Squad, Manager, and Gameweek data structures with basic validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Position = Literal["GK", "DEF", "MID", "FWD"]


@dataclass
class Player:
    """Represents a single FPL player."""

    id: int
    name: str
    price: float
    position: Position
    team: str


@dataclass
class Squad:
    """Represents a squad of exactly 11 unique players."""

    player_ids: list[int]

    def __post_init__(self) -> None:
        """Validate squad size."""
        if len(self.player_ids) != 11:
            raise ValueError(
                f"Squad must have exactly 11 players, got {len(self.player_ids)}"
            )
        if len(set(self.player_ids)) != len(self.player_ids):
            raise ValueError("Squad must have unique players")

    def total_price(self, players_by_id: dict[int, Player]) -> float:
        """Calculate total price of the squad."""
        return sum(players_by_id[pid].price for pid in self.player_ids)


@dataclass
class Manager:
    """Represents a manager with budget and optional squad."""

    id: int
    name: str
    budget: float
    squad: Squad | None = None


@dataclass
class Gameweek:
    """Represents a gameweek with players and managers."""

    id: int
    players: list[Player]
    managers: list[Manager]


def players_by_id(players: list[Player]) -> dict[int, Player]:
    """Create a lookup dictionary for players by ID."""
    return {player.id: player for player in players}


def validate_squad_constraints(
    squad: Squad,
    players_by_id: dict[int, Player],
    budget: float,
    positions_required: dict[Position, tuple[int, int]] | None = None,
    club_cap: int = 3,
) -> bool:
    """Validate squad constraints.

    For Iteration 1, only validates squad size and budget.
    Position and club constraints will be added later.
    """
    if len(squad.player_ids) != 11:
        return False

    if squad.total_price(players_by_id) > budget:
        return False

    # TODO: Add position and club constraints in future iterations
    return True
