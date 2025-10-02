"""Command-line interface for FPL EO-Only Simulator.

Provides entry point for running simulations with synthetic data.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from fpl_eo_sim.models import Manager, Player, Position
from fpl_eo_sim.sampling import make_rng
from fpl_eo_sim.simulator import (
    ConcentratedNpcPicker,
    PositionBasedPointsModel,
    RandomNpcPicker,
    SimulationEngine,
)
from fpl_eo_sim.strategies import (
    pick_barbell,
    pick_highest_eo,
    pick_lowest_eo,
    pick_random,
)


def create_synthetic_players(n: int, rng: np.random.Generator) -> list[Player]:
    """Create synthetic players for testing.

    Args:
        n: Number of players to create
        rng: Random number generator

    Returns:
        List of synthetic players
    """
    positions: list[Position] = ["GK", "DEF", "MID", "FWD"]
    teams = [f"Team_{i % 10}" for i in range(n)]  # 10 teams, cyclical

    # Prices from lognormal distribution clipped to [4.0, 13.0]
    log_prices = rng.normal(2.0, 0.5, n)  # lognormal parameters
    prices = np.clip(np.exp(log_prices), 4.0, 13.0)

    players = []
    for i in range(n):
        player = Player(
            id=i,
            name=f"Player_{i}",
            price=float(prices[i]),
            position=rng.choice(positions),
            team=teams[i],
        )
        players.append(player)

    return players


def create_synthetic_managers(m: int, budget: float) -> list[Manager]:
    """Create synthetic managers for testing.

    Args:
        m: Number of managers to create
        budget: Budget for each manager

    Returns:
        List of synthetic managers
    """
    managers = []
    for i in range(m):
        manager = Manager(id=i, name=f"Manager_{i}", budget=budget, squad=None)
        managers.append(manager)

    return managers


def get_strategy_function(strategy_name: str):
    """Get strategy function by name."""
    strategies = {
        "lowest_eo": pick_lowest_eo,
        "highest_eo": pick_highest_eo,
        "barbell": pick_barbell,
        "random": pick_random,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name]


def get_points_model(points_type: str, **kwargs):
    """Get points model by type."""
    if points_type == "position_based":
        return PositionBasedPointsModel(**kwargs)
    else:
        raise ValueError(f"Unknown points model: {points_type}. Use 'position_based' for the new Poisson/Negative Binomial model.")


def run_monte_carlo(
    rng: np.random.Generator,
    engine: SimulationEngine,
    runs: int,
    managers: list[Manager],
    players: list[Player],
    npc_picker: RandomNpcPicker,
    my_strategy_fn,
    points_model,
    budget: float,
) -> dict[str, Any]:
    """Run Monte Carlo simulation.

    Args:
        rng: Random number generator
        engine: Simulation engine
        runs: Number of simulation runs
        managers: List of managers
        players: List of players
        npc_picker: NPC picker instance
        my_strategy_fn: Strategy function
        points_model: Points model instance
        budget: Budget constraint

    Returns:
        Dictionary with simulation results
    """
    my_scores = []
    my_ranks = []
    field_scores_matrix = []

    for i in range(runs):
        result = engine.simulate_once(
            rng, managers, players, npc_picker, my_strategy_fn, points_model, budget
        )

        my_scores.append(result["my_score"])
        my_ranks.append(result["my_rank"])
        field_scores_matrix.append(result["field_scores"])

        # Lightweight progress logging every ~5% or on last
        if runs >= 20:
            step = max(1, runs // 20)
            if (i + 1) % step == 0 or i + 1 == runs:
                print(f"Progress: {i + 1}/{runs}")

    my_scores = np.array(my_scores)
    my_ranks = np.array(my_ranks)
    field_scores_matrix = np.array(field_scores_matrix)

    from fpl_eo_sim.metrics import summary

    summary_stats = summary(my_scores, my_ranks, field_scores_matrix)

    return {
        "runs": runs,
        "managers": len(managers),
        "players": len(players),
        "budget": budget,
        "summary": summary_stats,
    }


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="FPL EO-Only Simulator")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--managers", type=int, default=100, help="Number of NPC managers"
    )
    parser.add_argument("--players", type=int, default=50, help="Number of players")
    parser.add_argument("--budget", type=float, default=100.0, help="Manager budget")
    parser.add_argument(
        "--runs", type=int, default=500, help="Number of simulation runs"
    )
    parser.add_argument(
        "--points",
        choices=["position_based"],
        default="position_based",
        help="Points model (Poisson for GK, Negative Binomial for others)",
    )
    parser.add_argument(
        "--strategy",
        choices=["lowest_eo", "highest_eo", "barbell", "random"],
        default="lowest_eo",
        help="Strategy",
    )
    parser.add_argument(
        "--k-safe",
        type=int,
        default=5,
        help="Number of safe players for barbell strategy",
    )
    parser.add_argument(
        "--concentration",
        type=float,
        default=0.7,
        help="Concentration level for NPC picker (0.0=uniform, 1.0=max concentration)",
    )

    args = parser.parse_args()

    # Create RNG
    rng = make_rng(args.seed)

    # Create synthetic data
    players = create_synthetic_players(args.players, rng)
    managers = create_synthetic_managers(args.managers, args.budget)

    # Create components
    npc_picker = ConcentratedNpcPicker(concentration=args.concentration)
    points_model = get_points_model(args.points)
    my_strategy_fn = get_strategy_function(args.strategy)

    # Create strategy wrapper for barbell
    if args.strategy == "barbell":

        def strategy_wrapper(eo, K, rng):
            return pick_barbell(eo, K, args.k_safe, rng)

        my_strategy_fn = strategy_wrapper

    # Run simulation
    engine = SimulationEngine()
    results = run_monte_carlo(
        rng,
        engine,
        args.runs,
        managers,
        players,
        npc_picker,
        my_strategy_fn,
        points_model,
        args.budget,
    )

    # Print results
    print("FPL EO-Only Simulator Results")
    print("=" * 40)
    print(f"Runs: {results['runs']}")
    print(f"Managers: {results['managers']}")
    print(f"Players: {results['players']}")
    print(f"Budget: {results['budget']}")
    print(f"Concentration: {args.concentration}")
    print()

    summary = results["summary"]
    print("Performance Metrics:")
    print(f"Win vs median: {summary['win_rate_vs_median']:.3f}")
    print(f"P(top-10%): {summary['prob_top_10_percent']:.3f}")
    print(f"P(top-1%): {summary['prob_top_1_percent']:.3f}")
    print()

    print("Score Statistics:")
    print(f"Mean: {summary['my_score_mean']:.3f}")
    print(f"SD: {summary['my_score_sd']:.3f}")
    print(f"Quantiles: {summary['my_score_quantiles']}")
    print()

    print("Rank Statistics:")
    print(f"Mean: {summary['my_rank_mean']:.3f}")
    print(f"SD: {summary['my_rank_sd']:.3f}")
    print(f"Quantiles: {summary['my_rank_quantiles']}")

    # Output JSON to stdout
    print("\n" + "=" * 40)
    print("JSON Output:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
