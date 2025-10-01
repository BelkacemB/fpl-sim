"""Metrics and analysis utilities for FPL EO-Only Simulator.

Functions for analyzing simulation results and computing performance metrics.
"""

from __future__ import annotations

import numpy as np


def quantiles(
    arr: np.ndarray, qs: tuple[float, ...] = (0.05, 0.5, 0.95)
) -> dict[float, float]:
    """Compute quantiles of an array.

    Args:
        arr: Input array
        qs: Quantile values to compute

    Returns:
        Dictionary mapping quantile values to computed quantiles
    """
    if len(arr) == 0:
        return {q: 0.0 for q in qs}

    computed_quantiles = np.quantile(arr, qs)
    return dict(zip(qs, computed_quantiles))


def win_rate_vs_median(my_scores: np.ndarray, field_scores_matrix: np.ndarray) -> float:
    """Calculate win rate against median field score.

    Args:
        my_scores: Array of my scores across runs
        field_scores_matrix: Array of shape (runs, managers) with field scores

    Returns:
        Fraction of runs where my score beats median field score
    """
    if len(my_scores) == 0 or field_scores_matrix.size == 0:
        return 0.0

    median_scores = np.median(field_scores_matrix, axis=1)
    wins = np.sum(my_scores > median_scores)
    return wins / len(my_scores)


def prob_top_k_percent(my_ranks: np.ndarray, M: int, pct: float = 10.0) -> float:
    """Calculate probability of finishing in top k% of field.

    Args:
        my_ranks: Array of my ranks across runs
        M: Number of managers in field
        pct: Top percentage threshold (e.g., 10.0 for top 10%)

    Returns:
        Fraction of runs in top k%
    """
    if len(my_ranks) == 0 or M <= 0:
        return 0.0

    k = max(1, int(M * pct / 100))
    top_k_threshold = k
    top_k_finishes = np.sum(my_ranks <= top_k_threshold)
    return top_k_finishes / len(my_ranks)


def summary(
    my_scores: np.ndarray, my_ranks: np.ndarray, field_scores_matrix: np.ndarray
) -> dict[str, float | dict[float, float]]:
    """Generate summary statistics for simulation results.

    Args:
        my_scores: Array of my scores across runs
        my_ranks: Array of my ranks across runs
        field_scores_matrix: Array of shape (runs, managers) with field scores

    Returns:
        Dictionary with summary statistics
    """
    if len(my_scores) == 0:
        return {
            "my_score_mean": 0.0,
            "my_score_sd": 0.0,
            "my_score_quantiles": {0.05: 0.0, 0.5: 0.0, 0.95: 0.0},
            "my_rank_mean": 0.0,
            "my_rank_sd": 0.0,
            "my_rank_quantiles": {0.05: 0.0, 0.5: 0.0, 0.95: 0.0},
            "win_rate_vs_median": 0.0,
            "prob_top_10_percent": 0.0,
            "prob_top_1_percent": 0.0,
        }

    M = field_scores_matrix.shape[1] if field_scores_matrix.size > 0 else 1

    return {
        "my_score_mean": float(np.mean(my_scores)),
        "my_score_sd": float(np.std(my_scores)),
        "my_score_quantiles": quantiles(my_scores),
        "my_rank_mean": float(np.mean(my_ranks)),
        "my_rank_sd": float(np.std(my_ranks)),
        "my_rank_quantiles": quantiles(my_ranks),
        "win_rate_vs_median": win_rate_vs_median(my_scores, field_scores_matrix),
        "prob_top_10_percent": prob_top_k_percent(my_ranks, M, 10.0),
        "prob_top_1_percent": prob_top_k_percent(my_ranks, M, 1.0),
    }
