# Cursor Prompts for FPL EO-Only Simulator (Iteration 1)

**Purpose:** build a tiny, readable simulator to study **ownership-only (EO)** strategies. The field (NPC managers) locks teams first; we observe EO; then "my" strategy picks an XI using EO only (no projections), scores are drawn from a simple random model, and we evaluate **relative performance** via Monte Carlo.

**Principles:**
- _Clarity first:_ standard library + **NumPy only**, strong typing, docstrings, and small OOP where it helps.
- _Determinism:_ thread a single `numpy.random.Generator` for reproducibility.
- _Lean scope:_ XI only (no bench/captain/chips/transfers); minimal constraints in Iteration 1; add realism later.
- _Testing culture:_ each module gets at least one quick deterministic test.

---

## Prompt 1 — Project scaffold & code style
**Why:** establish a clean, minimal layout so iteration is fast and tests are easy to run. Keeping modules small avoids premature abstraction.

Create a minimal Python project scaffold:
```
fpl_eo_sim/
  __init__.py
  models.py
  strategies.py
  simulator.py
  sampling.py
  metrics.py
  cli.py
tests/
  test_models.py
  test_sampling.py
  test_strategies.py
  test_simulator.py
pyproject.toml
README.md
```
Requirements: Python 3.11+, NumPy only (no pandas). Use type hints, dataclasses where suitable, small classes, clear docstrings. Keep files lean. Add `pyproject.toml` with black/isort config (no deps). README: quickstart and CLI usage.

---

## Prompt 2 — Models (Player, Manager, Gameweek)
**Why:** define simple data containers and basic validation. We only need enough structure to hold players, managers, and an XI per manager.

Implement **models.py** with:
- `Position = Literal["GK","DEF","MID","FWD"]`
- `Player`: `id: int`, `name: str`, `price: float`, `position: Position`, `team: str`
- `Squad`: holds exactly 11 unique player IDs; method `total_price(players_by_id) -> float`
- `Manager`: `id: int`, `name: str`, `budget: float`, `squad: Squad | None`
- `Gameweek`: `id: int`, `players: list[Player]`, `managers: list[Manager]`
Provide helpers:
- `players_by_id(players) -> dict[int, Player]`
- `validate_squad_constraints(squad, players_by_id, budget, positions_required: dict[Position, tuple[int,int]], club_cap=3) -> bool` but default to **Iteration 1**: only enforce `len==11` and `total_price<=budget` (no positions/club caps yet).

---

## Prompt 3 — Random sampling utilities
**Why:** centralized, reproducible randomness and sampling primitives keep the simulator deterministic and testable.

**What is RNG?**
- **RNG** = **Random Number Generator**. We will pass around a `numpy.random.Generator` instance instead of using the global `np.random` so results are reproducible with a fixed seed and independent across components.
- Helper: `make_rng(seed)` returns a `Generator`; `seed=None` uses system entropy; a fixed integer seed yields repeatable draws.

Implement **sampling.py**:
- Use `numpy.random.Generator` with seedable constructor `make_rng(seed: int|None) -> Generator`.
- `weighted_choice_without_replacement(rng, population: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray` using NumPy only.
- `normal_points(rng, n: int, mean=0.0, sd=1.0) -> np.ndarray`
- `student_t_points(rng, n: int, df=5) -> np.ndarray`
- Simple arg validation + docstrings.

---

## Prompt 4 — NPC manager picker (random XI under budget)
**Why:** we need a baseline field that forms teams blindly (no EO/meta). In Iteration 1, keep it uniform/random and only respect budget and uniqueness.

In **simulator.py**, add:
- `RandomNpcPicker` class with method:
  ```py
  def pick_xi(self, rng, players: list[Player], budget: float) -> np.ndarray:
      """
      Return an array of 11 unique player ids chosen uniformly at random
      (subject to total price <= budget). Retry with a capped loop; if not
      feasible within attempts, relax by removing the most expensive candidate.
      """
  ```
Keep it readable; no positions/club constraints in Iteration 1. Use `models.validate_squad_constraints` to check budget and size only.

---

## Prompt 5 — EO computation & “my” strategies (anti/template)
**Why:** EO is an **outcome of field picks**. “My” strategies should use EO only (ownership mechanics) to select an XI: the simplest levers are lowest EO (max leverage) and highest EO (template baseline).

Implement **strategies.py** with pure EO rules:
- `compute_effective_ownership(field_squads: np.ndarray, num_players: int) -> np.ndarray`  
  where `field_squads` is shape `(M, 11)` of player IDs; returns EO in `[0,1]`.
- `pick_lowest_eo(eo: np.ndarray, K=11, rng=None) -> np.ndarray` (ties broken randomly).
- `pick_highest_eo(eo: np.ndarray, K=11, rng=None) -> np.ndarray`
- A convenience `pick_barbell(eo, K=11, k_safe=5, rng=None)`.

---

## Prompt 6 — Gameweek simulation (points then scores/ranks)
**Why:** one run = one world. We want a small engine that (1) lets the field lock teams, (2) computes EO, (3) lets "me" pick from EO only, (4) samples points, (5) scores and ranks.

In **simulator.py**, add a small OOP engine:
- `PointsModel` protocol with `.sample_points(rng, players: list[Player]) -> dict[int,float]`
- `NormalPointsModel(mean=0.0, sd=1.0)`, `StudentTPointsModel(df=5)`
- `SimulationEngine` with:
  - `simulate_once(rng, gw: Gameweek, npc_picker, my_strategy_fn, points_model, budget: float) -> dict`
    Steps:
    1) NPCs pick XI (array of ids) for each manager (M).
    2) Compute EO.
    3) “My” team: call `my_strategy_fn(eo, K=11, rng)` to get ids.
    4) Sample per-player points vector.
    5) Sum scores for each team, compute my rank vs field (competition ranking: 1 + #strictly_greater + 0.5*#ties).
    6) Return a dict with `eo`, `my_team`, `my_score`, `field_scores`, `my_rank`.

---

## Prompt 7 — Monte Carlo loop + metrics
**Why:** single runs are noise. Monte Carlo reveals distributions: how often we beat median, top-10%, etc., which is the core question for EO-only leverage.

Create **metrics.py** with:
- `quantiles(arr, qs=(0.05,0.5,0.95)) -> dict[float,float]`
- `win_rate_vs_median(my_scores, field_scores_matrix) -> float`
- `prob_top_k_percent(my_ranks, M, pct=10) -> float`
- `summary(my_scores, my_ranks, field_scores_matrix) -> dict`
In **simulator.py**, add:
- `run_monte_carlo(rng, engine, runs: int, ...) -> dict`  
  that stores per-run `my_score`, `my_rank`, and `field_scores`, then calls `metrics.summary`.

---

## Prompt 8 — CLI entry point
**Why:** one command to run the whole pipeline with synthetic data so we can iterate quickly and share reproducible examples.

Implement **cli.py** to run Iteration 1 end-to-end:
- Args: `--seed`, `--managers M`, `--players N`, `--budget`, `--runs`, `--points {normal,studentt}`, `--strategy {lowest_eo,highest_eo,barbell}`, `--k-safe`
- Build a synthetic pool of `N` players with:
  - prices drawn from a lognormal clipped to [4.0, 13.0]
  - positions random (unused for now)
  - teams cyclical labels
- Create `M` NPC managers with equal budgets.
- Use `RandomNpcPicker`, `NormalPointsModel()` by default.
- Print summary metrics neatly and dump JSON to stdout.
- `if __name__ == "__main__": main()` guard.

---

## Prompt 9 — Tests (pytest) small & fast
**Why:** quick tests keep regressions from creeping in and ensure determinism with a fixed RNG seed.

Write unit tests:
- **test_models.py**: 15 dummy players, make a `Squad` of 11, check `total_price` and budget validation.
- **test_sampling.py**: `weighted_choice_without_replacement` returns unique ids of size k and favors higher weights over many trials.
- **test_strategies.py**: EO computation correctness, lowest/highest selection sizes & membership.
- **test_simulator.py**: one `simulate_once` run with fixed seed returns valid shapes, rank within `[1, M+1]`; Monte Carlo summary keys exist and probs within [0,1].
Keep tests deterministic with a fixed RNG seed.

---

## Prompt 10 — Readability & guardrails
**Why:** consistent style makes collaboration and future extensions painless.

Apply consistent style:
- Add module-level docstrings summarizing responsibilities.
- Add function/class docstrings (args/returns).
- Prefer `np.ndarray` for vectors; convert to Python types only at API boundaries (CLI/JSON).
- Validate inputs with `ValueError`.
- Keep functions short; factor helpers when >25 lines.
- Ensure `python -m compileall .` and `pytest -q` succeed.

---

## Prompt 11 — Small acceptance demo
**Why:** provide a reproducible command with expected ranges so you can sanity-check the pipeline quickly.

Add a target in README “Quick Demo”:
```
python -m fpl_eo_sim.cli --seed 42 --managers 100 --players 50 --budget 100.0 --runs 5000 --points studentt --strategy barbell --k-safe 5
```
Expected output (example):
```
Win vs median: ~0.47–0.53
P(top-10%): ~0.08–0.15
P(top-1%): ~0.008–0.02
Score mean/sd/quantiles...
Rank mean/sd/quantiles...
```
Include note: values vary with RNG; ranges are sanity bands.

---

## Prompt 12 — (Optional) Speed & seed hygiene
**Why:** cheap wins on performance and reproducibility once the basics work.

Micro-optimize hot paths:
- Vectorize score aggregation with `points[team_ids].sum()`.
- Pre-allocate arrays in Monte Carlo.
- Pass a single `rng` down the stack.
- Add a `--bench` flag in CLI to print runtime per 10k runs.
