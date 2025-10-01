# FPL EO-Only Simulator

A tiny, readable simulator to study **ownership-only (EO)** strategies for Fantasy Premier League.

## Purpose

This simulator studies how effective ownership (EO) strategies perform in FPL by:
1. Having NPC managers lock their teams first
2. Computing effective ownership across the field
3. Using EO-only strategies to pick an XI (no projections)
4. Drawing scores from simple random models
5. Evaluating relative performance via Monte Carlo simulation

## Principles

- **Clarity first**: Standard library + NumPy only, strong typing, docstrings, small OOP
- **Determinism**: Single `numpy.random.Generator` for reproducibility
- **Lean scope**: XI only (no bench/captain/chips/transfers), minimal constraints in Iteration 1
- **Testing culture**: Each module gets at least one quick deterministic test

## Installation

```bash
# Install dependencies
poetry install

# Or with pip
pip install numpy
```

## Quick Demo

Run a basic simulation with synthetic data:

```bash
python -m fpl_eo_sim.cli --seed 42 --managers 100 --players 50 --budget 100.0 --runs 5000 --points studentt --strategy barbell --k-safe 5
```

Expected output ranges:
- Win vs median: ~0.47–0.53
- P(top-10%): ~0.08–0.15
- P(top-1%): ~0.008–0.02

*Note: Values vary with RNG; ranges are sanity bands.*

## Usage

### Command Line Interface

```bash
python -m fpl_eo_sim.cli [OPTIONS]
```

Options:
- `--seed`: Random seed (default: None)
- `--managers`: Number of NPC managers (default: 100)
- `--players`: Number of players (default: 50)
- `--budget`: Manager budget (default: 100.0)
- `--runs`: Number of simulation runs (default: 1000)
- `--points`: Points model {normal,studentt} (default: normal)
- `--strategy`: Strategy {lowest_eo,highest_eo,barbell} (default: barbell)
- `--k-safe`: Number of safe players for barbell strategy (default: 5)

### Programmatic Usage

```python
from fpl_eo_sim.simulator import SimulationEngine, RandomNpcPicker, NormalPointsModel
from fpl_eo_sim.strategies import pick_lowest_eo
from fpl_eo_sim.sampling import make_rng

# Create components
rng = make_rng(42)
engine = SimulationEngine()
npc_picker = RandomNpcPicker()
points_model = NormalPointsModel(mean=0.0, sd=1.0)

# Run simulation
result = engine.simulate_once(
    rng, managers, players, npc_picker, pick_lowest_eo, points_model, budget
)
```

## Project Structure

```
fpl_eo_sim/
├── __init__.py          # Package initialization
├── models.py            # Data models (Player, Squad, Manager, Gameweek)
├── strategies.py        # EO computation and strategy functions
├── simulator.py         # Simulation engine and NPC pickers
├── sampling.py          # Random sampling utilities
├── metrics.py           # Performance metrics and analysis
└── cli.py              # Command-line interface

tests/
├── test_models.py       # Tests for data models
├── test_sampling.py     # Tests for sampling utilities
├── test_strategies.py   # Tests for strategy functions
└── test_simulator.py    # Tests for simulation engine
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fpl_eo_sim

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
black fpl_eo_sim/ tests/

# Sort imports
isort fpl_eo_sim/ tests/

# Type checking
mypy fpl_eo_sim/
```

## Iteration 1 Scope

This is Iteration 1 with minimal constraints:
- XI selection only (no bench/captain/chips/transfers)
- Budget constraint only (no position/club constraints yet)
- Random NPC picker (no sophisticated strategies)
- Simple points models (normal/Student's t distributions)

Future iterations will add:
- Position and club constraints
- More sophisticated NPC strategies
- Real FPL data integration
- Advanced points models

## License

MIT License

