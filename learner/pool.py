from __future__ import annotations

from typing import Dict

import numpy as np

from .constants import ALL_POSITIONS


class PlayerPool:
    def __init__(self, num_per_position: Dict[int, int], rng: np.random.Generator) -> None:
        self.position_to_ids: Dict[int, np.ndarray] = {}
        self.position_to_slice: Dict[int, slice] = {}
        start_index = 0
        for position in ALL_POSITIONS:
            count = num_per_position[position]
            ids = np.arange(start_index, start_index + count, dtype=int)
            self.position_to_ids[position] = ids
            self.position_to_slice[position] = slice(start_index, start_index + count)
            start_index += count
        self.num_players = start_index
        self.skill = rng.normal(0.0, 1.0, size=self.num_players).astype(np.float32)


