from __future__ import annotations

from typing import Dict, List

POS_GK, POS_DEF, POS_MID, POS_FWD = 0, 1, 2, 3
ALL_POSITIONS: List[int] = [POS_GK, POS_DEF, POS_MID, POS_FWD]
POSITION_NAMES: Dict[int, str] = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}

FORMATION: Dict[int, int] = {POS_GK: 1, POS_DEF: 3, POS_MID: 4, POS_FWD: 3}
TEAM_SIZE: int = sum(FORMATION.values())


