from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ActionCountCallback(BaseCallback):
    def __init__(self, n_actions: int, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.n_actions = n_actions
        self.log_freq = log_freq
        self.counts = np.zeros(n_actions, dtype=np.int64)
        self._steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            a = info.get("action")
            if a is not None and 0 <= a < self.n_actions:
                self.counts[a] += 1
        self._steps += 1
        if self._steps % self.log_freq == 0:
            total = int(self.counts.sum())
            if total > 0:
                fracs = self.counts / total
                for a in range(self.n_actions):
                    self.logger.record(f"actions/frac_{a}", float(fracs[a]))
        return True

    def _on_training_end(self) -> None:
        total = int(self.counts.sum())
        if total > 0:
            fracs = self.counts / total
            print(f"Action counts: {self.counts.tolist()} | fracs={fracs.tolist()}")


