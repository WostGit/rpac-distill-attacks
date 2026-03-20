"""PAC-style privacy leakage controller for query answers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class PACController:
    enabled: bool
    num_classes: int
    leakage_threshold: float = 8.0
    delta: float = 1e-5
    noise_std: float = 0.03

    leakage_total: float = 0.0
    queries: int = 0
    enforcement_triggered: bool = False

    def _query_leakage(self, probs: list[float]) -> float:
        return max(probs) - 1.0 / self.num_classes

    def process(self, probs: list[float], rng: random.Random) -> list[float]:
        self.queries += 1
        self.leakage_total += self._query_leakage(probs)
        if self.enabled and self.leakage_total > self.leakage_threshold:
            self.enforcement_triggered = True
            uniform = [1.0 / self.num_classes for _ in probs]
            noisy = [max(1e-6, p + rng.gauss(0.0, self.noise_std)) for p in uniform]
            s = sum(noisy)
            return [v / s for v in noisy]
        return probs

    def pac_bound(self) -> float:
        if self.queries <= 0:
            raise ValueError("PAC bound undefined with zero queries")
        return math.sqrt((2.0 * self.leakage_total + math.log(2.0 / self.delta)) / self.queries)
