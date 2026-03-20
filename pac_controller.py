"""PAC-style leakage controller for teacher outputs."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PACController:
    enabled: bool
    leakage_budget: float
    leakage_rate: float
    noise_std: float
    clip_min: float
    clip_max: float

    cumulative_leakage: float = 0.0
    enforcement_triggered: bool = False

    def process(self, probs: np.ndarray) -> np.ndarray:
        query_count = max(1, probs.shape[0])
        if not self.enabled:
            self.cumulative_leakage += self.leakage_rate * query_count
            return probs

        self.cumulative_leakage += self.leakage_rate * query_count
        if self.cumulative_leakage > self.leakage_budget:
            self.enforcement_triggered = True
            # Output flattening once budget is exceeded
            probs = np.full_like(probs, fill_value=1.0 / probs.shape[1])

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=probs.shape)
            probs = probs + noise

        probs = np.clip(probs, self.clip_min, self.clip_max)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs

    def pac_bound(self, n_queries: int, delta: float = 1e-5) -> float:
        if n_queries <= 0:
            raise ValueError("n_queries must be positive to compute PAC bound")
        if self.cumulative_leakage < 0:
            raise ValueError("cumulative leakage cannot be negative")
        return float(self.cumulative_leakage + np.sqrt(np.log(1.0 / delta) / (2.0 * n_queries)))
