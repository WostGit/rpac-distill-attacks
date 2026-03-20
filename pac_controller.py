"""PAC-style controller that tracks and limits cumulative leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PACConfig:
    delta: float = 1e-4
    epsilon_budget: float = 2.0
    max_queries: int = 500
    noise_std: float = 0.03
    clip_min: float = 1e-6


@dataclass
class PACState:
    queries_answered: int = 0
    cumulative_leakage: float = 0.0
    bound: float = 0.0
    enforcement_triggered: bool = False


class PACController:
    def __init__(self, config: PACConfig, enabled: bool) -> None:
        self.config = config
        self.enabled = enabled
        self.state = PACState()

    def _compute_bound(self) -> float:
        # Simple PAC-style concentration-style bound proxy.
        n = max(1, self.state.queries_answered)
        complexity = np.sqrt(np.log(2.0 / self.config.delta) / (2.0 * n))
        return float(self.state.cumulative_leakage / n + complexity)

    def process(self, probs: np.ndarray) -> np.ndarray:
        normalized = np.clip(probs, self.config.clip_min, 1.0)
        normalized /= normalized.sum()

        leakage = float(np.max(normalized) - 1.0 / normalized.shape[0])
        self.state.queries_answered += 1
        self.state.cumulative_leakage += leakage
        self.state.bound = self._compute_bound()

        if not self.enabled:
            return normalized

        if (
            self.state.queries_answered >= self.config.max_queries
            or self.state.bound > self.config.epsilon_budget
        ):
            self.state.enforcement_triggered = True

        if self.state.enforcement_triggered:
            softened = np.ones_like(normalized) / normalized.shape[0]
            noise = np.random.normal(0.0, self.config.noise_std, size=softened.shape)
            noisy = np.clip(softened + noise, self.config.clip_min, 1.0)
            noisy /= noisy.sum()
            return noisy

        return normalized

    def snapshot(self) -> dict[str, float | bool | int]:
        return {
            "queries_answered": self.state.queries_answered,
            "cumulative_leakage": self.state.cumulative_leakage,
            "pac_bound": self.state.bound,
            "enforcement_triggered": self.state.enforcement_triggered,
        }
