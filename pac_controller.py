"""PAC-style leakage controller for teacher outputs."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class PACConfig:
    enabled: bool = False
    leakage_budget: float = 90.0
    delta: float = 1e-5
    noise_std: float = 0.03


class PACController:
    def __init__(self, n_classes: int, cfg: PACConfig, seed: int = 7) -> None:
        self.n_classes = n_classes
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.cumulative_leakage = 0.0
        self.query_count = 0
        self.enforcement_triggered = False

    def _leakage_increment(self, probs: np.ndarray) -> float:
        eps = 1e-8
        ent = -np.sum(probs * np.log2(np.clip(probs, eps, 1.0)))
        max_ent = math.log2(self.n_classes)
        return float(max(0.0, max_ent - ent))

    def process(self, probs: np.ndarray) -> np.ndarray:
        probs = probs.astype(np.float64)
        self.query_count += 1
        self.cumulative_leakage += self._leakage_increment(probs)

        if not self.cfg.enabled:
            return probs.astype(np.float32)

        if self.cumulative_leakage > self.cfg.leakage_budget:
            self.enforcement_triggered = True
            noisy = probs + self.rng.normal(0.0, self.cfg.noise_std, size=probs.shape)
            noisy = np.clip(noisy, 1e-6, None)
            noisy /= noisy.sum()
            uniform_mix = np.full_like(noisy, 1.0 / self.n_classes)
            clipped = 0.35 * noisy + 0.65 * uniform_mix
            clipped /= clipped.sum()
            return clipped.astype(np.float32)

        return probs.astype(np.float32)

    def pac_bound(self) -> float:
        if self.query_count <= 0:
            raise ValueError("PAC bound undefined: no queries processed")
        delta_term = math.log(2.0 / self.cfg.delta)
        bound = math.sqrt((2.0 * self.cumulative_leakage + delta_term) / self.query_count)
        if not math.isfinite(bound):
            raise ValueError("PAC bound computation failed")
        return bound
