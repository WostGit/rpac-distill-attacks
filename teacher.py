"""Small softmax teacher model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TeacherConfig:
    lr: float = 0.4
    steps: int = 320
    l2: float = 1e-3
    seed: int = 42


class TeacherModel:
    def __init__(self, input_dim: int, n_classes: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w = 0.01 * rng.standard_normal((input_dim, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._softmax(x @ self.w + self.b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def train_supervised(self, x: np.ndarray, y: np.ndarray, cfg: TeacherConfig) -> None:
        y_one_hot = np.eye(self.b.shape[0], dtype=np.float32)[y]
        for _ in range(cfg.steps):
            probs = self.predict_proba(x)
            grad_logits = (probs - y_one_hot) / x.shape[0]
            grad_w = x.T @ grad_logits + cfg.l2 * self.w
            grad_b = grad_logits.sum(axis=0)
            self.w -= cfg.lr * grad_w
            self.b -= cfg.lr * grad_b

            if not np.isfinite(self.w).all() or not np.isfinite(self.b).all():
                raise FloatingPointError("teacher training diverged (NaN/inf)")
