"""Student model trained only from teacher outputs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StudentConfig:
    lr: float = 0.35
    steps: int = 260
    l2: float = 1e-3
    seed: int = 99


class StudentModel:
    def __init__(self, input_dim: int, n_classes: int, seed: int = 99, width_scale: float = 1.0) -> None:
        rng = np.random.default_rng(seed)
        hidden = max(8, int(input_dim * 0.35 * width_scale))
        self.w1 = 0.04 * rng.standard_normal((input_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = 0.04 * rng.standard_normal((hidden, n_classes)).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_raw = x @ self.w1 + self.b1
        h = np.tanh(h_raw)
        logits = h @ self.w2 + self.b2
        probs = self._softmax(logits)
        return h, logits, probs

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[2]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def train_distill(self, x: np.ndarray, teacher_probs: np.ndarray, cfg: StudentConfig, batch_size: int = 64) -> None:
        rng = np.random.default_rng(cfg.seed)
        n = x.shape[0]
        for _ in range(cfg.steps):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                xb = x[batch_idx]
                tb = teacher_probs[batch_idx]

                h, _, probs = self._forward(xb)
                diff = (probs - tb) / xb.shape[0]

                grad_w2 = h.T @ diff + cfg.l2 * self.w2
                grad_b2 = diff.sum(axis=0)
                dh = (diff @ self.w2.T) * (1 - h * h)
                grad_w1 = xb.T @ dh + cfg.l2 * self.w1
                grad_b1 = dh.sum(axis=0)

                self.w2 -= cfg.lr * grad_w2
                self.b2 -= cfg.lr * grad_b2
                self.w1 -= cfg.lr * grad_w1
                self.b1 -= cfg.lr * grad_b1

                if (
                    not np.isfinite(self.w1).all()
                    or not np.isfinite(self.w2).all()
                    or not np.isfinite(self.b1).all()
                    or not np.isfinite(self.b2).all()
                ):
                    raise FloatingPointError("student training diverged (NaN/inf)")
