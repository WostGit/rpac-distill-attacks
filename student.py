"""Student model utilities for query-based extraction simulation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class StudentModel:
    weights: np.ndarray
    bias: np.ndarray

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x)
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def init_student(n_features: int, n_classes: int, seed: int) -> StudentModel:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.02, size=(n_features, n_classes))
    bias = np.zeros((n_classes,), dtype=np.float64)
    return StudentModel(weights=weights, bias=bias)
