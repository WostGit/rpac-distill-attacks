"""Teacher model utilities for query-based distillation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


@dataclass
class TeacherConfig:
    random_state: int = 7
    hidden_layer_sizes: tuple[int, ...] = (64,)
    max_iter: int = 120
    learning_rate_init: float = 0.01


class TeacherModel:
    """A tiny text teacher model (TF-IDF + MLP)."""

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=1200,
                    ),
                ),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=config.hidden_layer_sizes,
                        random_state=config.random_state,
                        max_iter=config.max_iter,
                        learning_rate_init=config.learning_rate_init,
                        solver="adam",
                        early_stopping=True,
                        n_iter_no_change=8,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels: np.ndarray) -> "TeacherModel":
        self.pipeline.fit(texts, labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)
