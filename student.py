"""Student model that learns only from teacher outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class StudentConfig:
    random_state: int = 11
    c: float = 1.0
    max_iter: int = 300


class StudentModel:
    def __init__(self, config: StudentConfig) -> None:
        self.config = config
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=900,
                    ),
                ),
                (
                    "logreg",
                    LogisticRegression(
                        C=config.c,
                        random_state=config.random_state,
                        max_iter=config.max_iter,
                        multi_class="multinomial",
                    ),
                ),
            ]
        )

    def fit_from_teacher_probs(self, queries: list[str], teacher_probs: np.ndarray) -> "StudentModel":
        pseudo_labels = np.argmax(teacher_probs, axis=1)
        self.pipeline.fit(queries, pseudo_labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)
