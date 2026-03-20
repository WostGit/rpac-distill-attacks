"""Teacher model utilities for distillation attack simulation."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class TeacherModel:
    model: LogisticRegression

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)


def train_teacher(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> TeacherModel:
    model = LogisticRegression(
        max_iter=400,
        random_state=seed,
        solver="lbfgs",
        multi_class="multinomial",
    )
    model.fit(x_train, y_train)
    return TeacherModel(model=model)
