"""Query-based extraction attack orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pac_controller import PACConfig, PACController
from student import StudentConfig, StudentModel
from teacher import TeacherModel


@dataclass
class AttackConfig:
    protected: bool
    max_queries: int = 500


def run_extraction_attack(
    teacher: TeacherModel,
    query_texts: list[str],
    config: AttackConfig,
    *,
    seed: int,
) -> tuple[StudentModel, dict[str, float | bool | int]]:
    np.random.seed(seed)
    pac = PACController(
        PACConfig(max_queries=config.max_queries),
        enabled=config.protected,
    )

    gathered_queries: list[str] = []
    gathered_probs: list[np.ndarray] = []

    for query in query_texts[: config.max_queries]:
        raw = teacher.predict_proba([query])[0]
        protected = pac.process(raw)
        gathered_queries.append(query)
        gathered_probs.append(protected)

        if config.protected and pac.state.enforcement_triggered:
            break

    if not gathered_probs:
        raise RuntimeError("No teacher outputs collected; cannot train student.")

    teacher_matrix = np.array(gathered_probs)
    student = StudentModel(StudentConfig()).fit_from_teacher_probs(gathered_queries, teacher_matrix)
    metrics = pac.snapshot()
    metrics["queries_used"] = len(gathered_queries)
    return student, metrics
