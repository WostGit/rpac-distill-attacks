"""Query-based distillation attack orchestration."""

from dataclasses import dataclass

import numpy as np

from pac_controller import PACController
from student import StudentModel
from teacher import TeacherModel


@dataclass
class AttackResult:
    queries_used: int
    divergence_detected: bool


def distill_from_teacher(
    teacher: TeacherModel,
    student: StudentModel,
    controller: PACController,
    x_queries: np.ndarray,
    learning_rate: float,
    steps: int,
    batch_size: int,
) -> AttackResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    n = x_queries.shape[0]
    divergence = False
    total_queries = 0

    for step in range(steps):
        start = (step * batch_size) % n
        end = min(start + batch_size, n)
        batch = x_queries[start:end]
        if batch.size == 0:
            continue

        teacher_probs = teacher.predict_proba(batch)
        teacher_probs = controller.process(teacher_probs)
        total_queries += batch.shape[0]

        student_logits = batch @ student.weights + student.bias
        student_logits = student_logits - np.max(student_logits, axis=1, keepdims=True)
        exp_logits = np.exp(student_logits)
        student_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        grad = (student_probs - teacher_probs) / batch.shape[0]
        grad_w = batch.T @ grad
        grad_b = np.sum(grad, axis=0)

        student.weights -= learning_rate * grad_w
        student.bias -= learning_rate * grad_b

        if not np.isfinite(student.weights).all() or not np.isfinite(student.bias).all():
            divergence = True
            break

    return AttackResult(queries_used=total_queries, divergence_detected=divergence)
