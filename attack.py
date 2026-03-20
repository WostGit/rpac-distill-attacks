"""Query-only distillation attack experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

from dataset import build_splits
from pac_controller import PACController
from student import train_student
from teacher import train_teacher


@dataclass
class AttackResult:
    config: str
    student_accuracy: float
    teacher_student_agreement: float
    kl_divergence: float
    queries_used: int
    pac_bound: float
    enforcement_triggered: bool


def _kl_divergence(p: list[list[float]], q: list[list[float]]) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        row = 0.0
        for pv, qv in zip(pi, qi):
            a = max(pv, 1e-9)
            b = max(qv, 1e-9)
            row += a * (math.log(a) - math.log(b))
        total += row
    return total / max(len(p), 1)


def run_attack(
    pac_enabled: bool,
    seed: int = 1234,
    lr: float = 0.2,
    steps: int = 100,
    batch_size: int = 32,
    model_dim: int = 512,
) -> AttackResult:
    rng = random.Random(seed)
    splits = build_splits(seed=seed)
    teacher = train_teacher(splits.x_train, splits.y_train, seed=seed)

    sample_probs = teacher.predict_proba([splits.x_train[0]])[0]
    controller = PACController(enabled=pac_enabled, num_classes=len(sample_probs), leakage_threshold=8.0 if pac_enabled else 1e9)

    queried_probs = []
    for txt in splits.x_query:
        probs = teacher.predict_proba([txt])[0]
        queried_probs.append(controller.process(probs, rng))

    student = train_student(
        query_texts=splits.x_query,
        teacher_probs=queried_probs,
        seed=seed,
        lr=lr,
        steps=steps,
        batch_size=batch_size,
        model_dim=model_dim,
    )

    teacher_test_probs = teacher.predict_proba(splits.x_test)
    student_test_probs = student.predict_proba(splits.x_test)
    teacher_preds = [max(range(len(p)), key=lambda i: p[i]) for p in teacher_test_probs]
    student_preds = [max(range(len(p)), key=lambda i: p[i]) for p in student_test_probs]

    accuracy = sum(int(p == y) for p, y in zip(student_preds, splits.y_test)) / len(splits.y_test)
    agreement = sum(int(a == b) for a, b in zip(student_preds, teacher_preds)) / len(teacher_preds)
    kl = _kl_divergence(teacher_test_probs, student_test_probs)

    bound = controller.pac_bound()
    if not math.isfinite(bound):
        raise ValueError("PAC bound is not finite")

    return AttackResult(
        config="protected" if pac_enabled else "unprotected",
        student_accuracy=accuracy,
        teacher_student_agreement=agreement,
        kl_divergence=kl,
        queries_used=controller.queries,
        pac_bound=bound,
        enforcement_triggered=controller.enforcement_triggered,
    )


def to_json_dict(result: AttackResult) -> dict:
    return asdict(result)
