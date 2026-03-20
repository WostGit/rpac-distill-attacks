"""Extraction attack runner and metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from data import load_dataset, vectorize_texts
from pac_controller import PACConfig, PACController
from student import StudentConfig, StudentModel
from teacher import TeacherConfig, TeacherModel


@dataclass
class AttackConfig:
    name: str
    seed: int = 123
    protected: bool = False
    leakage_budget: float = 90.0
    student_lr: float = 0.35
    student_steps: int = 260
    batch_size: int = 64
    width_scale: float = 1.0
    max_seconds: int = 540


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def run_extraction(cfg: AttackConfig, output_path: Path) -> Dict[str, float | bool | int | str]:
    import time

    start = time.time()
    ds, vocab = load_dataset(seed=cfg.seed)

    x_train = vectorize_texts(ds.train_texts, vocab)
    x_query = vectorize_texts(ds.query_texts, vocab)
    x_test = vectorize_texts(ds.test_texts, vocab)

    n_classes = int(ds.train_labels.max()) + 1

    teacher = TeacherModel(input_dim=x_train.shape[1], n_classes=n_classes, seed=cfg.seed)
    teacher.train_supervised(x_train, ds.train_labels, TeacherConfig(seed=cfg.seed))

    controller = PACController(
        n_classes=n_classes,
        cfg=PACConfig(enabled=cfg.protected, leakage_budget=cfg.leakage_budget),
        seed=cfg.seed + 1,
    )

    teacher_responses = np.zeros((x_query.shape[0], n_classes), dtype=np.float32)
    for i in range(x_query.shape[0]):
        if time.time() - start > cfg.max_seconds:
            raise TimeoutError("Attack pipeline exceeded CI time budget")
        probs = teacher.predict_proba(x_query[i : i + 1])[0]
        teacher_responses[i] = controller.process(probs)

    student = StudentModel(
        input_dim=x_query.shape[1],
        n_classes=n_classes,
        seed=cfg.seed + 2,
        width_scale=cfg.width_scale,
    )
    student.train_distill(
        x_query,
        teacher_responses,
        StudentConfig(lr=cfg.student_lr, steps=cfg.student_steps, seed=cfg.seed + 3),
        batch_size=cfg.batch_size,
    )

    teacher_test_probs = teacher.predict_proba(x_test)
    student_test_probs = student.predict_proba(x_test)
    teacher_pred = np.argmax(teacher_test_probs, axis=1)
    student_pred = np.argmax(student_test_probs, axis=1)

    pac_bound = controller.pac_bound()

    metrics: Dict[str, float | bool | int | str] = {
        "config": cfg.name,
        "protected": cfg.protected,
        "student_accuracy": _accuracy(ds.test_labels, student_pred),
        "teacher_student_agreement": _accuracy(teacher_pred, student_pred),
        "teacher_student_kl": _kl_divergence(teacher_test_probs, student_test_probs),
        "query_count": int(controller.query_count),
        "pac_bound": float(pac_bound),
        "enforcement_triggered": bool(controller.enforcement_triggered),
        "cumulative_leakage": float(controller.cumulative_leakage),
        "attack_config": asdict(cfg),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def compare_protection(unprotected: Dict[str, float | bool | int | str], protected: Dict[str, float | bool | int | str]) -> None:
    if not math.isfinite(float(protected["pac_bound"])):
        raise RuntimeError("PAC bound cannot be computed for protected configuration")

    unprot_agreement = float(unprotected["teacher_student_agreement"])
    prot_agreement = float(protected["teacher_student_agreement"])
    if prot_agreement >= unprot_agreement - 0.01:
        raise RuntimeError(
            "Protected configuration did not reduce extraction success "
            f"(agreement protected={prot_agreement:.4f}, unprotected={unprot_agreement:.4f})"
        )
