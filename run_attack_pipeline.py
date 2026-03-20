"""Deterministic distillation attack pipeline for CI."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from attack import AttackConfig, run_extraction_attack
from teacher import TeacherConfig, TeacherModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def is_unstable(arr: np.ndarray) -> bool:
    return bool(np.isnan(arr).any() or np.isinf(arr).any())


def load_dataset(seed: int) -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    topic_terms = {
        0: ["orbit", "rocket", "nasa", "planet", "telescope", "lunar", "mission"],
        1: ["pitcher", "homerun", "inning", "stadium", "baseball", "batting", "league"],
        2: ["render", "gpu", "shader", "graphics", "texture", "pixel", "animation"],
        3: ["policy", "election", "senate", "voter", "debate", "government", "bill"],
    }
    cross_terms = ["analysis", "discussion", "update", "report", "signal", "trend"]

    texts: list[str] = []
    labels: list[int] = []
    for label, terms in topic_terms.items():
        for _ in range(520):
            topical = rng.choice(terms, size=6, replace=True).tolist()
            shared = rng.choice(cross_terms, size=4, replace=True).tolist()
            noise_label = int(rng.integers(0, 4))
            noise = rng.choice(topic_terms[noise_label], size=2, replace=True).tolist()
            sentence = " ".join(topical + shared + noise)
            texts.append(sentence)
            labels.append(label)
    labels_np = np.array(labels)

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        texts,
        labels_np,
        test_size=0.45,
        random_state=seed,
        stratify=labels,
    )
    x_query, x_test, y_query, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=y_tmp,
    )
    return x_train, y_train, x_test, y_test, x_query


def train_teacher_with_retries(
    x_train: list[str], y_train: np.ndarray, *, seed: int
) -> TeacherModel:
    attempts = [
        TeacherConfig(random_state=seed, learning_rate_init=0.01, max_iter=120),
        TeacherConfig(random_state=seed, learning_rate_init=0.005, max_iter=100),
        TeacherConfig(random_state=seed, learning_rate_init=0.003, max_iter=90, hidden_layer_sizes=(48,)),
    ]

    for config in attempts:
        teacher = TeacherModel(config).fit(x_train, y_train)
        probs = teacher.predict_proba(x_train[:64])
        if not is_unstable(probs):
            return teacher

    raise RuntimeError("Teacher training diverged after retries.")


def run_once(protected: bool, seed: int, max_queries: int) -> dict[str, float | bool | int]:
    set_seed(seed)
    x_train, y_train, x_test, y_test, x_query = load_dataset(seed)
    teacher = train_teacher_with_retries(x_train, y_train, seed=seed)

    student, pac_metrics = run_extraction_attack(
        teacher,
        x_query,
        AttackConfig(protected=protected, max_queries=max_queries),
        seed=seed,
    )

    teacher_test_probs = teacher.predict_proba(x_test)
    student_test_probs = student.predict_proba(x_test)
    if is_unstable(student_test_probs):
        raise RuntimeError("Student training diverged (NaN/Inf probabilities).")

    teacher_preds = np.argmax(teacher_test_probs, axis=1)
    student_preds = np.argmax(student_test_probs, axis=1)

    teacher_acc = accuracy_score(y_test, teacher_preds)
    student_acc = accuracy_score(y_test, student_preds)
    agreement = float(np.mean(teacher_preds == student_preds))
    kl = kl_divergence(teacher_test_probs, student_test_probs)

    result: dict[str, float | bool | int] = {
        "protected": protected,
        "teacher_accuracy": float(teacher_acc),
        "student_accuracy": float(student_acc),
        "agreement_rate": agreement,
        "kl_divergence": float(kl),
        **pac_metrics,
    }

    if not math.isfinite(result["pac_bound"]):
        raise RuntimeError("PAC-style bound is non-finite and cannot be computed.")

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["protected", "unprotected"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    protected = args.mode == "protected"

    try:
        result = run_once(protected=protected, seed=args.seed, max_queries=args.max_queries)
    except MemoryError as exc:
        raise RuntimeError("Memory error during pipeline; reduce batch/model size.") from exc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
