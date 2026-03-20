#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.special import rel_entr
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from attack import distill_from_teacher
from pac_controller import PACController
from student import init_student
from teacher import train_teacher


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_data(seed: int):
    x, y = make_classification(
        n_samples=4000,
        n_features=32,
        n_informative=16,
        n_redundant=4,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=seed,
    )
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    x_train, x_query, y_train, _ = train_test_split(
        x_temp, y_temp, test_size=0.375, random_state=seed, stratify=y_temp
    )
    return x_train, y_train, x_query, x_test, y_test


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(rel_entr(p, q), axis=1)))


def run(config: str, timeout_s: int, out_json: Path) -> int:
    seed = 1337
    set_deterministic(seed)

    x_train, y_train, x_query, x_test, y_test = build_data(seed)
    teacher = train_teacher(x_train, y_train, seed)

    protected = config == "protected"
    controller = PACController(
        enabled=protected,
        leakage_budget=240.0,
        leakage_rate=0.5,
        noise_std=0.02 if protected else 0.0,
        clip_min=1e-5,
        clip_max=1.0,
    )

    lr = 0.35
    steps = 120
    batch_size = 64

    start_time = __import__("time").time()
    max_retries = 4
    attempt = 0
    last_err = None
    while attempt < max_retries:
        attempt += 1
        try:
            student = init_student(x_query.shape[1], len(np.unique(y_train)), seed + attempt)
            result = distill_from_teacher(
                teacher=teacher,
                student=student,
                controller=controller,
                x_queries=x_query,
                learning_rate=lr,
                steps=steps,
                batch_size=batch_size,
            )
            if result.divergence_detected:
                raise FloatingPointError("student training diverged (non-finite params)")
            break
        except MemoryError as exc:
            last_err = exc
            batch_size = max(8, batch_size // 2)
        except FloatingPointError as exc:
            last_err = exc
            lr = max(0.01, lr * 0.5)
            steps = max(40, int(steps * 0.8))
    else:
        print(f"ERROR: unrecoverable training failure after retries: {last_err}", file=sys.stderr)
        return 42

    elapsed = __import__("time").time() - start_time
    if elapsed > timeout_s:
        print("ERROR: attack pipeline exceeded CI time budget", file=sys.stderr)
        return 43

    teacher_probs = teacher.predict_proba(x_test)
    student_probs = student.predict_proba(x_test)
    teacher_pred = np.argmax(teacher_probs, axis=1)
    student_pred = np.argmax(student_probs, axis=1)

    student_acc = float(accuracy_score(y_test, student_pred))
    agreement = float(np.mean(teacher_pred == student_pred))
    kl = kl_divergence(teacher_probs, student_probs)

    try:
        pac_bound = controller.pac_bound(n_queries=max(1, result.queries_used))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not compute PAC-style bound: {exc}", file=sys.stderr)
        return 44

    output = {
        "config": config,
        "seed": seed,
        "student_accuracy": student_acc,
        "teacher_student_agreement": agreement,
        "teacher_student_kl": kl,
        "queries_used": int(result.queries_used),
        "pac_bound": float(pac_bound),
        "enforcement_triggered": bool(controller.enforcement_triggered),
        "train_retries": attempt - 1,
        "learning_rate_final": lr,
        "steps_final": steps,
        "batch_size_final": batch_size,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["unprotected", "protected"], required=True)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return run(config=args.config, timeout_s=args.timeout_s, out_json=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
