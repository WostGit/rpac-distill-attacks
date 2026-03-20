#!/usr/bin/env python3
"""CI runner for robust extraction simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack import run_attack, to_json_dict


def execute_with_retries(pac_enabled: bool, seed: int, out_path: Path) -> dict:
    lr = 0.2
    steps = 100
    batch_size = 32
    model_dim = 512

    max_attempts = 4
    last_error = None
    for _ in range(max_attempts):
        try:
            result = run_attack(
                pac_enabled=pac_enabled,
                seed=seed,
                lr=lr,
                steps=steps,
                batch_size=batch_size,
                model_dim=model_dim,
            )
            payload = to_json_dict(result)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload
        except FloatingPointError as exc:
            last_error = exc
            lr *= 0.5
            steps = max(30, int(steps * 0.8))
        except MemoryError as exc:
            last_error = exc
            batch_size = max(8, batch_size // 2)
            model_dim = max(128, model_dim // 2)
        except Exception as exc:  # noqa: BLE001 - CI resilience path
            last_error = exc
            break
    raise RuntimeError(f"attack pipeline failed after retries: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    unprotected = execute_with_retries(False, args.seed, artifacts / "unprotected_metrics.json")
    protected = execute_with_retries(True, args.seed, artifacts / "protected_metrics.json")

    summary = {
        "unprotected": unprotected,
        "protected": protected,
        "protected_reduces_extraction": protected["teacher_student_agreement"] < unprotected["teacher_student_agreement"],
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "pac_bound" not in protected or protected["pac_bound"] is None:
        raise SystemExit("PAC bound cannot be computed")

    if not summary["protected_reduces_extraction"]:
        raise SystemExit("Protected configuration did not reduce extraction success")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
