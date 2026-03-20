"""Robust CI entrypoint for extraction simulation with retries and adaptation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from attack import AttackConfig, compare_protection, run_extraction


def run_with_retries(base_cfg: AttackConfig, output_path: Path, max_attempts: int = 4):
    cfg = base_cfg
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            metrics = run_extraction(cfg, output_path)
            metrics["attempt"] = attempt
            output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            return metrics
        except FloatingPointError as err:
            last_error = err
            cfg = AttackConfig(
                **{
                    **cfg.__dict__,
                    "student_lr": max(0.03, cfg.student_lr * 0.5),
                    "student_steps": max(80, int(cfg.student_steps * 0.8)),
                }
            )
        except RuntimeError as err:
            if "out of memory" in str(err).lower() or "memory" in str(err).lower():
                last_error = err
                cfg = AttackConfig(
                    **{
                        **cfg.__dict__,
                        "batch_size": max(8, cfg.batch_size // 2),
                        "width_scale": max(0.5, cfg.width_scale * 0.8),
                    }
                )
                continue
            raise
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt == max_attempts:
                raise

    raise RuntimeError(f"Attack failed after retries: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    try:
        unprotected = run_with_retries(
            AttackConfig(name="unprotected", protected=False, leakage_budget=10_000.0),
            artifacts / "metrics_unprotected.json",
        )
        protected = run_with_retries(
            AttackConfig(name="protected", protected=True, leakage_budget=70.0),
            artifacts / "metrics_protected.json",
        )
        compare_protection(unprotected, protected)
    except Exception as err:  # noqa: BLE001
        failure = {
            "status": "failed",
            "error": str(err),
            "traceback": traceback.format_exc(),
        }
        (artifacts / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 1

    summary = {
        "status": "ok",
        "unprotected": unprotected,
        "protected": protected,
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
