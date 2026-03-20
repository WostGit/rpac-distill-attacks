#!/usr/bin/env python3
"""Compare protected vs unprotected extraction metrics and enforce CI gates."""

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    unp = load("outputs/unprotected/metrics.json")
    pro = load("outputs/protected/metrics.json")

    issues: list[str] = []
    if "pac_bound" not in unp or "pac_bound" not in pro:
        issues.append("PAC-style bound missing")
    if not isinstance(pro.get("enforcement_triggered"), bool):
        issues.append("Protected run missing enforcement status")
    if pro["teacher_student_agreement"] >= unp["teacher_student_agreement"]:
        issues.append("Protected run did not reduce extraction success (agreement)")
    if pro["teacher_student_kl"] <= unp["teacher_student_kl"]:
        issues.append("Protected run did not increase divergence vs baseline")

    if issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 55

    print("Protection validation passed")
    print("unprotected:", json.dumps(unp, sort_keys=True))
    print("protected:", json.dumps(pro, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
