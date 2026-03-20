#!/usr/bin/env python3
"""Validate required project imports for CI self-healing setup."""

import importlib
import sys

REQUIRED = [
    "teacher",
    "student",
    "attack",
    "pac_controller",
    "scripts.run_attack_ci",
]


def main() -> int:
    for mod in REQUIRED:
        importlib.import_module(mod)
    print("Import validation succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
