#!/usr/bin/env python3
"""
Wrapper: Generate prompts (A→Z pipeline step 1)

This is a convenience entrypoint that mirrors scripts/generate_experiment_prompts.py
but lives in the structured pipeline folder. It simply delegates to the original
script so existing docs/commands keep working.

Usage:
  python scripts/01_prompts/generate_prompts.py --help
  python scripts/01_prompts/generate_prompts.py --experiment pilot_study --version 5
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
TARGET = ROOT / "scripts" / "generate_experiment_prompts.py"

if __name__ == "__main__":
    if not TARGET.exists():
        sys.stderr.write(f"[error] Target script not found: {TARGET}\n")
        sys.exit(1)
    # Execute as if called directly; preserves CLI parsing in the target
    runpy.run_path(str(TARGET), run_name="__main__")
