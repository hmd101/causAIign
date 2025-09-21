#!/usr/bin/env python3
"""Compatibility shim for the new canonical location.

New path: scripts/01_prompts/generate_experiment_prompts.py
This stub will execute the new script and print a gentle note.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "01_prompts" / "generate_experiment_prompts.py"


def main() -> None:
    if not TARGET.exists():
        sys.stderr.write(f"[error] New target not found: {TARGET}\n")
        sys.exit(1)
    print(
        "[note] generate_experiment_prompts moved to scripts/01_prompts/. Executing new path..."
    )
    # Execute the new script as if called directly
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()

