#!/usr/bin/env python3
"""Compatibility shim: script moved to scripts/04_cbn_fit_and_eval/export_cbn_best_fits.py"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "04_cbn_fit_and_eval" / "export_cbn_best_fits.py"


def main() -> None:
    if not TARGET.exists():
        sys.stderr.write(f"[error] New target not found: {TARGET}\n")
        sys.exit(1)
    print("[note] export_cbn_best_fits moved to scripts/04_cbn_fit_and_eval/. Executing new path...")
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
