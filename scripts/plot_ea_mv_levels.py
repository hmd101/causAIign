"""Deprecation shim: use scripts/05_downstream_and_viz/plot_ea_mv_levels.py"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve()
TARGET = HERE.parent / "05_downstream_and_viz" / "plot_ea_mv_levels.py"
print("[DEPRECATION] Please call scripts/05_downstream_and_viz/plot_ea_mv_levels.py; this wrapper will be removed.")
if not TARGET.exists():
    print(f"[ERROR] Canonical script missing: {TARGET}")
    sys.exit(1)
runpy.run_path(str(TARGET), run_name="__main__")
