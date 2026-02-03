"""Deprecation shim: use scripts/03_analysis_raw/create_llm_coverage_csv.py"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve()
TARGET = HERE.parent / "03_analysis_raw" / "create_llm_coverage_csv.py"
print("[DEPRECATION] Please call scripts/03_analysis_raw/create_llm_coverage_csv.py; this wrapper will be removed.")
if not TARGET.exists():
    print(f"[ERROR] Canonical script missing: {TARGET}")
    sys.exit(1)
runpy.run_path(str(TARGET), run_name="__main__")

