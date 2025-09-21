#!/usr/bin/env python3
from pathlib import Path
from runpy import run_path
import sys

_HERE = Path(__file__).resolve()
_CANON = _HERE.parent / "03_analysis_raw" / "domain_differences.py"

if __name__ == "__main__":
    sys.exit(run_path(str(_CANON), run_name="__main__").get("__return__", 0))
