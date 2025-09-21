#!/usr/bin/env python3
"""Deprecated shim: forwards to scripts/03_analysis_raw/aggregate_cognitive_strategies.py"""
from pathlib import Path
import runpy
import sys


def main() -> None:
    target = Path(__file__).parent / "03_analysis_raw" / "aggregate_cognitive_strategies.py"
    sys.stderr.write(
        "[DEPRECATION] scripts/aggregate_cognitive_strategies.py moved to scripts/03_analysis_raw/aggregate_cognitive_strategies.py.\n"
    )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
