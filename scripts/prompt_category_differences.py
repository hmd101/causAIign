#!/usr/bin/env python3
"""Deprecated shim: forwards to scripts/03_analysis_raw/prompt_category_differences.py"""
from pathlib import Path
import runpy
import sys


def main() -> None:
    target = Path(__file__).parent / "03_analysis_raw" / "prompt_category_differences.py"
    sys.stderr.write(
        "[DEPRECATION] scripts/prompt_category_differences.py moved to scripts/03_analysis_raw/prompt_category_differences.py.\n"
    )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
