#!/usr/bin/env python3
"""
DEPRECATION SHIM
This script moved to scripts/03_analysis_raw/human_llm_alignment_correlation.py
Keeping this shim for backward compatibility only.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    here = Path(__file__).resolve()
    staged = here.parent / "03_analysis_raw" / "human_llm_alignment_correlation.py"
    if not staged.exists():
        sys.stderr.write(
            "[ERROR] Canonical script not found: scripts/03_analysis_raw/human_llm_alignment_correlation.py\n"
        )
        sys.exit(1)
    sys.stderr.write(
        "[DEPRECATION] scripts/human_llm_alignment_correlation.py moved to scripts/03_analysis_raw/human_llm_alignment_correlation.py.\n"
    )
    runpy.run_path(str(staged), run_name="__main__")


if __name__ == "__main__":
    main()