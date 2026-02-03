#!/usr/bin/env python3
from __future__ import annotations

"""
DEPRECATED: use scripts/02_llm_and_processing/run_llm_prompts.py

This shim forwards all arguments to the new wrapper and preserves behavior.
"""

from pathlib import Path
import runpy
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    target = root / "scripts" / "02_llm_and_processing" / "run_llm_prompts.py"
    if not target.exists():
        sys.stderr.write(f"[error] New wrapper not found: {target}\n")
        return 2
    sys.stderr.write(
        "[deprecated] scripts/02_llm_and_processing/run_llm_and_process.py is deprecated.\n"
        "            Please use scripts/02_llm_and_processing/run_llm_prompts.py instead.\n"
    )
    # Preserve arguments; just rewrite argv[0] to reflect the target script
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
