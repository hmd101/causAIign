#!/usr/bin/env python3
"""
Wrapper: Run LLMs and process outputs (A→Z pipeline step 2)

This wrapper delegates to one script at a time. It does not chain the two steps.
Run the LLM step and the processing step as two separate invocations.

Delegates to:
- run_experiment.py (project root): runs prompts through selected LLM(s)
- scripts/run_data_pipeline.py: collects LLM outputs and optionally merges with humans

Examples:
  # Run LLMs (use run_experiment.py flags)
  python scripts/02_llm_and_processing/run_llm_prompts.py --delegate run_experiment -- --version 10 --experiment pilot_study --model gpt-4o

  # Process data later (use run_data_pipeline.py flags). This will scan outputs under data/output_llm/.
  python scripts/02_llm_and_processing/run_llm_prompts.py --delegate pipeline -- --experiment pilot_study --version 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
DELEGATES = {
    "run_experiment": ROOT / "run_experiment.py",
    "pipeline": ROOT / "scripts" / "run_data_pipeline.py",
}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Delegate to either run_experiment.py (LLM runs) or scripts/run_data_pipeline.py (processing). "
            "Note: this wrapper never runs both in a single call."
        )
    )
    ap.add_argument("--delegate", choices=DELEGATES.keys(), required=True, help="Which tool to run")
    ap.add_argument("--", dest="sep", action="store_true", help="Separator before delegated args (optional)")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the delegated tool")
    ns = ap.parse_args(argv)

    target = DELEGATES[ns.delegate]
    if not target.exists():
        sys.stderr.write(f"[error] Delegate not found: {target}\n")
        return 2
    # Strip optional leading separator
    deli_args = ns.args
    if deli_args and deli_args[0] == "--":
        deli_args = deli_args[1:]

    # Emulate command-line execution for the target script
    sys.argv = [str(target), *deli_args]
    runpy.run_path(str(target), run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
