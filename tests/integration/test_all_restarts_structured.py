"""Integration test verifying all restarts are captured in structured JSON.

The pipeline now always records all restarts (former --record-all-restarts flag
removed). Runs a small fit with >1 restarts and inspects the hashed structured
JSON file to assert that len(restarts) equals the requested restart count. Uses
minimal epochs for speed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from subprocess import run

DATA_VERSION = "2"
EXPERIMENT = "rw17_indep_causes"
AGENT = "gpt-4o"
PROMPT_CATEGORY = "single_numeric_response"


def test_structured_includes_all_restarts(tmp_path: Path):
    out_dir = tmp_path / "model_fitting" / EXPERIMENT
    cmd = [
        "python","-m","causalign.analysis.model_fitting.cli",
        "--version", DATA_VERSION,
        "--experiment", EXPERIMENT,
        "--model","logistic",
        "--params","3",
        "--optimizer","lbfgs",
        "--epochs","3",
        "--restarts","3",
        "--agents", AGENT,
        "--prompt-categories", PROMPT_CATEGORY,
        "--loss","mse",
        "--no-roman-numerals",
        "--no-aggregated",
        "--temperature","0.0",
        "--output-dir", str(out_dir),
    ]
    proc = run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"CLI failed code={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    effective_dir = out_dir / EXPERIMENT if (out_dir / EXPERIMENT).exists() else out_dir
    hashed_pattern = re.compile(r"fit_[0-9a-f]{12}_[0-9a-f]{12}\.json")
    files = list(effective_dir.iterdir())
    hashed = [f for f in files if hashed_pattern.fullmatch(f.name)]
    assert hashed, "Structured hashed JSON not found"
    with open(hashed[0]) as fh:
        data = json.load(fh)
    restarts = data.get("restarts")
    assert isinstance(restarts, list), "restarts not a list"
    assert len(restarts) == 3, f"Expected 3 restarts, found {len(restarts)}"
    # Ensure aggregate fields included in Parquet index (restart_count) by loading index parquet
    import pandas as pd
    idx_path = effective_dir / "fit_index.parquet"
    assert idx_path.exists(), "fit_index.parquet missing"
    idx = pd.read_parquet(idx_path)
    row = idx.sort_values("restart_count", ascending=False).iloc[0]
    assert row.restart_count >= 3, "restart_count column not reflecting captured restarts"
