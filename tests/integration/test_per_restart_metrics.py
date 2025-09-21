"""Integration test verifying per-restart metrics are included when flag enabled.

Runs a small fit with multiple restarts and the --per-restart-metrics flag. Asserts that each
restart record contains expected metric keys (rmse, mae, r2, aic, bic, ece_10bin) and that
aggregate index still writes successfully. Keeps epochs small for speed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from subprocess import run

import pandas as pd

DATA_VERSION = "2"
EXPERIMENT = "rw17_indep_causes"
AGENT = "gpt-4o"
PROMPT_CATEGORY = "single_numeric_response"

METRIC_KEYS = {"rmse", "mae", "r2", "aic", "bic", "ece_10bin"}


def test_per_restart_metrics_included(tmp_path: Path):
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
        "--per-restart-metrics",
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
    assert isinstance(restarts, list) and restarts, "restarts list missing or empty"
    for r in restarts:
        missing = METRIC_KEYS - set(r.keys())
        assert not missing, f"Missing per-restart metric keys: {missing}"
        # Basic sanity: metrics should be numeric (None would indicate computation skipped)
        for k in METRIC_KEYS:
            assert isinstance(r[k], (int, float)), f"Metric {k} not numeric: {r[k]}"

    # Index parquet must still exist
    idx_path = effective_dir / "fit_index.parquet"
    assert idx_path.exists(), "fit_index.parquet missing"
    idx = pd.read_parquet(idx_path)
    # Ensure the row has restart_count >= 3
    assert (idx.restart_count >= 3).any(), "restart_count in index not reflecting restarts"
