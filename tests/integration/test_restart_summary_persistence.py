from __future__ import annotations

import json
import re
from pathlib import Path
from subprocess import run

import pandas as pd

DATA_VERSION = "2"
EXPERIMENT = "rw17_indep_causes"
AGENT = "gpt-4o"
PROMPT_CATEGORY = "numeric"


def test_restart_summary_in_json_and_index(tmp_path: Path):
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
    rs = data.get("restart_summary")
    assert rs and rs.get("count") == 3, "restart_summary missing or count incorrect in JSON"
    for key in ["loss_mean","loss_median","primary_metric"]:
        assert key in rs, f"{key} missing in restart_summary JSON"

    # Index check
    idx_path = effective_dir / "fit_index.parquet"
    assert idx_path.exists(), "fit_index.parquet missing"
    idx = pd.read_parquet(idx_path)
    # Ensure new columns are present
    for col in ["restart_loss_mean","restart_loss_median","primary_metric","primary_selection_rule"]:
        assert col in idx.columns, f"Index column {col} missing"
    # At least one row should have restart_count==3
    assert (idx.restart_count == 3).any(), "Index restart_count not reflecting captured restarts"
