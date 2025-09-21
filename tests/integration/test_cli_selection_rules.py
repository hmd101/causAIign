from __future__ import annotations

import json
import re
from pathlib import Path
from subprocess import run

DATA_VERSION = "2"
EXPERIMENT = "rw17_indep_causes"
AGENT = "gpt-4o"
PROMPT_CATEGORY = "single_numeric_response"


def _run_cli(tmp_path: Path, extra: list[str]):
    out_dir = tmp_path / "model_fitting" / EXPERIMENT
    base_cmd = [
        "python","-m","causalign.analysis.model_fitting.cli",
        "--version", DATA_VERSION,
        "--experiment", EXPERIMENT,
        "--model","logistic",
        "--params","3",
        "--optimizer","lbfgs",
        "--epochs","3",
        "--restarts","5",
        "--agents", AGENT,
        "--prompt-categories", PROMPT_CATEGORY,
        "--loss","mse",
        "--no-roman-numerals",
        "--no-aggregated",
        "--temperature","0.0",
        "--per-restart-metrics",
        "--output-dir", str(out_dir),
    ] + extra
    proc = run(base_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"CLI failed code={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    effective_dir = out_dir / EXPERIMENT if (out_dir / EXPERIMENT).exists() else out_dir
    hashed_pattern = re.compile(r"fit_[0-9a-f]{12}_[0-9a-f]{12}\.json")
    files = list(effective_dir.iterdir())
    hashed = [f for f in files if hashed_pattern.fullmatch(f.name)]
    assert hashed, "Structured hashed JSON not found"
    with open(hashed[0]) as fh:
        data = json.load(fh)
    return data


def test_cli_selection_rule_median_loss(tmp_path: Path):
    data = _run_cli(tmp_path, ["--selection-rule","median_loss"])
    restarts = data.get("restarts") or []
    assert len(restarts) == 5, "Expected 5 restarts"
    losses = sorted([(r.get("loss_final"), r.get("restart_index")) for r in restarts], key=lambda t: t[0])
    median_idx = losses[len(losses)//2][1]
    assert data.get("best_restart_index") == median_idx, "median_loss selection did not pick median restart"


def test_cli_selection_rule_best_primary_metric_rmse(tmp_path: Path):
    data = _run_cli(tmp_path, ["--primary-metric","rmse","--selection-rule","best_primary_metric"])
    restarts = data.get("restarts") or []
    rmses = [(r.get("rmse"), r.get("restart_index")) for r in restarts if r.get("rmse") is not None]
    assert rmses, "Missing per-restart rmse values"
    best_rmse_idx = min(rmses, key=lambda t: t[0])[1]
    assert data.get("best_restart_index") == best_rmse_idx, "best_primary_metric did not pick lowest rmse restart"
