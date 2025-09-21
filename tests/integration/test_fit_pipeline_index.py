"""Lightweight regression test for model fitting indexing pipeline.

This test runs a minimal fit (few epochs, one restart) on a small subset
and asserts that:
  * Legacy JSON result file is created.
  * New structured hashed JSON (with spec_hash & group_hash) is created.
  * Parquet index is updated and contains a row matching those hashes.
  * Key metric columns (rmse, aic, bic) are present and not null.

It protects against regressions where the CLI stops writing the new schema or
fails to append/update the Parquet index after a refactor.

Requires that processed data for version '2' and experiment 'rw17_indep_causes'
exists (as in current repository fixture/data). If data layout changes, adjust
constants at top.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from subprocess import run, CompletedProcess

DATA_VERSION = "2"
EXPERIMENT = "rw17_indep_causes"
AGENT = "gpt-4o"
PROMPT_CATEGORY = "single_numeric_response"


def _run_cli(tmpdir: Path) -> CompletedProcess:
    cmd = [
        "python",
        "-m",
        "causalign.analysis.model_fitting.cli",
        "--version",
        DATA_VERSION,
        "--experiment",
        EXPERIMENT,
        "--model",
        "logistic",
        "--params",
        "3",
        "--optimizer",
        "lbfgs",
        "--epochs",
        "3",  # keep fast
        "--restarts",
        "1",
        "--agents",
        AGENT,
        "--prompt-categories",
        PROMPT_CATEGORY,
        "--loss",
        "mse",
        "--enable-loocv",
        "--no-roman-numerals",
        "--no-aggregated",
        "--temperature",
        "0.0",
        "--output-dir",
        str(tmpdir),
    ]
    # Use check=True later; here capture output for debugging on failure.
    return run(cmd, capture_output=True, text=True)


def test_fit_creates_index_and_hashed_result(tmp_path: Path):
    # CLI builds: <output_dir>/<experiment>/ ... we pass out_dir as base, so results end up under out_dir/EXPERIMENT
    # In some configurations we observed an extra experiment nesting; handle both.
    out_dir = tmp_path / "model_fitting" / EXPERIMENT
    proc = _run_cli(out_dir)
    if proc.returncode != 0:
        raise AssertionError(
            f"CLI failed (code {proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # Legacy JSON pattern: fit_<agent>_<promptCat>_logistic_3p_lr*.json
    legacy_pattern = re.compile(rf"fit_{AGENT}_{PROMPT_CATEGORY}_logistic_3p_lr.+\.json")
    # Allow for potential extra nesting of experiment name
    effective_dir = out_dir / EXPERIMENT if (out_dir / EXPERIMENT).exists() else out_dir
    files = list(effective_dir.iterdir())
    legacy_json = [f for f in files if legacy_pattern.match(f.name)]
    assert legacy_json, "Legacy JSON result not found"

    # Hashed JSON pattern: fit_<12hex>_<12hex>.json (spec short + group short)
    hashed_pattern = re.compile(r"fit_[0-9a-f]{12}_[0-9a-f]{12}\.json")
    hashed_json = [f for f in files if hashed_pattern.fullmatch(f.name) and f not in legacy_json]
    assert hashed_json, "Hashed JSON result not found"

    # Load hashed JSON and extract hashes
    with open(hashed_json[0]) as fh:
        data = json.load(fh)
    spec_hash = data.get("spec_hash")
    group_hash = data.get("group_hash")
    assert spec_hash and group_hash, "spec_hash/group_hash missing in structured JSON"

    # Index parquet must exist
    index_path = effective_dir / "fit_index.parquet"
    assert index_path.exists(), "fit_index.parquet not created"
    idx = pd.read_parquet(index_path)

    # Find matching row
    row = idx[(idx.spec_hash == spec_hash) & (idx.group_hash == group_hash)]
    assert len(row) == 1, "Index does not contain exactly one matching row for hashes"

    rec = row.iloc[0]
    for col in ["rmse", "aic", "bic"]:
        assert pd.notna(rec[col]), f"Metric {col} is NaN in index row"

    # Ensure basic columns present for provenance
    expected_cols = {"spec_hash", "group_hash", "agent", "prompt_category", "link", "loss_name"}
    missing = expected_cols - set(idx.columns)
    assert not missing, f"Missing expected columns in index: {missing}"

    # Quick check that data_hash is stable-looking (length 64 hex)
    dh = rec.get("data_hash")
    assert isinstance(dh, str) and re.fullmatch(r"[0-9a-f]{64}", dh), "data_hash not a 64-hex string"
