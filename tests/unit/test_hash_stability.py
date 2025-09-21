from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from causalign.analysis.model_fitting.trainer import FitConfig
from causalign.analysis.model_fitting.api import fit_dataset
from causalign.analysis.model_fitting.adapters import compute_spec_hash, compute_group_hash


def test_spec_and_group_hash_stability(tmp_path: Path):
    # Minimal synthetic dataset (single agent group, no prompt_category/domain columns)
    df = pd.DataFrame({
        "subject": ["agent1"] * 5,
        "task": ["I", "II", "III", "IV", "V"],
        "response": [0.1, 0.3, 0.5, 0.7, 0.9],
    })
    cfg = FitConfig(epochs=10, restarts=2, optimizer="adam", loss_name="mse", num_params=3, link="logistic")
    out_dir = tmp_path / "fits"
    records = fit_dataset(df, cfg, output_dir=out_dir, per_restart_metrics=False)
    assert records, "No records produced"

    # Locate saved JSON file
    json_files = list(out_dir.glob("fit_*_*.json"))
    assert json_files, "Structured JSON file not found"
    with open(json_files[0]) as f:
        data = json.load(f)

    # Recompute spec hash from persisted spec
    recomputed_spec_hash, recomputed_short = compute_spec_hash(data["spec"])
    assert recomputed_spec_hash == data["spec_hash"], "Spec hash mismatch (instability detected)"
    assert recomputed_short == data["short_spec_hash"], "Short spec hash mismatch"

    # Recompute group hash (no domain list, no prompt category, temperature=None, extra=None)
    agent = data["group_key"].get("agent")
    recomputed_group_hash, recomputed_short_group = compute_group_hash(agent, [], None, None, None)
    assert recomputed_group_hash == data["group_hash"], "Group hash mismatch"
    assert recomputed_short_group == data["short_group_hash"], "Short group hash mismatch"
