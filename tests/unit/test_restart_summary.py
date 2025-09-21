from __future__ import annotations

import pandas as pd
from pathlib import Path

from causalign.analysis.model_fitting.trainer import FitConfig
from causalign.analysis.model_fitting.api import fit_dataset
from causalign.analysis.model_fitting.specs import RankingPolicy


def _df():
    return pd.DataFrame({
        "subject": ["agentZ"] * 6,
        "prompt_category": ["pc"] * 6,
        "domain": ["d"] * 6,
        "task": ["I","II","III","IV","V","VI"],
        "response": [0.1,0.3,0.5,0.7,0.2,0.4],
    })


def test_restart_summary_basic_stats(tmp_path: Path):
    df = _df()
    # Use more restarts to get non-trivial variance; keep epochs small for speed
    cfg = FitConfig(epochs=6, restarts=5, lr=0.1, optimizer="lbfgs")
    rp = RankingPolicy(selection_rule="best_loss")
    res = fit_dataset(df, cfg, output_dir=tmp_path / "summ", ranking_policy=rp, per_restart_metrics=False)[0]
    summary = res.get("restart_summary")
    assert summary and summary.get("count") == 5, "restart_summary missing or wrong count"
    # Loss summary presence
    for k in ["loss_mean", "loss_median"]:
        assert k in summary, f"{k} missing in restart_summary"
        assert summary[k] is None or isinstance(summary[k], (int, float))
    # Variance should be None only if degenerate (all equal); allow either but key must exist
    assert "loss_var" in summary
    # Duration fields present (may be None if trainer didn't capture durations)
    assert "duration_mean" in summary and "duration_sum" in summary
    # Primary metric metadata always present
    assert summary.get("primary_metric") == rp.primary_metric


def test_restart_summary_with_primary_metric(tmp_path: Path):
    df = _df()
    cfg = FitConfig(epochs=6, restarts=5, lr=0.1, optimizer="lbfgs")
    # Collect per-restart metrics so primary stats can be computed for rmse
    rp = RankingPolicy(primary_metric="rmse", selection_rule="best_primary_metric")
    res = fit_dataset(df, cfg, output_dir=tmp_path / "summ_rmse", ranking_policy=rp, per_restart_metrics=True)[0]
    summary = res.get("restart_summary")
    assert summary and summary.get("count") == 5
    # Primary metric stats should exist when metric collected
    for k in ["primary_mean", "primary_median"]:
        assert k in summary, f"Missing {k} in restart_summary"
        assert summary[k] is None or isinstance(summary[k], (int, float))
    # primary_metric field should echo configuration
    assert summary.get("primary_metric") == "rmse"
