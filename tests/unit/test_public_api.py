from __future__ import annotations

from pathlib import Path
import pandas as pd

from causalign.analysis.model_fitting.trainer import FitConfig
from causalign.analysis.model_fitting.api import (
    fit_dataset,
    load_index,
    load_result_by_hash,
    rank_index,
    top_by_metric,
    query_top_by_primary_metric,
)


def _dummy_df():
    return pd.DataFrame({
        "subject": ["agentA"] * 5,
        "prompt_category": ["cat1"] * 5,
        "domain": ["dom1"] * 5,
        "task": ["I", "II", "III", "IV", "V"],
        "response": [0.1, 0.2, 0.9, 0.4, 0.7],
    })


def test_fit_dataset_and_loaders(tmp_path: Path):
    df = _dummy_df()
    cfg = FitConfig(epochs=5, restarts=3, lr=0.1, optimizer="lbfgs")
    results = fit_dataset(df, cfg, output_dir=tmp_path, per_restart_metrics=True)
    assert results, "No results returned"
    idx = load_index(tmp_path)
    assert not idx.empty
    # Restart aggregate columns should exist
    for col in [
        "restart_count","restart_loss_mean","restart_loss_median","restart_rmse_mean","restart_aic_mean"
    ]:
        assert col in idx.columns
    rec = results[0]
    loaded = load_result_by_hash(tmp_path, rec["spec_hash"], rec["group_hash"])
    assert loaded is not None
    assert loaded["spec_hash"] == rec["spec_hash"]
    assert loaded["group_hash"] == rec["group_hash"]

    # Ranking helpers
    ranked = rank_index(idx, "aic")
    assert "aic_rank" in ranked.columns
    top = top_by_metric(idx, "aic")
    assert len(top) == 1
    qtop = query_top_by_primary_metric(tmp_path, "aic")
    assert len(qtop) == 1
