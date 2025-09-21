from __future__ import annotations

from pathlib import Path
import pandas as pd

from causalign.analysis.model_fitting.trainer import FitConfig
from causalign.analysis.model_fitting.api import fit_dataset
from causalign.analysis.model_fitting.specs import RankingPolicy


def _df():
    # Provide enough rows to get variation across restarts
    return pd.DataFrame({
        "subject": ["agentZ"] * 8,
        "prompt_category": ["pc"] * 8,
        "domain": ["d"] * 8,
        "task": ["I","II","III","IV","V","VI","VII","VIII"],
        "response": [0.15,0.35,0.55,0.75,0.25,0.45,0.65,0.85],
    })


def _loss_list(rec):
    return [(r.get("loss_final"), int(r.get("restart_index"))) for r in rec.get("restarts")]


def test_selection_rule_best_and_median(tmp_path: Path):
    df = _df()
    cfg = FitConfig(epochs=10, restarts=5, lr=0.1, optimizer="lbfgs")

    # best_loss rule
    rp_best = RankingPolicy(primary_metric="aic", selection_rule="best_loss")
    res_best = fit_dataset(df, cfg, output_dir=tmp_path / "best", ranking_policy=rp_best, per_restart_metrics=True)[0]
    losses = _loss_list(res_best)
    min_loss, min_idx = min(losses, key=lambda t: t[0])
    assert res_best.get("best_restart_index") == min_idx, "best_loss rule did not pick minimal loss restart"

    # median_loss rule
    rp_median = RankingPolicy(primary_metric="aic", selection_rule="median_loss")
    res_median = fit_dataset(df, cfg, output_dir=tmp_path / "median", ranking_policy=rp_median, per_restart_metrics=True)[0]
    losses_med = sorted(_loss_list(res_median), key=lambda t: t[0])
    mid = len(losses_med)//2
    median_idx = losses_med[mid][1]
    assert res_median.get("best_restart_index") == median_idx, "median_loss rule did not pick median loss restart"


def test_selection_rule_best_primary_metric(tmp_path: Path):
    df = _df()
    cfg = FitConfig(epochs=10, restarts=5, lr=0.1, optimizer="lbfgs")
    rp_primary = RankingPolicy(primary_metric="rmse", selection_rule="best_primary_metric")
    res = fit_dataset(df, cfg, output_dir=tmp_path / "primary", ranking_policy=rp_primary, per_restart_metrics=True)[0]
    restarts = res.get("restarts") or []
    # Collect rmse values (ensure they exist because per_restart_metrics=True)
    rmse_list = [(r.get("rmse"), int(r.get("restart_index"))) for r in restarts if r.get("rmse") is not None]
    assert rmse_list, "Per-restart rmse metrics missing"
    best_rmse_idx = min(rmse_list, key=lambda t: t[0])[1]
    assert res.get("best_restart_index") == best_rmse_idx, "best_primary_metric rule did not pick lowest primary metric restart"
