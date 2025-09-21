from __future__ import annotations

from pathlib import Path
import pandas as pd

from causalign.analysis.model_fitting.trainer import FitConfig
from causalign.analysis.model_fitting.api import fit_dataset
from causalign.analysis.model_fitting.specs import RankingPolicy


def _make_df(noise: float = 0.05):
    # Create synthetic tasks where higher r2 corresponds to lower noise in responses.
    tasks = ["I", "II", "III", "IV", "V"]
    base = [0.1, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for seed, jitter in enumerate([noise, noise*2, noise*3, noise*4, noise*5]):
        for t, b in zip(tasks, base):
            # Larger jitter -> poorer fit -> smaller r2 after training, expect selection to pick smallest jitter
            resp = min(1.0, max(0.0, b + (jitter * ((seed*3 + len(t)) % 5 - 2) / 4)))
            rows.append({"subject": "agentA", "task": t, "response": resp})
    return pd.DataFrame(rows)


def test_best_primary_metric_r2(tmp_path: Path):
    df = _make_df()
    cfg = FitConfig(epochs=30, restarts=5, optimizer="adam", loss_name="mse", num_params=3, link="logistic")
    out_dir = tmp_path / "fits"
    policy = RankingPolicy(primary_metric="r2", selection_rule="best_primary_metric")
    records = fit_dataset(df, cfg, output_dir=out_dir, per_restart_metrics=True, ranking_policy=policy, primary_metric="r2")
    assert records, "No records returned"
    rec = records[0]
    restarts = rec["restarts"]
    r2_values = [(r.get("r2"), r.get("restart_index")) for r in restarts if r.get("r2") is not None]
    assert r2_values, "Missing r2 metrics"
    best_idx = max(r2_values, key=lambda t: t[0])[1]
    assert rec["best_restart_index"] == best_idx, "Selection rule did not pick restart with highest r2"
