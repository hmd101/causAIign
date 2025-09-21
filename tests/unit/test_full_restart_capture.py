from __future__ import annotations

from pathlib import Path
import pandas as pd

from causalign.analysis.model_fitting.trainer import FitConfig
from causalign.analysis.model_fitting.api import fit_dataset


def _df():
    return pd.DataFrame({
        "subject": ["agentX"] * 6,
        "prompt_category": ["pc"] * 6,
        "domain": ["d"] * 6,
        "task": ["I","II","III","IV","V","VI"],
        "response": [0.2,0.4,0.6,0.8,0.3,0.5],
    })


def test_full_restart_capture(tmp_path: Path):
    df = _df()
    cfg = FitConfig(epochs=5, restarts=4, lr=0.1, optimizer="lbfgs")
    results = fit_dataset(df, cfg, output_dir=tmp_path, per_restart_metrics=True)
    assert results, "No results returned"
    rec = results[0]
    restarts = rec.get("restarts") or []
    assert len(restarts) == cfg.restarts, "Did not capture all restarts"
    # Ensure per-restart metrics presence for first restart (rmse may be None if disabled)
    assert {"restart_index","seed","loss_final","params"}.issubset(restarts[0].keys())
    # Representative best_restart_index must be within range
    assert 0 <= rec.get("best_restart_index", -1) < cfg.restarts
