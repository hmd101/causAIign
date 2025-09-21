from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

from causalign.analysis.model_fitting.trainer import FitConfig
from causalign.analysis.model_fitting.api import fit_dataset


def _df(n=8):
    # Use only supported tasks (I-VIII) to match task mapping
    tasks = ["I","II","III","IV","V","VI","VII","VIII"][:n]
    vals = [0.1,0.3,0.5,0.7,0.2,0.4,0.6,0.8][:n]
    return pd.DataFrame({
        "subject": ["agentU"] * n,
        "prompt_category": ["pc"] * n,
        "domain": ["d"] * n,
        "task": tasks,
        "response": vals,
    })


def test_uncertainty_block_present_and_reasonable(tmp_path: Path):
    df = _df()
    # Use LBFGS for more reliable local curvature capture
    cfg = FitConfig(epochs=25, restarts=2, lr=0.1, optimizer="lbfgs", compute_uncertainty=True)
    res = fit_dataset(df, cfg, output_dir=tmp_path / "unc", per_restart_metrics=False)[0]
    u = res.get("uncertainty")
    assert u and u.get("method") == "gauss_newton", "uncertainty block missing or wrong method"
    se = u.get("se") or {}
    # Allow fallback where SEs could be empty but then must have warning recorded
    if not se:
        warnings = u.get("warnings") or []
        assert warnings, "Expected warnings when SE dict empty"
    else:
        for k, v in se.items():
            assert isinstance(v, (int, float)) and v >= 0.0 and math.isfinite(v), f"Invalid SE for {k}: {v}"
    cond = u.get("condition_number")
    if cond is not None:
        assert cond > 0, "Condition number should be positive"


def test_uncertainty_absent_when_flag_false(tmp_path: Path):
    df = _df(8)
    cfg = FitConfig(epochs=20, restarts=2, lr=0.2, optimizer="adam", compute_uncertainty=False)
    res = fit_dataset(df, cfg, output_dir=tmp_path / "unc_off", per_restart_metrics=False)[0]
    assert res.get("uncertainty") is None, "Uncertainty block should be None when flag disabled"
