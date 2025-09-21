from __future__ import annotations
"""Build a multi-metric overview of model fits.

Reads the unified fit index (fit_index.parquet) plus optional long-form
restart metrics/params (restart_metrics.parquet, restart_params.parquet) in a
results/model_fitting/<experiment>/ directory and produces:
  - overview.csv : one row per (agent,prompt_category,domain,metric) pointing to best spec
  - leaderboard_wide.csv : single row per (agent,prompt_category,domain) with best spec for each metric
  - stability.csv : restart dispersion stats for each (spec_hash, group_hash)

Selection logic (per metric):
  1. Consider only rows with non-null metric value.
  2. Lower-is-better metrics: loss,aic,bic,rmse. Higher-is-better: r2.
  3. Tie-breakers: simpler params_tying (lower), then lower loss, then earlier spec_hash lexicographically.

Stability metrics are sourced from restart_* columns in index; if missing we
fall back to recomputing from long-form restart_metrics.

Parameter dispersion (uncertainty proxy) is computed from restart_params if
available: per parameter std/iqr/range then aggregated (mean/std across params).
"""
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict

LOWER_IS_BETTER = {"loss", "aic", "bic", "rmse"}
HIGHER_IS_BETTER = {"r2"}
ALL_METRICS = ["loss", "aic", "bic", "rmse", "r2"]


def _choose_best(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = metric
    if col not in df.columns:
        return pd.DataFrame(columns=list(df.columns)+["target_metric","target_value"])
    sub = df[df[col].notna()].copy()
    if sub.empty:
        return pd.DataFrame(columns=list(df.columns)+["target_metric","target_value"])
    ascending = metric in LOWER_IS_BETTER
    sub.sort_values(
        by=[col, "params_tying", "loss", "spec_hash"],
        ascending=[ascending, True, True, True],
        inplace=True,
        kind="mergesort",  # stable
    )
    best = sub.groupby(["agent","prompt_category","domain"], dropna=False).head(1).copy()
    best["target_metric"] = metric
    best["target_value"] = best[col]
    return best


def _compute_param_dispersion(restart_params: pd.DataFrame) -> pd.DataFrame:
    if restart_params.empty:
        return pd.DataFrame()
    # Per (spec_hash,group_hash,param) dispersion
    agg = restart_params.groupby(["spec_hash","group_hash","param_name"]).agg(
        value_std=("value","std"),
        value_iqr=("value", lambda s: s.quantile(0.75)-s.quantile(0.25)),
        value_range=("value", lambda s: s.max()-s.min()),
    ).reset_index()
    # Aggregate across params
    agg2 = agg.groupby(["spec_hash","group_hash"]).agg(
        param_std_mean=("value_std","mean"),
        param_std_max=("value_std","max"),
        param_iqr_mean=("value_iqr","mean"),
        param_range_mean=("value_range","mean"),
    ).reset_index()
    return agg2


def build_overview(base_dir: Path, experiment: str) -> Dict[str, Path]:
    root = base_dir / "results" / "model_fitting" / experiment
    index_path = root / "fit_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    df_idx = pd.read_parquet(index_path)

    # Load long-form (optional)
    metrics_long = pd.DataFrame()
    params_long = pd.DataFrame()
    if (root / "restart_metrics.parquet").exists():
        metrics_long = pd.read_parquet(root / "restart_metrics.parquet")
    if (root / "restart_params.parquet").exists():
        params_long = pd.read_parquet(root / "restart_params.parquet")

    # Multi-metric BEST table
    best_rows: List[pd.DataFrame] = []
    for m in ALL_METRICS:
        if m in df_idx.columns:
            best_rows.append(_choose_best(df_idx, m))
    best_all = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    overview_path = root / "overview.csv"
    if not best_all.empty:
        cols_export = [
            "agent","prompt_category","domain","target_metric","target_value","spec_hash","short_spec_hash","link","params_tying","optimizer","lr","loss","aic","bic","rmse","r2","restart_loss_mean","restart_loss_std","restart_loss_iqr","restart_rmse_mean","restart_rmse_var","selection_rule","primary_metric_name"
        ]
        # Add synthesized std/iqr if present via restart_summary
        if "restart_loss_var" in df_idx.columns and "restart_loss_std" not in best_all.columns:
            best_all["restart_loss_std"] = best_all["restart_loss_var"].pow(0.5)
        if "restart_loss_range" not in best_all.columns and "restart_loss_min" in best_all.columns and "restart_loss_max" in best_all.columns:
            best_all["restart_loss_range"] = best_all["restart_loss_max"] - best_all["restart_loss_min"]
        best_all.to_csv(overview_path, index=False)
    else:
        best_all.to_csv(overview_path, index=False)

    # Stability table (per spec/group)
    stability_cols = [c for c in df_idx.columns if c.startswith("restart_")]
    stability = df_idx[["spec_hash","group_hash","agent","prompt_category","domain"] + stability_cols].copy()

    # Parameter dispersion
    param_disp = _compute_param_dispersion(params_long) if not params_long.empty else pd.DataFrame()
    if not param_disp.empty:
        stability = stability.merge(param_disp, on=["spec_hash","group_hash"], how="left")

    stability_path = root / "stability.csv"
    stability.to_csv(stability_path, index=False)

    # Leaderboard wide: pivot best rows so each metric chooses a spec
    leaderboard = None
    if not best_all.empty:
        wide_parts = []
        for m in ALL_METRICS:
            sub = best_all[best_all["target_metric"] == m][["agent","prompt_category","domain","spec_hash"]].copy()
            sub = sub.rename(columns={"spec_hash": f"best_spec_{m}"})
            wide_parts.append(sub)
        if wide_parts:
            from functools import reduce
            leaderboard = reduce(lambda l,r: l.merge(r, on=["agent","prompt_category","domain"], how="outer"), wide_parts)
    leaderboard_path = root / "leaderboard_wide.csv"
    if leaderboard is not None:
        leaderboard.to_csv(leaderboard_path, index=False)
    else:
        pd.DataFrame().to_csv(leaderboard_path, index=False)

    return {
        "overview": overview_path,
        "stability": stability_path,
        "leaderboard": leaderboard_path,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build multi-metric overview of model fits")
    ap.add_argument("--experiment", default="pilot_study")
    ap.add_argument("--base-dir", help="Project base directory (default: inferred)")
    args = ap.parse_args(argv)
    base = Path(args.base_dir) if args.base_dir else Path.cwd()
    paths = build_overview(base, args.experiment)
    print("Wrote:")
    for k,p in paths.items():
        print(f"  {k}: {p}")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
