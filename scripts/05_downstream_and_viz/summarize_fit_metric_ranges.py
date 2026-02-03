#!/usr/bin/env python3
"""
Summarize error metric ranges from winners.csv across experiments and tags.

This script discovers winners.csv files produced by export_cbn_best_fits.py under
results/parameter_analysis/<experiment>/<tag>/winners.csv for a specific --tag
across all provided --experiments, and computes:

Outputs
-------
1) summary_by_experiment_prompt.csv
   - One row per (tag, experiment, prompt_category)
   - For each available metric, three columns: <metric>_min, <metric>_median, <metric>_max.

2) (removed) LaTeX table output is no longer produced.

3) long_by_experiment_prompt_agent.csv
   - Long-form rows per (tag, experiment, prompt_category, agent, metric)
   - Columns: value, group_min, group_median, group_max (grouped within the same tag/experiment/prompt_category).

Notes
-----
- A single exact --tag is used; outputs include a `tag` column.
- Metrics are detected from numeric columns typically emitted in winners.csv
  (e.g., mae, rmse, loss, loocv_r2, loocv_rmse, r2, r2_task, rmse_task, cv_r2). Non-numeric/ID columns are ignored.
- Domain is not part of the grouping; if winners.csv contains per-domain rows, they will be pooled across agents within
  the (experiment, prompt_category) condition. Pass --domains all when exporting winners to keep pooled rows if desired.

Example
-------
python scripts/05_downstream_and_viz/summarize_fit_metric_ranges.py \
    --experiments rw17_indep_causes random_abstract \
    --tag v2_noisy_or_2025_09_30 \
    --prompt-categories numeric cot \
    --output-dir results/parameter_analysis/cbn_fit_metric_analysis
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

# In staged location (scripts/05_downstream_and_viz), repo root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Canonicalization of prompt categories to ensure consistent filtering
_NUMERIC_SYNS = {
    "numeric",
    "num",
    "pcnum",
    "single_numeric",
    "single_numeric_response",
    "direct",
}
_COT_SYNS = {
    "cot",
    "pccot",
    "chain_of_thought",
    "chain-of-thought",
    "cot_stepwise",
}


def _canon_prompt_category(label: str) -> str:
    """Map various prompt label aliases to canonical values: 'numeric' or 'cot'.

    Unknown labels are returned lower-cased unchanged.
    """
    t = str(label).strip().lower()
    if t in _NUMERIC_SYNS:
        return "numeric"
    if t in _COT_SYNS:
        return "cot"
    return t


def find_winner_dirs_for_tag(experiments: List[str], tag: str) -> list[Path]:
    """Discover tag directories containing winners.csv for the given experiments and exact tag.

    Looks under results/parameter_analysis/<experiment>/<tag>/winners.csv.
    Returns list of tag directories (Path objects) where winners.csv exists.
    """
    found: list[Path] = []
    base = PROJECT_ROOT / "results" / "parameter_analysis"
    for exp in experiments:
        tag_dir = base / exp / tag
        if tag_dir.is_dir() and (tag_dir / "winners.csv").exists():
            found.append(tag_dir)
    return found


def _numeric_metrics(df: pd.DataFrame) -> list[str]:
    """Heuristically identify metric columns from winners.csv.

    We consider numeric columns and exclude obvious ID/meta fields.
    """
    if df.empty:
        return []
    # Known non-metric/meta columns
    exclude = {
        "link", "params_tying", "agent", "domain", "version", "prompt_category",
        "loss_name", "optimizer", "spec_hash", "short_spec_hash", "group_hash", "short_group_hash",
        "tag", "experiment",
    }
    candidates = []
    for c in df.columns:
        if c in exclude:
            continue
        # float-like numeric
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            candidates.append(c)
    # Keep common metric names first if present
    preferred_order = [
        "loocv_r2", "loocv_rmse", "r2_task", "rmse_task", "r2", "rmse", "mae", "loss", "cv_r2",
    ]
    ordered = [c for c in preferred_order if c in candidates]
    # Append any remaining numeric columns (stable order)
    for c in candidates:
        if c not in ordered:
            ordered.append(c)
    return ordered


def _fmt_num(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    try:
        xf = float(x)
    except Exception:
        return "--"
    if xf == 0.0:
        return f"{0.0:.{nd}f}"
    ax = abs(xf)
    if ax < 1e-3 or ax >= 1e3:
        return f"{xf:.2e}"
    return f"{xf:.{nd}f}"


def summarize(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Return wide summary with min/median/max per metric, grouped by available meta columns.

    Grouping keys: (tag, experiment, [prompt_category if present]).
    """
    if df.empty:
        return pd.DataFrame(columns=["tag", "experiment", "prompt_category"])
    group_cols = [c for c in ["tag", "experiment", "prompt_category"] if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols):
        # keys is scalar if one group col; normalize to tuple
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = {k: v for k, v in zip(group_cols, keys)}
        for m in metrics:
            s = pd.to_numeric(g[m], errors="coerce")
            s = s.dropna()
            if s.empty:
                row[f"{m}_min"] = np.nan
                row[f"{m}_median"] = np.nan
                row[f"{m}_max"] = np.nan
            else:
                row[f"{m}_min"] = float(np.min(s))
                row[f"{m}_median"] = float(np.median(s))
                row[f"{m}_max"] = float(np.max(s))
        rows.append(row)
    out = pd.DataFrame(rows)
    # Sort for readability
    return out.sort_values(group_cols).reset_index(drop=True)


def make_long(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Return long table per (tag, experiment, [prompt_category], agent, metric) with value and group ranges."""
    if df.empty:
        return pd.DataFrame(columns=["tag", "experiment", "prompt_category", "agent", "metric", "value", "group_min", "group_median", "group_max"])
    group_cols = [c for c in ["tag", "experiment", "prompt_category"] if c in df.columns]
    records = []
    # Pre-compute group stats per metric
    stats = {}
    for keys, g in df.groupby(group_cols):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        stats[key_tuple] = {}
        for m in metrics:
            s = pd.to_numeric(g[m], errors="coerce").dropna()
            if s.empty:
                stats[key_tuple][m] = (np.nan, np.nan, np.nan)
            else:
                stats[key_tuple][m] = (float(np.min(s)), float(np.median(s)), float(np.max(s)))
    # Emit rows per agent
    for _, row in df.iterrows():
        keys = tuple(row[c] for c in group_cols)
        # Map back literal fields for output (avoid relying on positional indices)
        tag_val = str(row.get("tag", ""))
        exp_val = str(row.get("experiment", ""))
        prompt_val = str(row.get("prompt_category", "")) if "prompt_category" in df.columns else ""
        agent = str(row.get("agent", ""))
        for m in metrics:
            val = pd.to_numeric(pd.Series([row.get(m)]), errors="coerce").iloc[0]
            gmin, gmed, gmax = stats.get(keys, {}).get(m, (np.nan, np.nan, np.nan))
            records.append({
                "tag": tag_val,
                "experiment": exp_val,
                "prompt_category": prompt_val,
                "agent": agent,
                "metric": m,
                "value": (float(val) if pd.notna(val) else np.nan),
                "group_min": gmin,
                "group_median": gmed,
                "group_max": gmax,
            })
    out = pd.DataFrame(records)
    sort_cols = [c for c in ["tag", "experiment", "prompt_category", "metric", "agent"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


# LaTeX output removed


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize range (min/median/max) of winners.csv metrics across experiments and tags")
    ap.add_argument("--experiments", nargs="*", required=True, help="Experiments to include (e.g., rw17_indep_causes random_abstract)")
    ap.add_argument("--tag", required=True, help="Exact tag under results/parameter_analysis/<experiment>/ to include")
    ap.add_argument("--prompt-categories", nargs="*", default=["numeric", "cot"], help="Prompt categories to include (filter on winners.csv column)")
    ap.add_argument("--output-dir", default="results/parameter_analysis/cbn_fit_metric_analysis", help="Directory to write outputs")
    ap.add_argument("--include-baseline", action="store_true", help="Also compute baseline medians & 95%% CI from random_init_metrics.parquet if available")
    args = ap.parse_args(argv)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_dirs = find_winner_dirs_for_tag(args.experiments, args.tag)
    if not tag_dirs:
        print("[warn] No tag directories with winners.csv found for the given scope.")
        return 0

    rows = []
    for tag_dir in tag_dirs:
        winners_path = tag_dir / "winners.csv"
        try:
            df = pd.read_csv(winners_path)
        except Exception as e:
            print(f"[warn] Failed to read {winners_path}: {e}")
            continue
        # Ensure required columns exist
        if "agent" not in df.columns:
            print(f"[warn] Missing 'agent' in {winners_path}; skipping")
            continue
        # Infer experiment from parent directory name and attach tag/experiment
        experiment = tag_dir.parent.name
        tag_value = tag_dir.name
        df["experiment"] = experiment
        df["tag"] = tag_value
        # Normalize and filter prompt categories with canonicalization
        if "prompt_category" in df.columns and args.prompt_categories:
            df["prompt_category"] = df["prompt_category"].astype(str).map(_canon_prompt_category)
            wanted = {_canon_prompt_category(str(x)) for x in args.prompt_categories}
            df = df[df["prompt_category"].isin(wanted)].copy()
        # Gather rows
        rows.append(df)

    if not rows:
        print("[warn] No data rows collected.")
        return 0

    full = pd.concat(rows, ignore_index=True)
    # Detect metrics present
    metrics = _numeric_metrics(full)
    if not metrics:
        print("[warn] No numeric metric columns detected in winners.csv files.")
        return 0

    # Build compact working set
    keep_cols = [c for c in ["tag", "experiment", "prompt_category", "agent"] if c in full.columns]
    keep_cols += metrics
    full_w: pd.DataFrame = pd.DataFrame(full.loc[:, keep_cols]).copy()

    # Summary by (tag, experiment, prompt)
    summary_df = summarize(full_w, metrics)
    # Long per-agent with group ranges
    long_df = make_long(full_w, metrics)

    # Write/append CSVs for winners summaries
    p_summary_csv = out_dir / "summary_by_experiment_prompt.csv"
    p_long_csv = out_dir / "long_by_experiment_prompt_agent.csv"
    if p_summary_csv.exists():
        prev = pd.read_csv(p_summary_csv)
        summary_out = pd.concat([prev, summary_df], ignore_index=True).drop_duplicates()
    else:
        summary_out = summary_df
    if p_long_csv.exists():
        prev_long = pd.read_csv(p_long_csv)
        long_out = pd.concat([prev_long, long_df], ignore_index=True).drop_duplicates()
    else:
        long_out = long_df
    summary_out.to_csv(p_summary_csv, index=False)
    long_out.to_csv(p_long_csv, index=False)
    print(f"[ok] Updated {p_summary_csv}")
    print(f"[ok] Updated {p_long_csv}")

    # Optionally compute baseline medians and 95% CI from random baselines
    if args.include_baseline:
        baseline_rows = []
        for exp in args.experiments:
            baseline_path = PROJECT_ROOT / "results" / "model_fitting" / exp / "random_baseline" / "random_init_metrics.parquet"
            if not baseline_path.exists():
                print(f"[warn] Baseline not found for {exp}: {baseline_path}")
                continue
            try:
                bdf = pd.read_parquet(baseline_path)
            except Exception as e:
                print(f"[warn] Failed to read baseline for {exp}: {e}")
                continue
            # Normalize prompt categories and filter
            if "prompt_category" in bdf.columns and args.prompt_categories:
                bdf = bdf.copy()
                bdf["prompt_category"] = bdf["prompt_category"].astype(str).map(_canon_prompt_category)
                wanted = {_canon_prompt_category(str(x)) for x in args.prompt_categories}
                bdf = bdf[bdf["prompt_category"].isin(wanted)]
            # Determine candidate baseline metrics (skip ids)
            exclude = {"agent", "prompt_category", "experiment", "version", "link", "params_tying", "loss_name", "draw_index", "seed", "included_domains", "run_timestamp", "n"}
            candidates = [c for c in bdf.columns if c not in exclude]
            # Keep metrics overlapping with winners first
            base_metrics = [m for m in metrics if m in candidates]
            # And include any other numeric candidates at the end
            for c in candidates:
                if c not in base_metrics:
                    # ensure numeric
                    s = pd.to_numeric(bdf[c], errors="coerce")
                    if s.notna().any():
                        base_metrics.append(c)
            # Group by experiment/prompt
            group_cols = [c for c in ["prompt_category"] if c in bdf.columns]
            if not group_cols:
                # No prompt column; compute a single pooled row
                bdf = bdf.copy()
                bdf["prompt_category"] = ""
                group_cols = ["prompt_category"]
            for keys, g in bdf.groupby(group_cols):
                prompt_val = keys if isinstance(keys, str) else keys[0]
                row = {"experiment": exp, "prompt_category": prompt_val}
                for m in base_metrics:
                    s = pd.to_numeric(g[m], errors="coerce").dropna()
                    if s.empty:
                        row[f"baseline_{m}_median"] = float("nan")
                        row[f"baseline_{m}_ci_low"] = float("nan")
                        row[f"baseline_{m}_ci_high"] = float("nan")
                    else:
                        # Median and 95% CI (2.5, 97.5 percentiles)
                        row[f"baseline_{m}_median"] = float(np.median(s))
                        row[f"baseline_{m}_ci_low"] = float(np.percentile(s, 2.5))
                        row[f"baseline_{m}_ci_high"] = float(np.percentile(s, 97.5))
                baseline_rows.append(row)
        if baseline_rows:
            bdf_out = pd.DataFrame(baseline_rows)
            p_base_csv = out_dir / "baseline_by_experiment_prompt.csv"
            if p_base_csv.exists():
                prev_b = pd.read_csv(p_base_csv)
                base_out = pd.concat([prev_b, bdf_out], ignore_index=True).drop_duplicates()
            else:
                base_out = bdf_out
            base_out.to_csv(p_base_csv, index=False)
            print(f"[ok] Updated {p_base_csv}")
        else:
            print("[warn] No baseline rows produced.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
