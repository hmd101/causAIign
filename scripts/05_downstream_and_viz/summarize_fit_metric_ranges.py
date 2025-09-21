#!/usr/bin/env python3
"""
Summarize error metric ranges from winners.csv across experiments and tags.

This script discovers winners.csv files produced by export_cbn_best_fits.py under
results/parameter_analysis/<experiment>/<tag>/winners.csv (mirroring the
discovery approach used by cross-experiment aggregation utilities) and computes:

Outputs
-------
1) summary_by_experiment_prompt.csv
   - One row per (tag, experiment, prompt_category)
   - For each available metric, three columns: <metric>_min, <metric>_median, <metric>_max.

2) summary_by_experiment_prompt.tex
   - A LaTeX tabular summarizing min/median/max per metric for each (experiment, prompt_category), grouped by tag.

3) long_by_experiment_prompt_agent.csv
   - Long-form rows per (tag, experiment, prompt_category, agent, metric)
   - Columns: value, group_min, group_median, group_max (grouped within the same tag/experiment/prompt_category).

Notes
-----
- If multiple tags match --tag-glob, all are included; outputs include a `tag` column to disambiguate.
- Metrics are detected from numeric columns typically emitted in winners.csv
  (e.g., mae, rmse, loss, loocv_r2, loocv_rmse, r2, r2_task, rmse_task, cv_r2). Non-numeric/ID columns are ignored.
- Domain is not part of the grouping; if winners.csv contains per-domain rows, they will be pooled across agents within
  the (experiment, prompt_category) condition. Pass --domains all when exporting winners to keep pooled rows if desired.

Example
-------
python scripts/summarize_fit_cbn_fit_metric_analysis.py \
  --experiments rw17_indep_causes random_abstract \
  --tag-glob "v2_noisy_or_*" \
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


def find_winner_dirs(experiments: List[str], tag_glob: str) -> list[Path]:
    """Discover tag directories containing winners.csv for the given experiments.

    Looks under results/parameter_analysis/<experiment>/<tag_glob>/winners.csv.
    Returns list of tag directories (Path objects) where winners.csv exists.
    """
    found: list[Path] = []
    base = PROJECT_ROOT / "results" / "parameter_analysis"
    for exp in experiments:
        exp_dir = base / exp
        if not exp_dir.exists():
            continue
        for tag_dir in exp_dir.glob(tag_glob):
            if not tag_dir.is_dir():
                continue
            if (tag_dir / "winners.csv").exists():
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


def to_latex_table(summary_df: pd.DataFrame, metrics: list[str]) -> str:
    """Render a LaTeX table string with subcolumns (min/med/max) per metric.

    Columns: Tag, Experiment, Prompt, then for each metric: (min, med, max).
    """
    if summary_df.empty:
        return "% No data to render"
    # Build header (booktabs recommended)
    # Column spec: 3 for meta + 3 per metric (r columns)
    col_spec = "l l l " + " ".join(["r r r" for _ in metrics])
    lb = " \\"  # LaTeX line break
    lines: list[str] = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    # First header row: metric spans
    first = ["", "", ""]
    for m in metrics:
        first.append(f"\\multicolumn{{3}}{{c}}{{\\texttt{{{m}}}}}")
    lines.append(" & ".join(first) + lb)
    # Second header row: min/med/max labels
    second = ["\\textbf{Tag}", "\\textbf{Experiment}", "\\textbf{Prompt}"]
    for _ in metrics:
        second.extend(["min", "med", "max"])
    lines.append(" & ".join(second) + lb)
    lines.append("\\midrule")
    # Body
    for _, rr in summary_df.iterrows():
        vals = [str(rr.get("tag", "")), str(rr.get("experiment", "")), str(rr.get("prompt_category", ""))]
        for m in metrics:
            vals.append(_fmt_num(rr.get(f"{m}_min")))
            vals.append(_fmt_num(rr.get(f"{m}_median")))
            vals.append(_fmt_num(rr.get(f"{m}_max")))
        lines.append(" & ".join(vals) + lb)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize range (min/median/max) of winners.csv metrics across experiments and tags")
    ap.add_argument("--experiments", nargs="*", required=True, help="Experiments to include (e.g., rw17_indep_causes random_abstract)")
    ap.add_argument("--tag-glob", default="*", help="Glob to select tag directories under results/parameter_analysis/<experiment> (default: * )")
    ap.add_argument("--prompt-categories", nargs="*", default=["numeric", "cot"], help="Prompt categories to include (filter on winners.csv column)")
    ap.add_argument("--output-dir", default="results/parameter_analysis/cbn_fit_metric_analysis", help="Directory to write outputs")
    args = ap.parse_args(argv)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_dirs = find_winner_dirs(args.experiments, args.tag_glob)
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
        # Filter experiments/prompt categories if present
        # Infer experiment from parent directory name
        experiment = tag_dir.parent.name
        tag = tag_dir.name
        df["experiment"] = experiment
        df["tag"] = tag
        # Normalize and filter prompt categories case-insensitively (e.g., "CoT" -> "cot")
        if "prompt_category" in df.columns and args.prompt_categories:
            df["prompt_category"] = df["prompt_category"].astype(str).str.strip().str.lower()
            wanted = {str(x).strip().lower() for x in args.prompt_categories}
            df = df[df["prompt_category"].isin(wanted)].copy()
        # Keep only minimal columns plus metrics; gather rows
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
    full_w = full[keep_cols].copy()

    # Summary by (tag, experiment, prompt)
    summary_df = summarize(full_w, metrics)
    # Long per-agent with group ranges
    long_df = make_long(full_w, metrics)

    # Write CSVs
    p_summary_csv = out_dir / "summary_by_experiment_prompt.csv"
    p_long_csv = out_dir / "long_by_experiment_prompt_agent.csv"
    summary_df.to_csv(p_summary_csv, index=False)
    long_df.to_csv(p_long_csv, index=False)
    print(f"[ok] Wrote {p_summary_csv}")
    print(f"[ok] Wrote {p_long_csv}")

    # Write LaTeX table
    tex_str = to_latex_table(summary_df, metrics)
    p_summary_tex = out_dir / "summary_by_experiment_prompt.tex"
    with open(p_summary_tex, "w", encoding="utf-8") as f:
        f.write(tex_str + "\n")
    print(f"[ok] Wrote {p_summary_tex}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
