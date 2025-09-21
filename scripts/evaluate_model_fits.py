#!/usr/bin/env python3
"""Evaluate CBN fit results using the structured index (fit_index.parquet).

Reads per-experiment `fit_index.parquet` files, merges optional `spec_manifest.csv`
for friendly model labels, filters, ranks, and writes summary CSV outputs:
    - combined_index.csv: filtered slice of the unified index
    - best_by_group.csv: one row per group (arg --group-by) with best model (by --metric)
    - ranks_by_group.csv: long form ranks of each model within each group

Metrics are drawn from columns in the index (produced during fitting). Legacy
`summary.csv` parsing and on-the-fly heatmap generation have been removed –
plotting should be handled by dedicated visualization scripts.

Full parameter example (showing most options):

    python scripts/evaluate_model_fits.py \
            --experiments rw17_indep_causes abstract_reasoning \
            --versions 2 3 \
            --agents gpt-4o claude-3-opus humans \
            --domains weather economy health \
            --prompt-categories numeric numeric-conf \
            --models logistic noisy_or \
            --param-counts 3 4 5 \
            --metric aic \
            --group-by experiment version prompt_category agent domain \
            --output-dir results/modelfits/combined_eval

Minimal example (discover everything automatically):

    python scripts/evaluate_model_fits.py

Selecting LOOCV metric (if present in index):

    python scripts/evaluate_model_fits.py --metric loocv_rmse --group-by experiment agent
"""
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import argparse
import pandas as pd
from causalign.analysis.model_fitting.discovery import (
    load_index,
    load_spec_manifest,
    merge_index_with_manifest,
    best_rows,
    ranks_long,
    _LOWER_IS_BETTER,
    _HIGHER_IS_BETTER,
)


def _load_combined_index(base_dir: Path, experiments: Optional[List[str]]) -> pd.DataFrame:
    exps = experiments
    if not exps:
        root = base_dir / "results" / "model_fitting"
        if not root.exists():
            return pd.DataFrame()
        exps = [p.name for p in root.iterdir() if (p / "fit_index.parquet").exists()]
    frames = []
    for exp in exps:
        df = load_index(base_dir, exp)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _augment_with_manifest(base_dir: Path, experiments: List[str], df_index: pd.DataFrame) -> pd.DataFrame:
    man_frames = []
    for exp in experiments:
        mf = load_spec_manifest(base_dir, exp)
        if mf is not None and not mf.empty:
            man_frames.append(mf)
    if not man_frames:
        return df_index
    manifest_df = pd.concat(man_frames, ignore_index=True)
    return merge_index_with_manifest(df_index, manifest_df)


def _filter_df(
    df: pd.DataFrame,
    versions: Optional[List[str]],
    agents: Optional[List[str]],
    domains: Optional[List[str]],
    prompt_categories: Optional[List[str]],
    models: Optional[List[str]],
    param_counts: Optional[List[str]],
) -> pd.DataFrame:
    """Apply simple inclusion filters based on provided lists (if not None)."""
    out = df.copy()
    if versions is not None and "version" in out.columns:
        out = out[out["version"].astype(str).isin([str(v) for v in versions])]
    if agents is not None and "agent" in out.columns:
        out = out[out["agent"].isin(agents)]
    if domains is not None and "domain" in out.columns:
        out = out[out["domain"].isin(domains)]
    if prompt_categories is not None and "prompt_category" in out.columns:
        out = out[out["prompt_category"].isin(prompt_categories)]
    if models is not None and "link" in out.columns:
        out = out[out["link"].isin(models)]
    if param_counts is not None and "params_tying" in out.columns:
        out = out[out["params_tying"].astype(str).isin(param_counts)]
    return out


def _compose_model_label(row: pd.Series) -> str:
    friendly = row.get("friendly_name")
    if isinstance(friendly, str) and friendly:
        return friendly
    return f"{row.get('link','?')}_{row.get('params_tying','?')}p"


def compute_best_and_ranks(df: pd.DataFrame, metric: str, group_by: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["model_label"] = work.apply(_compose_model_label, axis=1)
    if "loss_name" in work.columns and (work["loss_name"].nunique() > 1 or work["optimizer"].nunique() > 1):
        work["model_label"] = work.apply(lambda r: f"{r['model_label']}-{r['loss_name']}-{r['optimizer']}", axis=1)
    best_df = best_rows(work, metric, group_by)
    ranks_df = ranks_long(work, metric, group_by, label_col="model_label")
    return best_df, ranks_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate structured CBN fit artifacts")
    p.add_argument("--experiments", nargs="*", help="Experiments to include (default: all discovered)")
    p.add_argument("--experiment", dest="experiments", nargs="*", help="Alias for --experiments")
    p.add_argument("--versions", nargs="*", help="Version(s) to include")
    p.add_argument("--version", dest="versions", nargs="*", help="Alias for --versions")
    p.add_argument("--agents", nargs="*", help="Agents to include")
    p.add_argument("--agent", dest="agents", nargs="*", help="Alias for --agents")
    p.add_argument("--domains", nargs="*", help="Domains to include")
    p.add_argument("--prompt-categories", nargs="*", help="Prompt categories to include")
    p.add_argument("--models", nargs="*", choices=["logistic", "noisy_or"], help="Model link types to include")
    p.add_argument("--param-counts", nargs="*", choices=["3", "4", "5"], help="Parameter tying counts")
    p.add_argument(
        "--metric",
        default="aic",
        choices=[
            "loss","aic","bic","rmse","mae","r2",
            "cv_rmse","cv_r2",  # backwards compatibility aliases if present
            "loocv_rmse","loocv_mae","loocv_r2","loocv_consistency","loocv_calibration","loocv_bias"
        ],
        help="Metric column to rank by",
    )
    p.add_argument("--group-by", nargs="*", default=["experiment","version","prompt_category","agent","domain"], help="Grouping columns for best/ranks")
    p.add_argument("--output-dir", help="Output directory (default results/modelfits)")
    p.add_argument("--plot-metric", action="store_true", help="Save a PNG distribution (box+strip) of metric by model_label")
    p.add_argument("--print-top", type=int, help="Print top N rows globally by metric")
    # Stability flag options
    p.add_argument("--stability-metric", default="loss", choices=["loss","rmse","aic","bic","mae","r2"], help="Restart summary metric prefix used for stability (restart_<metric>_*)")
    p.add_argument("--stability-cv-threshold", type=float, default=0.02, help="Max coefficient of variation (std/|mean|) to consider stable")
    p.add_argument("--stability-range-threshold", type=float, default=0.10, help="Max relative range ((max-min)/|mean|) to consider stable")
    p.add_argument("--stability-require-all", action="store_true", help="Require BOTH CV and relative range under thresholds (default: either)")
    p.add_argument("--stability-epsilon", type=float, default=1e-8, help="Small value to avoid division by zero when computing CV/relative range")
    p.add_argument("--skip-stability", action="store_true", help="Skip deriving restart summaries and computing stability flags (stable_opt, stability_cv, stability_rel_range)")
    p.add_argument("--include-params", action="store_true", help="Include best-restart parameter values as columns when printing top rows")
    p.add_argument("--list-metrics", action="store_true", help="List available metric columns after filtering and exit")
    p.add_argument("--show-selection-metadata", action="store_true", help="Display unique (primary_metric, selection_rule) tuples present")
    p.add_argument("--list-learning-rates", action="store_true", help="List distinct learning rates (lr column) after filtering and exit")
    return p


def _ensure_output_dir(base_dir: Path, output_dir: Optional[str]) -> Path:
    if output_dir:
        out = Path(output_dir)
    else:
        out = base_dir / "results" / "modelfits"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _compute_stability_flags(df: pd.DataFrame, metric: str, cv_thr: float, range_thr: float, require_all: bool, epsilon: float) -> pd.DataFrame:
    """Compute optimization stability flags using restart summary stats present in index.

    Expects columns like restart_<metric>_mean, restart_<metric>_std, restart_<metric>_min, restart_<metric>_max.
    Stability heuristics:
      cv = std / |mean| (guarding zero/NaN)
      rel_range = (max - min) / |mean|
    A row is flagged stable_opt = True if (cv <= cv_thr) OR (rel_range <= range_thr) unless --stability-require-all, in which case both must hold.
    Rows lacking required columns get stable_opt = NaN.
    """
    prefix = metric if metric != "loss" else "loss"
    mean_col = f"restart_{prefix}_mean"
    std_col = f"restart_{prefix}_std"
    min_col = f"restart_{prefix}_min"
    max_col = f"restart_{prefix}_max"
    df = df.copy()
    if all(c in df.columns for c in [mean_col, std_col, min_col, max_col]):
        # Coerce potentially object / None-containing columns to numeric; invalid parse -> NaN
        for c in [mean_col, std_col, min_col, max_col]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        mean_vals = df[mean_col]
        std_vals = df[std_col]
        min_vals = df[min_col]
        max_vals = df[max_col]
        # Denominator with epsilon floor; mean may be NaN -> stays NaN (will propagate to cv/rel_range)
        denom = mean_vals.abs().where(mean_vals.abs() > epsilon, other=epsilon)
        cv = std_vals / denom
        rel_range = (max_vals - min_vals) / denom
        # Determine validity mask: need mean/std/min/max not NaN
        valid = mean_vals.notna() & std_vals.notna() & min_vals.notna() & max_vals.notna()
        # Comparisons produce False where NaN; enforce NA explicitly for invalid rows
        cond_cv = (cv <= cv_thr) & valid
        cond_range = (rel_range <= range_thr) & valid
        if require_all:
            stable = (cond_cv & cond_range).where(valid, other=pd.NA)
        else:
            stable = (cond_cv | cond_range).where(valid, other=pd.NA)
        df["stability_cv"] = cv.where(valid, other=pd.NA)
        df["stability_rel_range"] = rel_range.where(valid, other=pd.NA)
        df["stable_opt"] = stable
        # Optional debug if any coercions produced NaN where original value was not NaN/NA like None
        if (mean_vals.isna() & df[mean_col].isna()).any():  # simple hook; can be expanded
            pass
    else:
        missing = [c for c in [mean_col, std_col, min_col, max_col] if c not in df.columns]
        df["stable_opt"] = pd.NA
        df["stability_cv"] = pd.NA
        df["stability_rel_range"] = pd.NA
        if missing:
            # Informative print for user; kept lightweight.
            print(f"Info: Cannot compute stability flag – missing columns: {missing}")
    return df


def _locate_fit_json(base_dir: Path, experiment: str, row: pd.Series) -> Optional[Path]:
    """Best-effort location of the fit JSON for a row using short hashes if available."""
    root = base_dir / "results" / "model_fitting" / experiment
    short_spec = row.get("short_spec_hash") or row.get("short_spec")
    short_group = row.get("short_group_hash") or row.get("short_group")
    patterns = []
    if short_spec and short_group:
        patterns.append(f"fit_{short_spec}_{short_group}.json")
    # Fallback broad search
    patterns.append("fit_*.json")
    for pat in patterns:
        matches = list(root.rglob(pat))
        if not matches:
            continue
        if short_spec and short_group and pat.startswith("fit_") and pat.endswith(".json") and len(matches) > 1:
            # try to narrow by containing both hashes
            narrowed = [m for m in matches if short_spec in m.name and short_group in m.name]
            if narrowed:
                matches = narrowed
        # choose most recent
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]
    return None


def _extract_best_restart_params(fit_path: Path) -> Dict[str, float]:
    try:
        js = json.loads(fit_path.read_text())
    except Exception:
        return {}
    restarts = js.get("restarts") or []
    if not restarts:
        return {}
    best = min(restarts, key=lambda r: r.get("loss_final", float("inf")))
    params = best.get("params") or {}
    out: Dict[str, float] = {}
    for k, v in params.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = Path(__file__).parent.parent
    out_dir = _ensure_output_dir(base_dir, args.output_dir)
    index_df = _load_combined_index(base_dir, args.experiments)
    if index_df.empty:
        print("No fit_index.parquet files discovered.")
        return 1
    exps = args.experiments or sorted(index_df["experiment"].unique())
    index_df = _augment_with_manifest(base_dir, exps, index_df)
    filtered = _filter_df(
        index_df,
        versions=args.versions,
        agents=args.agents,
        domains=args.domains,
        prompt_categories=args.prompt_categories,
        models=args.models,
        param_counts=args.param_counts,
    )
    # Establish metric name early for validation
    metric_name = args.metric
    if args.show_selection_metadata:
        meta_cols = [c for c in ["primary_metric","primary_metric_name","primary_selection_rule","selection_rule"] if c in filtered.columns]
        if meta_cols:
            print("Selection metadata (unique rows):")
            print(filtered[meta_cols].drop_duplicates().to_string(index=False))
        else:
            print("No selection metadata columns present in index.")
    if metric_name not in filtered.columns:
        print(f"Warning: metric '{metric_name}' not present in index columns: {sorted(filtered.columns)}")
        return 1

    # Drop rows with NA metric
    metric_series = filtered[metric_name]
    if metric_series.isna().all():
        print(f"Metric '{args.metric}' is entirely NA after filtering; nothing to rank/plot.")
        return 1
    filtered_non_na = filtered[~metric_series.isna()].copy()

    if not args.skip_stability:
        # Fallback: if restart summary columns for chosen stability metric are missing, attempt to derive them
        stability_prefix = args.stability_metric if args.stability_metric != "loss" else "loss"
        needed_cols = [f"restart_{stability_prefix}_{s}" for s in ["mean","std","min","max","iqr","var","median","range"]]
        missing_summary = any(col not in filtered_non_na.columns for col in needed_cols if col not in [f"restart_{stability_prefix}_median", f"restart_{stability_prefix}_range"])  # median/range may not be critical
        if missing_summary:
            rm_frames = []
            experiments_present = sorted(filtered_non_na["experiment"].dropna().unique()) if "experiment" in filtered_non_na.columns else []
            for exp in experiments_present:
                exp_root = base_dir / "results" / "model_fitting" / exp
                if not exp_root.exists():
                    continue
                cand_paths = []
                root_file = exp_root / "restart_metrics.parquet"
                if root_file.exists():
                    cand_paths.append(root_file)
                for sub in exp_root.glob("lr*/restart_metrics.parquet"):
                    cand_paths.append(sub)
                for rp in cand_paths:
                    try:
                        rm_frames.append(pd.read_parquet(rp))
                    except Exception:
                        pass
            if rm_frames:
                rm_all = pd.concat(rm_frames, ignore_index=True)
                metric_field_map = {"loss": "loss_final"}
                metric_field = metric_field_map.get(stability_prefix, stability_prefix)
                if metric_field in rm_all.columns and "spec_hash" in rm_all.columns:
                    grp = rm_all.groupby("spec_hash")[metric_field]
                    agg_df = grp.agg(["mean","std","min","max",lambda s: s.quantile(0.75)-s.quantile(0.25),"var","median"])
                    agg_df.columns = ["mean","std","min","max","iqr","var","median"]
                    agg_df["range"] = agg_df["max"] - agg_df["min"]
                    rename_map = {c: f"restart_{stability_prefix}_{c}" for c in agg_df.columns}
                    agg_df.rename(columns=rename_map, inplace=True)
                    agg_df.reset_index(inplace=True)
                    for col in list(agg_df.columns):
                        if col != 'spec_hash' and col in filtered_non_na.columns:
                            agg_df.drop(columns=[col], inplace=True)
                    if 'spec_hash' in filtered_non_na.columns:
                        filtered_non_na = filtered_non_na.merge(agg_df, on="spec_hash", how="left")
                        print(f"Info: Derived restart summary columns for stability metric '{stability_prefix}' from long-form restarts.")
                else:
                    print(f"Info: Could not derive restart summaries (missing column '{metric_field}' or 'spec_hash').")
            else:
                print("Info: No restart_metrics.parquet files found for deriving stability summaries.")
        else:
            var_c = f"restart_{stability_prefix}_var"
            std_c = f"restart_{stability_prefix}_std"
            max_c = f"restart_{stability_prefix}_max"
            min_c = f"restart_{stability_prefix}_min"
            iqr_c = f"restart_{stability_prefix}_iqr"
            range_c = f"restart_{stability_prefix}_range"
            need_std = std_c not in filtered_non_na.columns and var_c in filtered_non_na.columns
            need_extrema = any(c not in filtered_non_na.columns for c in [min_c, max_c, iqr_c, range_c])
            if need_std or need_extrema:
                specs_needed = filtered_non_na['spec_hash'].unique().tolist()
                rm_frames_specs = []
                experiments_present = sorted(filtered_non_na['experiment'].dropna().unique()) if 'experiment' in filtered_non_na.columns else []
                for exp in experiments_present:
                    exp_root = base_dir / 'results' / 'model_fitting' / exp
                    cand_paths = []
                    root_file = exp_root / 'restart_metrics.parquet'
                    if root_file.exists():
                        cand_paths.append(root_file)
                    for sub in exp_root.glob('lr*/restart_metrics.parquet'):
                        cand_paths.append(sub)
                    for rp in cand_paths:
                        try:
                            tmp = pd.read_parquet(rp)
                            tmp = tmp[tmp.spec_hash.isin(specs_needed)]
                            if not tmp.empty:
                                rm_frames_specs.append(tmp[['spec_hash','loss_final']])
                        except Exception:
                            pass
                if rm_frames_specs:
                    rm_all_sp = pd.concat(rm_frames_specs, ignore_index=True)
                    grp_loss = rm_all_sp.groupby('spec_hash')['loss_final']
                    if need_std:
                        std_series = grp_loss.std(ddof=0)
                        filtered_non_na = filtered_non_na.merge(std_series.rename(std_c), on='spec_hash', how='left')
                    if need_extrema:
                        extrema = grp_loss.agg(['min','max',lambda s: s.quantile(0.75)-s.quantile(0.25)])
                        extrema.columns = ['_min_tmp','_max_tmp','_iqr_tmp']
                        filtered_non_na = filtered_non_na.merge(extrema, left_on='spec_hash', right_index=True, how='left')
                        if min_c not in filtered_non_na.columns:
                            filtered_non_na.rename(columns={'_min_tmp': min_c}, inplace=True)
                        else:
                            filtered_non_na.drop(columns=['_min_tmp'], inplace=True, errors='ignore')
                        if max_c not in filtered_non_na.columns:
                            filtered_non_na.rename(columns={'_max_tmp': max_c}, inplace=True)
                        else:
                            filtered_non_na.drop(columns=['_max_tmp'], inplace=True, errors='ignore')
                        if iqr_c not in filtered_non_na.columns:
                            filtered_non_na.rename(columns={'_iqr_tmp': iqr_c}, inplace=True)
                        else:
                            filtered_non_na.drop(columns=['_iqr_tmp'], inplace=True, errors='ignore')
                        if range_c not in filtered_non_na.columns and all(c in filtered_non_na.columns for c in [min_c,max_c]):
                            filtered_non_na[range_c] = filtered_non_na[max_c] - filtered_non_na[min_c]
                    print('Info: Synthesized missing restart std/extrema for stability computation from long-form data.')

        filtered_non_na = _compute_stability_flags(
            filtered_non_na,
            metric=args.stability_metric,
            cv_thr=args.stability_cv_threshold,
            range_thr=args.stability_range_threshold,
            require_all=args.stability_require_all,
            epsilon=args.stability_epsilon,
        )

        if ("stable_opt" in filtered_non_na.columns and filtered_non_na["stable_opt"].isna().all()):
            prefix = args.stability_metric if args.stability_metric != "loss" else "loss"
            req = [f"restart_{prefix}_{c}" for c in ["mean","std","min","max","iqr","var","median","range"]]
            present = [c for c in req if c in filtered_non_na.columns]
            print("Debug: stability columns remained NA. Present restart columns:")
            for c in present:
                na_rate = filtered_non_na[c].isna().mean() if c in filtered_non_na else None
                print(f"  {c} (NA%={na_rate:.2%})")
            missing = [c for c in req if c not in filtered_non_na.columns]
            if missing:
                print("Debug: missing restart columns:", missing)
            suffix_cols = [c for c in filtered_non_na.columns if c.endswith('_x') or c.endswith('_y')]
            if suffix_cols:
                print("Debug: found suffixed columns that may need coalescing:", suffix_cols[:12])
    else:
        print("Info: Skipping stability derivation and flag computation (--skip-stability).")

    # Persist enriched combined index (with stability columns)
    combined_path = out_dir / "combined_index.csv"
    filtered_non_na.to_csv(combined_path, index=False)

    # Optional plotting
    if args.plot_metric:
        try:
            import seaborn as sns  # type: ignore
            import matplotlib.pyplot as plt  # type: ignore
            work_plot = filtered_non_na.copy()
            work_plot["model_label"] = work_plot.apply(_compose_model_label, axis=1)
            # Enrich label with loss & optimizer if multiple present
            if work_plot["loss_name"].nunique() > 1 or work_plot["optimizer"].nunique() > 1:
                work_plot["model_label"] = work_plot.apply(lambda r: f"{r['model_label']}-{r['loss_name']}-{r['optimizer']}", axis=1)
            n_models = work_plot["model_label"].nunique()
            plt.figure(figsize=(max(6, 1.1 * n_models), 4.2))
            sns.boxplot(data=work_plot, x="model_label", y=metric_name, color="#a6cee3")
            sns.stripplot(data=work_plot, x="model_label", y=metric_name, color="#fb9a99", alpha=0.55, jitter=0.25)
            plt.ylabel(metric_name)
            plt.xlabel("model")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plot_path = out_dir / f"{metric_name}_by_model.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved plot: {plot_path}")
        except Exception as e:  # pragma: no cover
            print(f"Plotting failed: {e}")

    # Optional top-N print
    if args.print_top and args.print_top > 0:
        ascending = metric_name in _LOWER_IS_BETTER and metric_name not in _HIGHER_IS_BETTER
        ordered = filtered_non_na.sort_values(metric_name, ascending=ascending)
        # Enrich model label similarly for print
        if "loss_name" in ordered.columns and "optimizer" in ordered.columns and (
            ordered["loss_name"].nunique() > 1 or ordered["optimizer"].nunique() > 1
        ):
            ordered["model_label"] = ordered.apply(_compose_model_label, axis=1)
            ordered.loc[:, "model_label"] = ordered.apply(lambda r: f"{r['model_label']}-{r['loss_name']}-{r['optimizer']}", axis=1)

        top_subset = ordered.head(args.print_top).copy()

        # Optionally attach parameter values (best restart) for printed rows
        all_param_keys: set = set()
        if args.include_params:
            param_rows = []
            for idx, r in top_subset.iterrows():
                exp_name = r.get("experiment")
                if not isinstance(exp_name, str) or not exp_name:
                    fit_path = None
                else:
                    fit_path = _locate_fit_json(base_dir, exp_name, r)
                params = _extract_best_restart_params(fit_path) if fit_path else {}
                param_rows.append(params)
                all_param_keys.update(params.keys())
            # Add columns
            for key in sorted(all_param_keys):
                col_name = f"param_{key}"
                vals = []
                for params in param_rows:
                    vals.append(params.get(key))
                top_subset[col_name] = vals
        cols_base = [c for c in ["experiment","version","prompt_category","agent","domain","link","params_tying","lr","model_label",metric_name,"stable_opt","stability_cv","stability_rel_range"] if c in top_subset.columns]
        # Append param columns at end if present
        param_cols = [c for c in top_subset.columns if c.startswith("param_")]
        print_cols = cols_base + param_cols
        direction = "min" if ascending else "max"
        print(f"Top {len(top_subset)} rows by {direction} {metric_name}:")
        print(top_subset[print_cols].to_string(index=False))

    # Remove group-by columns not present
    group_by_cols = [c for c in args.group_by if c in filtered_non_na.columns]
    missing_gb = set(args.group_by) - set(group_by_cols)
    if missing_gb:
        print(f"Info: dropping missing group-by columns: {sorted(missing_gb)}")
    best_df, ranks_df = compute_best_and_ranks(filtered_non_na, metric=metric_name, group_by=group_by_cols)
    best_path = out_dir / "best_by_group.csv"
    ranks_path = out_dir / "ranks_by_group.csv"
    best_df.to_csv(best_path, index=False)
    ranks_df.to_csv(ranks_path, index=False)
    print(f"Saved: {combined_path}\nSaved: {best_path}\nSaved: {ranks_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



