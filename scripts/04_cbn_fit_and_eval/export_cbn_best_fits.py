#!/usr/bin/env python3
"""
Export LaTeX tables of best CBN model fits per experiment, similar to the
Milestone B table exporter. One row per (agent, domain) winner selected by
LOOCV R^2 (with robust fallbacks), ordered by loocv_r2 descending.

Outputs (per experiment):
 - publication/thesis/tuebingen_thesis_msc/tables/<experiment>/cbn_best_fits_{tag}.tex
 - results/parameter_analysis/<experiment>/{tag}/winners.csv
 - (optional) results/parameter_analysis/<experiment>/{tag}/winners_with_params.csv

Usage examples:
 # for rw17_indep_causes, humans pooled, some metric excluded
 python scripts/export_cbn_best_fits.py --experiment rw17_indep_causes --version 2 --exclude-metrics rmse r2 AIC BIC --humans-mode pooled --model noisy_or --lr 0.1 --include-domains all --include-readout-weights --params 3 4 --prompt-categories numeric

 # for random_abstract
python scripts/export_cbn_best_fits.py --experiment random_abstract --version 1 --exclude-metrics rmse r2 AIC BIC loocv_rmse  --exclude-humans --model noisy_or --lr 0.1 --include-domains all --include-readout-weights --params 3 4 --prompt-categories cot
  # Default: discover latest version, all models; collapse across prompt_category
  python scripts/export_model_fit_tables.py --experiment rw17_indep_causes

  # Restrict to noisy_or and version 2
  python scripts/export_model_fit_tables.py --experiment rw17_indep_causes --versions 2 --models noisy_or

  # Also export parameter values of the winning specifications
  python scripts/export_model_fit_tables.py --experiment rw17_indep_causes --export-params

    # Exclude 5-parameter tying models (keep 3p and 4p only)
    python scripts/export_model_fit_tables.py --experiment rw17_indep_causes --params 3 4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from causalign.analysis.model_fitting.data import load_processed_data, prepare_dataset
from causalign.analysis.model_fitting.discovery import load_index
from causalign.analysis.model_fitting.tasks import roman_task_to_probability
from causalign.config.paths import PathManager


def latex_escape(val) -> str:
    """Escape LaTeX special characters for safe use in tabular text cells."""
    if pd.isna(val):
        return "--"
    s = str(val)
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("_", r"\_")
    s = s.replace("#", r"\#")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def _fmt_num(x: Any, nd: int = 3) -> str:
    """Format numbers for LaTeX cells.

    - Returns "--" for None/NaN/Inf
    - Uses fixed decimals for typical magnitudes
    - Uses scientific notation for very small/large magnitudes to avoid 0.000 artifacts
    """
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    try:
        xf = float(x)
    except Exception:
        return "--"
    if xf == 0.0:
        return f"{0.0:.{nd}f}"
    ax = abs(xf)
    # Scientific notation for extreme magnitudes
    if ax < 1e-3 or ax >= 1e3:
        return f"{xf:.2e}"
    return f"{xf:.{nd}f}"


def _latest_version(df: pd.DataFrame) -> Optional[str]:
    if "version" not in df.columns or df["version"].isna().all():
        return None
    vals = df["version"].dropna().astype(str).unique().tolist()
    # Try numeric max first
    nums = []
    for v in vals:
        try:
            nums.append(float(v))
        except Exception:
            pass
    if nums:
        m = max(nums)
        # Return the original string representation matching the numeric max (best-effort)
        for v in vals:
            try:
                if float(v) == m:
                    return v
            except Exception:
                continue
    # Fallback: lexical max
    return max(vals)


def _filter_to_scope(df: pd.DataFrame, versions: Optional[List[str]], models: Optional[List[str]], param_counts: Optional[List[str]]) -> pd.DataFrame:
    out = df.copy()
    if versions:
        out = out[out["version"].astype(str).isin([str(v) for v in versions])]
    if models and "link" in out.columns:
        out = out[out["link"].isin(models)]
    if param_counts and "params_tying" in out.columns:
        out = out[out["params_tying"].astype(str).isin(param_counts)]
    return out


def _winner_row(group: pd.DataFrame) -> pd.Series:
    """Select the winner row within a group using a robust policy.

    Primary: maximize loocv_r2 (drop NaN if any finite exist).
    Fallback chain if all loocv_r2 are NaN/missing:
        cv_r2 -> r2 -> -bic -> -aic
    Tie-breakers within a tiny epsilon on the primary metric:
        lower loocv_rmse -> lower bic -> lower aic -> lower loss -> fewer params_tying -> min short_spec_hash
    """
    work = group.copy()
    # Establish candidate mask for primary metric
    if "loocv_r2" in work.columns:
        metric = "loocv_r2"
        non_na = work[metric].notna()
        if non_na.any():
            work = work[non_na]
            # Sort by descending loocv_r2
            work = work.sort_values([metric, "loocv_rmse", "bic", "aic", "loss", "params_tying"], ascending=[False, True, True, True, True, True])
            return work.iloc[0]
    # Fallbacks
    for metric, asc, prefer_high in [
        ("cv_r2", False, True),
        ("r2", False, True),
        ("bic", True, False),
        ("aic", True, False),
    ]:
        if metric in work.columns and work[metric].notna().any():
            work2 = work[work[metric].notna()].copy()
            work2 = work2.sort_values([metric, "loocv_rmse" if "loocv_rmse" in work2.columns else metric, "bic" if "bic" in work2.columns else metric, "aic" if "aic" in work2.columns else metric, "loss" if "loss" in work2.columns else metric, "params_tying" if "params_tying" in work2.columns else metric], ascending=[asc, True, True, True, True, True])
            return work2.iloc[0]
    # If everything is NA, just return the first row
    return work.iloc[0]


def _locate_fit_json(base_dir: Path, experiment: str, row: pd.Series) -> Optional[Path]:
    root = base_dir / "results" / "model_fitting" / experiment
    short_spec = row.get("short_spec_hash") or row.get("short_spec")
    short_group = row.get("short_group_hash") or row.get("short_group")
    patterns: list[str] = []
    if isinstance(short_spec, str) and isinstance(short_group, str) and short_spec and short_group:
        patterns.append(f"fit_{short_spec}_{short_group}.json")
    patterns.append("fit_*.json")
    for pat in patterns:
        matches = list(root.rglob(pat))
        if not matches:
            continue
        if pat.startswith("fit_") and pat.endswith(".json") and len(matches) > 1 and short_spec and short_group:
            narrowed = [m for m in matches if short_spec in m.name and short_group in m.name]
            if narrowed:
                matches = narrowed
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


def _canonicalize_params(raw: Dict[str, float], link: Optional[str], tying: Optional[int]) -> Dict[str, Optional[float]]:
    """Map raw param dict to canonical CBN parameters and expand ties.

    Canonical keys: b, m1, m2, pC1, pC2
    Aliases handled (case-insensitive):
      - b: b, bias, b0
      - m1/m2: m1, m2, m (duplicates to both), w_m1, w_m2
      - pC1/pC2: pc1, pc2, p_c1, p_c2, pC1, pC2, pC (duplicates to both)
    If tying indicates fewer unique params, duplicate available value across the pair.
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("-", "_")

    # Normalize keys
    low = {norm(k): v for k, v in (raw or {}).items()}

    b = None
    m1 = None
    m2 = None
    pc1 = None
    pc2 = None

    # b aliases
    for key in ("b", "bias", "b0"):
        if key in low and b is None:
            try:
                b = float(low[key])
            except Exception:
                pass
    # m aliases
    m_single = None
    for key in ("m", "m_", "w_m"):
        if key in low and m_single is None:
            try:
                m_single = float(low[key])
            except Exception:
                pass
    for key in ("m1", "w_m1"):
        if key in low and m1 is None:
            try:
                m1 = float(low[key])
            except Exception:
                pass
    for key in ("m2", "w_m2"):
        if key in low and m2 is None:
            try:
                m2 = float(low[key])
            except Exception:
                pass
    # pC aliases
    pc_single = None
    for key in ("pc", "p_c", "p_c_", "p"):
        if key in low and pc_single is None:
            try:
                pc_single = float(low[key])
            except Exception:
                pass
    for key in ("pc1", "p_c1", "pc_1", "p_c_1"):
        if key in low and pc1 is None:
            try:
                pc1 = float(low[key])
            except Exception:
                pass
    for key in ("pc2", "p_c2", "pc_2", "p_c_2"):
        if key in low and pc2 is None:
            try:
                pc2 = float(low[key])
            except Exception:
                pass

    # If single aliases present, duplicate
    if m_single is not None:
        if m1 is None:
            m1 = m_single
        if m2 is None:
            m2 = m_single
    if pc_single is not None:
        if pc1 is None:
            pc1 = pc_single
        if pc2 is None:
            pc2 = pc_single

    # Expand by tying hints
    try:
        t = int(float(tying)) if tying is not None else None
    except Exception:
        t = None
    if t in (3, 4):
        # Commonly: m1==m2 and/or pC1==pC2
        if m1 is not None and m2 is None:
            m2 = m1
        if m2 is not None and m1 is None:
            m1 = m2
        if pc1 is not None and pc2 is None:
            pc2 = pc1
        if pc2 is not None and pc1 is None:
            pc1 = pc2

    return {"b": b, "m1": m1, "m2": m2, "pC1": pc1, "pC2": pc2}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export LaTeX tables of best CBN fits per experiment")
    p.add_argument("--experiments", nargs="*", help="Experiments to include (default: discover from results/model_fitting)")
    p.add_argument("--experiment", dest="experiments", nargs="*", help="Alias for --experiments")
    p.add_argument("--versions", nargs="*", help="Version(s) to include; default: latest per experiment")
    p.add_argument("--models", nargs="*", choices=["logistic", "noisy_or"], help="Model link types to include")
    # p.add_argument("--param-counts", nargs="*", choices=["3", "4", "5"], help="Parameter tying counts to include")
    p.add_argument("--params", dest="param_counts", nargs="*", help="Alias for --param-counts; e.g., --params 3 4 to exclude 5p")
    p.add_argument("--tables-dir", default="publication/thesis/tuebingen_thesis_msc/tables", help="Base tables directory")
    p.add_argument("--collapse-prompt", action="store_true", default=True, help="Collapse across prompt_category (default: True)")
    p.add_argument("--no-collapse-prompt", dest="collapse_prompt", action="store_false", help="Do not collapse across prompt_category")
    p.add_argument("--export-params", action="store_true", default=True, help="Export winners_with_params.csv alongside table (default: True)")
    p.add_argument("--no-export-params", dest="export_params", action="store_false", help="Disable exporting winners_with_params.csv")
    p.add_argument("--pooled-domain-label", default="all", help="Label to use in LaTeX and CSVs when pooled domain is selected (default: 'all')")
    p.add_argument("--include-domains", nargs="+", help="Domain scope: pass names (one or more) to include; pass 'all' to select pooled-only")
    p.add_argument("--prompt-categories", nargs="*", default=["numeric"], help="Prompt categories to include (default: numeric)")
    p.add_argument("--lr", nargs="*", help="Learning-rate folders to include (e.g., lr0p1 lr0p01; or numeric 0.1 0.01; also 'base' for top-level, 'all' for no filter)")
    p.add_argument("--exclude-humans", action="store_true", help="Exclude rows where agent is 'humans' (case-insensitive)")
    p.add_argument("--include-readout-weights", action="store_true", help="Also include readout weights (w0,w1,w2) in winners_with_params.csv if present")
    p.add_argument("--fail-missing-canonical", action="store_true", help="Hard-error if required canonical parameters cannot be constructed from the fit JSON")
    # Metric selection controls (affects LaTeX columns only; winners.csv remains full)
    p.add_argument(
        "--metrics-in-sample",
        nargs="*",
        choices=["loss", "rmse", "r2", "r2_task", "mae", "aic", "bic"],
        help="In-sample metrics to include (ordered). Defaults to: rmse r2_task",
    )
    p.add_argument(
        "--metrics-out-of-sample",
        nargs="*",
        choices=["loocv_rmse", "loocv_r2", "loocv_mae"],
        help="Out-of-sample metrics to include (ordered). Defaults to: loocv_r2 loocv_rmse",
    )
    p.add_argument(
        "--exclude-metrics",
        dest="exclude_metrics",
        nargs="*",
        default=[],
        help="Metrics to exclude (applies after selection). Example: --exclude-metrics loocv_rmse rmse",
    )
    # Short alias
    p.add_argument("--exclude", dest="exclude_metrics", nargs="*", help="Alias for --exclude-metrics")
    # Humans-mode filtering: select which human agents to include while keeping all non-human agents
    p.add_argument(
        "--humans-mode",
        choices=["all", "aggregated", "pooled", "individual"],
        default="all",
        help="Restrict included human agents to a mode: aggregated ('humans'), pooled ('humans-pooled'), or individual ('human-<id>'). Non-human agents are always included. Default: all",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = Path(__file__).parent.parent

    # Heuristic: interpret accidental positional 'params 3 4' usage as '--params 3 4'
    if not args.param_counts and args.experiments and "params" in args.experiments:
        try:
            i = args.experiments.index("params")
            vals: List[str] = []
            j = i + 1
            while j < len(args.experiments):
                tok = str(args.experiments[j])
                if tok.isdigit():
                    vals.append(tok)
                    j += 1
                else:
                    break
            if vals:
                args.param_counts = vals
                # drop 'params' and following numeric tokens from experiments list
                args.experiments = args.experiments[:i] + args.experiments[j:]
                print(f"Note: interpreted 'params {' '.join(vals)}' as '--params {' '.join(vals)}'.")
        except Exception:
            pass
    # Heuristic: if user wrote '--prompt-categories numeric params 3 4', recover and treat as '--params 3 4'
    if not args.param_counts and args.prompt_categories and "params" in args.prompt_categories:
        try:
            i = args.prompt_categories.index("params")
            vals: List[str] = []
            j = i + 1
            while j < len(args.prompt_categories) and str(args.prompt_categories[j]).isdigit():
                vals.append(str(args.prompt_categories[j]))
                j += 1
            if vals:
                args.param_counts = vals
                args.prompt_categories = args.prompt_categories[:i] + args.prompt_categories[j:]
                print(f"Note: interpreted '--prompt-categories ... params {' '.join(vals)}' as '--params {' '.join(vals)}'.")
        except Exception:
            pass

    # Normalize/clean param_counts if provided (ignore stray tokens like 'model', 'noisy')
    if args.param_counts:
        cleaned: list[str] = []
        junk: list[str] = []
        for tok in args.param_counts:
            s = str(tok)
            if s.isdigit():
                cleaned.append(s)
            else:
                junk.append(s)
        if junk:
            print(f"Warning: ignoring non-numeric --params tokens: {', '.join(junk)}")
        args.param_counts = cleaned or None

    # Discover experiments if not provided
    experiments: List[str] = []
    if args.experiments:
        experiments = list(dict.fromkeys(args.experiments))
    else:
        root = base_dir / "results" / "model_fitting"
        if root.exists():
            experiments = [p.name for p in sorted(root.iterdir()) if (p / "fit_index.parquet").exists()]
    if not experiments:
        print("No experiments found.")
        return 1

    # Tables root
    tables_root = base_dir / args.tables_dir
    tables_root.mkdir(parents=True, exist_ok=True)

    # Helpful nudge if models not specified (defaults to all available in index)
    if not args.models:
        print("Note: --models not specified; including all links present in the index (e.g., logistic and noisy_or). Use --models to restrict.")

    for exp in experiments:
        idx = load_index(base_dir, exp)
        if idx is None or idx.empty:
            print(f"Skip {exp}: no index.")
            continue
        # Determine version scope (latest by default)
        versions = args.versions
        if not versions:
            lv = _latest_version(idx)
            versions = [lv] if lv is not None else None

        df_scoped = _filter_to_scope(idx, versions=versions, models=args.models, param_counts=args.param_counts)
        # Filter by learning-rate folders if requested
        if args.lr:
            lr_tokens = [str(x) for x in args.lr]
            if not (len(lr_tokens) == 1 and lr_tokens[0].lower() == "all"):
                # normalize tokens to candidate lr_subdir names
                wanted_lr_subdir: set[str] = set()
                for tok in lr_tokens:
                    t = tok.lower()
                    if t in {"base", "root", "top"}:
                        wanted_lr_subdir.add("")  # represent base directory with empty
                        continue
                    if t.startswith("lr"):
                        wanted_lr_subdir.add(t)
                        continue
                    # try numeric like 0.1 -> lr0p1 ; 0.01 -> lr0p01
                    try:
                        f = float(t)
                        # Build folder name; handle common values up to 1e-6 precisely
                        s = ("%g" % f).rstrip("0").rstrip(".")
                        s = s.replace(".", "p").replace("-", "m")  # 0.1 -> 0p1, -0.1 -> m0p1 (unlikely)
                        wanted_lr_subdir.add("lr" + s)
                    except Exception:
                        # Keep as-is for robustness
                        wanted_lr_subdir.add(t)
                # df_scoped may or may not have lr_subdir column (top-level rows absent). Treat NaN as base ('').
                if "lr_subdir" in df_scoped.columns:
                    sub = df_scoped["lr_subdir"].fillna("").astype(str)
                    df_scoped = df_scoped[sub.isin(wanted_lr_subdir)]
                else:
                    # Only base-level rows exist; keep them if base specified, else drop all
                    if "" not in wanted_lr_subdir:
                        df_scoped = df_scoped.iloc[0:0]
            # If after filtering no rows, skip
            if df_scoped.empty:
                print(f"Skip {exp}: no rows after filtering by --lr {lr_tokens}.")
                continue
        # Filter by prompt categories (default: numeric)
        if "prompt_category" in df_scoped.columns and args.prompt_categories:
            # Legacy safety: map legacy/new names both ways and compare case-insensitively.
            pcs_in = [str(x) for x in args.prompt_categories]
            req = [p.lower() for p in pcs_in]
            legacy_map = {
                # numeric family
                "numeric": {"single_numeric", "single_numeric_response"},
                "single_numeric": {"numeric"},
                "single_numeric_response": {"numeric"},
                # CoT family
                "cot": {"numeric-confidence-cot"},
                "numeric-confidence-cot": {"cot"},
                # numeric-conf family
                "numeric-conf": {"numeric-certainty"},
                "numeric-certainty": {"numeric-conf"},
            }
            targets: set[str] = set()
            for p in req:
                targets.add(p)
                if p in legacy_map:
                    targets.update(legacy_map[p])
            pc_l = df_scoped["prompt_category"].astype(str).str.lower()
            df_scoped = df_scoped[pc_l.isin(targets)].copy()
            if df_scoped.empty:
                print(f"Skip {exp}: no rows after filtering by prompt categories {sorted(targets)}.")
                continue
        if df_scoped.empty:
            print(f"Skip {exp}: empty after filtering.")
            continue

        # Humans-mode filtering (keep all non-human agents; restrict human-like rows by mode)
        if "agent" in df_scoped.columns and args.humans_mode and args.humans_mode != "all":
            # Determine human-mode per row using both agent label and domain nullability.
            # Rules:
            #  - 'humans' with domain notna -> aggregated
            #  - 'humans' with domain na -> pooled (pooled-all)
            #  - 'humans-pooled'/'human-pooled' -> pooled
            #  - 'human-*' / 'humans-*' -> individual
            def _row_mode(row) -> str:
                ls = str(row.get("agent")).strip().lower()
                dom = row.get("domain") if "domain" in row.index else None
                is_dom_na = (dom is None) or (isinstance(dom, float) and pd.isna(dom))
                if ls in {"humans-pooled", "human-pooled"}:
                    return "pooled"
                if ls == "humans":
                    return "pooled" if is_dom_na else "aggregated"
                if ls.startswith("human-") or ls.startswith("humans-"):
                    return "individual"
                return "nonhuman"

            modes = df_scoped.apply(_row_mode, axis=1)
            before = len(df_scoped)
            if args.humans_mode == "aggregated":
                df_scoped = df_scoped[(modes == "aggregated") | (modes == "nonhuman")]
            elif args.humans_mode == "pooled":
                df_scoped = df_scoped[(modes == "pooled") | (modes == "nonhuman")]
            elif args.humans_mode == "individual":
                df_scoped = df_scoped[(modes == "individual") | (modes == "nonhuman")]
            if df_scoped.empty:
                print(f"Skip {exp}: no rows after --humans-mode {args.humans_mode} filter.")
                continue
            if before != len(df_scoped):
                print(f"Humans-mode filter '{args.humans_mode}': kept {len(df_scoped)}/{before} rows (non-humans always kept).")
            if df_scoped.empty:
                print(f"Skip {exp}: no rows after --humans-mode {args.humans_mode} filter.")
                continue
            if before != len(df_scoped):
                print(f"Humans-mode filter '{args.humans_mode}': kept {len(df_scoped)}/{before} rows (non-humans always kept).")

        # Domain scoping (new): --include-domains
        # Semantics:
        #  - include-domains all -> keep pooled rows only (domain is NaN in index; we map to label later)
        #  - include-domains A B -> keep rows where domain in {A,B}
        # Back-compat: if user still passes --all-domains-only, treat as include-domains all
        include_domains = args.include_domains
        if getattr(args, "all_domains_only", False) and not include_domains:
            include_domains = ["all"]
        if include_domains:
            toks = [str(x) for x in include_domains]
            if len(toks) == 1 and toks[0].lower() == "all":
                # pooled-only
                if "domain" in df_scoped.columns:
                    df_scoped = df_scoped[df_scoped["domain"].isna()].copy()
                if df_scoped.empty:
                    print(f"Skip {exp}: no pooled 'all' domain rows after include-domains all.")
                    continue
            else:
                if "domain" in df_scoped.columns:
                    df_scoped = df_scoped[df_scoped["domain"].astype(str).isin(toks)].copy()
                if df_scoped.empty:
                    print(f"Skip {exp}: no rows after include-domains {toks}.")
                    continue

        # Optionally exclude humans
        if args.exclude_humans and "agent" in df_scoped.columns:
            mask = df_scoped["agent"].astype(str).str.lower() != "humans"
            df_scoped = df_scoped[mask].copy()
            if df_scoped.empty:
                print(f"Skip {exp}: no rows after excluding humans.")
                continue
        # if args.require_domain and "domain" in df_scoped.columns:
        #     df_scoped = df_scoped[df_scoped["domain"].notna()].copy()
        #     if df_scoped.empty:
        #         print(f"Skip {exp}: no domain-specific rows after filtering.")
        #         continue

        # Group by (agent, domain) within this experiment/version; optionally collapse prompt_category
        group_cols = ["agent", "domain"]
        scope_cols = ["experiment", "version"]
        if not args.collapse_prompt and "prompt_category" in df_scoped.columns:
            group_cols = ["prompt_category"] + group_cols
        winners: List[pd.Series] = []
        # Partition by scope (experiment, version) and group keys
        for (_, _), df_scope in df_scoped.groupby(scope_cols, dropna=False):
            for _, df_grp in df_scope.groupby(group_cols, dropna=False):
                if df_grp.empty:
                    continue
                winners.append(_winner_row(df_grp))
        if not winners:
            print(f"Skip {exp}: no winners found.")
            continue
        winners_df = pd.DataFrame(winners).copy()

        # Optional backfill: compute task-aggregated in-sample metrics (r2_task, rmse_task)
        # for rows where they're missing by reconstructing predictions from the best params.
        needs_any_r2t = ("r2_task" not in winners_df.columns) or winners_df["r2_task"].isna().any()
        needs_any_rmset = ("rmse_task" not in winners_df.columns) or winners_df["rmse_task"].isna().any()
        if needs_any_r2t or needs_any_rmset:
            paths = PathManager(base_dir)
            # Build a cache for loaded dataframes by (version, use_aggregated)
            df_cache: Dict[tuple, pd.DataFrame] = {}
            # Ensure columns exist to assign into
            if "r2_task" not in winners_df.columns:
                winners_df["r2_task"] = pd.NA
            if "rmse_task" not in winners_df.columns:
                winners_df["rmse_task"] = pd.NA
            # Iterate and backfill per row if missing
            for i, row in winners_df.iterrows():
                has_r2t = pd.notna(row.get("r2_task"))
                has_rmset = pd.notna(row.get("rmse_task"))
                if (not needs_any_r2t or has_r2t) and (not needs_any_rmset or has_rmset):
                    continue
                r2t: Optional[float] = None
                rmset: Optional[float] = None
                try:
                    version_raw = row.get("version")
                    version = str(version_raw) if version_raw is not None else ""
                    # Decide whether to use aggregated data (LLMs and aggregated/pooled humans) or
                    # individual-level data (for agents like 'human-123').
                    agent_label = str(row.get("agent") or "")
                    ls = agent_label.strip().lower()
                    is_individual_human = ls.startswith("human-") or ls.startswith("humans-")
                    use_agg = not is_individual_human
                    cache_key = (version, use_agg)
                    if cache_key not in df_cache:
                        df_cache[cache_key] = load_processed_data(
                            paths,
                            version=(version if version else None),
                            experiment_name=exp,
                            graph_type="collider",
                            use_roman_numerals=True,
                            use_aggregated=use_agg,
                            pipeline_mode="llm_with_humans",
                        )
                    df_all = df_cache[cache_key]
                    agent = agent_label
                    prompt_cat = str(row.get("prompt_category")) if not pd.isna(row.get("prompt_category")) else (args.prompt_categories[0] if args.prompt_categories else None)
                    # For pooled domain rows, domain is NaN; prepare_dataset will ignore None
                    domain_val = row.get("domain")
                    domains = None if (domain_val is None or (isinstance(domain_val, float) and pd.isna(domain_val))) else [str(domain_val)]
                    df_grp = prepare_dataset(
                        df_all,
                        agents=[agent],
                        domains=domains,
                        prompt_categories=[prompt_cat] if prompt_cat else None,
                    )
                    # Fallback: retry without prompt category restriction if empty (legacy naming issues)
                    if df_grp.empty:
                        df_grp = prepare_dataset(
                            df_all,
                            agents=[agent],
                            domains=domains,
                            prompt_categories=None,
                        )
                    if df_grp.empty:
                        raise RuntimeError("Group data not found for backfill")
                    # Unique tasks and per-task mean targets
                    gb = df_grp.groupby("task")
                    y_task = gb["response"].mean().astype(float)
                    tasks_unique = [str(t) for t in y_task.index.tolist()]
                    # Build predictions from best params
                    fit_path = _locate_fit_json(base_dir, exp, row)
                    params = _extract_best_restart_params(fit_path) if fit_path else {}
                    link_val = row.get("link") if "link" in winners_df.columns else None
                    link = str(link_val) if link_val is not None else "noisy_or"
                    # Canonicalize param dict to expected keys for probability eval
                    tying_val = row.get("params_tying") if "params_tying" in winners_df.columns else None
                    try:
                        tying_int = int(float(tying_val)) if tying_val is not None and not (isinstance(tying_val, float) and pd.isna(tying_val)) else None
                    except Exception:
                        tying_int = None
                    canon_params = _canonicalize_params(params, link, tying_int)
                    # Convert to tensors
                    param_tensors = {k: torch.tensor(float(v), dtype=torch.float32) for k, v in canon_params.items() if v is not None}
                    preds = []
                    for roman in tasks_unique:
                        try:
                            pt = roman_task_to_probability(roman, link, param_tensors)
                            preds.append(float(pt.item()))
                        except Exception:
                            preds.append(float("nan"))
                    import numpy as _np
                    p_task = _np.array(preds, dtype=float)
                    y_task_arr = _np.array(y_task.values, dtype=float)
                    mask = _np.isfinite(p_task) & _np.isfinite(y_task_arr)
                    if mask.sum() >= 2:
                        dy = p_task[mask] - y_task_arr[mask]
                        sse = float(_np.sum(dy ** 2))
                        sst = float(_np.sum((y_task_arr[mask] - _np.mean(y_task_arr[mask])) ** 2))
                        r2t_calc = float(1.0 - (sse / sst)) if sst > 0 else float("nan")
                        rmset_calc = float(_np.sqrt(_np.mean(dy ** 2)))
                        r2t = r2t_calc if _np.isfinite(r2t_calc) else None
                        rmset = rmset_calc if _np.isfinite(rmset_calc) else None
                        # Guard against tiny numerical >1.0
                        r2t = None if r2t is not None and (r2t > 1.0 and r2t < 1.0 + 1e-8) else r2t
                except Exception:
                    r2t = None
                    rmset = None
                if (not has_r2t) and (r2t is not None):
                    winners_df.at[i, "r2_task"] = r2t
                if (not has_rmset) and (rmset is not None):
                    winners_df.at[i, "rmse_task"] = rmset

        # Ensure a 'link' column exists for downstream filtering/labeling.
        # Some indices (e.g., certain experiments) may lack 'link'; backfill per winner by reading the fit JSON.
        need_backfill = ("link" not in winners_df.columns) or winners_df["link"].isna().any()
        if need_backfill:
            link_vals = []
            for _, row in winners_df.iterrows():
                link_val = row.get("link") if "link" in winners_df.columns else None
                if link_val is None or (isinstance(link_val, float) and pd.isna(link_val)):
                    fit_path = _locate_fit_json(base_dir, exp, row)
                    if fit_path and fit_path.exists():
                        try:
                            js = json.loads(fit_path.read_text())
                            # Common keys where link is found
                            for key in ("link", "cbn_link", "model", "model_link"):
                                if key in js and isinstance(js[key], str) and js[key]:
                                    link_val = js[key]
                                    break
                        except Exception:
                            link_val = None
                link_vals.append(link_val)
            winners_df["link"] = link_vals

        # If user requested specific models, filter winners_df now (even if index lacked link)
        if args.models:
            before_n = len(winners_df)
            winners_df = winners_df[winners_df["link"].astype(str).isin(args.models)]
            if winners_df.empty:
                print(f"Skip {exp}: no winners match requested --models {args.models} after backfilling link.")
                continue
            if before_n != len(winners_df):
                print(f"Filtered winners by --models {args.models}: kept {len(winners_df)}/{before_n} rows.")

        # Determine tag for filenames
        tag_parts = []
        if versions:
            vtag = "v" + "-".join(sorted({str(v) for v in versions}))
            tag_parts.append(vtag)
        if args.models:
            mtag = "-".join(sorted(set(args.models)))
            tag_parts.append(mtag)
        # Prompt category abbreviations to disambiguate outputs
        if args.prompt_categories:
            def _pc_abbrev(s: str) -> str:
                t = (s or "").lower()
                t = t.replace("_", "-")
                mapping = {
                    "numeric": "num",
                    "single-numeric": "num",
                    "single-numeric-response": "num",
                    "cot": "cot",
                    "numeric-confidence-cot": "cot",
                    "numeric-conf": "nconf",
                    "numeric-certainty": "nconf",
                }
                if t in mapping:
                    return mapping[t]
                # fallback: compact alphanum only
                compact = "".join(ch for ch in t if ch.isalnum())
                return compact[:8] if compact else "pc"
            pc_abbrs = sorted({ _pc_abbrev(x) for x in args.prompt_categories })
            if pc_abbrs:
                tag_parts.append("pc" + "-".join(pc_abbrs))
        if args.param_counts:
            ptag = "p" + "-".join(sorted(set(args.param_counts)))
            tag_parts.append(ptag)
        if args.lr and not (len(args.lr) == 1 and str(args.lr[0]).lower() == "all"):
            # produce a short lr tag
            lr_tag = "lr" + "-".join(sorted({str(x) for x in args.lr}))
            tag_parts.append(lr_tag)
        if args.exclude_humans:
            tag_parts.append("noh")
        # Humans-mode tag suffix
        if args.humans_mode and args.humans_mode != "all":
            hm_map = {"aggregated": "hm-agg", "pooled": "hm-pooled", "individual": "hm-indiv"}
            tag_parts.append(hm_map.get(args.humans_mode, args.humans_mode))
        tag = "_".join(tag_parts) if tag_parts else "all"

        # Sort rows by LOOCV R^2 descending (NaN at the end)
        if "loocv_r2" in winners_df.columns:
            winners_df = winners_df.sort_values(["loocv_r2"], ascending=[False], na_position="last")

        # Collect caption info from winners
        losses = sorted(x for x in winners_df.get("loss_name", pd.Series(dtype=str)).dropna().unique().tolist())
        opts = sorted(x for x in winners_df.get("optimizer", pd.Series(dtype=str)).dropna().unique().tolist())
        loss_txt = ", ".join(latex_escape(x) for x in losses) if losses else "n/a"
        opt_txt = ", ".join(latex_escape(x) for x in opts) if opts else "n/a"

        # Determine learning rates per winner (from columns, fit JSON, or lr_subdir) and how to present them
        lr_values: List[Optional[float]] = []
        # Prefer any existing LR-like columns
        lr_col: Optional[str] = None
        for cand in ["learning_rate", "lr", "opt_lr"]:
            if cand in winners_df.columns:
                lr_col = cand
                break
        for _, row in winners_df.iterrows():
            lr_val: Optional[float] = None
            if lr_col is not None:
                try:
                    v = row.get(lr_col)
                    if v is not None and not (isinstance(v, float) and pd.isna(v)):
                        lr_val = float(v)
                except Exception:
                    lr_val = None
            if lr_val is None:
                # Try reading the fit JSON for optimizer kwargs
                fit_path = _locate_fit_json(base_dir, exp, row)
                if fit_path and fit_path.exists():
                    try:
                        js = json.loads(fit_path.read_text())
                        found = None
                        # Check common locations
                        for key in ("learning_rate", "lr"):
                            if key in js:
                                try:
                                    found = float(js[key])
                                    break
                                except Exception:
                                    pass
                        if found is None:
                            opt_cfg = js.get("optimizer_kwargs") or js.get("optimizer_params") or js.get("optimizer_config") or js.get("optimizer")
                            if isinstance(opt_cfg, dict):
                                for key in ("learning_rate", "lr"):
                                    if key in opt_cfg:
                                        try:
                                            found = float(opt_cfg[key])
                                            break
                                        except Exception:
                                            pass
                        lr_val = found
                    except Exception:
                        lr_val = None
            if lr_val is None:
                # Fallback: attempt to parse from lr_subdir token, e.g., lr0p1 -> 0.1
                lr_sub: Optional[str] = None
                if "lr_subdir" in winners_df.columns:
                    lr_sub = str(row.get("lr_subdir")) if row.get("lr_subdir") is not None else None
                if lr_sub:
                    try:
                        token = lr_sub.strip().lower()
                        if token.startswith("lr"):
                            token = token[2:]
                        token = token.replace("p", ".")
                        token = token.lstrip("m")  # ignore potential leading minus marker
                        lr_val = float(token)
                    except Exception:
                        lr_val = None
            lr_values.append(lr_val)

        uniq_lrs = sorted({round(v, 10) for v in lr_values if isinstance(v, (float, int))})
        # If varied across rows, add a LR column; else include in caption
        add_lr_column = len(uniq_lrs) > 1
        single_lr_value: Optional[float] = uniq_lrs[0] if len(uniq_lrs) == 1 else None
        if add_lr_column:
            winners_df["learning_rate"] = lr_values

        # Prepare LaTeX lines
        # Metric selection: group by in-sample vs out-of-sample; defaults and excludes
        def _has_non_na(col: str) -> bool:
            return (col in winners_df.columns) and bool(pd.to_numeric(winners_df[col], errors="coerce").notna().any())

        # Default metric sets
        default_insample = ["rmse", "r2_task"]
        default_oos = ["loocv_r2", "loocv_rmse"]

        insample_req = list(args.metrics_in_sample) if args.metrics_in_sample else default_insample.copy()
        oos_req = list(args.metrics_out_of_sample) if args.metrics_out_of_sample else default_oos.copy()

        # Apply excludes
        excludes = set(str(x) for x in (args.exclude_metrics or []))
        insample_req = [m for m in insample_req if m not in excludes]
        oos_req = [m for m in oos_req if m not in excludes]

        # Map fallbacks for r2/r2_task and rmse/rmse_task: prefer task variants when available
        def _resolve_metric_single(m: str) -> Optional[str]:
            # Prefer task-aggregated forms; fall back to base; never include both
            if m == "r2_task":
                if _has_non_na("r2_task"):
                    return "r2_task"
                if _has_non_na("r2"):
                    print("Warning: r2_task not available; falling back to r2.")
                    return "r2"
                return None
            if m == "rmse_task":
                if _has_non_na("rmse_task"):
                    return "rmse_task"
                if _has_non_na("rmse"):
                    print("Warning: rmse_task not available; falling back to rmse.")
                    return "rmse"
                return None
            return m if _has_non_na(m) else None

        insample_cols: list[str] = []
        missing_insample: list[str] = []
        for m in insample_req:
            col = _resolve_metric_single(m)
            if col is None:
                missing_insample.append(m)
            elif col not in insample_cols:
                insample_cols.append(col)

        oos_cols: list[str] = []
        missing_oos: list[str] = []
        for m in oos_req:
            if _has_non_na(m):
                oos_cols.append(m)
            else:
                missing_oos.append(m)

        # Warnings for unavailable metrics after resolution
        if missing_insample:
            print("Warning: the following in-sample metrics are not available and will be omitted: " + ", ".join(sorted(set(missing_insample))))
        if missing_oos:
            print("Warning: the following out-of-sample metrics are not available and will be omitted: " + ", ".join(sorted(set(missing_oos))))

        # Compose LaTeX column order: model/meta first, then in-sample, then out-of-sample
        cols_order = [
            "link",
            "params_tying",
            # in-sample metrics
            *insample_cols,
            # other useful in-sample metrics if explicitly requested and present
            # out-of-sample metrics
            *oos_cols,
        ]
        # Filter out any Nones or missing columns
        present_cols = [c for c in cols_order if isinstance(c, str) and c in winners_df.columns]

        # If only a single model/link is present, drop it from LaTeX columns and add to caption
        single_model_label: Optional[str] = None
        if "link" in winners_df.columns:
            uniq_links = sorted([str(x) for x in winners_df["link"].dropna().unique().tolist()])
            if len(uniq_links) == 1:
                single_model_label = uniq_links[0]
        latex_cols = present_cols.copy()
        if single_model_label and "link" in latex_cols:
            latex_cols.remove("link")
        if add_lr_column:
            # Insert LR after params_tying if present, else append
            insert_at = latex_cols.index("params_tying") + 1 if "params_tying" in latex_cols else len(latex_cols)
            if "learning_rate" not in latex_cols:
                latex_cols.insert(insert_at, "learning_rate")

        # Persist CSV of winners for traceability (canonical parameter_analysis location)
        out_eval_dir_param = base_dir / "results" / "parameter_analysis" / exp / tag
        out_eval_dir_param.mkdir(parents=True, exist_ok=True)
        winners_payload = winners_df[present_cols + [c for c in ["agent","domain","version","prompt_category","loss_name","optimizer","spec_hash","short_spec_hash","group_hash","short_group_hash"] if c in winners_df.columns]]
        # Normalize pooled domain label in winners.csv for clarity if pooled was selected
        if (args.include_domains and len(args.include_domains) == 1 and str(args.include_domains[0]).lower() == "all") and "domain" in winners_payload.columns:
            winners_payload = winners_payload.copy()
            winners_payload["domain"] = winners_payload["domain"].where(winners_payload["domain"].notna(), other=args.pooled_domain_label)
        winners_payload.to_csv(out_eval_dir_param / "winners.csv", index=False)

        # Optional parameters export (now enriched with minimal metrics and provenance)
        params_rows: List[Dict[str, Any]] = []
        extras_param_keys: set[str] = set()
        if args.export_params:
            for _, row in winners_df.iterrows():
                fit_path = _locate_fit_json(base_dir, exp, row)
                params = _extract_best_restart_params(fit_path) if fit_path else {}
                # Include tying parameter count for traceability
                tying = row.get("params_tying")
                try:
                    tying = int(float(tying)) if tying is not None and not (isinstance(tying, float) and pd.isna(tying)) else None
                except Exception:
                    tying = None
                # Canonicalize core CBN parameters
                link = row.get("link")
                canon = _canonicalize_params(params, link, tying)
                # Guardrail: ensure canonical parameters present if requested
                if args.fail_missing_canonical:
                    missing = [k for k, v in canon.items() if v is None]
                    if missing:
                        raise RuntimeError(f"Missing canonical params for winner (agent={row.get('agent')}, domain={row.get('domain')}, link={link}, tying={tying}): {missing}")

                # Resolve domain label (pooled as configured)
                domain_val = row.get("domain")
                domain_out = domain_val if (domain_val is not None and not (isinstance(domain_val, float) and pd.isna(domain_val))) else args.pooled_domain_label

                # Learning rate: prefer column if present (added above when varied); else None
                lr_val = None
                try:
                    if "learning_rate" in winners_df.columns:
                        v = row.get("learning_rate")
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            lr_val = float(v)
                except Exception:
                    lr_val = None

                record: Dict[str, Any] = {
                    # Identity
                    "agent": row.get("agent"),
                    "domain": domain_out,
                    "version": row.get("version"),
                    "prompt_category": row.get("prompt_category") if "prompt_category" in winners_df.columns else None,
                    # Model provenance
                    "link": link,
                    "params_tying": tying,
                    "spec_hash": row.get("spec_hash") if "spec_hash" in winners_df.columns else None,
                    "short_spec_hash": row.get("short_spec_hash") if "short_spec_hash" in winners_df.columns else None,
                    "group_hash": row.get("group_hash") if "group_hash" in winners_df.columns else None,
                    "short_group_hash": row.get("short_group_hash") if "short_group_hash" in winners_df.columns else None,
                    "lr_subdir": row.get("lr_subdir") if "lr_subdir" in winners_df.columns else None,
                    "learning_rate": lr_val,
                    # Minimal metrics for plotting/QA
                    "loocv_r2": row.get("loocv_r2") if "loocv_r2" in winners_df.columns else None,
                    "cv_r2": row.get("cv_r2") if "cv_r2" in winners_df.columns else None,
                    "r2_task": row.get("r2_task") if "r2_task" in winners_df.columns else (row.get("r2") if "r2" in winners_df.columns else None),
                    "rmse_task": row.get("rmse_task") if "rmse_task" in winners_df.columns else (row.get("rmse") if "rmse" in winners_df.columns else None),
                    "aic": row.get("aic") if "aic" in winners_df.columns else None,
                    "bic": row.get("bic") if "bic" in winners_df.columns else None,
                    "loocv_rmse": row.get("loocv_rmse") if "loocv_rmse" in winners_df.columns else None,
                    # Canonical parameters
                    **canon,
                }

                # Optionally include readout weights if present
                if args.include_readout_weights:
                    for wkey in ("w0", "w1", "w2"):
                        if wkey in params:
                            record[wkey] = params[wkey]
                            extras_param_keys.add(wkey)

                params_rows.append(record)

            # Write CSV with unified, stable column order
            if params_rows:
                canonical_params = ["b", "m1", "m2", "pC1", "pC2"]
                # Base columns in desired order
                base_cols = [
                    "agent",
                    "domain",
                    "version",
                    "prompt_category",
                    "link",
                    "params_tying",
                    "learning_rate",
                    "lr_subdir",
                    "spec_hash",
                    "short_spec_hash",
                    "group_hash",
                    "short_group_hash",
                    # Metrics
                    "loocv_r2",
                    "cv_r2",
                    "r2_task",
                    "rmse_task",
                    "aic",
                    "bic",
                    "loocv_rmse",
                ]
                # Any extra param-like fields beyond canonical params (e.g., readout weights)
                all_keys = set().union(*(r.keys() for r in params_rows))
                extra_params_sorted = [k for k in sorted(all_keys) if k not in set(base_cols) | set(canonical_params)]

                ordered_cols = [c for c in base_cols if c in all_keys] + canonical_params + [k for k in extra_params_sorted if k not in canonical_params]
                params_df = pd.DataFrame(params_rows)
                # Reindex to ordered columns (missing columns will be filled with NaN/None)
                params_df = params_df.reindex(columns=ordered_cols)
                params_df.to_csv(out_eval_dir_param / "winners_with_params.csv", index=False)

        # Write manifest.json for provenance
        manifest = {
            "experiment": exp,
            "tag": tag,
            "versions": [str(v) for v in (versions or [])],
            "models": (args.models or []),
            "prompt_categories": (args.prompt_categories or []),
            "param_counts": (args.param_counts or []),
            "lr": (args.lr or []),
            "include_domains": (args.include_domains or []),
            "exclude_humans": bool(args.exclude_humans),
            "include_readout_weights": bool(args.include_readout_weights),
            "pooled_domain_label": args.pooled_domain_label,
            "humans_mode": args.humans_mode,
        }
        (out_eval_dir_param / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Create LaTeX table
        out_dir = tables_root / exp
        out_dir.mkdir(parents=True, exist_ok=True)
        tex_path = out_dir / f"cbn_best_fits_{tag}.tex"
        lines: List[str] = []
        lines.append("% Auto-generated; do not edit by hand\n")
        lines.append("\\begin{table*}[t]\n\\centering\n")
        lines.append("\\scriptsize\n")
        # Column spec: 2 text columns (Agent, Domain) then numeric right-aligned columns for metrics
        colspec = "ll" + ("r" * len(latex_cols))
        lines.append(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        header_map = {
            "link": "Model",
            "params_tying": "num params",
            "loss": "loss",
            "learning_rate": "LR",
            "rmse": "RMSE",
            "rmse_task": "RMSE",
            "r2": "$R^2$",
            "r2_task": "task $R^2$",
            "mae": "MAE",
            "aic": "AIC",
            "bic": "BIC",
            "loocv_rmse": "LOOCV RMSE",
            "loocv_r2": "LOOCV $R^2$",
            "loocv_mae": "LOOCV MAE",
        }
        # Format headers: make smaller and break multi-word headers into two lines
        def _fmt_header_cell(label: str) -> str:
            lab = header_map.get(label, label)
            parts = lab.split()
            if len(parts) >= 2:
                top = parts[0]
                bottom = " ".join(parts[1:])
                return "{\\tiny \\shortstack{" + top + " \\\\ " + bottom + "}}"
            else:
                return "{\\tiny " + lab + "}"

        headers = [_fmt_header_cell(c) for c in latex_cols]
        lines.append("Agent & Domain & " + " & ".join(headers) + " \\\\ \n\\midrule\n")

        # Determine best values per metric for bolding (minimize or maximize)
        metrics_min = {"loss", "rmse", "rmse_task", "mae", "aic", "bic", "loocv_rmse"}
        metrics_max = {"r2", "r2_task", "loocv_r2"}
        best_fmt_by_col: Dict[str, str] = {}
        for c in latex_cols:
            if c in metrics_min | metrics_max and c in winners_df.columns:
                col = pd.to_numeric(winners_df[c], errors="coerce")
                if col.notna().any():
                    best_val = col.min() if c in metrics_min else col.max()
                    best_fmt_by_col[c] = _fmt_num(best_val)

        for _, r in winners_df.iterrows():
            agent = latex_escape(r.get("agent"))
            dval = r.get("domain")
            if (isinstance(dval, float) and pd.isna(dval)) or dval is None or str(dval).strip() == "":
                dom = latex_escape(args.pooled_domain_label)
            else:
                dom = latex_escape(dval)
            cells: List[str] = []
            for c in latex_cols:
                if c in ("link",):
                    cells.append(latex_escape(r.get(c)))
                elif c == "params_tying":
                    val = r.get(c)
                    try:
                        if pd.isna(val):
                            cells.append("--")
                        else:
                            cells.append(str(int(float(val))))
                    except Exception:
                        cells.append(latex_escape(val))
                elif c == "learning_rate":
                    v = r.get("learning_rate")
                    cells.append(_fmt_num(v))
                else:
                    vstr = _fmt_num(r.get(c))
                    if c in best_fmt_by_col and vstr == best_fmt_by_col[c] and vstr != "--":
                        cells.append("\\textbf{" + vstr + "}")
                    else:
                        cells.append(vstr)
            # Important: four backslashes in Python to emit two backslashes in LaTeX
            lines.append(f"{agent} & {dom} & " + " & ".join(cells) + " \\\\ \n")

        lines.append("\\bottomrule\n\\end{tabular}\n")
        # Avoid math in caption (e.g., $R^2$) to keep hyperref/bookmarks/aux clean
        model_txt = f"; model: {latex_escape(single_model_label)}" if single_model_label else ""
        lr_txt = f"; learning rate: {_fmt_num(single_lr_value)}" if (not add_lr_column and single_lr_value is not None) else ""

        # Build metric grouping for caption (plain text: use R2, not $R^2$)
        insample_set = {"loss", "rmse", "rmse_task", "r2", "r2_task", "mae", "aic", "bic"}
        oos_set = {"loocv_rmse", "loocv_r2", "loocv_mae"}
        insample_metrics = [c for c in latex_cols if c in insample_set]
        oos_metrics = [c for c in latex_cols if c in oos_set]
        cap_name_map = {
            "r2_task": "task R2",
            "r2": "R2",
            "rmse_task": "task RMSE",
            "rmse": "RMSE",
            "mae": "MAE",
            "aic": "AIC",
            "bic": "BIC",
            "loss": "loss",
            "loocv_r2": "LOOCV R2",
            "loocv_rmse": "LOOCV RMSE",
            "loocv_mae": "LOOCV MAE",
        }
        ins_txt = ", ".join(cap_name_map[m] for m in insample_metrics) if insample_metrics else None
        oos_txt = ", ".join(cap_name_map[m] for m in oos_metrics) if oos_metrics else None

        comparability_note = ""
        if ("r2_task" in insample_metrics or "r2" in insample_metrics) and ("loocv_r2" in oos_metrics):
            comparability_note = " In-sample R2 (task-aggregated); selection uses LOOCV R2."

        metrics_clause_parts = []
        if ins_txt:
            metrics_clause_parts.append(f"in-sample: {ins_txt}")
        if oos_txt:
            metrics_clause_parts.append(f"out-of-sample: {oos_txt}")
        metrics_clause = (" Metrics: " + "; ".join(metrics_clause_parts) + ".") if metrics_clause_parts else ""

        cap = (
            f"Best CBN fits per agent for {latex_escape(exp)} (loss: {loss_txt}; optimizer: {opt_txt}{model_txt}{lr_txt}). "
            f"Sorted by LOOCV R2." + metrics_clause + comparability_note
        )
        # Do not escape in label; keep it simple for hyperref/bookmarks
        safe_label = f"{str(exp)}-{str(tag)}".replace("_", "-")
        lines.append(f"\\caption{{{cap}}}\\label{{tab:cbn-best-{safe_label}}}\n")
        lines.append("\\end{table*}\n")
        tex_path.write_text("".join(lines))
        # Report all outputs
        print("Saved outputs:")
        print(f" - winners.csv: {out_eval_dir_param / 'winners.csv'}")
        if args.export_params:
            print(f" - winners_with_params.csv: {out_eval_dir_param / 'winners_with_params.csv'}")
        print(f" - manifest.json: {out_eval_dir_param / 'manifest.json'}")
        print(f" - table: {tex_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
    raise SystemExit(main())
