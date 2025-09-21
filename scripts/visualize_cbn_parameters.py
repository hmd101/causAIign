#!/usr/bin/env python3
"""
Analyze and visualize fitted parameters for winning models (heatmaps only).

This script loads winners_with_params.csv (and the companion winners.csv for metadata)
from results/parameter_analysis/<experiment>/<tag>/ (or legacy results/modelfits as fallback), reshapes parameters, computes summary
statistics, and produces heatmaps of mean fitted parameter values per agent|domain.

Outputs are written under results/parameter_analysis by default.

Usage examples (discovery and selection):
- List matching tags interactively using a glob (quote patterns in zsh):
    python scripts/visualize_cbn_parameters.py --experiment random_abstract --tag-glob 'v1*noisy*'
- Substring convenience (no wildcard provided):
    python scripts/visualize_cbn_parameters.py --experiment rw17_indep_causes --tag-glob 'v2noisy'
    # Interpreted as '*v2noisy*' and shows discovered tag folders to pick from.
- Pick an exact tag directly (no prompt):
    python scripts/visualize_cbn_parameters.py --experiment rw17_indep_causes --tag v2_noisy_or_pcnum_p3-4_lr0.1_hm-pooled
- Non-interactive (CI/batch):
    python scripts/visualize_cbn_parameters.py --experiment rw17_indep_causes --tag-glob 'v2*noisy*' --non-interactive

Metric control (ordering and label value):
- Default metric is LOOCV R² (key: loocv_r2). Rows are ordered by this metric and the value is appended to each y-axis label.
- Choose a different metric (e.g., task R² or LOOCV RMSE):
    python scripts/visualize_cbn_parameters.py --experiment rw17_indep_causes --tag v2_noisy_or_pcnum_p3-4_lr0.1_hm-pooled --metric r2_task
    python scripts/visualize_cbn_parameters.py --experiment rw17_indep_causes --tag v2_noisy_or_pcnum_p3-4_lr0.1_hm-pooled --metric loocv_rmse
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import shutil
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import bundles, fonts

# from tueplots.figsizes import rel_width as tw_rel_width

# w, _ = tw_rel_width(rel_width=0.9)  # 90% of a NeurIPS column width
# fig, ax = plt.subplots(figsize=(w, height))

# NeurIPS-like, LaTeX, serif
config = bundles.neurips2023(nrows=2, ncols=1, rel_width=0.3, usetex=True, family="serif")

config = bundles.neurips2023(
    nrows=2, ncols=1,
    rel_width=0.8,   # <- was 0.3
    usetex=True, family="serif"
)


config["legend.title_fontsize"] = 12
config["font.size"] = 14
config["axes.labelsize"] = 14
config["axes.titlesize"] = 16
config["xtick.labelsize"] = 12
config["ytick.labelsize"] = 12
config["legend.fontsize"] = 12
config["text.latex.preamble"] = r"\usepackage{amsmath,bm,xcolor} \definecolor{inference}{HTML}{FF5B59}"

font_config = fonts.neurips2022_tex(family="serif")
config = {**config, **font_config}

# Make seaborn inherit the rcParams (don’t override fonts)
sns.set_theme(context="paper", style="white", rc=mpl.rcParams)
mpl.rcParams.update(config)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CAblue: RGB (10, 80, 110)
CAblue = (10/255, 80/255, 110/255)       # (0.039, 0.314, 0.431)

# CAlightblue: RGB (58, 160, 171)
CAlightblue = (58/255, 160/255, 171/255) # (0.227, 0.627, 0.671)


def find_winner_dirs(experiments: Optional[List[str]] = None, tag_glob: Optional[str] = None) -> List[Path]:
    """Find tag directories containing both winners_with_params.csv and winners.csv.

    Prefer results/parameter_analysis over results/modelfits for discovery; if none found,
    fall back to results/modelfits. Deduplicate by (experiment, tag).
    """
    bases = [
        PROJECT_ROOT / "results" / "parameter_analysis",
        PROJECT_ROOT / "results" / "modelfits",
    ]
    # Determine experiments across available bases
    exp_candidates: set[str] = set()
    for b in bases:
        if b.exists():
            for p in b.iterdir():
                if p.is_dir():
                    exp_candidates.add(p.name)
    exps = experiments if experiments else sorted(exp_candidates)
    seen: set[tuple[str, str]] = set()
    found: List[Path] = []
    pattern = tag_glob if tag_glob else "*"
    for exp in exps:
        for base in bases:
            exp_dir = base / exp
            if not exp_dir.exists():
                continue
            for tag_dir in exp_dir.glob(pattern):
                if not tag_dir.is_dir():
                    continue
                if (tag_dir / "winners_with_params.csv").exists() and (tag_dir / "winners.csv").exists():
                    key = (exp, tag_dir.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(tag_dir)
    return found


def load_and_merge(tag_dir: Path, experiment: str) -> pd.DataFrame:
    """Load winners_with_params, merge metadata, and return long-form rows.

    Robust to missing params_tying: if absent in winners_with_params.csv, it will be
    merged from winners.csv when available; otherwise the column is omitted.
    """
    params_df = pd.read_csv(tag_dir / "winners_with_params.csv")
    winners_df = pd.read_csv(tag_dir / "winners.csv")

    # Normalize pooled domain label
    params_df["domain"] = params_df["domain"].where(params_df["domain"].notna(), other="all")
    winners_df["domain"] = winners_df["domain"].where(winners_df["domain"].notna(), other="all")

    # Try to bring params_tying over if it is missing in params
    if "params_tying" not in params_df.columns and "params_tying" in winners_df.columns:
        pt = winners_df[["agent", "domain", "params_tying"]].drop_duplicates(["agent", "domain"])
        params_df = params_df.merge(pt, on=["agent", "domain"], how="left")

    # Optional metadata from winners.csv (include metrics to enable ordering/labeling)
    meta_cols = [
        c
        for c in [
            "link",
            "prompt_category",
            "version",
            "learning_rate",
            "params_tying",
            # Metrics commonly present in winners.csv
            "loocv_r2",
            "loocv_rmse",
            "r2_task",
            "rmse_task",
            "r2",
            "rmse",
        ]
        if c in winners_df.columns
    ]
    # If params_tying already exists in params_df, don't pull it again to avoid _x/_y collisions
    if "params_tying" in params_df.columns and "params_tying" in meta_cols:
        meta_cols = [c for c in meta_cols if c != "params_tying"]
    key_cols = ["agent", "domain"]
    win_meta = winners_df[key_cols + meta_cols].drop_duplicates(key_cols)

    merged = params_df.merge(win_meta, on=key_cols, how="left")
    merged["experiment"] = experiment
    merged["tag"] = tag_dir.name
    # Coalesce metric columns to canonical names to avoid _x/_y collisions after merge
    metric_keys = [
        "loocv_r2",
        "loocv_rmse",
        "r2_task",
        "rmse_task",
        "r2",
        "rmse",
        "cv_r2",
    ]
    for k in metric_keys:
        kx, ky = f"{k}_x", f"{k}_y"
        if k in merged.columns:
            continue
        if kx in merged.columns or ky in merged.columns:
            left = merged[kx] if kx in merged.columns else None
            right = merged[ky] if ky in merged.columns else None
            if left is not None and right is not None:
                merged[k] = left.combine_first(right)
            elif left is not None:
                merged[k] = left
            elif right is not None:
                merged[k] = right
            # Drop the suffixed columns if they exist
            if kx in merged.columns:
                merged = merged.drop(columns=[kx])
            if ky in merged.columns:
                merged = merged.drop(columns=[ky])
    # If exporter recorded humans_mode in manifest, propagate as a column for downstream filters
    manifest_path = tag_dir / "manifest.json"
    if manifest_path.exists():
        try:
            import json as _json
            _m = _json.loads(manifest_path.read_text())
            hm = _m.get("humans_mode")
            if hm:
                merged["humans_mode"] = hm
        except Exception:
            pass

    # Build id_vars dynamically and melt parameter columns
    id_vars = ["experiment", "tag", "agent", "domain"]
    if "params_tying" in merged.columns:
        id_vars.append("params_tying")
    # Add available meta columns (avoid duplicates)
    id_vars += [c for c in meta_cols if c in merged.columns and c not in id_vars]

    exclude = set(id_vars)
    # Candidate parameter columns = remaining numeric-looking columns excluding known artifacts
    param_cols_all = [c for c in merged.columns if c not in exclude]
    # Drop any accidental merge artifacts or tying-related columns
    bad_substrings = ("params_tying",)
    bad_suffixes = ("_x", "_y")
    param_cols_all = [
        c for c in param_cols_all
        if (not any(bs in c for bs in bad_substrings)) and (not any(c.endswith(suf) for suf in bad_suffixes))
    ]
    # Restrict to the five model parameters if available (ordered)
    five_params_order = ["b", "m1", "m2", "pC1", "pC2"]
    param_cols = [p for p in five_params_order if p in param_cols_all]
    if not param_cols:
        # Fallback: use up to 5 remaining columns sorted for stability
        param_cols = sorted(param_cols_all)[:5]

    long_df = merged.melt(id_vars=id_vars, value_vars=param_cols, var_name="parameter", value_name="value")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])  # keep numeric values only
    return long_df


def summarize_params(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """Aggregate parameter values by group and parameter, computing summary stats."""
    agg = (
        df.groupby(group_by + ["parameter"]).agg(
            n=("value", "count"),
            mean=("value", "mean"),
            std=("value", "std"),
            min=("value", "min"),
            p25=("value", lambda x: np.percentile(x, 25)),
            median=("value", "median"),
            p75=("value", lambda x: np.percentile(x, 75)),
            max=("value", "max"),
        ).reset_index()
    )
    return agg


# Violin plots removed: focus is solely on heatmaps


def _parse_human_id(agent: str) -> Optional[int]:
    """Extract integer human subject id from agent like 'human-12' or 'humans-34'."""
    s = str(agent).strip().lower()
    if s.startswith("human-"):
        suffix = s.split("-", 1)[1]
    elif s.startswith("humans-"):
        suffix = s.split("-", 1)[1]
    else:
        return None
    try:
        return int(suffix)
    except Exception:
        return None


def _metric_display_name(metric_key: str) -> str:
    mk = metric_key.lower().replace("-", "_")
    mapping = {
        "loocv_r2": "LOOCV R²",
        "loocv_rmse": "LOOCV RMSE",
        "r2_task": "Task R²",
        "rmse_task": "Task RMSE",
        "r2": "R²",
        "rmse": "RMSE",
    }
    return mapping.get(mk, metric_key)


# def plot_heatmaps(df: pd.DataFrame, out_dir: Path, by: str = "agent", split_by: Optional[str] = None, metric: str = "loocv_r2") -> None:
#     """Heatmaps of mean parameter values per agent|domain (no faceting)."""
#     out_dir.mkdir(parents=True, exist_ok=True)
#     sub = df.copy()
#     # Base key: Agent | Domain
#     sub["row_key"] = sub[by].astype(str) + " | " + sub["domain"].astype(str)
#     # Count number of CBN parameters per Agent|Domain: prefer 'params_tying' if present, else fallback to distinct parameter names
#     param_counts: dict[str, int] = {}
#     try:
#         if "params_tying" in sub.columns:
#             tmp = sub[["row_key", "params_tying"]].copy()
#             tmp["params_tying"] = pd.to_numeric(tmp["params_tying"], errors="coerce")
#             # Take the first non-null per row_key
#             pc = tmp.dropna(subset=["params_tying"]).groupby("row_key")["params_tying"].first().astype(int)
#             param_counts = pc.to_dict()
#         else:
#             param_counts = sub.groupby("row_key")["parameter"].nunique().astype(int).to_dict()
#     except Exception:
#         try:
#             param_counts = sub.groupby("row_key")["parameter"].nunique().astype(int).to_dict()
#         except Exception:
#             param_counts = {}
#     # Compute ordering scores (default: LOOCV R²)
#     metric_key = (metric or "loocv_r2").strip()
#     metric_key_norm = metric_key.lower().replace("-", "_")
#     used_metric_name = None
#     row_scores: dict[str, Optional[float]] = {}
#     ordered_row_labels: Optional[List[str]] = None
#     if metric_key_norm in {c.lower() for c in sub.columns}:
#         try:
#             # Align actual column name matching metric_key_norm
#             col_map = {c.lower(): c for c in sub.columns}
#             metric_col = col_map[metric_key_norm]
#             tmp = sub[["row_key", metric_col]].copy()
#             tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
#             scores = tmp.groupby("row_key")[metric_col].mean().to_dict()
#             row_scores = {k: (float(v) if pd.notna(v) else None) for k, v in scores.items()}
#             # Determine sort direction: RMSE-like -> ascending (lower better), else descending
#             ml = metric_key_norm
#             is_rmse_like = ("rmse" in ml) or ("error" in ml)
#             def sort_key(item: tuple[str, Optional[float]]):
#                 rname, val = item
#                 is_nan = (val is None) or (isinstance(val, float) and np.isnan(val))
#                 # For ascending metrics (lower better), use val; for descending metrics (higher better), use -val
#                 comp = float(val if val is not None else 0.0)
#                 comp = comp if is_rmse_like else -comp
#                 return (1 if is_nan else 0, comp, rname)
#             ordered_keys = [r for r, _ in sorted(row_scores.items(), key=sort_key)]
#             used_metric_name = _metric_display_name(metric_key_norm)
#             # Build human-friendly labels with metric values
#             def _mk_label(k: str) -> str:
#                 val = row_scores.get(k)
#                 if val is None or (isinstance(val, float) and np.isnan(val)):
#                     val_str = "--"
#                 else:
#                     val_str = f"{val:.3f}"
#                 # Include number of CBN params between domain and metric value
#                 pc = param_counts.get(k)
#                 try:
#                     pc_str = f"{int(pc)}" if pc is not None and not (isinstance(pc, float) and np.isnan(pc)) else "--"
#                 except Exception:
#                     pc_str = "--"
#                 # Format: Agent | Domain | num CBN params | metric value
#                 return f"{k} | {pc_str} | {val_str}"
#             label_map = {k: _mk_label(k) for k in row_scores.keys()}
#             sub["row_label"] = sub["row_key"].map(lambda k: label_map.get(str(k), str(k)))
#             ordered_row_labels = [label_map[k] for k in ordered_keys]
#         except Exception:
#             used_metric_name = None
#             # Fallback label includes number of params if available
#             def _mk_label_nom(k: str) -> str:
#                 pc = param_counts.get(k)
#                 try:
#                     pc_str = f"{int(pc)}" if pc is not None and not (isinstance(pc, float) and np.isnan(pc)) else "--"
#                 except Exception:
#                     pc_str = "--"
#                 return f"{k} | {pc_str}"
#             sub["row_label"] = sub["row_key"].map(lambda k: _mk_label_nom(str(k)))
#     else:
#         # Metric column not present; include number of params in labels
#         def _mk_label_nom(k: str) -> str:
#             pc = param_counts.get(k)
#             try:
#                 pc_str = f"{int(pc)}" if pc is not None and not (isinstance(pc, float) and np.isnan(pc)) else "--"
#             except Exception:
#                 pc_str = "--"
#             return f"{k} | {pc_str}"
#         sub["row_label"] = sub["row_key"].map(lambda k: _mk_label_nom(str(k)))

#     piv = sub.pivot_table(index="row_label", columns="parameter", values="value", aggfunc="mean")
#     if ordered_row_labels:
#         piv = piv.reindex(ordered_row_labels)

#     # >>> NEW: LaTeX-ify x-axis parameter names
#     piv = piv.rename(columns={c: _latex_param(c) for c in piv.columns})

#     plt.figure(figsize=(max(8, 0.7 * len(piv.columns)), max(6, 0.3 * len(piv.index))))
#     ax = sns.heatmap(piv, annot=True, fmt=".3f", cmap="RdYlBu_r", cbar_kws={"label": r"Fitted Parameter Value $\in [0,1]$"})
#     title = "CBN Parameter Values"
#     if used_metric_name:
#         title += f" — ordered by {used_metric_name}"
#     # Add a subtitle line indicating which metric is shown in the row labels
#     # if used_metric_name:
#     #     # ax.set_title(f"{title}\nMetric: {used_metric_name}")

#     # else:
#     #     ax.set_title(title)
#     if used_metric_name:
#         ax.set_ylabel(f"Agent | Domain | #CBN params | Metric: {used_metric_name}")
#     else:
#         ax.set_ylabel("Agent | Domain | #CBN params")
#     ax.set_xlabel("CBN Parameter")
#     plt.tight_layout()
#     # Determine tag suffix for filenames
#     tag_suffix = ""
#     if "tag" in sub.columns:
#         uniq_tags = sorted({str(t) for t in sub["tag"].dropna().unique().tolist()})
#         if len(uniq_tags) == 1:
#             tag_suffix = f"_{uniq_tags[0]}"
#         elif len(uniq_tags) > 1:
#             tag_suffix = "_multi"
#     for ext in ("png", "pdf"):
#         plt.savefig(out_dir / f"heatmap_{tag_suffix}.{ext}", dpi=300, bbox_inches="tight")
#     plt.close()


# NEURIPSY like heatmaps 


def latex_escape(s: str) -> str:
    """Escape LaTeX specials in plain text segments."""
    return re.sub(r'([&%$#_\{\}~\^])', r'\\\1', str(s))

def _metric_display_name(metric_key_norm: str) -> str:
    return metric_key_norm.replace("_", " ")

def _latex_param(name: str) -> str:
    name = str(name)
    if name == "b":
        return r"$b$"
    m = re.fullmatch(r"m(\d+)", name)
    if m:
        return rf"$m_{{{m.group(1)}}}$"
    p = re.fullmatch(r"pC(\d+)", name)
    if p:
        return rf"$p(C_{{{p.group(1)}}})$"
    return rf"${name}$"

def _mk_row_label_tex(agent: str, domain: str, pc_str: str, val_str: Optional[str] = None) -> str:
    """Build a TeX-safe row label with vertical bars via \\textbar."""
    sep = r" \textbar\ "
    left = latex_escape(agent) + sep + latex_escape(domain) + sep + pc_str
    return left if val_str is None else (left + sep + val_str)

def plot_heatmaps(df: pd.DataFrame, out_dir: Path, by: str = "agent",
                  split_by: Optional[str] = None, metric: str = "loocv_r2") -> None:
    """Heatmaps of mean parameter values per agent|domain (no faceting)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df.copy()

    # Base key: "agent | domain" (plain, just for grouping)
    sub["row_key"] = sub[by].astype(str) + " | " + sub["domain"].astype(str)

    # Count CBN params per row_key
    param_counts: Dict[str, int] = {}
    try:
        if "params_tying" in sub.columns:
            tmp = sub[["row_key", "params_tying"]].copy()
            tmp["params_tying"] = pd.to_numeric(tmp["params_tying"], errors="coerce")
            pc = tmp.dropna(subset=["params_tying"]).groupby("row_key")["params_tying"].first().astype(int)
            param_counts = pc.to_dict()
        else:
            param_counts = sub.groupby("row_key")["parameter"].nunique().astype(int).to_dict()
    except Exception:
        try:
            param_counts = sub.groupby("row_key")["parameter"].nunique().astype(int).to_dict()
        except Exception:
            param_counts = {}

    # Compute ordering and TeX row labels
    metric_key = (metric or "loocv_r2").strip()
    metric_key_norm = metric_key.lower().replace("-", "_")
    used_metric_name = None
    row_scores: Dict[str, Optional[float]] = {}
    ordered_row_labels: Optional[List[str]] = None

    if metric_key_norm in {c.lower() for c in sub.columns}:
        col_map = {c.lower(): c for c in sub.columns}
        metric_col = col_map[metric_key_norm]
        tmp = sub[["row_key", metric_col]].copy()
        tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
        scores = tmp.groupby("row_key")[metric_col].mean().to_dict()
        row_scores = {k: (float(v) if pd.notna(v) else None) for k, v in scores.items()}

        is_rmse_like = ("rmse" in metric_key_norm) or ("error" in metric_key_norm)
        def sort_key(item):
            k, v = item
            is_nan = (v is None) or (isinstance(v, float) and np.isnan(v))
            comp = float(v if v is not None else 0.0)
            comp = comp if is_rmse_like else -comp
            return (1 if is_nan else 0, comp, k)

        ordered_keys = [r for r, _ in sorted(row_scores.items(), key=sort_key)]
        used_metric_name = _metric_display_name(metric_key_norm)

        def _mk_label(k: str) -> str:
            # k is "agent | domain"
            agent, domain = k.split(" | ", 1)
            val = row_scores.get(k)
            val_str = "--" if (val is None or (isinstance(val, float) and np.isnan(val))) else f"{val:.3f}"
            pc = param_counts.get(k)
            try:
                pc_str = f"{int(pc)}" if pc is not None and not (isinstance(pc, float) and np.isnan(pc)) else "--"
            except Exception:
                pc_str = "--"
            return _mk_row_label_tex(agent, domain, pc_str, val_str)

        label_map = {k: _mk_label(k) for k in row_scores.keys()}
        sub["row_label"] = sub["row_key"].map(lambda k: label_map.get(str(k), str(k)))
        ordered_row_labels = [label_map[k] for k in ordered_keys]
    else:
        def _mk_label_nom(k: str) -> str:
            agent, domain = k.split(" | ", 1)
            pc = param_counts.get(k)
            try:
                pc_str = f"{int(pc)}" if pc is not None and not (isinstance(pc, float) and np.isnan(pc)) else "--"
            except Exception:
                pc_str = "--"
            return _mk_row_label_tex(agent, domain, pc_str)
        sub["row_label"] = sub["row_key"].map(lambda k: _mk_label_nom(str(k)))

    # Pivot (rows already TeX strings), and LaTeX-ify x-axis parameter names
    piv = sub.pivot_table(index="row_label", columns="parameter", values="value", aggfunc="mean")
    if ordered_row_labels:
        piv = piv.reindex(ordered_row_labels)
    piv = piv.rename(columns={c: _latex_param(c) for c in piv.columns})


    #the block below works well for rw17)indep_caues
    # # Figure size: keep NeurIPS width; increase height (less squeeze)
    # base_w, _ = mpl.rcParams["figure.figsize"]
    # height = max(4.2, 0.36 * len(piv.index))   # taller than before
    # fig, ax = plt.subplots(figsize=(base_w, height), layout="constrained")

    # # Heatmap
    # ax = sns.heatmap(
    #     piv, annot=True, fmt=".3f", cmap="RdYlBu_r",
    #     cbar_kws={"label": r"Fitted Parameter Value $\in [0,1]$", "pad": 0.02},
    #     ax=ax
    # )

    # # Axis labels (TeX-safe). Show LOOCV R^2 in math.
    # if used_metric_name:
    #     metric_tex = r"$\mathrm{LOOCV}\ R^2$" if metric_key_norm == "loocv_r2" else latex_escape(used_metric_name)
    #     ax.set_ylabel(r"Agent \textbar\ Domain \textbar\ \# CBN params \textbar\ Metric: " + metric_tex)
    # else:
    #     ax.set_ylabel(r"Agent \textbar\ Domain \textbar\ \# CBN params")

    # ax.set_xlabel("CBN Parameter")

    # # Ticks: use our TeX row labels; keep x labels horizontal
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    # ax.set_yticklabels(list(piv.index), rotation=0)


    # --- sizing based on table shape ---
    # --- sizing based on label length and table shape ---
    n_rows, n_cols = len(piv.index), len(piv.columns)

    # NeurIPS base width from tueplots rcParams
    base_w, _ = mpl.rcParams["figure.figsize"]

    # 1) Estimate left margin from y-tick label length (inches)
    #    Rule of thumb: avg char width ≈ 0.55 * font_size points; 72 pt = 1 inch
    ytick_fs = mpl.rcParams.get("ytick.labelsize", 12)
    max_label_chars = max(len(str(s)) for s in piv.index) if n_rows > 0 else 0
    avg_char_width_in = 0.55 * float(ytick_fs) / 72.0
    left_margin = 0.30 + avg_char_width_in * max_label_chars   # 0.30" padding
    left_margin = max(left_margin, 1.4)                         # hard floor
    left_margin = min(left_margin, 5.0)                         # hard ceiling

    # 2) Per-column cell width (be generous)
    if   n_cols <= 6:   cell_w = 0.8
    elif n_cols <= 10:  cell_w = 0.70
    elif n_cols <= 14:  cell_w = 0.58
    elif n_cols <= 18:  cell_w = 0.50
    else:               cell_w = 0.44

    # 3) Colorbar + right padding (inches)
    right_margin = 0.80

    # 4) Height per row (inches)
    cell_h = 0.40
    top_bottom_pad = 0.9
    fig_h = max(4.8, n_rows * cell_h + top_bottom_pad)

    # 5) Final width: left margin + heatmap + right margin,
    #    but at least some multiplier of the NeurIPS base width
    width_needed = left_margin + n_cols * cell_w + right_margin
    min_multiplier = 1.75                                     # <- widen aggressively
    fig_w = max(width_needed, base_w * min_multiplier)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    # --- annotations: adapt font/precision to density ---
    if   n_cols <= 8:   ann_size, ann_fmt = 11, ".3f"
    elif n_cols <= 12:  ann_size, ann_fmt = 9,  ".3f"
    elif n_cols <= 18:  ann_size, ann_fmt = 8,  ".3f"
    else:               ann_size, ann_fmt = 7,  ".2f"

    ax = sns.heatmap(
        piv,
        annot=True,
        fmt=ann_fmt,
        annot_kws={"fontsize": ann_size},
        cmap="RdYlBu_r",
        square=False,   # don't force square cells
        cbar_kws={"label": r"Fitted Parameter Value $\in [0,1]$", "pad": 0.02, "shrink": 0.9},
        ax=ax,
    )

    # Axis labels (unchanged)
    if used_metric_name:
        metric_tex = r"Reasoning Consistency ($\mathrm{LOOCV}\ R^2$)" if metric_key_norm == "loocv_r2" else latex_escape(used_metric_name)
        ax.set_ylabel(r"Agent \textbar\ Domain \textbar\ \# CBN Parameters \textbar\ " + metric_tex)
    else:
        ax.set_ylabel(r"Agent \textbar\ Domain \textbar\ \# CBN Parameters")
    ax.set_xlabel("Causal Bayes Net (CBN) Parameter")

    # Ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(list(piv.index), rotation=0)

    # n_rows, n_cols = len(piv.index), len(piv.columns)

    # # base size from tueplots; we’ll scale width if needed
    # base_w, _ = mpl.rcParams["figure.figsize"]

    # # per-cell target sizes (inches)
    # cell_h = 0.36                         # row height
    # cell_w = 0.55 if n_cols <= 8 else 0.45 if n_cols <= 12 else 0.38

    # # left/right margins (inches) – leave room for long y labels + colorbar
    # left_margin  = 2.0                    # y tick labels can be long
    # right_margin = 0.7                    # colorbar
    # width_needed  = left_margin + n_cols * cell_w + right_margin
    # height_needed = max(4.2, n_rows * cell_h + 0.8)

    # # final width: at least base_w, otherwise grow as needed
    # fig_w = max(base_w, width_needed)
    # fig_h = height_needed

    # fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    # # --- annotations: adapt font/precision to density ---
    # if n_cols <= 8:
    #     ann_size, ann_fmt = 10, ".3f"
    # elif n_cols <= 12:
    #     ann_size, ann_fmt = 8, ".3f"
    # else:
    #     ann_size, ann_fmt = 7, ".2f"      # a bit coarser when very dense

    # ax = sns.heatmap(
    #     piv,
    #     annot=True,
    #     fmt=ann_fmt,
    #     annot_kws={"fontsize": ann_size},
    #     cmap="RdYlBu_r",
    #     square=False,                      # don’t force square cells
    #     cbar_kws={"label": r"Fitted Parameter Value $\in [0,1]$", "pad": 0.02, "shrink": 0.9},
    #     ax=ax,
    # )

    # # labels (unchanged logic except now with more space)
    # if used_metric_name:
    #     metric_tex = r"$\mathrm{LOOCV}\ R^2$" if metric_key_norm == "loocv_r2" else latex_escape(used_metric_name)
    #     ax.set_ylabel(r"Agent \textbar\ Domain \textbar\ \# CBN params \textbar\ Metric: " + metric_tex)
    # else:
    #     ax.set_ylabel(r"Agent \textbar\ Domain \textbar\ \# CBN params")

    # ax.set_xlabel("CBN Parameter")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    # ax.set_yticklabels(list(piv.index), rotation=0)


    # Save
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)




# def _metric_display_name(metric_key_norm: str) -> str:
#     # your existing helper; keep as-is
#     return metric_key_norm.replace("_", " ")

# def _latex_param(name: str) -> str:
#     name = str(name)
#     if name == "b":
#         return r"$b$"
#     m = re.fullmatch(r"m(\d+)", name)
#     if m:
#         return rf"$m_{{{m.group(1)}}}$"
#     p = re.fullmatch(r"pC(\d+)", name)
#     if p:
#         return rf"$p(C_{{{p.group(1)}}})$"
#     return rf"${name}$"

# def plot_heatmaps(df: pd.DataFrame, out_dir: Path, by: str = "agent",
#                   split_by: Optional[str] = None, metric: str = "loocv_r2") -> None:
#     """Heatmaps of mean parameter values per agent|domain (no faceting)."""
#     out_dir.mkdir(parents=True, exist_ok=True)
#     sub = df.copy()

#     # Base key: Agent | Domain
#     sub["row_key"] = sub[by].astype(str) + " | " + sub["domain"].astype(str)

#     # Count CBN params per row_key (your existing logic, unchanged)
#     param_counts: dict[str, int] = {}
#     try:
#         if "params_tying" in sub.columns:
#             tmp = sub[["row_key", "params_tying"]].copy()
#             tmp["params_tying"] = pd.to_numeric(tmp["params_tying"], errors="coerce")
#             pc = tmp.dropna(subset=["params_tying"]).groupby("row_key")["params_tying"].first().astype(int)
#             param_counts = pc.to_dict()
#         else:
#             param_counts = sub.groupby("row_key")["parameter"].nunique().astype(int).to_dict()
#     except Exception:
#         try:
#             param_counts = sub.groupby("row_key")["parameter"].nunique().astype(int).to_dict()
#         except Exception:
#             param_counts = {}

#     # Compute ordering scores (default: LOOCV R^2), build row labels
#     metric_key = (metric or "loocv_r2").strip()
#     metric_key_norm = metric_key.lower().replace("-", "_")
#     used_metric_name = None
#     row_scores: dict[str, Optional[float]] = {}
#     ordered_row_labels: Optional[list[str]] = None

#     if metric_key_norm in {c.lower() for c in sub.columns}:
#         col_map = {c.lower(): c for c in sub.columns}
#         metric_col = col_map[metric_key_norm]
#         tmp = sub[["row_key", metric_col]].copy()
#         tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
#         scores = tmp.groupby("row_key")[metric_col].mean().to_dict()
#         row_scores = {k: (float(v) if pd.notna(v) else None) for k, v in scores.items()}

#         is_rmse_like = ("rmse" in metric_key_norm) or ("error" in metric_key_norm)
#         def sort_key(item):
#             k, v = item
#             is_nan = (v is None) or (isinstance(v, float) and np.isnan(v))
#             comp = float(v if v is not None else 0.0)
#             comp = comp if is_rmse_like else -comp
#             return (1 if is_nan else 0, comp, k)

#         ordered_keys = [r for r, _ in sorted(row_scores.items(), key=sort_key)]
#         used_metric_name = _metric_display_name(metric_key_norm)

#         def _mk_label(k: str) -> str:
#             val = row_scores.get(k)
#             val_str = "--" if (val is None or (isinstance(val, float) and np.isnan(val))) else f"{val:.3f}"
#             pc = param_counts.get(k)
#             try:
#                 pc_str = f"{int(pc)}" if pc is not None and not (isinstance(pc, float) and np.isnan(pc)) else "--"
#             except Exception:
#                 pc_str = "--"
#             return f"{k} | {pc_str} | {val_str}"

#         label_map = {k: _mk_label(k) for k in row_scores.keys()}
#         sub["row_label"] = sub["row_key"].map(lambda k: label_map.get(str(k), str(k)))
#         ordered_row_labels = [label_map[k] for k in ordered_keys]
#     else:
#         def _mk_label_nom(k: str) -> str:
#             pc = param_counts.get(k)
#             try:
#                 pc_str = f"{int(pc)}" if pc is not None and not (isinstance(pc, float) and np.isnan(pc)) else "--"
#             except Exception:
#                 pc_str = "--"
#             return f"{k} | {pc_str}"
#         sub["row_label"] = sub["row_key"].map(lambda k: _mk_label_nom(str(k)))

#     # Pivot and LaTeX-ify parameter names for the x-axis
#     piv = sub.pivot_table(index="row_label", columns="parameter", values="value", aggfunc="mean")
#     if ordered_row_labels:
#         piv = piv.reindex(ordered_row_labels)
#     piv = piv.rename(columns={c: _latex_param(c) for c in piv.columns})

#     # Figure size: width from NeurIPS column, height from rows

#     # base_w, _ = mpl.rcParams["figure.figsize"]
  
#     # fig, ax = plt.subplots(figsize=(base_w, height))

#     # # Heatmap
#     # ax = sns.heatmap(
#     #     piv, annot=True, fmt=".3f", cmap="RdYlBu_r",
#     #     cbar_kws={"label": r"Fitted Parameter Value $\in [0,1]$"},
#     #     ax=ax
#     # )


#     base_w, _ = mpl.rcParams["figure.figsize"]
#     height = max(3.0, 0.28 * len(piv.index))  # scale per row (adjust 0.25–0.35 to taste)

#     fig, ax = plt.subplots(figsize=(base_w, height), layout="constrained")  # <- use one engine

#     ax = sns.heatmap(
#         piv, annot=True, fmt=".3f", cmap="RdYlBu_r",
#         cbar_kws={"label": r"Fitted Parameter Value $\in [0,1]$", "pad": 0.02},  # optional: tighter cbar spacing
#         ax=ax
#     )


#     # Axes labels and title (TeX-safe; avoid & and Unicode em-dash)
#     title = "CBN Parameter Values"
#     if used_metric_name:
#         title += rf" --- ordered by {used_metric_name}"
#         ax.set_ylabel(r"Agent \textbar\ Domain \textbar\ \# CBN params \textbar\ Metric: " + latex_escape(used_metric_name))
#     else:
#         ax.set_ylabel(r"Agent \textbar\ Domain \textbar\ #CBN params")

#     ax.set_xlabel("CBN Parameter")

#     # Ticks: keep x-labels horizontal and nicely centered
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

#     # Slightly smaller annotation text if crowded (optional)
#     # for t in ax.texts: t.set_fontsize(10)

#     # fig.tight_layout()
#     for ext in ("png", "pdf"):
#         fig.savefig(out_dir / f"heatmap.{ext}", dpi=300, bbox_inches="tight")
#     plt.close(fig)


def plot_correlations(df: pd.DataFrame, out_dir: Path, split_by: str = "link") -> None:
    """Deprecated: correlations disabled as focus is on heatmaps only."""
    return


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point.

    Args:
      --experiments [NAMES ...]   : Experiments to include (default: discover all).
      --experiment NAME           : Alias for a single experiment; same as --experiments NAME.
    --tag-glob GLOB             : Glob for tags (quote in zsh).
    --tag TAG                   : Exact tag to include (single tag); overrides --tag-glob.
      --agents [AGENTS ...]       : Only include listed agents (use 'all' to include all).
      --exclude-agents [NAMES ...]: Exclude these agents (alias: --exclude).
      --domains [DOMAINS ...]     : Only include listed domains.
      --pooled-only               : Keep only pooled (domain == 'all').
    --out-root PATH             : Output root (default: results/parameter_analysis).
    --no-plots                  : Skip plots.
    """
    ap = argparse.ArgumentParser(description="Analyze and visualize fitted parameters for winning models")
    ap.add_argument("--experiments", nargs="*", help="Experiments to include; default: discover all with winners files")
    ap.add_argument("--experiment", help="Single experiment alias; equivalent to --experiments NAME")
    ap.add_argument("--tag-glob", help="Glob to match tags (e.g., 'v2*noisy*'); quote in zsh")
    ap.add_argument("--tag", help="Exact tag to include (single tag); overrides --tag-glob")
    ap.add_argument("--non-interactive", action="store_true", help="Do not prompt; if multiple matches, proceed with all (may average across tags)")
    ap.add_argument("--agents", nargs="*", help="Filter to these agents; use 'all' to include all agents")
    ap.add_argument("--exclude-agents", "--exclude", nargs="*", dest="exclude_agents", help="Agents to exclude (case-insensitive)")
    ap.add_argument("--domains", nargs="*", help="Filter to these domains; use 'all' for pooled")
    ap.add_argument("--pooled-only", action="store_true", help="Filter to pooled rows only (domain aggregated as 'all')")
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "results" / "parameter_analysis"), help="Output root directory for analyses")
    ap.add_argument("--humans-mode", choices=["all", "aggregated", "pooled", "individual"], default="all", help="Filter human agents by mode; non-human agents are always included. When set, also narrows tags if they carry hm-* suffix.")
    ap.add_argument("--metric", default="loocv_r2", help="Metric to order agents by and display in y-axis labels (e.g., loocv_r2, loocv_rmse, r2_task, rmse_task, r2, rmse)")
    ap.add_argument("--min-human-tasks", type=int, default=0, help="If >0, keep only individual humans with at least this many tasks (requires coverage CSV; non-human agents unaffected)")
    ap.add_argument("--human-coverage-csv", type=str, help="Optional path to human coverage summary CSV (with columns human_subj_id,n_tasks). If not provided, will look for results/human_coverage/<experiment>_human_prompt_coverage_summary.csv")
    ap.add_argument("--no-plots", action="store_true", help="Only compute summaries, skip plots")
    args = ap.parse_args(argv)

    if args.experiment and not args.experiments:
        args.experiments = [args.experiment]

    # Determine search pattern; if user passed --tag-glob without any wildcard, interpret as substring match
    pattern = args.tag if args.tag else args.tag_glob
    if args.tag_glob and not any(ch in args.tag_glob for ch in "*?[]"):
        # Auto-upgrade to substring search and inform the user
        pattern = f"*{args.tag_glob}*"
        print(f"[INFO] Interpreting --tag-glob '{args.tag_glob}' as '{pattern}' (substring match)")
    # Validate experiments exist; collect available experiments across bases
    bases = [PROJECT_ROOT / "results" / "parameter_analysis", PROJECT_ROOT / "results" / "modelfits"]
    available_exps: set[str] = set()
    for b in bases:
        if b.exists():
            for p in b.iterdir():
                if p.is_dir():
                    available_exps.add(p.name)
    if args.experiments:
        missing = [e for e in args.experiments if e not in available_exps]
        if missing:
            print("Experiment folder not found:")
            for e in missing:
                print(f"  - {e}")
            if available_exps:
                print("Available experiments:")
                for e in sorted(available_exps):
                    print(f"  - {e}")
            return 1

    tag_dirs = find_winner_dirs(args.experiments, pattern)
    # If humans-mode requested and discovering multiple tags, prefer those suffixed with hm-* matching the mode
    if args.humans_mode != "all":
        suffix_map = {"aggregated": "hm-agg", "pooled": "hm-pooled", "individual": "hm-indiv"}
        suf = suffix_map.get(args.humans_mode)
        if suf:
            narrowed = [td for td in tag_dirs if td.name.endswith(suf)]
            if narrowed:
                tag_dirs = narrowed
    if not tag_dirs:
        # No matches with current filters. If experiment(s) provided, list available tag subfolders within them.
        print("No winners_with_params.csv + winners.csv pairs found.")
        if args.experiments:
            for exp in args.experiments:
                print(f"\nAvailable tag folders under experiment '{exp}':")
                any_listed = False
                for b in bases:
                    exp_dir = b / exp
                    if not exp_dir.exists():
                        continue
                    for td in sorted([p for p in exp_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                        has_winners = (td / "winners.csv").exists() and (td / "winners_with_params.csv").exists()
                        mark = "(has winners)" if has_winners else ""
                        try:
                            rel = td.relative_to(PROJECT_ROOT)
                        except Exception:
                            rel = td
                        print(f"  - {rel} {mark}")
                        any_listed = True
                if not any_listed:
                    print("  (no subfolders)")
            # If a glob was provided, suggest close matches among tag names with winners
            if args.tag_glob and args.experiments:
                cleaned = args.tag_glob.replace("*", "").replace("?", "")
                if cleaned:
                    candidates: list[str] = []
                    for exp in args.experiments:
                        for b in bases:
                            exp_dir = b / exp
                            if not exp_dir.exists():
                                continue
                            for td in exp_dir.iterdir():
                                if td.is_dir() and (td / "winners.csv").exists() and (td / "winners_with_params.csv").exists():
                                    candidates.append(td.name)
                    if candidates:
                        suggestions = difflib.get_close_matches(cleaned, sorted(set(candidates)), n=3, cutoff=0.5)
                        if suggestions:
                            print("\nDid you mean one of these tag names?")
                            for s in suggestions:
                                print(f"  - {s}")
            return 1

    # If user used --tag-glob, list all discovered subfolders and optionally prompt to select one
    if args.tag_glob:
        print("Discovered candidate tag folders (with winners.csv + winners_with_params.csv):")
        for i, td in enumerate(tag_dirs, start=1):
            # Show relative path from project root for readability
            try:
                rel = td.relative_to(PROJECT_ROOT)
            except Exception:
                rel = td
            print(f"  [{i}] {rel}")
        if len(tag_dirs) > 1 and not args.tag and not args.non_interactive:
            # Prompt the user to pick one tag directory by index
            while True:
                sel = input(f"Select one by number (1-{len(tag_dirs)}), or press Enter to keep all: ").strip()
                if sel == "":
                    break
                if sel.isdigit():
                    idx = int(sel)
                    if 1 <= idx <= len(tag_dirs):
                        tag_dirs = [tag_dirs[idx - 1]]
                        break
                print("Invalid selection. Please enter a valid number or press Enter.")

    # Warn if multiple tag directories remain; list them for clarity
    matched_tags = sorted({td.name for td in tag_dirs})
    if len(matched_tags) > 1:
        print("[WARN] Multiple tags matched. Aggregating across tags will average parameter values in heatmaps.")
        print("[WARN] Matched tags:")
        for t in matched_tags:
            print(f"  - {t}")
        print("[WARN] To avoid averaging across tags, pass --tag <exact_tag>." )

    all_long: List[pd.DataFrame] = []
    for tag_dir in tag_dirs:
        exp = tag_dir.parent.name
        long_df = load_and_merge(tag_dir, exp)
        all_long.append(long_df)

    df = pd.concat(all_long, ignore_index=True)

    # Humans-mode agent filtering (keep non-human agents; restrict humans accordingly)
    if args.humans_mode != "all" and "agent" in df.columns:
        def _is_keep(agent: str) -> bool:
            s = str(agent).strip().lower()
            is_aggr = s == "humans"
            is_pooled = s in {"humans-pooled", "human-pooled"}
            is_indiv = s.startswith("human-") or s.startswith("humans-")
            is_human = is_aggr or is_pooled or is_indiv
            if not is_human:
                return True
            if args.humans_mode == "aggregated":
                return is_aggr
            if args.humans_mode == "pooled":
                return is_pooled
            if args.humans_mode == "individual":
                return is_indiv
            return True
        df = df[df["agent"].apply(_is_keep)]

    # Optional: filter individual humans by minimum number of tasks, using coverage summary CSV per experiment
    if args.min_human_tasks and args.min_human_tasks > 0:
        # Build coverage tables per experiment present in df
        coverage_tables: dict[str, pd.DataFrame] = {}
        for exp in sorted(df["experiment"].dropna().astype(str).unique().tolist()):
            cov_path: Optional[Path] = None
            if args.human_coverage_csv:
                cp = Path(args.human_coverage_csv)
                cov_path = cp if cp.exists() else None
            else:
                candidate = PROJECT_ROOT / "results" / "human_coverage" / f"{exp}_human_prompt_coverage_summary.csv"
                cov_path = candidate if candidate.exists() else None
            if cov_path is None:
                print(f"[WARN] Human coverage summary not found for experiment '{exp}'. Expected at results/human_coverage/{exp}_human_prompt_coverage_summary.csv. Skipping min-human-tasks filter for this experiment.")
                continue
            try:
                cov = pd.read_csv(cov_path)
                # Normalize column name
                if "human_subj_id" not in cov.columns and "humans_subj_id" in cov.columns:
                    cov = cov.rename(columns={"humans_subj_id": "human_subj_id"})
                coverage_tables[exp] = cov[["human_subj_id", "n_tasks"]].copy()
            except Exception as e:
                print(f"[WARN] Failed to read coverage summary for '{exp}': {e}. Skipping filter for this experiment.")
        if coverage_tables:
            # Compute human_id and join n_tasks using a vectorized mapping
            df["human_id"] = df["agent"].apply(_parse_human_id)
            # Build a mapping for quick lookup: (experiment, human_subj_id) -> n_tasks
            mapping: dict[tuple[str, int], int] = {}
            for exp_name, cov in coverage_tables.items():
                try:
                    for sid, n in zip(cov["human_subj_id"].astype(int).tolist(), cov["n_tasks"].astype(int).tolist()):
                        mapping[(str(exp_name), int(sid))] = int(n)
                except Exception:
                    pass
            human_n_tasks_list: list[Optional[int]] = []
            for exp_val, hid in zip(df["experiment"].astype(str).tolist(), df["human_id"].tolist()):
                if hid is None or (isinstance(hid, float) and np.isnan(hid)):
                    human_n_tasks_list.append(None)
                else:
                    human_n_tasks_list.append(mapping.get((str(exp_val), int(hid))))
            df["human_n_tasks"] = human_n_tasks_list
            before = len(df)
            # Keep rows where either not an individual human or human_n_tasks >= min threshold
            def _keep_min_tasks(agent: str, n_tasks: Optional[float]) -> bool:
                sid = _parse_human_id(agent)
                if sid is None:
                    return True  # non-individual human
                try:
                    return (n_tasks is not None) and (int(n_tasks) >= int(args.min_human_tasks))
                except Exception:
                    return False
            df = df[df.apply(lambda r: _keep_min_tasks(r["agent"], r.get("human_n_tasks")), axis=1)]
            after = len(df)
            removed = before - after
            print(f"Applied min-human-tasks >= {args.min_human_tasks}: removed {removed} rows (kept {after}).")
        else:
            print("[WARN] No coverage tables available; --min-human-tasks was requested but could not be applied.")

    # Filters
    if args.agents and not any(str(a).lower() == "all" for a in args.agents):
        df = df[df["agent"].isin(args.agents)]
    if args.domains:
        df = df[df["domain"].isin(args.domains)]
    if args.pooled_only:
        df = df[df["domain"].astype(str).str.lower() == "all"]
    if args.exclude_agents:
        excl = {str(a).lower() for a in args.exclude_agents}
        df = df[~df["agent"].astype(str).str.lower().isin(excl)]

    if df.empty:
        print("No data after filtering.")
        return 0

    multi = len(df["experiment"].unique()) > 1
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # If exactly one tag matched and one experiment, write directly into that tag directory
    if len(matched_tags) == 1 and not multi:
        exp_name = df["experiment"].iloc[0]
        # Include humans-mode subdir when filtering to avoid overwriting sibling runs
        if args.humans_mode != "all":
            hm_map = {"aggregated": "hm-agg", "pooled": "hm-pooled", "individual": "hm-indiv"}
            out_dir = out_root / exp_name / matched_tags[0] / hm_map.get(args.humans_mode, args.humans_mode)
        else:
            out_dir = out_root / exp_name / matched_tags[0]
    else:
        out_dir = out_root / ("multi_experiment" if multi else df["experiment"].iloc[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = out_dir / "winner_parameters_long.csv"
    df.to_csv(combined_csv, index=False)
    print(f"Saved: {combined_csv}")

    # Diagnostics table: agent|domain metrics and human task counts (if available)
    metric_keys = [
        "loocv_r2",
        "loocv_rmse",
        "r2_task",
        "rmse_task",
        "r2",
        "rmse",
    ]
    present_metrics = [m for m in metric_keys if m in df.columns]
    agg_dict: dict[str, tuple[str, str]] = {
        "n_param_values": ("value", "count"),
        "n_param_types": ("parameter", "nunique"),
    }
    for m in present_metrics:
        agg_dict[m] = (m, "mean")
    # Include humans_mode and human_n_tasks summary if present
    if "human_n_tasks" in df.columns:
        agg_dict["human_n_tasks_max"] = ("human_n_tasks", "max")
        agg_dict["human_n_tasks_min"] = ("human_n_tasks", "min")
    if "humans_mode" in df.columns:
        agg_dict["humans_mode_first"] = ("humans_mode", "first")
    g = df.groupby(["experiment", "tag", "agent", "domain"])
    diagnostics = g["value"].count().rename("n_param_values").to_frame()
    # Default fallback: distinct canonical parameter names observed
    diagnostics["n_param_types"] = g["parameter"].nunique()
    # Prefer params_tying if available
    if "params_tying" in df.columns:
        try:
            df_pt = df.copy()
            # Access column directly to avoid Optional[Series]
            df_pt["params_tying_num"] = pd.to_numeric(df_pt["params_tying"], errors="coerce")
            pt_series = df_pt.groupby(["experiment", "tag", "agent", "domain"])['params_tying_num'].first()
            # Align to diagnostics index and overwrite where available
            pt_series = pt_series.reindex(diagnostics.index)
            override = pt_series.notna()
            diagnostics.loc[override, "n_param_types"] = pt_series.loc[override].astype(int)
        except Exception:
            pass
    for m in present_metrics:
        diagnostics[m] = g[m].mean()
    if "human_n_tasks" in df.columns:
        diagnostics["human_n_tasks_max"] = g["human_n_tasks"].max()
        diagnostics["human_n_tasks_min"] = g["human_n_tasks"].min()
    if "humans_mode" in df.columns:
        try:
            diagnostics["humans_mode_first"] = g["humans_mode"].first()
        except Exception:
            pass
    diagnostics = diagnostics.reset_index()
    diag_path = out_dir / "diagnostics_agent_domain.csv"
    diagnostics.to_csv(diag_path, index=False)
    print(f"Saved diagnostics: {diag_path}")

    for group_by in [
        ["agent"],
        ["agent", "domain"],
        ["agent", "link"] if "link" in df.columns else ["agent"],
        ["agent", "domain", "link"] if "link" in df.columns else ["agent", "domain"],
    ]:
        summary = summarize_params(df, group_by)
        name = "by_" + "_".join(group_by) + ".csv"
        summary.to_csv(out_dir / name, index=False)
        print(f"Saved summary: {out_dir / name}")

    if not args.no_plots:
        # If writing into a tag dir, use it directly; otherwise create a plots subdir
        if len(matched_tags) == 1 and not multi:
            plots_dir = out_dir
        else:
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
        # Only heatmaps (no violins, no correlations)
        plot_heatmaps(df, plots_dir, metric=args.metric)
        # If not already in the tag dir, copy source winners artifacts next to plots
        if len(matched_tags) != 1 or multi:
            for td in tag_dirs:
                tag = td.name
                src1 = td / "winners.csv"
                src2 = td / "winners_with_params.csv"
                if src1.exists():
                    shutil.copy2(src1, plots_dir / f"winners_{tag}.csv")
                if src2.exists():
                    shutil.copy2(src2, plots_dir / f"winners_with_params_{tag}.csv")

    print(f"All artifacts in: {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
