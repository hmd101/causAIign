#!/usr/bin/env python3
"""
Plot distributions of fit metrics (loss, MAE, RMSE) per experiment × prompt-category.

This script reads the long-form CSV produced by scripts/summarize_fit_cbn_fit_metric_analysis.py
(long_by_experiment_prompt_agent.csv) and creates, for each metric requested:
  - A box plot figure (per experiment, separate boxes for each prompt-category)
  - A violin plot figure (same layout)

Color scheme follows the R^2 box plots in cbn_aggregate_cross_experiment.py:
  - numeric: (0.85, 0.60, 0.55)
  - cot:     (0.00, 0.20, 0.55)

Usage (from repo root):
    python3 scripts/plot_fit_metric_distributions.py \
    --input results/parameter_analysis/cbn_fit_metric_analysis/long_by_experiment_prompt_agent.csv \
    --metrics loss mae rmse \
    --output-dir results/parameter_analysis/cbn_fit_metric_analysis/plots \
        [--show-human-baseline] \
        [--experiments rw17_indep_causes random_abstract] \
        [--prompt-categories numeric cot]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib as mpl
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# # Prompt-category palette (aligned to cbn_aggregate_cross_experiment.py)
# PROMPT_CATEGORY_COLORS: Dict[str, Tuple[float, float, float]] = {
#     "numeric": (0.85, 0.60, 0.55),
#     "cot": (0.00, 0.20, 0.55),
# }


# Prompt-category synonyms (case-insensitive)
NUMERIC_SYNS = {
    "numeric", "pcnum", "num", "single_numeric", "single_numeric_response",
}
COT_SYNS = {
    "cot", "pccot", "chain_of_thought", "chain-of-thought", "cot_stepwise", "CoT",
}

# Global palette for prompt categories
try:
    from causalign.plotting.palette import (
        COT_LABEL,
        NUMERIC_LABEL,
        PROMPT_CATEGORY_COLORS,
        canon_prompt_category,
    )
except Exception:
    # Fallback if src/ not on path when running the script directly
    PROMPT_CATEGORY_COLORS = {
        "Direct": (0.85, 0.60, 0.55),
        "CoT": (0.00, 0.20, 0.55),
    }
    def canon_prompt_category(label: str) -> str:  # type: ignore
        t = str(label).strip().lower()
        if t in NUMERIC_SYNS or t == "numeric":
            return "Direct"
        if t in COT_SYNS or t == "cot":
            return "CoT"
        return str(label)
    NUMERIC_LABEL = "Direct"
    COT_LABEL = "CoT"

# Experiment pretty names
exp_name_map = {
    "random_abstract": "Abstract",
    "rw17_indep_causes": "RW17",
    "abstract_overloaded_lorem_de": "Abstract-Overloaded",
    "rw17_overloaded_de": "RW17-Overloaded-DE",
    "rw17_overloaded_d": "RW17-Overloaded-D",
    "rw17_overloaded_e": "RW17-Overloaded",
}


def _ensure_tueplots(usetex: bool = False) -> None:
    """Configure NeurIPS-like plotting defaults using tueplots if available."""
    try:
        from tueplots import bundles  # type: ignore
        from tueplots import fonts as _fonts  # type: ignore
    except Exception:
        mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 12})
        return
    cfg = bundles.neurips2023(nrows=1, ncols=1, rel_width=0.9, usetex=usetex, family="serif")
    cfg["legend.title_fontsize"] = 11
    cfg["font.size"] = 12
    cfg["axes.labelsize"] = 12
    cfg["axes.titlesize"] = 13
    cfg["xtick.labelsize"] = 10
    cfg["ytick.labelsize"] = 10
    cfg["legend.fontsize"] = 10
    if usetex:
        cfg["text.latex.preamble"] = r"\usepackage{amsmath,bm}"
    fnt = _fonts.neurips2022_tex(family="serif")
    mpl.rcParams.update({**cfg, **fnt})

def _canon_prompt(p: str) -> str:
    return canon_prompt_category(p)

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    req = {"experiment", "prompt_category", "agent", "metric", "value"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    return df


def _metric_label(m: str) -> str:
    # Pretty labels
    m_low = m.lower()
    if m_low == "rmse":
        return "RMSE"
    if m_low == "mae":
        return "MAE"
    if m_low == "loss":
        return "Loss"
    if m_low == "loocv_r2":
        return "Reasoning consistency"
    return m


def _collect_groups(df: pd.DataFrame, metric: str, prompt_order_canon: List[str]) -> Tuple[List[str], Dict[Tuple[str, str], np.ndarray]]:
    # Returns experiments list and mapping (experiment, prompt_canon) -> values array
    sub = df[df["metric"].str.lower() == metric.lower()].copy()
    if sub.empty:
        return [], {}
    experiments = sorted(sub["experiment"].astype(str).unique().tolist())
    data: Dict[Tuple[str, str], np.ndarray] = {}
    for exp in experiments:
        gexp = sub[sub["experiment"].astype(str) == exp]
        for pc in prompt_order_canon:
            vals = pd.to_numeric(gexp[gexp["pc_canon"].astype(str) == pc]["value"], errors="coerce").dropna().values
            data[(exp, pc)] = vals
    return experiments, data


def _legend_handles(prompt_order_canon: List[str]) -> List[Line2D]:
    handles: List[Line2D] = []
    for pc in prompt_order_canon:
        key = canon_prompt_category(pc)
        color = PROMPT_CATEGORY_COLORS.get(key, PROMPT_CATEGORY_COLORS.get(str(key).title(), (0.3, 0.3, 0.3)))
        label = key
        handles.append(Line2D([0], [0], color=color, lw=6, label=label))
    return handles


def _rw17_human_baseline(df: pd.DataFrame, metric: str) -> float | None:
    """
    Compute the human baseline for RW17 Numeric for a given metric from the long table.
    Heuristic: rows with agent == 'humans', experiment maps to RW17 (exp_name_map value 'RW17'),
    and prompt category numeric.
    Returns the mean across matched rows, or None if not found.
    """
    m = metric.lower()
    sub = df[df["metric"].str.lower() == m].copy()
    if sub.empty:
        return None
    # Canonical experiment id for RW17 (key in exp_name_map is 'rw17_indep_causes')
    # Accept either key or pretty label in case the CSV stores labels
    is_rw17 = (sub["experiment"].astype(str) == "rw17_indep_causes") | (
        sub["experiment"].astype(str).str.fullmatch("RW17", case=False, na=False)
    )
    is_human = sub["agent"].astype(str).str.lower() == "humans"
    # Normalize prompt label using canon_prompt_category if available
    sub["pc_canon_tmp"] = sub["prompt_category"].apply(canon_prompt_category)
    is_numeric = sub["pc_canon_tmp"].astype(str) == NUMERIC_LABEL
    hs = pd.to_numeric(sub[is_rw17 & is_human & is_numeric]["value"], errors="coerce").dropna()
    if hs.empty:
        return None
    return float(hs.mean())


def plot_box(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    prompt_order_canon: List[str],
    *,
    show_human_baseline: bool = False,
    filename_suffix: str = "",
    baseline_map: dict[tuple[str, str, str], tuple[float, float, float]] | None = None,
    baseline_with_cis: bool = False,
) -> Path:
    experiments, data = _collect_groups(df, metric, prompt_order_canon)
    if not experiments:
        return out_dir / f"box_{metric}_EMPTY.pdf"
    n_exp = len(experiments)
    group_gap = 1.6  # larger gap between experiments
    intra = 0.18     # tighter spacing within experiment
    layout_n_exp = 2 if n_exp == 1 else n_exp
    base_pos = np.arange(layout_n_exp) * group_gap
    fig_w = max(6.0, 0.8 + 0.9 * layout_n_exp * group_gap)
    fig_h = 3.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Positions: per experiment at integer i, with offsets for prompt categories
    x = base_pos
    n_pc = len(prompt_order_canon)
    width = 0.32 if n_pc == 2 else 0.22
    offsets = np.linspace(-intra, intra, n_pc)

    for j, pc in enumerate(prompt_order_canon):
        color = PROMPT_CATEGORY_COLORS.get(pc, PROMPT_CATEGORY_COLORS.get(pc.title(), (0.3, 0.3, 0.3)))
        boxes = []
        positions = []
        exps_for_layout = experiments + [""] if n_exp == 1 else experiments
        for i, exp in enumerate(exps_for_layout):
            vals = data.get((exp, pc), np.array([]))
            if vals.size == 0:
                continue
            boxes.append(vals)
            positions.append(x[i] + offsets[j])
        if not boxes:
            continue
        bp = ax.boxplot(
            boxes,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_alpha(0.8)
        for element in ['whiskers', 'caps', 'medians']:
            for artist in bp[element]:
                artist.set_color('black')
        # Overlay mean as yellow upward triangle markers
        means = [float(np.nanmean(v)) for v in boxes]
        ax.plot(positions, means, linestyle='None', marker='^', markersize=6,
                markerfacecolor='yellow', markeredgecolor='black', zorder=5)

    # Overlay random-init baseline (median or median ± 95% CI) per experiment × prompt
    baseline_handle = None
    baseline_color = 'gray'
    if baseline_map:
        exps_for_layout = experiments + [""] if n_exp == 1 else experiments
        for i, exp in enumerate(exps_for_layout):
            if not exp:
                continue
            for j, pc in enumerate(prompt_order_canon):
                key = (exp, pc, metric.lower())
                stats = baseline_map.get(key)
                if not stats:
                    continue
                med, lo, hi = stats
                xpos_base = x[i] + offsets[j]
                if baseline_with_cis:
                    xpos = xpos_base + width * 0.12  # slight right offset to avoid overlap
                    ax.hlines(med, xpos - width*0.35, xpos + width*0.35, colors=baseline_color, linestyles='--', linewidth=1.5, zorder=6)
                    ax.vlines(xpos, lo, hi, colors=baseline_color, linestyles='-', linewidth=1.2, alpha=0.9, zorder=6)
                    ax.hlines([lo, hi], xpos - width*0.20, xpos + width*0.20, colors=baseline_color, linewidth=1.2, alpha=0.9, zorder=6)
                else:
                    # median only, short dashed line at box center position
                    ax.hlines(med, xpos_base - width*0.35, xpos_base + width*0.35, colors=baseline_color, linestyles='--', linewidth=1.5, zorder=6)
        blabel = 'Random init baseline (median ±95% CI)' if baseline_with_cis else 'Random init baseline (median)'
        baseline_handle = Line2D([0, 1], [0, 0], color=baseline_color, linestyle='--', lw=1.5, label=blabel)

    ax.set_xticks(x)
    # Pretty experiment names
    exps_for_layout = experiments + [""] if n_exp == 1 else experiments
    xticklabels = [exp_name_map.get(e, e) if e else "" for e in exps_for_layout]
    ax.set_xticklabels(xticklabels, rotation=25, ha='right')
    ax.set_ylabel(_metric_label(metric))
    ax.set_xlabel("Experiment")
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    # Optional human baseline
    human_line = None
    if show_human_baseline:
        hb = _rw17_human_baseline(df, metric)
        if hb is not None and np.isfinite(hb):
            human_line = ax.axhline(y=hb, color=(1.0, 0.4, 0.7), linestyle="--", linewidth=1.5)
    # Legend: single row above the plot, no title
    prompt_handles = _legend_handles(prompt_order_canon)
    median_handle = Line2D([0, 1], [0, 0], color='black', lw=1.2, label='Median')
    mean_handle = Line2D([0], [0], marker='^', color='black', markerfacecolor='yellow', markeredgecolor='black', linestyle='None', markersize=6, label='Mean')
    handles = [*prompt_handles, median_handle, mean_handle]
    if human_line is not None:
        handles.append(Line2D([0, 1], [0, 0], color=(1.0, 0.4, 0.7), linestyle='--', lw=1.5, label=f"Humans (RW17 {NUMERIC_LABEL})"))
    if baseline_handle is not None:
        handles.append(baseline_handle)
    # Two-row legend above the axes (outside plot area)
    ncols = max(2, int(np.ceil(len(handles) / 2)))
    # Make room for legend above
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.87), ncol=ncols)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{filename_suffix}" if filename_suffix else ""
    out_pdf = out_dir / f"box_{metric}{suffix}.pdf"
    out_png = out_dir / f"box_{metric}{suffix}.png"
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_pdf


def plot_violin(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    prompt_order_canon: List[str],
    *,
    show_human_baseline: bool = False,
    filename_suffix: str = "",
    baseline_map: dict[tuple[str, str, str], tuple[float, float, float]] | None = None,
    baseline_with_cis: bool = False,
) -> Path:
    experiments, data = _collect_groups(df, metric, prompt_order_canon)
    if not experiments:
        return out_dir / f"violin_{metric}_EMPTY.pdf"
    n_exp = len(experiments)
    group_gap = 1.6
    intra = 0.18
    layout_n_exp = 2 if n_exp == 1 else n_exp
    base_pos = np.arange(layout_n_exp) * group_gap
    fig_w = max(6.0, 0.8 + 0.9 * layout_n_exp * group_gap)
    fig_h = 3.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = base_pos
    n_pc = len(prompt_order_canon)
    width = 0.32 if n_pc == 2 else 0.22
    offsets = np.linspace(-intra, intra, n_pc)

    for j, pc in enumerate(prompt_order_canon):
        color = PROMPT_CATEGORY_COLORS.get(pc, PROMPT_CATEGORY_COLORS.get(pc.title(), (0.3, 0.3, 0.3)))
        boxes = []
        positions = []
        exps_for_layout = experiments + [""] if n_exp == 1 else experiments
        for i, exp in enumerate(exps_for_layout):
            vals = data.get((exp, pc), np.array([]))
            if vals.size == 0:
                continue
            boxes.append(vals)
            positions.append(x[i] + offsets[j])
        if not boxes:
            continue
        vp = ax.violinplot(boxes, positions=positions, widths=width * 0.9, showmeans=False, showmedians=True, showextrema=False)
        bodies = vp.get('bodies', []) if isinstance(vp, dict) else getattr(vp, 'bodies', [])
        if not isinstance(bodies, (list, tuple)):
            bodies = [bodies] if bodies is not None else []
        for body in bodies:
            body.set_facecolor(color)
            body.set_edgecolor('black')
            body.set_alpha(0.7)
        # medians are LineCollection; set color to black
        if isinstance(vp, dict) and 'cmedians' in vp:
            vp['cmedians'].set_color('black')
        else:
            cm = getattr(vp, 'cmedians', None)
            if cm is not None:
                cm.set_color('black')
        # Overlay mean as yellow upward triangle markers
        means = [float(np.nanmean(v)) for v in boxes]
        ax.plot(positions, means, linestyle='None', marker='^', markersize=6,
                markerfacecolor='yellow', markeredgecolor='black', zorder=5)

    # Overlay random-init baseline (median or median ± 95% CI)
    baseline_handle = None
    baseline_color = 'gray'
    if baseline_map:
        exps_for_layout = experiments + [""] if n_exp == 1 else experiments
        for i, exp in enumerate(exps_for_layout):
            if not exp:
                continue
            for j, pc in enumerate(prompt_order_canon):
                key = (exp, pc, metric.lower())
                stats = baseline_map.get(key)
                if not stats:
                    continue
                med, lo, hi = stats
                xpos_base = x[i] + offsets[j]
                if baseline_with_cis:
                    xpos = xpos_base + width * 0.12
                    ax.hlines(med, xpos - width*0.35, xpos + width*0.35, colors=baseline_color, linestyles='--', linewidth=1.5, zorder=6)
                    ax.vlines(xpos, lo, hi, colors=baseline_color, linestyles='-', linewidth=1.2, alpha=0.9, zorder=6)
                    ax.hlines([lo, hi], xpos - width*0.20, xpos + width*0.20, colors=baseline_color, linewidth=1.2, alpha=0.9, zorder=6)
                else:
                    ax.hlines(med, xpos_base - width*0.35, xpos_base + width*0.35, colors=baseline_color, linestyles='--', linewidth=1.5, zorder=6)
        blabel = 'Random init baseline (median ±95% CI)' if baseline_with_cis else 'Random init baseline (median)'
        baseline_handle = Line2D([0, 1], [0, 0], color=baseline_color, linestyle='--', lw=1.5, label=blabel)
    ax.set_xticks(x)
    exps_for_layout = experiments + [""] if n_exp == 1 else experiments
    ax.set_xticklabels([exp_name_map.get(e, e) if e else "" for e in exps_for_layout], rotation=25, ha='right')
    ax.set_ylabel(_metric_label(metric))
    ax.set_xlabel("Experiment")
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    # Optional human baseline
    human_line = None
    if show_human_baseline:
        hb = _rw17_human_baseline(df, metric)
        if hb is not None and np.isfinite(hb):
            human_line = ax.axhline(y=hb, color=(1.0, 0.4, 0.7), linestyle="--", linewidth=1.5)
    # Legend: single row above the plot, no title
    prompt_handles = _legend_handles(prompt_order_canon)
    median_handle = Line2D([0, 1], [0, 0], color='black', lw=1.2, label='Median')
    mean_handle = Line2D([0], [0], marker='^', color='black', markerfacecolor='yellow', markeredgecolor='black', linestyle='None', markersize=6, label='Mean')
    handles = [*prompt_handles, median_handle, mean_handle]
    if human_line is not None:
        handles.append(Line2D([0, 1], [0, 0], color=(1.0, 0.4, 0.7), linestyle='--', lw=1.5, label=f"Humans (RW17 {NUMERIC_LABEL})"))
    if baseline_handle is not None:
        handles.append(baseline_handle)
    # Two-row legend above the axes (outside plot area)
    ncols = max(2, int(np.ceil(len(handles) / 2)))
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.87), ncol=ncols)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{filename_suffix}" if filename_suffix else ""
    out_pdf = out_dir / f"violin_{metric}{suffix}.pdf"
    out_png = out_dir / f"violin_{metric}{suffix}.png"
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_pdf


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot box/violin distributions of fit metrics per experiment × prompt-category.")
    ap.add_argument("--input", default="results/parameter_analysis/cbn_fit_metric_analysis/long_by_experiment_prompt_agent.csv", help="Path to long CSV from summarize_fit_cbn_fit_metric_analysis.py")
    ap.add_argument("--output-dir", default="results/parameter_analysis/cbn_fit_metric_analysis/plots", help="Directory to write plots")
    ap.add_argument("--metrics", nargs="*", default=["loss", "mae", "rmse"], help="Metrics to plot (must match names in the long CSV)")
    ap.add_argument("--prompts", nargs="*", default=["numeric", "cot"], help="Prompt categories to include and order (case-insensitive; e.g., 'CoT' or 'cot')")
    ap.add_argument("--prompt-categories", dest="prompts", nargs="*", help="Alias for --prompts")
    ap.add_argument("--experiments", nargs="*", help="Experiments to include (match the 'experiment' column; e.g., rw17_indep_causes random_abstract)")
    ap.add_argument("--usetex", action="store_true", help="Use LaTeX rendering if available (tueplots NeurIPS config)")
    ap.add_argument("--show-human-baseline", action="store_true", help="Draw a pink dashed horizontal line at the human baseline (RW17 Direct) on box plots")
    ap.add_argument("--show-random-baseline", action="store_true", help="Overlay random-init baseline median ±95%% CI from baseline_by_experiment_prompt.csv")
    ap.add_argument("--baseline-csv", help="Path to baseline_by_experiment_prompt.csv (defaults to sibling of --input)")
    ap.add_argument(
        "--baseline-with-cis",
        action="store_true",
        help="When showing the random baseline, draw median ±95%% CI (gray, slightly right-offset). Default draws median-only (gray dashed).",
    )
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    # Configure plotting style
    _ensure_tueplots(args.usetex)
    df = pd.read_csv(in_path)
    _ensure_cols(df)
    # Normalize prompt labels for matching colors and selection (canonical keys: "numeric" and "CoT")
    df["pc_canon"] = df["prompt_category"].apply(canon_prompt_category)
    prompt_order_canon = [canon_prompt_category(p) for p in args.prompts]
    df = df[df["pc_canon"].isin(set(prompt_order_canon))].copy()
    # Filter experiments if provided
    if args.experiments:
        allowed_exps = set(str(e) for e in args.experiments)
        df = df[df["experiment"].astype(str).isin(allowed_exps)].copy()
    if df.empty:
        print("[warn] No data after filtering by prompt categories.")
        return 0

    # Load baseline medians/CI if requested
    baseline_map: dict[tuple[str, str, str], tuple[float, float, float]] | None = None
    if args.show_random_baseline:
        bcsv = Path(args.baseline_csv) if args.baseline_csv else in_path.with_name("baseline_by_experiment_prompt.csv")
        if not bcsv.exists():
            print(f"[warn] Baseline CSV not found: {bcsv}")
        else:
            bdf = pd.read_csv(bcsv)
            # Normalize prompt labels to plotting canonical labels (Direct/CoT)
            if "prompt_category" in bdf.columns:
                bdf = bdf.copy()
                bdf["pc_canon_b"] = bdf["prompt_category"].apply(canon_prompt_category)
                bdf = bdf[bdf["pc_canon_b"].isin(set(prompt_order_canon))]
            # If experiments filtered, apply same filter
            if args.experiments and "experiment" in bdf.columns:
                bdf = bdf[bdf["experiment"].astype(str).isin(set(str(e) for e in args.experiments))]
            baseline_map = {}
            # Build mapping for requested metrics only
            for _, row in bdf.iterrows():
                exp = str(row.get("experiment", ""))
                pc = str(row.get("pc_canon_b", row.get("prompt_category", "")))
                for m in args.metrics:
                    m_low = str(m).lower()
                    med_col = f"baseline_{m_low}_median"
                    lo_col = f"baseline_{m_low}_ci_low"
                    hi_col = f"baseline_{m_low}_ci_high"
                    if med_col in bdf.columns and lo_col in bdf.columns and hi_col in bdf.columns:
                        try:
                            med = float(row[med_col])
                            lo = float(row[lo_col])
                            hi = float(row[hi_col])
                        except Exception:
                            continue
                        key = (exp, pc, m_low)
                        baseline_map[key] = (med, lo, hi)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build filename suffix reflecting selected experiments and prompt categories
    exp_suffix = None
    if args.experiments:
        exp_suffix = "exp-" + "+".join([str(e) for e in args.experiments])
    pc_suffix = "pc-" + "+".join([str(canon_prompt_category(p)) for p in args.prompts])
    suffix_parts = [s for s in [exp_suffix, pc_suffix] if s]
    filename_suffix = "_" + "__".join(suffix_parts) if suffix_parts else ""
    for m in args.metrics:
        if df[df["metric"].str.lower() == m.lower()].empty:
            print(f"[warn] Metric '{m}' not found in input; skipping")
            continue
        p1 = plot_box(
            df,
            m,
            out_dir,
            prompt_order_canon,
            show_human_baseline=args.show_human_baseline,
            filename_suffix=filename_suffix,
            baseline_map=baseline_map,
            baseline_with_cis=args.baseline_with_cis,
        )
        p2 = plot_violin(
            df,
            m,
            out_dir,
            prompt_order_canon,
            show_human_baseline=args.show_human_baseline,
            filename_suffix=filename_suffix,
            baseline_map=baseline_map,
            baseline_with_cis=args.baseline_with_cis,
        )
        print(f"[ok] Wrote {p1}")
        print(f"[ok] Wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
