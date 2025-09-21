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
    --output-dir results/parameter_analysis/cbn_fit_metric_analysis/plots
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
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

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
    from causalign.plotting.palette import PROMPT_CATEGORY_COLORS, canon_prompt_category
except Exception:
    # Fallback if src/ not on path when running the script directly
    PROMPT_CATEGORY_COLORS = {
        "numeric": (0.85, 0.60, 0.55),
        "CoT": (0.00, 0.20, 0.55),
    }
    def canon_prompt_category(label: str) -> str:  # type: ignore
        t = str(label).strip().lower()
        if t in NUMERIC_SYNS or t == "numeric":
            return "numeric"
        if t in COT_SYNS or t == "cot":
            return "CoT"
        return str(label)

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
    t = str(p).strip().lower()
    if t in NUMERIC_SYNS or t == "numeric":
        return "numeric"
    if t in COT_SYNS or t == "cot":
        return "CoT"
    return str(p)

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
        color = PROMPT_CATEGORY_COLORS.get(pc, PROMPT_CATEGORY_COLORS.get(pc.title(), (0.3, 0.3, 0.3)))
        label = pc if pc == "CoT" else ("Numeric" if pc == "numeric" else pc)
        handles.append(Line2D([0], [0], color=color, lw=6, label=label))
    return handles


def plot_box(df: pd.DataFrame, metric: str, out_dir: Path, prompt_order_canon: List[str]) -> Path:
    experiments, data = _collect_groups(df, metric, prompt_order_canon)
    if not experiments:
        return out_dir / f"box_{metric}_EMPTY.pdf"
    n_exp = len(experiments)
    group_gap = 1.6  # larger gap between experiments
    intra = 0.18     # tighter spacing within experiment
    base_pos = np.arange(n_exp) * group_gap
    fig_w = max(6.0, 0.8 + 0.9 * n_exp * group_gap)
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
        for i, exp in enumerate(experiments):
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

    ax.set_xticks(x)
    # Pretty experiment names
    xticklabels = [exp_name_map.get(e, e) for e in experiments]
    ax.set_xticklabels(xticklabels, rotation=25, ha='right')
    ax.set_ylabel(_metric_label(metric))
    ax.set_xlabel("Experiment")
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    # Legends: place Prompt and Summary side-by-side at the top center, outside the axes, without overlap
    prompt_handles = _legend_handles(prompt_order_canon)
    median_handle = Line2D([0, 1], [0, 0], color='black', lw=1.2)
    mean_handle = Line2D([0], [0], marker='^', color='black', markerfacecolor='yellow', markeredgecolor='black', linestyle='None', markersize=6)
    # Reserve top margin for legends
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    # Anchor two legends around the center top
    dx = 0.20
    leg_prompt = ax.legend(
        handles=prompt_handles,
        title="Prompt",
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5 - dx, 1.12),
        ncol=max(1, len(prompt_order_canon)),
    )
    ax.add_artist(leg_prompt)
    leg_summary = ax.legend(
        handles=[median_handle, mean_handle],
        labels=["Median", "Mean"],
        title="Summary",
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5 + dx, 1.12),
        ncol=2,
    )
    ax.add_artist(leg_summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"box_{metric}.pdf"
    out_png = out_dir / f"box_{metric}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_pdf


def plot_violin(df: pd.DataFrame, metric: str, out_dir: Path, prompt_order_canon: List[str]) -> Path:
    experiments, data = _collect_groups(df, metric, prompt_order_canon)
    if not experiments:
        return out_dir / f"violin_{metric}_EMPTY.pdf"
    n_exp = len(experiments)
    group_gap = 1.6
    intra = 0.18
    base_pos = np.arange(n_exp) * group_gap
    fig_w = max(6.0, 0.8 + 0.9 * n_exp * group_gap)
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
        for i, exp in enumerate(experiments):
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

    ax.set_xticks(x)
    ax.set_xticklabels([exp_name_map.get(e, e) for e in experiments], rotation=25, ha='right')
    ax.set_ylabel(_metric_label(metric))
    ax.set_xlabel("Experiment")
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    # Legends: place Prompt and Summary side-by-side at the top center, outside the axes, without overlap
    prompt_handles = _legend_handles(prompt_order_canon)
    median_handle = Line2D([0, 1], [0, 0], color='black', lw=1.2)
    mean_handle = Line2D([0], [0], marker='^', color='black', markerfacecolor='yellow', markeredgecolor='black', linestyle='None', markersize=6)
    # Reserve top margin for legends
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    dx = 0.20
    leg_prompt = ax.legend(
        handles=prompt_handles,
        title="Prompt",
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5 - dx, 1.12),
        ncol=max(1, len(prompt_order_canon)),
    )
    ax.add_artist(leg_prompt)
    leg_summary = ax.legend(
        handles=[median_handle, mean_handle],
        labels=["Median", "Mean"],
        title="Summary",
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5 + dx, 1.12),
        ncol=2,
    )
    ax.add_artist(leg_summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"violin_{metric}.pdf"
    out_png = out_dir / f"violin_{metric}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_pdf


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot box/violin distributions of fit metrics per experiment × prompt-category.")
    ap.add_argument("--input", default="results/parameter_analysis/cbn_fit_metric_analysis/long_by_experiment_prompt_agent.csv", help="Path to long CSV from summarize_fit_cbn_fit_metric_analysis.py")
    ap.add_argument("--output-dir", default="results/parameter_analysis/cbn_fit_metric_analysis/plots", help="Directory to write plots")
    ap.add_argument("--metrics", nargs="*", default=["loss", "mae", "rmse"], help="Metrics to plot (must match names in the long CSV)")
    ap.add_argument("--prompts", nargs="*", default=["numeric", "cot"], help="Prompt categories to include and order (case-insensitive; e.g., 'CoT' or 'cot')")
    ap.add_argument("--usetex", action="store_true", help="Use LaTeX rendering if available (tueplots NeurIPS config)")
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
    if df.empty:
        print("[warn] No data after filtering by prompt categories.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for m in args.metrics:
        if df[df["metric"].str.lower() == m.lower()].empty:
            print(f"[warn] Metric '{m}' not found in input; skipping")
            continue
        p1 = plot_box(df, m, out_dir, prompt_order_canon)
        p2 = plot_violin(df, m, out_dir, prompt_order_canon)
        print(f"[ok] Wrote {p1}")
        print(f"[ok] Wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
