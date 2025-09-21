#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, cast

import matplotlib as mpl
from matplotlib import patches as mpatches, transforms as mtransforms
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

# This file is the canonical location for EA/MV/R² overlays.
# Content moved from scripts/plot_ea_mv_levels.py unchanged.



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

def _canon_prompt(p: str) -> str:
    t = str(p).strip().lower()
    if t in NUMERIC_SYNS or t == "numeric":
        return "numeric"
    if t in COT_SYNS or t == "cot":
        return "CoT"
    return str(p)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot EA/MV/R²/Normativity levels by agent with category overlays")
    p.add_argument("--input-csv", default="results/cross_cogn_strategies/masters_classified_strategy_metrics.csv",
                   help="Path to masters_classified_strategy_metrics.csv")
    p.add_argument(
        "--metric",
        choices=["ea", "mv", "r2", "norm", "normative", "normative_metric"],
        default="ea",
        help=(
            "Which metric to plot: ea (Explaining-Away), mv (Majority-Violation), r2 (LOOCV R²), "
            "or norm/normative_metric (Normativity-level Ψ_norm)"
        ),
    )
    p.add_argument("--experiments", nargs="+", help="Experiments to include; omit to plot all in CSV")
    p.add_argument("--tag", help="Optional tag filter (CSV column 'tag')")
    p.add_argument("--by-domain", action="store_true", help="If set, emit one plot per domain instead of pooled")
    p.add_argument("--prompt-categories", nargs="+", help="Prompt categories to include (default: overlay numeric & CoT)")
    p.add_argument(
        "--threshold",
        default=None,
        type=float,
        help=(
            "Optional vertical threshold line to draw. For r2, no line is drawn by default and the region "
            "R^2≥0.937 is shaded; if provided, the line and shading start at the given threshold. For ea/mv, "
            "no threshold is shown unless provided."
        ),
    )
    p.add_argument("--show_human_baseline", action="store_true",
                   help="If set, draw a dashed magenta line at the RW17 numeric human baseline value for the chosen metric")
    p.add_argument("--show_EA", dest="show_ea", nargs='?', const=0.0001, type=float, default=None,
                   help="If provided, shade the EA region GREATER than this value (default 0.02 if flag has no value) and label as 'EA'")
    # Styling / output
    p.add_argument("--title", help="Override plot title")
    p.add_argument("--legend-loc", default="upper left")
    p.add_argument("--fig-width", type=float, default=8.0)
    p.add_argument("--fig-height", type=float, default=6.0)
    p.add_argument("--usetex", action="store_true", help="Use LaTeX rendering (requires LaTeX installed)")
    p.add_argument("--output-dir", default="results/plots/ea_mv_levels")
    # CI/Bootstrap options for mean + whiskers
    p.add_argument("--show-ci", action="store_true", help="Draw horizontal CI whiskers and use mean as marker")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap samples for CI over domains")
    p.add_argument("--ci", type=float, default=95.0, help="Confidence level for bootstrap CIs (e.g., 95)")
    p.add_argument("--seed", type=int, default=123, help="Random seed for bootstrap")
    # R² shading control
    p.add_argument(
        "--no-r2-shade",
        dest="no_r2_shade",
        action="store_true",
        help="Disable gray shaded region for R²; by default a gray box is shown from threshold (default 0.937) to the right",
    )
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def _canon_metric_key(m: str) -> str:
    """Canonicalize metric name to one of: 'ea', 'mv', 'r2', 'norm'."""
    t = str(m).strip().lower()
    if t in {"ea"}:
        return "ea"
    if t in {"mv"}:
        return "mv"
    if t in {"r2", "loocv_r2", "loocv"}:
        return "r2"
    if t in {"norm", "normative", "normative_metric", "psi_norm", "psinorm"}:
        return "norm"
    return t


def _ensure_tueplots(usetex: bool = False) -> None:
    try:
        from tueplots import bundles  # type: ignore
    except Exception:
        mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 12})
        return
    cfg = bundles.neurips2023(nrows=1, ncols=1, rel_width=0.85, usetex=usetex, family="serif")
    cfg["legend.title_fontsize"] = 12
    cfg["font.size"] = 13
    cfg["axes.labelsize"] = 13
    cfg["axes.titlesize"] = 15
    cfg["xtick.labelsize"] = 11
    cfg["ytick.labelsize"] = 11
    cfg["legend.fontsize"] = 10
    if usetex:
        cfg["text.latex.preamble"] = r"\usepackage{amsmath,bm}"
    from tueplots import fonts as _fonts  # type: ignore
    fnt = _fonts.neurips2022_tex(family="serif")
    mpl.rcParams.update({**cfg, **fnt})


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Canonicalize prompt_category into {numeric, CoT} when applicable
    if "prompt_category" in df.columns:
        df["prompt_category"] = df["prompt_category"].astype(str).map(canon_prompt_category)
    return df


def _get_rw17_human_baseline(df_all: pd.DataFrame, metric: str) -> Optional[float]:
    """Return RW17 numeric human baseline for the given metric (ea/mv/r2/norm), if available.
    Prefer pooled (domain == 'all'); else average across domains.
    """
    mk = _canon_metric_key(metric)
    metric_col = (
        "EA_raw" if mk == "ea" else ("MV_raw" if mk == "mv" else ("loocv_r2" if mk == "r2" else "normative_metric"))
    )
    if metric_col not in df_all.columns:
        return None
    d = df_all.copy()
    # Canonicalize prompt_category as in _load_data to be safe
    if "prompt_category" in d.columns:
        d["prompt_category"] = d["prompt_category"].astype(str).map(canon_prompt_category)
    mask = (
        d["agent"].astype(str).str.lower().str.contains("human") &
        d["experiment"].astype(str).str.lower().eq("rw17_indep_causes") &
        d["prompt_category"].astype(str).eq("numeric")
    )
    sub = d[mask].copy()
    if sub.empty:
        return None
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub[sub[metric_col].notna()]
    if sub.empty:
        return None
    # If pooled exists, use that; else mean across domains
    pooled = sub[sub.get("domain", pd.Series(dtype=str)).astype(str).eq("all")]
    if not pooled.empty:
        return float(pooled.iloc[0][metric_col])
    return float(sub[metric_col].mean())


def _pick_domains(df: pd.DataFrame, by_domain: bool) -> List[str]:
    if "domain" not in df.columns:
        # Treat as pooled-only
        return ["all"]
    doms = sorted(df.get("domain", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    if not doms:
        return ["all"]
    if by_domain:
        return doms
    # Prefer pooled if present
    return ["all"] if "all" in doms else doms[:1]


def _agents_order(df: pd.DataFrame, metric: str) -> List[str]:
    # Order agents by numeric category if available; fallback to mean across categories
    mk = _canon_metric_key(metric)
    metric_col = (
        "EA_raw" if mk == "ea" else ("MV_raw" if mk == "mv" else ("loocv_r2" if mk == "r2" else "normative_metric"))
    )
    num = df[df["prompt_category"].astype(str) == "numeric"][ ["agent", metric_col] ].copy()
    if not num.empty and num[metric_col].notna().any():
        order_series = num.groupby("agent")[metric_col].mean()
    else:
        order_series = df.groupby("agent")[metric_col].mean()
    # Ordering so that the desired values appear at the TOP of the plot (last tick):
    # - EA: high at top  -> ascending sort (low .. high)
    # - MV: low at top   -> descending sort (high .. low)
    if mk in ("ea", "r2", "norm"):
        # EA and LOOCV R²: sort ascending so low..high from bottom to top (high at top)
        order = order_series.sort_values(ascending=True).index.tolist()
    else:
        # MV: order by absolute value; smaller magnitude near top (low MV preferred)
        # Descending puts large at bottom, small at top
        order = order_series.abs().sort_values(ascending=False).index.tolist()
    return order


def _bootstrap_ci_mean(values: np.ndarray, *, B: int, ci: float, rng: np.random.Generator) -> Tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mu = float(np.nanmean(values))
    if values.size == 1:
        return (mu, mu, mu)
    idx = rng.integers(0, values.size, size=(B, values.size))
    boots = np.nanmean(values[idx], axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(boots, alpha))
    hi = float(np.percentile(boots, 100 - alpha))
    return (mu, lo, hi)


def _plot_overlay(sub: pd.DataFrame, *, metric: str, experiment: str, domain: str, out_dir: Path,
                  title_override: Optional[str], legend_loc: str, fig_size: Tuple[float, float], threshold: Optional[float],
                  show_human: bool, human_baseline: Optional[float], ea_shade_from: Optional[float], show: bool,
                  show_ci: bool, B: int, ci: float, rng: np.random.Generator,
                  source_all_domains: Optional[pd.DataFrame] = None,
                  shared_xlim: Optional[Tuple[float, float]] = None,
                  shared_xticks: Optional[np.ndarray] = None,
                  shade_r2: bool = True) -> None:
    mk = _canon_metric_key(metric)
    metric_col = (
        "EA_raw" if mk == "ea" else ("MV_raw" if mk == "mv" else ("loocv_r2" if mk == "r2" else "normative_metric"))
    )
    # Filter to selected categories (numeric/CoT only present after canonization)
    cats_present = [c for c in ["numeric", "CoT"] if c in sub["prompt_category"].unique().tolist()]
    if not cats_present:
        return
    order = _agents_order(sub[sub["prompt_category"].isin(cats_present)], metric)
    if not order:
        return

    # Colors from global palette
    color_map = PROMPT_CATEGORY_COLORS
    y_offsets = {"numeric": -0.08, "CoT": +0.08}

    fig, ax = plt.subplots(figsize=fig_size)

    # Dynamic height could be used; keep as provided size for consistency
    n = len(order)
    ax.set_ylim(-0.5, n - 0.5)

    # faint dashed horizontal guide lines for every agent row
    for i in range(0, n):
        ax.axhline(i, color="0.9", lw=0.6, ls="--", alpha=0.6, zorder=0)

    # Always draw a dashed vertical line at 0 (also useful for R² if negative values occur)
    ax.axvline(0.0, color="0.25", ls=":", lw=.5, alpha=0.8, zorder=1)

    # Plot per category
    for cat in cats_present:
        # Build a unique row per agent to avoid Series/DataFrame ambiguity
        cols = ["agent", metric_col]
        if mk == "ea":
            cols += ["EA_raw_lo", "EA_raw_hi", "EA_raw_mean"]
        elif mk == "mv":
            cols += ["MV_raw_lo", "MV_raw_hi", "MV_raw_mean"]
        cols = [c for c in cols if c in sub.columns]
        ss = sub[sub["prompt_category"].astype(str) == cat][cols].copy()
        if "agent" not in ss.columns:
            continue
        s = ss.groupby("agent", as_index=True).first().reindex(order)

        # Build per-agent mean and precomputed CI arrays
        means: List[float] = []
        ci_l: List[float] = []
        ci_h: List[float] = []
        lo_key = "EA_raw_lo" if mk == "ea" else ("MV_raw_lo" if mk == "mv" else None)
        hi_key = "EA_raw_hi" if mk == "ea" else ("MV_raw_hi" if mk == "mv" else None)
        mu_key = "EA_raw_mean" if mk == "ea" else ("MV_raw_mean" if mk == "mv" else metric_col)

        for agent in order:
            # Ensure a Series row; if missing, use empty Series
            if agent in s.index:
                rtmp = s.loc[agent]
                if isinstance(rtmp, pd.DataFrame):
                    # take first row
                    row = rtmp.iloc[0]
                elif isinstance(rtmp, pd.Series):
                    row = rtmp
                else:
                    row = pd.Series(dtype=float)
            else:
                row = pd.Series(dtype=float)

            def _get_scalar(sr: pd.Series, key: str, fallback_key: Optional[str] = None) -> float:
                try:
                    if key in sr.index:
                        v = sr[key]
                        return float(v) if pd.notna(v) else float("nan")
                    if fallback_key and fallback_key in sr.index:
                        v = sr[fallback_key]
                        return float(v) if pd.notna(v) else float("nan")
                except Exception:
                    pass
                return float("nan")

            mu = _get_scalar(row, mu_key, metric_col)
            means.append(mu)
            if lo_key and hi_key:
                lo = _get_scalar(row, lo_key)
                hi = _get_scalar(row, hi_key)
            else:
                lo = hi = float("nan")
            ci_l.append(lo)
            ci_h.append(hi)

        vals = np.array(means, dtype=float)
        y = np.arange(n) + y_offsets.get(cat, 0.0)
        # Draw lines for numeric; CoT as dots only (no connecting line)
        # Suppress connecting line entirely for MV plots.
        if cat == "numeric" and mk != "mv":
            ax.plot(vals, y, color=color_map.get(cat, "gray"), lw=1.8, alpha=0.9, zorder=2)
        # CI whiskers using precomputed bounds only
        if show_ci:
            for i in range(n):
                lo = ci_l[i]
                hi = ci_h[i]
                if np.isfinite(lo) and np.isfinite(hi) and (hi != lo):
                    ax.plot([lo, hi], [y[i], y[i]], color=color_map.get(cat, "gray"), lw=1.6, alpha=0.85, zorder=1)
        # Scatter markers show mean
        ax.scatter(vals, y, s=28, color=color_map.get(cat, "gray"), zorder=3, label=f"{cat}")

    shaded_no_mv = False
    shaded_ea = False
    shaded_r2 = False
    # Threshold/shading behavior
    thr_provided = (threshold is not None and np.isfinite(threshold))
    if mk == "mv" and thr_provided:
        thr = abs(cast(float, threshold))
        # symmetric markers at +/- threshold and a light-gray band between
        ax.axvline(+thr, color="gray", ls="--", lw=1.2, alpha=0.9,
                   label=f"ROPE: $|{thr:g}|$")
        ax.axvline(-thr, color="gray", ls="--", lw=1.2, alpha=0.9)
        ax.axvspan(-thr, +thr, color="0.9", alpha=0.35, zorder=0.7)
        shaded_no_mv = True
    elif mk == "ea" and thr_provided:
        thr = cast(float, threshold)
        ax.axvline(thr, color="gray", ls="--", lw=1.2, alpha=0.9)
    elif mk == "r2":
        # Default: no threshold line; shade region >= 0.937
        shade_from: float = cast(float, threshold) if thr_provided else 0.937
        # draw optional line only if provided
        if thr_provided:
            ax.axvline(shade_from, color="gray", ls="--", lw=1.2, alpha=0.9, label=f" $R^2\\geq{shade_from:g}$")
        if shade_r2:
            ax.margins(x=0)
            xmax = (shared_xlim[1] if shared_xlim is not None else ax.get_xlim()[1]) + 0.01
            width = max(0.0, xmax - shade_from)
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            rect = mpatches.Rectangle((shade_from, 0.0), width, 1.0, transform=trans,
                                       facecolor="0.9", alpha=0.35, zorder=0.7, clip_on=False)
            ax.add_patch(rect)
            shaded_r2 = True
            r2_label_text = f"$R^2\\geq{shade_from:g}$"

    # Optional EA region shading: anything GREATER than the given threshold
    if mk == "ea" and ea_shade_from is not None and np.isfinite(ea_shade_from):
        left = float(ea_shade_from)
        # Remove x-margins so the span reaches the right spine without a white gap
        ax.margins(x=0)
        # Use blended transform so y spans full axes height, x remains in data coords to right edge
        xmax = (shared_xlim[1] if shared_xlim is not None else ax.get_xlim()[1]) + 0.01
        width = max(0.0, xmax - left)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        rect = mpatches.Rectangle((left, 0.0), width, 1.0, transform=trans,
                                   facecolor="0.9", alpha=0.35, zorder=0.7, clip_on=False)
        ax.add_patch(rect)
        shaded_ea = True

    # Optional human baseline line (always uses RW17 numeric baseline)
    if show_human and mk in ("ea", "mv", "r2", "norm") and human_baseline is not None and np.isfinite(human_baseline):
        # popping magenta
        ax.axvline(float(human_baseline), color=(0.8, 0.0, 0.8), ls=(0, (4, 2)), lw=1.8, alpha=0.95,
                   label="human baseline (RW17, numeric)")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(order)
    if mk == "ea":
        xlbl = "Explaining-Away (EA) level"
        # Use raw strings to avoid Python interpreting backslashes in LaTeX/mathtext sequences
        xlblmath = r"$\mathrm{EA}$-level:  $\Pr(C_1{=}1 \mid E{=}1, C_2{=}0)-\Pr(C_1{=}1 \mid E{=}1, C_2{=}1)$"
    elif mk == "mv":
        xlbl = "Markov-Violation (MV) level"
        xlblmath = r"$\mathrm{MV}$-level:  $\Pr(C_1{=}1\mid C_2{=}1) - \Pr(C_1{=}1\mid C_2{=}0)$"
    elif mk == "r2":
        xlbl = "LOOCV R²"
        xlblmath = r"$\mathrm{LOOCV}\ R^2$"
    else:
        xlbl = r"Leak-Adjusted Determinacy $\mathrm{LAD}$"
        xlblmath = r"$\mathrm{LAD}= \overline{m}-b$"
    ax.set_xlabel(xlblmath)
    ax.set_ylabel("Agent")
    # Use pretty experiment name if available
    pretty_exp = exp_name_map.get(experiment, experiment)
    default_title = f"{xlbl} for {pretty_exp}"
    # raw experiment name and domains
    # default_title = f"{xlbl} by agent — {experiment} — {dom_title}" 

    ax.set_title(title_override or default_title)
    # Build legend, injecting a proxy patch for the no-MV shaded band when present
    handles, labels = ax.get_legend_handles_labels()
    if shaded_no_mv:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor="0.9", edgecolor="none", alpha=0.35, label="no MV"))
        labels.append("no MV")
    if shaded_ea:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor="0.9", edgecolor="none", alpha=0.35, label="EA"))
        labels.append("EA")
    if shaded_r2:
        from matplotlib.patches import Patch

        # Label based on shaded-from value
        label_text = locals().get("r2_label_text", "$R^2\\geq0.937$")
        handles.append(Patch(facecolor="0.9", edgecolor="none", alpha=0.35, label=label_text))
        labels.append(label_text)
    # Add CI legend item when whiskers are shown: a small dot with short whiskers
    if show_ci:
        # short whisker line with dot in the middle
        # CI legend: whisker line with dot in the middle, lines touch the dot
        ci_line_left = Line2D([0, 0.8], [0, 0], color="0.3", lw=1.6)
        ci_dot = Line2D([0.5], [0], marker="o", linestyle="None", color="0.3", markersize=5)
        ci_line_right = Line2D([0.8, 1], [0, 0], color="0.3", lw=1.6)
        handles.append((ci_line_left, ci_dot, ci_line_right))  # type: ignore[arg-type]
        labels.append(f"{ci:g}% CI")
    # Place legend; for MV we prefer upper right to avoid overlaps
    legend_loc_use = "upper right" if mk == "mv" else legend_loc
    ax.legend(
        handles,
        labels,
        title="Prompt category",
        loc=legend_loc_use,
        frameon=True,
        framealpha=0.7,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    # Uniform 0.1-spaced x-ticks, include the rightmost tick — shared across plots when provided
    if shared_xticks is not None and shared_xlim is not None:
        ax.set_xticks(shared_xticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlim(shared_xlim[0], shared_xlim[1])
    else:
        xmin, xmax = ax.get_xlim()
        tick_start = np.floor(xmin * 10.0) / 10.0
        tick_end = np.ceil(xmax * 10.0) / 10.0
        ticks = np.round(np.arange(tick_start, tick_end + 1e-9, 0.1), 1)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlim(tick_start, tick_end)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{metric}_levels_overlay_{experiment}_{domain.replace(',', '+').replace(' ', '')}"
    fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{base}.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    _ensure_tueplots(args.usetex)

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = _load_data(csv_path)

    # Filter by tag/experiment if specified
    if args.tag and "tag" in df.columns:
        df = df[df["tag"].astype(str) == str(args.tag)].copy()
    if args.experiments:
        keep = set(map(str, args.experiments))
        df = df[df["experiment"].astype(str).isin(keep)].copy()

    # Determine prompt categories
    if args.prompt_categories:
        keep_cats = {_canon_prompt(c) for c in args.prompt_categories}
        df = df[df["prompt_category"].astype(str).isin(keep_cats)].copy()
    else:
        # default overlay of numeric & CoT only
        df = df[df["prompt_category"].astype(str).isin(["numeric", "CoT"])].copy()

    # Pick metric column and drop rows without it
    mk = _canon_metric_key(args.metric)
    metric_col = (
        "EA_raw" if mk == "ea" else ("MV_raw" if mk == "mv" else ("loocv_r2" if mk == "r2" else "normative_metric"))
    )
    if metric_col not in df.columns:
        raise KeyError(f"Column {metric_col} not in CSV")
    df = df[pd.to_numeric(df[metric_col], errors="coerce").notna()].copy()

    # Resolve experiments to iterate
    exps = sorted(df["experiment"].astype(str).unique().tolist())
    out_root = Path(args.output_dir) / mk / (args.tag or "all")

    # Iterate per experiment
    human_baseline = None
    if args.show_human_baseline:
        human_baseline = _get_rw17_human_baseline(_load_data(csv_path), args.metric)
    rng = np.random.default_rng(args.seed)
    # Establish shared x-range and ticks per metric across all plots in this run
    metric_col = (
        "EA_raw" if mk == "ea" else ("MV_raw" if mk == "mv" else ("loocv_r2" if mk == "r2" else "normative_metric"))
    )
    # Use pooled rows if available for range estimation, otherwise all
    df_range = df.copy()
    if "domain" in df_range.columns:
        pooled = df_range[df_range["domain"].astype(str) == "all"].copy()
        if not pooled.empty:
            df_range = pooled
    vals_for_range = pd.to_numeric(df_range[metric_col], errors="coerce")
    vals_for_range = vals_for_range[np.isfinite(vals_for_range)]
    if not vals_for_range.empty:
        vmin = float(np.floor(vals_for_range.min() * 10.0) / 10.0)
        vmax = float(np.ceil(vals_for_range.max() * 10.0) / 10.0)
        shared_ticks = np.round(np.arange(vmin, vmax + 1e-9, 0.1), 1)
        shared_xlim = (vmin, vmax)
    else:
        shared_ticks = None
        shared_xlim = None

    for exp in exps:
        dfe = df[df["experiment"].astype(str) == exp].copy()
        domains = _pick_domains(dfe, args.by_domain)
        for dom in domains:
            if dom == "all":
                # If domain column exists, prefer pooled rows; else treat all as pooled
                if "domain" in dfe.columns:
                    pooled = dfe[dfe["domain"].astype(str) == "all"].copy()
                    sub = pooled if not pooled.empty else (
                        dfe.groupby(["agent", "prompt_category"], dropna=False)[metric_col]
                           .mean().reset_index()
                    )
                    # Provide source rows across domains to estimate CIs when no precomputed CIs
                    src = dfe[dfe["domain"].astype(str) != "all"].copy()
                    if src.empty:
                        src = None
                        if args.show_ci:
                            print(f"[info] No non-pooled domain rows for {exp}; CI whiskers may be suppressed.")
                else:
                    sub = dfe.copy()
                    src = None
            else:
                sub = dfe[dfe["domain"].astype(str) == dom].copy()
                if sub.empty:
                    continue
                src = None  # single domain -> no CI across domains
            _plot_overlay(
                sub=sub,
                metric=args.metric,
                experiment=exp,
                domain=dom,
                out_dir=out_root / exp,
                title_override=args.title,
                legend_loc=args.legend_loc,
                fig_size=(args.fig_width, args.fig_height),
                threshold=args.threshold,
                show_human=args.show_human_baseline,
                human_baseline=human_baseline,
                ea_shade_from=args.show_ea,
                show=(args.show and not args.no_show),
                show_ci=args.show_ci,
                B=args.bootstrap,
                ci=args.ci,
                rng=rng,
                source_all_domains=src,
                shared_xlim=shared_xlim,
                shared_xticks=shared_ticks,
                shade_r2=(not args.no_r2_shade),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
