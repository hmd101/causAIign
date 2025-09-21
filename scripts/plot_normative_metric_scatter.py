#!/usr/bin/env python3
"""
Normative metric vs. EA/MV/R² scatter — per agent
=================================================

For each agent, generate one figure with three subplots (EA, MV, R² on x-axis; normative_metric on y-axis),
including all available experiments and prompt categories. Colors encode prompt category, and markers encode
experiment type:

- Abstract: small diamond
- Abstract-Overloaded: large diamond
- RW17: small square
- RW17-Overloaded: large square

If --show-ci is passed, horizontal x-whiskers are drawn when precomputed CI columns exist (EA_raw_lo/hi, MV_raw_lo/hi).

Input
-----
results/cross_cogn_strategies/masters_classified_strategy_metrics.csv

Output
------
results/plots/normative_scatter/<tag or all>/<agent>/normative_scatter_<agent>.pdf (and .png)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette and category canon
try:
    from causalign.plotting.palette import PROMPT_CATEGORY_COLORS, canon_prompt_category
except Exception:
    PROMPT_CATEGORY_COLORS = {
        "numeric": (0.85, 0.60, 0.55),
        "CoT": (0.00, 0.20, 0.55),
    }

    def canon_prompt_category(label: str) -> str:  # type: ignore
        t = str(label).strip()
        tl = t.lower()
        if tl in {"numeric", "pcnum", "num", "single_numeric", "single_numeric_response"}:
            return "numeric"
        if tl in {"cot", "pccot", "chain_of_thought", "chain-of-thought", "cot_stepwise", "cot", "CoT".lower()}:
            return "CoT"
        return t


EXP_FAMILIES: Dict[str, str] = {
    # map known experiments to family labels used for marker selection
    "random_abstract": "abstract",
    "abstract_overloaded_lorem_de": "abstract_overloaded",
    "rw17_indep_causes": "rw17",
    "rw17_overloaded_de": "rw17_overloaded",
    "rw17_overloaded_d": "rw17_overloaded",
    "rw17_overloaded_e": "rw17_overloaded",
}


def _infer_exp_family(exp: str) -> str:
    e = str(exp).strip()
    if e in EXP_FAMILIES:
        return EXP_FAMILIES[e]
    el = e.lower()
    if "abstract_overloaded" in el:
        return "abstract_overloaded"
    if "abstract" in el:
        return "abstract"
    if "rw17_overloaded" in el:
        return "rw17_overloaded"
    if "rw17" in el:
        return "rw17"
    return "other"


def _ensure_tueplots(usetex: bool = False) -> None:
    try:
        from tueplots import bundles  # type: ignore
    except Exception:
        mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 12})
        return
    cfg = bundles.neurips2023(nrows=1, ncols=3, rel_width=0.95, usetex=usetex, family="serif")
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-agent normative metric vs EA/MV/R² scatter plots")
    p.add_argument(
        "--input-csv",
        default="results/cross_cogn_strategies/masters_classified_strategy_metrics.csv",
        help="Path to masters_classified_strategy_metrics.csv",
    )
    p.add_argument("--experiments", nargs="+", help="Experiments to include; omit to include all")
    p.add_argument("--tag", help="Optional tag filter (CSV column 'tag')")
    p.add_argument("--prompt-categories", nargs="+", help="Prompt categories to include (default: numeric & CoT)")
    p.add_argument("--usetex", action="store_true", help="Use LaTeX rendering (requires LaTeX installed)")
    p.add_argument("--fig-width", type=float, default=12.0)
    p.add_argument("--fig-height", type=float, default=4.0)
    p.add_argument("--output-dir", default="results/plots/normative_scatter")
    p.add_argument("--show-ci", action="store_true", help="Draw horizontal CI whiskers on x where available")
    p.add_argument(
        "--share-axes",
        action="store_true",
        help=(
            "If set, all agent figures share the same y-range (based on min/max normative_metric across agents) "
            "and each metric subplot shares an x-range across agents (based on data for that metric)."
        ),
    )
    p.add_argument(
        "--show-human-baseline",
        action="store_true",
        help="Plot a pink star for the RW17 numeric human baseline and add a legend entry.",
    )
    p.add_argument(
        "--plot-missing-x",
        action="store_true",
        help=(
            "Plot rows with missing x-metric at a sentinel x position (hollow markers) instead of omitting them."
        ),
    )
    p.add_argument(
        "--missing-x-sentinel",
        type=float,
        default=-0.05,
        help="Sentinel x value to use for missing x-metric rows.",
    )
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--no-subtitle",
        action="store_true",
        help="Hide the per-figure subtitle (e.g., 'LLM: <agent>'), leaving only the suptitle.",
    )
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Hide the big suptitle, but keep the small per-figure subtitle (unless --no-subtitle is also set).",
    )
    return p.parse_args()


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prompt_category" in df.columns:
        df["prompt_category"] = df["prompt_category"].astype(str).map(canon_prompt_category)
    return df


def _metric_cols(metric: str) -> Tuple[str, Optional[str], Optional[str], str]:
    """Return (x_col, lo_col, hi_col, pretty_label) for a metric key 'ea'|'mv'|'r2'.

    For EA/MV, prefer *_raw_mean if present, otherwise *_raw.
    """
    m = metric.lower()
    if m == "ea":
        return ("EA_raw", "EA_raw_lo", "EA_raw_hi", r"Explaining-away (EA)-level")
    if m == "mv":
        return ("MV_raw", "MV_raw_lo", "MV_raw_hi", r"Markov violation (MV)-level")
    return ("loocv_r2", None, None, r"LOOCV $R^2$")


def _get_x_value(row: pd.Series, x_col: str) -> float:
    if x_col == "EA_raw":
        if "EA_raw_mean" in row and pd.notna(row["EA_raw_mean"]):
            return float(row["EA_raw_mean"])
    if x_col == "MV_raw":
        if "MV_raw_mean" in row and pd.notna(row["MV_raw_mean"]):
            return float(row["MV_raw_mean"])
    v = row.get(x_col, np.nan)
    return float(v) if pd.notna(v) else float("nan")


def _series_for_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    """Return a numeric Series of x-values for a metric, preferring *_raw_mean if present for EA/MV."""
    x_col, _, _, _ = _metric_cols(metric)
    if x_col == "EA_raw" and "EA_raw_mean" in df.columns:
        s = pd.to_numeric(df["EA_raw_mean"], errors="coerce")
        s = s.fillna(pd.to_numeric(df["EA_raw"], errors="coerce"))
        return s
    if x_col == "MV_raw" and "MV_raw_mean" in df.columns:
        s = pd.to_numeric(df["MV_raw_mean"], errors="coerce")
        s = s.fillna(pd.to_numeric(df["MV_raw"], errors="coerce"))
        return s
    return pd.to_numeric(df[x_col], errors="coerce")


def _get_rw17_human_baselines(df_all: pd.DataFrame) -> dict:
    """Return baselines for human RW17 numeric: {'y': normative_metric, 'ea': x_ea, 'mv': x_mv, 'r2': x_r2}.

    Uses pooled domain ('all') if present, otherwise mean across matching rows. Returns {} when not found.
    """
    d = df_all.copy()
    if "prompt_category" in d.columns:
        d["prompt_category"] = d["prompt_category"].astype(str).map(canon_prompt_category)
    mask = (
        d["agent"].astype(str).str.lower().str.contains("human")
        & d["experiment"].astype(str).eq("rw17_indep_causes")
        & d["prompt_category"].astype(str).eq("numeric")
    )
    sub = d[mask].copy()
    if sub.empty:
        return {}
    def pooled_or_mean(col: str) -> Optional[float]:
        if col not in sub.columns:
            return None
        ss = pd.to_numeric(sub[col], errors="coerce")
        sub2 = sub[ss.notna()].copy()
        if sub2.empty:
            return None
        if "domain" in sub2.columns:
            pool = sub2[sub2["domain"].astype(str) == "all"]
            if not pool.empty:
                return float(pd.to_numeric(pool.iloc[0][col], errors="coerce"))
        return float(ss.mean())
    y = pooled_or_mean("normative_metric")
    x_ea = pooled_or_mean("EA_raw")
    x_mv = pooled_or_mean("MV_raw")
    x_r2 = pooled_or_mean("loocv_r2")
    out = {}
    if y is not None:
        out["y"] = y
    if x_ea is not None:
        out["ea"] = x_ea
    if x_mv is not None:
        out["mv"] = x_mv
    if x_r2 is not None:
        out["r2"] = x_r2
    return out


def _get_rw17_human_baseline_ci(df_all: pd.DataFrame) -> dict:
    """Return CI bounds for human RW17 numeric pooled row if available.

    Keys: {'ea_lo','ea_hi','mv_lo','mv_hi'} when present.
    """
    d = df_all.copy()
    if "prompt_category" in d.columns:
        d["prompt_category"] = d["prompt_category"].astype(str).map(canon_prompt_category)
    mask = (
        d["agent"].astype(str).str.lower().str.contains("human")
        & d["experiment"].astype(str).eq("rw17_indep_causes")
        & d["prompt_category"].astype(str).eq("numeric")
    )
    sub = d[mask].copy()
    if sub.empty:
        return {}
    # Prefer pooled domain row for CI
    if "domain" in sub.columns:
        pooled = sub[sub["domain"].astype(str) == "all"].copy()
        if not pooled.empty:
            row = pooled.iloc[0]
        else:
            row = sub.iloc[0]
    else:
        row = sub.iloc[0]
    out: dict = {}
    for m, lo_key, hi_key in (
        ("ea", "EA_raw_lo", "EA_raw_hi"),
        ("mv", "MV_raw_lo", "MV_raw_hi"),
    ):
        lo = pd.to_numeric(pd.Series([row.get(lo_key, np.nan)])).iloc[0]
        hi = pd.to_numeric(pd.Series([row.get(hi_key, np.nan)])).iloc[0]
        if np.isfinite(lo) and np.isfinite(hi) and hi != lo:
            out[f"{m}_lo"] = float(lo)
            out[f"{m}_hi"] = float(hi)
    return out


def _marker_for_experiment(exp: str) -> Tuple[str, float]:
    fam = _infer_exp_family(exp)
    if fam == "abstract":
        # return ("v", 45.0)  # small down triangle
        return ("d", 45.0)  # small down triangle
    if fam == "abstract_overloaded":
        # return ("v", 170.0)  # large down triangle
        return ("d", 170.0)  # large down triangle
    if fam == "rw17":
        # return ("^", 45.0)  # small up triangle
        return ("s", 45.0)  # small up triangle
    if fam == "rw17_overloaded":
        # return ("^", 170.0)  # large up triangle
        return ("s", 170.0)  # large up triangle
    return ("o", 35.0)  # fallback


def _build_legends() -> Tuple[list, list, list, list]:
    # Prompt category color legend
    cat_handles, cat_labels = [], []
    for cat, color in PROMPT_CATEGORY_COLORS.items():
        h = Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=color, markersize=8)
        cat_handles.append(h)
        cat_labels.append(cat)
    # Experiment marker legend (shape/size only)
    exp_items = [
        # ("Abstract", ("v", 45.0)),
        # ("Abstract-Overloaded", ("v", 170)),
        # ("RW17", ("^", 45.0)),
        # ("RW17-Overloaded", ("^", 170)),
        ("Abstract", ("d", 45.0)),
        ("Abstract-Overloaded", ("d", 170)),
        ("RW17", ("s", 45.0)),
        ("RW17-Overloaded", ("s", 170)),
    ]
    exp_handles, exp_labels = [], []
    for label, (mk, sz) in exp_items:
        h = Line2D([0], [0], marker=mk, color="black", markerfacecolor="white", markeredgecolor="black", markersize=np.sqrt(sz))
        exp_handles.append(h)
        exp_labels.append(label)
    return cat_handles, cat_labels, exp_handles, exp_labels


def main() -> int:
    args = _parse_args()
    _ensure_tueplots(args.usetex)

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = _load_data(csv_path)

    # Filters
    if args.tag and "tag" in df.columns:
        df = df[df["tag"].astype(str) == str(args.tag)].copy()
    if args.experiments:
        keep = set(map(str, args.experiments))
        df = df[df["experiment"].astype(str).isin(keep)].copy()
    if args.prompt_categories:
        keep_cats = {canon_prompt_category(c) for c in args.prompt_categories}
        df = df[df["prompt_category"].astype(str).isin(keep_cats)].copy()
    else:
        df = df[df["prompt_category"].astype(str).isin(["numeric", "CoT"])].copy()

    # Require normative_metric
    if "normative_metric" not in df.columns:
        raise KeyError("Column 'normative_metric' not in CSV")
    df = df[pd.to_numeric(df["normative_metric"], errors="coerce").notna()].copy()

    # Prefer pooled domain ('all') if present
    if "domain" in df.columns:
        pooled = df[df["domain"].astype(str) == "all"].copy()
        if not pooled.empty:
            df = pooled

    out_root = Path(args.output_dir) / (args.tag or "all")
    agents = sorted(df["agent"].astype(str).unique().tolist())

    # Legends (shared handles)
    cat_handles, cat_labels, exp_handles, exp_labels = _build_legends()

    # Optionally compute shared axis limits across all agents
    shared_y_by_metric: Dict[str, Tuple[float, float]] = {}
    shared_x: Dict[str, Tuple[float, float]] = {}
    if args.share_axes:
        # per-metric y-limits based on rows with available x-values
        for mkey in ("ea", "mv", "r2"):
            s_all = _series_for_metric(df, mkey)
            mask = pd.to_numeric(s_all, errors="coerce").notna()
            yvals_m = pd.to_numeric(df.loc[mask, "normative_metric"], errors="coerce")
            yvals_m = yvals_m[np.isfinite(yvals_m)]
            if not yvals_m.empty:
                step = 0.2
                ymin_raw = float(yvals_m.min())
                ymax_raw = float(yvals_m.max())
                ymin = float(np.floor(ymin_raw / step) * step)
                ymax = float(np.ceil(ymax_raw / step) * step)
                if ymin == ymax:
                    ymin -= step
                    ymax += step
                shared_y_by_metric[mkey] = (ymin, ymax)
        # per-metric x-limits as before
        for mkey in ("ea", "mv", "r2"):
            s = _series_for_metric(df, mkey)
            s = s[np.isfinite(s)]
            if not s.empty:
                xmin = float(np.floor(s.min() * 10.0) / 10.0)
                xmax = float(np.ceil(s.max() * 10.0) / 10.0)
                shared_x[mkey] = (xmin, xmax)

    # Optional human baseline (RW17, numeric)
    loaded_all = _load_data(csv_path)
    human_baselines = _get_rw17_human_baselines(loaded_all) if args.show_human_baseline else {}
    human_ci = _get_rw17_human_baseline_ci(loaded_all) if args.show_human_baseline else {}
    human_star_handle = None
    if args.show_human_baseline and human_baselines.get("y") is not None:
        human_star_handle = Line2D([0], [0], marker='*', color='hotpink', markerfacecolor='hotpink', markeredgecolor='black', markersize=10, linestyle='None')
        cat_handles = cat_handles + [human_star_handle]
        cat_labels = cat_labels + ["human baseline (RW17, numeric)"]
    # Add CI proxy line to legend if requested
    if args.show_ci:
        ci_handle = Line2D([0, 1], [0, 0], color="0.3", lw=1.0, alpha=0.45)
        cat_handles = cat_handles + [ci_handle]
        cat_labels = cat_labels + ["95% CI"]
    # Add legend entry for missing-x sentinel if enabled
    if args.plot_missing_x:
        missing_handle = Line2D(
            [0], [0], marker="o", color="black", markerfacecolor="white", markeredgecolor="black", markersize=8, linestyle="None"
        )
        cat_handles = cat_handles + [missing_handle]
        cat_labels = cat_labels + ["x-metric missing (sentinel)"]

    for agent in agents:
        sub = df[df["agent"].astype(str) == agent].copy()
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(args.fig_width, args.fig_height), sharey=True)
        # Order: R² (left), EA (middle), MV (right)
        metrics = [("r2", axes[0]), ("ea", axes[1]), ("mv", axes[2])]
        axes_by_metric = {k: ax for k, ax in metrics}

        # Plot each metric
        for metric_key, ax in metrics:
            x_col, lo_col, hi_col, x_label = _metric_cols(metric_key)
            # precompute x values using helper to respect *_raw_mean overrides
            x_vals = sub.apply(lambda r: _get_x_value(r, x_col), axis=1)
            is_x = np.isfinite(x_vals.to_numpy(dtype=float))
            s = sub[is_x].copy()
            s_missing = sub[~is_x].copy()
            if s.empty and (not args.plot_missing_x or s_missing.empty):
                ax.set_title(f"No data for {metric_key.upper()}")
                ax.set_xlabel(x_label)
                # zero guide lines still helpful
                ax.axvline(0.0, color="0.5", ls="--", lw=0.8, zorder=0)
                ax.axhline(0.0, color="0.5", ls="--", lw=0.8, zorder=0)
                # share axes if requested
                if args.share_axes:
                    if metric_key in shared_y_by_metric:
                        ax.set_ylim(*shared_y_by_metric[metric_key])
                        # y ticks at 0.2 increments
                        y0, y1 = ax.get_ylim()
                        y_step = 0.2
                        yt_start = np.floor(y0 / y_step) * y_step
                        yt_end = np.ceil(y1 / y_step) * y_step
                        yticks = np.round(np.arange(yt_start, yt_end + 1e-9, y_step), 2)
                        if len(yticks) > 1:
                            ax.set_yticks(yticks)
                    if metric_key in shared_x:
                        ax.set_xlim(*shared_x[metric_key])
                continue

            # draw each row as a point with optional CI whisker
            for idx, row in s.iterrows():
                y = float(row["normative_metric"]) if pd.notna(row["normative_metric"]) else np.nan
                if not np.isfinite(y):
                    continue
                # use precomputed x value
                x = float(x_vals.loc[idx])
                if not np.isfinite(x):
                    continue
                color = PROMPT_CATEGORY_COLORS.get(canon_prompt_category(row["prompt_category"]), (0.5, 0.5, 0.5))
                mk, size = _marker_for_experiment(str(row["experiment"]))

                # CI whiskers on x where available and requested
                if args.show_ci and lo_col and hi_col:
                    lo = float(row.get(lo_col, np.nan))
                    hi = float(row.get(hi_col, np.nan))
                    if np.isfinite(lo) and np.isfinite(hi) and hi != lo:
                        # Make EA/MV whiskers a bit more prominent than defaults
                        if metric_key in ("ea", "mv"):
                            ax.plot([lo, hi], [y, y], color=color, lw=1.4, alpha=0.85, zorder=1)
                        else:
                            ax.plot([lo, hi], [y, y], color=color, lw=1.0, alpha=0.85, zorder=1)

                ax.scatter(
                    [x], [y],
                    marker=mk,
                    s=size,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.95,
                    zorder=2,
                )

            # plot missing-x rows at sentinel, if enabled
            used_sentinel = False
            if args.plot_missing_x and not s_missing.empty:
                for _, row in s_missing.iterrows():
                    y = float(row.get("normative_metric", np.nan))
                    if not np.isfinite(y):
                        continue
                    color = PROMPT_CATEGORY_COLORS.get(canon_prompt_category(row.get("prompt_category", "")), (0.5, 0.5, 0.5))
                    mk, size = _marker_for_experiment(str(row.get("experiment", "")))
                    ax.scatter(
                        [args.missing_x_sentinel], [y],
                        marker=mk,
                        s=size,
                        facecolor="white",
                        edgecolor=color,
                        linewidth=1.0,
                        alpha=0.95,
                        zorder=2,
                    )
                    used_sentinel = True

            # dashed zero guide lines
            ax.axvline(0.0, color="0.5", ls="--", lw=0.8, zorder=0)
            ax.axhline(0.0, color="0.5", ls="--", lw=0.8, zorder=0)

            # sentinel indicator line
            if args.plot_missing_x and used_sentinel:
                ax.axvline(args.missing_x_sentinel, color="0.6", ls=":", lw=0.8, zorder=1)

            # human baseline star
            if args.show_human_baseline and human_baselines.get("y") is not None:
                xhb = human_baselines.get(metric_key)
                yhb = human_baselines.get("y")
                if xhb is not None and yhb is not None and np.isfinite(xhb) and np.isfinite(yhb):
                    ax.scatter([xhb], [yhb], marker='*', s=110, color='hotpink', edgecolor='black', linewidth=0.6, zorder=3)
                    # Draw human CI whisker for EA/MV if available
                    if metric_key == "ea" and ("ea_lo" in human_ci and "ea_hi" in human_ci):
                        ax.plot([human_ci["ea_lo"], human_ci["ea_hi"]], [yhb, yhb], color='hotpink', lw=1.6, alpha=0.8, zorder=2)
                    if metric_key == "mv" and ("mv_lo" in human_ci and "mv_hi" in human_ci):
                        ax.plot([human_ci["mv_lo"], human_ci["mv_hi"]], [yhb, yhb], color='hotpink', lw=1.6, alpha=0.8, zorder=2)

            ax.set_xlabel(x_label)
            ax.grid(True, axis="both", color="0.92", linestyle="--", linewidth=0.6)
            # share axes if requested
            if args.share_axes:
                if metric_key in shared_y_by_metric:
                    ax.set_ylim(*shared_y_by_metric[metric_key])
                    # y ticks at 0.2 increments
                    y0, y1 = ax.get_ylim()
                    y_step = 0.2
                    yt_start = np.floor(y0 / y_step) * y_step
                    yt_end = np.ceil(y1 / y_step) * y_step
                    yticks = np.round(np.arange(yt_start, yt_end + 1e-9, y_step), 2)
                    if len(yticks) > 1:
                        ax.set_yticks(yticks)
                if metric_key in shared_x:
                    ax.set_xlim(*shared_x[metric_key])
            # x ticks at 0.1 increments (auto-range based on (possibly shared) limits)
            xmin, xmax = ax.get_xlim()
            # ensure sentinel is visible when used and axes not shared
            if args.plot_missing_x and used_sentinel and metric_key not in shared_x:
                xmin = min(xmin, args.missing_x_sentinel - 0.02)
                ax.set_xlim(left=xmin)
            tick_start = np.floor(xmin * 10.0) / 10.0
            tick_end = np.ceil(xmax * 10.0) / 10.0
            ticks = np.round(np.arange(tick_start, tick_end + 1e-9, 0.1), 1)
            if len(ticks) > 1:
                ax.set_xticks(ticks)

        axes[0].set_ylabel("Leak-Adjusted Determinacy $\\mathrm{LAD}$")
        # axes[0].set_ylabel("Normativity-level: $\\overline{m}-b$ higher is more normative")

        title_prefix = "Leak-Adjusted Determinacy ($\\mathrm{LAD}=\\overline{m}-b$) vs $R^2$/EA/MV "
        title = title_prefix
        subtitle = f"LLM: {agent}"
        if not args.no_title:
            fig.suptitle(title)
        fig.subplots_adjust(top=0.88)
        if not args.no_subtitle:
            fig.text(0.5, 0.85, subtitle, ha='center', va='center', fontsize=12, color='black')

        # Compose legends: EA subplot for categories, R² subplot for experiments
        leg1 = axes_by_metric["ea"].legend(cat_handles, cat_labels, title="Prompt category", loc="best", frameon=True, framealpha=0.8)
        axes_by_metric["ea"].add_artist(leg1)
        axes_by_metric["r2"].legend(exp_handles, exp_labels, title="Experiment", loc="best", frameon=True, framealpha=0.8)

        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))

        agent_dir = out_root / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        base = f"normative_scatter_{agent}"
        fig.savefig(agent_dir / f"{base}.pdf", bbox_inches="tight")
        fig.savefig(agent_dir / f"{base}.png", dpi=300, bbox_inches="tight")
        if args.show and not args.no_show:
            plt.show()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
