#!/usr/bin/env python3
"""
Normativity vs. model release date — single scatter
===================================================

Plots one figure with normative_metric on the y-axis and model columns on the x-axis.
Columns are grouped by release date (models sharing the same release date appear next to each other).
The x-tick labels are LLM names (agents). Every other model column is lightly shaded in gray to aid
visual separation. Colors encode prompt category and markers encode experiment family, matching
`plot_normative_metric_scatter.py` conventions (triangles for RW17/Abstract, size by overloaded).

Input
-----
- CSV: results/cross_cogn_strategies/masters_classified_strategy_metrics.csv
- YAML: src/causalign/config/model_releases.yaml (maps agent -> YYYY-MM-DD)

Output
------
- results/plots/normativity_vs_release/normativity_vs_release.{pdf,png}
- results/plots/normativity_vs_release/ea_vs_release.{pdf,png}
- results/plots/normativity_vs_release/mv_vs_release.{pdf,png}
- results/plots/normativity_vs_release/loocv_r2_vs_release.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

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
        if tl in {"cot", "pccot", "chain_of_thought", "chain-of-thought", "cot_stepwise"}:
            return "CoT"
        return t


# Experiment families for marker encoding (match existing script)
EXP_FAMILIES: Dict[str, str] = {
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


# def _marker_for_experiment(exp: str) -> Tuple[str, float]:
#     fam = _infer_exp_family(exp)
#     if fam == "abstract":
#         return ("v", 45.0)  # small down triangle
#     if fam == "abstract_overloaded":
#         return ("v", 170.0)  # large down triangle
#     if fam == "rw17":
#         return ("^", 45.0)  # small up triangle
#     if fam == "rw17_overloaded":
#         return ("^", 170.0)  # large up triangle
#     return ("o", 35.0)  # fallback



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


def _ensure_tueplots(usetex: bool = False) -> None:
    try:
        from tueplots import bundles  # type: ignore
    except Exception:
        mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 12})
        return
    cfg = bundles.neurips2023(nrows=1, ncols=1, rel_width=0.9, usetex=usetex, family="serif")
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
    p = argparse.ArgumentParser(description="Leak-Adjusted Determinacy (LAD vs model release date scatter plot")
    p.add_argument(
        "--input-csv",
        default="results/cross_cogn_strategies/masters_classified_strategy_metrics.csv",
        help="Path to masters_classified_strategy_metrics.csv",
    )
    p.add_argument(
        "--releases-yaml",
        default="src/causalign/config/model_releases.yaml",
        help="Path to YAML mapping agent -> YYYY-MM-DD",
    )
    p.add_argument("--experiments", nargs="+", help="Experiments to include; omit to include all")
    p.add_argument("--tag", help="Optional tag filter (CSV column 'tag')")
    p.add_argument("--prompt-categories", nargs="+", help="Prompt categories to include (default: numeric & CoT)")
    p.add_argument("--usetex", action="store_true", help="Use LaTeX rendering (requires LaTeX installed)")
    p.add_argument("--fig-width", type=float, default=8.5)
    p.add_argument("--fig-height", type=float, default=4.8)
    p.add_argument("--output-dir", default="results/plots/normativity_vs_release")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prompt_category" in df.columns:
        df["prompt_category"] = df["prompt_category"].astype(str).map(canon_prompt_category)
    return df


def _load_releases(path: Path) -> Dict[str, pd.Timestamp]:
    with open(path, "r") as f:
        mapping = yaml.safe_load(f) or {}
    out: Dict[str, pd.Timestamp] = {}
    for k, v in mapping.items():
        try:
            out[str(k)] = pd.to_datetime(str(v))
        except Exception:
            continue
    return out


def _get_rw17_human_baselines(df_all: pd.DataFrame) -> dict:
    """Return baselines for RW17 numeric human.

    Keys: {'y': normative_metric, 'ea': EA_raw, 'mv': MV_raw, 'r2': loocv_r2} when available.
    Prefers pooled domain ('all') if present; otherwise mean across matching rows.
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
    out: dict = {}
    if y is not None:
        out["y"] = y
    if x_ea is not None:
        out["ea"] = x_ea
    if x_mv is not None:
        out["mv"] = x_mv
    if x_r2 is not None:
        out["r2"] = x_r2
    return out


def main() -> int:
    args = _parse_args()
    _ensure_tueplots(args.usetex)

    csv_path = Path(args.input_csv)
    yaml_path = Path(args.releases_yaml)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Releases YAML not found: {yaml_path}")

    df = _load_data(csv_path)
    releases = _load_releases(yaml_path)

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
        # default: numeric & CoT
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

    # Map agents to release dates; drop rows without mapping
    df["release_date"] = pd.to_datetime(df["agent"].map(releases))
    missing = df["release_date"].isna().sum()
    if missing:
        missing_agents = sorted(df.loc[df["release_date"].isna(), "agent"].astype(str).unique().tolist())
        print(f"[warn] Skipping {missing} rows due to missing release dates for agents: {missing_agents}")
        df = df[df["release_date"].notna()].copy()

    if df.empty:
        print("[info] No rows to plot after filtering and release-date mapping.")
        return 0

    # Sort by release date for deterministic plotting order
    df = df.sort_values("release_date")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legends (shared handles)
    cat_handles, cat_labels, exp_handles, exp_labels = _build_legends()

    # Build grouped x positions: group by release_date, within group by agent
    # Create an ordered mapping: [(date, [agents...]) ...]
    order = []
    for date, sub in df.groupby("release_date"):
        agents = sorted(sub["agent"].astype(str).unique().tolist())
        order.append((date, agents))
    order.sort(key=lambda t: t[0])

    gap = 0.6  # horizontal gap between release-date groups (in x units)
    xpos: Dict[str, float] = {}
    xcenters: list[float] = []
    xlabels: list[str] = []
    idx = 0.0
    for date, agents in order:
        for a in agents:
            xpos[a] = idx
            xcenters.append(idx)
            xlabels.append(f"{pd.to_datetime(date).strftime('%m/%Y')} --  {a}")
            idx += 1.0
        idx += gap

    # Compute human baselines from the unfiltered CSV (RW17, numeric)
    loaded_all = _load_data(csv_path)
    human_baselines = _get_rw17_human_baselines(loaded_all)

    def _plot_single(y_values: pd.Series, ylabel: str, out_base: str, metric_key: str) -> None:
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

        # Background alternating shading for each model column
        yclean = pd.to_numeric(y_values, errors="coerce")
        arr = np.asarray(yclean, dtype=float)
        ymin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else -1.0
        ymax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = -1.0, 1.0
        ypad = 0.05 * (ymax - ymin if (ymax - ymin) > 0 else 1.0)
        for i, x in enumerate(xcenters):
            if i % 2 == 1:
                ax.add_patch(Rectangle((x - 0.5, ymin - ypad), 1.0, (ymax - ymin) + 2 * ypad, color="0.94", zorder=0))

        # Grid and guide
        ax.axhline(0.0, color="gray", lw=1.0, ls=(0, (3, 2)), alpha=0.7, zorder=1)
        ax.grid(True, which="major", axis="y", alpha=0.15)

        # Plot points at model column centers
        for i, (idx_row, row) in enumerate(df.iterrows()):
            a = str(row["agent"])  # model name
            x = xpos.get(a)
            if x is None:
                continue
            # Use position-based lookup to avoid index type issues
            y = y_values.iloc[i] if i < len(y_values) else np.nan
            if not (pd.notna(y) and np.isfinite(float(y))):
                continue
            color = PROMPT_CATEGORY_COLORS.get(str(row["prompt_category"]), (0.2, 0.2, 0.2))
            mk, msz = _marker_for_experiment(str(row["experiment"]))
            ax.scatter(
                x,
                float(y),
                s=msz,
                marker=mk,
                facecolors=color,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
                zorder=3,
            )

        # Axis labels and formatting
        ax.set_xlabel("Release Date -- LLM")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xcenters)
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_xlim(min(xcenters) - 0.6, max(xcenters) + 0.6)

        # Human baseline dashed line (if available for this metric)
        hb_map = {"norm": "y", "ea": "ea", "mv": "mv", "r2": "r2"}
        hb_key = hb_map.get(metric_key)
        hb_val = human_baselines.get(hb_key) if hb_key is not None else None
        # Compose legends (category + human baseline) and draw baseline
        cat_handles_local = list(cat_handles)
        cat_labels_local = list(cat_labels)
        if hb_val is not None and np.isfinite(hb_val):
            ax.axhline(hb_val, color="hotpink", lw=1.6, ls="--", alpha=0.9, zorder=2)
            hb_handle = Line2D([0, 1], [0, 0], color="hotpink", lw=1.6, ls="--")
            cat_handles_local.append(hb_handle)
            cat_labels_local.append("human baseline (RW17, numeric)")

        # Legends: for MV move legends to top positions; otherwise use bottom placement
        if metric_key == "mv":
            cat_legend = ax.legend(
                cat_handles_local,
                cat_labels_local,
                title="Prompt category",
                loc="upper right",
                frameon=True,
                framealpha=0.6,
            )
            ax.add_artist(cat_legend)
            ax.legend(exp_handles, exp_labels, title="Experiment", loc="upper center", frameon=True, framealpha=0.6)
        else:
            # default placement
            cat_legend = ax.legend(
                cat_handles_local,
                cat_labels_local,
                title="Prompt category",
                loc="lower left",
                frameon=True,
                framealpha=0.6,
            )
            ax.add_artist(cat_legend)
            ax.legend(exp_handles, exp_labels, title="Experiment", loc="lower center", frameon=True, framealpha=0.6)
        

        fig.tight_layout()

        out_pdf = out_dir / f"{out_base}.pdf"
        out_png = out_dir / f"{out_base}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[ok] Saved: {out_pdf} and {out_png}")

    # Prepare series for each metric
    norm_series = pd.to_numeric(df["normative_metric"], errors="coerce")
    ea_series = (
        pd.to_numeric(df["EA_raw_mean"], errors="coerce").fillna(pd.to_numeric(df["EA_raw"], errors="coerce"))
        if "EA_raw_mean" in df.columns
        else pd.to_numeric(df["EA_raw"], errors="coerce")
    )
    mv_series = (
        pd.to_numeric(df["MV_raw_mean"], errors="coerce").fillna(pd.to_numeric(df["MV_raw"], errors="coerce"))
        if "MV_raw_mean" in df.columns
        else pd.to_numeric(df["MV_raw"], errors="coerce")
    )
    r2_series = pd.to_numeric(df.get("loocv_r2", pd.Series(np.nan, index=df.index)), errors="coerce")

    # Plot and save figures
    _plot_single(norm_series, "Leak-Adjusted Determinacy ", "normativity_vs_release", "norm")
    _plot_single(ea_series, "Explaining-away (EA)-level", "ea_vs_release", "ea")
    _plot_single(mv_series, "Markov violation (MV)-level", "mv_vs_release", "mv")
    _plot_single(r2_series, "LOOCV $R^2$", "loocv_r2_vs_release", "r2")

    # Optional interactive show
    if args.show and not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
