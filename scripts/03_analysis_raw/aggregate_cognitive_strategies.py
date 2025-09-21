#!/usr/bin/env python3
"""
Aggregate cognitive strategies classifications across experiments/tags.

This script searches under results/parameter_analysis for any
  .../<experiment>/<tag>/cogn_analysis/**/classified_strategy_metrics.csv
files, concatenates them into a single master CSV, and produces:

1) results/cross_cogn_strategies/masters_classified_strategy_metrics.csv
   - A row-wise concatenation with added columns: experiment, tag,
     ea_diff_threshold, mv_diff_threshold, loocv_r2_threshold (parsed from
     file or parent folder when missing).

2) results/cross_cogn_strategies/normative_share_by_exp_pc.csv and .tex
   - Per (experiment × prompt_category × thresholds):
       n_normative, n_total, frac_normative, pct_normative

3) results/cross_cogn_strategies/normative_by_agent_matrix.csv
   - Wide table: one row per agent, one column per (experiment × prompt_category)
     indicating True/False/NA for "normative_reasoner". Includes aggregate
     columns per agent: normative_count, appearance_count, normative_frac, normative_pct.

Usage:
  python scripts/03_analysis_raw/aggregate_cognitive_strategies.py

Optionally specify a custom root via --root (default: results/parameter_analysis)
and an output directory via --out (default: results/cross_cogn_strategies).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple, cast

# Plotting
import matplotlib
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from tueplots import bundles, fonts

# Plotting configuration (NeurIPS-like with LaTeX fonts)
config = {**bundles.neurips2023(), **fonts.neurips2022_tex(family="serif")}
config["legend.title_fontsize"] = 9
config["font.size"] = 14
config["axes.labelsize"] = 14
config["axes.titlesize"] = 12
config["xtick.labelsize"] = 12
config["ytick.labelsize"] = 12
config["legend.fontsize"] = 9
config["text.latex.preamble"] = r"\usepackage{amsmath,amssymb,bm,xcolor} \definecolor{inference}{HTML}{FF5B59}"
mpl.rcParams.update(config)


# --- constants / styles -------------------------------------------------------

# Pretty label mapping for prompt categories (publication labels)
PC_PRINT_MAP = {"pcnum": "Num", "pccot": "CoT", "pcconf": "Num-Conf"}

# Experiment pretty names
exp_name_map = {
    "random_abstract": "Abstract",
    "rw17_indep_causes": "RW17",
    "abstract_overloaded_lorem_de": "Abstract-Over",
    "rw17_overloaded_de": "RW17-Over-DE",
    "rw17_overloaded_d": "RW17-Over-D",
    "rw17_overloaded_e": "RW17-Over",
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
        _NUMERIC_SYNS = {"pcnum", "numeric", "num", "single_numeric", "single_numeric_response"}
        _COT_SYNS = {"pccot", "cot", "chain_of_thought", "chain-of-thought", "cot_stepwise"}
        if t in _NUMERIC_SYNS or t == "numeric":
            return "numeric"
        if t in _COT_SYNS or t == "cot":
            return "CoT"
        return str(label)

# Assign colors from PROMPT_CATEGORY_COLORS
numeric_color = PROMPT_CATEGORY_COLORS["numeric"]
cot_color = PROMPT_CATEGORY_COLORS["CoT"]


# --- helpers -----------------------------------------------------------------

def _latex_escape(s: str) -> str:
    if s is None:
        return ""
    # Minimal LaTeX escaping for table text
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _to_bool(series: pd.Series) -> pd.Series:
    # Normalize typical representations of booleans
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "t": True, "f": False}

    def coerce(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return mapping.get(s, np.nan)

    return series.apply(coerce)


def _prompt_label(pc: str) -> str:
    t = (pc or "").strip().lower()
    if t == "pcnum" or t == "numeric":
        return "Numeric"
    if t == "pccot" or t == "cot":
        return "CoT"
    return pc or ""


_EA_RE = re.compile(r"ea[_-]?diff[_-]?([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MV_RE = re.compile(r"mv[_-]?diff[_-]?([0-9]*\.?[0-9]+)", re.IGNORECASE)
_R2_RE = re.compile(r"loocv[_-]?r2[_-]?([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_thresholds_from_path(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Expect a folder like: ea_diff_0.3_mv_diff_0.05_loocv_r2_0.89
    # Search in all parent names to be robust
    names = [p.name for p in path.parents]
    ea = mv = r2 = None
    for name in names:
        if ea is None:
            m = _EA_RE.search(name)
            if m:
                try:
                    ea = float(m.group(1))
                except Exception:
                    pass
        if mv is None:
            m = _MV_RE.search(name)
            if m:
                try:
                    mv = float(m.group(1))
                except Exception:
                    pass
        if r2 is None:
            m = _R2_RE.search(name)
            if m:
                try:
                    r2 = float(m.group(1))
                except Exception:
                    pass
    return ea, mv, r2


def _infer_experiment_and_tag(csv_path: Path) -> Tuple[str, str]:
    # results/parameter_analysis/<experiment>/<tag>/cogn_analysis/.../classified_strategy_metrics.csv
    parts = csv_path.parts
    # Find the index of "parameter_analysis" to be robust
    try:
        ix = parts.index("parameter_analysis")
        experiment = parts[ix + 1]
        tag = parts[ix + 2]
        return experiment, tag
    except Exception:
        return "", ""


def discover_classified_csvs(root: Path) -> List[Path]:
    # Recursive glob under root for any classified_strategy_metrics.csv
    # Typical layout has an extra thresholds subfolder in cogn_analysis
    matches = list(root.glob("**/cogn_analysis/**/classified_strategy_metrics.csv"))
    # Keep only files
    return [p for p in matches if p.is_file()]


def build_master(classified_files: List[Path]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for p in classified_files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        exp, tag = _infer_experiment_and_tag(p)
        if "experiment" not in df.columns:
            df["experiment"] = exp
        else:
            df["experiment"] = df["experiment"].fillna(exp).replace("", exp)
        if "tag" not in df.columns:
            df["tag"] = tag
        else:
            df["tag"] = df["tag"].fillna(tag).replace("", tag)

        # Ensure threshold columns exist; parse from path if missing
        ea, mv, r2 = _parse_thresholds_from_path(p)
        if "ea_diff_threshold" not in df.columns:
            df["ea_diff_threshold"] = ea
        else:
            df["ea_diff_threshold"] = df["ea_diff_threshold"].fillna(ea)
        if "mv_diff_threshold" not in df.columns:
            df["mv_diff_threshold"] = mv
        else:
            df["mv_diff_threshold"] = df["mv_diff_threshold"].fillna(mv)
        if "loocv_r2_threshold" not in df.columns:
            df["loocv_r2_threshold"] = r2
        else:
            df["loocv_r2_threshold"] = df["loocv_r2_threshold"].fillna(r2)

        # Normalize boolean
        if "normative_reasoner" in df.columns:
            df["normative_reasoner"] = _to_bool(df["normative_reasoner"]).astype("boolean")

        rows.append(df)

    if not rows:
        return pd.DataFrame()
    master = pd.concat(rows, ignore_index=True, sort=False)
    # Light dedupe: if duplicated rows exist, keep first
    master = master.drop_duplicates()
    return master


def summarize_by_experiment_pc(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=[
            "experiment", "prompt_category", "prompt_label",
            "ea_diff_threshold", "mv_diff_threshold", "loocv_r2_threshold",
            "n_normative", "n_total", "frac_normative", "pct_normative"
        ])

    cols_needed = [
        "experiment", "prompt_category", "normative_reasoner",
        "ea_diff_threshold", "mv_diff_threshold", "loocv_r2_threshold",
    ]
    for c in cols_needed:
        if c not in master.columns:
            master[c] = np.nan

    df = master.copy()
    df["prompt_label"] = df["prompt_category"].astype(str).map(_prompt_label)
    # Count only rows with an explicit boolean (True/False), ignore NA
    is_bool = df["normative_reasoner"].notna()
    df_present = df[is_bool].copy()
    grp_cols = [
        "experiment", "prompt_category", "prompt_label",
        "ea_diff_threshold", "mv_diff_threshold", "loocv_r2_threshold",
    ]
    agg = (
        df_present
        .groupby(grp_cols, dropna=False)
        .agg(n_total=("normative_reasoner", "size"),
             n_normative=("normative_reasoner", lambda s: int(pd.Series(s).astype(bool).sum())))
        .reset_index()
    )
    agg["frac_normative"] = np.where(agg["n_total"] > 0, agg["n_normative"] / agg["n_total"], np.nan)
    agg["pct_normative"] = agg["frac_normative"] * 100.0
    return agg


def to_latex_table(df: pd.DataFrame, out_tex: Path) -> None:
    if df.empty:
        out_tex.write_text("% Empty table: no data found\n")
        return
    # Minimal LaTeX tabular
    cols = [
        "experiment", "prompt_label",
        "ea_diff_threshold", "mv_diff_threshold", "loocv_r2_threshold",
        "n_normative", "n_total", "frac_normative", "pct_normative",
    ]
    df2 = df.copy()
    df2 = df2[cols]
    # Round numeric columns
    df2["frac_normative"] = df2["frac_normative"].round(3)
    df2["pct_normative"] = df2["pct_normative"].round(1)
    # Escape text
    df2["experiment"] = df2["experiment"].map(_latex_escape)
    df2["prompt_label"] = df2["prompt_label"].map(_latex_escape)

    headers = [
        "Experiment", "Prompt",
        "$\\mathrm{EA}_\\mathrm{thr}$", "$|\\mathrm{MV}|_\\mathrm{thr}$", "$R^2_\\mathrm{thr}$",
        "$n$ Norm.", "$n$ Total", "Frac", "Pct (\\%)",
    ]
    aligns = "l l r r r r r r r"

    lines: List[str] = []
    lines.append("% Auto-generated by aggregate_cognitive_strategies.py")
    lines.append("\\begin{tabular}{" + aligns + "}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + r" \\ ")
    lines.append("\\hline")
    for _, r in df2.iterrows():
        row = [
            r["experiment"], r["prompt_label"],
            (f"{r['ea_diff_threshold']:g}" if not pd.isna(r["ea_diff_threshold"]) else ""),
            (f"{r['mv_diff_threshold']:g}" if not pd.isna(r["mv_diff_threshold"]) else ""),
            (f"{r['loocv_r2_threshold']:g}" if not pd.isna(r["loocv_r2_threshold"]) else ""),
            (str(int(r["n_normative"])) if not pd.isna(r["n_normative"]) else ""),
            (str(int(r["n_total"])) if not pd.isna(r["n_total"]) else ""),
            (f"{r['frac_normative']:.3f}" if not pd.isna(r["frac_normative"]) else ""),
            (f"{r['pct_normative']:.1f}" if not pd.isna(r["pct_normative"]) else ""),
        ]
        lines.append(" & ".join(row) + r" \\ ")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_tex.write_text("\n".join(lines))


def build_agent_matrix(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    df = master.copy()
    # Ensure columns
    for c in ["experiment", "prompt_category", "agent", "normative_reasoner"]:
        if c not in df.columns:
            df[c] = np.nan
    # Take only rows where normative_reasoner was computed (True/False)
    df["normative_reasoner"] = _to_bool(df["normative_reasoner"]).astype("boolean")
    # Deduplicate if multiple rows per (exp, pc, agent). Prefer domain == "all" if present.
    # Strategy: sort so that domain == "all" comes first, then drop duplicates keeping first
    if "domain" in df.columns:
        df["_domain_rank"] = (df["domain"].astype(str).str.lower() != "all")
        df = df.sort_values(["agent", "experiment", "prompt_category", "_domain_rank"])  # all first
        df = df.drop(columns=["_domain_rank"])
    df = df.drop_duplicates(subset=["agent", "experiment", "prompt_category"], keep="first")

    # Build wide columns per (experiment × prompt_category)
    df["col"] = df["experiment"].astype(str) + "__" + df["prompt_category"].astype(str)
    mat = df.pivot_table(index="agent", columns="col", values="normative_reasoner", aggfunc="first")
    # Represent as strings: True/False/empty (cast to object to avoid masked boolean type issues)
    mat = mat.astype(object).replace({True: "True", False: "False"}).fillna("")

    # Count stats per agent
    present_mask = df.groupby("agent")["normative_reasoner"].apply(lambda s: s.notna().sum())
    norm_count = df.groupby("agent")["normative_reasoner"].apply(lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()))
    stats = pd.DataFrame({
        "normative_count": norm_count,
        "appearance_count": present_mask,
    })
    stats["normative_frac"] = np.where(stats["appearance_count"] > 0, stats["normative_count"] / stats["appearance_count"], np.nan)
    stats["normative_pct"] = stats["normative_frac"] * 100.0

    out = mat.join(stats, how="outer")
    out = out.reset_index().rename(columns={"index": "agent"})
    return out


def agent_matrix_to_latex(agent_mat: pd.DataFrame, out_tex: Path) -> None:
    """Write a LaTeX tabular for the agent-wise normative matrix.

    - Renames condition columns from '<experiment>__<prompt_category>' to
      '<PrettyExperiment> <PC label>' using exp_name_map and PC_PRINT_MAP.
    - Leaves empty cells for missing values.
    - Rounds normative_frac to 3 decimals and normative_pct to 1 decimal.
    """
    if agent_mat.empty:
        out_tex.write_text("% Empty agent matrix: no data found\n")
        return

    df = agent_mat.copy()
    # Identify condition columns
    agg_cols = {"normative_count", "appearance_count", "normative_pct"}
    base_cols = ["agent"]
    cond_cols = [c for c in df.columns if c not in agg_cols and c != "agent"]

    # Build pretty column names for conditions
    pretty_map: dict[str, str] = {}
    for c in cond_cols:
        if "__" in c:
            exp, pc = c.split("__", 1)
            exp_pretty = str(exp_name_map.get(exp, exp))
            pc_key = str(pc).strip().lower()
            pc_pretty = PC_PRINT_MAP.get(pc_key, pc)
            pretty = f"{exp_pretty} {pc_pretty}"
        else:
            pretty = c
        pretty_map[c] = pretty

    # Reorder columns: agent, condition columns (as seen), aggregates
    ordered_cols = base_cols + cond_cols + [
        "normative_count", "appearance_count", "normative_pct"
    ]
    df = df[ordered_cols]

    # Round aggregates
    if "normative_pct" in df.columns:
        df["normative_pct"] = pd.to_numeric(df["normative_pct"], errors="coerce").round(1)

    # Escape headers and cells
    headers = [
        _latex_escape("Agent"),
        *[_latex_escape(pretty_map.get(c, c)) for c in cond_cols],
        _latex_escape("Norm. count"),
        _latex_escape("Appear-ances"),
        _latex_escape("Pct (\\%)"),
    ]

    lines: List[str] = []
    lines.append("% Auto-generated by aggregate_cognitive_strategies.py (agent matrix)")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\label{tab:cognitive_strategies}")
    lines.append("\\scriptsize")
    lines.append("\\caption{Cognitive strategy classifications per agent across experiments and prompt categories.}")
    lines.append("\\begin{tabular}{l p{.7cm} p{.7cm} p{.7cm} p{.7cm} p{.7cm} p{.7cm}  p{.7cm} p{.7cm} p{.4cm} p{.4cm} p{.4cm} p{.4cm}}")
    lines.append(" & ".join(headers) + r" \\ ")
    lines.append("\\hline")

    def cell(v: object) -> str:
        if v is None:
            return ""
        try:
            if isinstance(v, float) and np.isnan(v):
                return ""
        except Exception:
            pass
        s = str(v)
        return _latex_escape(s)

    for _, row in df.iterrows():
        vals = [
            _latex_escape(str(row.get("agent", ""))),
            *[cell(row.get(c, np.nan)) for c in cond_cols],
            cell(row.get("normative_count", np.nan)),
            cell(row.get("appearance_count", np.nan)),
            cell(row.get("normative_pct", np.nan)),
        ]
        lines.append(" & ".join(vals) + r" \\ ")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines))


def _violin_bodies(v) -> List[PolyCollection]:
    """Best-effort extraction of violin body artists as a list."""
    try:
        # Matplotlib returns a dict with 'bodies'
        bodies = getattr(v, 'bodies', None)
        if bodies is None and isinstance(v, dict):
            bodies = v.get('bodies', [])
        if bodies is None:
            return []
        try:
            return [cast(PolyCollection, x) for x in list(bodies)]
        except Exception:
            return []
    except Exception:
        return []


def _violin_for_param(ax: Axes, exps: List[str], df: pd.DataFrame, param: str, subset_mask: pd.Series) -> None:
    """Draw per-experiment paired violins for pcnum and pccot with mean/median overlays."""
    num_data: List[np.ndarray] = []
    cot_data: List[np.ndarray] = []
    for e in exps:
        sub_e = df[(df["experiment"] == e) & subset_mask]
        s_num = pd.to_numeric(sub_e[sub_e["pc_norm"] == "pcnum"][param], errors="coerce").dropna().values
        s_cot = pd.to_numeric(sub_e[sub_e["pc_norm"] == "pccot"][param], errors="coerce").dropna().values
        num_data.append(np.asarray(s_num, dtype=float))
        cot_data.append(np.asarray(s_cot, dtype=float))

    group_spacing = 1.20
    base = np.arange(len(exps)) * group_spacing
    pair_offset = 0.2
    positions_num = base - pair_offset
    positions_cot = base + pair_offset
    width = 0.26

    for i in range(len(exps)):
        if len(num_data[i]) > 0:
            v = ax.violinplot([num_data[i]], positions=[positions_num[i]], widths=width, showextrema=False)
            bodies = _violin_bodies(v)
            if bodies:
                b = bodies[0]
                b.set_facecolor(numeric_color)
                b.set_edgecolor('black')
                b.set_alpha(0.85)
        if len(cot_data[i]) > 0:
            v = ax.violinplot([cot_data[i]], positions=[positions_cot[i]], widths=width, showextrema=False)
            bodies = _violin_bodies(v)
            if bodies:
                b = bodies[0]
                b.set_facecolor(cot_color)
                b.set_edgecolor('black')
                b.set_alpha(0.85)

    # Overlay mean (yellow triangle) and median (black line)
    for i in range(len(exps)):
        if len(num_data[i]) > 0:
            mean = float(np.mean(num_data[i]))
            med = float(np.median(num_data[i]))
            ax.scatter([positions_num[i]], [mean], marker='^', color='yellow', zorder=3)
            ax.hlines(med, positions_num[i]-0.12, positions_num[i]+0.12, colors='black', linewidth=1.4)
        if len(cot_data[i]) > 0:
            mean = float(np.mean(cot_data[i]))
            med = float(np.median(cot_data[i]))
            ax.scatter([positions_cot[i]], [mean], marker='^', color='yellow', zorder=3)
            ax.hlines(med, positions_cot[i]-0.12, positions_cot[i]+0.12, colors='black', linewidth=1.4)

    ax.set_xticks(base)
    ax.set_xticklabels([str(exp_name_map.get(e, e) or "") for e in exps], rotation=25)
    ax.set_xlabel('Experiment')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', linestyle=':', alpha=0.4)


def plot_violin_params_by_experiment_and_pc(master: pd.DataFrame, out_dest: Path, normative: bool) -> None:
    """2×2 grouped violins by experiment×prompt-category for b, m1, m2, pCavg."""
    df = master.copy()
    # Normalize prompt-category synonyms to canonical keys for plotting
    _NUMERIC_SYNS = {"pcnum", "numeric", "num", "single_numeric", "single_numeric_response"}
    _COT_SYNS = {"pccot", "cot", "chain_of_thought", "chain-of-thought", "cot_stepwise"}
    pc_series = df.get("prompt_category", pd.Series([], dtype=str)).astype(str).str.strip().str.lower()
    df["pc_norm"] = pc_series.map(lambda t: ("pcnum" if t in _NUMERIC_SYNS else ("pccot" if t in _COT_SYNS else t)))
    df = df[df["pc_norm"].isin(["pcnum", "pccot"])].copy()

    # pCavg per row
    if {"pC1", "pC2"}.issubset(df.columns):
        df["pCavg"] = (pd.to_numeric(df["pC1"], errors="coerce") + pd.to_numeric(df["pC2"], errors="coerce")) / 2.0
    else:
        df["pCavg"] = np.nan

    params = ["b", "m1", "m2", "pCavg"]
    exps = sorted(df["experiment"].dropna().unique().tolist(), reverse=True)

    out_dest = Path(out_dest)
    suffix = "normative" if normative else "non_normative"
    if out_dest.suffix.lower() == ".pdf":
        out_path = out_dest
        out_dir = out_dest.parent
    else:
        out_dir = out_dest
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"violin_params_{suffix}.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=False)

    # Normative mask: ensure NA are excluded in both subsets
    bool_series = _to_bool(df["normative_reasoner"]).astype("boolean")
    if normative:
        subset_mask = (bool_series == True).fillna(False)  # noqa: E712
    else:
        subset_mask = (bool_series == False).fillna(False)  # noqa: E712

    # Draw subplots
    for idx, p in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        _violin_for_param(ax, exps, df, p, subset_mask)
        ax.set_ylabel(p if p != "pCavg" else "p(C)")
        ax.set_title(f"CBN-parameter {p if p != 'pCavg' else 'p(C)'}")

    # Legend similar to other plots
    handles = [
        Patch(facecolor=numeric_color),
        Patch(facecolor=cot_color),
        Line2D([], [], color='yellow', marker='^', linestyle='None'),
        Line2D([], [], color='black', linestyle='-'),
    ]
    labels = ['Numeric', 'CoT', 'Mean', 'Median']
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_loocv_r2_by_experiment_and_pc(master: pd.DataFrame, out_dest: Path, normative: bool) -> None:
    """Single-axes grouped violins by experiment×prompt-category for LOOCV R².

    - Two violins per experiment: numeric vs CoT, colored via global palette.
    - Overlays mean (yellow triangle) and median (black line), consistent with other violins.
    - Y-axis fixed to [0, 1] for R².
    """
    df = master.copy()
    # Normalize prompt-category synonyms to canonical keys for plotting
    _NUMERIC_SYNS = {"pcnum", "numeric", "num", "single_numeric", "single_numeric_response"}
    _COT_SYNS = {"pccot", "cot", "chain_of_thought", "chain-of-thought", "cot_stepwise"}
    pc_series = df.get("prompt_category", pd.Series([], dtype=str)).astype(str).str.strip().str.lower()
    df["pc_norm"] = pc_series.map(lambda t: ("pcnum" if t in _NUMERIC_SYNS else ("pccot" if t in _COT_SYNS else t)))
    df = df[df["pc_norm"].isin(["pcnum", "pccot"])].copy()

    # Ensure R² column exists
    if "loocv_r2" not in df.columns:
        # Nothing to plot
        return

    exps = sorted(df["experiment"].dropna().unique().tolist(), reverse=True)

    out_dest = Path(out_dest)
    suffix = "normative" if normative else "non_normative"
    if out_dest.suffix.lower() == ".pdf":
        out_path = out_dest
        out_dir = out_dest.parent
    else:
        out_dir = out_dest
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"violin_loocv_r2_{suffix}.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normative mask: ensure NA are excluded in both subsets
    bool_series = _to_bool(df["normative_reasoner"]).astype("boolean")
    if normative:
        subset_mask = (bool_series == True).fillna(False)  # noqa: E712
    else:
        subset_mask = (bool_series == False).fillna(False)  # noqa: E712

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    # Reuse the shared drawer for consistency
    _violin_for_param(ax, exps, df, "loocv_r2", subset_mask)
    ax.set_ylabel(r"LOOCV $R^2$")
    ax.set_title("")

    # Legend consistent with other violins
    handles = [
        Patch(facecolor=numeric_color),
        Patch(facecolor=cot_color),
        Line2D([], [], color='yellow', marker='^', linestyle='None'),
        Line2D([], [], color='black', linestyle='-'),
    ]
    labels = ['Numeric', 'CoT', 'Mean', 'Median']
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate cognitive strategies classification outputs across experiments.")
    p.add_argument("--root", type=str, default=str(Path("results") / "parameter_analysis"), help="Root folder to crawl")
    p.add_argument("--out", type=str, default=str(Path("results") / "cross_cogn_strategies"), help="Output directory")
    p.add_argument("--plot", action="store_true", help="If set, generate violin plots for normative and non-normative subsets")
    args = p.parse_args(argv)

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_classified_csvs(root)
    if not files:
        print(f"No classified_strategy_metrics.csv files found under {root}")
        # Still write empty placeholders
        (out_dir / "masters_classified_strategy_metrics.csv").write_text("")
        (out_dir / "normative_share_by_exp_pc.csv").write_text("")
        (out_dir / "normative_share_by_exp_pc.tex").write_text("% Empty table\n")
        (out_dir / "normative_by_agent_matrix.csv").write_text("")
        return 0

    master = build_master(files)
    master_path = out_dir / "masters_classified_strategy_metrics.csv"
    master.to_csv(master_path, index=False)
    print(f"Wrote master: {master_path}")

    # Summary by experiment × prompt_category × thresholds
    summary = summarize_by_experiment_pc(master)
    summary_path = out_dir / "normative_share_by_exp_pc.csv"
    summary.to_csv(summary_path, index=False)
    tex_path = out_dir / "normative_share_by_exp_pc.tex"
    to_latex_table(summary, tex_path)
    print(f"Wrote summary CSV and LaTeX: {summary_path}, {tex_path}")

    # Agent matrix
    agent_mat = build_agent_matrix(master)
    agent_mat_path = out_dir / "normative_by_agent_matrix.csv"
    agent_mat.to_csv(agent_mat_path, index=False)
    print(f"Wrote agent matrix: {agent_mat_path}")
    # Also write LaTeX
    agent_mat_tex = out_dir / "normative_by_agent_matrix.tex"
    agent_matrix_to_latex(agent_mat, agent_mat_tex)
    print(f"Wrote agent matrix LaTeX: {agent_mat_tex}")

    if args.plot:
        plot_violin_params_by_experiment_and_pc(master, out_dir, normative=True)
        plot_violin_params_by_experiment_and_pc(master, out_dir, normative=False)
        # Also LOOCV R^2 violins for normative and non-normative
        plot_loocv_r2_by_experiment_and_pc(master, out_dir, normative=True)
        plot_loocv_r2_by_experiment_and_pc(master, out_dir, normative=False)
        print(f"Wrote violin plots in {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
