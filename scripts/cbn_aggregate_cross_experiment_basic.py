#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CBN cross-experiment aggregator:
- Sweeps results/parameter_analysis/*/*/normat_analysis/
- Builds a master table with experiment, prompt_category, tag, agent, metrics
- Runs matched within-agent paired tests:
    (A) experiment contrasts vs a baseline experiment (per prompt category)
    (B) prompt-category contrasts (pcnum vs pccot) within each experiment
- Fits mixed-effects models to handle unbalanced agent sets:
    * LOOCV R^2: MixedLM (Gaussian) with agent random intercept
      (fallback: OLS + agent fixed effects + cluster-robust SE)
    * AIC winner (3 vs 4): GLM Binomial + agent fixed effects + cluster-robust SE
- Writes CSVs + PDF plots + LaTeX tables of key comparisons.

Author: causAIign
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional statsmodels (mixed effects). If missing, we fallback.
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

# Matplotlib for plots (no seaborn, single-plot PDFs).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- Discovery & I/O -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[0]

def find_normat_dirs(root: Path, experiments: Optional[List[str]] = None) -> List[Path]:
    """
    Discover all normat_analysis directories under:
      root/<experiment>/<tag>/normat_analysis
    Returns list of Paths to normat_analysis folders.
    """
    found: List[Path] = []
    if not root.exists():
        return found
    for exp_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if experiments and exp_dir.name not in experiments:
            continue
        for tag_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            na = tag_dir / "normat_analysis"
            if na.exists() and na.is_dir():
                found.append(na)
    return found


def read_parameters_wide(normat_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load analysis_parameters_wide.csv (preferred).
    Fallback: winners_with_params.csv in tag dir, optionally merging params_tying from winners.csv.
    Returns DataFrame with required columns or None if not found.
    """
    pref = normat_dir / "analysis_parameters_wide.csv"
    if pref.exists():
        df = pd.read_csv(pref)
        return df

    # Fallback to winners in tag parent
    tag_dir = normat_dir.parent
    wwp = tag_dir / "winners_with_params.csv"
    win = tag_dir / "winners.csv"
    if wwp.exists():
        df = pd.read_csv(wwp)
        df["domain"] = df["domain"].where(df["domain"].notna(), "all")
        if ("params_tying" not in df.columns) and win.exists():
            w = pd.read_csv(win)
            w["domain"] = w["domain"].where(w["domain"].notna(), "all")
            meta = w[["agent", "domain", "params_tying"]].drop_duplicates(["agent", "domain"])
            df = df.merge(meta, on=["agent", "domain"], how="left")
        return df
    return None


def parse_prompt_category_from_tag(tag: str) -> Optional[str]:
    s = str(tag).lower()
    if "_pcnum_" in s or s.endswith("_pcnum") or s.startswith("pcnum_") or "_pcnum-" in s:
        return "pcnum"
    if "_pccot_" in s or s.endswith("_pccot") or s.startswith("pccot_") or "_pccot-" in s:
        return "pccot"
    return None


# ------------------------- Utilities -------------------------

def latex_escape(s: str) -> str:
    if s is None:
        return ""
    out = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def bh_fdr(pvals: Iterable[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg. Returns (reject_flags, adjusted_p)."""
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    denom = np.arange(1, n + 1)
    adj_vals = ranked * n / denom
    adj[::-1] = np.minimum.accumulate(adj_vals[::-1])
    adjusted = np.empty_like(adj)
    adjusted[order] = adj
    reject = adjusted <= alpha
    return reject, adjusted


def share_three_param(series: pd.Series) -> str:
    s = to_num(series).dropna()
    if s.empty:
        return "--"
    num = int((s == 3).sum())
    den = int(len(s))
    return f"{num}/{den}"


# ------------------------- Master table build -------------------------

REQUIRED = ["agent", "loocv_r2", "b", "m1", "m2", "pC1", "pC2", "params_tying"]

def build_master(root: Path, experiments: Optional[List[str]] = None) -> pd.DataFrame:
    """Sweep normat_analysis folders and assemble a master table."""
    rows: List[pd.DataFrame] = []
    for normat_dir in find_normat_dirs(root, experiments):
        tag_dir = normat_dir.parent
        exp = tag_dir.parent.name
        tag = tag_dir.name
        pc = parse_prompt_category_from_tag(tag)  # "pcnum" | "pccot" | None

        df = read_parameters_wide(normat_dir)
        if df is None or df.empty:
            continue

        # Coerce required columns
        for c in REQUIRED:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = to_num(df[c]) if c != "agent" else df[c]

        # Standardize fields we care about
        keep = ["agent", "domain", "loocv_r2", "b", "m1", "m2", "pC1", "pC2", "params_tying"]
        # include family if present for later subgroup views
        if "family" in df.columns:
            keep.append("family")
        sub = df[keep].copy()
        sub["experiment"] = exp
        sub["tag"] = tag
        sub["prompt_category"] = pc if pc else "unknown"
        sub["abs_mdiff"] = (sub["m1"] - sub["m2"]).abs()
        rows.append(sub)

    master = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=REQUIRED + ["experiment", "tag", "prompt_category"])
    # one row per (experiment, prompt_category, tag, agent); if there are multiple domains, prefer pooled/all if present
    if "domain" in master.columns:
        # Prefer pooled/all over domain-specific for the master; if multiple, keep the row with domain == 'all'
        master["domain_norm"] = master["domain"].astype(str).str.lower()
        master = master.sort_values(by=["domain_norm"], key=lambda s: (s != "all"))  # 'all' first
        master = master.drop_duplicates(["experiment", "prompt_category", "tag", "agent"], keep="first")
        master = master.drop(columns=["domain_norm"], errors="ignore")
    return master


# ------------------------- Paired comparisons -------------------------

def matched_agents(master: pd.DataFrame,
                   expA: str, expB: str,
                   pc: str) -> pd.DataFrame:
    """Return wide frame (one row per agent) for agents present in both expA and expB under the same prompt_category."""
    A = master[(master["experiment"] == expA) & (master["prompt_category"] == pc)].copy()
    B = master[(master["experiment"] == expB) & (master["prompt_category"] == pc)].copy()
    cols = ["loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_mdiff", "params_tying"]
    A = A.set_index("agent")[cols]
    B = B.set_index("agent")[cols]
    common = A.index.intersection(B.index)
    if len(common) == 0:
        return pd.DataFrame()
    W = A.loc[common].add_suffix(f"__{expA}").join(B.loc[common].add_suffix(f"__{expB}"), how="inner")
    W["agent"] = W.index
    W.reset_index(drop=True, inplace=True)
    W["prompt_category"] = pc
    W["expA"] = expA
    W["expB"] = expB
    return W


def matched_pc(master: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Return wide frame for agents present in both prompt categories within the same experiment."""
    A = master[(master["experiment"] == exp) & (master["prompt_category"] == "pcnum")].copy()
    B = master[(master["experiment"] == exp) & (master["prompt_category"] == "pccot")].copy()
    cols = ["loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_mdiff", "params_tying"]
    A = A.set_index("agent")[cols]
    B = B.set_index("agent")[cols]
    common = A.index.intersection(B.index)
    if len(common) == 0:
        return pd.DataFrame()
    W = A.loc[common].add_suffix("__pcnum").join(B.loc[common].add_suffix("__pccot"), how="inner")
    W["agent"] = W.index
    W.reset_index(drop=True, inplace=True)
    W["experiment"] = exp
    return W


def paired_wilcoxon(delta: pd.Series) -> Tuple[float, float]:
    """Return (statistic, pvalue) for Wilcoxon signed-rank; safe fallback if SciPy isn't present."""
    try:
        from scipy.stats import wilcoxon
        x = to_num(delta).dropna()
        if len(x) == 0:
            return (np.nan, np.nan)
        stat, p = wilcoxon(x)
        return float(stat), float(p)
    except Exception:
        # Fallback: sign test (very conservative)
        x = to_num(delta).dropna()
        if len(x) == 0:
            return (np.nan, np.nan)
        n_pos = int((x > 0).sum())
        n_neg = int((x < 0).sum())
        n = n_pos + n_neg
        if n == 0:
            return (0.0, 1.0)
        # two-sided binomial exact under p=0.5
        from math import comb
        p_tail = sum(comb(n, k) for k in range(0, min(n_pos, n_neg) + 1)) / (2 ** n)
        return (float(n_pos - n_neg), float(2 * p_tail))


def paired_summary_table(W: pd.DataFrame,
                         lhs_suffix: str,
                         rhs_suffix: str,
                         label_cols: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (deltas_per_agent, test_summary) dataframes for a matched wide table W.
    label_cols: e.g., {"groupA": "expA", "groupB": "expB"} to carry metadata in outputs
    """
    metrics = ["loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_mdiff"]
    deltas: Dict[str, pd.Series] = {}
    for m in metrics:
        a = to_num(W[f"{m}{lhs_suffix}"])
        b = to_num(W[f"{m}{rhs_suffix}"])
        deltas[f"delta_{m}"] = b - a

    # Include 3-param winner delta (binary to {-1,0,1})
    if f"params_tying{lhs_suffix}" in W.columns and f"params_tying{rhs_suffix}" in W.columns:
        lhs3 = (to_num(W[f"params_tying{lhs_suffix}"]) == 3).astype(float)
        rhs3 = (to_num(W[f"params_tying{rhs_suffix}"]) == 3).astype(float)
        deltas["delta_three_param_share"] = rhs3 - lhs3

    D = pd.DataFrame({"agent": W["agent"], **deltas})
    # Add identifying cols
    for out_name, src_col in label_cols.items():
        D[out_name] = W[src_col]

    # Wilcoxon + effect sizes (Hodges–Lehmann)
    rows = []
    for k, s in deltas.items():
        x = to_num(s).dropna()
        if x.empty:
            rows.append({"metric": k, "N": 0, "median_delta": np.nan, "wilcoxon_stat": np.nan,
                         "p_value": np.nan})
            continue
        stat, p = paired_wilcoxon(x)
        # Hodges–Lehmann (median of all pairwise averages for signed deltas == just median)
        med = float(np.median(x))
        rows.append({"metric": k, "N": int(len(x)), "median_delta": med,
                     "wilcoxon_stat": stat, "p_value": p})
    T = pd.DataFrame(rows)
    # BH-FDR adjust within this comparison
    rej, adj = bh_fdr(T["p_value"].fillna(1.0).values)
    T["p_fdr"] = adj
    T["reject_fdr"] = rej
    # Carry labels on the summary too
    for out_name, src_col in label_cols.items():
        T[out_name] = W[src_col].iloc[0] if len(W) else None
    return D, T


# ------------------------- Mixed-effects models -------------------------

def fit_mixed_R2(master: pd.DataFrame,
                 experiments: List[str],
                 include_unknown_pc: bool = False) -> pd.DataFrame:
    """
    Fit LOOCV R^2 ~ experiment + prompt_category + experiment:prompt_category + (1|agent)
    Returns coefficients table. Fallback to OLS + agent FE + cluster-robust SE if MixedLM unavailable.
    """
    df = master.copy()
    if not include_unknown_pc:
        df = df[df["prompt_category"].isin(["pcnum", "pccot"])]

    # Make factors
    df["experiment"] = df["experiment"].astype("category")
    df["prompt_category"] = df["prompt_category"].astype("category")
    df = df.dropna(subset=["loocv_r2"])

    if HAS_SM:
        # Mixed effects: Gaussian
        # Use treatment coding with the first level as baseline (sorted order)
        formula = "loocv_r2 ~ C(experiment) + C(prompt_category) + C(experiment):C(prompt_category)"
        try:
            md = smf.mixedlm(formula, data=df, groups=df["agent"])
            m = md.fit(reml=False, method="lbfgs")
            coefs = m.summary().tables[1]  # params
            tab = coefs.reset_index()
            tab.columns = ["term", "coef", "std_err", "z", "p_value", "[0.025", "0.975]"]
            tab["model"] = "MixedLM"
            return tab
        except Exception as e:
            pass

    # Fallback: OLS + agent fixed effects + cluster-robust SE (cluster by agent)
    # Build design with agent FE
    formula = "loocv_r2 ~ C(experiment) + C(prompt_category) + C(experiment):C(prompt_category) + C(agent)"
    ols = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["agent"]}) if HAS_SM else None
    if ols is not None:
        summ = ols.summary2().tables[1].reset_index()
        summ = summ.rename(columns={"index": "term", "Coef.": "coef", "Std.Err.": "std_err", "P>|t|": "p_value"})
        summ["model"] = "OLS_FE_cluster(agent)"
        # Filter out the many agent dummies when reporting
        return summ[~summ["term"].str.startswith("C(agent)")].reset_index(drop=True)
    # last-resort: empty frame
    return pd.DataFrame(columns=["term", "coef", "std_err", "p_value", "model"])


def fit_glm_three_param(master: pd.DataFrame) -> pd.DataFrame:
    """
    GLM Binomial on 3-param winner:
      I(params_tying==3) ~ C(experiment) + C(prompt_category) + C(experiment):C(prompt_category) + C(agent)
    with cluster-robust SE by agent (requires statsmodels).
    """
    if not HAS_SM:
        return pd.DataFrame(columns=["term", "coef", "std_err", "p_value", "model"])

    df = master.copy()
    df = df.dropna(subset=["params_tying"])
    df["y"] = (to_num(df["params_tying"]) == 3).astype(int)
    df["experiment"] = df["experiment"].astype("category")
    df["prompt_category"] = df["prompt_category"].astype("category")

    formula = "y ~ C(experiment) + C(prompt_category) + C(experiment):C(prompt_category) + C(agent)"
    glm = smf.glm(formula, data=df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": df["agent"]})
    summ = glm.summary2().tables[1].reset_index()
    summ = summ.rename(columns={"index": "term", "Coef.": "coef", "Std.Err.": "std_err", "P>|z|": "p_value"})
    summ["model"] = "GLM_Binomial_FE_cluster(agent)"
    return summ[~summ["term"].str.startswith("C(agent)")].reset_index(drop=True)


# ------------------------- Plots -------------------------

def plot_box_R2_by_exp_pc(master: pd.DataFrame, out_path: Path) -> None:
    """Simple boxplot of LOOCV R^2 per experiment × prompt_category."""
    df = master.copy()
    df = df[df["prompt_category"].isin(["pcnum", "pccot"])]
    # sort x by experiment then pc
    groups = sorted(df.groupby(["experiment", "prompt_category"]).groups.keys())
    labels = [f"{e}\n{pc}" for e, pc in groups]
    data = [to_num(df[(df["experiment"] == e) & (df["prompt_category"] == pc)]["loocv_r2"]).dropna().values for e, pc in groups]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("LOOCV $R^2$")
    ax.set_title("Normativity by experiment × prompt-category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_slope_R2_pair(W: pd.DataFrame, colA: str, colB: str, title: str, out_path: Path) -> None:
    """Slope chart per agent for matched pair comparison."""
    x = [0, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for _, r in W.iterrows():
        y = [float(r[colA]), float(r[colB])]
        if any(np.isnan(y)):
            continue
        ax.plot(x, y, marker="o")
        # optionally annotate agent names lightly:
        # ax.text(1.01, y[1], str(r["agent"]), fontsize=6)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A", "B"])
    ax.set_ylabel("LOOCV $R^2$")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------- LaTeX tables -------------------------

# def tex_per_condition_medians(master: pd.DataFrame, out_path: Path) -> None:
#     """LaTeX table with per (experiment, prompt_category) medians and 3-par share."""
#     df = master.copy()
#     def med(s): return float(np.median(to_num(s).dropna())) if s is not None else np.nan
#     rows = []
#     for (exp, pc), sub in df.groupby(["experiment", "prompt_category"], dropna=False):
#         rows.append({
#             "experiment": exp, "prompt_category": pc, "N": int(sub["agent"].nunique()),
#             "R2": med(sub["loocv_r2"]), "b": med(sub["b"]), "m1": med(sub["m1"]), "m2": med(sub["m2"]),
#             "pC": med((to_num(sub["pC1"]) + to_num(sub["pC2"])) / 2.0),
#             "three_par": share_three_param(sub["params_tying"]),
#         })
#     T = pd.DataFrame(rows).sort_values(["experiment", "prompt_category"])
#     # Build LaTeX
#     lines = []
#     lines.append("% Auto-generated; do not edit by hand\n")
#     lines.append("\\begin{table}[t]\n\\centering\n")
#     lines.append("\\caption{Per-condition medians and 3-par share.}\n")
#     lines.append("\\label{tab:cbn_per_condition_medians}\n")
#     lines.append("\\begin{tabular}{l l r r r r r r}\n\\toprule\n")
#     lines.append("Experiment & Prompt & $N$ & $R^2$ & $b$ & $m_1$ & $m_2$ & $p(C)$ & 3-par \\\\\n\\midrule\n")
#     for _, r in T.iterrows():
#         lines.append(f"{latex_escape(r['experiment'])} & {latex_escape(r['prompt_category'])} & {int(r['N'])} & "
#                      f"{r['R2']:.3f} & {r['b']:.3f} & {r['m1']:.3f} & {r['m2']:.3f} & {r['pC']:.3f} & {r['three_par']} \\\\\n")
#     lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
#     out_path.write_text("".join(lines))

def tex_per_condition_medians(master: pd.DataFrame, out_path: Path) -> None:
    df = master.copy()
    def med(s): return float(np.median(to_num(s).dropna())) if s is not None else np.nan
    rows = []
    for (exp, pc), sub in df.groupby(["experiment", "prompt_category"], dropna=False):
        rows.append({
            "experiment": exp, "prompt_category": pc, "N": int(sub["agent"].nunique()),
            "R2": med(sub["loocv_r2"]), "b": med(sub["b"]), "m1": med(sub["m1"]), "m2": med(sub["m2"]),
            "pC": med((to_num(sub["pC1"]) + to_num(sub["pC2"])) / 2.0),
            "three_par": share_three_param(sub["params_tying"]),
        })
    T = pd.DataFrame(rows).sort_values(["experiment", "prompt_category"])
    lines = []
    lines.append("% Auto-generated; do not edit by hand\n")
    lines.append("\\begin{table}[t]\n\\centering\n")
    lines.append("\\caption{Per-condition medians and 3-par share.}\n")
    lines.append("\\label{tab:cbn_per_condition_medians}\n")
    # 9 columns here:
    lines.append("\\begin{tabular}{l l r r r r r r r}\n\\toprule\n")
    lines.append("Experiment & Prompt & $N$ & $R^2$ & $b$ & $m_1$ & $m_2$ & $p(C)$ & 3-par \\\\\n\\midrule\n")
    for _, r in T.iterrows():
        lines.append(f"{latex_escape(r['experiment'])} & {latex_escape(r['prompt_category'])} & {int(r['N'])} & "
                     f"{r['R2']:.3f} & {r['b']:.3f} & {r['m1']:.3f} & {r['m2']:.3f} & {r['pC']:.3f} & {r['three_par']} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))

def tex_paired_summary(T: pd.DataFrame, caption: str, label: str, out_path: Path) -> None:
    """LaTeX table for paired test summary (already BH-FDR adjusted)."""
    # Keep core stats, format cleanly
    disp = T.copy()
    # Order by metric for stable layout
    disp = disp.sort_values("metric")
    lines = []
    lines.append("% Auto-generated; do not edit by hand\n")
    lines.append("\\begin{table}[t]\n\\centering\n")
    lines.append(f"\\caption{{{caption}}}\n")
    lines.append(f"\\label{{{label}}}\n")
    lines.append("\\begin{tabular}{l r r r r}\n\\toprule\n")
    lines.append("Metric & $N$ & Median $\\Delta$ & Wilcoxon & $p_\\mathrm{FDR}$ \\\\\n\\midrule\n")
    for _, r in disp.iterrows():
        med = r["median_delta"]
        w = r["wilcoxon_stat"]
        pfdr = r["p_fdr"]
        lines.append(f"{latex_escape(r['metric'])} & {int(r['N'])} & {med if pd.isna(med) else f'{med:.3f}'} & "
                     f"{w if pd.isna(w) else f'{w:.3f}'} & {pfdr if pd.isna(pfdr) else f'{pfdr:.3g}'} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))


# def tex_model_coefs(tab: pd.DataFrame, caption: str, label: str, out_path: Path) -> None:
#     """LaTeX table for model coefficients (R^2 mixed model or GLM)."""
#     if tab is None or tab.empty:
#         out_path.write_text("% (no model coefficients to report)\n")
#         return
#     keep = tab.copy()
#     # Standardize column names if present
#     for cand in ["coef", "std_err", "p_value"]:
#         if cand not in keep.columns:
#             keep[cand] = np.nan
#     keep = keep[["term", "coef", "std_err", "p_value"]].copy()
#     lines = []
#     lines.append("% Auto-generated; do not edit by hand\n")
#     lines.append("\\begin{table}[t]\n\\centering\n")
#     lines.append(f"\\caption{{{caption}}}\n")
#     lines.append(f"\\label{{{label}}}\n")
#     lines.append("\\begin{tabular}{l r r r}\n\\toprule\n")
#     lines.append("Term & Coef & SE & $p$ \\\\\n\\midrule\n")
#     for _, r in keep.iterrows():
#         lines.append(f"{latex_escape(r['term'])} & {r['coef']:.3f} & {r['std_err']:.3f} & {r['p_value']:.3g} \\\\\n")
#     lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
#     out_path.write_text("".join(lines))

def tex_model_coefs(tab: pd.DataFrame, caption: str, label: str, out_path: Path) -> None:
    """LaTeX table for model coefficients (R^2 mixed model or GLM). Robust to strings/NaNs."""
    if tab is None or tab.empty:
        out_path.write_text("% (no model coefficients to report)\n")
        return

    keep = tab.copy()

    # Normalize column names if they vary by model type
    rename_map = {
        "Coef.": "coef", "coef": "coef",
        "Std.Err.": "std_err", "std_err": "std_err",
        "P>|t|": "p_value", "P>|z|": "p_value", "p_value": "p_value",
        "index": "term"
    }
    for k, v in rename_map.items():
        if k in keep.columns and v not in keep.columns:
            keep = keep.rename(columns={k: v})

    # Ensure required columns exist
    for col in ["term", "coef", "std_err", "p_value"]:
        if col not in keep.columns:
            keep[col] = np.nan

    # Coerce numerics safely
    keep["coef"] = pd.to_numeric(keep["coef"], errors="coerce")
    keep["std_err"] = pd.to_numeric(keep["std_err"], errors="coerce")
    keep["p_value"] = pd.to_numeric(keep["p_value"], errors="coerce")

    # Only keep display columns
    keep = keep[["term", "coef", "std_err", "p_value"]].copy()

    def fmtf(x, dec=3, empty="--"):
        try:
            x = float(x)
            if not np.isfinite(x):
                return empty
            return f"{x:.{dec}f}"
        except Exception:
            return empty

    def fmtp(x, empty="--"):
        try:
            x = float(x)
            if not np.isfinite(x):
                return empty
            # compact scientific if very small
            return f"{x:.3g}"
        except Exception:
            return empty

    lines = []
    lines.append("% Auto-generated; do not edit by hand\n")
    lines.append("\\begin{table}[t]\n\\centering\n")
    lines.append(f"\\caption{{{caption}}}\n")
    lines.append(f"\\label{{{label}}}\n")
    lines.append("\\begin{tabular}{l r r r}\n\\toprule\n")
    lines.append("Term & Coef & SE & $p$ \\\\\n\\midrule\n")
    for _, r in keep.iterrows():
        term = latex_escape(r["term"])
        lines.append(f"{term} & {fmtf(r['coef'])} & {fmtf(r['std_err'])} & {fmtp(r['p_value'])} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))


# ------------------------- Main -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate cross-experiment CBN analyses; export CSVs, PDFs, LaTeX.")
    ap.add_argument("--root", default=str(PROJECT_ROOT / "results" / "parameter_analysis"),
                    help="Root containing <experiment>/<tag>/normat_analysis/")
    ap.add_argument("--experiments", nargs="*", help="Optional list of experiments to include; default: discover all.")
    ap.add_argument("--baseline", default="rw17_indep_causes",
                    help="Baseline experiment for experiment-pair comparisons (default: rw17_indep_causes).")
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "../" / "results" / "parameter_analysis" / "cbn_agg"),
                    help="Output root for aggregated artifacts.")
    ap.add_argument("--export-tex", action="store_true", help="Write LaTeX tables for key summaries.")
    ap.add_argument("--plots", action="store_true", help="Write PDF plots for summaries and paired comparisons.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Build master table
    master = build_master(root, experiments=args.experiments)
    if master.empty:
        print("[WARN] No data discovered. Check --root.")
        return 0

    master_path = out_root / "master_table.csv"
    master.to_csv(master_path, index=False)
    print(f"Saved master table: {master_path}")

    # 2) Per-condition medians & share of 3-param winners (CSV + optional LaTeX)
    per_cond_rows = []
    for (exp, pc), sub in master.groupby(["experiment", "prompt_category"], dropna=False):
        def med(s): return float(np.median(to_num(s).dropna())) if not to_num(s).dropna().empty else np.nan
        per_cond_rows.append({
            "experiment": exp, "prompt_category": pc,
            "N_agents": int(sub["agent"].nunique()),
            "median_R2": med(sub["loocv_r2"]),
            "median_b": med(sub["b"]),
            "median_m1": med(sub["m1"]),
            "median_m2": med(sub["m2"]),
            "median_pC": med((to_num(sub["pC1"]) + to_num(sub["pC2"])) / 2.0),
            "share_3param": share_three_param(sub["params_tying"]),
        })
    per_cond = pd.DataFrame(per_cond_rows).sort_values(["experiment", "prompt_category"])
    per_cond_path = out_root / "per_condition_medians.csv"
    per_cond.to_csv(per_cond_path, index=False)
    print(f"Saved per-condition medians: {per_cond_path}")
    if args.export_tex:
        tex_per_condition_medians(master, out_root / "per_condition_medians.tex")
        print(f"Saved LaTeX: {out_root / 'per_condition_medians.tex'}")

    # 3) Experiment-pair comparisons vs baseline (within prompt_category)
    experiments = sorted(master["experiment"].unique().tolist())
    if args.baseline not in experiments:
        print(f"[WARN] baseline '{args.baseline}' not found among experiments: {experiments}")
    else:
        others = [e for e in experiments if e != args.baseline]
        pcs = ["pcnum", "pccot"]
        for pc in pcs:
            for exp in others:
                W = matched_agents(master, args.baseline, exp, pc)
                if W.empty:
                    continue
                # deltas and summary
                deltas, summary = paired_summary_table(
                    W,
                    lhs_suffix=f"__{args.baseline}",
                    rhs_suffix=f"__{exp}",
                    label_cols={"baseline": "expA", "other": "expB", "prompt_category": "prompt_category"}
                )
                pair_dir = out_root / "experiment_pairs" / f"{args.baseline}__vs__{exp}" / pc
                pair_dir.mkdir(parents=True, exist_ok=True)
                deltas.to_csv(pair_dir / "paired_deltas.csv", index=False)
                summary.to_csv(pair_dir / "paired_summary.csv", index=False)
                print(f"Saved experiment-pair CSVs: {pair_dir}")

                # Plots
                if args.plots:
                    title = f"LOOCV $R^2$ per agent: {args.baseline} (A) → {exp} (B) [{pc}]"
                    plot_slope_R2_pair(
                        W,
                        colA=f"loocv_r2__{args.baseline}",
                        colB=f"loocv_r2__{exp}",
                        title=title,
                        out_path=pair_dir / "slope_R2.pdf"
                    )

                # LaTeX
                if args.export_tex:
                    tex_paired_summary(
                        summary,
                        caption=f"Paired deltas ({pc}) for {args.baseline}→{exp} (matched agents).",
                        label=f"tab:paired_{pc}_{args.baseline}_to_{exp}",
                        out_path=pair_dir / "paired_summary.tex"
                    )

    # 4) Prompt-category comparisons within each experiment
    for exp in experiments:
        W = matched_pc(master, exp)
        if W.empty:
            continue
        deltas, summary = paired_summary_table(
            W,
            lhs_suffix="__pcnum",
            rhs_suffix="__pccot",
            label_cols={"experiment": "experiment"}
        )
        pc_dir = out_root / "prompt_category_pairs" / exp
        pc_dir.mkdir(parents=True, exist_ok=True)
        deltas.to_csv(pc_dir / "paired_deltas_pcnum_to_pccot.csv", index=False)
        summary.to_csv(pc_dir / "paired_summary_pcnum_to_pccot.csv", index=False)
        print(f"Saved pcnum vs pccot CSVs: {pc_dir}")

        if args.plots:
            title = f"LOOCV $R^2$ per agent: pcnum (A) → pccot (B) [{exp}]"
            plot_slope_R2_pair(
                W,
                colA="loocv_r2__pcnum",
                colB="loocv_r2__pccot",
                title=title,
                out_path=pc_dir / "slope_R2_pcnum_to_pccot.pdf"
            )

        if args.export_tex:
            tex_paired_summary(
                summary,
                caption=f"Paired deltas within {exp} (pcnum→pccot; matched agents).",
                label=f"tab:paired_pc_{exp}",
                out_path=pc_dir / "paired_summary_pcnum_to_pccot.tex"
            )

    # 5) Mixed-effects models on the full (unbalanced) set
    #    5a) LOOCV R^2
    coef_R2 = fit_mixed_R2(master, experiments)
    coef_R2_path = out_root / "mixed_effects_R2.csv"
    coef_R2.to_csv(coef_R2_path, index=False)
    print(f"Saved mixed-effects (R^2) coefficients: {coef_R2_path}")
    if args.export_tex:
        tex_model_coefs(
            coef_R2,
            caption="Mixed-effects coefficients for LOOCV $R^2$ (Gaussian MixedLM or OLS-FE fallback).",
            label="tab:mixed_R2",
            out_path=out_root / "mixed_effects_R2.tex"
        )
        print(f"Saved LaTeX: {out_root / 'mixed_effects_R2.tex'}")

    #    5b) AIC winner: 3-parameter vs 4-parameter
    coef_3par = fit_glm_three_param(master)
    coef_3par_path = out_root / "glm_three_param_share.csv"
    coef_3par.to_csv(coef_3par_path, index=False)
    print(f"Saved GLM (3-param share) coefficients: {coef_3par_path}")
    if args.export_tex:
        tex_model_coefs(
            coef_3par,
            caption="GLM (Binomial) coefficients for 3-parameter winner (agent FE; cluster-robust SE by agent).",
            label="tab:glm_three_param",
            out_path=out_root / "glm_three_param_share.tex"
        )
        print(f"Saved LaTeX: {out_root / 'glm_three_param_share.tex'}")

    # 6) High-level plots (R^2 distributions)
    if args.plots:
        plot_box_R2_by_exp_pc(master, out_root / "box_R2_by_experiment_and_pc.pdf")
        print(f"Saved plot: {out_root / 'box_R2_by_experiment_and_pc.pdf'}")

    print("\nDone. Artifacts under:", out_root.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
