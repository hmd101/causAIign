#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Generate LaTeX table of normativity by agent across experiments and prompts.

Per experiment, there are two prompt groups (Num, CoT). Under each prompt, four
subcolumns: EA, MV, R2 (soft marks) and ALL (overall, regular mark).
The tabular alignment is:  \begin{tabular}{l *{32}{c} c c}
Wrapped in a table* environment.

Thresholds default to --th_ea 0.3, --th_mv 0.05, --th_r2 0.89. If the CSV
contains meets_ea/meets_mv/meets_loocv_r or normative_reasoner, they are used;
otherwise they're computed from EA_raw/MV_raw/loocv_r2.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# Order and pretty labels
EXPERIMENT_ORDER: List[str] = [
    "rw17_indep_causes",            # RW17
    "random_abstract",              # Abstract
    "rw17_overloaded_e",            # RW17-Over
    "abstract_overloaded_lorem_de", # Abstract-Over
]
EXPERIMENT_LABEL: Dict[str, str] = {
    "rw17_indep_causes": "RW17",
    "random_abstract": "Abstract",
    "rw17_overloaded_e": "RW17-Over",
    "abstract_overloaded_lorem_de": "Abstract-Over",
}
PROMPT_ORDER: List[str] = ["pcnum", "pccot"]
PROMPT_LABEL: Dict[str, str] = {"pcnum": "Num", "pccot": "CoT"}

# CSV column candidates
EA_CANDS  = ["EA_raw", "ea_raw", "EA", "ea"]
MV_CANDS  = ["MV_raw", "mv_raw", "MV", "mv", "Delta_MV_raw", "delta_mv_raw"]
R2_CANDS  = ["loocv_r2", "LOOCV_R2", "loocv_R2", "r2"]
NORM_CANDS= ["normative_reasoner", "normative", "is_normative"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate normativity-by-agent LaTeX table with EA/MV/R2/ALL subcolumns.")
    p.add_argument("--csv", default="results/cross_cogn_strategies/masters_classified_strategy_metrics.csv")
    p.add_argument("--th_ea", type=float, default=0.3)
    p.add_argument("--th_mv", type=float, default=0.05)
    p.add_argument("--th_r2", type=float, default=0.89)
    p.add_argument("--output", default="results/plots/tables/normativity_by_agent.tex")
    return p.parse_args()


def _first_present(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _latex_escape(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("_", r"\_")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ["agent", "experiment", "prompt_category"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    # Canonicalize prompt category values
    if "prompt_category" in df.columns:
        canon = {
            "numeric": "pcnum",
            "num": "pcnum",
            "single_numeric_response": "pcnum",
            "pcnum": "pcnum",
            "cot": "pccot",
            "pccot": "pccot",
            "cot_reasoning": "pccot",
            "CoT": "pccot",
        }
        df["prompt_category"] = df["prompt_category"].apply(lambda x: canon.get(x, canon.get(str(x).lower(), str(x).lower())))
    return df


def ensure_flags(df: pd.DataFrame, th_ea: float, th_mv: float, th_r2: float) -> pd.DataFrame:
    out = df.copy()
    # Overall flag if present
    norm_col = _first_present(out, NORM_CANDS)
    if norm_col is not None:
        raw = out[norm_col]
        nums = pd.to_numeric(raw, errors="coerce")
        bool_from_nums = nums.notna() & (nums > 0)
        strmap = raw.astype(str).str.strip().str.lower().map({
            "true": True, "t": True, "yes": True, "y": True, "1": True,
            "false": False, "f": False, "no": False, "n": False, "0": False,
        })
        out["normative_reasoner"] = np.where(nums.notna(), bool_from_nums, strmap).astype(bool)
    # Component flags
    if {"meets_ea", "meets_mv", "meets_loocv_r"}.issubset(out.columns) or {"meets_ea", "meets_mv", "meets_loocv_r2"}.issubset(out.columns):
        out["meets_ea"] = out["meets_ea"].astype(bool)
        out["meets_mv"] = out["meets_mv"].astype(bool)
        if "meets_loocv_r" not in out.columns and "meets_loocv_r2" in out.columns:
            out = out.rename(columns={"meets_loocv_r2": "meets_loocv_r"})
        out["meets_loocv_r"] = out["meets_loocv_r"].astype(bool)
    else:
        cea = _first_present(out, EA_CANDS)
        cmv = _first_present(out, MV_CANDS)
        cr2 = _first_present(out, R2_CANDS)
        if cea is None or cmv is None or cr2 is None:
            if "normative_reasoner" in out.columns:
                out["meets_ea"], out["meets_mv"], out["meets_loocv_r"] = np.nan, np.nan, np.nan
                return out
            raise ValueError("CSV must contain 'normative_reasoner' or raw metrics to compute it.")
        out["meets_ea"] = pd.to_numeric(out[cea], errors="coerce") > th_ea
        out["meets_mv"] = pd.to_numeric(out[cmv], errors="coerce").abs() < th_mv
        out["meets_loocv_r"] = pd.to_numeric(out[cr2], errors="coerce") > th_r2
    if "normative_reasoner" not in out.columns:
        out["normative_reasoner"] = (out["meets_ea"] & out["meets_mv"] & out["meets_loocv_r"]).astype(bool)
    return out


def mark_soft(v) -> str:
    if pd.isna(v):
        return "--"
    return "\\cmarksoft" if bool(v) else "\\xmarksoft"


def mark_all(v) -> str:
    if pd.isna(v):
        return "--"
    return "\\cmark" if bool(v) else "\\xmark"


def build_table(df: pd.DataFrame) -> str:
    # Filter to relevant rows
    df = df[df["experiment"].isin(EXPERIMENT_ORDER) & df["prompt_category"].isin(PROMPT_ORDER)].copy()

    # Build wide matrix per agent
    records: List[dict] = []
    for agent, sub in df.groupby("agent", sort=False):
        rec: dict = {"agent": agent}
        for exp in EXPERIMENT_ORDER:
            for pc in PROMPT_ORDER:
                mask = (sub["experiment"] == exp) & (sub["prompt_category"] == pc)
                if mask.any():
                    row = sub.loc[mask].iloc[0]
                    rec[(exp, pc, "ea")] = row.get("meets_ea", np.nan)
                    rec[(exp, pc, "mv")] = row.get("meets_mv", np.nan)
                    # allow either meets_loocv_r or meets_loocv_r2
                    r2v = row.get("meets_loocv_r", np.nan)
                    if pd.isna(r2v) and "meets_loocv_r2" in row.index:
                        r2v = row.get("meets_loocv_r2", np.nan)
                    rec[(exp, pc, "r2")] = r2v
                    rec[(exp, pc, "all")] = row.get("normative_reasoner", np.nan)
                else:
                    rec[(exp, pc, "ea")] = np.nan
                    rec[(exp, pc, "mv")] = np.nan
                    rec[(exp, pc, "r2")] = np.nan
                    rec[(exp, pc, "all")] = np.nan
        records.append(rec)
    wide = pd.DataFrame.from_records(records)

    # Summary over ALL
    all_keys = [(e, p, "all") for e in EXPERIMENT_ORDER for p in PROMPT_ORDER]
    if wide.empty:
        raise ValueError("No rows after filtering by experiment and prompt_category.")
    wide["count_present"] = wide[all_keys].notna().sum(axis=1)
    wide["count_norm"] = wide[all_keys].apply(lambda r: float(np.nansum(pd.to_numeric(r.values, errors="coerce"))), axis=1)
    wide["percent"] = np.where(wide["count_present"] > 0, 100.0 * wide["count_norm"] / wide["count_present"], 0.0)

    # LaTeX assembly
    L: List[str] = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\scriptsize")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append(
        "\\caption{Normativity evaluation per agent across experiments and prompts (\\cmark{}=passes rule, \\xmark{}=fails; -- = not run). "
        "An agent is considered normative when explaining-away (EA) is strong, there is no Markov violation (MV), and LOOCV $R^2$ is high.}"
    )
    L.append("\\label{tab:normative-by-agent-matrix}")
    L.append("\\begin{tabular}{l *{32}{c} c c}")
    L.append("\\toprule")

    # Header 1: experiments
    h1 = ["\\textbf{Agent}"]
    for exp in EXPERIMENT_ORDER:
        h1.append(f"\\multicolumn{{8}}{{c}}{{\\textbf{{{EXPERIMENT_LABEL[exp]}}}}}")
    h1.extend(["\\textbf{Fraction Normative}", "\\textbf{Percent}"])
    L.append(" & ".join(h1) + " \\")

    # cmidrules per experiment block (8 columns each) after Agent
    cmis: List[str] = []
    start_col = 2
    for _ in EXPERIMENT_ORDER:
        end_col = start_col + 7
        cmis.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    L.append("".join(cmis))

    # Header 2: prompts under each experiment
    h2 = [""]
    for _ in EXPERIMENT_ORDER:
        h2.append(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{PROMPT_LABEL['pcnum']}}}}}")
        h2.append(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{PROMPT_LABEL['pccot']}}}}}")
    h2.extend(["", ""])  # placeholders for summary columns
    L.append(" & ".join(h2) + " \\")

    # Header 3: EA MV R2 ALL repeats
    h3 = [""]
    for _ in EXPERIMENT_ORDER:
        h3.extend(["EA", "MV", "R2", "ALL", "EA", "MV", "R2", "ALL"])
    h3.extend(["", ""])  # placeholders
    L.append(" & ".join(h3) + " \\")
    L.append("\\midrule")

    # Body rows
    for _, row in wide.sort_values("agent").iterrows():
        cells: List[str] = [_latex_escape(row["agent"])]
        for exp in EXPERIMENT_ORDER:
            for pc in PROMPT_ORDER:
                cells.append(mark_soft(row[(exp, pc, "ea")]))
                cells.append(mark_soft(row[(exp, pc, "mv")]))
                cells.append(mark_soft(row[(exp, pc, "r2")]))
                cells.append(mark_all(row[(exp, pc, "all")]))
        cells.append(f"{int(row['count_norm'])}/{int(row['count_present'])}")
        cells.append(f"{row['percent']:.1f}\\%")
        L.append(" & ".join(cells) + " \\")

    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    return "\n".join(L) + "\n"


def main() -> int:
    args = parse_args()
    df = load_data(Path(args.csv))
    df = ensure_flags(df, args.th_ea, args.th_mv, args.th_r2)
    tex = build_table(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX table to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
