#!/usr/bin/env python3
"""
Cognitive strategies analysis: compute EA and Markov-violation indices per agent
from raw responses and from the fitted CBN predictions, and emit robustness tables.

Outputs are written under:
  results/parameter_analysis/<experiment>/<tag>/cogn_analysis/

Key metrics (per agent x prompt_category):
- EA_raw, EA_model: Explaining-Away index = p(C1=1|E=1,C2=0) - p(C1=1|E=1,C2=1)
- MV_raw, MV_model: Markov-violation index = p(C1=1|C2=1) - p(C1=1|C2=0)
- |m1-m2|, m_bar, b, pC1, pC2 (from winners_with_params)
- loocv_r2 (if available from winners.csv)

Bootstrap extensions (added):
- For each (agent × prompt_category), compute EA and MV per-domain and derive
    domain-bootstrap percentile CIs on the mean (default B=2000, 95%). We write:
    EA_raw_lo/hi, EA_raw_median, EA_raw_mean and MV_raw_lo/hi, MV_raw_median, MV_raw_mean
    into the strategy_metrics.csv. These are attached on the pooled row for that
    agent × prompt_category (this script only writes one row per agent×PC).

CBN-derived normative metric (added):
- normative_metric = (m1 + m2)/2 − b, computed from canonical CBN params in
    winners_with_params.csv and included in the output.

Robustness (pcnum → pccot within-experiment):
- For agents present in both prompt categories, write a CSV with deltas of
  loocv_r2, b, m1, m2, pC1, pC2 and a boolean 'normative' using thresholds.

Notes
- Raw probabilities are computed as per-task mean responses for the relevant
  tasks (letters a,c,d,e) using Roman mapping in tasks.py.
- Model probabilities use roman_task_to_probability with the winners params.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# Ensure project src/ is on sys.path before any project imports
_sys.path.append(_os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "src"))

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from tueplots import bundles, fonts

from causalign.analysis.model_fitting.data import (  # type: ignore
    load_processed_data,
    prepare_dataset,
)
from causalign.analysis.model_fitting.tasks import (  # type: ignore
    LETTER_TO_ROMAN,
    ROMAN_TO_LETTER,
    roman_task_to_probability,
)
from causalign.config.paths import PathManager  # type: ignore

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




# After imports, define project root Path if needed elsewhere
PROJECT_ROOT = Path(__file__).parent.parent


ROMAN_ORDER: List[str] = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"
]

# Indices defined in terms of task letters (see tasks.py):
# - EA (Explaining Away): p(C1=1 | E=1, C2=0) - p(C1=1 | E=1, C2=1) = Task 'c' - Task 'a'.
#   Roman tasks used in our datasets: 'c' -> VIII, 'a' -> VI. So EA = VIII - VI.
# - MV (Markov Violation): p(C1=1 | C2=1) - p(C1=1 | C2=0) = Task 'd' - Task 'e'.
#   Roman tasks: 'd' -> IV, 'e' -> V. So MV = IV - V.
LETTERS_FOR_INDICES = {
    "EA": ("c", "a"),
    "MV": ("d", "e"),
}

# Prompt-category normalization: some manifests use synonyms (especially for humans)
_NUMERIC_SYNS = {"pcnum", "numeric", "num", "single_numeric", "single_numeric_response"}
_COT_SYNS = {"pccot", "cot", "chain_of_thought", "chain-of-thought", "cot_stepwise"}


def _normalize_prompt_category(agent: str, pc: Optional[str]) -> Optional[str]:
    """Return a normalized prompt-category label for data selection.

    - For humans, we collapse variants like 'single_numeric_response' to 'numeric'.
    - For LLMs, we pass through as-is unless it's one of common synonyms.
    """
    if pc is None or str(pc).lower() == "nan":
        return None
    t = str(pc).strip().lower()
    if t in _NUMERIC_SYNS:
        # Canonical internal label
        return "pcnum"
    if t in _COT_SYNS:
        # Canonical internal label
        return "pccot"
    # No change
    return pc


def _infer_link(tag: str, winners_df: Optional[pd.DataFrame], params_df: pd.DataFrame) -> str:
    if "link" in params_df.columns and isinstance(params_df["link"].dropna().iloc[0], str):
        return str(params_df["link"].dropna().iloc[0])
    if winners_df is not None and "link" in winners_df.columns and winners_df["link"].notna().any():
        return str(winners_df["link"].dropna().iloc[0])
    tl = str(tag).lower()
    if "noisy_or" in tl or "noisyor" in tl:
        return "noisy_or"
    if "logistic" in tl:
        return "logistic"
    # fall back: if param names suggest noisy_or
    if all(c in params_df.columns for c in ["b", "m1", "m2", "pC1", "pC2"]):
        return "noisy_or"
    return "logistic"


def _letters_to_roman(letters: Tuple[str, str]) -> Tuple[str, str]:
    a, b = letters
    # guard lower
    a = a.lower()
    b = b.lower()
    try:
        return (LETTER_TO_ROMAN[a], LETTER_TO_ROMAN[b])
    except Exception:
        # default to common mapping if missing
        inv = {v: k for k, v in ROMAN_TO_LETTER.items()}
        return (inv.get(a, ""), inv.get(b, ""))


def _to_float(val: object) -> float:
    try:
        if val is None:
            return float("nan")
        f = float(val)  # type: ignore[arg-type]
        return f
    except Exception:
        return float("nan")


def _raw_task_means(
    paths: PathManager,
    version: str,
    experiment: str,
    agent: str,
    prompt_category: Optional[str],
    pipeline_mode: str = "llm_with_humans",
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
    input_file: Optional[str] = None,
) -> pd.DataFrame:
    df = load_processed_data(
        paths,
        version=version,
        experiment_name=experiment,
        graph_type=graph_type,
        use_roman_numerals=use_roman_numerals,
        use_aggregated=use_aggregated,
        pipeline_mode=pipeline_mode,
        input_file=input_file,
    )
    # Normalize prompt-category to a type (numeric vs cot), then accept multiple label synonyms
    pc_norm = _normalize_prompt_category(agent, prompt_category)
    candidates: List[str] = []
    if pc_norm is not None and str(pc_norm).lower() != "nan":
        t = str(pc_norm).strip().lower()
        if t in _NUMERIC_SYNS or t == "numeric":
            candidates = [
                # Likely labels across pipelines
                "pcnum", "numeric", "numeric-conf", "num", "single_numeric", "single_numeric_response",
            ]
        elif t in _COT_SYNS or t in {"cot", "cot_stepwise", "chain_of_thought", "chain-of-thought"}:
            candidates = [
                # Likely labels across pipelines
                "pccot", "cot", "CoT", "cot_stepwise", "chain_of_thought", "chain-of-thought",
            ]
        else:
            # Unknown label; try as-is and a lower-cased variant
            candidates = [str(pc_norm), str(pc_norm).strip(), str(pc_norm).strip().lower()]
    # Always include the original if present and not already captured
    if prompt_category is not None:
        for c in [str(prompt_category), str(prompt_category).strip(), str(prompt_category).strip().lower()]:
            if c and c not in candidates:
                candidates.append(c)

    sub = pd.DataFrame()
    if candidates:
        # Accept any of the synonyms at once to avoid missing variants (e.g., 'numeric-conf')
        sub = prepare_dataset(df, agents=[agent], domains=None, prompt_categories=candidates)
    else:
        # No filtering on prompt_category if we couldn't determine a candidate
        sub = prepare_dataset(df, agents=[agent], domains=None, prompt_categories=None)
    if sub.empty:
        return pd.DataFrame({"task": [], "response_mean": []})
    g = (
        sub.groupby("task", dropna=False)["response"].mean().reset_index().rename(columns={"response": "response_mean"})
    )
    g["task"] = g["task"].astype(str)
    return g


def _raw_task_means_with_domain(
    paths: PathManager,
    version: str,
    experiment: str,
    agent: str,
    prompt_category: Optional[str],
    *,
    pipeline_mode: str = "llm_with_humans",
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
    input_file: Optional[str] = None,
) -> pd.DataFrame:
    """Return per-domain, per-task mean responses for an agent and prompt_category.

    Output columns: [domain, task, response_mean]. Domain may include pooled/NA rows;
    we will exclude pooled ('all') later for bootstrap CIs.
    """
    df = load_processed_data(
        paths,
        version=version,
        experiment_name=experiment,
        graph_type=graph_type,
        use_roman_numerals=use_roman_numerals,
        use_aggregated=use_aggregated,
        pipeline_mode=pipeline_mode,
        input_file=input_file,
    )
    pc_norm = _normalize_prompt_category(agent, prompt_category)
    candidates: List[str] = []
    if pc_norm is not None and str(pc_norm).lower() != "nan":
        t = str(pc_norm).strip().lower()
        if t in _NUMERIC_SYNS or t == "numeric":
            candidates = [
                "pcnum", "numeric", "numeric-conf", "num", "single_numeric", "single_numeric_response",
            ]
        elif t in _COT_SYNS or t in {"cot", "cot_stepwise", "chain_of_thought", "chain-of-thought"}:
            candidates = [
                "pccot", "cot", "CoT", "cot_stepwise", "chain_of_thought", "chain-of-thought",
            ]
        else:
            candidates = [str(pc_norm), str(pc_norm).strip(), str(pc_norm).strip().lower()]
    if prompt_category is not None:
        for c in [str(prompt_category), str(prompt_category).strip(), str(prompt_category).strip().lower()]:
            if c and c not in candidates:
                candidates.append(c)

    sub = pd.DataFrame()
    if candidates:
        # Accept any of the synonyms at once
        sub = prepare_dataset(df, agents=[agent], domains=None, prompt_categories=candidates)
    else:
        sub = prepare_dataset(df, agents=[agent], domains=None, prompt_categories=None)
    if sub.empty:
        return pd.DataFrame({"domain": [], "task": [], "response_mean": []})

    grp_cols = [c for c in ["domain", "task"] if c in sub.columns]
    if not grp_cols:
        return pd.DataFrame({"domain": [], "task": [], "response_mean": []})
    g = (
        sub.groupby(grp_cols, dropna=False)["response"].mean().reset_index().rename(columns={"response": "response_mean"})
    )
    g["task"] = g["task"].astype(str)
    # Normalize domain labels to strings; treat NaN as empty string
    g["domain"] = g["domain"].astype(str)
    return g


def _model_task_probs(
    roman: List[str],
    link: str,
    row_params: Dict[str, float],
) -> Dict[str, float]:
    tensors = {k: torch.tensor(float(v), dtype=torch.float32) for k, v in row_params.items()}
    out: Dict[str, float] = {}
    for r in roman:
        try:
            y = roman_task_to_probability(r, link, tensors).item()
        except Exception:
            y = float("nan")
        out[r] = float(y)
    return out


def _ea_mv_from_series(task_vals: Dict[str, float]) -> Tuple[float, float, str, str]:
    """
    Compute EA and MV from a mapping of task -> mean probability.

    Supports both Roman-numeral keys (e.g., 'VI', 'VIII') and letter keys ('a','c',...).

    Returns
    - EA value, MV value (floats)
    - EA task descriptor, MV task descriptor (strings), e.g., 'VIII - VI' and 'IV - V'.

    Notes
    - Previously, both metrics were returned as NaN if any of the four tasks (VI, VIII, IV, V)
      were missing. Now EA and MV are computed independently: if only the EA pair is available
      we compute EA and leave MV as NaN (and vice versa).

    Probability semantics (collider graph):
    - EA (Explaining Away) = p(C1=1 | E=1, C2=0) - p(C1=1 | E=1, C2=1)
        Tasks: 'c' - 'a' which map to Roman VIII - VI in our datasets.
    - MV (Markov Violation) = p(C1=1 | C2=1) - p(C1=1 | C2=0)
        Tasks: 'd' - 'e' which map to Roman IV - V.
    """
    # Normalize incoming keys: support keys like 'VI', 'Task VI', 'vi', or letters 'a'..'k'
    def _normalize_task_map(vals: Dict[str, float]) -> Dict[str, float]:
        norm: Dict[str, float] = {}
        romans = list(ROMAN_TO_LETTER.keys())  # ['I','II',...,'XI']
        letters = set(ROMAN_TO_LETTER.values())  # {'i','j',...,'h'} etc
        for k, v in vals.items():
            ks = str(k).strip()
            ks_upper = ks.upper()
            ks_lower = ks.lower()
            # Try exact roman match first
            if ks_upper in romans:
                norm[ks_upper] = float(v)
                continue
            # Try to find roman as substring token (e.g., 'Task VI')
            found = False
            for r in sorted(romans, key=len, reverse=True):
                if r in ks_upper:
                    norm[r] = float(v)
                    found = True
                    break
            if found:
                continue
            # Try letters
            if ks_lower in letters:
                norm[ks_lower] = float(v)
                continue
            for letter in letters:
                if letter in ks_lower:
                    norm[letter] = float(v)
                    break
        return norm if norm else {str(k): float(v) for k, v in vals.items()}

    task_vals = _normalize_task_map(task_vals)
    # Derive both letter and roman task IDs
    c_letter, a_letter = LETTERS_FOR_INDICES["EA"]  # c, a
    d_letter, e_letter = LETTERS_FOR_INDICES["MV"]  # d, e
    r_c, r_a = _letters_to_roman((c_letter, a_letter))
    r_d, r_e = _letters_to_roman((d_letter, e_letter))

    keys = set(task_vals.keys())
    # Compute EA and MV independently, preferring Roman when both pairwise keys exist
    ea_val = float("nan")
    mv_val = float("nan")
    ea_desc = f"{r_c} - {r_a}"
    mv_desc = f"{r_d} - {r_e}"

    # EA with Roman keys
    if {r_c, r_a}.issubset(keys):
        ea_val = float(task_vals.get(r_c, np.nan)) - float(task_vals.get(r_a, np.nan))
        ea_desc = f"{r_c} - {r_a}"
    else:
        # EA with letter keys
        lower_map = {str(k).lower(): float(v) for k, v in task_vals.items()}
        if {c_letter, a_letter}.issubset(set(lower_map.keys())):
            ea_val = float(lower_map.get(c_letter, np.nan)) - float(lower_map.get(a_letter, np.nan))
            ea_desc = f"{c_letter} - {a_letter}"

    # MV with Roman keys
    if {r_d, r_e}.issubset(keys):
        mv_val = float(task_vals.get(r_d, np.nan)) - float(task_vals.get(r_e, np.nan))
        mv_desc = f"{r_d} - {r_e}"
    else:
        # MV with letter keys
        lower_map = {str(k).lower(): float(v) for k, v in task_vals.items()}
        if {d_letter, e_letter}.issubset(set(lower_map.keys())):
            mv_val = float(lower_map.get(d_letter, np.nan)) - float(lower_map.get(e_letter, np.nan))
            mv_desc = f"{d_letter} - {e_letter}"

    return ea_val, mv_val, ea_desc, mv_desc


def analyze(args: argparse.Namespace) -> int:
    paths = PathManager()
    # Resolve tag dir and winners
    tag_dir = Path(paths.base_dir) / "results" / "parameter_analysis" / args.experiment / args.tag
    winners_params = tag_dir / "winners_with_params.csv"
    winners_csv = tag_dir / "winners.csv"
    if not winners_params.exists():
        print(f"[ERROR] winners_with_params.csv not found: {winners_params}")
        return 2
    params_df = pd.read_csv(winners_params)
    winners_df = pd.read_csv(winners_csv) if winners_csv.exists() else None
    link = _infer_link(args.tag, winners_df, params_df)

    # Normalize useful columns
    key_cols = [c for c in ["agent", "prompt_category", "domain"] if c in params_df.columns]
    meta_keep = [c for c in ["loocv_r2", "cv_r2", "r2", "params_tying"] if winners_df is not None and c in winners_df.columns]
    meta = winners_df[key_cols + meta_keep].drop_duplicates() if (winners_df is not None and not winners_df.empty) else pd.DataFrame(columns=key_cols)

    # Compute metrics per (agent, prompt_category)
    rows: List[Dict[str, object]] = []
    # We also collect domain-level EA/MV to derive bootstrap CIs per agent×PC
    per_agent_pc_domains: Dict[Tuple[str, Optional[str]], Dict[str, Tuple[float, float]]] = {}
    for _, r in params_df.iterrows():
        agent = str(r.get("agent"))
        pc = str(r.get("prompt_category")) if "prompt_category" in params_df.columns else None

        # Merge meta metrics
        mrow: Dict[str, object] = {}
        if not meta.empty:
            q = meta
            if "agent" in q.columns:
                q = q[q["agent"].astype(str) == agent]
            if pc is not None and "prompt_category" in q.columns:
                q = q[q["prompt_category"].astype(str) == pc]
            if not q.empty:
                mrow = q.iloc[0].to_dict()

        # Raw task means
        raw_means = _raw_task_means(
            paths,
            version=args.version,
            experiment=args.experiment,
            agent=agent,
            prompt_category=pc,
            pipeline_mode=args.pipeline_mode,
            graph_type=args.graph_type,
            use_roman_numerals=True,
            use_aggregated=not args.no_aggregated,
            input_file=args.input_file,
        )
        raw_map = {str(t): float(v) for t, v in zip(raw_means.get("task", []), raw_means.get("response_mean", []))}
        # Observed probabilities from data
        ea_raw, mv_raw, ea_tasks_desc, mv_tasks_desc = _ea_mv_from_series(raw_map)

        # Model task probabilities
        param_keys = ["b", "m1", "m2", "pC1", "pC2"] if link == "noisy_or" else ["w0", "w1", "w2", "pC1", "pC2"]
        rp: Dict[str, float] = {}
        for k in param_keys:
            v = r.get(k)
            if k in params_df.columns and pd.notna(v):
                try:
                    rp[k] = float(v)  # type: ignore[arg-type]
                except Exception:
                    pass
        model_map = _model_task_probs(ROMAN_ORDER, link, rp) if len(rp) == len(param_keys) else {}
        ea_model, mv_model, _, _ = _ea_mv_from_series(model_map)

        # Derived param stats
        b = _to_float(r.get("b"))
        m1 = _to_float(r.get("m1"))
        m2 = _to_float(r.get("m2"))
        pC1 = _to_float(r.get("pC1"))
        pC2 = _to_float(r.get("pC2"))
        m_bar = np.nanmean([m1, m2])
        m_gap = np.nan if (np.isnan(m1) or np.isnan(m2)) else abs(m1 - m2)
        normative_metric = (m_bar - b) if (not np.isnan(m_bar) and not np.isnan(b)) else float("nan")

        row_out: Dict[str, object] = {
            "agent": agent,
            "prompt_category": pc,
            "experiment": args.experiment,
            "link": link,
            "EA_raw": ea_raw,
            "MV_raw": mv_raw,
            "EA_raw_tasks": ea_tasks_desc,
            "MV_raw_tasks": mv_tasks_desc,
            "EA_model": ea_model,
            "MV_model": mv_model,
            "b": b,
            "m1": m1,
            "m2": m2,
            "pC1": pC1,
            "pC2": pC2,
            "m_bar": m_bar,
            "m_gap": m_gap,
            "normative_metric": normative_metric,
        }
        row_out.update(mrow)
        rows.append(row_out)

        # Collect per-domain EA/MV for bootstrapping later (per agent×PC)
        dom_df = _raw_task_means_with_domain(
            paths,
            version=args.version,
            experiment=args.experiment,
            agent=agent,
            prompt_category=pc,
            pipeline_mode=args.pipeline_mode,
            graph_type=args.graph_type,
            use_roman_numerals=True,
            use_aggregated=not args.no_aggregated,
            input_file=args.input_file,
        )
        if not dom_df.empty and {"domain", "task", "response_mean"}.issubset(dom_df.columns):
            # Build per-domain maps, exclude pooled 'all' if present
            by_dom: Dict[str, Dict[str, float]] = {}
            for _, r2 in dom_df.iterrows():
                dlab = str(r2.get("domain"))
                if dlab.strip().lower() == "all":
                    continue
                tname = str(r2.get("task"))
                val = r2.get("response_mean")
                try:
                    by_dom.setdefault(dlab, {})[tname] = float(val) if val is not None else float("nan")
                except Exception:
                    by_dom.setdefault(dlab, {})[tname] = float("nan")
            if by_dom:
                ea_mv_by_dom: Dict[str, Tuple[float, float]] = {}
                for dlab, tmap in by_dom.items():
                    ea_d, mv_d, _, _ = _ea_mv_from_series(tmap)
                    ea_mv_by_dom[dlab] = (ea_d, mv_d)
                per_agent_pc_domains[(agent, pc)] = ea_mv_by_dom

    out = pd.DataFrame(rows)

    # Attach domain-bootstrap CIs and medians on EA/MV per agent×PC
    if not out.empty and per_agent_pc_domains:
        rng = np.random.default_rng(getattr(args, "seed", 123))
        B = int(getattr(args, "bootstrap", 2000) or 2000)
        ci = float(getattr(args, "ci", 95.0) or 95.0)
        alpha = (100.0 - ci) / 2.0
        # Compute per key and merge back
        ci_records: List[Dict[str, object]] = []
        for (agent, pc), dom_map in per_agent_pc_domains.items():
            ea_vals = np.array([v[0] for v in dom_map.values()], dtype=float)
            mv_vals = np.array([v[1] for v in dom_map.values()], dtype=float)
            # Filter finite
            ea_vals = ea_vals[np.isfinite(ea_vals)]
            mv_vals = mv_vals[np.isfinite(mv_vals)]
            rec: Dict[str, object] = {"agent": agent, "prompt_category": pc}
            def _boot(vals: np.ndarray) -> Tuple[float, float, float, float]:
                if vals.size == 0:
                    return (float("nan"), float("nan"), float("nan"), float("nan"))
                mu = float(np.nanmean(vals))
                med = float(np.nanmedian(vals))
                if vals.size == 1 or B <= 0:
                    return (mu, med, float("nan"), float("nan"))
                idx = rng.integers(0, vals.size, size=(B, vals.size))
                boots = np.nanmean(vals[idx], axis=1)
                lo = float(np.percentile(boots, alpha))
                hi = float(np.percentile(boots, 100 - alpha))
                return (mu, med, lo, hi)
            ea_mu, ea_med, ea_lo, ea_hi = _boot(ea_vals)
            mv_mu, mv_med, mv_lo, mv_hi = _boot(mv_vals)
            rec.update({
                "EA_raw_mean": ea_mu,
                "EA_raw_median": ea_med,
                "EA_raw_lo": ea_lo,
                "EA_raw_hi": ea_hi,
                "MV_raw_mean": mv_mu,
                "MV_raw_median": mv_med,
                "MV_raw_lo": mv_lo,
                "MV_raw_hi": mv_hi,
                "n_domains": int(len(dom_map)),
            })
            ci_records.append(rec)
        if ci_records:
            ci_df = pd.DataFrame(ci_records)
            out = out.merge(ci_df, on=["agent", "prompt_category"], how="left")

    out_dir = tag_dir / "cogn_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strategy_metrics.csv"
    out.sort_values(["agent", "prompt_category"], inplace=True, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"Saved metrics: {out_path}")

    # Helper: classification applied to a DataFrame, returns a copy with boolean columns
    def _classify_df(df: pd.DataFrame, ea_thr: float, mv_thr: float, r2_thr: float) -> pd.DataFrame:
        out_df = df.copy()
        if "EA_raw" in out_df.columns:
            ea_vals = pd.to_numeric(out_df["EA_raw"], errors="coerce")
        else:
            ea_vals = pd.Series(np.nan, index=out_df.index)
        if "MV_raw" in out_df.columns:
            mv_vals = pd.to_numeric(out_df["MV_raw"], errors="coerce").abs()
        else:
            mv_vals = pd.Series(np.nan, index=out_df.index)
        if "loocv_r2" in out_df.columns:
            r2_vals = pd.to_numeric(out_df["loocv_r2"], errors="coerce")
        else:
            r2_vals = pd.Series(np.nan, index=out_df.index)

        out_df["meets_ea"] = ea_vals > float(ea_thr)
        out_df["meets_mv"] = mv_vals < float(mv_thr)
        out_df["meets_loocv_r2"] = r2_vals > float(r2_thr)
        out_df["normative_reasoner"] = out_df[["meets_ea", "meets_mv", "meets_loocv_r2"]].all(axis=1)
        return out_df

    # Optional: classify reasoning patterns based on thresholds and save a copy with flags
    if getattr(args, "classify_reasoning", False):
        # Build subdirectory name like: ea_diff_0.3_mv_diff_0.05_loocv_r2_0.89
        def _fmt(v: float) -> str:
            try:
                # compact formatting without trailing zeros
                return "%g" % float(v)
            except Exception:
                return str(v)

        subdir_name = f"ea_diff_{_fmt(args.ea_diff)}_mv_diff_{_fmt(args.mv_diff)}_loocv_r2_{_fmt(args.loocv_r2)}"
        out_dir_cls: Path = out_dir / subdir_name
        out_dir_cls.mkdir(parents=True, exist_ok=True)
        classified = _classify_df(out, args.ea_diff, args.mv_diff, args.loocv_r2)
        # Add threshold columns for traceability
        classified["ea_diff_threshold"] = float(args.ea_diff)
        classified["mv_diff_threshold"] = float(args.mv_diff)
        classified["loocv_r2_threshold"] = float(args.loocv_r2)
        out_path_cls = out_dir_cls / "classified_strategy_metrics.csv"
        classified.to_csv(out_path_cls, index=False)
        print(f"Saved classified metrics: {out_path_cls}")

    # Plot violin of parameter distributions for normative and non-normative reasoners if requested
    if getattr(args, "plot", False):
        # Ensure we have a classified DataFrame
        def _fmt(v: float) -> str:
            try:
                return "%g" % float(v)
            except Exception:
                return str(v)
        subdir_name = f"ea_diff_{_fmt(args.ea_diff)}_mv_diff_{_fmt(args.mv_diff)}_loocv_r2_{_fmt(args.loocv_r2)}"
        out_dir_cls: Path = out_dir / subdir_name
        out_dir_cls.mkdir(parents=True, exist_ok=True)

        classified_df = _classify_df(out, args.ea_diff, args.mv_diff, args.loocv_r2)
        total = int(len(classified_df))

        def _plot_violin_for_subset(df_subset: pd.DataFrame, out_name: str, count_label: str) -> None:
            params = ["b", "m1", "m2", "pC1", "pC2"]
            data: List[np.ndarray] = []
            labels: List[str] = []
            for p in params:
                if p in df_subset.columns:
                    vals = pd.to_numeric(df_subset[p], errors="coerce").astype(float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) > 0:
                        data.append(np.asarray(vals.values, dtype=float))
                        labels.append(p)
            if not data:
                print(f"[WARN] No {count_label} or no valid parameter values to plot.")
                return

            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            vparts = ax.violinplot(data, showmeans=False, showextrema=True, showmedians=False, widths=0.9)
            if isinstance(vparts, dict):
                for k in ("cmins", "cmaxes", "cbars"):
                    if k in vparts and vparts[k] is not None:
                        vparts[k].set_color((0.4, 0.4, 0.4))
                        vparts[k].set_linewidth(1.0)

            xpos = np.arange(1, len(labels) + 1)
            means = [float(np.nanmean(vals)) for vals in data]
            medians = [float(np.nanmedian(vals)) for vals in data]
            ax.scatter(xpos, means, marker='o', color='black', s=30, zorder=3, label='Mean')
            ax.scatter(xpos, medians, marker='^', color='green', s=45, zorder=3, label='Median')

            label_map = {"b": r"$b$", "m1": r"$m_1$", "m2": r"$m_2$", "pC1": r"$p(C_1)$", "pC2": r"$p(C_2)$"}
            labels_tex = [label_map[p] if p in label_map else p for p in labels]
            ax.set_xticks(range(1, len(labels_tex) + 1))
            ax.set_xticklabels(labels_tex)
            ax.set_ylabel("Parameter value")

            # Title
            num = len(df_subset)
            title = rf"EA $>$ {_fmt(args.ea_diff)}, $|MV|$ $<$ {_fmt(args.mv_diff)}, $R^2$ $>$ {_fmt(args.loocv_r2)} --- {num}/{total} {count_label}"
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)

            dist_handle = Patch(facecolor=(0.3, 0.3, 0.8), edgecolor=(0.2, 0.2, 0.5), alpha=0.25, label="Distribution")
            whisk_handle = Line2D([0], [0], color=(0.4, 0.4, 0.4), lw=1.0, label="Min/max")
            mean_handle = Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=6, label='Mean')
            med_handle = Line2D([0], [0], marker='^', color='green', linestyle='None', markersize=7, label=r'Median')
            ax.legend(handles=[dist_handle, whisk_handle, mean_handle, med_handle], frameon=False)

            fig.tight_layout()
            out_fig = out_dir_cls / out_name
            fig.savefig(out_fig)
            plt.close(fig)
            print(f"Saved plot: {out_fig}")

        # Plot for normative reasoners
        norm_df = classified_df[classified_df["normative_reasoner"]].copy()
        _plot_violin_for_subset(norm_df, "violin_normative_params.pdf", "normative reasoners")

        # Plot for non-normative reasoners
        non_norm_df = classified_df[~classified_df["normative_reasoner"]].copy()
        _plot_violin_for_subset(non_norm_df, "violin_non_normative_params.pdf", "NON-normative reasoners")

    # Robustness across prompt categories (pcnum → pccot) within this experiment/tag
    if {"prompt_category"}.issubset(out.columns):
        try:
            pcnum = args.pcnum_label
            pccot = args.pccot_label
            A = out[out["prompt_category"].astype(str) == pcnum].copy()
            B = out[out["prompt_category"].astype(str) == pccot].copy()
            if not A.empty and not B.empty:
                cols_to_delta = [c for c in ["loocv_r2", "b", "m1", "m2", "pC1", "pC2"] if c in out.columns]
                merged = pd.merge(A, B, on=["agent"], suffixes=("__A", "__B"))
                for c in cols_to_delta:
                    colA = f"{c}__A"
                    colB = f"{c}__B"
                    if colA in merged.columns and colB in merged.columns:
                        merged[f"delta_{c}"] = pd.to_numeric(merged[colB], errors="coerce") - pd.to_numeric(merged[colA], errors="coerce")
                # Normative boolean: small deltas and good absolute levels (high r2, m_bar; low b)
                def good_level(row) -> bool:
                    ok_r2 = True
                    if "loocv_r2__A" in merged.columns and "loocv_r2__B" in merged.columns:
                        ok_r2 = (float(row.get("loocv_r2__A", np.nan)) >= args.r2_min) and (float(row.get("loocv_r2__B", np.nan)) >= args.r2_min)
                    ok_m = True
                    if "m_bar__A" in merged.columns and "m_bar__B" in merged.columns:
                        ok_m = (float(row.get("m_bar__A", np.nan)) >= args.m_min) and (float(row.get("m_bar__B", np.nan)) >= args.m_min)
                    ok_b = True
                    if "b__A" in merged.columns and "b__B" in merged.columns:
                        ok_b = (float(row.get("b__A", np.nan)) <= args.b_max) and (float(row.get("b__B", np.nan)) <= args.b_max)
                    return bool(ok_r2 and ok_m and ok_b)
                def small_deltas(row) -> bool:
                    checks = []
                    for c in cols_to_delta:
                        dv = float(row.get(f"delta_{c}", np.nan))
                        if np.isnan(dv):
                            continue
                        thresh = args.delta_tol_params if c != "loocv_r2" else args.delta_tol_r2
                        checks.append(abs(dv) <= thresh)
                    return all(checks) if checks else False
                merged["normative"] = merged.apply(lambda r: small_deltas(r) and good_level(r), axis=1)
                out_cols = ["agent"] + [f"delta_{c}" for c in cols_to_delta] + ["normative"]
                rob = merged[out_cols].copy()
                rob_path = out_dir / f"robustness_{pcnum}_to_{pccot}.csv"
                rob.to_csv(rob_path, index=False)
                print(f"Saved robustness: {rob_path}")
        except Exception as e:
            print(f"[WARN] Robustness export failed: {e}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute EA/MV indices and robustness from raw and CBN predictions")
    p.add_argument("--experiment", required=True, help="Experiment name (e.g., rw17_indep_causes)")
    p.add_argument("--version", required=True, help="Version used when loading processed data")
    p.add_argument("--tag", required=True, help="Parameter-analysis tag under results/parameter_analysis/<experiment>/")

    # Data loading mirroring other scripts
    p.add_argument("--graph-type", choices=["collider", "fork", "chain"], default="collider")
    p.add_argument("--pipeline-mode", choices=["llm_with_humans", "llm", "humans"], default="llm_with_humans")
    p.add_argument("--no-aggregated", action="store_true")
    p.add_argument("--input-file", help="Override processed input CSV (optional)")

    # Prompt-category labels for within-experiment robustness
    p.add_argument("--pcnum-label", default="pcnum", help="Label used for numeric prompts (default: pcnum)")
    p.add_argument("--pccot-label", default="pccot", help="Label used for chain-of-thought prompts (default: pccot)")

    # Robustness thresholds
    p.add_argument("--delta-tol-r2", type=float, default=0.05, help="Max |Δ R2| to count as small (default 0.05)")
    p.add_argument("--delta-tol-params", type=float, default=0.05, help="Max |Δ param| to count as small for params (default 0.05)")
    p.add_argument("--r2-min", type=float, default=0.5, help="Min loocv_r2 on both sides to be normative (default 0.5)")
    p.add_argument("--m-min", type=float, default=0.6, help="Min m_bar on both sides to be normative (default 0.6)")
    p.add_argument("--b-max", type=float, default=0.2, help="Max b on both sides to be normative (default 0.2)")

    # Bootstrap CI settings (used to compute EA/MV domain-bootstrap CIs)
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap samples for domain-level CIs (default 2000)")
    p.add_argument("--ci", type=float, default=95.0, help="Confidence level for bootstrap CIs (e.g., 95)")
    p.add_argument("--seed", type=int, default=123, help="Random seed for bootstrap resampling")

    # Classification toggles and thresholds (applied to per-row values in strategy_metrics.csv)
    p.add_argument("--classify-reasoning", action="store_true", help="If set, append classification columns and save classified_strategy_metrics.csv in a thresholds-named subfolder")
    p.add_argument("--ea-diff", type=float, default=0.0995, help="Threshold for EA_raw: require EA_raw > ea_diff (0.0995 default human baseline)")
    p.add_argument("--mv-diff", type=float, default=0.100, help="Threshold for MV_raw: require abs(MV_raw) < mv_diff (0.100 default human baseline)")
    p.add_argument("--loocv-r2", type=float, default=0.937, help="Threshold for loocv_r2: require loocv_r2 > loocv_r2 (default human baseline)")
    p.add_argument("--plot", action="store_true", help="If set, generate a violin plot over parameters for normative reasoners and save it next to the classified CSV")
# ,0.0995138888888889,0.1004166666666666
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
