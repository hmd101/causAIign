#!/usr/bin/env python3
"""
Canonical location for EA/MV summarization (moved from scripts/summarize_ea_mv_levels.py).
See inline docstring in the original for details. This version is identical.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

NUMERIC_SYNS = {"numeric", "pcnum", "num", "single_numeric", "single_numeric_response"}
COT_SYNS = {"cot", "pccot", "chain_of_thought", "chain-of-thought", "cot_stepwise", "cot", "CoT"}

EXPERIMENT_LABEL: Dict[str, str] = {
    "rw17_indep_causes": "RW17",
    "random_abstract": "Abstract",
    "rw17_overloaded_e": "RW17-Over",
    "abstract_overloaded_lorem_de": "Abstract-Over",
}


def _canon_prompt(p: str) -> str:
    t = str(p).strip()
    tl = t.lower()
    if t in COT_SYNS or tl in COT_SYNS:
        return "CoT"
    if t in NUMERIC_SYNS or tl in NUMERIC_SYNS or tl == "numeric":
        return "numeric"
    return t


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prompt_category" in df.columns:
        df["prompt_category"] = df["prompt_category"].astype(str).map(_canon_prompt)
    if "domain" in df.columns:
        pooled = df[df["domain"].astype(str) == "all"].copy()
        if not pooled.empty:
            df = pooled
    for col in ("EA_raw", "MV_raw"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_rw17_human_baseline(df_all: pd.DataFrame, metric: str) -> Optional[float]:
    metric_col = "EA_raw" if metric == "ea" else "MV_raw"
    if metric_col not in df_all.columns:
        return None
    d = df_all.copy()
    if "prompt_category" in d.columns:
        d["prompt_category"] = d["prompt_category"].astype(str).map(_canon_prompt)
    mask = (
        d["agent"].astype(str).str.lower().str.contains("human")
        & d["experiment"].astype(str).eq("rw17_indep_causes")
        & d["prompt_category"].astype(str).eq("numeric")
    )
    sub = d[mask].copy()
    if sub.empty or metric_col not in sub.columns:
        return None
    sub = sub[pd.to_numeric(sub[metric_col], errors="coerce").notna()]
    if sub.empty:
        return None
    pooled = sub[sub.get("domain", pd.Series(dtype=str)).astype(str).eq("all")]
    if not pooled.empty:
        return float(pooled.iloc[0][metric_col])
    return float(sub[metric_col].mean())


def _metric_values(series: pd.Series, metric: str) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if metric == "mv":
        return np.abs(vals)
    return vals


def _bootstrap_median_ci(values: np.ndarray, B: int, ci: float, rng: np.random.Generator) -> Tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    med = float(np.median(values))
    if values.size == 1:
        return (med, med, med)
    alpha = (100.0 - ci) / 2.0
    idx = rng.integers(0, values.size, size=(B, values.size))
    boots = np.median(values[idx], axis=1)
    lo = float(np.percentile(boots, alpha))
    hi = float(np.percentile(boots, 100 - alpha))
    return (med, lo, hi)


@dataclass
class Criteria:
    metric: str  # "ea" or "mv"
    th_ea: float = 0.3
    th_mv: float = 0.05
    human: Optional[float] = None

    def passes_threshold(self, v: float) -> Optional[bool]:
        import numpy as _np
        if _np.isnan(v):
            return None
        if self.metric == "ea":
            return v > self.th_ea
        else:
            return abs(v) <= self.th_mv

    def passes_human(self, v: float) -> Optional[bool]:
        import numpy as _np
        if self.human is None or _np.isnan(v):
            return None
        if self.metric == "ea":
            return v > float(self.human)
        else:
            return abs(v) <= abs(float(self.human))


def _llm_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["agent"].astype(str).str.lower().str.contains("human")].copy()


def _pct_bool(series: pd.Series) -> Tuple[int, int, float]:
    vals = series.dropna().astype(bool)
    n = len(vals)
    k = int(vals.sum())
    pct = 0.0 if n == 0 else 100.0 * k / n
    return k, n, pct


def summarize_by_exp_pc(df: pd.DataFrame, crit: Criteria, B: int, ci: float, rng: np.random.Generator) -> pd.DataFrame:
    metric_col = "EA_raw" if crit.metric == "ea" else "MV_raw"
    rows = []
    for (exp, pc), grp in _llm_only(df).groupby(["experiment", "prompt_category"], dropna=False):
        if metric_col not in grp.columns:
            continue
        vals_series = grp[metric_col]
        over_th = vals_series.apply(crit.passes_threshold)
        over_hu = vals_series.apply(crit.passes_human)
        k_th, n_th, pct_th = _pct_bool(pd.Series(over_th))
        k_hu, n_hu, pct_hu = _pct_bool(pd.Series(over_hu))
        valid = vals_series.notna()
        vals_all = _metric_values(vals_series[valid], crit.metric)
        med_all, lo_all, hi_all = _bootstrap_median_ci(vals_all, B, ci, rng)
        mask_th = pd.Series(over_th)[valid].fillna(False).to_numpy()
        mask_hu = pd.Series(over_hu)[valid].fillna(False).to_numpy()
        med_th, lo_th, hi_th = _bootstrap_median_ci(vals_all[mask_th], B, ci, rng)
        med_hu, lo_hu, hi_hu = (
            _bootstrap_median_ci(vals_all[mask_hu], B, ci, rng) if crit.human is not None else (float("nan"),) * 3
        )
        rows.append(
            {
                "experiment": exp,
                "prompt_category": pc,
                "n": int(len(vals_series.dropna())),
                "k_threshold": k_th,
                "pct_threshold": pct_th,
                "k_human": k_hu,
                "pct_human": pct_hu,
                "median_metric_all_agents": med_all,
                "ci_low_median_metric_all_agents": lo_all,
                "ci_high_median_metric_all_agents": hi_all,
                "median_metric_agents_meeting_threshold": med_th,
                "ci_low_median_metric_agents_meeting_threshold": lo_th,
                "ci_high_median_metric_agents_meeting_threshold": hi_th,
                "median_metric_agents_meeting_human_baseline": med_hu,
                "ci_low_median_metric_agents_meeting_human_baseline": lo_hu,
                "ci_high_median_metric_agents_meeting_human_baseline": hi_hu,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["experiment", "prompt_category"]).reset_index(drop=True)


def _collapse_across_prompts(
    df: pd.DataFrame, crit: Criteria, B: int, ci: float, rng: np.random.Generator
) -> pd.DataFrame:
    metric_col = "EA_raw" if crit.metric == "ea" else "MV_raw"
    d = _llm_only(df).copy()
    if metric_col not in d.columns:
        return pd.DataFrame()
    d["passes_th"] = d[metric_col].apply(crit.passes_threshold)
    d["passes_hu"] = d[metric_col].apply(crit.passes_human)
    rows = []
    for exp, g in d.groupby("experiment", dropna=False):
        per_agent = g.groupby("agent").agg(
            any_th=("passes_th", lambda s: None if s.dropna().empty else bool(pd.Series(s).dropna().any())),
            any_hu=("passes_hu", lambda s: None if s.dropna().empty else bool(pd.Series(s).dropna().any())),
            both_th=("passes_th", lambda s: None if s.dropna().empty else bool(pd.Series(s).dropna().all())),
            both_hu=("passes_hu", lambda s: None if s.dropna().empty else bool(pd.Series(s).dropna().all())),
        )
        agent_vals = g.groupby("agent")[metric_col].apply(lambda s: np.median(_metric_values(s, crit.metric))).astype(float)
        k_any_hu, n_any_hu, pct_any_hu = _pct_bool(per_agent["any_hu"]) if crit.human is not None else (0, 0, 0.0)
        k_any_th, n_any_th, pct_any_th = _pct_bool(per_agent["any_th"])  # threshold always defined
        k_both_hu, n_both_hu, pct_both_hu = _pct_bool(per_agent["both_hu"]) if crit.human is not None else (0, 0, 0.0)
        k_both_th, n_both_th, pct_both_th = _pct_bool(per_agent["both_th"])  # threshold always defined
        med_all, lo_all, hi_all = _bootstrap_median_ci(agent_vals.to_numpy(), B, ci, rng)
        any_th_mask = per_agent["any_th"].fillna(False).to_numpy()
        any_hu_mask = per_agent["any_hu"].fillna(False).to_numpy()
        med_any_th, lo_any_th, hi_any_th = _bootstrap_median_ci(agent_vals.to_numpy()[any_th_mask], B, ci, rng)
        if crit.human is not None:
            med_any_hu, lo_any_hu, hi_any_hu = _bootstrap_median_ci(agent_vals.to_numpy()[any_hu_mask], B, ci, rng)
        else:
            med_any_hu, lo_any_hu, hi_any_hu = (float("nan"),) * 3
        both_th_mask = per_agent["both_th"].fillna(False).to_numpy()
        both_hu_mask = per_agent["both_hu"].fillna(False).to_numpy()
        med_both_th, lo_both_th, hi_both_th = _bootstrap_median_ci(agent_vals.to_numpy()[both_th_mask], B, ci, rng)
        if crit.human is not None:
            med_both_hu, lo_both_hu, hi_both_hu = _bootstrap_median_ci(agent_vals.to_numpy()[both_hu_mask], B, ci, rng)
        else:
            med_both_hu, lo_both_hu, hi_both_hu = (float("nan"),) * 3
        rows.append(
            {
                "experiment": exp,
                "n_any": n_any_th,
                "k_any_threshold": k_any_th,
                "pct_any_threshold": pct_any_th,
                "k_any_human": k_any_hu,
                "pct_any_human": pct_any_hu,
                "n_both": n_both_th,
                "k_both_threshold": k_both_th,
                "pct_both_threshold": pct_both_th,
                "k_both_human": k_both_hu,
                "pct_both_human": pct_both_hu,
                "median_metric_all_agents_across_prompts": med_all,
                "ci_low_median_metric_all_agents_across_prompts": lo_all,
                "ci_high_median_metric_all_agents_across_prompts": hi_all,
                "median_metric_agents_meeting_threshold_ANY": med_any_th,
                "ci_low_median_metric_agents_meeting_threshold_ANY": lo_any_th,
                "ci_high_median_metric_agents_meeting_threshold_ANY": hi_any_th,
                "median_metric_agents_meeting_threshold_BOTH": med_both_th,
                "ci_low_median_metric_agents_meeting_threshold_BOTH": lo_both_th,
                "ci_high_median_metric_agents_meeting_threshold_BOTH": hi_both_th,
                "median_metric_agents_meeting_human_baseline_ANY": med_any_hu,
                "ci_low_median_metric_agents_meeting_human_baseline_ANY": lo_any_hu,
                "ci_high_median_metric_agents_meeting_human_baseline_ANY": hi_any_hu,
                "median_metric_agents_meeting_human_baseline_BOTH": med_both_hu,
                "ci_low_median_metric_agents_meeting_human_baseline_BOTH": lo_both_hu,
                "ci_high_median_metric_agents_meeting_human_baseline_BOTH": hi_both_hu,
            }
        )
    return pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)


def _overall_across_experiments(df: pd.DataFrame, crit: Criteria, B: int, ci: float, rng: np.random.Generator) -> pd.DataFrame:
    metric_col = "EA_raw" if crit.metric == "ea" else "MV_raw"
    d = _llm_only(df).copy()
    if metric_col not in d.columns:
        return pd.DataFrame()
    d["passes_th"] = d[metric_col].apply(crit.passes_threshold)
    d["passes_hu"] = d[metric_col].apply(crit.passes_human)
    per_agent = d.groupby("agent").agg(
        any_th=("passes_th", lambda s: None if s.dropna().empty else bool(pd.Series(s).dropna().any())),
        any_hu=("passes_hu", lambda s: None if s.dropna().empty else bool(pd.Series(s).dropna().any())),
        all_th=("passes_th", lambda s: None if s.dropna().empty else bool(pd.Series(s).dropna().all())),
        all_hu=("passes_hu", lambda s: None if s.dropna().empty else bool(pd.Series(s).dropna().all())),
    )
    k_any_hu, n_any_hu, pct_any_hu = _pct_bool(per_agent["any_hu"]) if crit.human is not None else (0, 0, 0.0)
    k_any_th, n_any_th, pct_any_th = _pct_bool(per_agent["any_th"])  # threshold always defined
    k_all_hu, n_all_hu, pct_all_hu = _pct_bool(per_agent["all_hu"]) if crit.human is not None else (0, 0, 0.0)
    k_all_th, n_all_th, pct_all_th = _pct_bool(per_agent["all_th"])  # threshold always defined
    agent_vals = d.groupby("agent")[metric_col].apply(lambda s: np.median(_metric_values(s, crit.metric))).astype(float)
    med_all, lo_all, hi_all = _bootstrap_median_ci(agent_vals.to_numpy(), B, ci, rng)
    any_th_mask = per_agent["any_th"].fillna(False).to_numpy()
    any_hu_mask = per_agent["any_hu"].fillna(False).to_numpy()
    med_any_th, lo_any_th, hi_any_th = _bootstrap_median_ci(agent_vals.to_numpy()[any_th_mask], B, ci, rng)
    if crit.human is not None:
        med_any_hu, lo_any_hu, hi_any_hu = _bootstrap_median_ci(agent_vals.to_numpy()[any_hu_mask], B, ci, rng)
    else:
        med_any_hu, lo_any_hu, hi_any_hu = (float("nan"),) * 3
    all_th_mask = per_agent["all_th"].fillna(False).to_numpy()
    all_hu_mask = per_agent["all_hu"].fillna(False).to_numpy()
    med_all_th, lo_all_th, hi_all_th = _bootstrap_median_ci(agent_vals.to_numpy()[all_th_mask], B, ci, rng)
    if crit.human is not None:
        med_all_hu, lo_all_hu, hi_all_hu = _bootstrap_median_ci(agent_vals.to_numpy()[all_hu_mask], B, ci, rng)
    else:
        med_all_hu, lo_all_hu, hi_all_hu = (float("nan"),) * 3
    return pd.DataFrame(
        [
            {
                "scope": "overall",
                "n_any": n_any_th,
                "k_any_threshold": k_any_th,
                "pct_any_threshold": pct_any_th,
                "k_any_human": k_any_hu,
                "pct_any_human": pct_any_hu,
                "n_all": n_all_th,
                "k_all_threshold": k_all_th,
                "pct_all_threshold": pct_all_th,
                "k_all_human": k_all_hu,
                "pct_all_human": pct_all_hu,
                "median_metric_all_agents_overall": med_all,
                "ci_low_median_metric_all_agents_overall": lo_all,
                "ci_high_median_metric_all_agents_overall": hi_all,
                "median_metric_agents_meeting_threshold_ANY_overall": med_any_th,
                "ci_low_median_metric_agents_meeting_threshold_ANY_overall": lo_any_th,
                "ci_high_median_metric_agents_meeting_threshold_ANY_overall": hi_any_th,
                "median_metric_agents_meeting_threshold_ALL_overall": med_all_th,
                "ci_low_median_metric_agents_meeting_threshold_ALL_overall": lo_all_th,
                "ci_high_median_metric_agents_meeting_threshold_ALL_overall": hi_all_th,
                "median_metric_agents_meeting_human_baseline_ANY_overall": med_any_hu,
                "ci_low_median_metric_agents_meeting_human_baseline_ANY_overall": lo_any_hu,
                "ci_high_median_metric_agents_meeting_human_baseline_ANY_overall": hi_any_hu,
                "median_metric_agents_meeting_human_baseline_ALL_overall": med_all_hu,
                "ci_low_median_metric_agents_meeting_human_baseline_ALL_overall": lo_all_hu,
                "ci_high_median_metric_agents_meeting_human_baseline_ALL_overall": hi_all_hu,
            }
        ]
    )


def to_latex_block(metric: str, exp_pc: pd.DataFrame, exp: pd.DataFrame, overall: pd.DataFrame) -> str:
    lines: List[str] = []
    mm = metric.upper()
    lines.append(f"% Summary for {mm} levels vs. human baseline and threshold")
    for _, r in exp_pc.iterrows():
        exp_name = EXPERIMENT_LABEL.get(str(r["experiment"]), str(r["experiment"]))
        pc = str(r["prompt_category"]) or "—"
        lines.append(
            f"% {exp_name} {pc}: threshold {r['k_threshold']}/{r['n']} = {r['pct_threshold']:.1f}\\%, "
            f"human {r['k_human']}/{r['n']} = {r['pct_human']:.1f}\\%"
        )
    for _, r in exp.iterrows():
        exp_name = EXPERIMENT_LABEL.get(str(r["experiment"]), str(r["experiment"]))
        lines.append(
            f"% {exp_name} ANY: threshold {r['k_any_threshold']}/{r['n_any']} = {r['pct_any_threshold']:.1f}\\%, "
            f"human {r['k_any_human']}/{r['n_any']} = {r['pct_any_human']:.1f}\\%"
        )
        lines.append(
            f"% {exp_name} BOTH: threshold {r['k_both_threshold']}/{r['n_both']} = {r['pct_both_threshold']:.1f}\\%, "
            f"human {r['k_both_human']}/{r['n_both']} = {r['pct_both_human']:.1f}\\%"
        )
    if not overall.empty:
        r = overall.iloc[0]
        lines.append(
            f"% OVERALL ANY: threshold {r['k_any_threshold']}/{r['n_any']} = {r['pct_any_threshold']:.1f}\\%, "
            f"human {r['k_any_human']}/{r['n_any']} = {r['pct_any_human']:.1f}\\%"
        )
        lines.append(
            f"% OVERALL ALL: threshold {r['k_all_threshold']}/{r['n_all']} = {r['pct_all_threshold']:.1f}\\%, "
            f"human {r['k_all_human']}/{r['n_all']} = {r['pct_all_human']:.1f}\\%"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize EA/MV levels vs human baseline and thresholds")
    ap.add_argument("--csv", default="results/cross_cogn_strategies/masters_classified_strategy_metrics.csv")
    ap.add_argument("--metric", choices=["ea", "mv"], required=True)
    ap.add_argument("--th_ea", type=float, default=0.3)
    ap.add_argument("--th_mv", type=float, default=0.05)
    ap.add_argument("--experiments", nargs="+", help="Optional experiments to include (default: all)")
    ap.add_argument("--tag", help="Optional tag filter")
    ap.add_argument("--out-prefix", default="results/plots/ea_mv_levels/summary")
    ap.add_argument("--no_human", action="store_true", help="Skip human baseline comparisons even if present")
    ap.add_argument("--human", type=float, help="Override human baseline value (EA or MV depending on --metric)")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap samples for median CIs")
    ap.add_argument("--ci", type=float, default=95.0, help="Confidence level for bootstrap CIs (e.g., 95)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for bootstrap")
    args = ap.parse_args()

    df = _load_data(Path(args.csv))
    if args.tag and "tag" in df.columns:
        df = df[df["tag"].astype(str) == str(args.tag)].copy()
    if args.experiments:
        keep = set(map(str, args.experiments))
        df = df[df["experiment"].astype(str).isin(keep)].copy()

    human = None if args.no_human else (args.human if args.human is not None else _get_rw17_human_baseline(df, args.metric))
    crit = Criteria(metric=args.metric, th_ea=args.th_ea, th_mv=args.th_mv, human=human)
    rng = np.random.default_rng(args.seed)
    per_exp_pc = summarize_by_exp_pc(df, crit, args.bootstrap, args.ci, rng)
    per_exp = _collapse_across_prompts(df, crit, args.bootstrap, args.ci, rng)
    overall = _overall_across_experiments(df, crit, args.bootstrap, args.ci, rng)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_exp_pc.to_csv(out_prefix.with_suffix("").as_posix() + f"_{args.metric}_exp_pc.csv", index=False)
    per_exp.to_csv(out_prefix.with_suffix("").as_posix() + f"_{args.metric}_exp.csv", index=False)
    overall.to_csv(out_prefix.with_suffix("").as_posix() + f"_{args.metric}_overall.csv", index=False)
    tex = to_latex_block(args.metric, per_exp_pc, per_exp, overall)
    (out_prefix.with_suffix(".tex").parent / f"summary_{args.metric}.tex").write_text(tex, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
