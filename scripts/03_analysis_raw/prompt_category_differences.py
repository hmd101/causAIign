#!/usr/bin/env python3
"""
Prompt-category effects within experiments, plus optional across-experiments omnibus.

Goal
-----
Answer two questions using raw likelihood judgments (0–100):
  1) Within each experiment, does prompt_category (numeric vs CoT) lead to different
     response distributions for a given agent?
  2) Across experiments, does a given agent’s behavior differ (omnibus across experiments),
     optionally per prompt category?

Core analyses
-------------
- Within-experiment per-agent test: Mann–Whitney U (two-sided) comparing numeric vs CoT.
  * Report: n per group, U, p_value, rank-biserial effect (from U), Wasserstein distance (W) and
    optional permutation p-value for W by shuffling prompt labels within task strata.
  * Multiple testing: BH-FDR within each experiment across agents; also a global FDR across
    all (experiment, agent) tests.

- Across-experiments omnibus: For each agent and prompt category (numeric and CoT separately),
  Kruskal–Wallis across experiments where the agent has data (with min_n per group filter).
  * Multiple testing: BH-FDR within each prompt category across agents; and optional global.

Notes
-----
- By default, humans are excluded (focus on agent behavior). Use --include-humans to include.
- We normalize prompt_category labels to lowercase and keep only {"numeric", "cot"} by default.
- We create an agent_variant label analogous to scripts/domain_differences.py: GPT‑5 variants
  get suffixes -v_<verbosity>-r_<reasoning_effort> when present.

Outputs
-------
- results/prompt_category_differences/<experiment>/mwu_numeric_vs_cot_by_agent_v1.csv
- results/prompt_category_differences/summary_per_experiment_v1.csv
- results/prompt_category_differences/across_experiments_kw_by_agent_prompt_v1.csv

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, wasserstein_distance

try:
    from statsmodels.stats.multitest import multipletests as _sm_multipletests
except Exception:  # pragma: no cover - fallback if statsmodels missing
    _sm_multipletests = None


def _fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return BH-FDR adjusted p-values. Fallback to a minimal implementation if needed."""
    pvals = np.asarray(pvals, dtype=float)
    if _sm_multipletests is not None:
        _, p_adj, _, _ = _sm_multipletests(pvals, alpha=alpha, method="fdr_bh")
        return p_adj
    # Minimal BH fallback (monotone p-adjust)
    m = np.sum(np.isfinite(pvals))
    if m == 0:
        return pvals * np.nan
    idx = np.argsort(pvals)
    p_sorted = pvals[idx]
    adj = np.empty_like(p_sorted)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = p_sorted[i] * m / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.zeros_like(pvals)
    out[idx] = adj
    return out


def project_root() -> Path:
    # scripts/03_analysis_raw/<file>.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _ensure_agent_variant(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "verbosity" not in df.columns:
        df["verbosity"] = "n/a"
    if "reasoning_effort" not in df.columns:
        df["reasoning_effort"] = "n/a"
    def _norm(x: pd.Series) -> pd.Series:
        x = x.astype(str).str.strip().str.lower()
        x = x.replace({"": "n/a", "nan": "n/a", "none": "n/a"})
        return x
    df["verbosity"] = _norm(df["verbosity"])
    df["reasoning_effort"] = _norm(df["reasoning_effort"])
    subj = df["subject"].astype(str)
    is_gpt5 = subj.str.startswith("gpt-5")
    already_variant = subj.str.contains(r"-v_.*-r_.*", regex=True)
    has_meta = ~(
        df["verbosity"].isin(["n/a", "unspecified"]) & df["reasoning_effort"].isin(["n/a", "unspecified"])
    )
    df["agent_variant"] = subj
    mask = is_gpt5 & has_meta & ~already_variant
    df.loc[mask, "agent_variant"] = (
        df.loc[mask, "subject"].astype(str)
        + "-v_" + df.loc[mask, "verbosity"].astype(str)
        + "-r_" + df.loc[mask, "reasoning_effort"].astype(str)
    )
    return df


def _normalize_prompt_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "prompt_category" not in df.columns:
        df["prompt_category"] = "unspecified"
    df["prompt_category"] = df["prompt_category"].astype(str).str.strip().str.lower()
    # Normalize common variants
    df["prompt_category"] = df["prompt_category"].replace({
        "cot": "cot",
        "chain-of-thought": "cot",
        "chain_of_thought": "cot",
        "numeric": "numeric",
        "numeric-conf": "numeric-conf",
    })
    return df


def _load_experiment_raw(experiment: str, include_humans: bool) -> pd.DataFrame:
    """Load all available cleaned CSVs for an experiment under data/processed/llm_with_humans/rw17/<experiment>.

    Returns a concatenated DataFrame with normalized columns and agent_variant.
    """
    base = project_root() / "data" / "processed" / "llm_with_humans" / "rw17" / experiment
    if not base.exists():
        raise FileNotFoundError(f"Experiment folder not found: {base}")
    csvs = sorted(base.glob("*_cleaned_data*.csv"))
    # Also include optional humans files if present; keep in df then filter later if needed
    csvs += sorted(base.glob("*_humans*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No input CSVs found under {base}")
    frames: List[pd.DataFrame] = []
    for fp in csvs:
        try:
            fdf = pd.read_csv(fp)
            frames.append(fdf)
        except Exception as e:
            print(f"Warning: failed reading {fp}: {e}", file=sys.stderr)
    if not frames:
        raise RuntimeError(f"Failed to load any CSVs in {base}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    # Ensure required columns
    required = ["subject", "likelihood", "prompt_category"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in data for {experiment}")
    # Drop NaN likelihoods
    df = df[pd.to_numeric(df["likelihood"], errors="coerce").notna()].copy()
    df["likelihood"] = df["likelihood"].astype(float)
    # Normalize prompt categories and agent variants
    df = _normalize_prompt_category(df)
    df = _ensure_agent_variant(df)
    # Exclude humans unless requested
    if not include_humans and "subject" in df.columns:
        df = df[df["subject"].astype(str) != "humans"]
    return df


def _rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float((2.0 * u) / (n1 * n2) - 1.0)


def _wasserstein_perm_p(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray | None,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Return (W, permutation p-value) for Wasserstein distance, shuffling labels within strata.

    If strata is None, shuffle globally. Small-sample adjusted p = (1 + #>=) / (n_perm + 1).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0 or n_perm <= 0:
        return float(wasserstein_distance(x, y)) if x.size and y.size else np.nan, np.nan
    W_obs = float(wasserstein_distance(x, y))
    # Build pooled arrays
    vals = np.concatenate([x, y])
    labels = np.array([0] * x.size + [1] * y.size, dtype=int)
    if strata is None:
        strata = np.zeros_like(vals)
    else:
        # Expand strata to match pooled length: we expect caller to pass per-row strata for x and y
        pass
    # To support optional strata, we require caller to pass aligned arrays for strata_x and strata_y
    # We detect that by checking if strata has same length as vals; otherwise, disable stratification.
    if strata.shape[0] != vals.shape[0]:  # disable if misaligned
        strata = np.zeros_like(vals)
    unique_strata = np.unique(strata)
    count = 0
    for _ in range(n_perm):
        permuted_labels = np.empty_like(labels)
        for s in unique_strata:
            mask = strata == s
            idx = np.where(mask)[0]
            lab = labels[idx].copy()
            rng.shuffle(lab)
            permuted_labels[idx] = lab
        x_b = vals[permuted_labels == 0]
        y_b = vals[permuted_labels == 1]
        if x_b.size == 0 or y_b.size == 0:
            continue
        W_b = float(wasserstein_distance(x_b, y_b))
        if W_b >= W_obs:
            count += 1
    p = (1.0 + count) / (n_perm + 1.0)
    return W_obs, float(p)


@dataclass
class MWUResult:
    experiment: str
    agent_variant: str
    n_numeric: int
    n_cot: int
    U: float
    p_value: float
    effect_rb: float
    W: float
    ws_perm_p: float


def compute_within_experiment_tests(
    experiments: List[str],
    include_humans: bool,
    min_n: int,
    ws_permute: int,
    rng_seed: int | None,
    prompt_keep: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run numeric vs CoT MWU per agent within each experiment; return (per_row_df, summary_df)."""
    rng = np.random.default_rng(rng_seed) if ws_permute and rng_seed is not None else np.random.default_rng()
    all_rows: List[MWUResult] = []

    for exp in experiments:
        df = _load_experiment_raw(exp, include_humans=include_humans)
        df = df[df["prompt_category"].isin(prompt_keep)].copy()
        # Optional: normalize case of prompt_category already done; create per-agent tests
        for agent, g in sorted(df.groupby("agent_variant")):
            gnum = g[g["prompt_category"] == "numeric"]["likelihood"].astype(float).to_numpy()
            gcot = g[g["prompt_category"] == "cot"]["likelihood"].astype(float).to_numpy()
            if gnum.size < min_n or gcot.size < min_n:
                continue
            # MWU two-sided
            try:
                U, p = mannwhitneyu(gnum, gcot, alternative="two-sided", method="asymptotic")
            except Exception:
                # fall back to exact when small without ties
                U, p = mannwhitneyu(gnum, gcot, alternative="two-sided")
            rb = _rank_biserial_from_u(U, gnum.size, gcot.size)
            # Optional stratified permutation by task for W distance
            if ws_permute and "task" in g.columns:
                # Build aligned strata arrays for gnum and gcot
                s_num = g[g["prompt_category"] == "numeric"]["task"].astype(str).to_numpy()
                s_cot = g[g["prompt_category"] == "cot"]["task"].astype(str).to_numpy()
                strata = np.concatenate([s_num, s_cot])
                W, p_ws = _wasserstein_perm_p(gnum, gcot, strata=strata, n_perm=ws_permute, rng=rng)
            else:
                W = float(wasserstein_distance(gnum, gcot)) if gnum.size and gcot.size else np.nan
                p_ws = np.nan
            all_rows.append(MWUResult(str(exp), str(agent), int(gnum.size), int(gcot.size), float(U), float(p), float(rb), float(W), float(p_ws)))

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()
    df_rows = pd.DataFrame([r.__dict__ for r in all_rows])
    # BH within each experiment
    df_rows["p_fdr_within_experiment"] = np.nan
    df_rows["ws_p_fdr_within_experiment"] = np.nan
    for exp, g in df_rows.groupby("experiment"):
        if g["p_value"].notna().any():
            df_rows.loc[g.index, "p_fdr_within_experiment"] = _fdr_bh(g["p_value"].to_numpy())
        if g["ws_perm_p"].notna().any():
            df_rows.loc[g.index, "ws_p_fdr_within_experiment"] = _fdr_bh(g["ws_perm_p"].to_numpy())
    # Global BH across all (experiment, agent)
    if df_rows["p_value"].notna().any():
        df_rows["p_fdr_global"] = _fdr_bh(df_rows["p_value"].to_numpy())
    else:
        df_rows["p_fdr_global"] = np.nan
    if df_rows["ws_perm_p"].notna().any():
        df_rows["ws_p_fdr_global"] = _fdr_bh(df_rows["ws_perm_p"].to_numpy())
    else:
        df_rows["ws_p_fdr_global"] = np.nan

    # Per-experiment summary
    def _count_sig(series) -> int:
        return int(np.sum(pd.to_numeric(series, errors="coerce") < 0.05))
    summaries = []
    for exp, g in df_rows.groupby("experiment"):
        summaries.append({
            "experiment": exp,
            "n_tests": int(g.shape[0]),
            "n_sig_fdr_within_exp": _count_sig(g["p_fdr_within_experiment"]),
            "n_sig_fdr_global": _count_sig(g["p_fdr_global"]),
            "n_sig_ws_within_exp": _count_sig(g["ws_p_fdr_within_experiment"]),
            "n_sig_ws_global": _count_sig(g["ws_p_fdr_global"]),
        })
    summary_df = pd.DataFrame(summaries).sort_values("experiment")
    return df_rows, summary_df


def omnibus_across_experiments(
    experiments: List[str],
    include_humans: bool,
    min_n: int,
    prompt_keep: List[str],
) -> pd.DataFrame:
    """Kruskal–Wallis across experiments per (agent_variant, prompt_category)."""
    # Load all experiments and concatenate with a column
    frames = []
    for exp in experiments:
        df = _load_experiment_raw(exp, include_humans=include_humans)
        df = df[df["prompt_category"].isin(prompt_keep)].copy()
        df["experiment"] = exp
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for (agent, prompt), g in df_all.groupby(["agent_variant", "prompt_category"]):
        # Build groups per experiment
        groups = []
        exps_present = []
        for exp, gexp in g.groupby("experiment"):
            vals = gexp["likelihood"].astype(float).to_numpy()
            if vals.size >= min_n:
                groups.append(vals)
                exps_present.append(exp)
        if len(groups) < 2:
            continue
        try:
            stat, p = kruskal(*groups)
        except Exception:
            stat, p = np.nan, np.nan
        rows.append({
            "agent_variant": agent,
            "prompt_category": prompt,
            "n_groups": len(groups),
            "experiments": ",".join(sorted(exps_present)),
            "kw_stat": float(stat),
            "p_value": float(p) if np.isfinite(p) else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # BH-FDR within each prompt category across agents
    out["p_fdr_within_prompt"] = np.nan
    for prompt, g in out.groupby("prompt_category"):
        if g["p_value"].notna().any():
            out.loc[g.index, "p_fdr_within_prompt"] = _fdr_bh(g["p_value"].to_numpy())
    # Global across both prompts
    if out["p_value"].notna().any():
        out["p_fdr_global"] = _fdr_bh(out["p_value"].to_numpy())
    else:
        out["p_fdr_global"] = np.nan
    return out


def posthoc_pairwise_across_experiments(
    experiments: List[str],
    include_humans: bool,
    min_n: int,
    prompt_keep: List[str],
    kw_df: pd.DataFrame,
    alpha: float,
    ws_permute: int,
    rng_seed: int | None,
) -> pd.DataFrame:
    """For each (agent_variant, prompt_category) with significant KW (BH within prompt),
    run pairwise MWU tests between experiments with sufficient n.

    Returns a DataFrame with pairwise stats and BH-FDR within-family and global.
    """
    # Load all experiments once
    frames = []
    for exp in experiments:
        df = _load_experiment_raw(exp, include_humans=include_humans)
        df = df[df["prompt_category"].isin(prompt_keep)].copy()
        df["experiment"] = exp
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True, sort=False)

    if kw_df is None or kw_df.empty:
        return pd.DataFrame()

    # Determine which (agent, prompt) to test based on KW BH within prompt
    if "p_fdr_within_prompt" in kw_df.columns:
        sig_mask = pd.to_numeric(kw_df["p_fdr_within_prompt"], errors="coerce") < alpha
    else:
        sig_mask = pd.to_numeric(kw_df["p_value"], errors="coerce") < alpha
    kw_sig = kw_df[sig_mask].copy()
    if kw_sig.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    rows = []
    for _, row in kw_sig.iterrows():
        agent = row["agent_variant"]
        prompt = row["prompt_category"]
        # Collect groups per experiment for this (agent, prompt)
        g = df_all[(df_all["agent_variant"] == agent) & (df_all["prompt_category"] == prompt)]
        # map experiment -> values, tasks
        groups = {}
        tasks = {}
        for exp, gexp in g.groupby("experiment"):
            vals = gexp["likelihood"].astype(float).to_numpy()
            if vals.size >= min_n:
                groups[str(exp)] = vals
                if "task" in gexp.columns:
                    tasks[str(exp)] = gexp["task"].astype(str).to_numpy()
                else:
                    tasks[str(exp)] = np.array(["_" for _ in range(vals.size)])
        exps = sorted(groups.keys())
        # Need at least two experiments with sufficient n
        if len(exps) < 2:
            continue
        # All unordered pairs
        for i in range(len(exps)):
            for j in range(i + 1, len(exps)):
                ea, eb = exps[i], exps[j]
                xa, xb = groups[ea], groups[eb]
                if xa.size < min_n or xb.size < min_n:
                    continue
                # MWU two-sided
                try:
                    U, p = mannwhitneyu(xa, xb, alternative="two-sided", method="asymptotic")
                except Exception:
                    U, p = mannwhitneyu(xa, xb, alternative="two-sided")
                rb = _rank_biserial_from_u(U, xa.size, xb.size)
                # Wasserstein and optional permutation p within task strata
                s_a = tasks.get(ea, np.array(["_" for _ in range(xa.size)]))
                s_b = tasks.get(eb, np.array(["_" for _ in range(xb.size)]))
                if ws_permute and xa.size and xb.size:
                    strata = np.concatenate([s_a, s_b])
                    W, p_ws = _wasserstein_perm_p(xa, xb, strata=strata, n_perm=ws_permute, rng=rng)
                else:
                    W = float(wasserstein_distance(xa, xb)) if xa.size and xb.size else np.nan
                    p_ws = np.nan
                rows.append({
                    "agent_variant": str(agent),
                    "prompt_category": str(prompt),
                    "experiment_a": str(ea),
                    "experiment_b": str(eb),
                    "n_a": int(xa.size),
                    "n_b": int(xb.size),
                    "U": float(U),
                    "p_value": float(p),
                    "effect_rb": float(rb),
                    "W": float(W),
                    "ws_perm_p": float(p_ws) if np.isfinite(p_ws) else np.nan,
                })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # BH-FDR within family (agent, prompt)
    out["p_fdr_within_family"] = np.nan
    out["ws_p_fdr_within_family"] = np.nan
    for (agent, prompt), g in out.groupby(["agent_variant", "prompt_category"]):
        if g["p_value"].notna().any():
            out.loc[g.index, "p_fdr_within_family"] = _fdr_bh(g["p_value"].to_numpy())
        if g["ws_perm_p"].notna().any():
            out.loc[g.index, "ws_p_fdr_within_family"] = _fdr_bh(g["ws_perm_p"].to_numpy())
    # Global BH across all pairwise
    if out["p_value"].notna().any():
        out["p_fdr_global"] = _fdr_bh(out["p_value"].to_numpy())
    else:
        out["p_fdr_global"] = np.nan
    if out["ws_perm_p"].notna().any():
        out["ws_p_fdr_global"] = _fdr_bh(out["ws_perm_p"].to_numpy())
    else:
        out["ws_p_fdr_global"] = np.nan
    return out


def discover_experiments(base: Path) -> List[str]:
    """Discover experiment folder names under processed llm_with_humans/rw17.

    Returns folder names (e.g., 'abstract_overloaded_lorem_de').
    """
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prompt-category effects within experiments (numeric vs CoT)")
    ap.add_argument("--experiments", nargs="*", default=None, help="Experiments to analyze. Default: discover all.")
    ap.add_argument("--include-humans", action="store_true", help="Include human rows in analysis")
    ap.add_argument("--min-n", type=int, default=10, help="Minimum n per group for a test to run")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for reporting thresholds (used in summaries)")
    ap.add_argument("--ws-permute", type=int, default=0, help="If >0, run permutation test count for W distance within task strata")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for permutations")
    ap.add_argument("--prompts", default="numeric,cot", help="Comma-separated prompt categories to keep (normalized to lowercase)")
    ap.add_argument("--no-omnibus", action="store_true", help="Skip across-experiments omnibus")
    ap.add_argument("--posthoc", action="store_true", help="If set, run post-hoc pairwise across experiments where KW is significant")
    ap.add_argument("--posthoc-alpha", type=float, default=0.05, help="Alpha on KW BH-FDR (within prompt) to trigger pairwise")
    ap.add_argument("--outdir", default=None, help="Override output directory root")
    args = ap.parse_args(argv)

    root = project_root()
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = discover_experiments(root / "data" / "processed" / "llm_with_humans" / "rw17")
    if not experiments:
        print("No experiments found", file=sys.stderr)
        return 2
    prompt_keep = [p.strip().lower() for p in args.prompts.split(",") if p.strip()]
    out_root = Path(args.outdir) if args.outdir else (root / "results" / "prompt_category_differences")
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Within-experiment tests
    per_row_df, summary_df = compute_within_experiment_tests(
        experiments=experiments,
        include_humans=args.include_humans,
        min_n=args.min_n,
        ws_permute=args.ws_permute,
        rng_seed=args.seed,
        prompt_keep=prompt_keep,
    )
    if not per_row_df.empty:
        for exp, g in per_row_df.groupby("experiment"):
            exp_dir = out_root / str(exp)
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "mwu_numeric_vs_cot_by_agent_v1.csv").write_text(g.to_csv(index=False))
        (out_root / "summary_per_experiment_v1.csv").write_text(summary_df.to_csv(index=False))
        print(f"Saved within-experiment results under {out_root}")
    else:
        print("No eligible within-experiment tests (check min_n and available prompts)")

    # 2) Across-experiments omnibus per agent/prompt (optional)
    if not args.no_omnibus:
        kw_df = omnibus_across_experiments(
            experiments=experiments,
            include_humans=args.include_humans,
            min_n=args.min_n,
            prompt_keep=prompt_keep,
        )
        if not kw_df.empty:
            (out_root / "across_experiments_kw_by_agent_prompt_v1.csv").write_text(kw_df.to_csv(index=False))
            print(f"Saved across-experiments omnibus to {out_root}")
            # Optional post-hoc pairwise if KW significant
            if args.posthoc:
                ph_df = posthoc_pairwise_across_experiments(
                    experiments=experiments,
                    include_humans=args.include_humans,
                    min_n=args.min_n,
                    prompt_keep=prompt_keep,
                    kw_df=kw_df,
                    alpha=args.posthoc_alpha,
                    ws_permute=args.ws_permute,
                    rng_seed=args.seed,
                )
                if not ph_df.empty:
                    (out_root / "across_experiments_pairwise_by_agent_prompt_v1.csv").write_text(ph_df.to_csv(index=False))
                    print(f"Saved post-hoc pairwise across experiments to {out_root}")
                else:
                    print("No eligible post-hoc pairs (KW not significant or insufficient n)")
        else:
            print("No eligible across-experiments omnibus tests (insufficient groups)")

    # Brief console summary
    if not per_row_df.empty:
        alpha = args.alpha
        hits = per_row_df[per_row_df["p_fdr_within_experiment"] < alpha]
        print(f"Significant (BH within experiment, alpha={alpha}): {hits.shape[0]} / {per_row_df.shape[0]} tests")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
