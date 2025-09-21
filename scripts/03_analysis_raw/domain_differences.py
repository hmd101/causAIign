#!/usr/bin/env python3
"""
Domain differences within and between agents (numeric prompts).

Purpose
-------
This module quantifies how likelihood values (0–100) differ across domains and agents.
It runs nonparametric omnibus and pairwise tests, optional distribution distances,
and produces ECDF overlays. Results are saved to results/domain_differences/<experiment>/<prompt_category>.

Data model
----------
- Response variable: likelihood in [0, 100].
- Grouping factors:
    - agent_variant: agent label (e.g., subject optionally augmented with verbosity/reasoning for GPT-5 variants).
    - domain: RW17 domain/task family.
- Humans are excluded unless explicitly requested via --include-humans.
- Optional replicate handling: when multiple completions per prompt id exist, collapse to a single value
    per (agent_variant, domain, id) to avoid pseudo-replication (see --collapse-replicates).

Analyses and hypotheses
-----------------------
Within-agent across domains (numeric prompts by default)
    1) Kruskal–Wallis omnibus across domains for each agent.
         - H0: All domain distributions are identical (same distribution).
         - H1: At least one domain distribution differs.
         - Multiple comparisons: p-values are BH–FDR corrected across agents (p_fdr_bh_across_agents).

    2) Pairwise Mann–Whitney U (two-sided) for every domain pair within each agent.
         - H0: The two domain distributions are identical (no stochastic dominance/location shift).
         - H1: The distributions differ (either direction).
         - Effect size: rank-biserial r = 2U/(n1*n2) − 1 in [−1, 1] (positive: first domain tends higher).
         - Multiple comparisons: BH–FDR within each agent over its domain pairs (p_fdr_bh_within_agent).

    3) Optional distance metric (1-Wasserstein / Earth Mover’s Distance) per pair.
         - Measures overall distributional separation in original units; we also report a normalized version
             W_norm in [0, 1] by dividing by the measurement range (default [0, 100]).
         - Optional 95% bootstrap CIs for W and a one-sided permutation p-value (H0: label exchangeability; statistic: W).
         - Multiple comparisons for permutation p-values: BH–FDR within agent (ws_p_fdr_bh_within_agent).

Within-domain across agents
    1) Kruskal–Wallis omnibus across agents for each domain.
         - H0: All agent distributions are identical.
         - H1: At least one agent differs.
         - Multiple comparisons: BH–FDR across domains (p_fdr_bh_across_domains).

    2) Pairwise Mann–Whitney U (two-sided) for agent pairs within each domain.
         - H0: The two agent distributions are identical.
         - H1: They differ (two-sided).
         - Effect size: rank-biserial r as above.
         - Multiple comparisons: BH–FDR within each domain across agent pairs (p_fdr_bh_within_domain),
             plus a pooled global BH across all domains for these pairwise tests (p_fdr_bh_global_within_domain_stratum).

    3) Optional 1-Wasserstein distance, bootstrap CI, and one-sided permutation p-value, with BH–FDR within domain
         for the permutation p-values (ws_p_fdr_bh_within_domain).

Spike composition (optional)
    - Three-bin composition per group (low [0,5], mid (5,95), high [95,100]).
    - Pairwise Pearson 2×3 chi-square tests:
            * Within-agent across domains and within-domain across agents.
            * H0: The 3-bin compositions are identical between the two groups.
            * H1: Compositions differ (e.g., more zero/one inflation or directional skew).
        BH–FDR within each family is applied to the chi-square p-values.
    - Note: A task-stratified permutation approach is described in-code to avoid confounding from task mix,
        but is not executed here.

Multiple-comparison policy
--------------------------
- Omnibus KW p-values: BH–FDR across agents (within-agent analysis) or across domains (within-domain analysis).
- Pairwise MWU p-values: BH–FDR within the coherent family:
    * within-agent across that agent’s domain pairs, and
    * within-domain across that domain’s agent pairs.
- Global BH pooling across domains is added for within-domain pairwise MWU as a secondary adjusted column.
- When Wasserstein permutation p-values are computed, BH–FDR is applied within the same family.

Outputs
-------
- kw_across_domains_per_agent_vX.csv              (KW omnibus per agent; + BH across agents)
- pairwise_mwu_across_domains_per_agent_vX.csv   (pairwise MWU within agent; + BH within agent; optional W, CI, perm p, + BH)
- kw_across_agents_per_domain_vX.csv              (KW omnibus per domain; + BH across domains)
- pairwise_mwu_across_agents_per_domain_vX.csv   (pairwise MWU within domain; + BH within domain; + pooled global BH; optional W, CI, perm p, + BH)
- ecdf_by_domain_agent_*.pdf, ecdf_by_agent_domain_*.pdf (visual overlays)
- Optional spike composition summaries and chi-square pairwise p-values with BH–FDR.

Example
-------
python scripts/domain_differences.py \
    --version 1 --experiment rw17_overloaded_e \
    --compute-wasserstein --ws-seed 0 --ws-permute 10 \
    --compute-spike-composition --spike-pairwise-tests \
    --prompt_category numeric

Conversion to LaTeX tables is supported via scripts/export_domain_differences_tables.py.
"""

from __future__ import annotations

# Standard library imports
import argparse  # command-line argument parsing
import math  # numeric helpers (e.g., isnan)
from pathlib import Path  # filesystem path helpers
import re  # regex for splitting lists
from typing import Iterable, Tuple  # typing annotations

import matplotlib.pyplot as plt

# Third-party libs
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, wasserstein_distance
import seaborn as sns

# Try to import statsmodels' multipletests; if unavailable, provide a minimal fallback BH procedure.
try:
    from statsmodels.stats.multitest import (  # type: ignore
        multipletests as _sm_multipletests,
    )

    def fdr_bh(pvals: np.ndarray, alpha: float) -> np.ndarray:
        # Use statsmodels to return adjusted p-values only (Benjamini-Hochberg).
        return _sm_multipletests(pvals, alpha=alpha, method="fdr_bh")[1]
except Exception:
    # Fallback minimal BH-FDR: takes array-like p-values and returns adjusted p-values.
    def fdr_bh(pvals: np.ndarray, alpha: float) -> np.ndarray:  # noqa: ARG001 (alpha unused in fallback)
        arr = np.asarray(pvals, dtype=float)
        finite = np.isfinite(arr)
        out = np.full_like(arr, np.nan, dtype=float)
        if not finite.any():
            return out
        p = arr[finite]
        m = p.size
        # Order p-values and compute BH adjusted values
        order = np.argsort(p)
        ranked = p[order]
        bh = ranked * m / (np.arange(1, m + 1))
        # Ensure monotonicity: cumulative minimum from largest to smallest
        adj = np.minimum.accumulate(bh[::-1])[::-1]
        # Cap at 1
        adj = np.minimum(adj, 1.0)
        # Place adjusted values back in original order positions
        out_idx = np.where(finite)[0][order]
        out[out_idx] = adj
        return out


def _as_float(x) -> float:
    """Best-effort conversion to float; returns NaN on failure."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def project_root() -> Path:
    """
    Return the project root (two levels up from this file).
    Useful for constructing relative data/output paths when no explicit input/output is provided.
    """
    # File is under scripts/03_analysis_raw/, so repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def load_processed(args) -> pd.DataFrame:
    """
    Load the processed input CSV.

    If args.input_file is provided, load that directly. Otherwise, construct a path from
    the project structure using experiment/version/graph and load the cleaned CSV.

    After loading, normalize column names by stripping whitespace.
    """
    if args.input_file:
        df = pd.read_csv(args.input_file)
    else:
        base = project_root() / "data" / "processed" / "llm_with_humans" / "rw17" / args.experiment
        fn = f"{args.version}_v_{args.graph}_cleaned_data.csv"
        df = pd.read_csv(base / fn)
    # Normalize column names to simple stripped strings to avoid subtle name mismatches.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'verbosity' and 'reasoning_effort' columns exist and produce an 'agent_variant' label.

    Behavior:
    - If verbosity or reasoning_effort are missing, create them with 'n/a'.
    - Normalize the two metadata columns to lowercase stripped strings, mapping blanks/'nan' to 'n/a'.
    - For subjects that start with 'gpt-5' and have non-default metadata, append the metadata
      to the subject string to create an 'agent_variant' like 'gpt-5...-v_high-r_strong'.
    - Otherwise, agent_variant defaults to the subject string.
    """
    df = df.copy()
    if "verbosity" not in df.columns:
        df["verbosity"] = "n/a"
    if "reasoning_effort" not in df.columns:
        df["reasoning_effort"] = "n/a"
    for c in ["verbosity", "reasoning_effort"]:
        # Normalize strings, coerce missing-like tokens to 'n/a'
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"": "n/a", "nan": "n/a"})
    subj_str = df["subject"].astype(str)
    is_gpt5 = subj_str.str.startswith("gpt-5")  # flag GPT-5 subjects
    already_variant = subj_str.str.contains(r"-v_.*-r_.*", regex=True)  # already have variant suffix
    has_meta = ~(
        df["verbosity"].isin(["n/a", "unspecified"]) & df["reasoning_effort"].isin(["n/a", "unspecified"])
    )
    # Default agent_variant to subject
    df["agent_variant"] = df["subject"].astype(str)
    # For GPT-5 rows that have metadata and don't already include it in the subject, append it
    df.loc[is_gpt5 & has_meta & ~already_variant, "agent_variant"] = (
        df.loc[is_gpt5 & has_meta, "subject"].astype(str)
        + "-v_" + df.loc[is_gpt5 & has_meta, "verbosity"].astype(str)
        + "-r_" + df.loc[is_gpt5 & has_meta, "reasoning_effort"].astype(str)
    )
    return df


def ecdf(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical CDF (ECDF) for numeric array y.

    Returns a tuple (x_sorted, f_values) where f_values are the ECDF values at sorted x.
    Filters NaNs and returns empty arrays if no finite data.

    Provides a visual comparison of likelihood distributions.
    Highlights differences in distribution shapes and cumulative probabilities.

    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return np.array([]), np.array([])
    x = np.sort(y)
    n = x.size
    f = np.arange(1, n + 1) / n  # ECDF values 1/n, 2/n, ..., 1
    return x, f


def _pairwise(iterable: Iterable[str]) -> Iterable[Tuple[str, str]]:
    """
    Yield unordered unique pairs (i, j) for items in iterable with i < j ordering.

    Used to iterate through combinations for pairwise tests without duplication.
    """
    items = list(iterable)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def _rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    """
    Convert Mann-Whitney U statistic to a rank-biserial effect size in [-1, 1].

    Formula: r = 2*U / (n1*n2) - 1
    Returns NaN if group sizes are non-positive.
    """
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float((2.0 * u) / (n1 * n2) - 1.0)


def _wasserstein_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float],
    bootstrap: int = 0,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Compute 1-Wasserstein distance (Earth Mover's) between arrays x and y.

    Returns a dictionary with:
      - W: raw Wasserstein distance
      - W_norm: normalized distance divided by provided bounds span (hi - lo)
      - W_ci_low, W_ci_high: 95% CI percentiles for bootstrap replicates (if requested)
      - W_ci_low_norm, W_ci_high_norm: normalized CI endpoints

    If either sample is empty, returns dict with NaNs. Bootstrapping is done by resampling
    with replacement and computing Wasserstein on each replicate using provided RNG.
    """
    out = {
        "W": np.nan,
        "W_norm": np.nan,
        "W_ci_low": np.nan,
        "W_ci_high": np.nan,
        "W_ci_low_norm": np.nan,
        "W_ci_high_norm": np.nan,
    }
    if x.size == 0 or y.size == 0:
        return out
    # Compute observed Wasserstein distance
    W = float(wasserstein_distance(x, y))
    lo, hi = float(bounds[0]), float(bounds[1])
    rng_span = hi - lo if hi > lo else np.nan  # denominator for normalization
    out["W"] = W
    out["W_norm"] = (W / rng_span) if np.isfinite(rng_span) and rng_span > 0 else np.nan
    if bootstrap and bootstrap > 0:
        if rng is None:
            rng = np.random.default_rng()
        Ws = np.empty(bootstrap, dtype=float)
        # Bootstrap replicates by resampling each sample with replacement
        for b in range(bootstrap):
            xb = rng.choice(x, size=x.size, replace=True)
            yb = rng.choice(y, size=y.size, replace=True)
            Ws[b] = float(wasserstein_distance(xb, yb))
        # Compute 2.5 and 97.5 percentiles as a simple 95% CI
        out["W_ci_low"] = float(np.percentile(Ws, 2.5))
        out["W_ci_high"] = float(np.percentile(Ws, 97.5))
        if np.isfinite(rng_span) and rng_span > 0:
            out["W_ci_low_norm"] = out["W_ci_low"] / rng_span
            out["W_ci_high_norm"] = out["W_ci_high"] / rng_span
    return out


def _wasserstein_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """
    Compute a permutation p-value for the Wasserstein distance (one-sided greater-than test).

    Procedure:
      - Compute observed W_obs between x and y.
      - Pool all values, shuffle, split into two groups of original sizes, compute W_b.
      - Count how many permuted W_b >= W_obs.
      - Return small-sample-adjusted p = (1 + count) / (n_perm + 1).

    Returns NaN if inputs are invalid.
    """
    if x.size == 0 or y.size == 0 or n_perm <= 0:
        return np.nan
    W_obs = float(wasserstein_distance(x, y))
    all_vals = np.concatenate([x, y])
    n1 = x.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(all_vals)  # in-place shuffle of pooled values
        xb = all_vals[:n1]
        yb = all_vals[n1:]
        W_b = float(wasserstein_distance(xb, yb))
        if W_b >= W_obs:
            count += 1
    p = (1.0 + count) / (n_perm + 1.0)  # add-one correction for permutation tests
    return float(p)


def _spike_counts(y: np.ndarray, low: float = 5.0, high: float = 95.0) -> tuple[int, int, int]:
    """
    Count frequencies of values in three bins (spike composition):
      - low bin: [0, low] (e.g., 0-5)
      - mid bin: (low, high)
      - high bin: [high, 100] (e.g., 95-100)

    NaNs and non-finite values are removed before counting. Returns counts (low, mid, high).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    c_low = int(np.sum((y >= 0.0) & (y <= low)))
    c_mid = int(np.sum((y > low) & (y < high)))
    c_high = int(np.sum((y >= high) & (y <= 100.0)))
    return c_low, c_mid, c_high


def analyze_within_agent_across_domains(
    df: pd.DataFrame,
    alpha: float,
    min_n: int,
    include_humans: bool,
    *,
    compute_wasserstein: bool = False,
    ws_bootstrap: int = 0,
    ws_permute: int = 0,
    bounds: tuple[float, float] = (0.0, 100.0),
    rng: np.random.Generator | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each non-human agent (agent_variant), perform the within-agent analysis across domains.

    Hypotheses and interpretation
    ----------------------------
    - Kruskal–Wallis omnibus (per agent):
        H0: All domain distributions are identical; H1: At least one differs.
        Detects any distributional difference (location and/or shape).
        We apply BH–FDR across agents to these omnibus p-values.
    - Pairwise Mann–Whitney U (two-sided) for every domain pair:
        H0: The two domain distributions are identical; H1: They differ.
        Sensitive primarily to location (median/rank) differences, but as a rank test,
        can reflect broader distributional changes. Rank-biserial r summarizes effect direction/magnitude.
        We apply BH–FDR within each agent across its domain pairs.
    - Optional 1-Wasserstein distance and permutation p-value (one-sided, greater-than):
        Measures overall distributional separation; permutation H0 is label exchangeability.
        BH–FDR is applied to permutation p-values within agent when requested.

    Returns
    -------
    - kw_df: Kruskal–Wallis omnibus results per agent (+ BH across agents).
    - pair_df: Pairwise MWU (and optional Wasserstein) per domain pair (+ BH within agent).
    """
    results_kw = []
    results_pair = []
    # Consider agents for within-agent analyses; optionally include humans
    if include_humans:
        agents = df["agent_variant"].dropna().astype(str).unique().tolist()
    else:
        agents = (
            df.loc[df["subject"] != "humans", "agent_variant"].dropna().astype(str).unique().tolist()
        )
    for agent in sorted(agents):
        sub = df[df["agent_variant"] == agent]
        # Group by domain and extract numeric 'likelihood' arrays
        groups = {d: g["likelihood"].dropna().to_numpy(float) for d, g in sub.groupby("domain")}
        # Keep only groups with at least min_n samples
        groups = {d: v for d, v in groups.items() if v.size >= min_n}
        if len(groups) < 2:
            # Not enough domains with sufficient samples to compare
            continue
        # Kruskal–Wallis omnibus across the domain groups for this agent
        try:
            stat, p = kruskal(*groups.values())
        except Exception:
            stat, p = (np.nan, np.nan)
        results_kw.append({
            "agent": agent,
            "k": int(len(groups)),
            "H": float(stat) if stat is not None else np.nan,
            "df": int(max(0, len(groups) - 1)),
            "p_value": float(p) if p is not None else np.nan,
            "n_total": int(sum(len(v) for v in groups.values())),
        })
        # Prepare pairwise MWU tests within this agent across domains (always run; not gated on KW)
        pairs = list(_pairwise(sorted(groups.keys())))
        pvals = []
        tmp_rows = []
        ws_pvals_for_fdr: list[float] = []
        for a, b in pairs:
            x, y = groups[a], groups[b]
            try:
                u_stat, p_mwu = mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                u_stat, p_mwu = (np.nan, np.nan)
            # Convert U to rank-biserial effect size
            effect = _rank_biserial_from_u(u_stat if not math.isnan(u_stat) else np.nan, len(x), len(y))
            # Prepare Wasserstein placeholders and optionally compute
            W_dict = {k: np.nan for k in ["W", "W_norm", "W_ci_low", "W_ci_high", "W_ci_low_norm", "W_ci_high_norm"]}
            ws_p = np.nan
            if compute_wasserstein:
                # Compute Wasserstein and optionally bootstrap CIs
                W_dict = _wasserstein_with_ci(x, y, bounds=bounds, bootstrap=ws_bootstrap, rng=rng)
                # Optionally compute permutation p-value for Wasserstein if RNG provided
                if ws_permute and rng is not None:
                    try:
                        ws_p = _wasserstein_permutation_p(x, y, n_perm=ws_permute, rng=rng)
                    except Exception:
                        ws_p = np.nan
            # Collect per-pair results
            tmp_rows.append({
                "agent": agent,
                "domain_a": a,
                "domain_b": b,
                "n_a": int(len(x)),
                "n_b": int(len(y)),
                "U": float(u_stat) if u_stat is not None else np.nan,
                "p_value": float(p_mwu) if p_mwu is not None else np.nan,
                "effect_rb": float(effect) if effect is not None else np.nan,
                "wasserstein": float(W_dict["W"]) if np.isfinite(W_dict["W"]) else np.nan,
                "wasserstein_norm": float(W_dict["W_norm"]) if np.isfinite(W_dict["W_norm"]) else np.nan,
                "wasserstein_ci_low": float(W_dict["W_ci_low"]) if np.isfinite(W_dict["W_ci_low"]) else np.nan,
                "wasserstein_ci_high": float(W_dict["W_ci_high"]) if np.isfinite(W_dict["W_ci_high"]) else np.nan,
                "wasserstein_ci_low_norm": float(W_dict["W_ci_low_norm"]) if np.isfinite(W_dict["W_ci_low_norm"]) else np.nan,
                "wasserstein_ci_high_norm": float(W_dict["W_ci_high_norm"]) if np.isfinite(W_dict["W_ci_high_norm"]) else np.nan,
                "ws_p_value": float(ws_p) if np.isfinite(ws_p) else np.nan,
            })
            # Collect MWU p-values for FDR; could contain NaNs
            pvals.append(tmp_rows[-1]["p_value"])
            if np.isfinite(ws_p):
                ws_pvals_for_fdr.append(ws_p)
        # Apply BH–FDR correction to finite MWU p-values within this agent (family: all pairs for this agent)
        finite_mask = np.isfinite(pvals)
        if finite_mask.any():
            p_adj = fdr_bh(np.array(pvals)[finite_mask], alpha=alpha)
            adj_iter = iter(p_adj)
            for row in tmp_rows:
                if np.isfinite(row["p_value"]):
                    # Assign the adjusted p-value in the same order as original finite p-values
                    row["p_fdr_bh_within_agent"] = float(next(adj_iter))
                else:
                    row["p_fdr_bh_within_agent"] = np.nan
        else:
            for row in tmp_rows:
                row["p_fdr_bh_within_agent"] = np.nan
        # If Wasserstein permutation p-values were computed, BH–FDR-correct them within the same family
        if compute_wasserstein and ws_permute and len(tmp_rows) > 0:
            wp = np.array([r.get("ws_p_value", np.nan) for r in tmp_rows], dtype=float)
            finite_mask_ws = np.isfinite(wp)
            if finite_mask_ws.any():
                p_adj_ws = fdr_bh(wp[finite_mask_ws], alpha=alpha)
                it = iter(p_adj_ws)
                for r in tmp_rows:
                    if np.isfinite(r.get("ws_p_value", np.nan)):
                        r["ws_p_fdr_bh_within_agent"] = float(next(it))
                    else:
                        r["ws_p_fdr_bh_within_agent"] = np.nan
            else:
                for r in tmp_rows:
                    r["ws_p_fdr_bh_within_agent"] = np.nan
        # Append pairwise rows for this agent to the global list
        results_pair.extend(tmp_rows)

    # Create DataFrame for KW results and apply BH–FDR across agents for the omnibus p-values
    kw_df = pd.DataFrame(results_kw)
    if not kw_df.empty and kw_df["p_value"].notna().any():
        mask = kw_df["p_value"].notna()
        kw_df.loc[mask, "p_fdr_bh_across_agents"] = fdr_bh(
            np.asarray(kw_df.loc[mask, "p_value"].values, dtype=float), alpha=alpha
        )
    else:
        kw_df["p_fdr_bh_across_agents"] = np.nan
    pair_df = pd.DataFrame(results_pair)
    return kw_df, pair_df


def analyze_within_domain_across_agents(
    df: pd.DataFrame,
    alpha: float,
    min_n: int,
    include_humans: bool,
    *,
    compute_wasserstein: bool = False,
    ws_bootstrap: int = 0,
    ws_permute: int = 0,
    bounds: tuple[float, float] = (0.0, 100.0),
    rng: np.random.Generator | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each domain, perform the within-domain analysis across agent_variants.

    Hypotheses and interpretation
    ----------------------------
    - Kruskal–Wallis omnibus (per domain):
        H0: All agent distributions are identical; H1: At least one differs.
        BH–FDR is applied across domains for these omnibus p-values.
    - Pairwise Mann–Whitney U (two-sided) for every agent pair within the domain:
        H0: The two agent distributions are identical; H1: They differ.
        Rank-biserial r summarizes effect direction/magnitude.
        BH–FDR is applied within the domain across its agent pairs; we also report a pooled global BH
        across all domains as an additional adjusted column for these pairwise tests.
    - Optional Wasserstein, bootstrap CI, and one-sided permutation p-values; BH–FDR applied within domain
      for permutation p-values when requested.

    Returns
    -------
    - kw_df: Kruskal–Wallis omnibus results per domain (+ BH across domains).
    - pair_df: Pairwise MWU (and optional Wasserstein) per agent pair (+ BH within domain; + pooled global BH column later).
    """
    results_kw = []
    results_pair = []
    for domain, g_dom in sorted(df.groupby("domain")):
        # Build group for domain, excluding humans if requested
        if include_humans:
            g = g_dom.copy()
        else:
            g = g_dom[g_dom["subject"] != "humans"].copy()
        # Map agent_variant -> numeric likelihood arrays
        groups = {a: gsub["likelihood"].dropna().to_numpy(float) for a, gsub in g.groupby("agent_variant")}
        # Keep only sufficiently large groups
        groups = {a: v for a, v in groups.items() if v.size >= min_n}
        if len(groups) < 2:
            continue
        # Kruskal–Wallis omnibus across agents for this domain
        try:
            stat, p = kruskal(*groups.values())
        except Exception:
            stat, p = (np.nan, np.nan)
        results_kw.append({
            "domain": domain,
            "k": int(len(groups)),
            "H": float(stat) if stat is not None else np.nan,
            "df": int(max(0, len(groups) - 1)),
            "p_value": float(p) if p is not None else np.nan,
            "n_total": int(sum(len(v) for v in groups.values())),
        })
        # For pairwise tests, sort keys by string to ensure deterministic ordering
        names = sorted((str(k) for k in groups.keys()))
        pairs = list(_pairwise(names))
        pvals = []
        tmp_rows = []
        for a, b in pairs:
            x, y = groups[a], groups[b]
            try:
                u_stat, p_mwu = mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                u_stat, p_mwu = (np.nan, np.nan)
            effect = _rank_biserial_from_u(u_stat if not math.isnan(u_stat) else np.nan, len(x), len(y))
            W_dict = {k: np.nan for k in ["W", "W_norm", "W_ci_low", "W_ci_high", "W_ci_low_norm", "W_ci_high_norm"]}
            ws_p = np.nan
            if compute_wasserstein:
                W_dict = _wasserstein_with_ci(x, y, bounds=bounds, bootstrap=ws_bootstrap, rng=rng)
                if ws_permute and rng is not None:
                    try:
                        ws_p = _wasserstein_permutation_p(x, y, n_perm=ws_permute, rng=rng)
                    except Exception:
                        ws_p = np.nan
            tmp_rows.append({
                "domain": domain,
                "agent_a": a,
                "agent_b": b,
                "n_a": int(len(x)),
                "n_b": int(len(y)),
                "U": float(u_stat) if u_stat is not None else np.nan,
                "p_value": float(p_mwu) if p_mwu is not None else np.nan,
                "effect_rb": float(effect) if effect is not None else np.nan,
                "wasserstein": float(W_dict["W"]) if np.isfinite(W_dict["W"]) else np.nan,
                "wasserstein_norm": float(W_dict["W_norm"]) if np.isfinite(W_dict["W_norm"]) else np.nan,
                "wasserstein_ci_low": float(W_dict["W_ci_low"]) if np.isfinite(W_dict["W_ci_low"]) else np.nan,
                "wasserstein_ci_high": float(W_dict["W_ci_high"]) if np.isfinite(W_dict["W_ci_high"]) else np.nan,
                "wasserstein_ci_low_norm": float(W_dict["W_ci_low_norm"]) if np.isfinite(W_dict["W_ci_low_norm"]) else np.nan,
                "wasserstein_ci_high_norm": float(W_dict["W_ci_high_norm"]) if np.isfinite(W_dict["W_ci_high_norm"]) else np.nan,
                "ws_p_value": float(ws_p) if np.isfinite(ws_p) else np.nan,
            })
            pvals.append(tmp_rows[-1]["p_value"])
        # BH–FDR correction within this domain for the MWU pairwise p-values (family: all pairs for this domain)
        finite_mask = np.isfinite(pvals)
        if finite_mask.any():
            p_adj = fdr_bh(np.array(pvals)[finite_mask], alpha=alpha)
            adj_iter = iter(p_adj)
            for row in tmp_rows:
                if np.isfinite(row["p_value"]):
                    row["p_fdr_bh_within_domain"] = float(next(adj_iter))
                else:
                    row["p_fdr_bh_within_domain"] = np.nan
        else:
            for row in tmp_rows:
                row["p_fdr_bh_within_domain"] = np.nan
        # If Wasserstein permutation p-values were computed, BH–FDR-correct them within the same family
        if compute_wasserstein and ws_permute and len(tmp_rows) > 0:
            wp = np.array([r.get("ws_p_value", np.nan) for r in tmp_rows], dtype=float)
            finite_mask_ws = np.isfinite(wp)
            if finite_mask_ws.any():
                p_adj_ws = fdr_bh(wp[finite_mask_ws], alpha=alpha)
                it = iter(p_adj_ws)
                for r in tmp_rows:
                    if np.isfinite(r.get("ws_p_value", np.nan)):
                        r["ws_p_fdr_bh_within_domain"] = float(next(it))
                    else:
                        r["ws_p_fdr_bh_within_domain"] = np.nan
            else:
                for r in tmp_rows:
                    r["ws_p_fdr_bh_within_domain"] = np.nan
        results_pair.extend(tmp_rows)

    # Create KW DataFrame and apply BH–FDR across domains for omnibus results
    kw_df = pd.DataFrame(results_kw)
    if not kw_df.empty and kw_df["p_value"].notna().any():
        mask = kw_df["p_value"].notna()
        kw_df.loc[mask, "p_fdr_bh_across_domains"] = fdr_bh(
            np.asarray(kw_df.loc[mask, "p_value"].values, dtype=float), alpha=alpha
        )
    else:
        kw_df["p_fdr_bh_across_domains"] = np.nan
    pair_df = pd.DataFrame(results_pair)
    return kw_df, pair_df


def _summarize_spike_composition(df: pd.DataFrame, include_humans: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize three-bin spike composition for non-human agents by (agent, domain) and by (domain, agent).

    For each group, compute:
      - n: total counted values (low+mid+high)
      - proportions p_low_0_5, p_mid_5_95, p_high_95_100
      - zero_one_inflation: proportion in low or high bins combined
      - directionality_high_minus_low: (high - low) / n, ranges [-1, 1]

    Returns a tuple:
      - by_agent_domain: rows keyed by (agent, domain)
      - by_domain_agent: rows keyed by (domain, agent)
    """
    rows_ad = []
    df_within_agent = df if include_humans else df[df["subject"] != "humans"]
    for (agent, domain), g in df_within_agent.groupby(["agent_variant", "domain"]):
        y = g["likelihood"].dropna().to_numpy(float)
        c_low, c_mid, c_high = _spike_counts(y)
        n = int(c_low + c_mid + c_high)
        if n == 0:
            continue
        rows_ad.append({
            "agent": str(agent),
            "domain": str(domain),
            "n": n,
            "p_low_0_5": c_low / n,
            "p_mid_5_95": c_mid / n,
            "p_high_95_100": c_high / n,
            "zero_one_inflation": (c_low + c_high) / n,
            "directionality_high_minus_low": (c_high - c_low) / n,
        })
    by_agent_domain = pd.DataFrame(rows_ad)

    rows_da = []
    for (domain, agent), g in df.groupby(["domain", "agent_variant"]):
        y = g["likelihood"].dropna().to_numpy(float)
        c_low, c_mid, c_high = _spike_counts(y)
        n = int(c_low + c_mid + c_high)
        if n == 0:
            continue
        rows_da.append({
            "domain": str(domain),
            "agent": str(agent),
            "n": n,
            "p_low_0_5": c_low / n,
            "p_mid_5_95": c_mid / n,
            "p_high_95_100": c_high / n,
            "zero_one_inflation": (c_low + c_high) / n,
            "directionality_high_minus_low": (c_high - c_low) / n,
        })
    by_domain_agent = pd.DataFrame(rows_da)
    return by_agent_domain, by_domain_agent


def _pairwise_spike_tests(df: pd.DataFrame, alpha: float, include_humans: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform pairwise chi-square tests on 3-bin spike composition.

    For each agent, compare domains pairwise; for each domain, compare agents pairwise.
    Apply BH-FDR correction within each group of pairwise tests (within-agent across domains,
    and within-domain across agents).

    Returns two DataFrames:
      - within_agent_across_domains (rows with agent, domain_a, domain_b, chi2, df, p_value, p_fdr_bh_within_agent_spike)
      - within_domain_across_agents (rows with domain, agent_a, agent_b, chi2, df, p_value, p_fdr_bh_within_domain_spike)

     Note on permutation-based inference for spike composition (conceptual description):
     Some RW17 tasks are expected to induce responses at or near the bounds even under
     normative behavior (e.g., Task III near 100; Task I near 0; Tasks IX-XI typically
     closer to 0 than 50). A naïve (unstratified) permutation of labels between two
     groups can therefore overstate differences that are actually driven by differing
     task mixes, not true domain/agent effects.

     To address this, a task-stratified permutation test can be used. The idea is to
     preserve the task composition while testing whether the 3-bin spike proportions
     differ between the two groups:
        1) Choose a two-sample comparison (e.g., domain A vs domain B for a fixed agent).
        2) For each task stratum, pool the two groups' observations for that task only,
            then randomly reassign labels back to the two groups within that stratum,
            keeping the original per-task group sizes fixed.
        3) After relabeling within all strata, compute the 2x3 contingency table of
            [low, mid, high] counts for the permuted groups and a test statistic such as
            Pearson's chi-square.
        4) Repeat for N permutations and form a small-sample adjusted p-value
            p = (1 + #{T_perm >= T_obs}) / (N + 1).
        5) Apply BH-FDR to the set of permutation p-values within each coherent family
            (within-agent across domains; within-domain across agents).

     This conditional permutation respects task-specific expected spikes and isolates
     whether group labeling explains additional composition differences beyond tasks.
     The same approach can be implemented by stratifying on another column if desired
     (e.g., counterbalance cell), provided each stratum has observations in both groups.
    """
    rows_agent = []
    df_within_agent = df if include_humans else df[df["subject"] != "humans"]
    for agent, g in df_within_agent.groupby("agent_variant"):
        # Build 3-bin counts per domain for this agent
        groups = {d: gsub["likelihood"].dropna().to_numpy(float) for d, gsub in g.groupby("domain")}
        groups = {d: v for d, v in groups.items() if v.size > 0}
        if len(groups) < 2:
            continue
        names = sorted((str(k) for k in groups.keys()))
        pairs = list(_pairwise(names))
        pvals = []
        tmp = []
        for a, b in pairs:
            ca = _spike_counts(groups[a])
            cb = _spike_counts(groups[b])
            table = np.array([ca, cb])  # contingency table 2x3
            try:
                chi2_val, p_val, dof_val, _ = chi2_contingency(table)
            except Exception:
                chi2_val, p_val, dof_val = (np.nan, np.nan, 2)
            chi2_f = _as_float(chi2_val)
            p_f = _as_float(p_val)
            tmp.append({
                "agent": str(agent),
                "domain_a": str(a),
                "domain_b": str(b),
                "chi2": chi2_f if math.isfinite(chi2_f) else np.nan,
                "df": int(dof_val) if isinstance(dof_val, (int, np.integer)) else 2,
                "p_value": p_f if math.isfinite(p_f) else np.nan,
            })
            pvals.append(tmp[-1]["p_value"])
        # BH-FDR on the finite p-values within this agent
        finite = np.isfinite(pvals)
        if np.any(finite):
            padj = fdr_bh(np.array(pvals)[finite], alpha=alpha)
            it = iter(padj)
            for r in tmp:
                if np.isfinite(r["p_value"]):
                    r["p_fdr_bh_within_agent_spike"] = float(next(it))
                else:
                    r["p_fdr_bh_within_agent_spike"] = np.nan
        else:
            for r in tmp:
                r["p_fdr_bh_within_agent_spike"] = np.nan
        rows_agent.extend(tmp)

    rows_domain = []
    for domain, g in df.groupby("domain"):
        groups = {a: gsub["likelihood"].dropna().to_numpy(float) for a, gsub in g.groupby("agent_variant")}
        groups = {a: v for a, v in groups.items() if v.size > 0}
        if len(groups) < 2:
            continue
        names = sorted((str(k) for k in groups.keys()))
        pairs = list(_pairwise(names))
        pvals = []
        tmp = []
        for a, b in pairs:
            ca = _spike_counts(groups[a])
            cb = _spike_counts(groups[b])
            table = np.array([ca, cb])
            try:
                chi2_val, p_val, dof_val, _ = chi2_contingency(table)
            except Exception:
                chi2_val, p_val, dof_val = (np.nan, np.nan, 2)
            chi2_f = _as_float(chi2_val)
            p_f = _as_float(p_val)
            tmp.append({
                "domain": str(domain),
                "agent_a": str(a),
                "agent_b": str(b),
                "chi2": chi2_f if math.isfinite(chi2_f) else np.nan,
                "df": int(dof_val) if isinstance(dof_val, (int, np.integer)) else 2,
                "p_value": p_f if math.isfinite(p_f) else np.nan,
            })
            pvals.append(tmp[-1]["p_value"])
        finite = np.isfinite(pvals)
        if np.any(finite):
            padj = fdr_bh(np.array(pvals)[finite], alpha=alpha)
            it = iter(padj)
            for r in tmp:
                if np.isfinite(r["p_value"]):
                    r["p_fdr_bh_within_domain_spike"] = float(next(it))
                else:
                    r["p_fdr_bh_within_domain_spike"] = np.nan
        else:
            for r in tmp:
                r["p_fdr_bh_within_domain_spike"] = np.nan
        rows_domain.extend(tmp)

    return pd.DataFrame(rows_agent), pd.DataFrame(rows_domain)


def plot_ecdf_overlays(
    df: pd.DataFrame,
    out_dir: Path,
    max_agents_per_domain: int | None = None,
    include_humans: bool = False,
) -> None:
    """
    Produce ECDF overlay plots:
      - For each non-human agent: plot ECDF curves for each domain the agent experienced.
      - For each domain: plot ECDF curves for each agent (optionally cap number of agents for readability).

    Saves PDF files to out_dir with deterministic file names (slashing replaced).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-agent ECDF by domain (optionally include humans)
    df_agents = df if include_humans else df[df["subject"] != "humans"]
    for agent, g in df_agents.groupby("agent_variant"):
        plt.figure(figsize=(6, 4))
        domains = sorted(g["domain"].astype(str).unique())
        palette = sns.color_palette(n_colors=len(domains))
        colors = {d: c for d, c in zip(domains, palette)}
        # For each domain, compute ECDF and plot a step function
        for d, gdom in g.groupby("domain"):
            x, f = ecdf(gdom["likelihood"].to_numpy(float))
            if x.size == 0:
                continue
            plt.step(x, f, where="post", label=str(d), color=colors[str(d)])
        plt.xlabel("likelihood")
        plt.ylabel("ECDF")
        plt.title(f"ECDF by domain — {agent}")
        plt.legend(title="domain", loc="best")
        plt.tight_layout()
        # Save with agent string safe for filenames (replace '/' with '-')
        plt.savefig(out_dir / f"ecdf_by_domain_agent_{str(agent).replace('/', '-')}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # Per-domain ECDF by agent (optionally cap number of agents for readability)
    for domain, g in df.groupby("domain"):
        agents = sorted(g["agent_variant"].astype(str).unique())
        if max_agents_per_domain is not None and len(agents) > max_agents_per_domain:
            # Truncate agent list for the plot to avoid legend clutter
            agents = agents[:max_agents_per_domain]
        plt.figure(figsize=(7, 5))
        palette = sns.color_palette(n_colors=len(agents))
        colors = {a: c for a, c in zip(agents, palette)}
        for a in agents:
            x, f = ecdf(g[g["agent_variant"] == a]["likelihood"].to_numpy(float))
            if x.size == 0:
                continue
            plt.step(x, f, where="post", label=str(a), color=colors[a], alpha=0.9)
        plt.xlabel("likelihood")
        plt.ylabel("ECDF")
        plt.title(f"ECDF by agent — {domain}")
        # Packaging legend with small font and multiple columns may help readability
        plt.legend(title="agent", loc="best", fontsize="small", ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"ecdf_by_agent_domain_{str(domain)}.pdf", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """
    Command-line entry point:
      - Parse arguments for input, filtering, analysis options (Wasserstein, spike composition), and outputs.
      - Load and normalize data, filter rows based on prompt_category/domains/agents provided.
      - Run within-agent and within-domain analyses, save resulting CSVs.
      - Produce ECDF overlay plots and optionally spike composition CSVs/tests.
      - Print saved outputs directory.
    """
    ap = argparse.ArgumentParser(
        description=(
            "domain differences per agent. "
            "Tip: use --list-agents to print available agent_variant labels given current filters."
        )
    )
    # Basic dataset and experiment metadata
    ap.add_argument("--version", default="2")
    ap.add_argument("--experiment", default="rw17_indep_causes")
    ap.add_argument("--graph", default="collider")
    ap.add_argument("--input-file")
    ap.add_argument("--output-dir", default="results/domain_differences")
    ap.add_argument("--prompt_category", default="numeric", help="Focus prompt_category (default: numeric)")
    ap.add_argument("--domains", default="all", help="Comma/space separated list or 'all'")
    ap.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        metavar="AGENT",
        help=(
            "One or more agent_variant labels (space-separated) or 'all'. "
            "Comma-separated in a single token also works. Run with --list-agents to see available labels."
        ),
    )
    ap.add_argument(
        "--list-agents",
        action="store_true",
        help=(
            "List available agent_variant labels (after applying --promptcategory and --domains filters). "
            "Respects --include-humans. Exits afterward."
        ),
    )
    ap.add_argument(
        "--include-humans",
        action="store_true",
        help="Include humans everywhere applicable (analyses and ECDFs) when set",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-n", type=int, default=8, help="Minimum observations per group to include in tests")
    ap.add_argument("--max-agents-per-domain", type=int, default=None, help="Cap ECDF legend clutter for per-domain plots")
    # Wasserstein options
    ap.add_argument("--compute-wasserstein", action="store_true", help="Compute 1-Wasserstein distances for each pairwise comparison")
    ap.add_argument("--ws-bootstrap", type=int, default=0, help="Bootstrap reps for Wasserstein CIs (0 to disable)")
    ap.add_argument("--ws-seed", type=int, default=0, help="Random seed for Wasserstein bootstrap")
    ap.add_argument("--bounds-lower", type=float, default=0.0, help="Lower bound for likelihood normalization (default 0)")
    ap.add_argument("--bounds-upper", type=float, default=100.0, help="Upper bound for likelihood normalization (default 100)")
    ap.add_argument("--ws-permute", type=int, default=0, help="Permutation reps for Wasserstein p-values (0 to disable)")
    # Spike composition options
    ap.add_argument("--compute-spike-composition", action="store_true", help="Compute 3-bin spike composition summaries and CSVs")
    ap.add_argument("--spike-pairwise-tests", action="store_true", help="Run pairwise chi-square tests on spike composition with BH-FDR")
    # Replicate handling
    ap.add_argument(
        "--collapse-replicates",
        choices=["none", "mean", "median", "first"],
        default="none",
        help=(
            "If multiple completions per prompt exist (replicates per id), collapse to one value per (agent_variant, domain, id). "
            "This prevents pseudo-replication and 10× inflation in n_total."
        ),
    )
    args = ap.parse_args()

    # Load and normalize the processed dataset
    df = load_processed(args)
    df = ensure_variant_columns(df)

    # Filter by prompt_category unless 'all' is specified
    # Note: Some datasets label human rows as 'single_numeric_response' while LLM rows are 'numeric'.
    # Treat these as synonyms so humans aren't dropped when requesting numeric prompts.
    if args.prompt_category and str(args.prompt_category).lower() != "all":
        spec = str(args.prompt_category).strip().lower()
        pc_norm = df["prompt_category"].astype(str).str.strip().str.lower()
        accepted = {spec}
        if spec == "numeric":
            accepted.update({"single_numeric_response"})
        df = df[pc_norm.isin(accepted)].copy()

    # Filter domains: accepts comma/space-separated list or 'all'
    domain_spec = str(args.domains).strip()
    if domain_spec and domain_spec.lower() != "all":
        selected = [d for d in re.split(r"[,\s]+", domain_spec) if d]
        df = df[df["domain"].astype(str).isin(selected)].copy()

    # Optionally list agents (after prompt/domain filters) and exit
    if args.list_agents:
        pool = df.copy()
        if not args.include_humans:
            pool = pool[pool["subject"] != "humans"]
        labels = sorted(pool["agent_variant"].dropna().astype(str).unique().tolist())
        print("Available agent_variant labels (after current filters):")
        for a in labels:
            print(f"  - {a}")
        print(f"Total: {len(labels)} agents")
        return

    # Filter agents by agent_variant labels if provided
    # Normalize --agents values into a list of labels unless 'all'
    agent_tokens: list[str] = []
    if isinstance(args.agents, list):
        # Flatten any comma-separated tokens within the list
        for tok in args.agents:
            if tok is None:
                continue
            for part in re.split(r"[\,]+", str(tok)):
                p = part.strip()
                if p:
                    agent_tokens.append(p)
    else:
        # Fallback if argparse delivers a single string
        for part in re.split(r"[\,\s]+", str(args.agents)):
            p = part.strip()
            if p:
                agent_tokens.append(p)

    if len(agent_tokens) == 1 and agent_tokens[0].lower() == "all":
        pass  # no filtering
    elif len(agent_tokens) > 0:
        # If including humans, make sure they are not inadvertently excluded by the agent filter
        if args.include_humans and "humans" not in agent_tokens:
            agent_tokens.append("humans")
        df = df[df["agent_variant"].astype(str).isin(agent_tokens)].copy()

    # Ensure required columns exist; fail early with a clear message if missing
    needed = ["agent_variant", "domain", "likelihood", "subject"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Prepare output directory under project root
    out_dir = project_root() / args.output_dir / args.experiment / args.prompt_category
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure bounds and RNG for Wasserstein procedures if requested
    bounds = (float(args.bounds_lower), float(args.bounds_upper))
    rng = None
    # Use RNG only if either bootstrap or permutation is requested (so tests are reproducible via seed)
    if (args.ws_bootstrap and args.ws_bootstrap > 0) or (args.ws_permute and args.ws_permute > 0):
        rng = np.random.default_rng(args.ws_seed)

    # Optionally collapse per-prompt replicates to a single observation per (agent_variant, domain, id)
    if args.collapse_replicates != "none" and "id" in df.columns:
        agg_map = {
            "mean": np.mean,
            "median": np.median,
            "first": (lambda x: x.iloc[0] if len(x) > 0 else np.nan),
        }
        agg_fn = agg_map.get(args.collapse_replicates)
        if agg_fn is not None:
            # Keep minimal columns required for downstream analysis
            keep_cols = [c for c in ["agent_variant", "domain", "likelihood", "subject", "id"] if c in df.columns]
            df_small = df[keep_cols].copy()
            # Aggregate likelihood per (agent_variant, domain, id)
            grouped = (
                df_small.groupby([c for c in ["agent_variant", "domain", "id"] if c in df_small.columns], as_index=False)
                .agg({"likelihood": agg_fn, "subject": "first"})
            )
            # Restore expected columns
            df = grouped[[c for c in ["agent_variant", "domain", "likelihood", "subject", "id"] if c in grouped.columns]].copy()
            print(f"Collapsed replicates per id using '{args.collapse_replicates}'; new rows: {len(df)}")
        else:
            print("Unknown collapse_replicates option; proceeding without collapsing.")

    # Run analyses: within-agent across domains, and within-domain across agents
    kw_agents_df, pair_agents_df = analyze_within_agent_across_domains(
        df, alpha=args.alpha, min_n=args.min_n,
        include_humans=args.include_humans,
        compute_wasserstein=args.compute_wasserstein,
        ws_bootstrap=args.ws_bootstrap,
        ws_permute=args.ws_permute,
        bounds=bounds,
        rng=rng,
    )
    kw_domains_df, pair_domains_df = analyze_within_domain_across_agents(
        df, alpha=args.alpha, min_n=args.min_n, include_humans=args.include_humans,
        compute_wasserstein=args.compute_wasserstein,
        ws_bootstrap=args.ws_bootstrap,
        ws_permute=args.ws_permute,
        bounds=bounds,
        rng=rng,
    )

    # Add a global BH (pooled across all domains) column for the within-domain across agents pairwise MWU p-values
    if not pair_domains_df.empty and pair_domains_df["p_value"].notna().any():
        mask = pair_domains_df["p_value"].notna()
        try:
            pair_domains_df.loc[mask, "p_fdr_bh_global_within_domain_stratum"] = fdr_bh(
                np.asarray(pair_domains_df.loc[mask, "p_value"].values, dtype=float), alpha=args.alpha
            )
        except Exception:
            # On any failure, fall back to NaNs to avoid breaking the pipeline
            pair_domains_df["p_fdr_bh_global_within_domain_stratum"] = np.nan
    else:
        pair_domains_df["p_fdr_bh_global_within_domain_stratum"] = np.nan

    # Save CSV outputs for the KW omnibus and pairwise MWU results
    kw_agents_df.to_csv(out_dir /  f"kw_across_domains_per_agent_v{args.version}.csv", index=False)
    pair_agents_df.to_csv(out_dir /  f"pairwise_mwu_across_domains_per_agent_v{args.version}.csv", index=False)
    kw_domains_df.to_csv(out_dir /  f"kw_across_agents_per_domain_v{args.version}.csv", index=False)
    pair_domains_df.to_csv(out_dir /  f"pairwise_mwu_across_agents_per_domain_v{args.version}.csv", index=False)

    # Generate ECDF overlay plots for visual comparison
    plot_ecdf_overlays(
        df,
        out_dir=out_dir,
        max_agents_per_domain=args.max_agents_per_domain,
        include_humans=args.include_humans,
    )

    # Optionally compute and save spike composition summaries (3-bin proportions)
    if args.compute_spike_composition:
        ad_tbl, da_tbl = _summarize_spike_composition(df, include_humans=args.include_humans)
        ad_tbl.to_csv(out_dir / f"spike_composition_by_agent_domain_v{args.version}.csv", index=False)
        da_tbl.to_csv(out_dir / f"spike_composition_by_domain_agent_v{args.version}.csv", index=False)

    # Optionally run pairwise chi-square tests on spike composition and save CSV outputs
    if args.compute_spike_composition and args.spike_pairwise_tests:
        spike_agent_df, spike_domain_df = _pairwise_spike_tests(
            df, alpha=args.alpha, include_humans=args.include_humans
        )
        spike_agent_df.to_csv(
            out_dir / f"pairwise_spike_chi2_within_agent_across_domains_v{args.version}.csv", index=False
        )
        spike_domain_df.to_csv(
            out_dir / f"pairwise_spike_chi2_within_domain_across_agents_v{args.version}.csv", index=False
        )

    # Final status message
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
