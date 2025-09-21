"""
Analyze noisy-OR CBN fits (winners_with_params.csv) and export CSV summaries.

This script  follows the  discovery logic (experiments/tags) and CLI, and writes analysis CSVs *into the same directory* that
contains the input winners_with_params.csv (i.e., the discovered tag directory).

Focus:
- Ranking by LOOCV R²
- Descriptive stats for b, m1, m2, pC1, pC2
- Top-K vs Bottom-K contrasts
- Humans vs. models deltas on parameters
- Family/tier/effort patterns (gpt-5: nano/mini/v; o3 vs o3-mini; Claude/Gemini, etc.)
- Tying (3 vs 4) distributions
- Correlations (parameters ↔ LOOCV R²) and |m1-m2|
- Optional domain heterogeneity diagnostics across domains per agent

Example:
python scripts/cbn_normative_analysis.py --experiment rw17_indep_causes --tag v2_noisy_or_pcnum_p3-4_lr0.1_hm-pooled --no-domain-tests

If multiple tags match --tag-glob, the script will show a menu unless --non-interactive
is given; in non-interactive mode, it will process *all* matches, writing artifacts
to each corresponding tag directory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# CAblue: RGB (10, 80, 110)
CAblue = (10/255, 80/255, 110/255)       # (0.039, 0.314, 0.431)

# CAlightblue: RGB (58, 160, 171)
CAlightblue = (58/255, 160/255, 171/255) # (0.227, 0.627, 0.671)

######################################### 
########################################
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_winner_dirs(experiments: Optional[List[str]] = None,
                     tag_glob: Optional[str] = None) -> List[Path]:
    """
    Find tag directories containing both winners_with_params.csv and winners.csv.

    Prefer results/parameter_analysis over results/modelfits for discovery; if none found,
    fall back to results/modelfits. Deduplicate by (experiment, tag).
    """
    bases = [
        PROJECT_ROOT / "results" / "parameter_analysis",
        PROJECT_ROOT / "results" / "modelfits",
    ]
    exp_candidates: set[str] = set()
    for b in bases:
        if b.exists():
            for p in b.iterdir():
                if p.is_dir():
                    exp_candidates.add(p.name)
    exps = experiments if experiments else sorted(exp_candidates)
    seen: set[tuple[str, str]] = set()
    found: List[Path] = []
    pattern = tag_glob if tag_glob else "*"
    for exp in exps:
        for base in bases:
            exp_dir = base / exp
            if not exp_dir.exists():
                continue
            for tag_dir in exp_dir.glob(pattern):
                if not tag_dir.is_dir():
                    continue
                if (tag_dir / "winners_with_params.csv").exists() and (tag_dir / "winners.csv").exists():
                    key = (exp, tag_dir.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(tag_dir)
    return found


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV at {path}: {e}")


def load_and_merge(tag_dir: Path, experiment: str) -> pd.DataFrame:
    """
    Load winners_with_params and winners.csv, normalize, and return a *wide* table
    with one row per (agent, domain). Robust to missing params_tying in params CSV.
    """
    params_df = _read_csv_safe(tag_dir / "winners_with_params.csv")
    winners_df = _read_csv_safe(tag_dir / "winners.csv")

    # Normalize pooled domain label
    params_df["domain"] = params_df["domain"].where(params_df["domain"].notna(), other="all")
    winners_df["domain"] = winners_df["domain"].where(winners_df["domain"].notna(), other="all")

    # Bring params_tying over if missing
    if "params_tying" not in params_df.columns and "params_tying" in winners_df.columns:
        pt = winners_df[["agent", "domain", "params_tying"]].drop_duplicates(["agent", "domain"])
        params_df = params_df.merge(pt, on=["agent", "domain"], how="left")

    # Meta columns we may want from winners.csv
    meta_cols = [
        c for c in [
            "link", "prompt_category", "version", "learning_rate", "params_tying",
            "loocv_r2", "loocv_rmse", "r2_task", "rmse_task", "r2", "rmse", "cv_r2"
        ] if c in winners_df.columns
    ]
    # Avoid duplicate pull of params_tying
    if "params_tying" in params_df.columns and "params_tying" in meta_cols:
        meta_cols = [c for c in meta_cols if c != "params_tying"]

    key_cols = ["agent", "domain"]
    win_meta = winners_df[key_cols + meta_cols].drop_duplicates(key_cols)

    merged = params_df.merge(win_meta, on=key_cols, how="left")
    merged["experiment"] = experiment
    merged["tag"] = tag_dir.name

    # Coalesce possible _x/_y metric dupes
    for k in ["loocv_r2", "loocv_rmse", "r2_task", "rmse_task", "r2", "rmse", "cv_r2"]:
        kx, ky = f"{k}_x", f"{k}_y"
        if k not in merged.columns and (kx in merged.columns or ky in merged.columns):
            left = merged[kx] if kx in merged.columns else None
            right = merged[ky] if ky in merged.columns else None
            if left is not None and right is not None:
                merged[k] = left.combine_first(right)
            elif left is not None:
                merged[k] = left
            elif right is not None:
                merged[k] = right
            merged = merged.drop(columns=[c for c in (kx, ky) if c in merged.columns])

    # Manifest (optional) → humans_mode
    manifest_path = tag_dir / "manifest.json"
    if manifest_path.exists():
        try:
            _m = json.loads(manifest_path.read_text())
            hm = _m.get("humans_mode")
            if hm:
                merged["humans_mode"] = hm
        except Exception:
            pass

    # Keep canonical noisy-OR params if present
    five = ["b", "m1", "m2", "pC1", "pC2"]
    for c in five + ["loocv_r2", "params_tying"]:
        if c not in merged.columns:
            merged[c] = np.nan

    # Force numerics where appropriate
    for c in five + ["loocv_r2", "params_tying"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged


######################################### Utilities for labels & parsing
########################################

# --- LaTeX export helpers -----------------------------------------------------

_LATEX_ESC_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "_": r"\_",
    "#": r"\#",
    "{": r"\{",
    "}": r"\}",
}

def _esc_latex(s: str) -> str:
    if s is None:
        return ""
    out = str(s)
    for k, v in _LATEX_ESC_MAP.items():
        out = out.replace(k, v)
    return out

def _median_numeric(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.median(x)) if len(x) else float("nan")

def _frac_three_param(sub: pd.DataFrame) -> str:
    """Return 'a/b' for the share of AIC winners with params_tying == 3."""
    if "params_tying" not in sub.columns:
        return "--"
    s = pd.to_numeric(sub["params_tying"], errors="coerce").dropna()
    if s.empty:
        return "--"
    num = int((s == 3).sum())
    den = int(len(s))
    return f"{num}/{den}"

def _is_humans_agent(agent: str) -> bool:
    s = str(agent).strip().lower()
    return s in {"humans", "human-pooled", "humans-pooled"} or s.startswith("humans (pooled)")

def _parse_gpt5_effort_fields(agent: str) -> tuple[str | None, str | None]:
    """
    Parse 'verbosity' (v) and 'effort' (r) from agent strings like:
      'gpt-5-nano-v_low-r_minimal' or 'gpt-5-v_low-r_medium'
    Returns (verbosity, effort) lowercased, or (None, None) if not a GPT-5 name.
    """
    s = str(agent).lower()
    if not s.startswith("gpt-5"):
        return (None, None)
    # tolerate variants with hyphen/underscore/space separators
    m_v = re.search(r"\bv\s*[-_ ]\s*(low|medium|high)\b", s)
    m_r = re.search(r"\br\s*[-_ ]\s*(minimal|low|medium|high)\b", s)
    return (m_v.group(1) if m_v else None, m_r.group(1) if m_r else None)



def _parse_human_id(agent: str) -> Optional[int]:
    s = str(agent).strip().lower()
    m = re.match(r"(?:humans?-)?(?:subj-)?(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # Common format "human-<id>" or "humans-<id>"
    m = re.search(r"human[s-]*?(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None
\

def _family(agent: str) -> str:
    s = str(agent).strip().lower()
    if _is_human(s):
        return "human"
    if s.startswith("gpt-5"):
        return "gpt-5"
    if s.startswith("gpt-4o"):
        return "gpt-4o"
    if s.startswith("gpt-4.1"):
        return "gpt-4.1"
    if s.startswith("gpt-4"):
        return "gpt-4"
    if s.startswith("gpt-3.5"):
        return "gpt-3.5"
    if s.startswith("o3-mini"):
        return "o3-mini"
    if s.startswith("o3"):
        return "o3"
    if s.startswith("claude"):
        return "claude"
    if s.startswith("gemini"):
        return "gemini"
    return "other"


def _gpt5_tier(agent: str) -> Optional[str]:
    s = str(agent).strip().lower()
    if not s.startswith("gpt-5"):
        return None
    if "gpt-5-nano-v" in s:
        return "nano"
    if "gpt-5-mini-v" in s:
        return "mini"
    if "gpt-5-v" in s:
        return "flagship"
    return None


def _reasoning_effort_text(agent: str) -> Optional[str]:
    """
    Extract effort label following 'low-r ' if present:
      e.g., 'gpt-5-mini-v low-r minimal' -> 'minimal'
            'gpt-5-v low-r high'         -> 'high'
    """
    s = str(agent).strip().lower()
    m = re.search(r"\blow-r\s+(minimal|low|medium|high)\b", s)
    return m.group(1) if m else None


EFFORT_ORDER = {"minimal": 0, "low": 1, "medium": 2, "high": 3}


def _effort_ord(txt: Optional[str]) -> Optional[int]:
    if txt is None:
        return None
    return EFFORT_ORDER.get(txt)


def _coerce01(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _bh_fdr(pvals: Iterable[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR. Returns (reject_flags, adjusted_pvals)."""
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    # BH step-up
    denom = np.arange(1, n + 1)
    adj_vals = ranked * n / denom
    # enforce monotonicity
    adj[::-1] = np.minimum.accumulate(adj_vals[::-1])
    # Revert to original order
    adjusted = np.empty_like(adj)
    adjusted[order] = adj
    reject = adjusted <= alpha
    return reject, adjusted



# Orders for ordinal encodings
VERBOSITY_ORDER = {"low": 0, "medium": 1, "high": 2}
EFFORT_ORDER    = {"minimal": 0, "low": 1, "medium": 2, "high": 3}

# Patterns:
#  - underscore style: "... v_low-r_minimal" or "... v_low_r_minimal"
#  - hyphen/space style (legacy): "... low-r minimal"
RE_V_UDS = re.compile(r"\bv[_\-]?(low|medium|high)\b", re.IGNORECASE)     # v_low | v-low | vlow (we accept vlow too)
RE_R_UDS = re.compile(r"\br[_\-]?(minimal|low|medium|high)\b", re.IGNORECASE)  # r_minimal | r-minimal | rminimal
RE_LOW_R = re.compile(r"\blow-r[_\-\s]+(minimal|low|medium|high)\b", re.IGNORECASE)

def parse_family(agent: str) -> str:
    s = str(agent).strip().lower()
    if "human" in s:
        return "human"
    for fam in ("gpt-5", "gpt-4.1", "gpt-4o", "gpt-4", "gpt-3.5", "o3-mini", "o3", "claude", "gemini"):
        if s.startswith(fam):
            return fam
    return "other"

def parse_gpt5_tier(agent: str) -> str | None:
    s = str(agent).strip().lower()
    if not s.startswith("gpt-5"):
        return None
    if "gpt-5-nano-v" in s:
        return "nano"
    if "gpt-5-mini-v" in s:
        return "mini"
    if "gpt-5-v" in s:
        return "flagship"
    return None

def parse_verbosity_and_effort(agent: str) -> tuple[str | None, str | None]:
    """
    Extracts:
      - verbosity_text ∈ {low, medium, high}
      - effort_text    ∈ {minimal, low, medium, high}
    Supports:
      gpt-5-nano-v_low-r_minimal
      gpt-5-v_low-r_medium
      (legacy) "... low-r minimal"
    """
    s = str(agent).strip().lower()
    v = None
    e = None
    m_v = RE_V_UDS.search(s)
    if m_v:
        v = m_v.group(1)
    # Try r_<level> first
    m_r = RE_R_UDS.search(s)
    if m_r:
        e = m_r.group(1)
    else:
        # Fallback: "low-r minimal" style
        m2 = RE_LOW_R.search(s)
        if m2:
            e = m2.group(1)
    return v, e

def _is_human(agent: str) -> bool:
    s = str(agent).strip().lower()
    return s == "humans" or s in {"humans-pooled", "human-pooled"} or s.startswith("human")

def annotate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["family"]     = out["agent"].apply(parse_family)
    out["gpt5_tier"]  = out["agent"].apply(parse_gpt5_tier)
    # NEW: verbosity/effort populated for any model that encodes them in the name
    ve = out["agent"].apply(parse_verbosity_and_effort)
    out["verbosity_text"] = ve.apply(lambda t: t[0])
    out["effort_text"]    = ve.apply(lambda t: t[1])
    out["verbosity_ord"]  = out["verbosity_text"].map(VERBOSITY_ORDER)
    out["effort_ord"]     = out["effort_text"].map(EFFORT_ORDER)
    out["is_human"]       = out["agent"].apply(_is_human)
    out["abs_m_diff"]     = (out["m1"] - out["m2"]).abs()

    # bounds check for parameters in [0,1]
    for c in ["b", "m1", "m2", "pC1", "pC2"]:
        out[f"{c}_in01"] = out[c].between(0.0, 1.0, inclusive="both")
    # RENAME: all_in01 -> params_in_unit_interval (more descriptive)
    out["params_in_unit_interval"] = out[[f"{c}_in01" for c in ["b", "m1", "m2", "pC1", "pC2"]]].all(axis=1)
    out = out.drop(columns=[f"{c}_in01" for c in ["b", "m1", "m2", "pC1", "pC2"]], errors="ignore")
    return out



def quartiles_by_loocv_r2(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Bin agents into quartiles (Q1..Q4) by loocv_r2 (per domain if present),
    summarize medians per quartile, and emit interquartile (Q2∪Q3) members.
    """
    dd = df.dropna(subset=["loocv_r2"]).copy()
    # If multiple domains exist, compute quartiles globally (pooled) — consistent with pooled analysis
    dd = dd.sort_values("loocv_r2", ascending=True).reset_index(drop=True)
    n = len(dd)
    if n == 0:
        # nothing to do
        return (
            write_csv(pd.DataFrame(), out_dir / "analysis_quartiles_summary.csv", "quartiles_summary"),
            write_csv(pd.DataFrame(), out_dir / "analysis_quartile_members.csv", "quartile_members"),
            write_csv(pd.DataFrame(), out_dir / "analysis_interquartile_members.csv", "interquartile_members"),
        )

    # Bin edges via qcut; label as Q1..Q4
    qlabels = ["Q1", "Q2", "Q3", "Q4"]
    dd["quartile"] = pd.qcut(dd["loocv_r2"], q=4, labels=qlabels, duplicates="drop")

    # Summary per quartile
    params = ["loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_m_diff"]
    rows = []
    for q, block in dd.groupby("quartile", observed=True):
        med = {f"median_{p}": float(pd.to_numeric(block[p], errors="coerce").median()) for p in params if p in block.columns}
        share3 = float((block.get("params_tying", np.nan) == 3).mean()) if "params_tying" in block.columns else np.nan
        rows.append({"quartile": str(q), "n": int(len(block)), "share_3param": share3, **med})
    qsum = pd.DataFrame(rows).sort_values("quartile")

    # Members per quartile
    members = dd[["quartile", "agent", "family", "gpt5_tier", "verbosity_text", "effort_text",
                  "params_tying", "loocv_r2", "b", "m1", "m2", "pC1", "pC2"]].copy()

    # Interquartile (Q2 ∪ Q3)
    iq_members = members[members["quartile"].isin(["Q2", "Q3"])].copy()

    p1 = write_csv(qsum, out_dir / "analysis_quartiles_summary.csv", "quartiles_summary")
    p2 = write_csv(members.sort_values(["quartile", "loocv_r2"], ascending=[True, False]),
                   out_dir / "analysis_quartile_members.csv", "quartile_members")
    p3 = write_csv(iq_members.sort_values("loocv_r2", ascending=False),
                   out_dir / "analysis_interquartile_members.csv", "interquartile_members")
    return p1, p2, p3

def write_csv(df: pd.DataFrame, path: Path, name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {name}: {path}")
    return path


def rank_agents(df: pd.DataFrame, out_dir: Path) -> Path:
    keep_cols = [
        "experiment", "tag", "agent", "domain", "family", "gpt5_tier", "effort_text", "effort_ord",
        "params_tying", "loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_m_diff", "all_in01"
    ]
    cols = [c for c in keep_cols if c in df.columns]
    ranked = df.sort_values(["domain", "loocv_r2"], ascending=[True, False]).reset_index(drop=True)
    return write_csv(ranked[cols], out_dir / "analysis_ranking_by_loocv_r2.csv",
                     "ranking_by_loocv_r2")


#

def top_bottom_summary(df: pd.DataFrame, out_dir: Path, top_k: int = 15, bottom_k: int = 15) -> Tuple[Path, Path]:
    dd = df.dropna(subset=["loocv_r2"]).copy()
    dd = dd.sort_values("loocv_r2", ascending=False).reset_index(drop=True)
    k = min(top_k, len(dd))
    j = min(bottom_k, len(dd))
    top = dd.head(k).copy()
    bot = dd.tail(j).copy()

    metrics = ["loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_m_diff"]
    stats   = ["count", "mean", "std", "min", "median", "max"]

    def _tidy(block: pd.DataFrame, label: str) -> pd.DataFrame:
        rows = []
        for m in metrics:
            if m not in block.columns:
                continue
            col = pd.to_numeric(block[m], errors="coerce")
            rows.append({"group": label, "metric": m, "stat": "count",  "value": int(col.notna().sum())})
            rows.append({"group": label, "metric": m, "stat": "mean",   "value": float(col.mean())})
            rows.append({"group": label, "metric": m, "stat": "std",    "value": float(col.std())})
            rows.append({"group": label, "metric": m, "stat": "min",    "value": float(col.min())})
            rows.append({"group": label, "metric": m, "stat": "median", "value": float(col.median())})
            rows.append({"group": label, "metric": m, "stat": "max",    "value": float(col.max())})
        return pd.DataFrame(rows)

    top_stats = _tidy(top, "top")
    bot_stats = _tidy(bot, "bottom")

    top_list = top[["agent", "family", "gpt5_tier", "verbosity_text", "effort_text", "params_tying",
                    "loocv_r2", "b", "m1", "m2", "pC1", "pC2"]]
    bot_list = bot[["agent", "family", "gpt5_tier", "verbosity_text", "effort_text", "params_tying",
                    "loocv_r2", "b", "m1", "m2", "pC1", "pC2"]]

    stats_path = write_csv(pd.concat([top_stats, bot_stats], ignore_index=True),
                           out_dir / "analysis_top_bottom_summary.csv", "top_bottom_summary")
    members_path = write_csv(
        pd.concat([top_list.assign(group="top"), bot_list.assign(group="bottom")], ignore_index=True),
        out_dir / "analysis_top_bottom_members.csv", "top_bottom_members"
    )
    return stats_path, members_path


def humans_reference(df: pd.DataFrame) -> Optional[pd.Series]:
    """Pick a reference row for humans: prefer pooled 'humans' / 'humans-pooled' if available."""
    pooled_labels = {"humans", "humans-pooled", "human-pooled"}
    cand = df[df["agent"].astype(str).str.lower().isin(pooled_labels)]
    if not cand.empty:
        # Prefer domain == 'all' if present
        c2 = cand[cand["domain"].astype(str).str.lower() == "all"]
        return (c2.iloc[0] if not c2.empty else cand.iloc[0]).copy()
    # else average across individual humans (domain == all if available)
    indiv = df[df["agent"].astype(str).str.lower().str.startswith("human")]
    if indiv.empty:
        return None
    pref = indiv[indiv["domain"].astype(str).str.lower() == "all"]
    block = pref if not pref.empty else indiv
    # simple mean over noisy-OR params and loocv_r2 if present
    avg = {}
    for c in ["b", "m1", "m2", "pC1", "pC2", "loocv_r2"]:
        if c in block.columns:
            avg[c] = pd.to_numeric(block[c], errors="coerce").mean()
    s = pd.Series(avg)
    s["agent"] = "humans-avg"
    s["domain"] = "all"
    return s


def compare_to_humans(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    ref = humans_reference(df)
    if ref is None:
        print("[INFO] No human reference found; skipping humans comparison.")
        return None
    params = ["b", "m1", "m2", "pC1", "pC2"]
    comp_cols = ["agent", "family", "gpt5_tier", "effort_text", "params_tying", "domain", "loocv_r2"] + params
    have_cols = [c for c in comp_cols if c in df.columns]
    block = df[have_cols].copy()
    for p in params:
        block[f"delta_{p}_vs_humans"] = pd.to_numeric(block[p], errors="coerce") - float(ref.get(p, np.nan))
    # Simple distance score
    deltas = [f"delta_{p}_vs_humans" for p in params if f"delta_{p}_vs_humans" in block.columns]
    if deltas:
        block["L2_param_delta_vs_humans"] = np.sqrt(np.nansum(np.square(block[deltas]), axis=1))
    return write_csv(block.sort_values("L2_param_delta_vs_humans", na_position="last"),
                     out_dir / "analysis_humans_comparison.csv",
                     "humans_comparison")


def family_and_tying_summaries(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    # Family summary
    fam_keys = ["family", "gpt5_tier", "effort_text", "params_tying"]
    num_cols = ["loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_m_diff"]
    present = [c for c in num_cols if c in df.columns]
    fam_sum = (
        df.groupby(fam_keys, dropna=False)[present]
          .agg(["count", "mean", "std", "median"])
          .reset_index()
    )
    fam_path = write_csv(fam_sum, out_dir / "analysis_family_summary.csv", "family_summary")

    # Tying distribution by family/tier/effort
    if "params_tying" in df.columns:
        tying = (
            df.groupby(fam_keys, dropna=False)["agent"]
              .count().rename("n")
              .reset_index()
        )
        tying_path = write_csv(tying, out_dir / "analysis_tying_distribution.csv", "tying_distribution")
    else:
        tying_path = None
        print("[INFO] params_tying not found; skipping tying distribution.")
    return fam_path, tying_path


def gpt5_effort_trends(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Analyze the relationship between effort levels and various metrics for GPT-5 model entries.

    This function filters the input DataFrame for entries corresponding to the "gpt-5" family with non-null
    effort labels. For each GPT-5 tier (e.g., nano, mini, flagship), it computes the Spearman and Kendall
    correlation coefficients between the ordinal effort label (`effort_ord`) and a set of specified metrics.
    The results are summarized in a table and written to a CSV file in the specified output directory.

    Args:
        df (pd.DataFrame): Input DataFrame containing model data, including 'family', 'effort_ord', 'gpt5_tier', and metrics.
        out_dir (Path): Directory where the output CSV file will be saved.

    Returns:
        Optional[Path]: Path to the generated CSV file if analysis was performed, otherwise None.
    """
    g = df[(df["family"] == "gpt-5") & df["effort_ord"].notna()].copy()
    if g.empty:
        print("[INFO] No gpt-5 entries with effort labels; skipping effort trends.")
        return None
    # Within each tier (nano/mini/flagship), summarize correlations effort_ord ↔ metrics
    rows = []
    metrics = ["loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_m_diff"]
    for tier, block in g.groupby("gpt5_tier"):
        for m in metrics:
            if m not in block.columns:
                continue
            x = pd.to_numeric(block["effort_ord"], errors="coerce")
            y = pd.to_numeric(block[m], errors="coerce")
            # Spearman rho
            rho = pd.Series(x).corr(pd.Series(y), method="spearman")
            # Kendall tau (robust for small N)
            tau = pd.Series(x).corr(pd.Series(y), method="kendall")
            rows.append({"tier": tier, "metric": m, "spearman": rho, "kendall": tau, "n": len(block)})
    table = pd.DataFrame(rows)
    return write_csv(table, out_dir / "analysis_gpt5_effort_trends.csv", "gpt5_effort_trends")


def o3_comparison(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Filters the input DataFrame for entries belonging to the 'o3' or 'o3-mini' families,
    selects relevant columns, sorts the results by family and LOOCV R² score, and writes
    the output to a CSV file in the specified directory.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing model results.
        out_dir (Path): The directory where the output CSV file will be saved.

    Returns:
        Optional[Path]: The path to the written CSV file if entries are found; otherwise, None.
    """
    o = df[df["family"].isin(["o3", "o3-mini"])].copy()
    if o.empty:
        print("[INFO] No o3/o3-mini entries; skipping o3 comparison.")
        return None
    keep = ["family", "agent", "domain", "params_tying", "loocv_r2", "b", "m1", "m2", "pC1", "pC2", "abs_m_diff"]
    keep = [c for c in keep if c in o.columns]
    return write_csv(o[keep].sort_values(["family", "loocv_r2"], ascending=[True, False]),
                     out_dir / "analysis_o3_family.csv", "o3_family")


def param_correlations(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Computes Pearson and Spearman correlation matrices for selected model parameters in the given DataFrame,
    and writes the results to CSV files in the specified output directory.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing model parameters as columns.
        out_dir (Path): Directory where the correlation CSV files will be saved.

    Returns:
        Optional[Path]: Path to the saved Pearson correlation CSV file, or None if insufficient data is available.
    """
    params = ["b", "m1", "m2", "pC1", "pC2", "abs_m_diff", "loocv_r2"]
    have = [c for c in params if c in df.columns]
    block = df[have].copy()
    if block.empty or block.dropna(how="all").empty:
        print("[INFO] Not enough numeric data for correlations; skipping.")
        return None
    # Pearson
    pearson = block.corr(method="pearson")
    spearman = block.corr(method="spearman")
    p_path = write_csv(pearson.reset_index().rename(columns={"index": "metric"}),
                       out_dir / "analysis_correlations_pearson.csv", "correlations_pearson")
    s_path = write_csv(spearman.reset_index().rename(columns={"index": "metric"}),
                       out_dir / "analysis_correlations_spearman.csv", "correlations_spearman")
    return p_path


def invalid_value_report(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["b", "m1", "m2", "pC1", "pC2"]
    rows = []
    for _, r in df.iterrows():
        flags = {c: not bool(r.get(f"{c}_in01", True)) for c in cols}
        if any(flags.values()):
            rows.append({
                "agent": r["agent"], "domain": r.get("domain", None),
                "family": r.get("family", None), "params_tying": r.get("params_tying", None),
                "loocv_r2": r.get("loocv_r2", None), **{f"invalid_{c}": flags[c] for c in cols}
            })
    tab = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["agent"] + [f"invalid_{c}" for c in cols])
    return write_csv(tab, out_dir / "analysis_invalid_values.csv", "invalid_values")


def domain_heterogeneity(df: pd.DataFrame, out_dir: Path,
                         alpha: float = 0.05) -> Optional[Path]:
    """
    Kruskal-Wallis across domains within each agent on LOOCV R² and parameters.
    Adjust p-values per-agent (BH).
    """
    from scipy.stats import kruskal

    keys = ["b", "m1", "m2", "pC1", "pC2", "loocv_r2"]
    rows = []
    for agent, group in df.groupby("agent"):
        doms = group["domain"].astype(str).unique().tolist()
        if len(doms) < 2:
            continue
        for metric in keys:
            if metric not in group.columns:
                continue
            # group values by domain
            samples = [pd.to_numeric(group.loc[group["domain"] == d, metric], errors="coerce").dropna().values
                       for d in doms]
            samples = [s for s in samples if len(s) > 0]
            if len(samples) < 2:
                continue
            try:
                H, p = kruskal(*samples)
            except Exception:
                H, p = np.nan, np.nan
            rows.append({"agent": agent, "metric": metric, "H": H, "p": p, "n_domains": len(doms)})

    if not rows:
        print("[INFO] Not enough multi-domain data for Kruskal-Wallis.")
        return None

    res = pd.DataFrame(rows)
    # BH within each agent
    adj_list = []
    for agent, g in res.groupby("agent", as_index=False):
        reject, adj = _bh_fdr(g["p"].values, alpha=alpha)
        gg = g.copy()
        gg["p_fdr"] = adj
        gg["reject_fdr"] = reject
        adj_list.append(gg)
    out = pd.concat(adj_list, ignore_index=True)
    return write_csv(out.sort_values(["reject_fdr", "p_fdr", "agent", "metric"]),
                     out_dir / "analysis_domain_heterogeneity_kw.csv",
                     "domain_heterogeneity_kw")


############################
# LATEX TABLE export
def export_latex_cbn_tables(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """
    Create two LaTeX tables from winners_with_params rows:
      (A) Quartiles by LOOCV R^2 + humans (pooled)
      (B) GPT-5 'reasoning effort' summary (minimal/low/medium/high)

    Saves .tex files into out_dir and returns their paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_paths: list[Path] = []

    # --- Preprocess: keep one row per agent (best LOOCV R2), keep humans separately
    df = df.copy()
    for c in ["b", "m1", "m2", "pC1", "pC2", "loocv_r2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Humans (pooled) row (if present)
    humans_df = df[df["agent"].apply(_is_humans_agent)].copy()
    humans_row = None
    if not humans_df.empty:
        if "domain" in humans_df.columns:
            pref = humans_df[humans_df["domain"].astype(str).str.lower().isin(["all", "pooled"])]
            humans_row = (pref if not pref.empty else humans_df).sort_values("loocv_r2", ascending=False).iloc[0]
        else:
            humans_row = humans_df.sort_values("loocv_r2", ascending=False).iloc[0]

    # Non-human best rows
    best_by_agent = df.sort_values("loocv_r2", ascending=False).groupby("agent", as_index=False).first()
    best_nonhuman = best_by_agent[~best_by_agent["agent"].apply(_is_humans_agent)].copy()

    # --- (A) Quartile table ---------------------------------------------------
    N = len(best_nonhuman)
    if N > 0:
        top_k = int(np.ceil(N / 4))
        bottom_k = int(np.ceil(N / 4))
        top = best_nonhuman.nlargest(top_k, "loocv_r2").copy()
        bottom = best_nonhuman.nsmallest(bottom_k, "loocv_r2").copy()

        def _summ(sub: pd.DataFrame) -> dict[str, object]:
            out: dict[str, object] = {}
            out["R2"] = _median_numeric(sub["loocv_r2"])
            out["b"] = _median_numeric(sub["b"])
            out["m1"] = _median_numeric(sub["m1"])
            out["m2"] = _median_numeric(sub["m2"])
            # median of per-row average of p(C1), p(C2)
            pc = []
            for _, r in sub.iterrows():
                vals = []
                if not pd.isna(r.get("pC1")): vals.append(float(r.get("pC1")))
                if not pd.isna(r.get("pC2")): vals.append(float(r.get("pC2")))
                if vals: pc.append(float(np.mean(vals)))
            out["pC"] = float(np.median(pc)) if pc else float("nan")
            out["three_par"] = _frac_three_param(sub)
            out["N"] = len(sub)
            return out

        top_s = _summ(top)
        bot_s = _summ(bottom)

        lines = []
        lines.append("% Auto-generated; do not edit by hand\n")
        lines.append("\\begin{table}[t]\n\\centering\n")
        lines.append("\\caption{Parameter regimes by $\\mathrm{LOOCV}\\ R^2$ quartiles and humans (pooled). Medians shown; ``3-par'' is the fraction of AIC winners with tied strengths.}\n")
        lines.append("\\label{tab:cbn-summaries}\n")
        lines.append("\\begin{tabular}{lcccccc}\n\\toprule\n")
        lines.append("Group & $R^2$ & $b$ & $m_1$ & $m_2$ & $p(C_1){=}p(C_2)$ & 3-par \\\\\n\\midrule\n")
        lines.append(f"Top quartile (N={top_s['N']}) & {top_s['R2']:.3f} & {top_s['b']:.3f} & {top_s['m1']:.3f} & {top_s['m2']:.3f} & {top_s['pC']:.3f} & {top_s['three_par']} \\\\\n")
        if humans_row is not None:
            hp = float(np.nanmean([humans_row.get("pC1", np.nan), humans_row.get("pC2", np.nan)]))
            h_three = "--"
            if "params_tying" in humans_row.index:
                try:
                    h_three = "3" if int(pd.to_numeric(humans_row["params_tying"], errors="coerce")) == 3 else "4"
                except Exception:
                    h_three = "--"
            lines.append(f"Humans (pooled) & {float(humans_row.get('loocv_r2', np.nan)):.3f} & {float(humans_row.get('b', np.nan)):.3f} & {float(humans_row.get('m1', np.nan)):.3f} & {float(humans_row.get('m2', np.nan)):.3f} & {hp:.3f} & {h_three} \\\\\n")
        lines.append(f"Bottom quartile (N={bot_s['N']}) & {bot_s['R2']:.3f} & {bot_s['b']:.3f} & {bot_s['m1']:.3f} & {bot_s['m2']:.3f} & {bot_s['pC']:.3f} & {bot_s['three_par']} \\\\\n")
        lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

        tex_quart = out_dir / "cbn_quartiles_and_humans.tex"
        tex_quart.write_text("".join(lines))
        tex_paths.append(tex_quart)

    # --- (B) GPT-5 effort table ----------------------------------------------
    best_nonhuman["verbosity"], best_nonhuman["effort"] = zip(*best_nonhuman["agent"].map(_parse_gpt5_effort_fields))
    gpt5 = best_nonhuman[best_nonhuman["agent"].str.lower().str.startswith("gpt-5")].copy()
    if not gpt5.empty and "effort" in gpt5.columns:
        eff_order = ["minimal", "low", "medium", "high"]
        rows = []
        for e in eff_order:
            sub = gpt5[gpt5["effort"] == e]
            if len(sub):
                d = {
                    "Effort": e,
                    "N": int(len(sub)),
                    "Median_R2": _median_numeric(sub["loocv_r2"]),
                    "Min_R2": float(pd.to_numeric(sub["loocv_r2"], errors="coerce").min()),
                    "Max_R2": float(pd.to_numeric(sub["loocv_r2"], errors="coerce").max()),
                    "Median_b": _median_numeric(sub["b"]),
                    "Median_m1": _median_numeric(sub["m1"]),
                    "Median_m2": _median_numeric(sub["m2"]),
                    "three_par": _frac_three_param(sub),
                }
                rows.append(d)
        if rows:
            lines = []
            lines.append("% Auto-generated; do not edit by hand\n")
            lines.append("\\begin{table}[t]\n\\centering\n")
            lines.append("\\caption{GPT-5: performance and parameters by declared reasoning effort. Medians shown; ``3-par'' is the share of AIC winners with tied strengths.}\n")
            lines.append("\\label{tab:gpt5-effort}\n")
            lines.append("\\begin{tabular}{lccccccc}\n\\toprule\n")
            lines.append("Effort & $N$ & Median $R^2$ & Min $R^2$ & Max $R^2$ & Median $b$ & Median $m_1$ & Median $m_2$ \\\\\n\\midrule\n")
            for d in rows:
                lines.append(
                    f"{_esc_latex(d['Effort'])} & {d['N']} & {d['Median_R2']:.3f} & {d['Min_R2']:.3f} & {d['Max_R2']:.3f} "
                    f"& {d['Median_b']:.3f} & {d['Median_m1']:.3f} & {d['Median_m2']:.3f} \\\\\n"
                )
            lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
            tex_eff = out_dir / "gpt5_effort.tex"
            tex_eff.write_text("".join(lines))
            tex_paths.append(tex_eff)

    return tex_paths



######################################### CLI and entry
########################################.  
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Analyze noisy-OR CBN parameter fits for winning models; export CSVs."
    )
    # File discovery
    ap.add_argument("--experiments", nargs="*", help="Experiments to include; default: discover all with winners files")
    ap.add_argument("--experiment", help="Single experiment alias; equivalent to --experiments NAME")
    ap.add_argument("--tag-glob", help="Glob for tags (e.g., 'v2*noisy*'); quote in zsh")
    ap.add_argument("--tag", help="Exact tag to include (single tag); overrides --tag-glob")
    ap.add_argument("--non-interactive", action="store_true",
                    help="Do not prompt; if multiple matches, process all")
    # Filters ( + a few useful knobs)
    ap.add_argument("--agents", nargs="*", help="Filter to these agents; use 'all' to include all agents")
    ap.add_argument("--exclude-agents", "--exclude", nargs="*", dest="exclude_agents",
                    help="Agents to exclude (case-insensitive)")
    ap.add_argument("--domains", nargs="*", help="Filter to these domains; use 'all' to include pooled")
    ap.add_argument("--pooled-only", action="store_true", help="Filter to pooled rows only (domain == 'all')")
    ap.add_argument("--humans-mode", choices=["all", "aggregated", "pooled", "individual"], default="all",
                    help="Filter human agents by mode; non-human agents are always included.")
    ap.add_argument("--min-human-tasks", type=int, default=0,
                    help="If >0, keep only individual humans with at least this many tasks (requires coverage CSV)")
    ap.add_argument("--human-coverage-csv", type=str,
                    help="Optional path to human coverage summary CSV (columns: human_subj_id,n_tasks)")
    # Analysis knobs
    ap.add_argument("--top-k", type=int, default=15, help="K for top-K summary (default: 15)")
    ap.add_argument("--bottom-k", type=int, default=15, help="K for bottom-K summary (default: 15)")
    ap.add_argument("--no-domain-tests", action="store_true",
                    help="Skip Kruskal-Wallis domain heterogeneity tests")
    # export latex tables
    ap.add_argument("--export-tex", action="store_true", default=True,
                help="Export LaTeX tables (quartiles+humans, GPT-5 effort) into the analysis output folder")

    # Output behavior
    ap.add_argument("--copy-winners-alongside", action="store_true",
                    help="Copy winners.csv and winners_with_params.csv to analysis filenames next to outputs")

    args = ap.parse_args(argv)

    if args.experiment and not args.experiments:
        args.experiments = [args.experiment]

    # Determine search pattern; if --tag-glob has no wildcard, interpret as substring
    pattern = args.tag if args.tag else args.tag_glob
    if args.tag_glob and not any(ch in args.tag_glob for ch in "*?[]"):
        pattern = f"*{args.tag_glob}*"
        print(f"[INFO] Interpreting --tag-glob '{args.tag_glob}' as '{pattern}' (substring match)")

    # Validate experiment names under both bases
    bases = [PROJECT_ROOT / "results" / "parameter_analysis", PROJECT_ROOT / "results" / "modelfits"]
    available_exps: set[str] = set()
    for b in bases:
        if b.exists():
            for p in b.iterdir():
                if p.is_dir():
                    available_exps.add(p.name)
    if args.experiments:
        missing = [e for e in args.experiments if e not in available_exps]
        if missing:
            print("Experiment folder not found:")
            for e in missing:
                print(f"  - {e}")
            if available_exps:
                print("Available experiments:")
                for e in sorted(available_exps):
                    print(f"  - {e}")
            return 1

    tag_dirs = find_winner_dirs(args.experiments, pattern)

    # Humans-mode tag narrowing if many tags discovered (optional suffixing convention)
    if args.humans_mode != "all":
        suffix_map = {"aggregated": "hm-agg", "pooled": "hm-pooled", "individual": "hm-indiv"}
        suf = suffix_map.get(args.humans_mode)
        if suf:
            narrowed = [td for td in tag_dirs if td.name.endswith(suf)]
            if narrowed:
                tag_dirs = narrowed

    if not tag_dirs:
        print("No winners_with_params.csv + winners.csv pairs found.")
        if args.experiments:
            for exp in args.experiments:
                print(f"\nAvailable tag folders under experiment '{exp}':")
                any_listed = False
                for b in bases:
                    exp_dir = b / exp
                    if not exp_dir.exists():
                        continue
                    for td in sorted([p for p in exp_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                        has_winners = (td / "winners.csv").exists() and (td / "winners_with_params.csv").exists()
                        mark = "(has winners)" if has_winners else ""
                        try:
                            rel = td.relative_to(PROJECT_ROOT)
                        except Exception:
                            rel = td
                        print(f"  - {rel} {mark}")
                        any_listed = True
                if not any_listed:
                    print("  (no subfolders)")
            if args.tag_glob and args.experiments:
                cleaned = args.tag_glob.replace("*", "").replace("?", "")
                if cleaned:
                    candidates: List[str] = []
                    for exp in args.experiments:
                        for b in bases:
                            exp_dir = b / exp
                            if not exp_dir.exists():
                                continue
                            for td in exp_dir.iterdir():
                                if td.is_dir() and (td / "winners.csv").exists() and (td / "winners_with_params.csv").exists():
                                    candidates.append(td.name)
                    if candidates:
                        suggestions = difflib.get_close_matches(cleaned, sorted(set(candidates)), n=3, cutoff=0.5)
                        if suggestions:
                            print("\nDid you mean one of these tag names?")
                            for s in suggestions:
                                print(f"  - {s}")
        return 1

    # If a glob matched multiple, list and optionally prompt
    if args.tag_glob:
        print("Discovered candidate tag folders (with winners.csv + winners_with_params.csv):")
        for i, td in enumerate(tag_dirs, start=1):
            try:
                rel = td.relative_to(PROJECT_ROOT)
            except Exception:
                rel = td
            print(f"  [{i}] {rel}")
        if len(tag_dirs) > 1 and not args.tag and not args.non_interactive:
            # Prompt to pick one
            while True:
                sel = input(f"Select one by number (1-{len(tag_dirs)}), or press Enter to process all: ").strip()
                if sel == "":
                    break
                if sel.isdigit():
                    idx = int(sel)
                    if 1 <= idx <= len(tag_dirs):
                        tag_dirs = [tag_dirs[idx - 1]]
                        break
                print("Invalid selection. Please enter a valid number or press Enter.")

    # Process each discovered tag directory independently
    for tag_dir in tag_dirs:
        exp = tag_dir.parent.name
        try:
            rel_td = tag_dir.relative_to(PROJECT_ROOT)
        except Exception:
            rel_td = tag_dir
        print(f"\n[INFO] Processing: {rel_td}")

        wide = load_and_merge(tag_dir, exp)
        # Humans-mode agent filtering (keep non-human; restrict humans accordingly)
        if args.humans_mode != "all" and "agent" in wide.columns:
            def _keep(agent: str) -> bool:
                s = str(agent).strip().lower()
                is_aggr = s == "humans"
                is_pooled = s in {"humans-pooled", "human-pooled"}
                is_indiv = s.startswith("human-") or s.startswith("humans-")
                is_human = is_aggr or is_pooled or is_indiv
                if not is_human:
                    return True
                if args.humans_mode == "aggregated":
                    return is_aggr
                if args.humans_mode == "pooled":
                    return is_pooled
                if args.humans_mode == "individual":
                    return is_indiv
                return True
            wide = wide[wide["agent"].apply(_keep)]

        # Optional human coverage filter for individuals
        if args.min-human-tasks if False else False:
            # Guard (we keep signature parity with the other script; noop unless truly needed)
            pass

        # Filters
        if args.agents and not any(str(a).lower() == "all" for a in args.agents):
            wide = wide[wide["agent"].isin(args.agents)]
        if args.exclude_agents:
            excl = {str(a).lower() for a in args.exclude_agents}
            wide = wide[~wide["agent"].astype(str).str.lower().isin(excl)]
        if args.domains:
            wide = wide[wide["domain"].isin(args.domains)]
        if args.pooled_only:
            wide = wide[wide["domain"].astype(str).str.lower() == "all"]

        if wide.empty:
            print("[WARN] No data after filtering; skipping this tag.")
            continue



        # Annotate and coerce numerics; output base = *input tag_dir* (as requested)
        out_dir = tag_dir / "normat_analysis"  # write artifacts to same directory as input and subdirectory named normat_analysis

        out_dir.mkdir(parents=True, exist_ok=True)

        df = annotate(wide)
        if args.export_tex:
            try:
                tex_paths = export_latex_cbn_tables(df, out_dir)  # df should be the winners_with_params rows you loaded/merged
                for p in tex_paths:
                    print(f"Saved LaTeX: {p}")
            except Exception as e:
                print(f"[WARN] Failed to export LaTeX tables: {e}")

        
        keep_wide = [
            "experiment", "tag", "agent", "domain", "family", "gpt5_tier",
            "verbosity_text", "verbosity_ord", "effort_text", "effort_ord",
            "params_tying", "loocv_r2", "b", "m1", "m2", "pC1", "pC2",
            "abs_m_diff", "params_in_unit_interval"
        ]
        write_csv(df[[c for c in keep_wide if c in df.columns]].copy(),
                out_dir / "analysis_parameters_wide.csv", "parameters_wide")

       

        long = df.melt(
            id_vars=[c for c in ["experiment", "tag", "agent", "domain", "family", "gpt5_tier",
                                 "effort_text", "effort_ord", "params_tying", "loocv_r2"] if c in df.columns],
            value_vars=[c for c in ["b", "m1", "m2", "pC1", "pC2"] if c in df.columns],
            var_name="parameter", value_name="value"
        )
        write_csv(long, out_dir / "analysis_parameters_long.csv", "parameters_long")  # Save long-format parameters table

        rank_agents(df, out_dir)  # Export ranking of agents by LOOCV R²
        top_bottom_summary(df, out_dir, top_k=args.top_k, bottom_k=args.bottom_k)  # Summarize top/bottom-K agents
        quartiles_by_loocv_r2(df, out_dir)  # Bin agents into LOOCV R² quartiles and summarize
        compare_to_humans(df, out_dir)  # Compare model parameters to human reference
        family_and_tying_summaries(df, out_dir)  # Summarize by model family/tier/effort and tying distribution
        gpt5_effort_trends(df, out_dir)  # Analyze effort trends for GPT-5 tiers
        o3_comparison(df, out_dir)  # Compare O3 and O3-mini family results
        param_correlations(df, out_dir)  # Compute parameter correlation matrices
        invalid_value_report(df, out_dir)  # Report agents with invalid parameter values

        # Domain heterogeneity tests (optional)
        if not args.no_domain_tests:
            domain_heterogeneity(df, out_dir)  # Test for parameter heterogeneity across domains

        # Optionally copy source winners into analysis filenames (handy when aggregating elsewhere)
        if args.copy_winners_alongside:
            src1 = tag_dir / "winners.csv"
            src2 = tag_dir / "winners_with_params.csv"
            if src1.exists():
                shutil.copy2(src1, out_dir / f"analysis_src_winners.csv")  # Copy winners.csv to analysis folder
            if src2.exists():
                shutil.copy2(src2, out_dir / f"analysis_src_winners_with_params.csv")  # Copy winners_with_params.csv to analysis folder

        print(f"[INFO] Done: {rel_td}")

    print("\nAll done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
