"""Public API helpers for model fitting artifacts.

This module provides a thin, stable surface area so downstream code does not
need to import internal CLI / adapter modules directly.

Functions:
    fit_dataset(df, config, *, per_restart_metrics) -> List[GroupFitResult]
  load_index(path) -> pandas.DataFrame
  load_result_by_hash(output_dir, spec_hash, group_hash) -> dict | None

Notes:
- fit_dataset currently mirrors CLI grouping logic on (subject, prompt_category, domain).
- config is a FitConfig; RankingPolicy can be passed separately.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import pandas as pd
from .validation import validate_group_fit_result_dict

from .trainer import FitConfig, fit_single_group
from .specs import DataFilterSpec, RankingPolicy
from .adapters import (
    build_fit_spec_from_config,
    build_group_fit_result,
    build_data_signature,
    compute_spec_hash,
    compute_group_hash,
)
from .io import save_group_fit_result_json, update_fit_index_parquet

# Metrics where higher values indicate better fit (others assumed lower is better)
_HIGHER_IS_BETTER = {"r2"}


def _metric_direction(metric: str) -> int:
    """Return +1 if higher is better for metric, else -1."""
    return 1 if metric in _HIGHER_IS_BETTER else -1


def fit_dataset(
    df: pd.DataFrame,
    fit_cfg: FitConfig,
    *,
    output_dir: Path,
    ranking_policy: Optional[RankingPolicy] = None,
    per_restart_metrics: bool = False,
    data_filters: Optional[DataFilterSpec] = None,
    primary_metric: str = "aic",
) -> List[Dict[str, Any]]:
    """Fit all groups in a dataframe and return list of GroupFitResult dicts.

    Groups on available columns: subject, prompt_category (if present), domain (if present).
    Writes structured JSON + updates index parquet; returns in-memory records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: List[Tuple[Tuple[str, Optional[str], Optional[str]], pd.DataFrame]] = []
    group_cols = ["subject"]
    if "prompt_category" in df.columns:
        group_cols.append("prompt_category")
    if "domain" in df.columns:
        group_cols.append("domain")
    for keys, g in df.groupby(group_cols, dropna=False):
        k_tuple = keys if isinstance(keys, tuple) else (keys,)
        key_dict = dict(zip(group_cols, k_tuple))
        agent = key_dict.get("subject")
        prompt_cat = key_dict.get("prompt_category")
        domain = key_dict.get("domain") if "domain" in key_dict else None
        groups.append(((agent, prompt_cat, domain), g))  # type: ignore[arg-type]

    results: List[Dict[str, Any]] = []
    for (agent, prompt_cat, domain), g in groups:
        res = fit_single_group(g, fit_cfg, collect_restart_metrics=per_restart_metrics)
        rp = ranking_policy or RankingPolicy(primary_metric=primary_metric)
        if rp.primary_metric == "cv_rmse" and not fit_cfg.enable_loocv:
            rp.primary_metric = "aic"
        fit_spec = build_fit_spec_from_config(
            fit_cfg,
            data_filters=data_filters,
            enable_loocv=fit_cfg.enable_loocv,
            ranking_policy=rp,
        )
        spec_hash, short_spec_hash = compute_spec_hash(fit_spec.to_minimal_dict())
        group_hash, short_group_hash = compute_group_hash(
            agent,
            sorted([d for d in g["domain"].dropna().unique()]) if "domain" in g.columns else [],
            prompt_cat,
            None,
            None,
        )
        data_spec, rows_signature = build_data_signature(g)
        group_key = {"agent": agent, "prompt_category": prompt_cat, "domain": domain}
        gfr = build_group_fit_result(
            fit_spec=fit_spec,
            spec_hash=spec_hash,
            short_spec_hash=short_spec_hash,
            group_hash=group_hash,
            short_group_hash=short_group_hash,
            group_key=group_key,
            data_spec=data_spec,
            data_rows_signature=rows_signature,
            legacy_fit_result=res,
        )
        save_group_fit_result_json(output_dir, f"fit_{short_spec_hash}_{short_group_hash}.json", gfr)
        results.append(gfr.to_dict())

    if results:
        update_fit_index_parquet(output_dir, "fit_index.parquet", results)
    return results


def load_index(output_dir: Path) -> pd.DataFrame:
    """Load the Parquet index for a given output directory."""
    path = output_dir / "fit_index.parquet"
    return pd.read_parquet(path)


def load_result_by_hash(output_dir: Path, spec_hash: str, group_hash: str, *, validate: bool = False) -> Optional[Dict[str, Any]]:
    """Load a structured result JSON by full spec & group hash. Returns dict or None."""
    # Filename uses short hashes; scan for match lazily (small directory assumption)
    short_spec = spec_hash[:12]
    short_group = group_hash[:12]
    candidate = output_dir / f"fit_{short_spec}_{short_group}.json"
    if candidate.exists():
        with open(candidate) as f:
            data = json.load(f)
        if validate:
            validate_group_fit_result_dict(data)
        return data
    # Fallback: brute scan
    for p in output_dir.glob("fit_????????????_????????????.json"):
        with open(p) as f:
            data = json.load(f)
        if data.get("spec_hash") == spec_hash and data.get("group_hash") == group_hash:
            if validate:
                validate_group_fit_result_dict(data)
            return data
    return None

def load_validated_result_by_hash(output_dir: Path, spec_hash: str, group_hash: str) -> Optional[Dict[str, Any]]:
    """Convenience wrapper that always validates the loaded JSON against the schema."""
    return load_result_by_hash(output_dir, spec_hash, group_hash, validate=True)

__all__ = ["fit_dataset", "load_index", "load_result_by_hash", "load_validated_result_by_hash"]


def rank_index(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a copy of index df with ranking column for given metric.

    Lower-is-better metrics (aic, bic, rmse, loss) ranked ascending; higher-is-better (r2) descending.
    Adds column '{metric}_rank' (1 = best).
    """
    direction = _metric_direction(metric)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in index columns")
    df_ranked = df.copy()
    ascending = direction == -1
    df_ranked[f"{metric}_rank"] = df_ranked[metric].rank(method="min", ascending=ascending)
    return df_ranked


def top_by_metric(df: pd.DataFrame, metric: str, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Select best row per group based on metric.

    If group_cols is None returns single best overall row.
    """
    direction = _metric_direction(metric)
    ascending = direction == -1
    if group_cols:
        # Sort then drop duplicates keeping first (best) per group
        ordered = df.sort_values(metric, ascending=ascending)
        return ordered.drop_duplicates(subset=group_cols, keep="first")
    else:
        return df.sort_values(metric, ascending=ascending).head(1)


def query_top_by_primary_metric(output_dir: Path, primary_metric: str, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load index and return best rows by primary metric.

    Convenience wrapper combining load_index + top_by_metric.
    """
    idx = load_index(output_dir)
    return top_by_metric(idx, primary_metric, group_cols=group_cols)


__all__.extend([
    "rank_index",
    "top_by_metric",
    "query_top_by_primary_metric",
])
