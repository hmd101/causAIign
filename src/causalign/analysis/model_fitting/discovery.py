"""Discovery & loading utilities for structured model fitting artifacts.

All functions are thin wrappers around reading the Parquet index and spec manifest
produced under results/model_fitting/<experiment>/.

Public functions:
- list_experiments(base_dir: Path) -> list[str]
- load_index(base_dir: Path, experiment: str) -> pd.DataFrame
- load_spec_manifest(base_dir: Path, experiment: str) -> pd.DataFrame | None
- merge_index_with_manifest(index_df, manifest_df) -> pd.DataFrame
- best_rows(df, metric, group_cols) -> pd.DataFrame
- ranks_long(df, metric, group_cols) -> pd.DataFrame

These helpers let legacy scripts transition away from ad‑hoc filename parsing.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import pandas as pd

_LOWER_IS_BETTER = {"loss", "aic", "bic", "rmse", "mae", "cv_rmse", "cv_mae"}
_HIGHER_IS_BETTER = {"r2", "cv_r2"}


def list_experiments(base_dir: Path) -> List[str]:
    root = base_dir / "results" / "model_fitting"
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if (p / "fit_index.parquet").exists():
            out.append(p.name)
    return out


def load_index(base_dir: Path, experiment: str) -> Optional[pd.DataFrame]:
    exp_dir = base_dir / "results" / "model_fitting" / experiment
    path = exp_dir / "fit_index.parquet"
    parts = []
    # 1) Load existing top-level index if present (but do NOT return early; we may need to augment with lr subdir indices)
    if path.exists():
        try:
            top_df = pd.read_parquet(path).copy()
            top_df["experiment"] = experiment
            parts.append(top_df)
        except Exception:
            pass
    # 2) Scan lr* subdirectories (e.g., lr0p1) produced by grid runner and merge any indices found
    if exp_dir.exists():
        for sub in sorted(exp_dir.iterdir()):
            if not (sub.is_dir() and sub.name.startswith("lr")):
                continue
            candidate_paths = []
            # layout A: lrX/fit_index.parquet
            candidate_paths.append(sub / "fit_index.parquet")
            # layout B: lrX/<experiment>/fit_index.parquet
            candidate_paths.append(sub / experiment / "fit_index.parquet")
            # layout C: lrX/variants/*/fit_index.parquet (per-agent subfolders)
            variants_dir = sub / "variants"
            if variants_dir.exists():
                for agent_dir in sorted(variants_dir.iterdir()):
                    if agent_dir.is_dir():
                        candidate_paths.append(agent_dir / "fit_index.parquet")
            for idx_path in candidate_paths:
                if not idx_path.exists():
                    continue
                try:
                    df_sub = pd.read_parquet(idx_path).copy()
                    df_sub["experiment"] = experiment
                    df_sub["lr_subdir"] = sub.name
                    parts.append(df_sub)
                except Exception:
                    continue
    if not parts:
        return None
    merged = pd.concat(parts, ignore_index=True)
    # De-duplicate on (spec_hash, group_hash) keeping last (lr_subdir entries may represent newer grid fits)
    if {"spec_hash", "group_hash"}.issubset(merged.columns):
        merged.sort_values(by=["spec_hash", "group_hash"], inplace=True)
        merged = merged.drop_duplicates(subset=["spec_hash", "group_hash"], keep="last")
    # Attempt to materialize combined index for faster future loads
    try:
        merged_no_lr = merged.drop(columns=["lr_subdir"], errors="ignore").copy()
        merged_no_lr.to_parquet(path, index=False)
    except Exception:
        pass
    return merged


def load_spec_manifest(base_dir: Path, experiment: str) -> Optional[pd.DataFrame]:
    path = base_dir / "results" / "model_fitting" / experiment / "spec_manifest.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.copy()
    df["experiment"] = experiment
    return df


def merge_index_with_manifest(index_df: pd.DataFrame, manifest_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if manifest_df is None or manifest_df.empty:
        return index_df
    keep = manifest_df[["spec_hash", "friendly_name"]].drop_duplicates()
    return index_df.merge(keep, on="spec_hash", how="left")


def _metric_direction(metric: str) -> int:
    if metric in _LOWER_IS_BETTER:
        return -1
    if metric in _HIGHER_IS_BETTER:
        return 1
    # default assume lower better
    return -1


def best_rows(df: pd.DataFrame, metric: str, group_cols: List[str]) -> pd.DataFrame:
    direction = _metric_direction(metric)
    ascending = direction == -1
    ordered = df.sort_values(metric, ascending=ascending)
    return ordered.drop_duplicates(subset=group_cols, keep="first")


def ranks_long(df: pd.DataFrame, metric: str, group_cols: List[str], label_col: str = "friendly_name") -> pd.DataFrame:
    direction = _metric_direction(metric)
    ascending = direction == -1
    ranked = df.copy()
    ranked[f"{metric}_rank"] = ranked.groupby(group_cols)[metric].rank(method="min", ascending=ascending)
    cols = group_cols + [label_col, metric, f"{metric}_rank"]
    return ranked[cols]

__all__ = [
    "list_experiments",
    "load_index",
    "load_spec_manifest",
    "merge_index_with_manifest",
    "best_rows",
    "ranks_long",
]
