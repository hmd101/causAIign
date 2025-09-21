from __future__ import annotations

"""utilities for legacy and refactored model fitting outputs.

This module now supports two parallel output pathways:

1. Legacy flat JSON + summary CSV (unchanged) used by existing analysis code.
2. New structured schema (GroupFitResult) enabling hashing, restart capture,
   ranking, and Parquet indexing for scalable querying.

1) writes both formats so downstream consumers can migrate
incrementally. 
2) Future versions will deprecate the summary CSV in favor of a
compact `fit_index.parquet`, which is now supported.
"""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .validation import validate_group_fit_result_dict


def _prepare_json(obj: Any):
    if is_dataclass(obj):  # runtime check; mypy may complain but acceptable here
        return asdict(obj)  # type: ignore[arg-type]
    return obj


def save_result_json(output_dir: Path, filename: str, data: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def append_summary_csv(output_dir: Path, filename: str, row: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df_row = pd.DataFrame([row])
    if path.exists():
        df_existing = pd.read_csv(path)
        df_out = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_out = df_row
    df_out.to_csv(path, index=False)
    return path


def save_loss_curve_plot(
    output_dir: Path,
    filename: str,
    loss_curve: list[float],
    title: str,
) -> Path:
    """Save a simple loss-vs-iteration plot for a fit.

    Title should include model, tying, agent, prompt_category, domain, and any other
    subset metadata to make the plot self-descriptive.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path_pdf = output_dir / f"{filename}.pdf"
    path_png = output_dir / f"{filename}.png"

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(path_png, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return path_pdf


# ---------------------------------------------------------------------------
# New schema writers / indexing
# ---------------------------------------------------------------------------

def save_group_fit_result_json(output_dir: Path, filename: str, group_fit_result, validate: bool = True) -> Path:
    """Write a structured GroupFitResult JSON file.

    The `group_fit_result` can be either the dataclass instance or a plain
    dictionary produced by its `to_dict()` method. Filenames should *generally*
    incorporate the short spec & group hashes to guarantee uniqueness, e.g.:

        fit_{short_spec_hash}_{short_group_hash}.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = group_fit_result.to_dict() if hasattr(group_fit_result, "to_dict") else group_fit_result
    if validate:
        # Validate before writing to catch schema regressions early
        validate_group_fit_result_dict(payload)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _row_for_index(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a minimal set of fields for Parquet indexing.

    Kept intentionally concise to keep the index small; full details remain in
    the per-group JSON files. This functions acts as the *schema contract*
    for the index. Adding a column here is a backward-compatible change; removing
    or renaming requires a migration step.
    """
    # Compute aggregate restart stats if multiple restarts captured
    # "restarts" may appear in legacy artifacts as an integer count instead of
    # a list of restart result dicts. Normalize to a list for uniform handling.
    raw_restarts = record.get("restarts") or []
    if isinstance(raw_restarts, int):  # legacy count-only form
        restarts = []  # no per-restart metrics available
    elif isinstance(raw_restarts, list):
        restarts = raw_restarts
    else:  # unexpected type (e.g., dict) – treat as empty to avoid iteration errors
        restarts = []
    losses = [r.get("loss_final") for r in restarts if r.get("loss_final") is not None]
    rmses = [r.get("rmse") for r in restarts if r.get("rmse") is not None]
    aics = [r.get("aic") for r in restarts if r.get("aic") is not None]
    loss_mean = float(sum(losses) / len(losses)) if losses else None
    loss_median = None
    loss_var = None
    rmse_mean = float(sum(rmses) / len(rmses)) if rmses else None
    rmse_median = None
    rmse_var = None
    aic_mean = float(sum(aics) / len(aics)) if aics else None
    aic_var = None
    if losses:
        sorted_losses = sorted(losses)
        mid = len(sorted_losses) // 2
        if len(sorted_losses) % 2 == 1:
            loss_median = float(sorted_losses[mid])
        else:
            loss_median = float(0.5 * (sorted_losses[mid - 1] + sorted_losses[mid]))
        if len(sorted_losses) > 1:
            mean_val = loss_mean  # already computed
            loss_var = float(sum((loss_val - mean_val) ** 2 for loss_val in sorted_losses) / (len(sorted_losses) - 1))
    if rmses:
        sorted_rmse = sorted(rmses)
        mid_r = len(sorted_rmse) // 2
        if len(sorted_rmse) % 2 == 1:
            rmse_median = float(sorted_rmse[mid_r])
        else:
            rmse_median = float(0.5 * (sorted_rmse[mid_r - 1] + sorted_rmse[mid_r]))
        if len(sorted_rmse) > 1:
            mean_rmse = rmse_mean
            rmse_var = float(sum((v - mean_rmse) ** 2 for v in sorted_rmse) / (len(sorted_rmse) - 1))
    if aics and len(aics) > 1:
        aic_var = float(sum((v - aic_mean) ** 2 for v in aics) / (len(aics) - 1))
    # Pull restart_summary if already computed (preferred authoritative aggregate)
    rs = record.get("restart_summary") or {}
    gk = record.get("group_key") or {}
    # Fallback to top-level legacy keys when group_key absent
    agent = gk.get("agent") or record.get("agent")
    prompt_category = gk.get("prompt_category") or record.get("prompt_category")
    domain = gk.get("domain") or record.get("domain")
    version = gk.get("version") or record.get("version")

    # Fallback model / optimizer fields for legacy (flat) JSON
    link_val = record.get("spec", {}).get("model", {}).get("link")
    if link_val is None:
        link_val = record.get("model")
    params_tying_val = record.get("spec", {}).get("model", {}).get("params_tying")
    if params_tying_val is None:
        params_tying_val = record.get("params_tying")
    optimizer_name = record.get("spec", {}).get("optimizer", {}).get("name") or record.get("optimizer")
    loss_name = record.get("spec", {}).get("loss", {}).get("name") or record.get("loss")
    lr_val = (
        record.get("spec", {}).get("optimizer", {}).get("lr")
        if isinstance(record.get("spec", {}), dict) else None
    )
    if lr_val is None:
        lr_val = record.get("lr")

    # If spec/group hashes absent (legacy), synthesize stable ones to avoid collapsing rows
    spec_hash = record.get("spec_hash")
    group_hash = record.get("group_hash")
    if spec_hash is None:
        spec_basis = {
            "agent": agent,
            "prompt_category": prompt_category,
            "domain": domain,
            "version": version,
            "model": link_val,
            "params_tying": params_tying_val,
            "optimizer": optimizer_name,
            "loss": loss_name,
            "lr": lr_val,
            "restarts": record.get("restarts"),
        }
        spec_hash = hashlib.sha1(json.dumps(spec_basis, sort_keys=True).encode()).hexdigest()
    if group_hash is None:
        group_basis = {
            "agent": agent,
            "prompt_category": prompt_category,
            "domain": domain,
            "version": version,
        }
        group_hash = hashlib.sha1(json.dumps(group_basis, sort_keys=True).encode()).hexdigest()
    # Short hashes (first 12 chars) if originals missing
    short_spec_hash = record.get("short_spec_hash") or spec_hash[:12]
    short_group_hash = record.get("short_group_hash") or group_hash[:12]
    return {
    "spec_hash": spec_hash,
    "short_spec_hash": short_spec_hash,
    "group_hash": group_hash,
    "short_group_hash": short_group_hash,
        "schema_version": record.get("schema_version"),
        # Group identity (with legacy fallback)
        "agent": agent,
        "prompt_category": prompt_category,
        "domain": domain,
        # Data version / experiment provenance (version newly added, legacy fallback)
        "version": version,
        # Metrics
        "loss": record.get("metrics", {}).get("loss"),
        "rmse": record.get("metrics", {}).get("rmse"),
    "r2": record.get("metrics", {}).get("r2"),
    "mae": record.get("metrics", {}).get("mae"),
        "aic": record.get("metrics", {}).get("aic"),
        "bic": record.get("metrics", {}).get("bic"),
    "ece_10bin": record.get("metrics", {}).get("ece_10bin"),
    "cv_rmse": record.get("metrics", {}).get("cv_rmse"),
    "cv_r2": record.get("metrics", {}).get("cv_r2"),
    # Dual-write LOOCV explicit naming (backward-compatible)
    "loocv_rmse": record.get("metrics", {}).get("cv_rmse") or record.get("metrics", {}).get("loocv_rmse"),
    "loocv_r2": record.get("metrics", {}).get("cv_r2") or record.get("metrics", {}).get("loocv_r2"),
    "loocv_mae": record.get("metrics", {}).get("loocv_mae"),
    "loocv_consistency": record.get("metrics", {}).get("loocv_consistency"),
    "loocv_calibration": record.get("metrics", {}).get("loocv_calibration"),
    "loocv_bias": record.get("metrics", {}).get("loocv_bias"),
        # Model structural info
    "link": link_val,
    "params_tying": params_tying_val,
    # Optimizer / loss
    "optimizer": optimizer_name,
    "loss_name": loss_name,
    "lr": lr_val,
        # Data filters provenance (subset for index)
        "num_rows": record.get("data_spec", {}).get("num_rows"),
        "data_hash": record.get("data_spec", {}).get("data_hash"),
    # Restart distribution aggregates (optional)
    "restart_count": rs.get("count", len(restarts)),
    "restart_loss_mean": rs.get("loss_mean", loss_mean),
    "restart_loss_median": rs.get("loss_median", loss_median),
    "restart_loss_var": rs.get("loss_var", loss_var),
    "restart_rmse_mean": rs.get("primary_mean") if rs.get("primary_metric") == "rmse" else rmse_mean,
    "restart_rmse_median": rs.get("primary_median") if rs.get("primary_metric") == "rmse" else rmse_median,
    "restart_rmse_var": rs.get("primary_var") if rs.get("primary_metric") == "rmse" else rmse_var,
    "restart_aic_mean": rs.get("primary_mean") if rs.get("primary_metric") == "aic" else aic_mean,
    "restart_aic_var": rs.get("primary_var") if rs.get("primary_metric") == "aic" else aic_var,
    # Generic primary metric columns (namespaced) for flexible querying
    "primary_metric": record.get("ranking", {}).get("primary_metric"),
    "primary_selection_rule": record.get("ranking", {}).get("selection_rule"),
    # Housekeeping / reproducibility
    "selection_rule": record.get("ranking", {}).get("selection_rule"),
    "primary_metric_name": record.get("ranking", {}).get("primary_metric"),
    }


def update_fit_index_parquet(output_dir: Path, index_filename: str, new_records: Iterable[Dict[str, Any]]) -> Path:
    """Append (or create) a Parquet index of group fit results.

    Each row corresponds to a single `GroupFitResult`. Idempotency is achieved
    by de-duplicating on (spec_hash, group_hash). This function loads any
    existing Parquet file, merges new rows, drops duplicates, and writes back.

    Args:
        output_dir: Directory containing the index file.
        index_filename: Typically 'fit_index.parquet'.
        new_records: Iterable of dicts (already `to_dict()`ed dataclasses).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / index_filename
    rows = [_row_for_index(r) for r in new_records]
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_existing = pd.read_parquet(path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all.drop_duplicates(subset=["spec_hash", "group_hash"], keep="last", inplace=True)
    else:
        df_all = df_new
    df_all.to_parquet(path, index=False)
    return path


__all__ = [
    "save_result_json",
    "append_summary_csv",
    "save_loss_curve_plot",
    "save_group_fit_result_json",
    "update_fit_index_parquet",
]

# ---------------------------------------------------------------------------
# Long-form restart metrics / params writers 
# ---------------------------------------------------------------------------

def append_restart_longforms(
    output_dir: Path,
    restart_records: list[dict],
    spec_hash: str,
    group_hash: str,
    short_spec_hash: str,
    short_group_hash: str,
    agent: str | None,
    prompt_category: str | None,
    domain: str | None,
    version: str | None,
    metrics_filename: str = "restart_metrics.parquet",
    params_filename: str = "restart_params.parquet",
) -> tuple[Path, Path]:
    """Append per-restart metrics & params in long-form Parquet tables.

    Schema (restart_metrics):
        spec_hash, group_hash, short_spec_hash, short_group_hash, agent, prompt_category, domain, version,
        restart_index, seed, loss_final, rmse, mae, r2, aic, bic, ece_10bin, duration_sec, status

    Schema (restart_params): long form one row per (restart_index, param_name)
        spec_hash, group_hash, restart_index, param_name, value, init_value
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    params_rows: list[dict] = []
    for r in restart_records:
        base_meta = {
            "spec_hash": spec_hash,
            "group_hash": group_hash,
            "short_spec_hash": short_spec_hash,
            "short_group_hash": short_group_hash,
            "agent": agent,
            "prompt_category": prompt_category,
            "domain": domain,
            "version": version,
            "restart_index": r.get("restart_index"),
            "seed": r.get("seed"),
        }
        metrics_rows.append({
            **base_meta,
            "loss_final": r.get("loss_final"),
            "rmse": r.get("rmse"),
            "mae": r.get("mae"),
            "r2": r.get("r2"),
            "aic": r.get("aic"),
            "bic": r.get("bic"),
            "ece_10bin": r.get("ece_10bin"),
            "duration_sec": r.get("duration_sec"),
            "status": r.get("status"),
        })
        params = r.get("params", {}) or {}
        init_params = r.get("init_params", {}) or {}
        for pname, pval in params.items():
            params_rows.append({
                **base_meta,
                "param_name": pname,
                "value": pval,
                "init_value": init_params.get(pname),
            })
    # Append / create Parquet files (idempotent concatenation; no dedupe to preserve history)
    metrics_path = output_dir / metrics_filename
    params_path = output_dir / params_filename
    if metrics_rows:
        df_m_new = pd.DataFrame(metrics_rows)
        if metrics_path.exists():
            df_m_old = pd.read_parquet(metrics_path)
            df_m_all = pd.concat([df_m_old, df_m_new], ignore_index=True)
        else:
            df_m_all = df_m_new
        df_m_all.to_parquet(metrics_path, index=False)
    if params_rows:
        df_p_new = pd.DataFrame(params_rows)
        if params_path.exists():
            df_p_old = pd.read_parquet(params_path)
            df_p_all = pd.concat([df_p_old, df_p_new], ignore_index=True)
        else:
            df_p_all = df_p_new
        df_p_all.to_parquet(params_path, index=False)
    return metrics_path, params_path


from datetime import datetime as _dt, timezone as _tz

# ---------------------------------------------------------------------------
# Spec manifest utilities
# ---------------------------------------------------------------------------
import pandas as _pd  # local alias to avoid polluting public namespace


def update_spec_manifest(output_dir: Path, spec_hash: str, short_spec_hash: str, spec_dict: Dict[str, Any]) -> Path:
    """Append a row describing a spec (if new) to spec_manifest.csv.

    Columns include human-friendly description to assist downstream analysis.
    Idempotent: if spec_hash already present, no duplicate row is added.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "spec_manifest.csv"
    # Build friendly name
    model = spec_dict.get("model", {})
    optimizer = spec_dict.get("optimizer", {})
    loss = spec_dict.get("loss", {})
    ranking = spec_dict.get("ranking", {})
    cv = spec_dict.get("cv", {})
    link = model.get("link")
    tying = model.get("params_tying")
    opt_name = optimizer.get("name")
    lr = optimizer.get("lr")
    epochs = optimizer.get("epochs")
    restarts = optimizer.get("restarts")
    loss_name = loss.get("name")
    primary_metric = ranking.get("primary_metric")
    selection_rule = ranking.get("selection_rule")
    cv_enabled = cv.get("enabled")
    parts = [f"{link}", f"p{tying}", opt_name, f"lr{lr}", f"e{epochs}", f"r{restarts}", loss_name]
    if cv_enabled:
        parts.append("cv")
    if primary_metric not in (None, "aic"):
        parts.append(f"pm-{primary_metric}")
    if selection_rule not in (None, "best_loss"):
        parts.append(f"sel-{selection_rule}")
    friendly = "-".join(str(p) for p in parts if p is not None)

    row = {
        "spec_hash": spec_hash,
        "short_spec_hash": short_spec_hash,
        "friendly_name": friendly,
        "link": link,
        "params_tying": tying,
        "optimizer": opt_name,
        "lr": lr,
        "epochs": epochs,
        "restarts": restarts,
        "loss": loss_name,
        "primary_metric": primary_metric,
        "selection_rule": selection_rule,
        "cv_enabled": cv_enabled,
        "timestamp_first_seen": _dt.now(_tz.utc).isoformat(),
    }
    df_new = _pd.DataFrame([row])
    if path.exists():
        df_old = _pd.read_csv(path)
        if spec_hash in set(df_old["spec_hash"].values):
            return path
        df_all = _pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(path, index=False)
    return path

__all__.append("update_spec_manifest")



__all__.append("update_spec_manifest")


