"""Adapters bridging legacy CLI args / FitConfig to new specification & result schema.

This module introduces *non-breaking* integration layer code that constructs
`FitSpec` objects and `GroupFitResult` payloads while the legacy trainer and
CLI still operate. The goal is to progressively adopt the new
spec→hash→result pipeline without forcing an immediate rewrite of existing
scripts or analyses.

Design philosophy:
1. Pure functions: deterministic, side-effect free builders for specs / hashes.
2. Explicit field selection: only whitelisted fields influence the spec hash.
3. Backwards compatibility: functions accept current `FitConfig` + CLI meta.
4. Rich inline documentation for reviewers (why each field exists).

Example (internal usage in CLI after a group fit):

    fit_spec = build_fit_spec_from_config(fit_cfg, data_filters=...)
    spec_hash, short_spec_hash = compute_spec_hash(fit_spec.to_minimal_dict())
    group_hash, short_group_hash = compute_group_hash(agent, domains, prompt_cat, temperature, None)
    result_payload = build_group_fit_result(
        fit_spec, spec_hash, short_spec_hash, group_hash, short_group_hash,
        group_key={"agent": agent, "prompt_category": prompt_cat, "domain": domain},
        data_spec=data_spec_dict, legacy_fit_result=legacy_res_dict,
    # legacy_fit_result expected to include full restart records
    )

Later versions will replace `legacy_fit_result` with richer restart capture, which is now supported
by the trainer. See `all_restarts` field in `build_group_fit_result`.
"""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .hashing import compute_data_hash, compute_group_hash, compute_spec_hash
from .result_types import SCHEMA_VERSION, GroupFitResult
from .specs import (
    CVSpec,
    DataFilterSpec,
    FitSpec,
    LossSpec,
    ModelSpec,
    OptimizerSpec,
    RankingPolicy,
    RunSpec,
)


def _compute_restart_summary(restarts: List[Dict[str, Any]], primary_metric: str) -> Optional[Dict[str, Any]]:
    """Compute rich aggregate statistics across restart records.

    For each metric in METRICS_TO_SUMMARIZE we compute:
        count, mean, median, var, std, iqr, min, max, range
    Additionally surface duration aggregates and the *primary* metric fields
    (for backward compatibility with earlier schema usage).
    Returns None if no restarts available.
    """
    if not restarts:
        return None

    METRICS_TO_SUMMARIZE = [
        ("loss_final", "loss"),  # (restart field name, public prefix)
        ("rmse", "rmse"),
        ("aic", "aic"),
        ("bic", "bic"),
        ("mae", "mae"),
        ("r2", "r2"),
        ("ece_10bin", "ece_10bin"),
    ]

    def _to_float(v):
        try:
            return float(v)
        except Exception:  # noqa: BLE001
            return None

    def _median(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        ys = sorted(xs)
        m = len(ys) // 2
        if len(ys) % 2 == 1:
            return ys[m]
        return 0.5 * (ys[m - 1] + ys[m])

    def _var(xs: List[float]) -> Optional[float]:
        if len(xs) < 2:
            return None
        mu = sum(xs) / len(xs)
        return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)

    def _iqr(xs: List[float]) -> Optional[float]:
        if len(xs) < 2:
            return None
        ys = sorted(xs)
        q1i = (len(ys) - 1) * 0.25
        q3i = (len(ys) - 1) * 0.75
        def _interp(pos: float) -> float:
            lo = int(pos)
            hi = min(lo + 1, len(ys) - 1)
            frac = pos - lo
            return ys[lo] * (1 - frac) + ys[hi] * frac
        return _interp(q3i) - _interp(q1i)

    summary: Dict[str, Any] = {"count": len(restarts)}
    for field_name, public in METRICS_TO_SUMMARIZE:
        values = [fv for r in restarts if (fv := _to_float(r.get(field_name))) is not None]
        if not values:
            continue
        mean_v = sum(values) / len(values)
        median_v = _median(values)
        var_v = _var(values)
        std_v = (var_v ** 0.5) if var_v is not None else None
        iqr_v = _iqr(values)
        min_v = min(values)
        max_v = max(values)
        rng_v = max_v - min_v if values else None
        summary[f"{public}_mean"] = mean_v
        summary[f"{public}_median"] = median_v
        summary[f"{public}_var"] = var_v
        summary[f"{public}_std"] = std_v
        summary[f"{public}_iqr"] = iqr_v
        summary[f"{public}_min"] = min_v
        summary[f"{public}_max"] = max_v
        summary[f"{public}_range"] = rng_v

    # Duration aggregates (if any)
    _dur_raw = [_to_float(r.get("duration_sec")) for r in restarts]
    durations = [d for d in _dur_raw if d is not None]
    if durations:
        summary["duration_mean"] = sum(durations) / len(durations)
        summary["duration_sum"] = sum(durations)

    # Backward compatibility primary metric fields
    if primary_metric in ["aic", "bic", "rmse", "mae", "r2", "ece_10bin", "loss"]:
        pm_field = "loss_final" if primary_metric == "loss" else primary_metric
        pm_values = [fv for r in restarts if (fv := _to_float(r.get(pm_field))) is not None]
        if pm_values:
            summary["primary_metric"] = primary_metric
            summary["primary_mean"] = sum(pm_values) / len(pm_values)
            summary["primary_median"] = _median(pm_values)
            summary["primary_var"] = _var(pm_values)

    return summary


def build_fit_spec_from_config(
    fit_config,  # legacy FitConfig instance
    data_filters: Optional[DataFilterSpec],
    enable_loocv: bool,
    ranking_policy: Optional[RankingPolicy] = None,
) -> FitSpec:
    """Construct a `FitSpec` from the legacy `FitConfig` + filter context.

    Args:
        fit_config: Existing `FitConfig` object used by trainer.
        data_filters: Filters applied to the dataset prior to grouping.
        enable_loocv: Whether LOOCV is turned on (maps into CVSpec for now).
        ranking_policy: Optional explicit ranking policy; falls back to default.

    Returns:
        FitSpec instance capturing model, optimizer, loss, run, ranking, CV & data filters.
    """
    cv_spec = CVSpec(enabled=enable_loocv, method="loocv" if enable_loocv else "kfold", k=None)
    model_spec = ModelSpec(link=fit_config.link, params_tying=fit_config.num_params)
    optimizer_spec = OptimizerSpec(
        name=fit_config.optimizer,
        lr=fit_config.lr,
        epochs=fit_config.epochs,
        restarts=fit_config.restarts,
    )
    loss_spec = LossSpec(name=fit_config.loss_name)
    ranking = ranking_policy or RankingPolicy()
    run_spec = RunSpec(seed=fit_config.seed, device=fit_config.device, enable_loocv=enable_loocv)
    spec = FitSpec(
        model=model_spec,
        optimizer=optimizer_spec,
        loss=loss_spec,
        ranking=ranking,
        run=run_spec,
        cv=cv_spec,
        data_filters=data_filters,
    )
    return spec


def build_data_signature(df_group) -> Tuple[Dict[str, Any], List[Tuple]]:
    """Create a compact representation of the group's data for hashing.

    Returns both a human-readable `data_spec` dictionary (e.g., counts, columns)
    and a *rows signature* list of tuples that is stable across ordering for
    hashing. Current implementation uses (task, response) pairs.
    """
    # Minimal descriptive spec (extend if needed)
    data_spec = {
        "num_rows": int(df_group.shape[0]),
        "columns": sorted(list(df_group.columns)),
    }
    # Stable signature: sorted tuples ensures consistent hash irrespective of input ordering.
    rows_signature = sorted([(str(r.task), float(r.response)) for r in df_group.itertuples()])
    return data_spec, rows_signature


def build_group_fit_result(
    fit_spec: FitSpec,
    spec_hash: str,
    short_spec_hash: str,
    group_hash: str,
    short_group_hash: str,
    group_key: Dict[str, Any],
    data_spec: Dict[str, Any],
    data_rows_signature: Sequence[Tuple],
    legacy_fit_result: Dict[str, Any],
) -> GroupFitResult:
    """Assemble a `GroupFitResult` in the new schema from legacy trainer output.

    Legacy output: We only have the *best* restart metrics from the legacy
    trainer. We wrap that in a single-item `restarts` list. 
    The trainer will emit every restart and this adapter flag will be removed.
    """
    # Build minimal restart list
    restarts: List[Dict[str, Any]] = []
    best_restart_index = int(legacy_fit_result.get("restart_index", 0) or 0)
    best_seed = int(legacy_fit_result.get("seed_used", fit_spec.run.seed))
    if "all_restarts" in legacy_fit_result:
        # Use the full restart records emitted by the updated trainer
        for rec in legacy_fit_result.get("all_restarts", []) or []:
            # Ensure minimal required fields present; fall back if missing. Pass through
            # optional per-restart metrics (rmse, mae, r2, aic, bic, ece_10bin) and
            # future CV metrics if present.
            restarts.append({
                "restart_index": rec.get("restart_index"),
                "seed": rec.get("seed"),
                "loss_final": float(rec.get("loss_final", float("nan"))),
                "params": rec.get("params", {}),
                "init_params": rec.get("init_params", {}),
                "curve": rec.get("curve"),
                "status": rec.get("status", "success"),
                # Per-restart metrics (only present when collection flag enabled)
                "rmse": rec.get("rmse"),
                "mae": rec.get("mae"),
                "r2": rec.get("r2"),
                "aic": rec.get("aic"),
                "bic": rec.get("bic"),
                "ece_10bin": rec.get("ece_10bin"),
            })
    else:
        # Single synthetic restart entry representing the chosen solution
        loss_value = legacy_fit_result.get("loss")
        loss_float = float(loss_value) if loss_value is not None else float("nan")
        restarts.append({
            "restart_index": best_restart_index,
            "seed": best_seed,
            "loss_final": loss_float,
            "params": legacy_fit_result.get("params", {}),
            "init_params": legacy_fit_result.get("init_params", {}),
            "curve": legacy_fit_result.get("loss_curve"),
            "status": "success",
        })
    # Apply selection rule to choose representative restart (may change best_restart_index)
    selection_rule = getattr(fit_spec.ranking, "selection_rule", "best_loss")
    chosen_restart = None
    if restarts:
        if selection_rule == "best_loss" or len(restarts) == 1:
            chosen_restart = min(restarts, key=lambda r: r.get("loss_final", float("inf")))
        elif selection_rule == "median_loss":
            valid = [r for r in restarts if r.get("loss_final") is not None]
            if valid:
                losses_sorted = sorted(valid, key=lambda r: float(r.get("loss_final", float("inf"))))
                mid = len(losses_sorted) // 2
                chosen_restart = losses_sorted[mid]
            else:
                chosen_restart = restarts[0]
        elif selection_rule == "best_primary_metric":
            primary = fit_spec.ranking.primary_metric
            if primary in ["aic", "bic", "rmse", "mae", "r2", "ece_10bin"]:
                have = [r for r in restarts if r.get(primary) is not None]
                if have:
                    reverse = primary in ["r2"]
                    def _key_fn(r: Dict[str, Any]) -> float:
                        val = r.get(primary)
                        return float(val) if val is not None else float("inf")
                    chosen_restart = (max if reverse else min)(have, key=_key_fn)
                else:
                    chosen_restart = min(restarts, key=lambda r: r.get("loss_final", float("inf")))
            else:
                chosen_restart = min(restarts, key=lambda r: r.get("loss_final", float("inf")))
        else:
            chosen_restart = min(restarts, key=lambda r: r.get("loss_final", float("inf")))
    if chosen_restart is not None:
        best_restart_index = int(chosen_restart.get("restart_index", best_restart_index))
        loss_float = float(chosen_restart.get("loss_final", float("nan")))
    else:
        loss_float = float("nan")

    metrics_block = legacy_fit_result.get("metrics", {}).copy()
    metrics_block["loss"] = loss_float
    if legacy_fit_result.get("loocv"):
        loocv_metrics = legacy_fit_result["loocv"].get("loocv_metrics", {})
        metrics_block["cv_rmse"] = loocv_metrics.get("loocv_rmse")
        metrics_block["cv_r2"] = loocv_metrics.get("loocv_r2")

    ranking_info = {
        "primary_metric": fit_spec.ranking.primary_metric,
        "fallbacks": fit_spec.ranking.fallbacks,
        "selection_rule": selection_rule,
        "cv_scope": "group_level_only" if fit_spec.cv and fit_spec.cv.enabled else "none",
    }

    # If multiple restarts captured and primary metric is loss-like (aic, bic, rmse, loss)
    # we can optionally compute a median-based representative loss for ranking.
    restart_losses_raw = [r.get("loss_final") for r in restarts]
    restart_losses = [float(x) for x in restart_losses_raw if x is not None]
    if len(restart_losses) > 1:
        sorted_losses = sorted(restart_losses)
        mid = len(sorted_losses) // 2
        if len(sorted_losses) % 2 == 1:
            median_loss = float(sorted_losses[mid])
        else:
            median_loss = float(0.5 * (sorted_losses[mid - 1] + sorted_losses[mid]))
        ranking_info["restart_loss_median"] = median_loss
        ranking_info["restart_loss_mean"] = float(sum(sorted_losses) / len(sorted_losses))
        ranking_info["restart_loss_min"] = float(sorted_losses[0])
        ranking_info["restart_loss_max"] = float(sorted_losses[-1])
        # Simple policy: if median restart differs from chosen best by >1% relative, annotate
        if restarts:
            best_loss_val = loss_float
            if best_loss_val and median_loss:
                rel = abs(median_loss - best_loss_val) / best_loss_val if best_loss_val != 0 else None
                ranking_info["median_vs_best_rel_diff"] = rel

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
    }

    # Data hash (rows signature)
    data_full_hash, data_short_hash = compute_data_hash(data_rows_signature)
    data_spec_hashes = {
        "data_hash": data_full_hash,
        "short_data_hash": data_short_hash,
    }

    result = GroupFitResult(
        schema_version=SCHEMA_VERSION,
        spec_hash=spec_hash,
        short_spec_hash=short_spec_hash,
        group_hash=group_hash,
        short_group_hash=short_group_hash,
        spec=fit_spec.to_minimal_dict(),
        group_key=group_key,
        data_spec={**data_spec, **data_spec_hashes},
        restarts=restarts,
        best_restart_index=best_restart_index,
    best_params=chosen_restart.get("params", legacy_fit_result.get("params", {})) if chosen_restart else legacy_fit_result.get("params", {}),
        metrics=metrics_block,
        ranking=ranking_info,
    uncertainty=legacy_fit_result.get("uncertainty"),
        environment=environment,
        provenance={
            "legacy_adapter_version": 2,
            "num_restarts_captured": len(restarts),
        },
        restart_summary=None,
    )

    # Build restart_summary
    result.restart_summary = _compute_restart_summary(restarts, fit_spec.ranking.primary_metric)
    return result


__all__ = [
    "build_fit_spec_from_config",
    "build_group_fit_result",
    "build_data_signature",
    "compute_spec_hash",
    "compute_group_hash",
    "compute_data_hash",
    "_compute_restart_summary",
]

