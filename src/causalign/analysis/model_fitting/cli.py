"""CLI for fitting collider CBN models (legacy + new structured schema output).

This  refactor writes two formats per group:
  1. Legacy JSON + summary.csv (backward compatibility)
  2. New spec+hash structured JSON + Parquet index (forward path)

Usage examples:
    python -m causalign.analysis.model_fitting.cli \
        --version 8 --experiment pilot_study --model logistic --params 3 \
        --optimizer lbfgs --epochs 300 --restarts 5 --agents gpt-4o,human

    python -m causalign.analysis.model_fitting.cli \
        --version 8 --experiment pilot_study --enable-loocv \
        --tasks VI,VII,VIII --prompt-categories abstract_reasoning

Virtual environment (if applicable):
    source ~/.virtualenvs/llm-causality/bin/activate

Outputs written into results/model_fitting/<experiment>/ (or --output-dir):
    Legacy: fit_<agent>_<promptCat>_<domain>_<model>_<params>p_lr*.json
    New:    fit_<shortSpecHash>_<shortGroupHash>.json
    Index:  fit_index.parquet (deduplicated by (spec_hash, group_hash))
    Plots:  Loss curve PDF/PNG
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ...config.paths import PathManager
from .adapters import (
    build_data_signature,
    build_fit_spec_from_config,
    build_group_fit_result,
    compute_group_hash,
    compute_spec_hash,
)
from .data import load_processed_data, prepare_dataset
from .io import (
    append_restart_longforms,
    append_summary_csv,
    save_group_fit_result_json,
    save_loss_curve_plot,
    save_result_json,
    update_fit_index_parquet,
    update_spec_manifest,
)
from .specs import DataFilterSpec
from .trainer import FitConfig, fit_single_group

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    # Suppress extremely verbose matplotlib font manager debug spam even when --verbose is used.
    for noisy in ["matplotlib", "matplotlib.font_manager"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit collider CBN models to processed data (PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data loading
    p.add_argument("--version", "-v", help="Version number to fit (e.g., '8')")
    p.add_argument("--experiment", "-e", default="pilot_study", help="Experiment name")
    p.add_argument("--input-file", help="Path to a specific CSV to load")
    p.add_argument("--pipeline-mode", choices=["llm_with_humans", "llm", "humans"], default="llm_with_humans")
    p.add_argument("--graph-type", choices=["collider", "fork", "chain"], default="collider")
    p.add_argument("--no-roman-numerals", action="store_true", help="Load non-Roman variant if available")
    p.add_argument("--no-aggregated", action="store_true", help="Do not use aggregated human responses")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature filter for LLMs (humans allow NaN)")
    # Subsetting
    p.add_argument("--agents", help="Comma-separated agents (subjects) to include")
    p.add_argument("--domains", help="Comma-separated domains to include")
    p.add_argument("--reasoning-types", help="Comma-separated reasoning types")
    p.add_argument("--tasks", help="Comma-separated Roman tasks (e.g., VI,VII,...)")
    p.add_argument("--prompt-categories", help="Comma-separated prompt categories")
    # Model config
    p.add_argument("--model", choices=["logistic", "noisy_or"], default="logistic")
    p.add_argument("--params", type=int, choices=[3,4,5], default=3, help="Number of parameters (tying scheme)")
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="Device: auto,cpu,cuda,mps")
    p.add_argument("--enable-loocv", action="store_true", help="Enable leave-one-out task CV")
    p.add_argument("--per-restart-metrics", action="store_true", help="Compute inexpensive per-restart metrics (rmse,aic,bic,mae,r2,ece) without CV")
    p.add_argument(
        "--compute-uncertainty",
        action="store_true",
        help="Estimate parameter standard errors via a Gauss-Newton (J^T J)^{-1} approximation at the best restart. Adds an 'uncertainty' block with per-parameter SEs (on constrained scale) to outputs.")
    # Ranking / selection
    p.add_argument(
        "--primary-metric",
        choices=["aic", "bic", "rmse", "loss", "cv_rmse", "r2"],
        default="aic",
        help="Primary metric for ranking/selection in structured schema (cv_rmse requires --enable-loocv)",
    )
    p.add_argument(
        "--selection-rule",
        choices=["best_loss", "median_loss", "best_primary_metric"],
        default="best_loss",
        help="How to choose representative restart params when multiple restarts captured",
    )
    p.add_argument(
        "--store-loss-curves",
        action="store_true",
        help="Persist per-restart loss curves (PDF/PNG). Off by default to reduce I/O and disk footprint.",
    )
    # Output / misc
    p.add_argument("--output-dir", help="Output directory base for fit results")
    p.add_argument("--verbose", action="store_true")
    return p


def _parse_list(arg: Optional[str]) -> Optional[List[str]]:
    return [s.strip() for s in arg.split(",")] if arg else None


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.graph_type != "collider":
        logger.error("Only collider graph type is implemented.")
        return 1

    paths = PathManager()
    run_timestamp = datetime.now(timezone.utc).isoformat()
    # Output directory resolution:
    # Previous behavior (before fix): when --output-dir was provided (e.g. results/model_fitting/exp/lr0p1)
    # the code appended /<experiment> again, producing a nested
    # results/model_fitting/exp/lr0p1/<experiment>/ layout. This caused downstream
    # notebooks expecting restart_metrics.parquet directly under the lr* folder
    # to miss files. We now:
    #   * Use exactly the user-supplied path when --output-dir is given.
    #   * Fallback to the legacy default when not provided.
    # Backward compatibility: if a user points to an *old* nested layout root
    # (ending in the experiment name) that already contains fit_index.parquet we leave it.
    if args.output_dir:
        candidate = Path(args.output_dir)
        # If user accidentally passed a parent (without experiment) we mimic old default by appending experiment.
        # Heuristic: only append if neither fit_index.parquet nor restart_metrics.parquet exist inside candidate
        # but a directory candidate/experiment exists (suggesting an old invocation pattern).
        if not any((candidate / f).exists() for f in ["fit_index.parquet", "restart_metrics.parquet"]) and (candidate / args.experiment).exists():
            # Likely the user passed results/model_fitting/<experiment>; keep behavior consistent.
            output_dir = candidate / args.experiment
        else:
            output_dir = candidate
    else:
        output_dir = paths.base_dir / "results" / "model_fitting" / args.experiment

    # Load data with fallbacks
    use_roman = not args.no_roman_numerals
    use_agg = not args.no_aggregated
    df = None
    try:
        df = load_processed_data(
            paths,
            version=args.version,
            experiment_name=args.experiment,
            graph_type=args.graph_type,
            use_roman_numerals=use_roman,
            use_aggregated=use_agg,
            pipeline_mode=args.pipeline_mode,
            input_file=args.input_file,
        )
    except FileNotFoundError:
        logger.warning("Preferred data not found; attempting fallbacks.")
        if use_roman:
            try:
                df = load_processed_data(
                    paths,
                    version=args.version,
                    experiment_name=args.experiment,
                    graph_type=args.graph_type,
                    use_roman_numerals=False,
                    use_aggregated=use_agg,
                    pipeline_mode=args.pipeline_mode,
                    input_file=args.input_file,
                )
                use_roman = False
            except FileNotFoundError:
                pass
        if df is None and use_agg:
            try:
                df = load_processed_data(
                    paths,
                    version=args.version,
                    experiment_name=args.experiment,
                    graph_type=args.graph_type,
                    use_roman_numerals=False,
                    use_aggregated=False,
                    pipeline_mode=args.pipeline_mode,
                    input_file=args.input_file,
                )
                use_roman = False
                use_agg = False
            except FileNotFoundError:
                pass
    if df is None:
        logger.error("No data located. Provide --input-file or generate processed data.")
        return 1

    # Global filters
    agents = _parse_list(args.agents)
    domains = _parse_list(args.domains)
    reasoning = _parse_list(args.reasoning_types)
    tasks = _parse_list(args.tasks)
    prompt_categories = _parse_list(args.prompt_categories)

    df_filt = prepare_dataset(
        df,
        agents=agents,
        domains=domains,
        temperature=args.temperature,
        reasoning_types=reasoning,
        tasks=tasks,
        prompt_categories=prompt_categories,
    )

    # Group splitting
    # List of ((agent, prompt_category, domain), DataFrame)
    groups: List[Tuple[Tuple[str, Optional[str], Optional[str]], pd.DataFrame]] = []
    group_cols = ["subject"]
    if "prompt_category" in df_filt.columns:
        group_cols.append("prompt_category")
    if domains is not None and "domain" in df_filt.columns:
        group_cols.append("domain")

    for keys, g in df_filt.groupby(group_cols, dropna=False):
        k_tuple = keys if isinstance(keys, tuple) else (keys,)
        key_dict = dict(zip(group_cols, k_tuple))
        agent = key_dict.get("subject")
        prompt_cat = key_dict.get("prompt_category")
        domain = key_dict.get("domain") if "domain" in key_dict else None
        # Keys may be pandas scalar types; treat them as str/Optional[str] and ignore strict type mismatch.
        groups.append(((agent, prompt_cat, domain), g))  # type: ignore[arg-type]

    if not groups:
        logger.error("No groups formed after filtering; nothing to fit.")
        return 1

    fit_cfg = FitConfig(
        link=args.model,
        num_params=args.params,
        loss_name=args.loss,
        optimizer=args.optimizer,
        lr=args.lr,
        epochs=args.epochs,
        restarts=args.restarts,
        seed=args.seed,
        device=args.device,
        enable_loocv=args.enable_loocv,
        compute_uncertainty=args.compute_uncertainty,
    )

    summary_rows = []
    new_schema_records = []

    for (agent, prompt_cat, domain), g in groups:
        logger.info(
            f"Fitting agent={agent} prompt_category={prompt_cat} domain={domain} rows={len(g)}"
        )
        res = fit_single_group(g, fit_cfg, collect_restart_metrics=args.per_restart_metrics)

        meta = {
            "agent": agent,
            "prompt_category": prompt_cat,
            "domain": domain,
            "version": args.version,
            "experiment": args.experiment,
            "graph_type": args.graph_type,
            "pipeline_mode": args.pipeline_mode,
            "temperature": args.temperature,
            "use_roman": use_roman,
            "use_aggregated": use_agg,
            "model": args.model,
            "params_tying": args.params,
            "loss": args.loss,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "lr": args.lr,
            "restarts": args.restarts,
            "seed": args.seed,
            "included_domains": sorted([d for d in g["domain"].dropna().unique()]) if "domain" in g.columns else [],
            "run_timestamp": run_timestamp,
        }
        out = {**meta, **res}
        lr_str = f"lr{args.lr:g}".replace(".", "p")
        base_name = (
            f"fit_{agent}{f'_{prompt_cat}' if prompt_cat else ''}{f'_{domain}' if domain else ''}_"
            f"{args.model}_{args.params}p_{lr_str}"
        )
        save_result_json(output_dir, f"{base_name}.json", out)

        data_filter_spec = DataFilterSpec(
            agents=agents,
            domains=domains,
            prompt_categories=prompt_categories,
            reasoning_types=reasoning,
            tasks=tasks,
            temperature=args.temperature,
        )
        from .specs import RankingPolicy
        fallbacks = [m for m in ["aic", "rmse", "loss", "bic"] if m != args.primary_metric]
        ranking_policy = RankingPolicy(primary_metric=args.primary_metric, fallbacks=fallbacks, selection_rule=args.selection_rule)
        if args.primary_metric == "cv_rmse" and not args.enable_loocv:
            logger.warning("--primary-metric cv_rmse specified without --enable-loocv; falling back to aic")
            ranking_policy.primary_metric = "aic"
        fit_spec = build_fit_spec_from_config(
            fit_cfg,
            data_filters=data_filter_spec,
            enable_loocv=args.enable_loocv,
            ranking_policy=ranking_policy,
        )
        spec_hash, short_spec_hash = compute_spec_hash(fit_spec.to_minimal_dict())
        group_hash, short_group_hash = compute_group_hash(
            agent,
            sorted([d for d in g["domain"].dropna().unique()]) if "domain" in g.columns else [],
            prompt_cat,
            args.temperature,
            None,
        )
        data_spec, rows_signature = build_data_signature(g)
        group_key = {
            "agent": agent,
            "prompt_category": prompt_cat,
            "domain": domain,
            "experiment": args.experiment,
            "version": args.version,
        }
        group_fit_result = build_group_fit_result(
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
        new_json_name = f"fit_{short_spec_hash}_{short_group_hash}.json"
        save_group_fit_result_json(output_dir, new_json_name, group_fit_result)
        # Update spec manifest (idem potent)
        update_spec_manifest(output_dir, spec_hash, short_spec_hash, group_fit_result.spec)
        # Long-form restart metrics & params ( enhancement)
        try:
            append_restart_longforms(
                output_dir=output_dir,
                restart_records=group_fit_result.restarts,
                spec_hash=spec_hash,
                group_hash=group_hash,
                short_spec_hash=short_spec_hash,
                short_group_hash=short_group_hash,
                agent=agent,
                prompt_category=prompt_cat,
                domain=domain,
                version=args.version,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to append restart longform tables: {e}")
        new_schema_records.append(group_fit_result.to_dict())

        if args.store_loss_curves:
            curve = res.get("loss_curve")
            if curve:
                title = (
                    f"Loss curve | model={args.model}, tying={args.params}, agent={agent}, "
                    f"prompt_category={prompt_cat}, domain={domain}, ver={args.version}, temp={args.temperature}, loss={args.loss}"
                )
                save_loss_curve_plot(output_dir, f"{base_name}_{args.loss}_loss", curve, title)

        fitted = res.get("params", {}) or {}
        init_p = res.get("init_params", {}) or {}
        metrics = res.get("metrics", {}) or {}
        loocv_results = res.get("loocv")
        loocv_metrics = loocv_results.get("loocv_metrics", {}) if loocv_results else {}
        p_keys = ["pC1", "pC2", "w0", "w1", "w2", "b", "m1", "m2"]
        row = {
            "agent": agent,
            "prompt_category": prompt_cat,
            "domain": domain,
            "loss": res["loss"],
            "loss_function": args.loss,
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "r2": metrics.get("r2"),
            # Task-aggregated (per unique task) in-sample metrics to align with LOOCV granularity
            "rmse_task": metrics.get("rmse_task"),
            "r2_task": metrics.get("r2_task"),
            "ece_10bin": metrics.get("ece_10bin"),
            "aic": metrics.get("aic"),
            "bic": metrics.get("bic"),
            "loocv_rmse": loocv_metrics.get("loocv_rmse"),
            "loocv_mae": loocv_metrics.get("loocv_mae"),
            "loocv_r2": loocv_metrics.get("loocv_r2"),
            "loocv_consistency": loocv_metrics.get("loocv_consistency"),
            "loocv_calibration": loocv_metrics.get("loocv_calibration"),
            "loocv_bias": loocv_metrics.get("loocv_bias"),
            "loocv_n_folds": loocv_results.get("n_successful_folds") if loocv_results else None,
            "num_rows": res["num_rows"],
            "model": args.model,
            "params_tying": args.params,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "version": args.version,
            "experiment": args.experiment,
            "graph_type": args.graph_type,
            "temperature": args.temperature,
            "included_domains": ";".join(
                sorted([str(d) for d in g["domain"].dropna().unique()])
            ) if "domain" in g.columns else "",
            "seed_used": res.get("seed_used"),
            "restart_index": res.get("restart_index"),
            "run_timestamp": run_timestamp,
        }
        for k in p_keys:
            row[f"init_{k}"] = init_p.get(k)
            row[f"fit_{k}"] = fitted.get(k)
        summary_rows.append(row)

    # Persist legacy summary
    for row in summary_rows:
        append_summary_csv(output_dir, "summary.csv", row=row)

    # Update new index
    if new_schema_records:
        update_fit_index_parquet(output_dir, "fit_index.parquet", new_schema_records)

    logger.info(f"Saved results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



