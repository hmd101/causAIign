#!/usr/bin/env python3
"""
Generate random-initialization baseline predictions and error metrics per agent
without fitting (domains pooled per experiment).

Outputs (under results/model_fitting/<experiment>/random_baseline/):
  - random_init_metrics.parquet: one row per (agent, prompt_category, draw)
  - random_init_params.parquet: optional per-draw parameter values (constrained scale)
"""
from __future__ import annotations

import argparse

# Ensure src/ is importable for project imports
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from causalign.analysis.model_fitting.data import (  # type: ignore
    load_processed_data,
    prepare_dataset,
)
from causalign.analysis.model_fitting.losses import LOSS_REGISTRY  # type: ignore
from causalign.analysis.model_fitting.models import (  # type: ignore
    create_parameter_module,
    device_from_string,
)
from causalign.analysis.model_fitting.tasks import (
    roman_task_to_probability,  # type: ignore
)
from causalign.config.paths import PathManager  # type: ignore


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)


def _metrics_from_preds_targets(pred: torch.Tensor, targ: torch.Tensor) -> Dict[str, float]:
    diff = pred - targ
    sse = float(torch.sum(diff ** 2).item())
    n = int(pred.shape[0])
    mse = sse / n if n else float("nan")
    rmse = float(torch.sqrt(torch.tensor(mse)).item()) if n else float("nan")
    mae = float(torch.mean(torch.abs(diff)).item()) if n else float("nan")
    y_mean = torch.mean(targ) if n else torch.tensor(float("nan"))
    sst = float(torch.sum((targ - y_mean) ** 2).item()) if n else float("nan")
    r2 = float(1.0 - (sse / sst)) if sst and sst > 0 else float("nan")

    def _ece_10(p: torch.Tensor, y: torch.Tensor) -> float:
        edges = torch.linspace(0.0, 1.0, steps=11)
        ece = 0.0
        total = p.shape[0]
        for b in range(10):
            lo = edges[b].item()
            hi = edges[b + 1].item()
            mask = (p >= lo) & (p < hi) if b < 9 else (p >= lo) & (p <= hi)
            if torch.any(mask):
                p_mean = float(torch.mean(p[mask]).item())
                y_mean_b = float(torch.mean(y[mask]).item())
                weight = float(torch.count_nonzero(mask).item()) / total
                ece += weight * abs(p_mean - y_mean_b)
        return float(ece)

    return {
        "sse": sse,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "ece_10bin": _ece_10(pred, targ) if n else float("nan"),
        "n": n,
    }


def _evaluate_random_draw(
    df_group: pd.DataFrame,
    link: str,
    num_params: int,
    loss_name: str,
    device: torch.device,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Single random parameter draw → predictions → metrics and loss."""
    _set_seeds(seed)
    module = create_parameter_module(link, num_params)
    module.to(device)
    module.eval()
    # Init same as in trainer
    for p in module.parameters():
        nn.init.normal_(p, mean=0.0, std=0.1)

    # Constrained params snapshot
    params = {k: float(v.detach().cpu().item()) for k, v in module.get_params().items()}  # type: ignore[arg-type]

    tasks = list(df_group["task"].astype(str).values)
    targets = torch.tensor(df_group["response"].astype(float).values, dtype=torch.float32, device=device)
    preds: List[torch.Tensor] = []
    for roman in tasks:
        preds.append(roman_task_to_probability(roman, link, module.get_params()))  # type: ignore[arg-type]
    pred_vec = torch.stack(preds, dim=0)

    metrics = _metrics_from_preds_targets(pred_vec, targets)
    loss_fn = LOSS_REGISTRY[loss_name]
    loss_val = float(loss_fn(pred_vec, targets).detach().cpu().item())
    metrics["loss"] = loss_val
    return metrics, params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Random-init baseline metrics for CBN models (no fitting)")
    # Data loading
    p.add_argument("--version", required=True, help="Data version (e.g., 8)")
    p.add_argument("--experiment", required=True, help="Experiment name (e.g., rw17_indep_causes)")
    p.add_argument("--pipeline-mode", choices=["llm_with_humans", "llm", "humans"], default="llm_with_humans")
    p.add_argument("--graph-type", choices=["collider", "fork", "chain"], default="collider")
    p.add_argument("--no-roman-numerals", action="store_true")
    p.add_argument("--no-aggregated", action="store_true")
    p.add_argument("--input-file")
    # Subsets
    p.add_argument("--agents", help="Comma-separated agents to include")
    p.add_argument("--prompt-categories", help="Comma-separated prompt categories to include")
    p.add_argument("--tasks", help="Comma-separated Roman tasks to include (e.g., VI,VII,X)")
    p.add_argument("--temperature", type=float, default=0.0)
    # Model config
    p.add_argument("--model", choices=["logistic", "noisy_or"], default="logistic")
    p.add_argument("--params", type=int, choices=[3, 4, 5], default=3)
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    # Random draws
    p.add_argument("--draws", type=int, default=100, help="Number of random parameter draws per group")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    # Output
    p.add_argument(
        "--output-dir",
        help="Output directory base (default: results/model_fitting/<experiment>/random_baseline)",
    )
    p.add_argument("--store-params", action="store_true", help="Also write constrained params per draw to Parquet")
    return p


def _parse_list(arg: Optional[str]) -> Optional[List[str]]:
    return [s.strip() for s in arg.split(",")] if arg else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    paths = PathManager()
    use_roman = not args.no_roman_numerals
    use_agg = not args.no_aggregated
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
    df = prepare_dataset(
        df,
        agents=_parse_list(args.agents),
        domains=None,  # we'll pool domains during grouping
        temperature=args.temperature,
        reasoning_types=None,
        tasks=_parse_list(args.tasks),
        prompt_categories=_parse_list(args.prompt_categories),
    )

    # Group by agent and prompt_category only (domains pooled)
    group_cols = ["subject"]
    if "prompt_category" in df.columns:
        group_cols.append("prompt_category")

    run_ts = datetime.now(timezone.utc).isoformat()
    device = device_from_string(args.device)
    print(f"[info] Using device: {device}")

    # Output paths
    if args.output_dir:
        base_out = Path(args.output_dir)
    else:
        base_out = paths.base_dir / "results" / "model_fitting" / args.experiment / "random_baseline"
    base_out.mkdir(parents=True, exist_ok=True)
    p_metrics = base_out / "random_init_metrics.parquet"
    p_params = base_out / "random_init_params.parquet"

    metrics_rows: List[Dict] = []
    params_rows: List[Dict] = []

    # Materialize groups so we can show progress (agent i/total)
    groups = list(df.groupby(group_cols, dropna=False))
    total_groups = len(groups)
    if not groups:
        print("[warn] No groups (agents) found after filtering; exiting.")
        return 0

    for g_idx, (keys, g) in enumerate(groups, start=1):
        k_tuple = keys if isinstance(keys, tuple) else (keys,)
        agent = str(k_tuple[0])
        prompt_cat = str(k_tuple[1]) if len(k_tuple) > 1 else None
        included_domains = sorted([str(d) for d in g["domain"].dropna().unique()]) if "domain" in g.columns else []

        print(f"[info] Processing agent {agent} (group {g_idx}/{total_groups})" + (f", prompt_category={prompt_cat}" if prompt_cat else "") + f"; draws={args.draws}")

        for d_i in range(args.draws):
            seed_i = int(args.seed) + d_i
            metrics, params = _evaluate_random_draw(
                df_group=g,
                link=args.model,
                num_params=args.params,
                loss_name=args.loss,
                device=device,
                seed=seed_i,
            )
            row = {
                "agent": agent,
                "prompt_category": prompt_cat,
                "experiment": args.experiment,
                "version": args.version,
                "link": args.model,
                "params_tying": args.params,
                "loss_name": args.loss,
                "draw_index": d_i,
                "seed": seed_i,
                "included_domains": ";".join(included_domains),
                "run_timestamp": run_ts,
                **metrics,
            }
            metrics_rows.append(row)
            if args.store_params:
                for pname, pval in params.items():
                    params_rows.append(
                        {
                            "agent": agent,
                            "prompt_category": prompt_cat,
                            "experiment": args.experiment,
                            "version": args.version,
                            "link": args.model,
                            "params_tying": args.params,
                            "loss_name": args.loss,
                            "draw_index": d_i,
                            "seed": seed_i,
                            "param_name": pname,
                            "value": float(pval),
                            "run_timestamp": run_ts,
                        }
                    )

    # Append/create outputs
    if metrics_rows:
        df_new = pd.DataFrame(metrics_rows)
        if p_metrics.exists():
            df_old = pd.read_parquet(p_metrics)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_parquet(p_metrics, index=False)
        print(f"[ok] Wrote baseline metrics → {p_metrics} (rows added: {len(df_new)})")
    else:
        print("[warn] No metrics rows produced (empty grouping?)")

    if params_rows:
        df_p_new = pd.DataFrame(params_rows)
        if p_params.exists():
            df_p_old = pd.read_parquet(p_params)
            df_p_all = pd.concat([df_p_old, df_p_new], ignore_index=True)
        else:
            df_p_all = df_p_new
        df_p_all.to_parquet(p_params, index=False)
        print(f"[ok] Wrote baseline params → {p_params} (rows added: {len(df_p_new)})")

    print(f"[done] Processed {total_groups} agent group(s). Total draws: {total_groups * args.draws}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
