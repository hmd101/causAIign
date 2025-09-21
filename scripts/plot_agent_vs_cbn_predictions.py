# #!/usr/bin/env python3
# """
# Plot Agent vs CBN Predictions (manifest-driven)
# ==============================================

# This script reads winners_with_params.csv (and merges winners.csv for metadata)
# to generate out-of-sample-looking comparisons: per agent, plot the agent's
# observed per-task means against the CBN predictions from the winning model
# parameters (the same ones used in tables and heatmaps).

# Examples
# --------
# Index-driven (exploration)
# --source index [--metric cv_r2|loocv_r2|r2_task|r2|aic]
# Honors --agent/--agents. Produces outputs under .../index_<metric>/.

# Manifest-driven (recommended for figures for publication aligning with other tables / figures presented in a publication)
# Use your tag or path:
# --source manifest --winners-manifest v2_pcnum
# --source manifest --winners-manifest results/parameter_analysis/<experiment>/<tag>/winners_with_params.csv


#   # Using a tag (auto-resolves CSV path under results/parameter_analysis/<experiment>/<tag>)
#   python scripts/plot_agent_vs_cbn_predictions.py \
#     --winners-manifest v2_noisy_or_pcnum_p3-4_lr0.1 \
#     --experiment rw17_indep_causes \
#     --version 2 \
#     --agents gpt-4o claude-3-opus-20240229

#   # Using a direct path to winners_with_params.csv
#   python scripts/plot_agent_vs_cbn_predictions.py \
#     --winners-manifest results/parameter_analysis/rw17_indep_causes/v2_noisy_or_pcnum_p3-4_lr0.1/winners_with_params.csv \
#     --experiment rw17_indep_causes \
#     --version 2 \
#     --agent gpt-4o
# """
# from __future__ import annotations

# import argparse
# import logging
# from pathlib import Path
# from typing import List, Optional, Dict, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch

# from causalign.analysis.model_fitting.tasks import roman_task_to_probability
# from causalign.config.paths import PathManager
# from causalign.analysis.model_fitting.data import load_processed_data, prepare_dataset
# from causalign.analysis.model_fitting import discovery as mf_discovery
# from causalign.analysis.model_fitting import api as mf_api


# ROMAN_ORDER: List[str] = [
#     "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"
# ]


# def setup_logging(verbose: bool = False) -> None:
#     level = logging.DEBUG if verbose else logging.INFO
#     logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


# def _resolve_manifest_paths(base_dir: Path, experiment: str, manifest_arg: str) -> Tuple[Path, Optional[Path], str]:
#     """Return (winners_with_params.csv, winners.csv, tag) given a path or a tag name."""
#     p = Path(manifest_arg)
#     if p.suffix.lower() == ".csv":
#         # Direct path to CSV; infer winners.csv sibling
#         winners_params = p
#         tag = winners_params.parent.name
#         winners_csv = winners_params.parent / "winners.csv"
#         return winners_params, (winners_csv if winners_csv.exists() else None), tag
#     # Otherwise treat as a tag under results/parameter_analysis/<experiment>/<tag>
#     tag = manifest_arg
#     base = base_dir / "results" / "parameter_analysis" / experiment / tag
#     winners_params = base / "winners_with_params.csv"
#     winners_csv = base / "winners.csv"
#     return winners_params, (winners_csv if winners_csv.exists() else None), tag

# essential_param_cols = ["b", "m1", "m2", "pC1", "pC2"]





# def _merge_params_with_metadata(params_df: pd.DataFrame, winners_df: Optional[pd.DataFrame]) -> pd.DataFrame:
#     df = params_df.copy()
#     if winners_df is not None and not winners_df.empty:
#         # Keep only a few metadata fields if present
#         keep_candidates = [
#             "agent",
#             "domain",
#             "link",
#             "prompt_category",
#             "version",
#             "loocv_r2",
#             "cv_r2",
#         ]
#         keep_cols = [c for c in keep_candidates if c in winners_df.columns]
#         key_cols = [c for c in ["agent", "domain"] if c in winners_df.columns]
#         if key_cols:
#             meta = winners_df[keep_cols].drop_duplicates(subset=key_cols)
#             df = df.merge(meta, on=key_cols, how="left")
#     return df


# def _predict_tasks_for_row(row: pd.Series, tasks: List[str]) -> pd.DataFrame:
#     """Generate CBN predictions for a winners_with_params row across tasks."""
#     link = str(row.get("link") or "noisy_or")
#     # Collect numeric params present in the row
#     params: Dict[str, float] = {}
#     for k in ["b", "m1", "m2", "pC1", "pC2", "w0", "w1", "w2"]:
#         if k in row and pd.notna(row[k]):
#             try:
#                 params[k] = float(row[k])
#             except Exception:
#                 pass
#     tensors: Dict[str, torch.Tensor] = {k: torch.tensor(v, dtype=torch.float32) for k, v in params.items()}
#     preds = []
#     for t in tasks:
#         try:
#             y = roman_task_to_probability(t, link, tensors)
#             preds.append({"task": t, "prediction": float(y.item())})
#         except Exception:
#             preds.append({"task": t, "prediction": np.nan})
#     return pd.DataFrame(preds)


# def _agent_task_means(
#     paths: PathManager,
#     version: str,
#     experiment: str,
#     agent: str,
#     prompt_category: Optional[str],
#     domain: Optional[str],
# ) -> pd.DataFrame:
#     """Load agent data and compute per-task mean responses.

#     If domain is None or 'all', we pool across domains; otherwise we filter to the exact domain.
#     """
#     df = load_processed_data(
#         paths,
#         version=version,
#         experiment_name=experiment,
#         graph_type="collider",
#         use_roman_numerals=True,
#         use_aggregated=True,
#         pipeline_mode="llm_with_humans",
#     )
#     doms = None if (domain is None or str(domain).lower() == "all") else [str(domain)]
#     pcs = [prompt_category] if (prompt_category and str(prompt_category) != "nan") else None
#     sub = prepare_dataset(df, agents=[agent], domains=doms, prompt_categories=pcs)
#     if sub.empty:
#         return pd.DataFrame(columns=["task", "response_mean"])  # empty
#     g = (
#         sub.groupby("task", dropna=False)["response"].mean().reset_index().rename(columns={"response": "response_mean"})
#     )
#     g["task"] = g["task"].astype(str)
#     g = g[g["task"].isin(ROMAN_ORDER)].copy()
#     g["task"] = pd.Categorical(g["task"], categories=ROMAN_ORDER, ordered=True)
#     g = g.sort_values("task")
#     return g


# def _save_plot(
#     df_agent: pd.DataFrame,
#     df_pred: pd.DataFrame,
#     out_dir: Path,
#     agent: str,
#     experiment: str,
#     tag: str,
#     domain_label: str,
#     prompt_category: Optional[str],
#     model_label: Optional[str],
#     show: bool,
# ) -> None:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     merged = pd.merge(df_agent, df_pred, on="task", how="outer")
#     plt.figure(figsize=(9, 4))
#     sns.set_style("whitegrid")
#     x = merged["task"].astype(str).tolist()
#     plt.plot(x, merged["response_mean"], marker="o", label=f"{agent} (mean)")
#     plot_label = model_label or "CBN (winner)"
#     plt.plot(x, merged["prediction"], marker="s", label=plot_label)
#     plt.ylim(0, 1)
#     plt.ylabel("Probability")
#     title = f"{agent} vs CBN — {experiment} ({domain_label}; {prompt_category or 'prompt'})"
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     safe_agent = agent.replace("/", "-")
#     base = f"{safe_agent}_{experiment}_{tag}_{domain_label}_{(prompt_category or 'pc').replace(' ', '-') }"
#     for ext in ("pdf", "png"):
#         plt.savefig(out_dir / f"{base}.{ext}", dpi=300, bbox_inches="tight")
#     if show:
#         plt.show()
#     plt.close()


# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(description="Plot Agent vs CBN predictions using winners manifest")
#     p.add_argument("--source", choices=["manifest", "index"], default="manifest", help="Where to read winning params from")
#     p.add_argument("--winners-manifest", required=False, help="Used when --source=manifest: tag (e.g., v2_noisy_or_pcnum_p3-4_lr0.1) or path to winners_with_params.csv")
#     p.add_argument("--experiment", required=True, help="Experiment name (e.g., rw17_indep_causes)")
#     p.add_argument("--version", required=True, help="Version string used in processed data loading")
#     p.add_argument("--agent", help="Single agent to plot")
#     p.add_argument("--agents", nargs="*", help="Multiple agents to plot")
#     p.add_argument("--output-dir", help="Output dir root (default: results/plots/agent_vs_cbn/<experiment>/<tag>)")
#     # Index-mode options
#     p.add_argument("--metric", default="cv_r2", help="Index metric to select best per (agent,domain,prompt_category) when --source=index")
#     p.add_argument("--no-show", action="store_true", help="Do not display plots interactively")
#     p.add_argument("--verbose", action="store_true")
#     return p


# def main(argv: Optional[List[str]] = None) -> int:
#     ap = build_parser()
#     args = ap.parse_args(argv)
#     setup_logging(args.verbose)
#     log = logging.getLogger(__name__)

#     paths = PathManager()
#     tag: str = ""
#     merged: pd.DataFrame

#     if args.source == "manifest":
#         if not args.winners_manifest:
#             log.error("--winners-manifest is required when --source=manifest")
#             return 2
#         winners_params_path, winners_csv_path, tag = _resolve_manifest_paths(paths.base_dir, args.experiment, args.winners_manifest)
#         if not winners_params_path.exists():
#             log.error(f"winners_with_params.csv not found: {winners_params_path}")
#             return 1

#         params_df = pd.read_csv(winners_params_path)
#         winners_df = pd.read_csv(winners_csv_path) if (winners_csv_path and winners_csv_path.exists()) else None

#         # Ensure link exists or infer
#         if "link" not in params_df.columns:
#             inferred = None
#             if winners_df is not None and "link" in winners_df.columns:
#                 inferred = winners_df["link"].iloc[0]
#             else:
#                 tl = str(tag).lower()
#                 if "noisy_or" in tl or "noisyor" in tl:
#                     inferred = "noisy_or"
#                 elif "logistic" in tl:
#                     inferred = "logistic"
#             if inferred is None and all(c in params_df.columns for c in essential_param_cols):
#                 inferred = "noisy_or"
#             if inferred is None:
#                 log.error("Could not infer 'link' from manifest or tag")
#                 return 1
#             params_df["link"] = inferred

#         merged = _merge_params_with_metadata(params_df, winners_df)
#     else:
#         # Index-driven discovery of winners and parameters
#         idx = mf_discovery.load_index(paths.base_dir, args.experiment)
#         if idx is None or idx.empty:
#             log.error("No fit_index.parquet found or empty; cannot use --source=index")
#             return 1
#         # Determine selection metric
#         metric_candidates = [args.metric, "cv_r2", "loocv_r2", "r2_task", "r2", "aic"]
#         metric = next((m for m in metric_candidates if m in idx.columns), None)
#         if metric is None:
#             log.error(f"None of candidate metrics present in index: {metric_candidates}")
#             return 1
#         # Best per (agent, domain, prompt_category)
#         group_cols = [c for c in ["agent", "domain", "prompt_category"] if c in idx.columns]
#         if not group_cols:
#             group_cols = ["agent"] if "agent" in idx.columns else []
#         best = mf_discovery.best_rows(idx, metric, group_cols) if group_cols else idx.sort_values(metric).head(1)

#         # Filter agents if requested
#         requested_agents: Optional[List[str]] = None
#         if args.agent and args.agents:
#             requested_agents = list(dict.fromkeys([args.agent] + list(args.agents)))
#         elif args.agent:
#             requested_agents = [args.agent]
#         elif args.agents:
#             requested_agents = list(args.agents)
#         if requested_agents and "agent" in best.columns:
#             best = best[best["agent"].astype(str).isin([str(a) for a in requested_agents])]
#         if best.empty:
#             log.error("No matching rows in index after filtering.")
#             return 1

#         records: List[Dict[str, object]] = []
#         exp_root = paths.base_dir / "results" / "model_fitting" / args.experiment
#         for _, r in best.iterrows():
#             spec_hash = r.get("spec_hash")
#             group_hash = r.get("group_hash")
#             if not (isinstance(spec_hash, str) and isinstance(group_hash, str)):
#                 continue
#             lr_subdir = r.get("lr_subdir") if "lr_subdir" in r else None
#             search_dirs = [exp_root]
#             if isinstance(lr_subdir, str) and lr_subdir:
#                 search_dirs.insert(0, exp_root / lr_subdir)
#                 search_dirs.insert(0, exp_root / lr_subdir / args.experiment)
#             gfr = None
#             for d in search_dirs:
#                 try:
#                     gfr = mf_api.load_result_by_hash(d, spec_hash, group_hash, validate=False)
#                 except Exception:
#                     gfr = None
#                 if gfr is not None:
#                     break
#             if gfr is None:
#                 log.warning(f"Could not locate JSON for spec={spec_hash[:8]} group={group_hash[:8]}")
#                 continue
#             # Extract link and params
#             link = None
#             try:
#                 link = gfr.get("spec", {}).get("model", {}).get("name") or gfr.get("spec", {}).get("model", {}).get("link")
#             except Exception:
#                 link = None
#             if not link and "link" in r:
#                 link = r.get("link")
#             params_map: Dict[str, float] = {}
#             try:
#                 for k, v in (gfr.get("best_params", {}) or {}).items():
#                     try:
#                         params_map[k] = float(v)
#                     except Exception:
#                         pass
#             except Exception:
#                 pass
#             # Pull a LOOCV-like metric for display if available
#             loocv_val = None
#             rv = r.get("cv_r2")
#             if isinstance(rv, (float, int)):
#                 loocv_val = float(rv)
#             rv = r.get("loocv_r2") if loocv_val is None else loocv_val
#             if loocv_val is None and isinstance(rv, (float, int)):
#                 loocv_val = float(rv)

#             rec: Dict[str, object] = {
#                 "agent": r.get("agent"),
#                 "domain": r.get("domain") if "domain" in r else None,
#                 "prompt_category": r.get("prompt_category") if "prompt_category" in r else None,
#                 "link": link or "noisy_or",
#             }
#             if loocv_val is not None:
#                 rec["loocv_r2"] = loocv_val
#             for k in ["b", "m1", "m2", "pC1", "pC2", "w0", "w1", "w2"]:
#                 if k in params_map:
#                     rec[k] = params_map[k]
#             records.append(rec)
#         merged = pd.DataFrame.from_records(records)
#         if merged.empty:
#             log.error("Failed to materialize any parameter records from index JSONs.")
#             return 1
#         tag = f"index_{metric}"

#     # Agents filter (for manifest mode only; index mode already filtered above)
#     if args.source == "manifest":
#         requested_agents2: Optional[List[str]] = None
#         if args.agent and args.agents:
#             requested_agents2 = list(dict.fromkeys([args.agent] + list(args.agents)))
#         elif args.agent:
#             requested_agents2 = [args.agent]
#         elif args.agents:
#             requested_agents2 = list(args.agents)
#         if requested_agents2:
#             merged = merged[merged["agent"].astype(str).isin([str(a) for a in requested_agents2])].copy()
#     if merged.empty:
#         log.error("No matching agents in winners manifest after filtering.")
#         return 1

#     # Output directory
#     out_root = Path(args.output_dir) if args.output_dir else (paths.base_dir / "results" / "plots" / "agent_vs_cbn" / args.experiment / tag)
#     out_root.mkdir(parents=True, exist_ok=True)

#     # Iterate per row
#     rows_plotted = 0
#     for _, row in merged.iterrows():
#         agent = str(row["agent"]) if "agent" in row and pd.notna(row["agent"]) else None
#         if not agent:
#             continue
#         domain_val = row.get("domain")
#         domain_label = str(domain_val) if pd.notna(domain_val) else "all"
#         prompt_category = str(row.get("prompt_category")) if ("prompt_category" in row and pd.notna(row.get("prompt_category"))) else None

#         pred_df = _predict_tasks_for_row(row, ROMAN_ORDER)
#         df_agent = _agent_task_means(paths, args.version, args.experiment, agent, prompt_category, domain_label)
#         if df_agent.empty:
#             log.warning(f"No agent data for agent={agent}, domain={domain_label}, prompt={prompt_category}; skipping plot.")
#             continue

#         # Build legend label with LOOCV R² if available
#         loocv_val = None
#         if "loocv_r2" in row:
#             v = row.get("loocv_r2")
#             if pd.notna(v):
#                 try:
#                     loocv_val = float(v)
#                 except Exception:
#                     loocv_val = None
#         if loocv_val is None and "cv_r2" in row:
#             v = row.get("cv_r2")
#             if pd.notna(v):
#                 try:
#                     loocv_val = float(v)
#                 except Exception:
#                     loocv_val = None
#         model_label = f"CBN (winner, LOOCV R²={loocv_val:.2f})" if isinstance(loocv_val, float) else "CBN (winner)"

#         subdir = out_root / agent
#         _save_plot(
#             df_agent,
#             pred_df,
#             subdir,
#             agent,
#             args.experiment,
#             tag,
#             domain_label,
#             prompt_category,
#             model_label,
#             show=(not args.no_show),
#         )
#         log.info(f"Saved plots for {agent} ({domain_label}) -> {subdir}")
#         rows_plotted += 1

#     if rows_plotted == 0:
#         log.error("No plots generated.")
#         return 2

#     return 0


# if __name__ == "__main__":  # pragma: no cover
#     raise SystemExit(main())


#!/usr/bin/env python3
"""
Plot Agent vs Model Predictions (Index-Based)
=============================================

Refactored to consume the structured model fitting artifacts introduced in the
new pipeline (fit_index.parquet + spec_manifest.csv + hashed fit JSON files)
instead of brittle filename pattern parsing. Only the small subset of JSON
files actually needed for the selected rows is opened (on‑demand parameter
extraction) which scales better than loading every legacy file.

Data Sources:
    * Agent / human / combined responses: processed CSVs (unchanged)
    * Model fits: results/model_fitting/<experiment>/
            - fit_index.parquet (lightweight tabular summary & metrics)
            - spec_manifest.csv (optional friendly names)
            - fit_<short_spec>_<short_group>.json (full restart details & params)

Flow:
    1. Load & filter the Parquet index by agent / prompt_category / domain / link /
         params_tying / loss function.
    2. (Optional) Select one best row per (agent, prompt_category, domain) using
         discovery.best_rows(metric).
    3. Resolve each retained (spec_hash, group_hash) to its hashed JSON file and
         read only the needed parameter set (best restart) + optimizer lr.
    4. Generate predictions for all tasks present in the agent data.
    5. Combine & plot via facet_lineplot with overlay lines for each model.

Metric Selection:
    The index currently stores: loss, aic, bic, rmse (and possibly cv_* variants).
    R² is not yet present in the index; requesting --metric r2 will raise a clear
    error until it is added upstream.

Examples:
    # Plot specific agent with aggregated domains (default behaviour)
    python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2

    # Separate lines for specified domains
    python scripts/plot_agent_vs_cbn_predictions.py --agent claude-3-opus --experiment rw17_indep_causes --version 2 --domains weather economy --prompt-categories numeric

    # Aggregate all domains explicitly into one line
    python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --domains all

    # Show only best fitting models (by AIC) using index ranking
    python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --best-only --metric aic

    # Filter model structure via index (logistic + 3 / 5 shared parameter schemes)
    python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --model-types logistic --param-counts 3 5

    # Use huber loss fits only
    python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --loss-functions huber

Legacy flags like --learning-rates now resolve lr from JSON spec (index does
not yet store lr explicitly) and are applied after initial index filtering.

Manifest-driven (winners manifest) examples:
    # Minimal: plot a single agent present in the manifest
    python scripts/plot_agent_vs_cbn_predictions.py \
        --winners-manifest results/parameter_analysis/rw17_indep_causes/v2_noisy_or_pcnum_p3-4_lr0.1/winners_with_params.csv \
        --experiment rw17_indep_causes \
        --version 2 \
        --agent gpt-4o

    # Plot a subset of agents from the manifest
    python scripts/plot_agent_vs_cbn_predictions.py \
        --winners-manifest results/parameter_analysis/rw17_indep_causes/v2_noisy_or_pcnum_p3-4_lr0.1/winners_with_params.csv \
        --experiment rw17_indep_causes \
        --version 2 \
        --agents gpt-4o claude-3-opus-20240229 gemini-1.5-pro

    # Random Abstract experiment, custom output dir, no UI
    python scripts/plot_agent_vs_cbn_predictions.py \
        --winners-manifest results/parameter_analysis/random_abstract/v1_noisy_or_pcnum_p3-4_lr0.1_noh/winners_with_params.csv \
        --experiment random_abstract \
        --version 1 \
        --agents gemini-1.5-flash \
        --output-dir results/plots/agent_vs_cbn \
        --no-show
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set

import pandas as pd
import torch
from matplotlib import cm

from causalign.analysis.model_fitting.tasks import roman_task_to_probability
from causalign.analysis.model_fitting.discovery import (
    load_index,
    best_rows,
)
from causalign.analysis.visualization.facet_lineplot import create_facet_line_plot
from causalign.config.paths import PathManager



from tueplots import bundles, fonts
# from tueplots.figsizes import rel_width as tw_rel_width

# w, _ = tw_rel_width(rel_width=0.9)  # 90% of a NeurIPS column width
# fig, ax = plt.subplots(figsize=(w, height))

import matplotlib as mpl

# NeurIPS-like, LaTeX, serif
# config = bundles.neurips2023(nrows=2, ncols=1, rel_width=0.3, usetex=True, family="serif")

config = bundles.neurips2023(
    nrows=2, ncols=1,
    rel_width=0.8,   # <- was 0.3
    usetex=True, family="serif"
)


config["legend.title_fontsize"] = 12
config["font.size"] = 14
config["axes.labelsize"] = 14
config["axes.titlesize"] = 16
config["xtick.labelsize"] = 12
config["ytick.labelsize"] = 12
config["legend.fontsize"] = 12
config["text.latex.preamble"] = r"\usepackage{amsmath,bm,xcolor} \definecolor{inference}{HTML}{FF5B59}"

font_config = fonts.neurips2022_tex(family="serif")
config = {**config, **font_config}

mpl.rcParams.update(config)


# Ensure the project's src directory is on sys.path for direct script execution
project_root = Path(__file__).parent.parent  # repository root (contains 'src')
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


# Prompt-category synonyms (case-insensitive)
NUMERIC_SYNS = {
    "numeric", "pcnum", "num", "single_numeric", "single_numeric_response",
}
COT_SYNS = {
    "cot", "pccot", "chain_of_thought", "chain-of-thought", "cot_stepwise",
}


def _expand_prompt_category_synonyms(cats: Optional[List[str]]) -> Optional[Set[str]]:
    """Return a lowercase set including provided categories plus known synonyms."""
    if not cats:
        return None
    out: Set[str] = set()
    for c in cats:
        if c is None:
            continue
        t = str(c).strip().lower()
        if t in NUMERIC_SYNS or t == "numeric":
            out.update(NUMERIC_SYNS)
            out.add("numeric")
        elif t in COT_SYNS or t == "cot":
            out.update(COT_SYNS)
            out.add("cot")
        else:
            out.add(t)
    return out


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_agent_data(
    paths: PathManager,
    version: str,
    experiment_name: str,
    agent: str,
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
    pipeline_mode: str = "llm_with_humans",
    temperature_filter: Optional[float] = None,
    domains: Optional[List[str]] = None,
    prompt_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load agent prediction data with filtering"""
    logger = logging.getLogger(__name__)

    # Determine data path using similar logic to plot_facet_line_plots.py
    processed_base = paths.base_dir / "data" / "processed"
    
    if pipeline_mode == "humans":
        experiment_dir = processed_base / "humans" / "rw17"
    elif pipeline_mode == "llm":
        experiment_dir = processed_base / "llm" / "rw17" / experiment_name
    else:  # "llm_with_humans" (default)
        experiment_dir = processed_base / "llm_with_humans" / "rw17" / experiment_name

    # Generate version string
    version_str = f"{version}_v_" if version else ""

    if pipeline_mode == "humans":
        data_path = experiment_dir / f"rw17_{graph_type}_humans_processed.csv"
    elif pipeline_mode == "llm":
        if use_roman_numerals:
            data_path = (
                experiment_dir
                / "reasoning_types"
                / f"{version_str}{graph_type}_llm_only_roman.csv"
            )
        else:
            data_path = experiment_dir / f"{version_str}{graph_type}_llm_only.csv"
    else:
        # Combined data files (llm_with_humans)
        if use_roman_numerals and use_aggregated:
            data_path = (
                experiment_dir
                / "reasoning_types"
                / f"{version_str}{graph_type}_cleaned_data_roman.csv"
            )
        elif use_aggregated:
            data_path = (
                experiment_dir
                / f"{version_str}humans_avg_equal_sample_size_cogsci.csv"
            )
        else:
            data_path = (
                experiment_dir / f"{version_str}{graph_type}_cleaned_data.csv"
            )

    logger.info(f"Loading agent data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Agent data file not found: {data_path}")
    
    # Filter to specific agent
    if "subject" in df.columns:
        df = df[df["subject"] == agent].copy()
    else:
        raise ValueError(f"'subject' column not found in data. Available columns: {df.columns.tolist()}")
    
    if df.empty:
        raise ValueError(f"No data found for agent '{agent}' in {data_path}")
    
    # Apply filters
    if temperature_filter is not None and "temperature" in df.columns:
        df = df[df["temperature"] == temperature_filter].copy()
    
    if domains is not None and "domain" in df.columns:
        df = df[df["domain"].isin(domains)].copy()
    
    if prompt_categories is not None and "prompt_category" in df.columns:
        syns = _expand_prompt_category_synonyms(prompt_categories)
        if syns:
            df = df[df["prompt_category"].astype(str).str.lower().isin(syns)].copy()
    
    # Standardize column names
    if "likelihood" in df.columns and "likelihood-rating" not in df.columns:
        df["likelihood-rating"] = df["likelihood"]
    
    logger.info(f"Loaded {len(df)} rows for agent {agent}")
    return df


def _roman_order() -> List[str]:
    return [
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
        "XI",
    ]


def _predict_noisy_or_tasks(params: Dict[str, float], tasks: List[str]) -> Dict[str, float]:
    """Compute noisy-or predictions for given Roman tasks using provided parameters.

    Returns probabilities scaled to 0-100 for plotting.
    """
    # Convert to torch tensors as expected by roman_task_to_probability
    tensors = {
        "b": torch.tensor(float(params["b"])),
        "m1": torch.tensor(float(params["m1"])),
        "m2": torch.tensor(float(params["m2"])),
        "pC1": torch.tensor(float(params["pC1"])),
        "pC2": torch.tensor(float(params["pC2"])),
    }
    out: Dict[str, float] = {}
    for t in tasks:
        y = roman_task_to_probability(t, "noisy_or", tensors).item()
        out[t] = float(y) * 100.0
    return out


def _make_subject_colors(subjects: List[str]) -> Dict[str, Any]:
    """Generate a distinct color mapping for each subject (pair label).

    Uses matplotlib categorical palettes when possible and falls back to an HSV spread.
    Returns dict mapping subject -> RGB tuple.
    """
    n = len(subjects)
    if n <= 10:
        cmap = cm.get_cmap('tab10', n)
    elif n <= 20:
        cmap = cm.get_cmap('tab20', n)
    else:
        # Wide gamut for many subjects
        cmap = cm.get_cmap('hsv', n)
    colors = {sub: tuple(cmap(i)[:3]) for i, sub in enumerate(subjects)}
    # Override humans to magenta
    for sub in subjects:
        s = str(sub).strip().lower()
        if s == "humans" or s.startswith("human"):
            colors[sub] = (1.0, 0.0, 1.0)  # magenta
    return colors


def _resolve_fit_json(experiment_dir: Path, short_spec: str, short_group: str, cache: Dict[str, Path]) -> Path:
    key = f"{short_spec}_{short_group}"
    if key in cache:
        return cache[key]
    pattern = f"fit_{short_spec}_{short_group}.json"
    matches = list(experiment_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Hashed fit JSON not found for {pattern}")
    cache[key] = matches[0]
    return matches[0]


def _extract_params_and_lr(fit_json: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[float]]:
    # Pick best restart (lowest loss_final)
    restarts = fit_json.get("restarts", [])
    if restarts:
        best = min(restarts, key=lambda r: r.get("loss_final", float("inf")))
        params = best.get("params", {})
    else:
        params = {}
    lr = None
    try:
        lr = fit_json["spec"]["optimizer"].get("lr")
    except Exception:
        pass
    return params, lr


def load_indexed_model_fits(
    experiment_dir: Path,
    agent: str,
    prompt_categories: Optional[List[str]],
    domains: Optional[List[str]],
    aggregate_all_domains: bool,
    model_types: Optional[List[str]],
    param_counts: Optional[List[str]],
    loss_functions: Optional[List[str]],
    learning_rates: Optional[List[float]],
    best_only: bool,
    metric: str,
) -> pd.DataFrame:
    """Load model fits using the structured index and (optionally) restrict to best rows.

    Returns a DataFrame with columns: agent, prompt_category, domain, model, params_tying,
    loss, aic, bic, rmse, loss_function (alias of loss_name), fitted_params, lr, file_path.
    """
    logger = logging.getLogger(__name__)

    index_df = load_index(PathManager().base_dir, experiment_dir.name)
    if index_df is None or index_df.empty:
        raise FileNotFoundError(f"fit_index.parquet not found for experiment {experiment_dir.name}")

    # Filter core dimensions
    df = index_df[index_df["agent"] == agent].copy()
    if prompt_categories:
        syns = _expand_prompt_category_synonyms(prompt_categories)
        if syns:
            df = df[df["prompt_category"].astype(str).str.lower().isin(syns)]
    if domains:
        # Only keep domain-specific fits if user specified explicit domains
        df = df[df["domain"].isin(domains)]

    # Model structure filters
    if model_types:
        df = df[df["link"].isin(model_types)]
    if param_counts:
        param_ints = [int(p) for p in param_counts]
        df = df[df["params_tying"].isin(param_ints)]
    if loss_functions:
        df = df[df["loss_name"].isin(loss_functions)]

    if df.empty:
        raise ValueError("No index rows match the specified filters")

    # Domain preference logic (aggregate_all_domains similar to legacy): prefer domain=None rows if aggregating
    if aggregate_all_domains or domains is None:
        aggregated_rows = df[df["domain"].isna()]
        if not aggregated_rows.empty:
            logger.info(f"Using {len(aggregated_rows)} aggregated (domain=None) fits")
            df = aggregated_rows
        else:
            logger.info("No aggregated fits found; will average domain-specific predictions later if requested")

    # Best-only selection
    if best_only:
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not present in index columns {list(df.columns)}. Cannot compute best-only selection yet.")
        group_cols = ["agent", "prompt_category", "domain"]
        df = best_rows(df, metric=metric, group_cols=group_cols)
        logger.info(f"Selected {len(df)} best rows via metric '{metric}'")

    # Load JSONs & extract params (apply learning rate filter late)
    records = []
    cache: Dict[str, Path] = {}
    available_lrs: Set[float] = set()
    for _, row in df.iterrows():
        try:
            fit_path = _resolve_fit_json(experiment_dir, row.short_spec_hash, row.short_group_hash, cache)
            with open(fit_path, "r") as f:
                fit_json = json.load(f)
            params, lr = _extract_params_and_lr(fit_json)
            if lr is not None:
                try:
                    available_lrs.add(float(lr))
                except Exception:
                    pass
            if learning_rates is not None and lr not in learning_rates:
                continue
            record = {
                "file_path": str(fit_path),
                "agent": row.agent,
                "prompt_category": row.prompt_category,
                "domain": row.domain if pd.notna(row.domain) else None,
                "model": row.link,
                "params_tying": row.params_tying,
                "loss": row.loss,
                "loss_function": row.loss_name,
                "aic": row.aic if "aic" in row else None,
                "bic": row.bic if "bic" in row else None,
                "rmse": row.rmse if "rmse" in row else None,
                "loocv_r2": row.loocv_r2 if hasattr(row, "loocv_r2") else (row["loocv_r2"] if "loocv_r2" in df.columns else None),
                "cv_r2": row.cv_r2 if hasattr(row, "cv_r2") else (row["cv_r2"] if "cv_r2" in df.columns else None),
                "r2_task": row.r2_task if hasattr(row, "r2_task") else (row["r2_task"] if "r2_task" in df.columns else None),
                "r2": row.r2 if hasattr(row, "r2") else (row["r2"] if "r2" in df.columns else None),
                "fitted_params": params,
                "lr": lr,
            }
            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to load params for spec={row.short_spec_hash} group={row.short_group_hash}: {e}")
            continue

    if not records:
        if learning_rates is not None:
            raise ValueError(
                "No model fits retained after learning rate filtering. "
                f"Requested learning_rates={learning_rates}; available learning rates among scanned fits={sorted(available_lrs) if available_lrs else 'none found'}"
            )
        raise ValueError("No model fits retained after JSON parameter extraction")
    out_df = pd.DataFrame(records)
    logger.info(f"Prepared {len(out_df)} model fits after structured loading")
    if learning_rates is None:
        uniq_lrs = sorted([lr for lr in out_df['lr'].dropna().unique().tolist()])
        if uniq_lrs:
            logger.info(f"Available learning rates (no filter applied): {uniq_lrs}")
    return out_df


def generate_model_predictions(model_fits_df: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
    """Generate model predictions for all tasks using fitted parameters (index-based fits)."""
    logger = logging.getLogger(__name__)
    
    prediction_rows = []
    
    for _, fit in model_fits_df.iterrows():
        model_type = fit['model']
        params = fit['fitted_params']
        # Try to surface LOOCV R² if present in index
        r2_value = None
        for r2_key in ("loocv_r2", "cv_r2", "r2_task", "r2"):
            if r2_key in fit and pd.notna(fit[r2_key]):
                try:
                    r2_value = float(fit[r2_key])
                    break
                except Exception:
                    pass
        
        # Convert parameters to tensors
        param_tensors = {}
        for key, value in params.items():
            param_tensors[key] = torch.tensor(float(value), dtype=torch.float32)
        
        # Generate predictions for each task
        for task in tasks:
            try:
                pred_tensor = roman_task_to_probability(task, model_type, param_tensors)
                prediction = float(pred_tensor.item()) * 100  # Convert to 0-100 scale
                
                prediction_rows.append({
                    'task': task,
                    'model_prediction': prediction,
                    'agent': fit['agent'],
                    'prompt_category': fit['prompt_category'], 
                    'domain': fit['domain'],
                    'model': fit['model'],
                    'params_tying': fit['params_tying'],
                    'loss': fit['loss'],
                    'loss_function': fit['loss_function'],
                    'lr': fit['lr'],
                    'r2': r2_value,
                    'loocv_r2': r2_value,
                    'aic': fit.get('aic'),
                    'file_path': fit['file_path'],
                })
                
            except Exception as e:
                logger.warning(f"Error generating prediction for task {task} with model {model_type}: {e}")
                continue
    
    if not prediction_rows:
        raise ValueError("No model predictions could be generated")
    
    df = pd.DataFrame(prediction_rows)
    logger.info(f"Generated {len(df)} model predictions")
    return df


## Legacy selection / filtering helpers removed; index-based filtering covers these responsibilities.


def create_experimental_condition_string(
    experiment: str,
    version: str,
    agent: str,
    domains: Optional[List[str]] = None,
    prompt_categories: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    max_line_length: int = 80,
    aggregate_all_domains: bool = False,
    loss_functions: Optional[List[str]] = None,
) -> str:
    """Create a string describing the experimental condition, breaking into multiple lines if needed"""
    parts = []
    
    # Add experiment and version
    parts.append(f"Experiment: {experiment}")
    parts.append(f"Version: {version}")
    parts.append(f"Agent: {agent}")
    
    # Add domains if specified
    if aggregate_all_domains:
        parts.append("Domains: All Domains")
    elif domains:
        if len(domains) == 1:
            parts.append(f"Domain: {domains[0]}")
        else:
            parts.append(f"Domains: {', '.join(sorted(domains))}")
    else:
        parts.append("Domains: Aggregated")
    
    # Add prompt categories if specified
    if prompt_categories:
        if len(prompt_categories) == 1:
            parts.append(f"Prompt: {prompt_categories[0]}")
        else:
            parts.append(f"Prompts: {', '.join(sorted(prompt_categories))}")
    
    # Add temperature if specified
    if temperature is not None:
        parts.append(f"Temperature: {temperature}")
    
    # Add loss function if specified and not default
    if loss_functions:
        if len(loss_functions) == 1 and loss_functions[0] != "mse":
            parts.append(f"Loss: {loss_functions[0]}")
        elif len(loss_functions) > 1:
            parts.append(f"Loss: {', '.join(sorted(loss_functions))}")
    
    # Join parts with " | " and break into multiple lines if too long
    full_string = " | ".join(parts)
    
    # If the string is too long, break it into multiple lines
    if len(full_string) > max_line_length:
        # Strategy: Split into logical groups and create multiple lines
        line1_parts = []
        line2_parts = []
        line3_parts = []
        
        # Group 1: Core experiment info (experiment, version, agent)
        line1_parts.extend([f"Experiment: {experiment}", f"Version: {version}", f"Agent: {agent}"])
        
        # Group 2: Domain info
        if aggregate_all_domains:
            line2_parts.append("Domains: All Domains")
        elif domains:
            if len(domains) == 1:
                line2_parts.append(f"Domain: {domains[0]}")
            else:
                line2_parts.append(f"Domains: {', '.join(sorted(domains))}")
        else:
            line2_parts.append("Domains: Aggregated")
        
        # Group 3: Prompt and temperature info
        if prompt_categories:
            if len(prompt_categories) == 1:
                line3_parts.append(f"Prompt: {prompt_categories[0]}")
            else:
                # If prompt categories are very long, they might need their own line
                prompt_str = f"Prompts: {', '.join(sorted(prompt_categories))}"
                if len(prompt_str) > max_line_length // 2:
                    line3_parts.append(prompt_str)
                else:
                    line2_parts.append(prompt_str) if not line2_parts or len(" | ".join(line2_parts + [prompt_str])) <= max_line_length else line3_parts.append(prompt_str)
        
        if temperature is not None:
            temp_str = f"Temperature: {temperature}"
            if line3_parts and len(" | ".join(line3_parts + [temp_str])) <= max_line_length:
                line3_parts.append(temp_str)
            elif line2_parts and len(" | ".join(line2_parts + [temp_str])) <= max_line_length:
                line2_parts.append(temp_str)
            else:
                line3_parts.append(temp_str)
        
        # Construct the multi-line string
        lines = []
        if line1_parts:
            lines.append(" | ".join(line1_parts))
        if line2_parts:
            lines.append(" | ".join(line2_parts))
        if line3_parts:
            lines.append(" | ".join(line3_parts))
        
        return "\n".join(lines)
    
    return full_string


def create_filename_components(
    experiment: str,
    version: str,
    agent: str,
    model_fits_df: pd.DataFrame,
    domains: Optional[List[str]] = None,
    prompt_categories: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    best_only: bool = False,
    metric: Optional[str] = None,
    aggregate_all_domains: bool = False,
    lr_filter_applied: bool = False,
) -> str:
    """Create filename component string similar to plot_facet_line_plots.py"""
    components = []
    
    # Add core components
    components.append(f"v{version}")
    components.append(experiment)
    components.append(agent.replace("-", "_"))
    
    # Add CBN model information
    if not model_fits_df.empty:
        # Get unique model types, parameter counts, learning rates, and loss functions
        model_types = sorted(model_fits_df['model'].unique())
        param_counts = sorted(model_fits_df['params_tying'].unique())
        learning_rates = sorted(model_fits_df['lr'].unique())
        loss_functions = sorted(model_fits_df['loss_function'].unique()) if 'loss_function' in model_fits_df.columns else []
        
        # Create model descriptor
        if len(model_types) == 1 and len(param_counts) == 1:
            # Single model type and parameter count
            components.append(f"{model_types[0]}{param_counts[0]}p")
        elif len(model_types) == 1:
            # Single model type, multiple parameter counts
            param_str = "_".join([f"{p}p" for p in param_counts])
            components.append(f"{model_types[0]}_{param_str}")
        elif len(param_counts) == 1:
            # Multiple model types, single parameter count
            model_str = "_".join(model_types)
            components.append(f"{model_str}_{param_counts[0]}p")
        else:
            # Multiple model types and parameter counts
            model_str = "_".join(model_types)
            param_str = "_".join([f"{p}p" for p in param_counts])
            components.append(f"{model_str}_{param_str}")
        
        # Add learning rate info:
        # - Always include if a filter was applied (even single default 0.1)
        # - Otherwise include only when multiple or single non-default
        if learning_rates:
            if lr_filter_applied or len(learning_rates) > 1 or (len(learning_rates) == 1 and learning_rates[0] != 0.1):
                if len(learning_rates) == 1:
                    lr_str = f"lr{learning_rates[0]:g}".replace(".", "p")
                    components.append(lr_str)
                else:
                    lr_strs = [f"lr{lr:g}".replace(".", "p") for lr in learning_rates]
                    components.append("_".join(lr_strs))
        
        # Add loss function information if not all default (mse)
        if len(loss_functions) == 1 and loss_functions[0] != "mse":
            components.append(loss_functions[0])
        elif len(loss_functions) > 1:
            components.append("_".join(loss_functions))
    
    # Add domains
    if aggregate_all_domains:
        components.append("all_domains")
    elif domains:
        if len(domains) == 1:
            components.append(domains[0])
        else:
            components.append("_".join(sorted(domains)))
    else:
        components.append("aggregated")
    
    # Add prompt categories
    if prompt_categories:
        if len(prompt_categories) == 1:
            components.append(prompt_categories[0])
        else:
            components.append("_".join(sorted(prompt_categories)))
    
    # Add temperature
    if temperature is not None:
        components.append(f"temp{temperature}")
    
    # Add best model indicator
    if best_only and metric:
        components.append(f"best_{metric}")
    
    return "_".join(components)


def create_cbn_output_subdirectory(
    base_output_dir: Path,
    model_fits_df: pd.DataFrame,
) -> Path:
    """Create organized subdirectory structure based on CBN model types and parameter counts"""
    if model_fits_df.empty:
        return base_output_dir
    
    # Get unique model types and parameter counts
    model_types = sorted(model_fits_df['model'].unique())
    param_counts = sorted(model_fits_df['params_tying'].unique())
    
    # Determine model subdirectory
    if len(model_types) == 1:
        if model_types[0] == 'logistic':
            model_subdir = "logistic"
        elif model_types[0] == 'noisy_or':
            model_subdir = "noisy_or"
        else:
            model_subdir = "other"
    else:
        # Multiple model types - use "both_models"
        model_subdir = "both_models"
    
    # Determine parameter subdirectory
    if len(param_counts) == 1:
        param_subdir = f"{param_counts[0]}p"
    else:
        # Multiple parameter counts - use "all_params"
        param_subdir = "all_params"
    
    # Create the full subdirectory path
    output_dir = base_output_dir / model_subdir / param_subdir
    
    return output_dir


def aggregate_domains_if_needed(
    df: pd.DataFrame, 
    domains: Optional[List[str]] = None,
    aggregate_all_domains: bool = False,
    is_model_data: bool = False
) -> pd.DataFrame:
    """Aggregate data across domains if needed, preserving individual data points for uncertainty bands"""
    logger = logging.getLogger(__name__)
    
    if df.empty:
        logger.warning("Input dataframe is empty - returning as is")
        return df
    
    if aggregate_all_domains or domains is None:
        # Aggregate all domains
        if 'domain' in df.columns:
            # For model data that already has domain=None (all-domain fits), don't aggregate
            if df['domain'].isna().all() or (df['domain'] == 'all_domains').all():
                logger.info(f"Data already aggregated (domain=None/all_domains): {len(df)} rows")
                if aggregate_all_domains:
                    df = df.copy()
                    df['domain'] = 'all_domains'
                return df
            
            # For model data with --domains all, check if we should aggregate domain-specific predictions
            if is_model_data and aggregate_all_domains:
                # This means we have domain-specific model fits but user wants --domains all
                # We should average the predictions across domains
                logger.info(f"Averaging {len(df)} domain-specific CBN predictions across domains")
                
                # Group by all columns except domain and average the predictions
                group_cols = [col for col in df.columns if col not in ['domain', 'model_prediction']]
                
                # Create aggregation functions
                agg_funcs = {'model_prediction': 'mean'}
                
                # For other columns, take first non-null value
                for col in df.columns:
                    if col not in group_cols and col not in agg_funcs and col != 'domain':
                        agg_funcs[col] = 'first'
                
                try:
                    aggregated_df = df.groupby(group_cols, as_index=False, dropna=False).agg(agg_funcs)
                    aggregated_df['domain'] = 'all_domains'
                    logger.info(f"Aggregated CBN predictions to {len(aggregated_df)} rows across all domains")
                    return aggregated_df
                    
                except Exception as e:
                    logger.error(f"CBN aggregation failed: {e}")
                    logger.info("Returning original CBN data without aggregation")
                    return df
            
            # For agent data, we want to preserve individual data points for uncertainty bands
            # Instead of aggregating to means, we just relabel the domain column
            if 'subject' in df.columns and 'task' in df.columns:
                
                # Simply relabel domain to indicate aggregation, but keep all individual rows
                df_copy = df.copy()
                if aggregate_all_domains:
                    df_copy['domain'] = 'all_domains'
                    logger.info(f"Relabeled {len(df_copy)} individual data points as 'all_domains' (preserving for uncertainty)")
                else:
                    df_copy['domain'] = 'aggregated'
                    logger.info(f"Relabeled {len(df_copy)} individual data points as 'aggregated' (preserving for uncertainty)")
                
                return df_copy
                
            else:
                logger.warning("Missing required columns for aggregation (subject, task)")
                return df
    
    # No aggregation needed - return original data
    return df


def combine_agent_and_model_data(
    agent_df: pd.DataFrame,
    model_predictions_df: pd.DataFrame,
    domains: Optional[List[str]] = None,
    aggregate_all_domains: bool = False,
) -> pd.DataFrame:
    """Combine agent predictions with model predictions into single DataFrame"""
    logger = logging.getLogger(__name__)
    
    # Apply domain aggregation if needed
    agent_data = aggregate_domains_if_needed(agent_df, domains, aggregate_all_domains, is_model_data=False)
    model_data = aggregate_domains_if_needed(model_predictions_df, domains, aggregate_all_domains, is_model_data=True)
    
    # Prepare agent data
    agent_data = agent_data.copy()
    agent_data['prediction_type'] = 'Agent'
    agent_data['prediction_value'] = agent_data['likelihood-rating']
    agent_data['model'] = 'Agent'
    agent_data['params_tying'] = None
    agent_data['loss'] = None
    agent_data['r2'] = None
    agent_data['aic'] = None
    
    # Prepare model data with separate prediction types for logistic vs noisy_or
    model_data = model_data.copy()
    
    # Create model-specific prediction types for better styling
    def create_prediction_type(row):
        if row['model'] == 'logistic':
            return 'CBN-Logistic'
        elif row['model'] == 'noisy_or':
            return 'CBN-NoisyOR'
        else:
            return 'CBN'
    
    model_data['prediction_type'] = model_data.apply(create_prediction_type, axis=1)
    model_data['prediction_value'] = model_data['model_prediction']
    model_data['subject'] = model_data['agent']
    
    # Create consistent columns
    common_cols = ['task', 'subject', 'domain', 'prompt_category', 'prediction_type', 'prediction_value', 'model', 'params_tying', 'loss', 'loss_function', 'lr', 'r2', 'aic']
    
    # Add missing columns with defaults
    for col in common_cols:
        if col not in agent_data.columns:
            agent_data[col] = None
        if col not in model_data.columns:
            model_data[col] = None
    
    # Select and combine
    combined_df = pd.concat([
        agent_data[common_cols],
        model_data[common_cols]
    ], ignore_index=True)
    
    # Add model label for legend with R², loss, AIC values and domain information
    def create_model_label(row):
        if row['prediction_type'] == 'Agent':
            # Use actual agent name instead of generic "Agent"
            agent_name = row['subject']
            if domains and len(domains) > 1 and not aggregate_all_domains:
                # Multiple specific domains - include domain in agent label
                return f"Agent ({agent_name}, {row['domain']})"
            else:
                return f'Agent ({agent_name})'
        else:
            base_label = f"{row['model']} ({row['params_tying']}p)"
            
            # Add learning rate if available and not default
            if row['lr'] is not None and not pd.isna(row['lr']) and row['lr'] != 0.1:
                base_label += f", LR={row['lr']:g}"
            
            # Add LOOCV R² value if available
            if row.get('r2') is not None and not pd.isna(row['r2']):
                base_label += f" (LOOCV R²={row['r2']:.3f}"
                
                # Add loss if available
                if row['loss'] is not None and not pd.isna(row['loss']):
                    base_label += f", Loss={row['loss']:.3f}"
                
                # Add AIC if available
                if row['aic'] is not None and not pd.isna(row['aic']):
                    base_label += f", AIC={row['aic']:.1f}"
                
                base_label += ")"
            
            # Add domain information if showing multiple specific domains
            if domains and len(domains) > 1 and not aggregate_all_domains:
                base_label += f" ({row['domain']})"
            
            return base_label
    
    combined_df['model_label'] = combined_df.apply(create_model_label, axis=1)
    
    logger.info(f"Combined data: {len(combined_df)} rows")
    return combined_df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create faceted plots comparing agent predictions with fitted CBN model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot specific agent with aggregated domains (default)
  python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2

  # Show separate lines for specific domains  
  python scripts/plot_agent_vs_cbn_predictions.py --agent claude-3-opus --experiment rw17_indep_causes --version 2 --domains weather economy --prompt-categories numeric

  # Aggregate all domains explicitly into one line
  python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --domains all

  # Show only best fitting models (by AIC)
  python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --best-only --metric aic

  # Facet by specific dimensions
  python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --facet-by domain prompt_category

  # Disable uncertainty bands for cleaner plots
  python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --no-uncertainty

  # Custom uncertainty visualization
  python scripts/plot_agent_vs_cbn_predictions.py --agent humans --experiment rw17_indep_causes --version 2 --uncertainty-type se --uncertainty-alpha 0.3

  # Filter to specific model types only (filename: gpt_4o_lineplot_v2_rw17_indep_causes_gpt_4o_logistic3p_...)
  python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --model-types logistic --param-counts 3

  # Compare logistic vs noisy-or models (filename: gpt_4o_lineplot_v2_rw17_indep_causes_gpt_4o_logistic_noisy_or_3p_5p_...)
  python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --model-types logistic noisy_or --param-counts 3 5

  # Position legend on the right (for smaller legends)
  python scripts/plot_agent_vs_cbn_predictions.py --agent gpt-4o --experiment rw17_indep_causes --version 2 --legend-position right

    # Manifest-driven mode (publication-safe): use exact winners_with_params.csv
    # Single agent
    python scripts/plot_agent_vs_cbn_predictions.py \
        --winners-manifest results/parameter_analysis/rw17_indep_causes/v2_noisy_or_pcnum_p3-4_lr0.1/winners_with_params.csv \
        --experiment rw17_indep_causes \
        --version 2 \
        --agent gpt-4o

    # Multiple agents subset
    python scripts/plot_agent_vs_cbn_predictions.py \
        --winners-manifest results/parameter_analysis/rw17_indep_causes/v2_noisy_or_pcnum_p3-4_lr0.1/winners_with_params.csv \
        --experiment rw17_indep_causes \
        --version 2 \
        --agents gpt-4o claude-3-opus-20240229 gemini-1.5-pro

    # Random Abstract example
    python scripts/plot_agent_vs_cbn_predictions.py \
        --winners-manifest results/parameter_analysis/random_abstract/v1_noisy_or_pcnum_p3-4_lr0.1_noh/winners_with_params.csv \
        --experiment random_abstract \
        --version 1 \
        --agents gemini-1.5-flash \
        --no-show

        # Manifest-driven using --tag (auto-resolve CSV path)
        # Single agent
        python scripts/plot_agent_vs_cbn_predictions.py \
            --experiment rw17_indep_causes \
            --version 2 \
            --tag v2_noisy_or_pcnum_p3-4_lr0.1 \
            --agent gpt-4o

        # Multiple agents subset
        python scripts/plot_agent_vs_cbn_predictions.py \
            --experiment rw17_indep_causes \
            --version 2 \
            --tag v2_noisy_or_pcnum_p3-4_lr0.1 \
            --agents gpt-4o claude-3-opus-20240229 gemini-1.5-pro
        """,
    )

    parser.add_argument(
        "--source",
        choices=["manifest", "index"],
        help="Optionally force source of model params: 'manifest' uses winners_with_params.csv, 'index' uses fit_index.parquet. If omitted, auto: manifest when --tag/--winners-manifest provided, else index.",
    )

    parser.add_argument(
        "--agent",
    required=False,
        help="Agent/subject name (e.g., 'gpt-4o', 'claude-3-opus')",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name (e.g., 'rw17_indep_causes')",
    )

    parser.add_argument(
        "--version",
    required=True,
        help="Version number (e.g., '2')",
    )

    parser.add_argument(
        "--graph-type",
        choices=["collider", "fork", "chain"],
        default="collider",
        help="Graph type (default: collider)",
    )

    parser.add_argument(
        "--pipeline-mode",
        choices=["llm_with_humans", "llm", "humans"],
        default="llm_with_humans",
        help="Pipeline mode to load data from (default: llm_with_humans)",
    )

    parser.add_argument(
        "--no-roman-numerals",
        action="store_true",
        help="Don't use Roman numerals version",
    )

    parser.add_argument(
        "--no-aggregated",
        action="store_true",
        help="Don't use aggregated human responses",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature filter for LLM responses",
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        help="List of domains to include (e.g., 'weather economy') or 'all' to aggregate all domains into one line",
    )

    parser.add_argument(
        "--prompt-categories",
        nargs="+",
        help="List of prompt categories to include (e.g., 'numeric')",
    )

    parser.add_argument(
        "--facet-by",
        nargs="+",
        help="Column(s) to facet by (e.g., 'domain' or 'domain prompt_category')",
    )

    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Show only best fitting models per condition",
    )

    parser.add_argument(
        "--metric",
        choices=["loss", "aic", "bic", "rmse"],
        default="aic",
        help="Metric to use for best-only selection (must exist in index; default: aic)",
    )

    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["logistic", "noisy_or"],
        help="Filter to specific model types (e.g., 'logistic' or 'logistic noisy_or')",
    )

    parser.add_argument(
        "--param-counts",
        nargs="+",
        choices=["3", "4", "5"],
        help="Filter to specific parameter counts (e.g., '3' or '3 5')",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for plots (default: results/plots/agent_vs_cbn)",
    )

    parser.add_argument(
        "--title",
        help="Title prefix for the plots",
    )

    parser.add_argument(
        "--filename-suffix",
        help="Custom suffix to add to the filename",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    parser.add_argument("--no-show", action="store_true", help="Don't display plots (useful for batch processing)")
    
    parser.add_argument(
        "--title-line-length",
        type=int,
        default=80,
        help="Maximum length of title lines before breaking (default: 80)",
    )
    
    parser.add_argument(
        "--title-fontsize",
        type=int,
        default=None,
        help="Override title font size (default: matplotlib default)",
    )
    
    parser.add_argument(
        "--show-uncertainty",
        action="store_true",
        default=True,
        help="Show uncertainty bands around agent predictions (default: True)",
    )
    
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Disable uncertainty bands around agent predictions",
    )
    
    parser.add_argument(
        "--uncertainty-type",
        choices=["ci", "se", "sd", "pi"],
        default="ci",
        help="Type of uncertainty to show: ci (confidence interval), se (standard error), sd (standard deviation), pi (percentile interval) (default: ci)",
    )
    
    parser.add_argument(
        "--uncertainty-level",
        type=float,
        default=95,
        help="Confidence level for CI or percentile level for PI (default: 95)",
    )
    
    parser.add_argument(
        "--uncertainty-alpha",
        type=float,
        default=0.2,
        help="Alpha transparency for uncertainty bands (default: 0.2)",
    )

    parser.add_argument(
        "--legend-position",
        choices=["right", "bottom", "top", "left"],
        default="bottom",
        help="Position of the legend (default: bottom)",
    )

    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Filter to specific learning rates (e.g., '0.01 0.1 1.0'). If not specified, shows all available.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Alias for specifying a single learning rate (equivalent to --learning-rates <value>).",
    )

    parser.add_argument(
    "--loss-functions",
    nargs="+",
    help="Filter to specific loss functions (index column loss_name, e.g. 'mse huber').",
    )

    # Manifest-driven mode (publication-safe): use winners_with_params.csv as ground truth
    parser.add_argument(
        "--winners-manifest",
        help="Path to results/parameter_analysis/.../winners_with_params.csv (bypasses index selection)",
    )
    parser.add_argument(
        "--tag",
        help="Parameter-analysis tag (folder name under results/parameter_analysis/<experiment>/). If provided, the script will auto-resolve winners_with_params.csv from --experiment and --tag.",
    )
    parser.add_argument(
        "--humans-mode",
        choices=["all", "aggregated", "pooled", "individual"],
        default="all",
        help="Filter human agents in the manifest: aggregated ('humans'), pooled ('humans-pooled'), or individual ('human-<id>'). Non-human agents are always included.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        help="Subset of agents to plot from the winners manifest (omit to use --agent)",
    )
    parser.add_argument(
        "--exclude-agents",
        nargs="+",
        help="Agents to exclude (useful with --agents all)",
    )
    # Plotting layout for multi-agent selection
    try:
        # Python 3.9+ supports BooleanOptionalAction
        parser.add_argument(
            "--same-plot",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="If multiple agents are provided, plot them in the same figure (default: True). Use --no-same-plot to create separate plots per agent.",
        )
    except Exception:
        # Fallback: separate flags
        parser.add_argument(
            "--same-plot",
            action="store_true",
            default=True,
            help="If multiple agents are provided, plot them in the same figure (default: True).",
        )
        parser.add_argument(
            "--no-same-plot",
            action="store_true",
            help="Disable same-plot behavior for multiple agents; create separate plots per agent.",
        )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Harmonize learning rate args
    if getattr(args, "learning_rate", None) is not None:
        if args.learning_rates is None:
            args.learning_rates = [args.learning_rate]
        else:
            # Ensure uniqueness
            if args.learning_rate not in args.learning_rates:
                args.learning_rates.append(args.learning_rate)

    # Initialize paths
    paths = PathManager()

    # Manifest-driven publication-safe mode (explicit or auto)
    force_manifest = args.source == "manifest"
    force_index = args.source == "index"

    if force_manifest and not (args.tag or args.winners_manifest):
        raise ValueError("--source manifest requires --tag or --winners-manifest to be provided")

    if (args.tag or args.winners_manifest) and not force_index:
        # Resolve winners CSV path from either --tag or --winners-manifest
        winners_path: Path
        tag: str
        if args.tag:
            tag = args.tag
            winners_path = (
                paths.base_dir
                / "results"
                / "parameter_analysis"
                / args.experiment
                / tag
                / "winners_with_params.csv"
            )
        else:
            # Backward compatible: --winners-manifest can be a full CSV path, a directory path, or a tag string
            raw = Path(args.winners_manifest)
            if raw.suffix.lower() == ".csv" and raw.exists():
                winners_path = raw
                tag = winners_path.parent.name
            else:
                # If it's a directory or a tag, construct the CSV path using --experiment
                if raw.exists() and raw.is_dir():
                    winners_path = raw / "winners_with_params.csv"
                    tag = raw.name
                else:
                    # Treat provided string as a tag under the experiment
                    tag = args.winners_manifest
                    winners_path = (
                        paths.base_dir
                        / "results"
                        / "parameter_analysis"
                        / args.experiment
                        / tag
                        / "winners_with_params.csv"
                    )

        if not winners_path.exists():
            raise FileNotFoundError(f"winners_with_params.csv not found at: {winners_path}")

        winners_df = pd.read_csv(winners_path)
        # Ensure a 'link' column exists; infer if missing
        if "link" not in winners_df.columns:
            logger = logging.getLogger(__name__)
            # Try common alternatives
            alt_cols = ["model", "link_type", "link_name"]
            found_alt = None
            for alt in alt_cols:
                if alt in winners_df.columns:
                    found_alt = alt
                    break
                # case-insensitive match
                for c in winners_df.columns:
                    if c.lower() == alt:
                        found_alt = c
                        break
                if found_alt:
                    break
            if found_alt:
                winners_df["link"] = winners_df[found_alt]
                logger.info(f"Inferred 'link' from manifest column '{found_alt}'")
            else:
                # Infer from tag or parameter columns
                tl = str(tag).lower()
                inferred = None
                if "noisy_or" in tl or "noisyor" in tl:
                    inferred = "noisy_or"
                elif "logistic" in tl:
                    inferred = "logistic"
                else:
                    # Heuristic: if noisy-or params are present, assume noisy_or
                    if all(col in winners_df.columns for col in ["b", "m1", "m2", "pC1", "pC2"]):
                        inferred = "noisy_or"
                if inferred:
                    winners_df["link"] = inferred
                    logger.info(f"Inferred 'link' from tag/params: {inferred}")
                else:
                    raise ValueError(
                        "Manifest missing required column 'link' and could not infer it. "
                        "Provide a 'link' column (values: 'noisy_or' or 'logistic'), or use --tag/--winners-manifest containing 'noisy_or' or 'logistic'."
                    )

        # Ensure 'agent' column exists; infer if missing
        if "agent" not in winners_df.columns:
            for alt in ["subject", "model_name", "name"]:
                if alt in winners_df.columns:
                    winners_df["agent"] = winners_df[alt]
                    logger.info(f"Inferred 'agent' from manifest column '{alt}'")
                    break
        # Ensure 'params_tying' column exists; infer if missing
        if "params_tying" not in winners_df.columns:
            for alt in ["pcnum", "param_count", "params", "n_params", "k"]:
                if alt in winners_df.columns:
                    winners_df["params_tying"] = winners_df[alt]
                    logger.info(f"Inferred 'params_tying' from manifest column '{alt}'")
                    break
        # Normalize params_tying to integers if possible
        if "params_tying" in winners_df.columns:
            try:
                winners_df["params_tying"] = winners_df["params_tying"].astype(int)
            except Exception:
                pass

        # Basic required columns
        needed = {"agent", "link", "params_tying", "b", "m1", "m2", "pC1", "pC2"}
        missing = needed - set(winners_df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        # Only use noisy_or winners (as in thesis tables/heatmaps)
        winners_df = winners_df[winners_df["link"] == "noisy_or"].copy()
        if winners_df.empty:
            raise ValueError("No noisy_or winners in manifest after filtering")

        # Determine which agents to plot (support --agents all and --exclude-agents)
        agents_to_plot: List[str]
        if args.agents:
            # Support keyword 'all'
            if any(a.lower() == "all" for a in args.agents):
                agents_to_plot = sorted(winners_df["agent"].dropna().astype(str).unique().tolist())
            else:
                agents_to_plot = args.agents
        elif args.agent:
            agents_to_plot = [args.agent]
        else:
            raise ValueError("Provide --agents or --agent when using --winners-manifest")

        # Apply blacklist exclusion if provided
        if args.exclude_agents:
            excl = set([e.strip() for e in args.exclude_agents])
            agents_to_plot = [a for a in agents_to_plot if a not in excl]
            if not agents_to_plot:
                raise ValueError("After applying --exclude-agents, no agents remain to plot")

        # Default prompt category handling
        if "prompt_category" not in winners_df.columns:
            winners_df["prompt_category"] = "numeric"

        # If user provided prompt-categories, filter manifest rows using synonym expansion
        if args.prompt_categories:
            syns = _expand_prompt_category_synonyms(args.prompt_categories)
            if syns:
                winners_df = winners_df[winners_df["prompt_category"].astype(str).str.lower().isin(syns)].copy()

        # Determine base output dir
        if args.output_dir:
            base_output_dir = Path(args.output_dir)
        else:
            base_output_dir = paths.base_dir / "results" / "plots" / "agent_vs_cbn"
        out_dir = (base_output_dir / args.experiment / tag)
        # If humans-mode is set and tag doesn't already encode it (hm-*), add a subdir to avoid collisions
        if args.humans_mode != "all" and not any(tag.endswith(suf) or ("_" + suf) in tag for suf in ("hm-agg", "hm-pooled", "hm-indiv")):
            hm_map = {"aggregated": "hm-agg", "pooled": "hm-pooled", "individual": "hm-indiv"}
            out_dir = out_dir / hm_map.get(args.humans_mode, args.humans_mode)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Iterate by (prompt_category x domain) and either combine agents into one plot or one-per-agent
        roman = _roman_order()

        # Normalize domain to a grouping key
        def _domain_key(val: Any) -> str:
            sval = str(val).strip().lower()
            return "all" if sval in ("all", "nan", "none", "") else sval

        # Apply humans-mode filter first (keeps non-human agents). Then apply agents subset.
        def _keep_by_mode(agent: str) -> bool:
            if args.humans_mode == "all":
                return True
            s = str(agent).strip().lower()
            is_aggr = s == "humans"
            is_pooled = s in {"humans-pooled", "human-pooled"}
            is_indiv = s.startswith("human-") or s.startswith("humans-")
            is_human = is_aggr or is_pooled or is_indiv
            if not is_human:
                return True
            return (
                (args.humans_mode == "aggregated" and is_aggr)
                or (args.humans_mode == "pooled" and is_pooled)
                or (args.humans_mode == "individual" and is_indiv)
            )

        filtered_winners = winners_df[winners_df["agent"].apply(_keep_by_mode)].copy()
        selected_winners = filtered_winners[filtered_winners["agent"].isin(agents_to_plot)].copy()
        if "domain" in selected_winners.columns:
            domain_series = selected_winners["domain"]
        else:
            # No domain column in manifest; treat as 'all' for every row
            domain_series = pd.Series(["all"] * len(selected_winners), index=selected_winners.index)
        selected_winners["_domain_key"] = domain_series.apply(_domain_key)

        # If same-plot flag was provided via fallback (--no-same-plot), respect it
        same_plot_flag = getattr(args, "same_plot", True)
        if hasattr(args, "no_same_plot") and getattr(args, "no_same_plot"):
            same_plot_flag = False

        # Normalize prompt category for grouping: map synonyms to canonical labels
        winners_pc = selected_winners.copy()
        def _pc_canon(val: Any) -> str:
            s = str(val).strip().lower()
            if s in NUMERIC_SYNS or s == "numeric":
                return "numeric"
            if s in COT_SYNS or s == "cot":
                return "cot"
            return s
        winners_pc["_pc_canon"] = winners_pc["prompt_category"].apply(_pc_canon)
        # Grouping: for same-plot, collapse across domains; otherwise, keep per-domain
        group_by_cols = ["_pc_canon"] if same_plot_flag else ["_pc_canon", "_domain_key"]
        for group_key, group in winners_pc.groupby(group_by_cols, dropna=False):
            if same_plot_flag:
                # Mixed domains allowed; determine unique domains in this group
                prompt_cat = group_key if not isinstance(group_key, tuple) else group_key[0]
                dom_keys = sorted({str(d) for d in group.get("_domain_key", pd.Series(dtype=str)).unique().tolist()})
                mixed_domains = len(dom_keys) > 1
                # Build combined plot across all agents (and domains) in this prompt-category
                combined_frames = []
                agents_in_group = group["agent"].astype(str).unique().tolist()
                for agent_name in agents_in_group:
                    # iterate per agent-domain row so we keep domain-specific filtering
                    for _, mrow in group[group["agent"] == agent_name].iterrows():
                        dom_key = str(mrow.get("_domain_key", "all"))
                        pooled = dom_key == "all"
                        domains_filter = None if pooled else [dom_key]

                        # Load agent data for this agent-domain
                        agent_df = load_agent_data(
                            paths,
                            version=args.version,
                            experiment_name=args.experiment,
                            agent=agent_name,
                            graph_type=args.graph_type,
                            use_roman_numerals=not args.no_roman_numerals,
                            use_aggregated=not args.no_aggregated,
                            pipeline_mode=args.pipeline_mode,
                            temperature_filter=args.temperature,
                            domains=domains_filter,
                            prompt_categories=[prompt_cat],
                        )

                        tasks_present = [t for t in roman if t in agent_df["task"].astype(str).unique().tolist()]

                        params = {k: float(mrow[k]) for k in ["b", "m1", "m2", "pC1", "pC2"]}
                        preds = _predict_noisy_or_tasks(params, tasks_present)
                        model_rows = []
                        for t in tasks_present:
                            model_rows.append({
                                'task': t,
                                'model_prediction': preds[t],
                                'agent': agent_name,
                                'prompt_category': prompt_cat,
                                'domain': None if pooled else dom_key,
                                'model': 'noisy_or',
                                'params_tying': int(mrow['params_tying']) if not pd.isna(mrow['params_tying']) else None,
                                'loss': None,
                                'loss_function': None,
                                'lr': None,
                                'r2': float(mrow['loocv_r2']) if 'loocv_r2' in mrow and pd.notna(mrow['loocv_r2']) else None,
                                'loocv_r2': float(mrow['loocv_r2']) if 'loocv_r2' in mrow and pd.notna(mrow['loocv_r2']) else None,
                                'aic': None,
                                'file_path': str(winners_path),
                            })
                        model_predictions_df = pd.DataFrame(model_rows)

                        combined_df = combine_agent_and_model_data(
                            agent_df,
                            model_predictions_df,
                            domains=domains_filter,
                            aggregate_all_domains=pooled,
                        )
                        # Ensure legend uses detailed labels (includes LOOCV R² when present)
                        combined_df["legend_label"] = combined_df.get("model_label", None)
                        combined_frames.append(combined_df)

                if not combined_frames:
                    logger.warning(f"No data to plot for prompt={prompt_cat} (no agent/domain rows)")
                    continue

                plot_df = pd.concat(combined_frames, ignore_index=True)
                plot_df['likelihood-rating'] = plot_df['prediction_value']
                # Use shared subject label per agent-domain pair for consistent color per pair across overlays
                def _pair_label(r):
                    base = r['subject']  # subject in combined_df is agent name
                    d = str(r.get('domain', '')).strip().lower() if 'domain' in r else ''
                    return base if d in ('', 'all', 'nan', 'none') else f"{base} ({r['domain']})"
                plot_df['subject'] = plot_df.apply(_pair_label, axis=1)
                # Create deterministic distinct colors per subject pair
                unique_subjects = sorted(plot_df['subject'].astype(str).unique().tolist())
                subject_colors = _make_subject_colors(unique_subjects)

                # Compose title and filename
                # Title/filename domain label
                dom_label = 'mixed' if mixed_domains else (dom_keys[0] if dom_keys else 'all')
                agents_str = ", ".join(agents_in_group)
                # Respect custom title override
                title_prefix = (
                    args.title
                    if args.title
                    else f"{args.experiment} | {tag} | Agents: {agents_str} | Domain: {dom_label} | Prompt: {prompt_cat}"
                )
                filename_base = f"agents_manifest_{args.experiment}_{tag}_{dom_label}_{prompt_cat}"

                overlay_styles = {
                    'Agent': {'linestyle': '-', 'alpha': 0.9, 'marker': 'o', 'markersize': 7},
                    'CBN-NoisyOR': {'linestyle': '-.', 'alpha': 0.85, 'marker': '^', 'markersize': 6},
                }

                create_facet_line_plot(
                    df=plot_df,
                    facet_by=None,
                    overlay_by='prediction_type',
                    group_subjects=True,  # force a single subplot when overlaying multiple agents
                    output_dir=out_dir,
                    title_prefix=title_prefix,
                    filename_suffix=f"_{filename_base}",
                    subject_colors=subject_colors,
                    show=not args.no_show,
                    legend_position=args.legend_position,
                    show_uncertainty=(args.show_uncertainty and not args.no_uncertainty),
                    uncertainty_type=args.uncertainty_type,
                    uncertainty_level=args.uncertainty_level,
                    uncertainty_alpha=args.uncertainty_alpha,
                    overlay_styles=overlay_styles,
                )

                generated = out_dir / f"facet_lineplot_overlay_prediction_type_{filename_base}.pdf"
                desired = out_dir / f"{filename_base}.pdf"
                if generated.exists():
                    try:
                        import shutil
                        shutil.move(str(generated), str(desired))
                        logger.info(f"Plot saved: {desired}")
                    except Exception as e:
                        logger.warning(f"Rename failed ({generated} -> {desired}): {e}")
            else:
                # One plot per agent (original behavior)
                # Here, group has both _pc_norm and _domain_key, so derive these
                prompt_cat = group_key[0]
                dom_key = group_key[1]
                pooled = dom_key == "all"
                domains_filter = None if pooled else [dom_key]
                for _, mrow in group.iterrows():
                    agent_name = str(mrow['agent'])
                    # Load agent data with matching filters
                    agent_df = load_agent_data(
                        paths,
                        version=args.version,
                        experiment_name=args.experiment,
                        agent=agent_name,
                        graph_type=args.graph_type,
                        use_roman_numerals=not args.no_roman_numerals,
                        use_aggregated=not args.no_aggregated,
                        pipeline_mode=args.pipeline_mode,
                        temperature_filter=args.temperature,
                        domains=domains_filter,
                        prompt_categories=[prompt_cat],
                    )

                    tasks_present = [t for t in roman if t in agent_df["task"].astype(str).unique().tolist()]

                    params = {k: float(mrow[k]) for k in ["b", "m1", "m2", "pC1", "pC2"]}
                    preds = _predict_noisy_or_tasks(params, tasks_present)
                    model_rows = []
                    for t in tasks_present:
                        model_rows.append({
                            'task': t,
                            'model_prediction': preds[t],
                            'agent': agent_name,
                            'prompt_category': prompt_cat,
                            'domain': None if pooled else dom_key,
                            'model': 'noisy_or',
                            'params_tying': int(mrow['params_tying']) if not pd.isna(mrow['params_tying']) else None,
                            'loss': None,
                            'loss_function': None,
                            'lr': None,
                            'r2': float(mrow['loocv_r2']) if 'loocv_r2' in mrow and pd.notna(mrow['loocv_r2']) else None,
                            'loocv_r2': float(mrow['loocv_r2']) if 'loocv_r2' in mrow and pd.notna(mrow['loocv_r2']) else None,
                            'aic': None,
                            'file_path': str(winners_path),
                        })
                    model_predictions_df = pd.DataFrame(model_rows)

                    combined_df = combine_agent_and_model_data(
                        agent_df,
                        model_predictions_df,
                        domains=domains_filter,
                        aggregate_all_domains=pooled,
                    )
                    combined_df["legend_label"] = combined_df.get("model_label", None)

                    plot_df = combined_df.copy()
                    plot_df['likelihood-rating'] = plot_df['prediction_value']
                    pair_label = agent_name if pooled else f"{agent_name} ({dom_key})"
                    plot_df['subject'] = pair_label
                    subject_colors = _make_subject_colors([pair_label])

                    dom_label = 'all' if pooled else dom_key
                    # Respect custom title override
                    title_prefix = (
                        args.title
                        if args.title
                        else f"{args.experiment} | {tag} | Agent: {agent_name} | Domain: {dom_label} | Prompt: {prompt_cat}"
                    )
                    filename_base = f"{agent_name.replace('-', '_')}_manifest_{args.experiment}_{tag}_{dom_label}_{prompt_cat}"

                    overlay_styles = {
                        'Agent': {'linestyle': '-', 'alpha': 0.9, 'marker': 'o', 'markersize': 7},
                        'CBN-NoisyOR': {'linestyle': '-.', 'alpha': 0.85, 'marker': '^', 'markersize': 6},
                    }

                    create_facet_line_plot(
                        df=plot_df,
                        facet_by=None,
                        overlay_by='prediction_type',
                        output_dir=out_dir,
                        title_prefix=title_prefix,
                        filename_suffix=f"_{filename_base}",
                        subject_colors=subject_colors,
                        show=not args.no_show,
                        legend_position=args.legend_position,
                        show_uncertainty=(args.show_uncertainty and not args.no_uncertainty),
                        uncertainty_type=args.uncertainty_type,
                        uncertainty_level=args.uncertainty_level,
                        uncertainty_alpha=args.uncertainty_alpha,
                        overlay_styles=overlay_styles,
                    )

                    generated = out_dir / f"facet_lineplot_overlay_prediction_type_{filename_base}.pdf"
                    desired = out_dir / f"{filename_base}.pdf"
                    if generated.exists():
                        try:
                            import shutil
                            shutil.move(str(generated), str(desired))
                            logger.info(f"Plot saved: {desired}")
                        except Exception as e:
                            logger.warning(f"Rename failed ({generated} -> {desired}): {e}")

        logger.info(f"Completed manifest-driven plots in: {out_dir}")
        return

    logger.info(f"Loading data for agent: {args.agent}")

    # Process domains argument - check for "all" option
    aggregate_all_domains = False
    domains_for_loading = args.domains

    if args.domains and len(args.domains) == 1 and args.domains[0].lower() == "all":
        aggregate_all_domains = True
        domains_for_loading = None  # Load all domains from data
        logger.info("Will aggregate all domains into one line")
    elif args.domains is None:
        aggregate_all_domains = False
        domains_for_loading = None  # Load all domains but aggregate by default
        logger.info("No domains specified - will aggregate all domains by default")
    else:
        logger.info(f"Will show separate lines for domains: {args.domains}")

    # Load agent data
    agent_df = load_agent_data(
        paths,
        version=args.version,
        experiment_name=args.experiment,
        agent=args.agent,
        graph_type=args.graph_type,
        use_roman_numerals=not args.no_roman_numerals,
        use_aggregated=not args.no_aggregated,
        pipeline_mode=args.pipeline_mode,
        temperature_filter=args.temperature,
        domains=domains_for_loading,
        prompt_categories=args.prompt_categories,
    )

    # Structured model fitting directory
    model_fitting_dir = paths.base_dir / "results" / "model_fitting" / args.experiment
    if not model_fitting_dir.exists():
        raise FileNotFoundError(f"Model fitting directory not found: {model_fitting_dir}")

    # Load model fits via index
    model_fits_df = load_indexed_model_fits(
        experiment_dir=model_fitting_dir,
        agent=args.agent,
        prompt_categories=args.prompt_categories,
        domains=domains_for_loading,
        aggregate_all_domains=aggregate_all_domains,
        model_types=args.model_types,
        param_counts=args.param_counts,
        loss_functions=args.loss_functions,
        learning_rates=args.learning_rates,
        best_only=args.best_only,
        metric=args.metric,
    )

    # Get unique tasks from agent data
    unique_tasks = sorted(agent_df['task'].unique())
    logger.info(f"Tasks: {unique_tasks}")

    # Generate model predictions
    model_predictions_df = generate_model_predictions(model_fits_df, unique_tasks)

    # Combine agent and model data with domain aggregation logic
    combined_df = combine_agent_and_model_data(
        agent_df, 
        model_predictions_df,
        domains=args.domains,
        aggregate_all_domains=aggregate_all_domains
    )

    # Check that we have both agent and model data
    agent_rows = len(combined_df[combined_df['prediction_type'] == 'Agent'])
    model_rows = len(combined_df[combined_df['prediction_type'].str.startswith('CBN')])

    if agent_rows == 0:
        logger.warning(f"No agent data found for {args.agent} with experiment={args.experiment}, version={args.version}, domains={args.domains}, prompt_categories={args.prompt_categories}")
    if model_rows == 0:
        logger.warning(f"No CBN model data found for {args.agent} with experiment={args.experiment}, version={args.version}, domains={args.domains}, prompt_categories={args.prompt_categories}")

    if agent_rows == 0 or model_rows == 0:
        logger.error("Cannot create plot without both agent and model data")
        sys.exit(1)

    logger.info(f"Loaded both agent ({agent_rows} rows) and CBN model ({model_rows} rows) data")

    # Determine uncertainty setting
    show_uncertainty = args.show_uncertainty and not args.no_uncertainty

    # Set base output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = paths.base_dir / "results" / "plots" / "agent_vs_cbn"
        if args.experiment:
            base_output_dir = base_output_dir / args.experiment
        
    # Create organized subdirectory based on CBN model information
    output_dir = create_cbn_output_subdirectory(base_output_dir, model_fits_df)

    # Create experimental condition string (include loss functions present in fits if not explicitly provided)
    inferred_losses = []
    if 'loss_function' in model_fits_df.columns:
        inferred_losses = [str(loss_name) for loss_name in model_fits_df['loss_function'].dropna().unique().tolist()]
    elif args.loss_functions:
        inferred_losses = args.loss_functions

    condition_str = create_experimental_condition_string(
            experiment=args.experiment,
            version=args.version,
            agent=args.agent,
            domains=args.domains,
            prompt_categories=args.prompt_categories,
            temperature=args.temperature,
            max_line_length=args.title_line_length,
            aggregate_all_domains=aggregate_all_domains,
            loss_functions=inferred_losses if inferred_losses else None,
        )

    # Create title with multi-line layout
    main_title = args.title if args.title else "Agent vs CBN Predictions"
    if args.best_only:
        main_title += f" (Best by {args.metric.upper()})"
    # If a custom title was provided, use it as the full title; otherwise, keep condition + default title
    full_title = (
        args.title
        if args.title
        else f"{condition_str}\n{main_title}"
    )

    # Create filename components (include loss functions)
    filename_components = create_filename_components(
            experiment=args.experiment,
            version=args.version,
            agent=args.agent,
            model_fits_df=model_fits_df,
            domains=args.domains,
            prompt_categories=args.prompt_categories,
            temperature=args.temperature,
            best_only=args.best_only,
            metric=args.metric if args.best_only else None,
            aggregate_all_domains=aggregate_all_domains,
            lr_filter_applied=bool(args.learning_rates),
        )

    # Create plot using the existing facet_lineplot function
    logger.info("Creating agent vs CBN prediction plot...")
        
    plot_df = combined_df.copy()
    plot_df['likelihood-rating'] = plot_df['prediction_value']
    # When plotting multiple agents, color by agent (pair) while legend shows detailed label
    # Create a separate legend label column but keep subject for color grouping
    plot_df['legend_label'] = plot_df['model_label']
    plot_df['subject'] = plot_df.apply(
        lambda r: (f"{r['subject']} ({r['domain']})" if r['prediction_type'] == 'Agent' and pd.notna(r['domain']) else r['subject'])
        if 'domain' in plot_df.columns else r['subject'],
        axis=1,
    )

    # Build deterministic distinct colors per subject (pair) for hue
    unique_subjects = sorted(plot_df['subject'].astype(str).unique().tolist())
    subject_colors = _make_subject_colors(unique_subjects)

    overlay_styles = {
        'Agent': {
            'linestyle': '-',
            'alpha': 0.9,
            'marker': 'o',
            'markersize': 7,
        },
        'CBN-Logistic': {
            'linestyle': '--',
            'alpha': 0.8,
            'marker': 's',  # Square for logistic
            'markersize': 6,
        },
        'CBN-NoisyOR': {
            'linestyle': '-.',
            'alpha': 0.8,
            'marker': '^',  # Triangle for noisy-OR
            'markersize': 6,
        }
    }

    # Build filename
    agent_filename = f"{args.agent.replace('-', '_')}_lineplot_{filename_components}"
        
    if args.filename_suffix:
        if not args.filename_suffix.startswith("_"):
            agent_filename += f"_{args.filename_suffix}"
        else:
            agent_filename += args.filename_suffix
        
    create_facet_line_plot(
            df=plot_df,
            facet_by=args.facet_by,
            overlay_by='prediction_type',  # Overlay agent vs CBN
            output_dir=output_dir,
            temperature_filter=args.temperature,
            title_prefix=full_title,
            x='task',
            y='likelihood-rating',
            filename_suffix=f"_{agent_filename}",  # This will replace default suffix
            subject_colors=subject_colors,
            overlay_styles=overlay_styles,
            show=not args.no_show,
            title_fontsize=args.title_fontsize,
            show_uncertainty=show_uncertainty,
            uncertainty_type=args.uncertainty_type,
            uncertainty_level=args.uncertainty_level,
            uncertainty_alpha=args.uncertainty_alpha,
            legend_position=args.legend_position,
        )
        
    generated_filename = f"facet_lineplot_overlay_prediction_type_{agent_filename}.pdf"
    desired_filename = f"{agent_filename}.pdf"
    generated_path = output_dir / generated_filename
    desired_path = output_dir / desired_filename
    if generated_path.exists():
        import shutil
        shutil.move(str(generated_path), str(desired_path))
        logger.info(f"Plot saved: {desired_path}")
    else:
        logger.info(f"Plot saved to: {output_dir}")

    # End main processing


if __name__ == "__main__":
    main()