#!/usr/bin/env python3
"""
Publication-Grade Agent vs CBN Plots (Manifest-Driven)
======================================================

This script creates single, publication-ready overlay plots that compare
agents' observed per-task means against CBN predictions from the
winners_with_params.csv manifest. By default, x-axis tick labels are
conditional probability expressions for each Roman task (I–XI).

Notes
-----
- Same-plot overlay: all selected agents (and domains) appear in one plot.
- Prompt-category synonyms normalized (numeric/pcnum, cot/pccot, ...).
- Humans are colored magenta; other subjects use a distinct palette.
- Title override supported via --title; otherwise uses a concise default.
- Graph cartoons can be optionally supported later via --use-graph-cartoons,
  but the default (and the current implementation) uses conditional labels.

Outputs
-------
results/plots/agent_vs_cbn_publication/<experiment>/<tag>/
    pub_overlay_<experiment>_<tag>_<prompt>_<domains>.pdf
    (and .png)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import numpy as np

import pandas as pd
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory

from causalign.analysis.model_fitting.tasks import roman_task_to_probability
from causalign.config.paths import PathManager
from causalign.analysis.model_fitting.data import load_processed_data, prepare_dataset


# Optional: TeX-like styling if available, but don't require LaTeX
try:
    from tueplots import bundles, fonts  # type: ignore
except Exception:
    _USE_TUEPLOTS = False
else:
    _USE_TUEPLOTS = True
    _tplt = bundles.neurips2023(nrows=1, ncols=1, rel_width=0.85, usetex=False, family="serif")
    _tplt["legend.title_fontsize"] = 12
    _tplt["font.size"] = 13
    _tplt["axes.labelsize"] = 13
    _tplt["axes.titlesize"] = 15
    _tplt["xtick.labelsize"] = 11
    _tplt["ytick.labelsize"] = 11
    _tplt["legend.fontsize"] = 11
    _fnt = fonts.neurips2022_tex(family="serif")
    mpl.rcParams.update({**_tplt, **_fnt})

if not _USE_TUEPLOTS:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
    })


# Roman task ordering used in the project
ROMAN_ORDER: List[str] = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"
]


# Prompt-category synonyms (case-insensitive)
NUMERIC_SYNS: Set[str] = {
    "numeric", "pcnum", "num", "single_numeric", "single_numeric_response",
}
COT_SYNS: Set[str] = {
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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _make_subject_colors(subjects: List[str]) -> Dict[str, Any]:
    n = len(subjects)
    if n <= 10:
        cmap = mpl.colormaps.get_cmap('tab10')
    elif n <= 20:
        cmap = mpl.colormaps.get_cmap('tab20')
    else:
        cmap = mpl.colormaps.get_cmap('hsv')
    colors = {sub: tuple(cmap(i)[:3]) for i, sub in enumerate(subjects)}
    # Humans => magenta
    for sub in subjects:
        s = str(sub).strip().lower()
        if s == "humans" or s.startswith("human"):
            colors[sub] = (1.0, 0.0, 1.0)
    return colors


def _cond_prob_label(task: str) -> str:
    r"""Return LaTeX math label for a Roman task's conditional probability.

    Uses TeX-friendly symbols (\mid instead of pipe) and wraps in $...$.
    """
    # Map tasks to (human-friendly) conditional probability forms
    mapping = {
        "VI": r"p(C_i{=}1\mid E{=}1,\, C_j{=}1)",
        "VII": r"p(C_i{=}1\mid E{=}1)",
        "VIII": r"p(C_i{=}1\mid E{=}1,\, C_j{=}0)",
        "IV": r"p(C_i{=}1\mid C_j{=}1)",
        "V": r"p(C_i{=}1\mid C_j{=}0)",
        "IX": r"p(C_i{=}1\mid E{=}0,\, C_j{=}1)",
        "X": r"p(C_i{=}1\mid E{=}0)",
        "XI": r"p(C_i{=}1\mid E{=}0,\, C_j{=}0)",
        "I": r"p(E{=}1\mid C_i{=}0,\, C_j{=}0)",
        "II": r"p(E{=}1\mid C_i{=}0,\, C_j{=}1)",
        "III": r"p(E{=}1\mid C_i{=}1,\, C_j{=}1)",
    }
    inner = mapping.get(task, task)
    return f"${inner}$"


def _task_labels_in_order() -> List[str]:
    return [_cond_prob_label(t) for t in ROMAN_ORDER]


# Optional: Graph cartoon helpers
def _roman_to_cartoon_basename(task: str) -> str:
    """Map Roman task to a canonical cartoon basename, e.g., 'graph_a'..'graph_k'.

    Assumes I..XI => a..k. Adjust here if your naming differs.
    """
    roman_to_letter = {
        "I": "a", "II": "b", "III": "c", "IV": "d", "V": "e",
        "VI": "f", "VII": "g", "VIII": "h", "IX": "i", "X": "j", "XI": "k",
    }
    letter = roman_to_letter.get(task, None)
    return f"01_graph_{letter}" if letter else task.lower()


def _find_cartoon_png(paths: PathManager, basename: str) -> Optional[Path]:
    """Search a few likely directories for a PNG named `<basename>.png`.

    Returns the first existing path or None if not found.
    """
    candidates = [
        paths.base_dir / "publication" / "graph_cartoons" / f"{basename}.png",
        paths.base_dir / "publication" / "build" / f"{basename}.png",
        paths.base_dir / "assets" / "graph_cartoons" / f"{basename}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _apply_graph_cartoons(ax: Axes, fig: Figure, paths: PathManager, use_cartoons: bool, tasks: List[str], zoom: float = 0.39) -> None:
    """If enabled, draw small PNG graph cartoons under each x-tick using AnnotationBbox.

    - Looks for PNG files named graph_a.png .. graph_k.png based on Roman tasks.
    - Adds extra bottom margin for visibility.
    - Falls back silently if images are missing.

        Tuning tips (examples):
        - Make images larger/smaller via `zoom`, e.g., zoom=0.22 for larger, zoom=0.15 for smaller.
        - If images get clipped at the bottom, increase bottom margin, e.g., `fig.subplots_adjust(bottom=0.33)`.
        - Vertical placement: adjust the y offset in AnnotationBbox coords (default -0.12). 
        Less negative raises images, more negative lowers them.
            For example, change `(i, -0.10)` or `(i, -0.16)` below.
    """
    if not use_cartoons:
        return
    try:
        # Increase bottom margin to make room for images
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.35))

        # Place images using blended transform: x in data coords, y in axes coords
        btf = blended_transform_factory(ax.transData, ax.transAxes)
        for i, task in enumerate(tasks):
            basename = _roman_to_cartoon_basename(task)
            img_path = _find_cartoon_png(paths, basename)
            if img_path is None:
                continue
            try:
                img = plt.imread(str(img_path))
            except Exception:
                continue
            oi = OffsetImage(img, zoom=zoom)
            # y = -0.12 places slightly below the axis; tweak zoom/offset as needed.
            # Example: ab = AnnotationBbox(oi, (i, -0.10), ...) to move images up a bit.
            ab = AnnotationBbox(oi, (i, -0.07), xycoords=btf, frameon=False, box_alignment=(0.5, 1.0))
            ax.add_artist(ab)
    except Exception:
        # Never fail plotting due to cartoons; just skip on errors
        pass


def _load_agent_data(
    paths: PathManager,
    version: str,
    experiment: str,
    agent: str,
    prompt_categories: Optional[List[str]],
    domain: Optional[str],
) -> pd.DataFrame:
    """Load agent per-task mean ratings and SE using project loaders.

    Returns columns: task, likelihood-rating (0-100), se (0-100), n
    """
    df = load_processed_data(
        paths,
        version=version,
        experiment_name=experiment,
        graph_type="collider",
        use_roman_numerals=True,
        use_aggregated=True,
        pipeline_mode="llm_with_humans",
    )
    doms = None if (domain is None or str(domain).lower() == "all") else [str(domain)]
    pcs = None
    if prompt_categories:
        syns = _expand_prompt_category_synonyms(prompt_categories)
        pcs = list(syns) if syns else None
    sub = prepare_dataset(df, agents=[agent], domains=doms, prompt_categories=pcs)
    if sub.empty:
        return pd.DataFrame(columns=["task", "likelihood-rating", "se", "n"])  # empty
    g = (
        sub.groupby("task", dropna=False)["response"].agg(["mean", "std", "count"]).reset_index()
        .rename(columns={"mean": "likelihood-rating", "count": "n"})
    )
    # SE with ddof=1 std; handle n<=1
    g["std"] = g["std"].fillna(0.0)
    g["se"] = g.apply(lambda r: (float(r["std"]) / np.sqrt(float(r["n"])) if float(r["n"]) > 1 else 0.0), axis=1)
    g.drop(columns=["std"], inplace=True)
    # Scale to [0,100] if needed (apply to both mean and se)
    if g["likelihood-rating"].max() <= 1.01:
        g["likelihood-rating"] = g["likelihood-rating"] * 100.0
        g["se"] = g["se"] * 100.0
    g["task"] = g["task"].astype(str)
    g = g[g["task"].isin(ROMAN_ORDER)].copy()
    return g


def _predict_noisy_or_tasks(params: Dict[str, float], tasks: List[str]) -> Dict[str, float]:
    """Compute noisy-or predictions for Roman tasks, return 0-100 scale."""
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publication-grade Agent vs CBN overlay plots (manifest-driven)")
    p.add_argument("--winners-manifest", help="Tag (under results/parameter_analysis/<experiment>/) or a path to winners_with_params.csv")
    p.add_argument("--experiment", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--agents", nargs="+", help="Agents to include; use 'all' to include all agents in manifest")
    p.add_argument("--exclude-agents", nargs="+", help="Agents to exclude")
    p.add_argument("--prompt-categories", nargs="+", default=["numeric"], help="Prompt categories to include (e.g., numeric, cot)")
    p.add_argument("--humans-mode", choices=["all", "aggregated", "pooled", "individual"], default="all",
                   help="How to handle humans in selection (non-human agents always included)")
    p.add_argument("--same-plot", action=argparse.BooleanOptionalAction, default=True,
                   help="Overlay all agents in one plot (default: True)")
    p.add_argument("--title", help="If provided, overrides the default title")
    p.add_argument("--output-dir", help="Base output dir (default: results/plots/agent_vs_cbn_publication)")
    p.add_argument("--use-graph-cartoons", action="store_true", help="Optional: replace x-ticks with small graph cartoons (not enabled by default)")
    p.add_argument("--domains", nargs="+", help="Domains to include (e.g., 'weather economy'), 'all' for all domains, or 'pooled' to aggregate across domains")
    # Publication knobs
    p.add_argument("--ylabel", default="p(query node is 1)", help="Y-axis label")
    p.add_argument("--legend-loc", default="lower center", help="Legend location (e.g., 'upper right', 'lower center')")
    p.add_argument("--legend-ncols", type=int, default=2, help="Number of columns in legend")
    p.add_argument("--fig-width", type=float, default=8.5, help="Figure width in inches")
    p.add_argument("--fig-height", type=float, default=5.0, help="Figure height in inches")
    p.add_argument("--cat-fig-width", type=float, default=3.5, help="Figure width for --by_reason_cat plots (default: 5.0)")
    p.add_argument("--cat-fig-height", type=float, default=5.0, help="Figure height for --by_reason_cat plots (default: 4.0)")
    p.add_argument("--usetex", action="store_true", help="Use LaTeX for text rendering (requires LaTeX installed)")
    p.add_argument("--show", action="store_true", help="Show the plot interactively")
    p.add_argument("--no-show", action="store_true", help="Do not display plots interactively (alias)")
    # Uncertainty bands for Agent lines
    p.add_argument("--uncertainty", action=argparse.BooleanOptionalAction, default=True, help="Show uncertainty bands for Agent lines (default: True)")
    p.add_argument("--uncertainty-level", type=float, default=95.0, help="Confidence level for bands (default: 95)")
    p.add_argument("--uncertainty-alpha", type=float, default=0.2, help="Alpha for uncertainty shading (default: 0.2)")
    # Per-reasoning-category plots
    p.add_argument("--by_reason_cat", action="store_true", help="Create a separate figure per reasoning category (I–III, IV–V, VI–VIII, IX–XI)")
    # Control which per-category figure(s) show a legend
    p.add_argument(
        "--legend_loc_figure_xs",
        nargs="+",
        help=(
            "Show a legend on these per-category figures. Choices: "
            "predictive, independence, effect-present, effect-absent, or all. "
            "You can pass multiple (e.g., --legend_loc_figure_xs predictive effect-present). "
            "Legends will be placed using --legend-loc and --legend-ncols."
        ),
    )
    # Label controls
    p.add_argument("--no-probs", action="store_true", help="Hide conditional probability expressions in x-tick labels")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _resolve_manifest_path(paths: PathManager, experiment: str, winners_manifest: str) -> Tuple[Path, str]:
    p = Path(winners_manifest)
    if p.suffix.lower() == ".csv":
        return p, p.parent.name
    # Otherwise treat as tag under results/parameter_analysis/<experiment>/
    tag = winners_manifest
    csv_path = paths.base_dir / "results" / "parameter_analysis" / experiment / tag / "winners_with_params.csv"
    return csv_path, tag


def _filter_humans_mode(agent: str, mode: str) -> bool:
    if mode == "all":
        return True
    s = str(agent).strip().lower()
    is_aggr = s == "humans"
    is_pooled = s in {"humans-pooled", "human-pooled"}
    is_indiv = s.startswith("human-") or s.startswith("humans-")
    is_human = is_aggr or is_pooled or is_indiv
    if not is_human:
        return True
    return (
        (mode == "aggregated" and is_aggr)
        or (mode == "pooled" and is_pooled)
        or (mode == "individual" and is_indiv)
    )


def main() -> int:
    args = _parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    paths = PathManager()
    winners_csv, tag = _resolve_manifest_path(paths, args.experiment, args.winners_manifest)
    if not winners_csv.exists():
        log.error(f"winners_with_params.csv not found: {winners_csv}")
        return 2

    winners_df = pd.read_csv(winners_csv)
    # Ensure essential columns exist (infer link if needed)
    if "link" not in winners_df.columns:
        inferred = None
        tl = str(tag).lower()
        if "noisy_or" in tl or "noisyor" in tl:
            inferred = "noisy_or"
        elif "logistic" in tl:
            inferred = "logistic"
        if inferred is None and all(c in winners_df.columns for c in ["b", "m1", "m2", "pC1", "pC2"]):
            inferred = "noisy_or"
        if inferred is None:
            log.error("Could not infer 'link' from manifest/tag; add 'link' column or include 'noisy_or'/'logistic' in tag.")
            return 2
        winners_df["link"] = inferred

    # Only use noisy_or winners (matches thesis figure focus)
    winners_df = winners_df[winners_df["link"] == "noisy_or"].copy()
    if winners_df.empty:
        log.error("No noisy_or winners in manifest after filtering")
        return 2

    # Prompt-category filter via synonyms
    if "prompt_category" not in winners_df.columns:
        winners_df["prompt_category"] = "numeric"
    syns = _expand_prompt_category_synonyms(args.prompt_categories)
    if syns:
        winners_df = winners_df[winners_df["prompt_category"].astype(str).str.lower().isin(syns)].copy()
    if winners_df.empty:
        log.error("No rows match the requested prompt-categories in manifest")
        return 2

    # Agent selection
    all_agents = winners_df["agent"].dropna().astype(str).unique().tolist() if "agent" in winners_df.columns else []
    if not args.agents or any(str(a).lower() == "all" for a in args.agents):
        agents_to_plot = all_agents
    else:
        agents_to_plot = [str(a) for a in args.agents]
    if args.exclude_agents:
        excl = {str(a) for a in args.exclude_agents}
        agents_to_plot = [a for a in agents_to_plot if a not in excl]
    if not agents_to_plot:
        log.error("No agents selected after filtering")
        return 2

    # Humans-mode filter
    winners_df = winners_df[winners_df["agent"].apply(lambda a: _filter_humans_mode(a, args.humans_mode))].copy()
    winners_df = winners_df[winners_df["agent"].astype(str).isin(agents_to_plot)].copy()
    if winners_df.empty:
        log.error("No winners rows remain after agent/humans-mode filters")
        return 2

    # Output directory
    base_out = Path(args.output_dir) if args.output_dir else (paths.base_dir / "results" / "plots" / "agent_vs_cbn_publication")
    out_dir = base_out / args.experiment / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collapse across domains for same-plot overlay per prompt-category
    winners_df["_pc_canon"] = winners_df["prompt_category"].astype(str).str.lower().map(
        lambda s: "numeric" if (s in NUMERIC_SYNS or s == "numeric") else ("cot" if (s in COT_SYNS or s == "cot") else s)
    )

    # Determine domains to plot
    def _available_domains_from_data() -> List[str]:
        try:
            df_all = load_processed_data(
                paths,
                version=args.version,
                experiment_name=args.experiment,
                graph_type="collider",
                use_roman_numerals=True,
                use_aggregated=True,
                pipeline_mode="llm_with_humans",
            )
            doms = (
                df_all["domain"].dropna().astype(str).unique().tolist()
                if "domain" in df_all.columns
                else []
            )
            return sorted(doms)
        except Exception:
            return []

    requested_domains: Optional[List[str]] = None
    pooled_mode: bool = False
    if args.domains:
        d0 = str(args.domains[0]).lower() if len(args.domains) == 1 else None
        if len(args.domains) == 1 and d0 == "all":
            requested_domains = _available_domains_from_data()
        elif len(args.domains) == 1 and d0 == "pooled":
            pooled_mode = True
            requested_domains = None
        else:
            requested_domains = [str(d) for d in args.domains]
    else:
        # infer from manifest; if none, fallback to data
        doms_manifest = winners_df.get("domain")
        if doms_manifest is not None:
            dlist = [str(d) for d in doms_manifest.dropna().unique().tolist()]
        else:
            dlist = []
        requested_domains = sorted(dlist) if dlist else _available_domains_from_data()

    # Prepare combined data for each prompt-category (single figure per prompt)
    for prompt_cat, group in winners_df.groupby("_pc_canon", dropna=False):
        combined_frames: List[pd.DataFrame] = []
        # Iterate each requested domain per agent, with pooled mode support
        for agent_name in agents_to_plot:
            if pooled_mode:
                # Pooled across domains
                agent_df = _load_agent_data(
                    paths=paths,
                    version=args.version,
                    experiment=args.experiment,
                    agent=agent_name,
                    prompt_categories=[str(prompt_cat)],
                    domain=None,  # pooled
                )
                tasks_present = [t for t in ROMAN_ORDER if t in agent_df["task"].astype(str).unique().tolist()]
                if tasks_present:
                    sel = group[group["agent"].astype(str) == str(agent_name)]
                    row_pooled = sel[sel["domain"].isna()] if "domain" in sel.columns else sel
                    use_row = row_pooled.iloc[0] if not row_pooled.empty else (sel.iloc[0] if not sel.empty else None)
                    model_df = pd.DataFrame([])
                    if use_row is not None:
                        params = {k: float(use_row[k]) for k in ["b", "m1", "m2", "pC1", "pC2"] if k in use_row and pd.notna(use_row[k])}
                        preds = _predict_noisy_or_tasks(params, tasks_present)
                        mrows: List[Dict[str, Any]] = []
                        for t in tasks_present:
                            mrows.append({
                                "task": t,
                                "subject": agent_name,  # no domain in legend
                                "domain": None,
                                "prompt_category": prompt_cat,
                                "prediction_type": "CBN",
                                "likelihood-rating": preds[t],
                                "model": "noisy_or",
                            })
                        model_df = pd.DataFrame(mrows)

                    arows: List[Dict[str, Any]] = []
                    for _, r in agent_df.iterrows():
                        arows.append({
                            "task": str(r["task"]),
                            "subject": agent_name,  # no domain in legend
                            "domain": None,
                            "prompt_category": prompt_cat,
                            "prediction_type": "Agent",
                            "likelihood-rating": float(r["likelihood-rating"]),
                            "se": float(r.get("se", 0.0)) if pd.notna(r.get("se", 0.0)) else 0.0,
                            "n": int(r.get("n", 0)) if pd.notna(r.get("n", np.nan)) else 0,
                            "model": "Agent",
                        })
                    agent_rows_df = pd.DataFrame(arows)
                    combined = pd.concat([agent_rows_df, model_df], ignore_index=True)
                    # annotate and store
                    group_map = {
                        "Predictive": ["I", "II", "III"],
                        "Independence": ["IV", "V"],
                        "Effect-Present": ["VI", "VII", "VIII"],
                        "Effect-Absent": ["IX", "X", "XI"],
                    }
                    combined["reasoning_group"] = None
                    for gname, gtasks in group_map.items():
                        combined.loc[combined["task"].isin(gtasks), "reasoning_group"] = gname
                    combined["task_label"] = combined["task"].map(_cond_prob_label)
                    combined_frames.append(combined)
            else:
                for dom_key in (requested_domains or ["all"]):
                    # Load agent data for the specific domain
                    agent_df = _load_agent_data(
                        paths=paths,
                        version=args.version,
                        experiment=args.experiment,
                        agent=agent_name,
                        prompt_categories=[str(prompt_cat)],
                        domain=dom_key,
                    )

                    tasks_present = [t for t in ROMAN_ORDER if t in agent_df["task"].astype(str).unique().tolist()]
                    if not tasks_present:
                        continue

                    # Try to find a winners row for this agent+domain; fallback to pooled
                    sel = group[group["agent"].astype(str) == str(agent_name)]
                    row_exact = sel[sel["domain"].astype(str) == str(dom_key)] if "domain" in sel.columns else pd.DataFrame()
                    row_pooled = sel[sel["domain"].isna()] if "domain" in sel.columns else sel
                    use_row = row_exact.iloc[0] if not row_exact.empty else (row_pooled.iloc[0] if not row_pooled.empty else None)

                    model_df = pd.DataFrame([])
                    if use_row is not None:
                        params = {k: float(use_row[k]) for k in ["b", "m1", "m2", "pC1", "pC2"] if k in use_row and pd.notna(use_row[k])}
                        preds = _predict_noisy_or_tasks(params, tasks_present)
                        mrows: List[Dict[str, Any]] = []
                        for t in tasks_present:
                            mrows.append({
                                "task": t,
                                "subject": f"{agent_name} ({dom_key})",
                                "domain": dom_key,
                                "prompt_category": prompt_cat,
                                "prediction_type": "CBN",
                                "likelihood-rating": preds[t],
                                "model": "noisy_or",
                            })
                        model_df = pd.DataFrame(mrows)

                    # Agent rows with SE
                    arows: List[Dict[str, Any]] = []
                    for _, r in agent_df.iterrows():
                        arows.append({
                            "task": str(r["task"]),
                            "subject": f"{agent_name} ({dom_key})",
                            "domain": dom_key,
                            "prompt_category": prompt_cat,
                            "prediction_type": "Agent",
                            "likelihood-rating": float(r["likelihood-rating"]),
                            "se": float(r.get("se", 0.0)) if pd.notna(r.get("se", 0.0)) else 0.0,
                            "n": int(r.get("n", 0)) if pd.notna(r.get("n", np.nan)) else 0,
                            "model": "Agent",
                        })
                    agent_rows_df = pd.DataFrame(arows)

                    combined = pd.concat([agent_rows_df, model_df], ignore_index=True)
                    # Add reasoning group and task labels for x-axis
                    group_map = {
                        "Predictive": ["I", "II", "III"],
                        "Independence": ["IV", "V"],
                        "Effect-Present": ["VI", "VII", "VIII"],
                        "Effect-Absent": ["IX", "X", "XI"],
                    }
                    combined["reasoning_group"] = None
                    for gname, gtasks in group_map.items():
                        combined.loc[combined["task"].isin(gtasks), "reasoning_group"] = gname
                    combined["task_label"] = combined["task"].map(_cond_prob_label)
                    combined_frames.append(combined)
                

        if not combined_frames:
            log.warning(f"No data to plot for prompt={prompt_cat}")
            continue

        plot_df = pd.concat(combined_frames, ignore_index=True)

        # Subject coloring
        unique_subjects = sorted(plot_df["subject"].astype(str).unique().tolist())
        subject_colors = _make_subject_colors(unique_subjects)

        # Title and filename
        domains_in_group = sorted({str(d) for d in plot_df.get("domain", pd.Series(dtype=str)).dropna().unique().tolist()})
        dom_label = "mixed" if len(domains_in_group) > 1 else (domains_in_group[0] if domains_in_group else "all")
        if pooled_mode:
            dom_label = "pooled"
        filename_base = f"pub_overlay_{args.experiment}_{tag}_{prompt_cat}_{dom_label}"

        # Configure Matplotlib LaTeX usage if requested
        if args.usetex:
            mpl.rcParams.update({
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath,bm}",
            })

        # Prepare plotting data ordered by Roman tasks
        tick_labels = [f"{t}\n{_cond_prob_label(t)}" for t in ROMAN_ORDER] if not args.no_probs else [f"{t}" for t in ROMAN_ORDER]
        plot_df = plot_df[plot_df["task"].isin(ROMAN_ORDER)].copy()

        # Build series per (subject, prediction_type)
        series: Dict[Tuple[str, str], List[Optional[float]]] = {}
        series_se: Dict[Tuple[str, str], List[Optional[float]]] = {}
        for (subject, ptype), subdf in plot_df.groupby(["subject", "prediction_type"], dropna=False):
            vals = [float(subdf[subdf["task"] == t]["likelihood-rating"].mean()) if (t in subdf["task"].values) else None for t in ROMAN_ORDER]
            series[(subject, ptype)] = vals
            if ptype == "Agent":
                se_vals = [
                    float(subdf[subdf["task"] == t]["se"].mean()) if ("se" in subdf.columns and t in subdf["task"].values) else None
                    for t in ROMAN_ORDER
                ]
                series_se[(subject, ptype)] = se_vals

        # Style and uncertainty helpers
        style_map = {
            "Agent": {
            "linestyle": "-", "marker": "o", "alpha": 0.9, "markersize": 6,
             "markeredgewidth": 1.2
            },
            "CBN": {
            "linestyle": ":", "marker": "^", "alpha": 0.7, "markersize": 5,
            "markeredgecolor": "black", "markeredgewidth": 1.2
            },
        }
        def _z_for_level(level: float) -> float:
            lvl = float(level)
            if abs(lvl - 90.0) < 1e-6:
                return 1.645
            if abs(lvl - 95.0) < 1e-6:
                return 1.96
            if abs(lvl - 99.0) < 1e-6:
                return 2.576
            return 1.96
        z = _z_for_level(args.uncertainty_level)

        # Per-reasoning-category figures
        group_defs = [
            ("Predictive Inference", "predictive", (0, 2)),
            ("Conditional Independence", "independence", (3, 4)),
            ("Effect-Present Diagnostic", "effect-present", (5, 7)),
            ("Effect-Absent Diagnostic", "effect-absent", (8, 10)),
        ]

        if args.by_reason_cat:
            for pretty, slug, (a, b) in group_defs:
                fig, ax = plt.subplots(figsize=(args.cat_fig_width, args.cat_fig_height))
                for (subject, ptype), vals in series.items():
                    color = subject_colors.get(subject, (0.2, 0.2, 0.2))
                    style = style_map.get(ptype, {"linestyle": "-", "marker": "o", "alpha": 0.8, "markersize": 5})
                    xs = list(range(0, (b - a + 1)))
                    ys = [vals[i] if vals[i] is not None else float('nan') for i in range(a, b + 1)]
                    ax.plot(xs, np.array(ys, dtype=float), label=f"{subject} ({ptype})", color=color, **style)
                    if args.uncertainty and ptype == "Agent" and (subject, ptype) in series_se:
                        se_vals = series_se.get((subject, ptype), [])
                        se_seg = [se_vals[i] if (i < len(se_vals) and se_vals[i] is not None) else float('nan') for i in range(a, b + 1)]
                        yarr = np.array(ys, dtype=float)
                        searr = np.array(se_seg, dtype=float)
                        lower = yarr - z * searr
                        upper = yarr + z * searr
                        ax.fill_between(xs, lower, upper, color=color, alpha=float(args.uncertainty_alpha), linewidth=0)

                # Axes formatting per category
                cat_tick_labels = (
                    [f"{ROMAN_ORDER[i]}\n{_cond_prob_label(ROMAN_ORDER[i])}" for i in range(a, b + 1)]
                    if not args.no_probs
                    else [f"{ROMAN_ORDER[i]}" for i in range(a, b + 1)]
                )
                ax.set_xlim(-0.5, (b - a + 1) - 0.5)
                ax.set_ylim(0, 100)
                ax.set_xticks(list(range(0, (b - a + 1))))
                if args.use_graph_cartoons:
                    # Keep Roman numerals when using cartoons; images serve as conditional context
                    ax.set_xticklabels([ROMAN_ORDER[i] for i in range(a, b + 1)])
                else:
                    ax.set_xticklabels(cat_tick_labels, rotation=45, ha="right")
                # Draw optional graph cartoons beneath x-ticks for this category
                _apply_graph_cartoons(ax, fig, paths, args.use_graph_cartoons, ROMAN_ORDER[a:b+1])
                ax.set_ylabel(args.ylabel)
                # ax.set_xlabel("Task")
                # Move x-label up/down examples:
                # ax.xaxis.labelpad = 8   # increase space between axis and label (down if positive)
                # ax.xaxis.set_label_coords(0.5, -0.12)  # (x,y) in axes fraction; smaller y pushes label down
                # Rename Conditional Independence title; show legend only on Predictive plot
                if slug == "independence":
                    ax.set_title("Independence of $C_1,C_2$")
                else:
                    ax.set_title(pretty)
                ax.grid(True, alpha=0.3)
                # Legend placement per user request
                if args.legend_loc_figure_xs:
                    # Normalize provided figure keys to lowercase
                    wanted = {str(x).strip().lower() for x in args.legend_loc_figure_xs}
                    if ("all" in wanted) or (slug in wanted):
                        ax.legend(loc=args.legend_loc, ncol=1, frameon=False)
                else:
                    # Default behavior: show legend only on Independence plot, slightly outside to the right
                    if slug == "independence":
                        ax.legend(
                            loc='lower right',
                            bbox_to_anchor=(1.02, 0),  # Increase x to move legend further right
                            borderaxespad=0.,
                            ncol=1,
                            frameon=False
                        )
                fig.tight_layout()
                out_pdf = out_dir / f"{filename_base}_{slug}.pdf"
                out_png = out_dir / f"{filename_base}_{slug}.png"
                fig.savefig(out_pdf, bbox_inches="tight")
                fig.savefig(out_png, dpi=300, bbox_inches="tight")
                if args.show and not args.no_show:
                    plt.show()
                plt.close(fig)
        # Always generate the unified big plot
        # Single figTure with background shading and segmented lines
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        # Background shading for groups
        for _, _, (a, b) in group_defs:
            ax.axvspan(a - 0.4, b + 0.4, color="#F7F7F7", zorder=0)
        for (subject, ptype), vals in series.items():
            color = subject_colors.get(subject, (0.2, 0.2, 0.2))
            style = style_map.get(ptype, {"linestyle": "-", "marker": "o", "alpha": 0.8, "markersize": 5})
            segments = [(0, 2), (3, 4), (5, 7), (8, 10)]
            for seg_idx, (a, b) in enumerate(segments):
                xs = list(range(a, b + 1))
                ys = [vals[i] if vals[i] is not None else float('nan') for i in xs]
                ax.plot(xs, np.array(ys, dtype=float), label=(f"{subject} ({ptype})" if seg_idx == 0 else "_nolegend_"), color=color, **style)
                if args.uncertainty and ptype == "Agent" and (subject, ptype) in series_se:
                    se_vals = series_se.get((subject, ptype), [])
                    se_seg = [se_vals[i] if (i < len(se_vals) and se_vals[i] is not None) else float('nan') for i in xs]
                    yarr = np.array(ys, dtype=float)
                    searr = np.array(se_seg, dtype=float)
                    lower = yarr - z * searr
                    upper = yarr + z * searr
                    ax.fill_between(xs, lower, upper, color=color, alpha=float(args.uncertainty_alpha), linewidth=0)

        ax.set_xlim(-0.5, len(ROMAN_ORDER)-0.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(list(range(len(ROMAN_ORDER))))
        if args.use_graph_cartoons:
            # Keep Roman numerals when using cartoons
            ax.set_xticklabels(ROMAN_ORDER)
        else:
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        # Draw optional graph cartoons beneath x-ticks for unified plot
        # Example to change cartoon size: pass zoom, e.g., zoom=0.22 for larger icons
        _apply_graph_cartoons(ax, fig, paths, args.use_graph_cartoons, ROMAN_ORDER)

        ax.set_ylabel(args.ylabel)
        # ax.set_xlabel("Task")
        # Move x-label up/down examples:
        # ax.xaxis.labelpad = 8   # increase space between axis and label (down if positive)
        # ax.xaxis.set_label_coords(0.5, -0.10)  # (x,y) in axes fraction; adjust y to move label
        ax.set_title("Agent vs its CBN predictions")
        ax.grid(True, alpha=0.3)
        # To force legend further right, you can use bbox_to_anchor, e.g.:
        # ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=args.legend_ncols)
        ax.legend(loc=args.legend_loc, ncol=args.legend_ncols, frameon=False)
        fig.tight_layout()
        out_pdf = out_dir / f"{filename_base}.pdf"
        out_png = out_dir / f"{filename_base}.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        if args.show and not args.no_show:
            plt.show()
        plt.close(fig)

    log.info(f"Completed publication plots in: {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
