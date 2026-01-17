#!/usr/bin/env python3
"""
Box and violin plots for EA, MV, and LAD per experiment × prompt-category.

Conventions mirrored from scripts/05_downstream_and_viz/plot_fit_metric_distributions.py:
- Prompt category colors identical to that script
- Mean (yellow triangle) and median (black line) indicated in legend
- Human baseline always drawn (RW17 Direct) as a horizontal pink dashed line when available
- Filenames include the metric and selected experiments/prompt categories

Input
-----
results/cross_cogn_strategies/masters_classified_strategy_metrics.csv

Output
------
results/plots/normative_bars/box_<metric>[_exp-...__pc-...].pdf/.png
results/plots/normative_bars/violin_<metric>[_exp-...__pc-...].pdf/.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib as mpl
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Experiment pretty names
exp_name_map = {
    "random_abstract": "Abstract",
    "rw17_indep_causes": "RW17",
    "abstract_overloaded_lorem_de": "Abstract-Overloaded",
    "rw17_overloaded_de": "RW17-Overloaded-DE",
    "rw17_overloaded_d": "RW17-Overloaded-D",
    "rw17_overloaded_e": "RW17-Overloaded",
}

# Use the exact same palette/canon behavior as the error metrics plotting script
try:
    from causalign.plotting.palette import (
        COT_LABEL,
        NUMERIC_LABEL,
        PROMPT_CATEGORY_COLORS,
        canon_prompt_category,
    )
except Exception:
    # Use CAblue/CAlightblue to match central palette
    CAblue = (10/255, 80/255, 110/255)
    CAlightblue = (58/255, 160/255, 171/255)
    PROMPT_CATEGORY_COLORS = {
        "Direct": CAlightblue,
        "CoT": CAblue,
    }
    def canon_prompt_category(label: str) -> str:  # type: ignore
        t = str(label).strip().lower()
        if t in {"numeric", "pcnum", "num", "single_numeric", "single_numeric_response"}:
            return "Direct"
        if t in {"cot", "pccot", "chain_of_thought", "chain-of-thought", "cot_stepwise", "cot"}:
            return "CoT"
        return str(label)
    NUMERIC_LABEL = "Direct"
    COT_LABEL = "CoT"


def _ensure_tueplots(usetex: bool = False) -> None:
    try:
        from tueplots import bundles  # type: ignore
        from tueplots import fonts as _fonts  # type: ignore
    except Exception:
        mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 12})
        return
    cfg = bundles.neurips2023(nrows=1, ncols=1, rel_width=0.9, usetex=usetex, family="serif")
    cfg["legend.title_fontsize"] = 11
    cfg["font.size"] = 12
    cfg["axes.labelsize"] = 12
    cfg["axes.titlesize"] = 13
    cfg["xtick.labelsize"] = 10
    cfg["ytick.labelsize"] = 10
    cfg["legend.fontsize"] = 10
    if usetex:
        cfg["text.latex.preamble"] = r"\usepackage{amsmath,bm}"
    fnt = _fonts.neurips2022_tex(family="serif")
    mpl.rcParams.update({**cfg, **fnt})


def _metric_info(m: str) -> Tuple[str, str]:
    m = m.lower()
    if m == "ea":
        return ("EA_raw", r"Explaining-away (EA)-level")
    if m == "mv":
        return ("MV_raw", r"Markov violation (MV)-level")
    if m == "lad":
        return ("normative_metric", r"Leak-Adjusted Determinacy (LAD)")
    raise ValueError("metric must be one of: ea, mv, lad")


# Fixed 4-slot order for experiments on x-axis
SLOT_ORDER: List[Tuple[str, str]] = [
    ("abstract_overloaded", "Abstract-Overloaded"),
    ("abstract", "Abstract"),
    ("rw17", "RW17"),
    ("rw17_overloaded", "RW17-Overloaded"),
]


def preferred_experiment_id_for_slot(slot: str) -> Optional[str]:
    """Return the concrete experiment id to use for a given canonical slot.

    - abstract_overloaded -> abstract_overloaded_lorem_de
    - abstract -> random_abstract
    - rw17 -> rw17_indep_causes
    - rw17_overloaded -> rw17_overloaded_e (as requested)
    Returns None if no preference.
    """
    slot = str(slot)
    if slot == "abstract_overloaded":
        return "abstract_overloaded_lorem_de"
    if slot == "abstract":
        return "random_abstract"
    if slot == "rw17":
        return "rw17_indep_causes"
    if slot == "rw17_overloaded":
        return "rw17_overloaded_e"
    return None


def _slot_experiment_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Pick one experiment id per slot based on availability in df.

    Uses preferred ids when present; otherwise attempts a fallback by pattern.
    For rw17_overloaded, strictly prefer rw17_overloaded_e.
    """
    available = set(df["experiment"].astype(str).unique())
    mapping: Dict[str, Optional[str]] = {}
    for slot, _ in SLOT_ORDER:
        pref = preferred_experiment_id_for_slot(slot)
        if pref in available:
            mapping[slot] = pref
            continue
        # Fallbacks by pattern (except rw17_overloaded, where we keep None if _e is missing)
        if slot == "abstract_overloaded":
            cand = [e for e in available if e.startswith("abstract_overloaded")] or []
            mapping[slot] = sorted(cand)[0] if cand else None
        elif slot == "abstract":
            mapping[slot] = "random_abstract" if "random_abstract" in available else None
        elif slot == "rw17":
            mapping[slot] = "rw17_indep_causes" if "rw17_indep_causes" in available else None
        elif slot == "rw17_overloaded":
            # Do not fall back to other variants when _e missing per request
            mapping[slot] = "rw17_overloaded_e" if "rw17_overloaded_e" in available else None
        else:
            mapping[slot] = None
    return mapping


def _preferred_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col == "EA_raw" and "EA_raw_mean" in df.columns:
        s = pd.to_numeric(df["EA_raw_mean"], errors="coerce")
        s = s.fillna(pd.to_numeric(df["EA_raw"], errors="coerce"))
        return s
    if col == "MV_raw" and "MV_raw_mean" in df.columns:
        s = pd.to_numeric(df["MV_raw_mean"], errors="coerce")
        s = s.fillna(pd.to_numeric(df["MV_raw"], errors="coerce"))
        return s
    return pd.to_numeric(df[col], errors="coerce")


def _rw17_human_baseline_from_masters(df: pd.DataFrame) -> Dict[str, float]:
    """Return human baselines from masters for RW17 Direct: keys ea, mv, lad."""
    d = df.copy()
    d["pc"] = d["prompt_category"].apply(canon_prompt_category)
    mask = (
        d["agent"].astype(str).str.lower().str.contains("human")
        & (d["experiment"].astype(str) == "rw17_indep_causes")
        & (d["pc"].astype(str) == NUMERIC_LABEL)
    )
    sub = d[mask].copy()
    out: Dict[str, float] = {}
    if sub.empty:
        return out
    # Prefer pooled domain
    if "domain" in sub.columns:
        pooled = sub[sub["domain"].astype(str) == "all"].copy()
        if not pooled.empty:
            sub = pooled
    # LAD
    lad = pd.to_numeric(sub["normative_metric"], errors="coerce").dropna()
    if not lad.empty:
        out["lad"] = float(lad.mean())
    # EA
    if "EA_raw_mean" in sub.columns:
        ea_s = pd.to_numeric(sub["EA_raw_mean"], errors="coerce").dropna()
    else:
        ea_s = pd.to_numeric(sub.get("EA_raw", pd.Series(dtype=float)), errors="coerce").dropna()
    if not ea_s.empty:
        out["ea"] = float(ea_s.mean())
    # MV
    if "MV_raw_mean" in sub.columns:
        mv_s = pd.to_numeric(sub["MV_raw_mean"], errors="coerce").dropna()
    else:
        mv_s = pd.to_numeric(sub.get("MV_raw", pd.Series(dtype=float)), errors="coerce").dropna()
    if not mv_s.empty:
        out["mv"] = float(mv_s.mean())
    return out


def _collect_groups_fixed_slots(
    df: pd.DataFrame,
    metric: str,
    prompt_order: List[str],
    slot_to_experiment: Dict[str, Optional[str]],
) -> Tuple[List[str], Dict[Tuple[str, str], np.ndarray]]:
    """Collect data per canonical slot and prompt.

    Returns:
      slots_in_order: fixed slot ids in SLOT_ORDER
      data: map (slot_id, pc) -> np.ndarray values
    """
    col, _ = _metric_info(metric)
    d = df.copy()
    d["value"] = _preferred_series(d, col)
    d = d[pd.to_numeric(d["value"], errors="coerce").notna()].copy()
    slots_in_order = [slot for slot, _ in SLOT_ORDER]
    data: Dict[Tuple[str, str], np.ndarray] = {}
    for slot in slots_in_order:
        exp_id = slot_to_experiment.get(slot)
        if not exp_id:
            for pc in prompt_order:
                data[(slot, pc)] = np.array([])
            continue
        gexp = d[d["experiment"].astype(str) == str(exp_id)]
        for pc in prompt_order:
            vals = pd.to_numeric(gexp[gexp["pc"].astype(str) == pc]["value"], errors="coerce").dropna().values
            data[(slot, pc)] = np.asarray(vals, dtype=float)
    return slots_in_order, data


def _legend_handles(prompt_order: List[str]) -> List[Line2D]:
    handles: List[Line2D] = []
    for pc in prompt_order:
        color = PROMPT_CATEGORY_COLORS.get(pc, PROMPT_CATEGORY_COLORS.get(pc.title(), (0.3, 0.3, 0.3)))
        handles.append(Line2D([0], [0], color=color, lw=6, label=pc))
    return handles


def plot_box(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    prompt_order: List[str],
    *,
    filename_suffix: str = "",
    include_slots: Optional[Iterable[str]] = None,
) -> Path:
    slot_map = _slot_experiment_mapping(df)
    slots, data = _collect_groups_fixed_slots(df, metric, prompt_order, slot_map)
    # Fixed 4-position layout regardless of which slots are included
    group_gap = 1.6
    intra = 0.18
    layout_n_exp = 4
    base_pos = np.arange(layout_n_exp) * group_gap
    fig_w = max(6.0, 0.8 + 0.9 * layout_n_exp * group_gap)
    fig_h = 3.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = base_pos
    n_pc = len(prompt_order)
    width = 0.32 if n_pc == 2 else 0.22
    offsets = np.linspace(-intra, intra, n_pc)

    include_set = set(include_slots) if include_slots is not None else {s for s, _ in SLOT_ORDER}
    for j, pc in enumerate(prompt_order):
        color = PROMPT_CATEGORY_COLORS.get(pc, PROMPT_CATEGORY_COLORS.get(pc.title(), (0.3, 0.3, 0.3)))
        boxes = []
        positions = []
        for i, slot in enumerate([s for s, _ in SLOT_ORDER]):
            if slot not in include_set:
                continue
            vals = data.get((slot, pc), np.array([]))
            if vals.size == 0:
                continue
            boxes.append(vals)
            positions.append(x[i] + offsets[j])
        if not boxes:
            continue
        bp = ax.boxplot(
            boxes,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_alpha(0.8)
        for element in ['whiskers', 'caps', 'medians']:
            for artist in bp[element]:
                artist.set_color('black')
        # Overlay mean as yellow triangle
        means = [float(np.nanmean(v)) for v in boxes]
        ax.plot(positions, means, linestyle='None', marker='^', markersize=6,
                markerfacecolor='yellow', markeredgecolor='black', zorder=5)

    # Axes formatting
    ax.set_xticks(x)
    # Show labels only for included slots; keep others blank but preserve positions
    tick_labels = [label if slot in include_set else "" for slot, label in SLOT_ORDER]
    ax.set_xticklabels(tick_labels, rotation=25, ha='right')
    _, y_label = _metric_info(metric)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Experiment")
    ax.grid(axis='y', linestyle=':', alpha=0.3)

    # Emphasize zero line for EA/MV metrics
    if str(metric).lower() in {"mv", "ea"}:
        ymin, ymax = ax.get_ylim()
        new_ymin, new_ymax = ymin, ymax
        if ymin > 0:
            new_ymin = 0
        if ymax < 0:
            new_ymax = 0
        if (new_ymin, new_ymax) != (ymin, ymax):
            ax.set_ylim(new_ymin, new_ymax)
        ax.axhline(y=0, color='gray', linewidth=1.2, alpha=0.9,  linestyle='--', zorder=2)

    # Human baseline (RW17 Direct) always when available
    hb_all = _rw17_human_baseline_from_masters(df)
    hb_val = hb_all.get(metric.lower())
    hb_line = None
    if hb_val is not None and np.isfinite(hb_val):
        hb_line = ax.axhline(y=hb_val, color=(1.0, 0.4, 0.7), linestyle='--', linewidth=1.5)

    # Legend (prompt colors + median + mean + human baseline)
    prompt_handles = _legend_handles(prompt_order)
    median_handle = Line2D([0, 1], [0, 0], color='black', lw=1.2, label='Median')
    mean_handle = Line2D([0], [0], marker='^', color='black', markerfacecolor='yellow', markeredgecolor='black', linestyle='None', markersize=6, label='Mean')
    handles = [*prompt_handles, median_handle, mean_handle]
    if hb_line is not None:
        handles.append(Line2D([0, 1], [0, 0], color=(1.0, 0.4, 0.7), linestyle='--', lw=1.5, label=f"Humans (RW17 {NUMERIC_LABEL})"))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    ax.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(handles))

    # Output
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{filename_suffix}" if filename_suffix else ""
    out_pdf = out_dir / f"box_{metric}{suffix}.pdf"
    out_png = out_dir / f"box_{metric}{suffix}.png"
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_pdf


def plot_violin(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    prompt_order: List[str],
    *,
    filename_suffix: str = "",
    include_slots: Optional[Iterable[str]] = None,
) -> Path:
    slot_map = _slot_experiment_mapping(df)
    slots, data = _collect_groups_fixed_slots(df, metric, prompt_order, slot_map)
    group_gap = 1.6
    intra = 0.18
    layout_n_exp = 4
    base_pos = np.arange(layout_n_exp) * group_gap
    fig_w = max(6.0, 0.8 + 0.9 * layout_n_exp * group_gap)
    fig_h = 3.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = base_pos
    n_pc = len(prompt_order)
    width = 0.32 if n_pc == 2 else 0.22
    offsets = np.linspace(-intra, intra, n_pc)

    include_set = set(include_slots) if include_slots is not None else {s for s, _ in SLOT_ORDER}
    for j, pc in enumerate(prompt_order):
        color = PROMPT_CATEGORY_COLORS.get(pc, PROMPT_CATEGORY_COLORS.get(pc.title(), (0.3, 0.3, 0.3)))
        boxes = []
        positions = []
        for i, slot in enumerate([s for s, _ in SLOT_ORDER]):
            if slot not in include_set:
                continue
            vals = data.get((slot, pc), np.array([]))
            if vals.size == 0:
                continue
            boxes.append(vals)
            positions.append(x[i] + offsets[j])
        if not boxes:
            continue
        vp = ax.violinplot(boxes, positions=positions, widths=width * 0.9, showmeans=False, showmedians=True, showextrema=False)
        bodies = vp.get('bodies', []) if isinstance(vp, dict) else getattr(vp, 'bodies', [])
        if not isinstance(bodies, (list, tuple)):
            bodies = [bodies] if bodies is not None else []
        for body in bodies:
            body.set_facecolor(color)
            body.set_edgecolor('black')
            body.set_alpha(0.7)
        cm = getattr(vp, 'cmedians', None)
        if cm is not None:
            cm.set_color('black')
        means = [float(np.nanmean(v)) for v in boxes]
        ax.plot(positions, means, linestyle='None', marker='^', markersize=6,
                markerfacecolor='yellow', markeredgecolor='black', zorder=5)

    ax.set_xticks(x)
    tick_labels = [label if slot in include_set else "" for slot, label in SLOT_ORDER]
    ax.set_xticklabels(tick_labels, rotation=25, ha='right')
    _, y_label = _metric_info(metric)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Experiment")
    ax.grid(axis='y', linestyle=':', alpha=0.3)

    # Emphasize zero line for EA/MV metrics
    if str(metric).lower() in {"mv", "ea"}:
        ymin, ymax = ax.get_ylim()
        new_ymin, new_ymax = ymin, ymax
        if ymin > 0:
            new_ymin = 0
        if ymax < 0:
            new_ymax = 0
        if (new_ymin, new_ymax) != (ymin, ymax):
            ax.set_ylim(new_ymin, new_ymax)
        ax.axhline(y=0, color='gray', linewidth=1.2, alpha=0.9,  linestyle='--', zorder=2)
    hb_all = _rw17_human_baseline_from_masters(df)
    hb_val = hb_all.get(metric.lower())
    hb_line = None
    if hb_val is not None and np.isfinite(hb_val):
        hb_line = ax.axhline(y=hb_val, color=(1.0, 0.4, 0.7), linestyle='--', linewidth=1.5)

    prompt_handles = _legend_handles(prompt_order)
    median_handle = Line2D([0, 1], [0, 0], color='black', lw=1.2, label='Median')
    mean_handle = Line2D([0], [0], marker='^', color='black', markerfacecolor='yellow', markeredgecolor='black', linestyle='None', markersize=6, label='Mean')
    handles = [*prompt_handles, median_handle, mean_handle]
    if hb_line is not None:
        handles.append(Line2D([0, 1], [0, 0], color=(1.0, 0.4, 0.7), linestyle='--', lw=1.5, label=f"Humans (RW17 {NUMERIC_LABEL})"))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    ax.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(handles))

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{filename_suffix}" if filename_suffix else ""
    out_pdf = out_dir / f"violin_{metric}{suffix}.pdf"
    out_png = out_dir / f"violin_{metric}{suffix}.png"
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return out_pdf

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Box and violin plots for EA, MV, and LAD per experiment × prompt-category")
    ap.add_argument("--input", default="results/cross_cogn_strategies/masters_classified_strategy_metrics.csv", help="Path to masters CSV")
    ap.add_argument("--output-dir", default="results/plots/normative_bars", help="Directory to write plots")
    ap.add_argument("--metrics", nargs="*", default=["ea", "mv", "lad"], help="Metrics to plot: ea mv lad")
    ap.add_argument("--prompts", nargs="*", default=["numeric", "cot"], help="Prompt categories to include and order")
    ap.add_argument("--prompt-categories", dest="prompts", nargs="*", help="Alias for --prompts")
    ap.add_argument("--experiments", nargs="*", help="Experiments to include (ids)")
    ap.add_argument("--usetex", action="store_true", help="Use LaTeX rendering if available")
    ap.add_argument("--auto-panels", action="store_true", help="Generate the 4 requested figure combinations per metric with fixed slot layout")
    args = ap.parse_args(argv)

    _ensure_tueplots(args.usetex)
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    df = pd.read_csv(in_path)
    # Canon prompt categories and filter
    df["pc"] = df["prompt_category"].apply(canon_prompt_category)
    prompt_order = [canon_prompt_category(p) for p in args.prompts] if args.prompts else [NUMERIC_LABEL, COT_LABEL]
    df = df[df["pc"].isin(set(prompt_order))].copy()
    # Filter by experiments if provided
    if args.experiments:
        allowed = set(str(e) for e in args.experiments)
        df = df[df["experiment"].astype(str).isin(allowed)].copy()
    # Prefer pooled domain if available
    if "domain" in df.columns:
        pooled = df[df["domain"].astype(str) == "all"].copy()
        if not pooled.empty:
            df = pooled
    if df.empty:
        print("[warn] No data after filtering by prompt categories/experiments.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.auto_panels:
        # Build the four panels per metric
        slot_sets = [
            ("rw17", ["rw17"]),
            ("abs+rw17", ["abstract", "rw17"]),
            ("all4-direct", ["abstract_overloaded", "abstract", "rw17", "rw17_overloaded"]),
            ("all4-allprompts", ["abstract_overloaded", "abstract", "rw17", "rw17_overloaded"]),
        ]
        for m in args.metrics:
            for key, slots in slot_sets:
                # Panel 3: Direct only; Panel 4: Both prompts
                if key.endswith("direct"):
                    porder = [NUMERIC_LABEL]
                elif key.endswith("allprompts"):
                    porder = [NUMERIC_LABEL, COT_LABEL]
                else:
                    # For rw17-only and abs+rw17, Direct-only as requested
                    porder = [NUMERIC_LABEL]
                try:
                    pc_join = "+".join(porder)
                    suffix = f"__slots-{key}__pc-{pc_join}"
                    p1 = plot_box(df, m, out_dir, porder, filename_suffix=suffix, include_slots=slots)
                    p2 = plot_violin(df, m, out_dir, porder, filename_suffix=suffix, include_slots=slots)
                    print(f"[ok] Wrote {p1}")
                    print(f"[ok] Wrote {p2}")
                except Exception as e:
                    print(f"[warn] Failed to plot metric '{m}' (panel {key}): {e}")
        return 0
    else:
        # Original behavior
        exp_suffix = None
        if args.experiments:
            exp_suffix = "exp-" + "+".join([str(e) for e in args.experiments])
        pc_suffix = "pc-" + "+".join([str(canon_prompt_category(p)) for p in (args.prompts or [])])
        suffix_parts = [s for s in [exp_suffix, pc_suffix] if s]
        filename_suffix = "_" + "__".join(suffix_parts) if suffix_parts else ""

        for m in args.metrics:
            try:
                p1 = plot_box(df, m, out_dir, prompt_order, filename_suffix=filename_suffix)
                p2 = plot_violin(df, m, out_dir, prompt_order, filename_suffix=filename_suffix)
                print(f"[ok] Wrote {p1}")
                print(f"[ok] Wrote {p2}")
            except Exception as e:
                print(f"[warn] Failed to plot metric '{m}': {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
