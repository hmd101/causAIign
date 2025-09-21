#!/usr/bin/env python3
"""
Human–Agent Alignment per Domain, with GPT-5 variant handling (canonical).

- Treat each distinct (subject, verbosity, reasoning_effort) for GPT-5 as a separate agent label: gpt-5-v_<v>-r_<r>.
- Compute Spearman correlations vs aggregated humans per domain and prompt_category.
- Bootstrap 95% CIs. Save CSV and plots. Column visibility preserved for verbosity/reasoning_effort.
"""

import argparse
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

try:
    # Shared palette for prompt categories
    from causalign.plotting.palette import (
        canon_prompt_category as _canon_pcat,  # type: ignore[assignment]
    )
    from causalign.plotting.palette import PROMPT_CATEGORY_COLORS as _PCOL
except Exception:
    _PCOL = {"numeric": (0.85, 0.60, 0.55), "CoT": (0.00, 0.20, 0.55)}
    def _canon_pcat(s: str) -> str:
        t = str(s).strip().lower()
        if t in {"numeric", "pcnum", "num", "single_numeric", "single_numeric_response"}:
            return "numeric"
        if t in {"cot", "pccot", "chain_of_thought", "chain-of-thought", "cot_stepwise", "CoT"}:
            return "CoT"
        return str(s)
from tueplots import bundles, fonts

# NeurIPS-like, LaTeX, serif
config = bundles.neurips2023(
    nrows=2, ncols=1,
    rel_width=0.8,
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


def project_root() -> Path:
    # scripts/03_analysis_raw/<file>.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def load_processed(args):
    """Load processed data, trying several known naming variants.

    Tries, in order:
      - reasoning_types/{version}_v_{graph}_cleaned_data_roman.csv (aggregated, Roman numerals)
      - {version}_v_{graph}_cleaned_data.csv (aggregated, non-Roman)
      - {version}_v_humans_avg_equal_sample_size_cogsci.csv (pooled humans variant)
      - {version}_v_{graph}_cleaned_data_indiv_humans.csv (individual humans)
    """
    if args.input_file:
        print(f"[INFO] Loading input data from: {args.input_file}")
        df = pd.read_csv(args.input_file)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    base = project_root() / "data" / "processed" / "llm_with_humans" / "rw17" / args.experiment
    candidates = [
        base / "reasoning_types" / f"{args.version}_v_{args.graph}_cleaned_data_roman.csv",
        base / f"{args.version}_v_{args.graph}_cleaned_data.csv",
        base / f"{args.version}_v_humans_avg_equal_sample_size_cogsci.csv",
        base / f"{args.version}_v_{args.graph}_cleaned_data_indiv_humans.csv",
    ]
    for p in candidates:
        if p.exists():
            print(f"[INFO] Loading input data from: {p}")
            df = pd.read_csv(p)
            df.columns = [str(c).strip() for c in df.columns]
            return df
    # Helpful error: show available CSVs in base and reasoning_types
    avail = []
    for folder in [base, base / "reasoning_types"]:
        if folder.exists():
            for f in sorted(folder.glob("*.csv")):
                try:
                    rel = f.relative_to(project_root())
                except Exception:
                    rel = f
                avail.append(str(rel))
    raise FileNotFoundError(
        "Could not find a processed CSV. Tried these names under 'data/processed/llm_with_humans/rw17/" +
        f"{args.experiment}':\n - " + "\n - ".join(str(c.relative_to(project_root())) for c in candidates) +
        ("\nAvailable CSVs:\n - " + "\n - ".join(avail) if avail else "\nNo CSV files found in expected folders.")
    )


def ensure_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "verbosity" not in df.columns:
        df["verbosity"] = "n/a"
    if "reasoning_effort" not in df.columns:
        df["reasoning_effort"] = "n/a"
    for c in ["verbosity", "reasoning_effort"]:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"": "n/a", "nan": "n/a"})
    # Build variant label
    subj_str = df["subject"].astype(str)
    is_gpt5 = subj_str.str.startswith("gpt-5")
    already_variant = subj_str.str.contains(r"-v_.*-r_.*", regex=True)
    has_meta = ~(
        df["verbosity"].isin(["n/a", "unspecified"]) & df["reasoning_effort"].isin(["n/a", "unspecified"])  # noqa: E501
    )
    df["agent_variant"] = df["subject"].astype(str)
    df.loc[is_gpt5 & has_meta & ~already_variant, "agent_variant"] = (
        df.loc[is_gpt5 & has_meta, "subject"].astype(str)
        + "-v_" + df.loc[is_gpt5 & has_meta, "verbosity"].astype(str)
        + "-r_" + df.loc[is_gpt5 & has_meta, "reasoning_effort"].astype(str)
    )
    return df


def _extract_corr(res):
    # SciPy versions vary: may return SpearmanrResult or (rho, p)
    if hasattr(res, "correlation"):
        return float(res.correlation)
    if hasattr(res, "statistic"):
        return float(res.statistic)
    if isinstance(res, (list, tuple)):
        return float(res[0])
    return float(res)


def bootstrap_spearman(x, y, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    m = ~(np.isnan(x) | np.isnan(y))
    x = x[m]
    y = y[m]
    if len(x) < 3:
        return np.nan, (np.nan, np.nan)
    rho = _extract_corr(spearmanr(x, y))
    if n <= 0:
        return float(rho), (np.nan, np.nan)
    idx = np.arange(len(x))
    boots = np.empty(n)
    for i in range(n):
        samp = rng.choice(idx, size=len(idx), replace=True)
        r = _extract_corr(spearmanr(x[samp], y[samp]))
        boots[i] = float(r)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(rho), (float(lo), float(hi))


def compute_alignment(df: pd.DataFrame, bootstrap=2000, include_pooled: bool = True) -> pd.DataFrame:
    # Aggregate human baseline per (domain, task, cntbl_cond, graph) to avoid ID mismatch issues
    key_cols = ["domain", "task", "cntbl_cond", "graph"]
    humans = (
        df[df["subject"] == "humans"][key_cols + ["likelihood"]]
        .groupby(key_cols, as_index=False)
        .agg(human_likelihood=("likelihood", "mean"))
    )

    agents = df[df["subject"] != "humans"].copy()
    out = []
    for (agent, domain, pcat), g in agents.groupby(["agent_variant", "domain", "prompt_category"]):
        # Merge by semantic keys shared between humans and LLMs
        merged = g.merge(humans, on=key_cols, how="inner")
        if merged.empty:
            continue
        rho, (lo, hi) = bootstrap_spearman(
            merged["likelihood"].to_numpy(dtype=float), merged["human_likelihood"].to_numpy(dtype=float), n=bootstrap
        )
        out.append({
            "agent": agent,
            "domain": domain,
            "prompt_category": pcat,
            "rho": rho,
            "ci_low": lo,
            "ci_high": hi,
            "n": int(len(merged)),
        })
    res = pd.DataFrame(out)

    if include_pooled:
        # Compute pooled across all domains per (agent, prompt_category)
        pooled_rows = []
        agents = df[df["subject"] != "humans"].copy()
        for (agent, pcat), g in agents.groupby(["agent_variant", "prompt_category"]):
            merged = g.merge(humans, on=key_cols, how="inner")
            if merged.empty:
                continue
            rho, (lo, hi) = bootstrap_spearman(
                merged["likelihood"].to_numpy(dtype=float), merged["human_likelihood"].to_numpy(dtype=float), n=bootstrap
            )
            # Use domain label 'all' to signal pooled
            pooled_rows.append({
                "agent": agent,
                "domain": "all",
                "prompt_category": pcat,
                "rho": rho,
                "ci_low": lo,
                "ci_high": hi,
                "n": int(len(merged)),
            })
        if pooled_rows:
            res = pd.concat([res, pd.DataFrame(pooled_rows)], ignore_index=True)

    return res


def plot_alignment(res: pd.DataFrame, out_dir: Path, domain_spec: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if res.empty:
        return

    for pcat, sub in res.groupby("prompt_category"):
        # Prefer ordering by pooled ('all')
        pooled_map = (
            sub[sub["domain"].astype(str) == "all"].set_index("agent")["rho"].to_dict()
        )
        if pooled_map:
            order = sorted(
                sub["agent"].unique().tolist(),
                key=lambda a: pooled_map.get(a, sub[sub["agent"] == a]["rho"].mean()),
                reverse=True,
            )
        else:
            order = (
                sub.groupby("agent")["rho"].mean()
                .sort_values(ascending=False)
                .index.tolist()
            )

        # Figure: width scale and height by number of agents
        n = len(order)
        height = max(2.5, 0.25 * n)
        base_w = mpl.rcParams["figure.figsize"][0]
        fig, ax = plt.subplots(figsize=(base_w * 1.6, height * 1.2))

        # Faint horizontal banding
        for i in range(1, n, 2):
            ax.axhline(i, color="0.85", lw=0.8, alpha=0.6, zorder=0)

        hue_levels = sub["domain"].nunique()
        dodge = hue_levels > 1

        # Palette: others in greys, pooled 'all' in prompt-category color
        domains = sorted(sub["domain"].astype(str).unique().tolist())
        if "all" in domains:
            domains = ["all"] + [d for d in domains if d != "all"]
        others = [d for d in domains if d != "all"]
        grey_palette = [(v, v, v) for v in np.linspace(0.3, 0.8, len(others))] if others else []
        palette_map = {d: c for d, c in zip(others, grey_palette)}
        _pc_key = _canon_pcat(str(pcat))
        palette_map["all"] = _PCOL.get(_pc_key, mcolors.to_rgb("#d62728"))

        sns.pointplot(
            data=sub,
            y="agent",
            x="rho",
            hue="domain",
            order=order,
            hue_order=domains,
            palette=palette_map,
            dodge=dodge,
            errorbar=None,
            ax=ax,
        )
        ax.legend(title="domain", loc="upper left", frameon=False)

        # CI whiskers with vertical offsets by hue
        if hue_levels > 1:
            y_jitter = 0.18
            y_offsets = {d: off for d, off in zip(domains, np.linspace(-y_jitter, y_jitter, len(domains)))}
        else:
            y_offsets = {domains[0]: 0.0}

        for _, row in sub.iterrows():
            base_y = order.index(row["agent"])
            d = str(row["domain"])
            y_off = y_offsets.get(d, 0.0)
            color = palette_map.get(d, "black")
            alpha = 0.95 if d == "all" else 0.7
            ax.plot([row["ci_low"], row["ci_high"]], [base_y + y_off, base_y + y_off], color=color, alpha=alpha, lw=2.0, zorder=1)

        pooled_vals = sub[sub["domain"].astype(str) == "all"]["rho"].dropna()
        if not pooled_vals.empty:
            xmin = float(pooled_vals.min())
            xmax = float(pooled_vals.max())
            ax.axvline(xmin, color="gray", lw=1, ls=":")
            if xmax != xmin:
                ax.axvline(xmax, color="gray", lw=1, ls=":")
        ax.set_yticks(range(n))
        ax.set_yticklabels(order)
        ax.set_xlabel(r"Human-LLM alignment (Spearman $\rho$)")
        ax.set_ylabel("LLM agent")
        ax.set_xlim(-0.13, 1.0)

        ax.legend(title="Domains:", loc="upper left", frameon=False, ncol=1, handlelength=1.5)
        plt.tight_layout()

        fn_suffix = (domain_spec if domain_spec != "all" else "all").replace(",", "+").replace(" ", "")
        fig.savefig(out_dir / f"alignment_spearman_{pcat}_dom_{fn_suffix}.pdf", bbox_inches="tight")
        if domain_spec == "all" and pcat == "numeric":
            fig.savefig(out_dir / "alignment_spearman_numeric.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_overlay_pooled_categories(res: pd.DataFrame, out_dir: Path, domain_spec: str = "all") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled = res[res["domain"].astype(str) == "all"].copy()
    pooled["prompt_category"] = pooled["prompt_category"].astype(str).map(_canon_pcat)
    pooled = pooled[pooled["prompt_category"].isin(["numeric", "CoT"])]
    if pooled.empty:
        return
    numeric = pooled[pooled["prompt_category"] == "numeric"].sort_values("rho", ascending=False)
    order = numeric["agent"].tolist()[::-1]
    numeric_color = _PCOL.get("numeric", (0.85, 0.60, 0.55))
    cot_color = _PCOL.get("CoT", (0.00, 0.20, 0.55))
    color_map = _PCOL
    n = len(order)
    base_w = mpl.rcParams["figure.figsize"][0]
    fig, ax = plt.subplots(figsize=(base_w * 1.6, max(2.5, 0.25 * n) * 1.2))
    ax.set_ylim(-0.5, n - 0.5)
    for i in range(1, n, 2):
        ax.axhline(i, color="0.88", lw=0.8, alpha=0.6, zorder=0)
    minmax_info: dict[str, tuple[float, float]] = {}
    y_offsets = {"numeric": -0.08, "CoT": +0.08}
    for pcat in ["numeric", "CoT"]:
        sub = pooled[pooled["prompt_category"] == pcat].set_index("agent").reindex(order).reset_index()
        color = color_map[pcat]
        y = np.arange(n) + y_offsets[pcat]
        for i, row in enumerate(sub.itertuples(index=False)):
            lo = float(getattr(row, "ci_low")) if pd.notna(getattr(row, "ci_low")) else np.nan
            hi = float(getattr(row, "ci_high")) if pd.notna(getattr(row, "ci_high")) else np.nan
            ax.plot([lo, hi], [float(y[i]), float(y[i])], color=color, lw=1.6, alpha=0.85, zorder=1)
        xvals = pd.to_numeric(sub["rho"], errors="coerce").to_numpy(dtype=float)
        ax.plot(xvals, y.astype(float), color=color, lw=1.8, alpha=0.85, zorder=2)
        ax.scatter(xvals, y.astype(float), s=28, color=color, zorder=3)
        vals = pd.to_numeric(sub["rho"], errors="coerce").dropna()
        if not vals.empty:
            minmax_info[pcat] = (float(vals.min()), float(vals.max()))
    def _draw_minmax_lines(xmin: float, xmax: float, color):
        ax.axvline(xmin, color=color, lw=1.0, ls=":", zorder=1)
        if xmax != xmin:
            ax.axvline(xmax, color=color, lw=1.0, ls=":", zorder=1)
    if "numeric" in minmax_info:
        _draw_minmax_lines(*minmax_info["numeric"], color=numeric_color)
    if "CoT" in minmax_info:
        _draw_minmax_lines(*minmax_info["CoT"], color=cot_color)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(order)
    ax.set_xlabel(r"Human-LLM alignment (Spearman $\rho$)")
    ax.set_ylabel("LLM agent")
    ax.set_xlim(-0.13, 1.00)
    ax.set_title("Human--LLM Alignment by Prompt-Category")
    handles = [
        mlines.Line2D([], [], color=numeric_color, lw=2, label="Numeric (all)"),
        mlines.Line2D([], [], color=cot_color, lw=2, label="CoT (all)"),
    ]
    if "numeric" in minmax_info:
        nmin, nmax = minmax_info["numeric"]
        handles.append(mlines.Line2D([], [], color=numeric_color, lw=1.5, ls=":", label=f"Numeric min/max: {nmin:.2f} / {nmax:.2f}"))
    if "CoT" in minmax_info:
        cmin, cmax = minmax_info["CoT"]
        handles.append(mlines.Line2D([], [], color=cot_color, lw=1.5, ls=":", label=f"CoT min/max: {cmin:.2f} / {cmax:.2f}"))
    leg = ax.legend(handles=handles, title="Prompt Category (Domains):", loc="upper left", frameon=False, ncol=1, handlelength=1.5)
    try:
        leg.get_title().set_fontweight("bold")
    except Exception:
        pass
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92)
    fig.savefig(out_dir / f"alignment_overlay_numeric_cot_pooled_{(domain_spec if domain_spec!='all' else 'all').replace(',', '+').replace(' ', '')}.pdf", bbox_inches="tight")
    plt.close(fig)


def export_latex_tables(res: pd.DataFrame, experiment: str, domain_spec: str) -> None:
    base = project_root() / "publication" / "thesis" / "tuebingen_thesis_msc" / "tables" / "human-llm-corr" / experiment
    base.mkdir(parents=True, exist_ok=True)
    def fmt(x) -> str:
        try:
            if pd.isna(x):
                return "--"
            return f"{float(x):.3f}"
        except Exception:
            return "--"
    def _esc(s: str) -> str:
        return (
            str(s)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )
    suffix = (domain_spec if domain_spec != "all" else "all").replace(",", "+").replace(" ", "")
    for pcat, sub in res.groupby("prompt_category"):
        sub = sub.copy()
        sub["domain"] = sub["domain"].replace({"all": "pooled"})
        pivot_rho = sub.pivot_table(index="agent", columns="domain", values="rho")
        pivot_lo = sub.pivot_table(index="agent", columns="domain", values="ci_low")
        pivot_hi = sub.pivot_table(index="agent", columns="domain", values="ci_high")
        all_cols = [str(c) for c in pivot_rho.columns]
        cols = ["pooled"] + [c for c in sorted(all_cols) if c != "pooled"] if "pooled" in all_cols else sorted(all_cols)
        if "pooled" in pivot_rho.columns:
            agent_order = pivot_rho["pooled"].sort_values(ascending=False).index.tolist()
        else:
            agent_order = pivot_rho.mean(axis=1).sort_values(ascending=False).index.tolist()
        col_max: dict[str, float] = {}
        for c in cols:
            if c in pivot_rho.columns:
                series = pd.to_numeric(pivot_rho[c], errors="coerce")
                col_max[c] = float(series.max(skipna=True)) if not series.dropna().empty else float("nan")
            else:
                col_max[c] = float("nan")
        lines: list[str] = []
        lines.append("% Auto-generated; do not edit by hand\n")
        lines.append("\\begin{table*}[t]\n\\centering\n")
        lines.append("\\scriptsize\n")
        colspec = "l" + ("r" * len(cols))
        lines.append(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        lines.append("Agent & \\multicolumn{" + str(len(cols)) + "}{c}{Domain} \\\\ \n")
        lines.append(" & " + " & ".join(cols) + " \\\\ \n\\midrule\n")
        for agent in agent_order:
            row_vals = []
            for c in cols:
                if c in pivot_rho.columns:
                    val = pivot_rho.loc[agent, c]
                    lo = pivot_lo.loc[agent, c] if c in pivot_lo.columns else np.nan
                    hi = pivot_hi.loc[agent, c] if c in pivot_hi.columns else np.nan
                    val_num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                    lo_num = pd.to_numeric(pd.Series([lo]), errors="coerce").iloc[0]
                    hi_num = pd.to_numeric(pd.Series([hi]), errors="coerce").iloc[0]
                    fval = fmt(val_num)
                    fci = f"[{fmt(lo_num)}, {fmt(hi_num)}]" if not (pd.isna(lo_num) or pd.isna(hi_num)) else ""
                    maxv = col_max.get(c, float("nan"))
                    is_max = (not pd.isna(val_num)) and (not pd.isna(maxv)) and (abs(float(val_num) - float(maxv)) < 1e-12)
                    cell = f"\\textbf{{{fval}}} {fci}" if is_max and fval != "--" else f"{fval} {fci}"
                else:
                    cell = "--"
                row_vals.append(cell)
            lines.append(f"{_esc(agent)} & " + " & ".join(row_vals) + " \\\\ \n")
        lines.append("\\bottomrule\n\\end{tabular}\n")
        dom_title = domain_spec if domain_spec != "all" else "all domains"
        lines.append(f"\\caption{{Human--LLM alignment (Spearman rho) across domains (prompt category: {pcat}; {dom_title}). Agents are ordered by pooled domain alignment. Each cell reports the bootstrapped Spearman $\\rho$ and 95\\% confidence interval [lower, upper]. Uncertainty reflects the range of $\\rho$ values obtained by nonparametric bootstrapping (2,000 resamples) for each agent--domain pair.}}\n")
        lines.append(f"\\label{{tab:alignment_{pcat}_dom_{suffix}}}\n")
        lines.append("\\end{table*}\n")
        tex_path = base / f"alignment_{pcat}_wide_dom_{suffix}.tex"
        tex_path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Human–Agent alignment with gpt-5 variant handling")
    ap.add_argument("--version", default="2")
    ap.add_argument("--experiment", default="rw17_indep_causes")
    ap.add_argument("--graph", default="collider")
    ap.add_argument("--input-file")
    ap.add_argument("--output-dir", default="results/human_llm_corr")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument(
        "--domains",
        default="all",
        help="Comma-separated list of domains to include (e.g., 'economy,weather') or 'all'",
    )
    args = ap.parse_args()

    df = load_processed(args)
    df = ensure_variant_columns(df)

    # Domain filtering
    domain_spec = str(args.domains).strip()
    if domain_spec and domain_spec.lower() != "all":
        selected = [d for d in re.split(r"[,\s]+", domain_spec) if d]
        df = df[df["domain"].astype(str).isin(selected)].copy()
    else:
        domain_spec = "all"

    res = compute_alignment(df, bootstrap=args.bootstrap)
    if res.empty or "prompt_category" not in res.columns:
        print("[WARNING] No alignment results were computed. This likely means there is no human data available for this experiment, so agent–human alignment cannot be calculated.")
        print("[INFO] Skipping further output and table generation.")
        return

    out_dir = project_root() / args.output_dir / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_suffix = (domain_spec if domain_spec != "all" else "all").replace(",", "+").replace(" ", "")
    combined_csv_path = out_dir / f"alignment_domain_agent_variants_v{args.version}_dom_{csv_suffix}.csv"
    res.to_csv(combined_csv_path, index=False)

    if domain_spec == "all" and not res.empty:
        for dom in sorted(res["domain"].astype(str).unique()):
            sub_dom = res[res["domain"].astype(str) == dom]
            sub_dom.to_csv(out_dir / f"alignment_domain_agent_variants_v{args.version}_dom_{dom}.csv", index=False)
    plot_alignment(res, out_dir, domain_spec)
    plot_overlay_pooled_categories(res, out_dir, domain_spec)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for pcat, sub in res.groupby("prompt_category"):
        sub = sub.copy()
        sub["domain"] = sub["domain"].replace({"all": "pooled"})
        pivot_rho = sub.pivot_table(index="agent", columns="domain", values="rho")
        pivot_lo = sub.pivot_table(index="agent", columns="domain", values="ci_low")
        pivot_hi = sub.pivot_table(index="agent", columns="domain", values="ci_high")
        all_cols = [str(c) for c in pivot_rho.columns]
        cols = ["pooled"] + [c for c in sorted(all_cols) if c != "pooled"] if "pooled" in all_cols else sorted(all_cols)
        if "pooled" in pivot_rho.columns:
            agent_order = pivot_rho["pooled"].sort_values(ascending=False).index.tolist()
        else:
            agent_order = pivot_rho.mean(axis=1).sort_values(ascending=False).index.tolist()
        out_rows = []
        header = ["Agent"] + cols
        out_rows.append(["Agent"] + ["Domain"] * len(cols))
        out_rows.append(header)
        for agent in agent_order:
            row = [agent]
            for c in cols:
                if c in pivot_rho.columns:
                    val = pivot_rho.loc[agent, c]
                    lo = pivot_lo.loc[agent, c] if c in pivot_lo.columns else np.nan
                    hi = pivot_hi.loc[agent, c] if c in pivot_hi.columns else np.nan
                    fval = f"{val:.3f}" if pd.notna(val) else "--"
                    fci = f"[{lo:.3f}, {hi:.3f}]" if pd.notna(lo) and pd.notna(hi) else ""
                    cell = f"{fval} {fci}".strip()
                else:
                    cell = "--"
                row.append(cell)
            out_rows.append(row)
        out_path = tables_dir / f"alignment_{pcat}_wide_dom_{csv_suffix}.csv"
        pd.DataFrame(out_rows).to_csv(out_path, index=False, header=False)

    export_latex_tables(res, args.experiment, domain_spec)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
