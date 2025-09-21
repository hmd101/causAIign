#!/usr/bin/env python3
"""
Export citation-ready LaTeX tables from Milestone B outputs.

Generates:
 - kw_across_agents_per_domain_v{ver}.tex (omnibus across agents per domain)
 - kw_across_domains_per_agent_v{ver}.tex (omnibus across domains per agent)
 - pairwise_top_by_wasserstein_norm_{domain}_v{ver}.tex (top pairs per domain)

Usage example:
  python scripts/export_domain_differences_tables.py --experiment rw17_indep_causes --version 2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "--"
    if p == 0:
        return "$<10^{-300}$"
    if p < 1e-4:
        return f"$\\num{{{p:.1e}}}$"
    if p < 0.01:
        return f"$\\num{{{p:.3f}}}$"
    return f"$\\num{{{p:.3f}}}$"


def latex_escape(val) -> str:
    """Escape LaTeX special characters for safe use in tabular text cells.
    Keep this minimal to avoid interfering with table syntax; we escape the most
    common offenders found in agent/domain names.
    """
    if pd.isna(val):
        return "--"
    s = str(val)
    # Order matters: backslash first to avoid double-escaping introduced braces
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("_", r"\_")
    s = s.replace("#", r"\#")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def save_kw_by_domain(df: pd.DataFrame, out_path: Path):
    cols = ["domain", "k", "H", "df", "p_value", "p_fdr_bh_across_domains", "n_total"]
    df = df[cols].copy()
    df["p_fmt"] = df["p_value"].map(fmt_p)
    df["p_fdr_fmt"] = df["p_fdr_bh_across_domains"].map(fmt_p)
    lines = []
    lines.append("% Auto-generated; do not edit by hand\n")
    lines.append("\\begin{table*}[t]\n\\centering\n")
    lines.append("\\footnotesize\n")
    lines.append("\\begin{tabular}{lrrrrr}\n\\toprule\n")
    lines.append("Domain & $k$ & $H$ & df & $p$ (raw) & $p_\\text{FDR}$ \\\\ \n\\midrule\n")
    for _, r in df.iterrows():
        dom = latex_escape(r["domain"])
        lines.append(
            f"{dom} & {int(r['k'])} & {r['H']:.2f} & {int(r['df'])} & {r['p_fmt']} & {r['p_fdr_fmt']} \\\\ \n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n")
    lines.append("\\caption{Kruskal--Wallis across agents within each domain.}\\label{tab:kw-by-domain}\n")
    lines.append("\\end{table*}\n")
    out_path.write_text("".join(lines))


def save_kw_by_agent(df: pd.DataFrame, out_path: Path):
    cols = ["agent", "k", "H", "df", "p_value", "p_fdr_bh_across_agents", "n_total"]
    df = df[cols].copy().sort_values("agent")
    df["p_fmt"] = df["p_value"].map(fmt_p)
    df["p_fdr_fmt"] = df["p_fdr_bh_across_agents"].map(fmt_p)
    lines = []
    lines.append("% Auto-generated; do not edit by hand\n")
    lines.append("\\begin{table*}[t]\n\\centering\n")
    lines.append("\\scriptsize\n")
    lines.append("\\begin{tabular}{lrrrrr}\n\\toprule\n")
    lines.append("Agent & $k$ & $H$ & df & $p$ (raw) & $p_\\text{FDR}$ \\\\ \n\\midrule\n")
    for _, r in df.iterrows():
        agent = latex_escape(r["agent"])
        lines.append(
            f"{agent} & {int(r['k'])} & {r['H']:.2f} & {int(r['df'])} & {r['p_fmt']} & {r['p_fdr_fmt']} \\\\ \n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n")
    lines.append("\\caption{Kruskal--Wallis across domains within each agent.}\\label{tab:kw-by-agent}\n")
    lines.append("\\end{table*}\n")
    out_path.write_text("".join(lines))


def save_top_pairs(df: pd.DataFrame, domain: str, out_path: Path, top_n: int = 12):
    dfd = df[df["domain"] == domain].copy()
    # Prefer BH-significant pairs; break ties by normalized Wasserstein distance
    dfd["sig"] = dfd["p_fdr_bh_within_domain"].fillna(1.0) <= 0.05
    dfd = dfd.sort_values(["sig", "wasserstein_norm"], ascending=[False, False])
    dfd = dfd.head(top_n)

    def fnum(x):
        if pd.isna(x):
            return "--"
        return f"{x:.3f}"

    lines = []
    lines.append("% Auto-generated; do not edit by hand\n")
    lines.append("\\begin{table*}[t]\n\\centering\n")
    lines.append("\\scriptsize\n")
    lines.append("\\begin{tabular}{llrrrrrrrr}\n\\toprule\n")
    lines.append("Domain & Pair & $n_a$ & $n_b$ & rb & $W_\\text{norm}$ & $p_\\text{MWU,FDR}$ & $p_{W,\\text{perm,FDR}}$ \\\\ \n\\midrule\n")
    for _, r in dfd.iterrows():
        pair = f"{latex_escape(r['agent_a'])} vs {latex_escape(r['agent_b'])}"
        dom = latex_escape(r["domain"])
        lines.append(
            f"{dom} & {pair} & {int(r['n_a'])} & {int(r['n_b'])} & {fnum(r['effect_rb'])} & {fnum(r['wasserstein_norm'])} & {fmt_p(r['p_fdr_bh_within_domain'])} & {fmt_p(r.get('ws_p_fdr_bh_within_domain', float('nan')))} \\\\ \n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n")
    # Escape domain for caption to avoid underscores entering text mode as subscripts
    dom_caption = latex_escape(domain)
    lines.append(
        f"\\caption{{Top {top_n} pairwise contrasts by normalized Wasserstein in {dom_caption} (BH within-domain).}}\\label{{tab:top-pairs-{domain}}}\n"
    )
    lines.append("\\end{table*}\n")
    out_path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="rw17_indep_causes")
    ap.add_argument("--version", default="2")
    ap.add_argument("--results-dir", default="results/domain_differences")
    ap.add_argument("--tables-dir", default="publication/thesis/tuebingen_thesis_msc/tables")
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--prompt_category", default="numeric", help="Focus prompt_category (default: numeric)")

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    res_dir = root / args.results_dir / args.experiment / args.prompt_category
    # Place tables under a subfolder named after the experiment, e.g., tables/<experiment>/...
    out_dir_base = root / args.tables_dir
    out_dir = out_dir_base / args.experiment / args.prompt_category
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    kw_by_domain = pd.read_csv(res_dir / f"kw_across_agents_per_domain_v{args.version}.csv")
    kw_by_agent = pd.read_csv(res_dir / f"kw_across_domains_per_agent_v{args.version}.csv")
    pairwise = pd.read_csv(res_dir / f"pairwise_mwu_across_agents_per_domain_v{args.version}.csv")

    # Save KW tables
    save_kw_by_domain(kw_by_domain, out_dir / f"kw_across_agents_per_domain_v{args.version}.tex")
    save_kw_by_agent(kw_by_agent, out_dir / f"kw_across_domains_per_agent_v{args.version}.tex")

    # Save top pairwise per domain
    for d in sorted(pairwise["domain"].dropna().unique().tolist()):
        save_top_pairs(pairwise, d, out_dir / f"pairwise_top_by_wasserstein_norm_{d}_v{args.version}.tex", top_n=args.top_n)

    print(f"Saved LaTeX tables to: {out_dir}")


if __name__ == "__main__":
    main()
