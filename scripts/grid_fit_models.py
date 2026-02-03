#!/usr/bin/env python3
"""Grid runner for causal Bayes net fits.

Loops over combinations of:
    - model link function: logistic, noisy_or
    - parameter tying sizes: 3,4,5
    - agents (default: all present in processed data after filtering flags)
    - prompt categories (default: all present)
    - domains (optional: if --by-domain provided, fits per domain; else pooled). When --by-domain is set, the script automatically restricts domains to those present for each (agent, prompt_category); if none are present it falls back to a pooled fit.
    - learning rates (single --lr or multiple --learning-rates)

For each Cartesian combination a call is made into `causalign.analysis.model_fitting.cli`
which persists a validated GroupFitResult JSON plus updates `fit_index.parquet`
under results/model_fitting/<experiment>/<lr_subdir>/.

Comprehensive example using most parameters:

    python scripts/grid_fit_models.py \
            --experiment rw17_indep_causes \
            --version 2 \
            --graph-type collider \
            --pipeline-mode llm_with_humans \
            --temperature 0.0 \
            --models logistic noisy_or \
            --params 3 4 5 \
            --prompt-categories numeric numeric-conf CoT \
            --agents gpt-4o claude-3-opus humans \
            --by-domain \
            --domains weather economy health \
            --learning-rates 0.01 0.05 0.1 \
            --loss mse \
            --optimizer lbfgs \
            --epochs 400 \
            --restarts 10 \
            --seed 123 \
            --device auto \
            --enable-loocv \
            --verbose

Minimal example (defaults discover agents / prompts / domains):

    python scripts/grid_fit_models.py --experiment rw17_indep_causes --version 2

Using Adam & Huber with explicit learning rates (and pooled domains):

    python scripts/grid_fit_models.py \
            --experiment rw17_indep_causes --version 2 \
            --models logistic \
            --params 3 5 \
            --learning-rates  0.1 \
            --loss huber --optimizer lbfgs --restarts 2 --epochs 50 \
            --agent gpt-4o

Humans fitting modes (aggregated vs pooled vs individual):
     There are three useful ways to fit to human data:
     1) Aggregated humans (default when you select agent 'humans' and aggregated rows exist):
         - Uses pre-aggregated human responses (typically mean per item), where rows have subject='humans'.
         - Invoke with: --agents humans  (and DO NOT set --no-aggregated)
     2) Pooled humans (new):
         - Pools all individual human rows (where a per-participant id exists) into a single synthetic agent.
         - Use: --agents humans --humans-mode pooled
            (or explicitly --agents humans-pooled)
     3) Individual humans:
         - Splits the aggregated label into per-participant agent variants (human-<id>) so each participant is fit separately.
         - Use: --agents all-humans  (requires a per-participant id column, auto-detected as human_subj_id)
         - You can also set --humans-mode individual and pass --agents humans to expand to all individuals.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure src/ is importable before importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from causalign.analysis.model_fitting.data import load_processed_data  # type: ignore  # noqa: E402
from causalign.config.paths import PathManager  # type: ignore  # noqa: E402
from causalign.analysis.model_fitting.cli import main as fit_main  # type: ignore  # noqa: E402

# Ensure src/ is importable
# (legacy duplicated import block removed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid-search CBN fits over models and data subsets")
    # Required
    p.add_argument("--experiment", required=True, help="Experiment name (e.g., abstract_reasoning)")
    p.add_argument("--version", required=True, help="Version (e.g., 2)")

    # Data loading parity with fit_models
    p.add_argument("--graph-type", choices=["collider", "fork", "chain"], default="collider")
    p.add_argument("--pipeline-mode", choices=["llm_with_humans", "llm", "humans"], default="llm_with_humans")
    p.add_argument("--no-roman-numerals", action="store_true")
    p.add_argument("--no-aggregated", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--input-file")
    p.add_argument(
        "--split-humans-by",
        help=(
            "Column name to split aggregated 'humans' into per-participant agents "
            "(e.g., human_subj_id, worker_id, participant_id). Common aliases like 'humans_subj_id' are accepted."
        ),
    )
    p.add_argument(
        "--humans-mode",
        choices=["auto", "aggregated", "pooled", "individual"],
        default="auto",
        help=(
            "How to treat human rows when 'humans' is requested as an agent: "
            "'aggregated' → use pre-aggregated humans rows only; "
            "'pooled' → pool all individual human rows into a single synthetic agent 'humans-pooled'; "
            "'individual' → expand 'humans' to all per-participant variants (requires split column); "
            "'auto' (default) → do not override: if aggregated rows exist, '--agents humans' targets aggregated."
        ),
    )

    # Subset control
    p.add_argument(
        "--agents",
        nargs="*",
        help=(
            "Limit to these agents (default: all present). "
            "Special token: 'all-humans' → expands to all per-individual human variants (requires split or auto-detected human_subj_id)."
        ),
    )
    p.add_argument("--prompt-categories", nargs="*", help="Limit to these prompt categories (default: all present)")
    p.add_argument(
        "--domains",
        nargs="*",
        help=(
            "Limit to these domains. Special token: 'all' → include all available domains. "
            "Interaction with --by-domain: \n"
            "  • Without --by-domain (default): pooled fit across the included domains.\n"
            "  • With --by-domain: fits each included domain separately."
        ),
    )
    p.add_argument(
        "--by-domain",
        action="store_true",
        help=(
            "When set, fits each domain separately (one fit per domain). "
            "When omitted, fits a single pooled model across the included domain rows."
        ),
    )

    # Model grid
    p.add_argument("--models", nargs="*", default=["logistic", "noisy_or"], choices=["logistic", "noisy_or"])
    p.add_argument("--params", nargs="*", default=["3", "4", "5"], choices=["3", "4", "5"], help="Parameter tying sizes")

    # Optimization knobs (forwarded)
    p.add_argument("--loss", default="mse", choices=["mse", "huber"]) 
    p.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "adam"]) 
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.1, help="Single learning rate (used if --learning-rates not specified)")
    p.add_argument("--learning-rates", nargs="*", type=float, help="Multiple learning rates to grid search over (e.g., 0.01 0.1 1.0)")
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--enable-loocv", action="store_true", help="Enable leave-one-out cross-validation for all fits")
    p.add_argument("--dry-run", action="store_true", help="List planned fit commands and exit without running")
    p.add_argument("--primary-metric", default="aic", choices=["aic","bic","rmse","loss","cv_rmse","r2"], help="Primary metric for restart ranking (forwarded)")
    p.add_argument("--selection-rule", default="best_loss", choices=["best_loss","median_loss","best_primary_metric"], help="Restart selection rule (forwarded)")
    p.add_argument("--skip-empty", action="store_true", help="Skip (do not invoke fitter) when a combination has zero matching rows instead of producing an error")
    p.add_argument("--inspect-filters", action="store_true", help="Print discovered agents/prompt categories/domains (and mismatches) before running the grid")

    # Output control
    p.add_argument("--output-dir", help="Override output dir for fitter (optional)")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    paths = PathManager()

    # Load once for discovery of present agents/domains/prompt_categories
    use_roman = not args.no_roman_numerals
    use_agg = not args.no_aggregated
    # If the user explicitly chose pooled or individual humans, prefer non-aggregated rows
    if getattr(args, "humans_mode", "auto") in {"pooled", "individual"} and use_agg:
        print("[INFO] --humans-mode is", args.humans_mode, "→ disabling aggregated rows for loading.")
        use_agg = False

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

    # Ensure GPT-5 variant labeling, mirroring logic in human_llm_alignment_correlation.py
    def _ensure_variant_columns(df_: pd.DataFrame) -> pd.DataFrame:
        df_ = df_.copy()
        if "verbosity" not in df_.columns:
            df_["verbosity"] = "n/a"
        if "reasoning_effort" not in df_.columns:
            df_["reasoning_effort"] = "n/a"
        for c in ["verbosity", "reasoning_effort"]:
            df_[c] = (
                df_[c].astype(str).str.strip().str.lower().replace({"": "n/a", "nan": "n/a"})
            )
        subj_str = df_["subject"].astype(str)
        is_gpt5 = subj_str.str.startswith("gpt-5")
        already_variant = subj_str.str.contains(r"-v_.*-r_.*", regex=True)
        has_meta = ~(
            df_["verbosity"].isin(["n/a", "unspecified"]) & df_["reasoning_effort"].isin(["n/a", "unspecified"])
        )
        df_["agent_variant"] = df_["subject"].astype(str)
        df_.loc[is_gpt5 & has_meta & ~already_variant, "agent_variant"] = (
            df_.loc[is_gpt5 & has_meta, "subject"].astype(str)
            + "-v_" + df_.loc[is_gpt5 & has_meta, "verbosity"].astype(str)
            + "-r_" + df_.loc[is_gpt5 & has_meta, "reasoning_effort"].astype(str)
        )
        return df_

    df = _ensure_variant_columns(df)

    # Auto-enable per-human splitting when a canonical id column is present
    if not args.split_humans_by and "human_subj_id" in df.columns:
        subj_lower = df["subject"].astype(str).str.strip().str.lower()
        if (subj_lower.isin(["humans", "human"]) | subj_lower.str.startswith("human-")).any() and df["human_subj_id"].notna().any():
            args.split_humans_by = "human_subj_id"
            print("[INFO] Auto-enabling --split-humans-by=human_subj_id to avoid overwriting per-individual human fits.")

    # Optionally split aggregated humans into per-participant agent variants
    if args.split_humans_by:
        requested_col = str(args.split_humans_by)
        col = requested_col
        if col not in df.columns:
            # Try common aliases
            aliases = [
                requested_col,
                requested_col.replace("s_subj", "_subj"),  # humans_subj_id -> human_subj_id
                requested_col.replace("humans_", "human_"),
                "human_subj_id",
                "participant_id",
                "worker_id",
                "subject_id",
                "human_id",
            ]
            col = next((c for c in aliases if c in df.columns), requested_col)
            if col != requested_col:
                print(f"[INFO] --split-humans-by={requested_col} not found; using existing column '{col}'.")
        if col not in df.columns:
            print(f"[WARN] --split-humans-by={requested_col} requested but column not found; keeping aggregated humans.")
        else:
            # Identify aggregated human rows; common labels: 'humans', 'human'
            subj = df["subject"].astype(str)
            is_human = subj.str.strip().str.lower().isin(["humans", "human"]) | subj.str.strip().str.lower().str.startswith("human-")
            has_id = df[col].notna()
            if not (is_human & has_id).any():
                print(f"[WARN] No split candidates found for column '{col}'; keeping aggregated humans.")
            else:
                def _mk_variant(v: object) -> str:
                    s = str(v)
                    # Normalize numeric IDs: 1.0 -> 1
                    try:
                        f = float(s)
                        if f.is_integer():
                            s = str(int(f))
                    except Exception:
                        pass
                    # Sanitize to filesystem/identifier friendly
                    import re as _re
                    s = _re.sub(r"[^A-Za-z0-9_.-]", "_", s)
                    return f"human-{s}"

                df = df.copy()
                df.loc[is_human & has_id, "agent_variant"] = df.loc[is_human & has_id, col].map(_mk_variant)
                # Ensure subject mirrors the variant when we isolate inputs for the fitter downstream
                # (we relabel subject later for matched variants in the loop; here we keep source intact)

    # Determine value spaces (use variant-aware agents)
    all_agents = (
        sorted(df["agent_variant"].dropna().unique().tolist()) if "agent_variant" in df.columns else []
    )
    all_prompts = sorted(df["prompt_category"].dropna().unique().tolist()) if "prompt_category" in df.columns else []
    all_domains = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []

    # If user specified agents, allow base names or exact variant names and special token 'all-humans'
    if args.agents:
        requested = [str(a) for a in args.agents]
        # Humans-mode rewrites for convenience
        hm = getattr(args, "humans_mode", "auto")
        # If user asked for 'humans' and wants individual, rewrite to 'all-humans'
        if hm == "individual":
            requested = ["all-humans" if a.lower().replace("_", "-") == "humans" else a for a in requested]
        # If user asked for 'humans' and wants pooled, rewrite to 'humans-pooled'
        if hm == "pooled":
            requested = ["humans-pooled" if a.lower().replace("_", "-") == "humans" else a for a in requested]
        # expand base names to their variants
        def expand(agent: str) -> list[str]:
            norm_agent = agent.lower().replace("_", "-")
            # Always allow special/explicit tokens through even if not discovered in all_agents
            if norm_agent in {"humans", "humans-pooled"}:
                return [norm_agent]
            if agent in all_agents:
                return [agent]
            # include any variant that starts with the agent token (e.g., 'gpt-5')
            return [a for a in all_agents if str(a).startswith(agent)] or ([agent] if agent in all_agents else [])

        expanded: list[str] = []
        # Handle special token 'all-humans'
        if any(a.lower().replace("_", "-") == "all-humans" for a in requested):
            # Support both legacy 'humans-' and new 'human-' prefixes
            human_variants = [a for a in all_agents if str(a).startswith("human-") or str(a).startswith("humans-")]
            expanded.extend(human_variants)
            print(f"[INFO] 'all-humans' expanded to {len(human_variants)} individual humans:")
            print(f"        {human_variants}")
        # Expand remaining tokens
        for a in requested:
            norm = a.lower().replace("_", "-")
            if norm == "all-humans":
                continue
            if norm == "humans-pooled":
                # Keep synthetic pooled humans token
                expanded.append("humans-pooled")
            else:
                expanded.extend(expand(a))
        # If user chose humans-mode=aggregated, drop any per-human variants unless explicitly listed
        if hm == "aggregated":
            explicitly_listed = {a.lower().replace("_", "-") for a in requested}
            expanded = [a for a in expanded if not (str(a).startswith("human-") or str(a).startswith("humans-")) or (str(a).lower() in explicitly_listed)]
        agents = sorted(set(expanded))
    else:
        agents = all_agents
    prompt_cats = args.prompt_categories or all_prompts
    # Map '--domains all' to no restriction (use discovered domains)
    if args.domains and any(str(d).strip().lower() == "all" for d in args.domains):
        print("[INFO] --domains all → using all available domains")
        args.domains = None
    domains = args.domains or all_domains

    if not agents:
        print("No agents found; nothing to fit")
        return 1
    if not prompt_cats:
        print("No prompt categories found; nothing to fit")
        return 1
    if args.by_domain and not domains:
        print("--by-domain set but no domains found; nothing to fit")
        return 1

    # Determine learning rates to use
    learning_rates = args.learning_rates if args.learning_rates else [args.lr]

    # Pre-compute total job count
    domain_factor = len(domains) if args.by_domain else 1
    total_jobs = (
        len(learning_rates)
        * len(args.models)
        * len(args.params)
        * len(prompt_cats)
        * domain_factor
        * len(agents)
    )

    if args.dry_run:
        print(f"[DRY-RUN] Planned fits: {total_jobs}")

    if args.inspect_filters:
        def _missing(requested, available):
            return sorted(set(requested) - set(available)) if requested else []
        missing_agents = _missing(args.agents, all_agents)
        missing_prompts = _missing(args.prompt_categories, all_prompts)
        missing_domains = _missing(args.domains, all_domains)
        print("=== Filter Inspection ===")
        print(f"Discovered agents ({len(all_agents)}): {all_agents}")
        if args.agents:
            print(f"Requested agents: {args.agents}")
            if missing_agents:
                print(f"  Missing agents (no rows): {missing_agents}")
        print(f"Discovered prompt categories ({len(all_prompts)}): {all_prompts}")
        if args.prompt_categories:
            print(f"Requested prompt categories: {args.prompt_categories}")
            if missing_prompts:
                print(f"  Missing prompt categories (no rows): {missing_prompts}")
        print(f"Discovered domains ({len(all_domains)}): {all_domains}")
        if args.domains:
            print(f"Requested domains: {args.domains}")
            if missing_domains:
                print(f"  Missing domains (no rows): {missing_domains}")
        print(f"Learning rates: {learning_rates}")
        print(f"Models: {args.models}; Params: {args.params}; Loss: {args.loss}; Optimizer: {args.optimizer}")
        print(f"Primary metric: {args.primary_metric}; Selection rule: {args.selection_rule}")
        print(f"Total planned jobs: {total_jobs}")
        print("Run mode:")
        if args.by_domain:
            print("  • Per-domain fits (one per domain)")
        else:
            print("  • Pooled across domains (single fit uses all included domain rows)")
        print("=========================")
        if args.dry_run and (missing_agents or missing_prompts or missing_domains):
            print("[DRY-RUN] Exiting after inspection due to missing requested values.")
            return 0

    job_index = 0

    # Helper to parse variant label into base subject and meta (if present)
    variant_re = re.compile(r"^(?P<base>.+?)-v_(?P<v>[^-]+)-r_(?P<r>.+)$")

    # Loop over grid (added learning rate dimension)
    for lr in learning_rates:
        for model in args.models:
            for params_tying in args.params:
                for prompt_cat in prompt_cats:
                    for agent in agents:
                        # Determine domain iteration early for logging
                        if args.by_domain:
                            agent_mask = df["subject"].astype(str) == str(agent)
                            pc_mask = df["prompt_category"].astype(str) == str(prompt_cat)
                            # Use agent_variant-based mask when available
                            if "agent_variant" in df.columns:
                                agent_mask = df["agent_variant"].astype(str) == str(agent)
                            dom_values = (
                                df.loc[agent_mask & pc_mask, "domain"].dropna().unique().tolist()
                                if "domain" in df.columns
                                else []
                            )
                            # If user provided a subset of domains (not 'all'), filter the discovered domains
                            if args.domains:
                                dom_values = [d for d in dom_values if str(d) in set(str(x) for x in args.domains)]
                            domain_iter = sorted(dom_values)
                            if not domain_iter:
                                # Fallback: pooled (None) if no per-domain data for this agent+prompt
                                if args.verbose:
                                    print(f"[FALLBACK] No domains for agent={agent} prompt={prompt_cat}; using pooled fit instead.")
                                domain_iter = [None]
                            elif args.verbose:
                                print(f"[INFO] Agent {agent} prompt {prompt_cat} domains: {domain_iter}")
                        else:
                            domain_iter = [None]

                        for domain in domain_iter:
                            job_index += 1
                            # Create learning rate subdirectory
                            lr_str = f"lr{lr:g}".replace(".", "p")  # e.g., lr0p01 for 0.01, lr0p1 for 0.1, lr1 for 1.0
                            
                            # Build variant-aware filtered input (and relabel subject to variant when applicable)
                            tmp_input_file = None
                            agent_arg = str(agent)
                            df_subset = df.copy()
                            # Special synthetic pooled humans agent
                            if agent_arg.lower().replace("_", "-") == "humans-pooled":
                                subj = df["subject"].astype(str).str.strip().str.lower()
                                # Find a suitable per-human id column
                                id_col = None
                                for cand in [
                                    "human_subj_id",
                                    "humans_subj_id",
                                    "participant_id",
                                    "worker_id",
                                    "subject_id",
                                    "human_id",
                                ]:
                                    if cand in df.columns:
                                        id_col = cand
                                        break
                                has_id = df[id_col].notna() if id_col else pd.Series([False] * len(df))
                                is_humanish = subj.isin(["humans", "human"]) | subj.str.startswith("human-") | subj.str.startswith("humans-")
                                sel = is_humanish & has_id
                                df_subset = df[sel].copy()
                                if not df_subset.empty:
                                    # Relabel to a stable synthetic subject for grouping and provenance
                                    df_subset.loc[:, "subject"] = "humans-pooled"
                                    # Ensure agent_variant matches subject for downstream discovery consistency
                                    if "agent_variant" in df_subset.columns:
                                        df_subset.loc[:, "agent_variant"] = "humans-pooled"
                            else:
                                m = variant_re.match(agent_arg)
                                if m:
                                    base = m.group("base")
                                    v = m.group("v").strip().lower()
                                    r = m.group("r").strip().lower()
                                    subj = df["subject"].astype(str)
                                    # Two matching strategies:
                                    #  1) Subject already carries variant suffix exactly matching agent_arg
                                    #  2) Subject is base and meta columns match v/r
                                    mask_variant_subject = subj == agent_arg
                                    mask_base_meta = (
                                        (subj == base)
                                        & (df["verbosity"].astype(str).str.lower() == v)
                                        & (df["reasoning_effort"].astype(str).str.lower() == r)
                                    )
                                    sel = mask_variant_subject | mask_base_meta
                                    df_subset = df[sel].copy()
                                    # Relabel to variant so downstream index records the variant as the agent
                                    if not df_subset.empty:
                                        df_subset.loc[:, "subject"] = agent_arg
                                else:
                                    # Non-variant: select by subject equals agent label if present in subject,
                                    # otherwise select by agent_variant equals agent and relabel subject
                                    if (df["subject"].astype(str) == agent_arg).any():
                                        df_subset = df[df["subject"].astype(str) == agent_arg].copy()
                                    elif ("agent_variant" in df.columns) and (df["agent_variant"].astype(str) == agent_arg).any():
                                        df_subset = df[df["agent_variant"].astype(str) == agent_arg].copy()
                                        if not df_subset.empty:
                                            df_subset.loc[:, "subject"] = agent_arg
                            # Determine effective prompt_category for this agent: map human-like 'numeric' → 'single_numeric_response'
                            def _is_humanish(label: str) -> bool:
                                s = str(label).strip().lower()
                                return (
                                    s in {"humans", "human", "humans-pooled"}
                                    or s.startswith("human-")
                                    or s.startswith("humans-")
                                )

                            effective_prompt_cat = str(prompt_cat)
                            if _is_humanish(agent_arg):
                                # If user asked for 'numeric' but this agent only has 'single_numeric_response', switch.
                                if effective_prompt_cat == "numeric":
                                    available_prompts = set(df_subset["prompt_category"].astype(str).unique().tolist())
                                    if "single_numeric_response" in available_prompts and "numeric" not in available_prompts:
                                        effective_prompt_cat = "single_numeric_response"

                            # Apply prompt_category and domain filters to subset if specified
                            df_subset = df_subset[df_subset["prompt_category"].astype(str) == str(effective_prompt_cat)]
                            if domain is not None:
                                df_subset = df_subset[df_subset["domain"].astype(str) == str(domain)]
                            # In pooled mode, apply optional domain subset filter if provided (and not 'all')
                            elif args.domains:
                                df_subset = df_subset[df_subset["domain"].astype(str).isin([str(x) for x in args.domains])]

                            # If empty, optionally skip
                            if df_subset.empty:
                                if args.verbose:
                                    print(f"[SKIP] No rows after variant filtering for agent={agent_arg}, prompt={prompt_cat}, domain={domain}")
                                if args.skip_empty:
                                    continue

                            # Write temp CSV for this specific combination
                            tmp_dir = Path(tempfile.gettempdir()) / "cbn_variant_inputs" / args.experiment / lr_str
                            tmp_dir.mkdir(parents=True, exist_ok=True)
                            safe_agent = re.sub(r"[^A-Za-z0-9_.-]", "_", agent_arg)
                            safe_dom = "all" if domain is None else re.sub(r"[^A-Za-z0-9_.-]", "_", str(domain))
                            tmp_input_file = tmp_dir / f"input_{safe_agent}_{prompt_cat}_{safe_dom}.csv"
                            df_subset.to_csv(tmp_input_file, index=False)

                            # Build argv for the fitter
                            # Prefer the concrete temperature found in this subset (if unique)
                            effective_temp = args.temperature
                            if "temperature" in df_subset.columns:
                                non_nan_temps = [t for t in df_subset["temperature"].dropna().unique().tolist()]
                                if len(non_nan_temps) == 1:
                                    try:
                                        effective_temp = float(non_nan_temps[0])
                                    except Exception:
                                        pass
                            fit_argv = [
                                "--experiment", args.experiment,
                                "--version", str(args.version),
                                "--graph-type", args.graph_type,
                                "--pipeline-mode", args.pipeline_mode,
                                "--temperature", str(effective_temp),
                                "--model", model,
                                "--params", str(params_tying),
                                "--loss", args.loss,
                                "--optimizer", args.optimizer,
                                "--epochs", str(args.epochs),
                                "--lr", str(lr),  # Use current learning rate from loop
                                "--restarts", str(args.restarts),
                                "--seed", str(args.seed),
                                "--device", args.device,
                                "--primary-metric", args.primary_metric,
                                "--selection-rule", args.selection_rule,
                                "--agents", agent_arg,
                                "--prompt-categories", effective_prompt_cat,
                            ]
                            # Always pass our filtered input file to isolate the variant
                            fit_argv += ["--input-file", str(tmp_input_file)]
                            if args.no_roman_numerals:
                                fit_argv += ["--no-roman-numerals"]
                            if args.no_aggregated:
                                fit_argv += ["--no-aggregated"]
                            if domain is not None:
                                fit_argv += ["--domains", domain]
                            
                            # Create output directory with learning rate subdirectory
                            if args.output_dir:
                                # New semantics: --output-dir is treated as a base root; we add experiment & lr subdir once
                                base_output_dir = Path(args.output_dir) / args.experiment / lr_str
                            else:
                                # Default root
                                base_output_dir = Path("results") / "model_fitting" / args.experiment / lr_str

                            # Route human modes into distinct subdirectories to avoid overwrites
                            def _human_mode_for(agent_label: str) -> Optional[str]:
                                s = str(agent_label).strip().lower()
                                if s == "humans-pooled":
                                    return "pooled"
                                if s == "humans":
                                    return "aggregated"
                                if s.startswith("human-") or s.startswith("humans-"):
                                    return "individual"
                                return None

                            hm_label = _human_mode_for(agent_arg)
                            if hm_label:
                                base_output_dir = base_output_dir / "humans_mode" / hm_label

                            # Isolate variant outputs to avoid collisions between variants of same base model
                            if variant_re.match(agent_arg):
                                base_output_dir = base_output_dir / "variants" / safe_agent
                            # Also isolate per-human variants produced via --split-humans-by (labels like 'human-<id>' or 'humans-<id>')
                            elif str(agent_arg).startswith("human-") or str(agent_arg).startswith("humans-"):
                                base_output_dir = base_output_dir / "variants" / safe_agent
                            fit_argv += ["--output-dir", str(base_output_dir)]
                            
                            if args.verbose:
                                fit_argv += ["--verbose"]
                            if args.enable_loocv:
                                fit_argv += ["--enable-loocv"]

                            if args.dry_run:
                                cmd_str = "grid_fit_models.py " + " ".join(fit_argv)
                                print(f"[DRY-RUN {job_index}/{total_jobs}] {cmd_str}")
                                continue

                            # No additional preflight needed; we've already filtered to concrete rows
                            # Execute one fit for this combination
                            return_code = fit_main(fit_argv)
                            if return_code != 0:
                                print(f"Fit failed for: lr={lr}, model={model}, params={params_tying}, agent={agent}, prompt_category={prompt_cat}, domain={domain}")

    if args.dry_run:
        print("[DRY-RUN] No fits executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


