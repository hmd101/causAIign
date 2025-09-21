#!/usr/bin/env python3
"""Batch plot Agent vs CBN predictions (index-based model fit discovery).

Modernized to use the structured fitting artifacts (`fit_index.parquet` + hashed
`fit_<short_spec>_<short_group>.json` files) instead of brittle filename parsing
to discover agents. Only the dedicated single-plot script
`plot_agent_vs_cbn_predictions.py` is invoked – this wrapper orchestrates
batch enumeration across agents and (optionally) domain / prompt_category
combinations.

Key behaviours:
  * Agents discovered from unique `agent` values in the experiment's index.
  * Supports filtering model fits via model types, param counts, loss functions,
    learning rates (alias: --learning-rate) and best-only selection metric
    limited to columns present in the index (loss, aic, bic, rmse).
  * Condition strategies:
     --individual-conditions  => one plot per (domain, prompt_category)
     --combined-conditions    => single plot with all specified/available
     (default)                => single plot with all available if none specified

Examples:
  # Aggregate all domains, single prompt category across all agents
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 \
      --domains all --prompt-categories numeric

  # Only selected agents, show separate domain facets
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 \
      --agents gpt-4o humans --domains weather economy --facet-by domain

  # Best logistic 3p fits only (AIC)
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 \
      --agents gpt-4o --model-types logistic --param-counts 3 --best-only --metric aic

  # Multiple learning rates (explicit filter)
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 \
      --learning-rates 0.01 0.1
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set

import pandas as pd

# Ensure src on path for direct execution (before importing causalign)
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from causalign.config.paths import PathManager  # noqa: E402
from causalign.analysis.model_fitting.discovery import load_index  # noqa: E402


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def discover_agents_with_fits(model_fitting_dir: Path, filter_agents: Optional[List[str]] = None) -> List[str]:
    """Discover agents present in the structured fit index (preferred over filename parsing)."""
    logger = logging.getLogger(__name__)
    index_df = load_index(model_fitting_dir.parent.parent.parent, model_fitting_dir.name)  # base_dir, experiment
    if index_df is None or index_df.empty:
        logger.warning(f"No fit_index.parquet found or empty for {model_fitting_dir}")
        return []
    agents: Set[str] = set(index_df['agent'].dropna().unique().tolist())
    if filter_agents:
        # Expand base GPT-5 names (e.g., 'gpt-5') to variant labels present in index
        expanded: Set[str] = set()
        for a in filter_agents:
            if a in agents:
                expanded.add(a)
            else:
                expanded.update({x for x in agents if x.startswith(a)})
        agents = agents.intersection(expanded)
    agents_list = sorted(agents)
    logger.info(f"Discovered {len(agents_list)} agents via index: {agents_list}")
    return agents_list


def discover_agent_data_conditions(
    paths: PathManager,
    version: str,
    experiment_name: str,
    agent: str,
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
    pipeline_mode: str = "llm_with_humans",
) -> Tuple[List[str], List[str]]:
    """Discover available domains and prompt categories for a specific agent"""
    logger = logging.getLogger(__name__)
    
    # Determine data path using similar logic to the main script
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

    try:
        df = pd.read_csv(data_path)
        
        # Filter to specific agent
        if "subject" in df.columns:
            agent_df = df[df["subject"] == agent].copy()
        else:
            logger.warning(f"No 'subject' column found in {data_path}")
            return [], []
        
        if agent_df.empty:
            logger.warning(f"No data found for agent '{agent}' in {data_path}")
            return [], []
        
        # Extract unique domains and prompt categories
        domains = sorted(agent_df["domain"].dropna().unique()) if "domain" in agent_df.columns else []
        prompt_categories = sorted(agent_df["prompt_category"].dropna().unique()) if "prompt_category" in agent_df.columns else []
        
        logger.debug(f"Agent {agent}: {len(domains)} domains, {len(prompt_categories)} prompt categories")
        return domains, prompt_categories
        
    except FileNotFoundError:
        logger.warning(f"Agent data file not found: {data_path}")
        return [], []
    except Exception as e:
        logger.warning(f"Error loading data for agent {agent}: {e}")
        return [], []


def run_agent_plot(
    agent: str,
    experiment: str,
    version: str,
    domains: Optional[List[str]] = None,
    prompt_categories: Optional[List[str]] = None,
    facet_by: Optional[List[str]] = None,
    best_only: bool = False,
    metric: str = "aic",
    model_types: Optional[List[str]] = None,
    param_counts: Optional[List[str]] = None,
    learning_rates: Optional[List[float]] = None,
    loss_functions: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    temperature: Optional[float] = None,
    title: Optional[str] = None,
    filename_suffix: Optional[str] = None,
    title_line_length: int = 80,
    title_fontsize: Optional[int] = None,
    verbose: bool = False,
    show_uncertainty: bool = True,
    no_uncertainty: bool = False,
    uncertainty_type: str = "ci",
    uncertainty_level: float = 95,
    uncertainty_alpha: float = 0.2,
    legend_position: str = "bottom",
) -> bool:
    """Run the plot_agent_vs_cbn_predictions.py script for a specific agent"""
    logger = logging.getLogger(__name__)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/plot_agent_vs_cbn_predictions.py",
        "--agent", agent,
        "--experiment", experiment,
        "--version", version,
    ]
    
    # Add optional arguments
    if domains:
        cmd.extend(["--domains"] + domains)
    
    if prompt_categories:
        cmd.extend(["--prompt-categories"] + prompt_categories)
    
    if facet_by:
        cmd.extend(["--facet-by"] + facet_by)
    
    if best_only:
        cmd.append("--best-only")
        cmd.extend(["--metric", metric])
    
    if model_types:
        cmd.extend(["--model-types"] + model_types)
    
    if param_counts:
        cmd.extend(["--param-counts"] + param_counts)
    
    if learning_rates:
        cmd.extend(["--learning-rates"] + [str(lr) for lr in learning_rates])
    
    if loss_functions:
        cmd.extend(["--loss-functions"] + loss_functions)
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    
    if title:
        cmd.extend(["--title", title])
    
    if filename_suffix:
        cmd.extend(["--filename-suffix", filename_suffix])
    
    if title_line_length != 80:
        cmd.extend(["--title-line-length", str(title_line_length)])
    
    if title_fontsize is not None:
        cmd.extend(["--title-fontsize", str(title_fontsize)])
    
    if verbose:
        cmd.append("--verbose")
    
    # Add uncertainty options
    if no_uncertainty:
        cmd.append("--no-uncertainty")
    elif not show_uncertainty:
        cmd.append("--no-uncertainty")
    else:
        cmd.extend(["--uncertainty-type", uncertainty_type])
        if uncertainty_level != 95:
            cmd.extend(["--uncertainty-level", str(uncertainty_level)])
        if uncertainty_alpha != 0.2:
            cmd.extend(["--uncertainty-alpha", str(uncertainty_alpha)])
    
    # Add legend position
    if legend_position != "bottom":
        cmd.extend(["--legend-position", legend_position])
    
    # Add show=False to prevent plot display
    cmd.append("--no-show")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully generated plot for {agent}")
            if verbose:
                logger.debug(f"Command output: {result.stdout}")
            return True
        else:
            logger.error(f"❌ Failed to generate plot for {agent}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout generating plot for {agent}")
        return False
    except Exception as e:
        logger.error(f"❌ Exception generating plot for {agent}: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Batch generate agent vs CBN prediction plots for all available agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots for all agents and conditions
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2

  # Generate plots only for specific agents
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --agents gpt-4o claude-3-opus

  # Generate plots with specific domains and prompt categories
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --domains all --prompt-categories numeric

  # Generate plots for specific domains (not aggregated)
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --domains weather economy --facet-by domain

  # Generate only best model plots with faceting
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --best-only --metric aic --facet-by domain

  # Custom output directory with uncertainty customization
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --output-dir custom_plots --uncertainty-type se --uncertainty-alpha 0.3

  # Generate plots with custom title and disable uncertainty
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --title "Custom Analysis" --no-uncertainty

  # Generate plots for specific CBN model types only
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --model-types logistic --param-counts 3

  # Compare different CBN model types and parameter counts
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --model-types logistic noisy_or --param-counts 3 5

  # Position legend at the bottom (default, better for large legends)
  python scripts/batch_plot_agent_vs_cbn.py --experiment rw17_indep_causes --version 2 --legend-position bottom
        """,
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
        "--agents",
        nargs="+",
        help="List of specific agents to process (default: all available agents)",
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
        "--domains",
        nargs="+",
        help="List of domains to include (e.g., 'weather economy') or 'all' to aggregate all domains into one line. If not specified, uses all available domains for each agent.",
    )

    parser.add_argument(
        "--prompt-categories",
        nargs="+",
        help="List of prompt categories to include (e.g., 'numeric'). If not specified, uses all available prompt categories for each agent.",
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
        help="Metric (index column) for --best-only selection (default: aic)",
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
        "--temperature",
        type=float,
        help="Temperature filter for LLM responses",
    )

    parser.add_argument(
        "--title",
        help="Title prefix for the plots",
    )

    parser.add_argument(
        "--filename-suffix",
        help="Custom suffix to add to the filename",
    )

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
        help="Disable uncertainty bands for all plots",
    )
    
    parser.add_argument(
        "--uncertainty-type",
        choices=["ci", "se", "sd", "pi"],
        default="ci",
        help="Type of uncertainty to show (default: ci)",
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
        help="Filter to specific learning rates (e.g., '0.01 0.1 1.0'). If omitted, all available are included.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Alias for specifying a single learning rate.",
    )

    parser.add_argument(
        "--loss-functions",
        nargs="+",
        choices=["mse", "huber"],
        help="Filter to specific loss functions (e.g., 'mse huber'). If not specified, includes all available loss functions.",
    )

    parser.add_argument(
        "--individual-conditions",
        action="store_true",
        help="Create separate plots for each domain/prompt category combination",
    )

    parser.add_argument(
        "--combined-conditions",
        action="store_true",
        help="Create plots with all conditions combined",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Harmonize single learning rate alias
    if getattr(args, "learning_rate", None) is not None:
        if args.learning_rates is None:
            args.learning_rates = [args.learning_rate]
        elif args.learning_rate not in args.learning_rates:
            args.learning_rates.append(args.learning_rate)

    try:
        # Initialize paths
        paths = PathManager()

        # Discover model fitting directory
        model_fitting_dir = paths.base_dir / "results" / "model_fitting" / args.experiment
        if not model_fitting_dir.exists():
            raise FileNotFoundError(f"Model fitting directory not found: {model_fitting_dir}")

        # Discover agents with model fits
        available_agents = discover_agents_with_fits(model_fitting_dir, args.agents)
        
        if not available_agents:
            logger.error("No agents with model fits found")
            sys.exit(1)

        # Process each agent
        total_plots = 0
        successful_plots = 0
        
        for agent in available_agents:
            logger.info(f"\n🔄 Processing agent: {agent}")
            
            # Discover available conditions for this agent
            domains, prompt_categories = discover_agent_data_conditions(
                paths,
                args.version,
                args.experiment,
                agent,
                args.graph_type,
                not args.no_roman_numerals,
                not args.no_aggregated,
                args.pipeline_mode,
            )
            
            if not domains and not prompt_categories:
                logger.warning(f"No data conditions found for agent {agent}, skipping")
                continue
            
            plots_for_agent = 0
            
            # Determine plotting strategy
            if args.individual_conditions:
                # Create separate plots for each combination
                target_domains = args.domains if args.domains else domains
                target_prompt_cats = args.prompt_categories if args.prompt_categories else prompt_categories
                
                for domain in target_domains or [None]:
                    for prompt_cat in target_prompt_cats or [None]:
                        domain_list = [domain] if domain else None
                        prompt_list = [prompt_cat] if prompt_cat else None
                        
                        total_plots += 1
                        success = run_agent_plot(
                            agent=agent,
                            experiment=args.experiment,
                            version=args.version,
                            domains=domain_list,
                            prompt_categories=prompt_list,
                            facet_by=args.facet_by,
                            best_only=args.best_only,
                            metric=args.metric,
                            model_types=args.model_types,
                            param_counts=args.param_counts,
                            learning_rates=args.learning_rates,
                            loss_functions=args.loss_functions,
                            output_dir=args.output_dir,
                            temperature=args.temperature,
                            title=args.title,
                            filename_suffix=args.filename_suffix,
                            title_line_length=args.title_line_length,
                            title_fontsize=args.title_fontsize,
                            verbose=args.verbose,
                            show_uncertainty=args.show_uncertainty,
                            no_uncertainty=args.no_uncertainty,
                            uncertainty_type=args.uncertainty_type,
                            uncertainty_level=args.uncertainty_level,
                            uncertainty_alpha=args.uncertainty_alpha,
                            legend_position=args.legend_position,
                        )
                        
                        if success:
                            successful_plots += 1
                            plots_for_agent += 1
            
            elif args.combined_conditions:
                # Create single plot with all conditions
                total_plots += 1
                success = run_agent_plot(
                    agent=agent,
                    experiment=args.experiment,
                    version=args.version,
                    domains=args.domains if args.domains else (domains if domains else None),
                    prompt_categories=args.prompt_categories if args.prompt_categories else (prompt_categories if prompt_categories else None),
                    facet_by=args.facet_by,
                    best_only=args.best_only,
                    metric=args.metric,
                    model_types=args.model_types,
                    param_counts=args.param_counts,
                    learning_rates=args.learning_rates,
                    loss_functions=args.loss_functions,
                    output_dir=args.output_dir,
                    temperature=args.temperature,
                    title=args.title,
                    filename_suffix=args.filename_suffix,
                    title_line_length=args.title_line_length,
                    title_fontsize=args.title_fontsize,
                    verbose=args.verbose,
                    show_uncertainty=args.show_uncertainty,
                    no_uncertainty=args.no_uncertainty,
                    uncertainty_type=args.uncertainty_type,
                    uncertainty_level=args.uncertainty_level,
                    uncertainty_alpha=args.uncertainty_alpha,
                    legend_position=args.legend_position,
                )
                
                if success:
                    successful_plots += 1
                    plots_for_agent += 1
            
            else:
                # Default: create plot with all available conditions
                total_plots += 1
                success = run_agent_plot(
                    agent=agent,
                    experiment=args.experiment,
                    version=args.version,
                    domains=args.domains if args.domains else (domains if domains else None),
                    prompt_categories=args.prompt_categories if args.prompt_categories else (prompt_categories if prompt_categories else None),
                    facet_by=args.facet_by,
                    best_only=args.best_only,
                    metric=args.metric,
                    model_types=args.model_types,
                    param_counts=args.param_counts,
                    learning_rates=args.learning_rates,
                    loss_functions=args.loss_functions,
                    output_dir=args.output_dir,
                    temperature=args.temperature,
                    title=args.title,
                    filename_suffix=args.filename_suffix,
                    title_line_length=args.title_line_length,
                    title_fontsize=args.title_fontsize,
                    verbose=args.verbose,
                    show_uncertainty=args.show_uncertainty,
                    no_uncertainty=args.no_uncertainty,
                    uncertainty_type=args.uncertainty_type,
                    uncertainty_level=args.uncertainty_level,
                    uncertainty_alpha=args.uncertainty_alpha,
                    legend_position=args.legend_position,
                )
                
                if success:
                    successful_plots += 1
                    plots_for_agent += 1
            
            logger.info(f"Generated {plots_for_agent} plots for agent {agent}")

        # Summary
        logger.info("\n📊 Batch processing complete!")
        logger.info(f"Total plots attempted: {total_plots}")
        logger.info(f"Successful plots: {successful_plots}")
        logger.info(f"Failed plots: {total_plots - successful_plots}")

        if args.output_dir:
            output_path = Path(args.output_dir)
        else:
            output_path = paths.base_dir / "results" / "plots" / "agent_vs_cbn" / args.experiment

        logger.info(f"Plots saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
