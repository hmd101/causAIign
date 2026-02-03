#!/usr/bin/env python3
"""Batch evaluate model fits across experiments using the structured index.

Generates evaluation CSVs (combined_index, best_by_group, ranks_by_group) for
systematic combinations of metrics, models, and agents by invoking
`scripts/evaluate_model_fits.py` repeatedly. (Legacy inline heatmap generation
removed – use dedicated visualization scripts for plotting.)

Primary use cases:
    * Enumerate multiple metrics (AIC, loss, r2, LOOCV metrics) over all agents.
    * Produce per-agent evaluations and combined-agent evaluations.
    * Slice by model family (logistic vs noisy_or) and parameter tying counts.

All invocations ultimately read `fit_index.parquet` + optional `spec_manifest.csv`
previously produced by fitting scripts (e.g., `grid_fit_models.py` or
`causalign.analysis.model_fitting.cli`).

Comprehensive example (maximal parameterization):

    python scripts/batch_evaluate_model_fits.py \
            --experiments rw17_indep_causes abstract_reasoning \
            --versions 2 3 \
            --agents gpt-4o claude-3-opus humans \
            --domains weather economy health \
            --prompt-categories numeric numeric-conf \
            --models logistic noisy_or \
            --param-counts 3 4 5 \
            --loss-functions mse huber \
            --learning-rates 0.01 0.1 \
            --optimizers lbfgs adam \
            --metrics aic loss r2 loocv_rmse loocv_r2 \
            --individual-agents --combined-agents \
            --individual-models --combined-models \
            --output-dir results/modelfits/batch_eval

Minimal example (discover experiments, agents, models automatically):

    python scripts/batch_evaluate_model_fits.py --experiments rw17_indep_causes --versions 2

Agents + specific metrics only:

    python scripts/batch_evaluate_model_fits.py \
            --experiments rw17_indep_causes --versions 2 \
            --agents gpt-4o humans \
            --metrics aic r2
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from causalign.config.paths import PathManager  # type: ignore  # noqa: E402
from causalign.analysis.model_fitting.discovery import load_index  # noqa: E402

# After imports, adjust sys.path if running from source checkout (best-effort)
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def discover_available_agents(model_fitting_dir: Path) -> List[str]:
    """Discover agents from the structured fit index (preferred over filename parsing)."""
    logger = logging.getLogger(__name__)
    index_df = load_index(model_fitting_dir.parent.parent.parent, model_fitting_dir.name)
    if index_df is None or index_df.empty:
        logger.warning(f"No fit_index.parquet found for {model_fitting_dir}")
        return []
    agents = sorted(index_df['agent'].dropna().unique().tolist())
    logger.info(f"Discovered {len(agents)} agents via index: {agents}")
    return agents


def expand_gpt5_variants(available_agents: List[str], requested: Optional[List[str]]) -> List[str]:
    """Expand base GPT-5 names (e.g., 'gpt-5', 'gpt-5-mini') into concrete variant labels present in available_agents.
    If requested already contain variant labels, keep them.
    """
    if not requested:
        return available_agents
    variants = set()
    available_set = set(available_agents)
    for a in requested:
        if a in available_set:
            variants.add(a)
            continue
        # Include any agent string that starts with the requested token
        expanded = [x for x in available_agents if x.startswith(a)]
        if expanded:
            variants.update(expanded)
    return sorted(variants) if variants else requested


def run_evaluate_model_fits(
    experiments: Optional[List[str]] = None,
    versions: Optional[List[str]] = None,
    agents: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    prompt_categories: Optional[List[str]] = None,
    learning_rates: Optional[List[float]] = None,
    optimizers: Optional[List[str]] = None,
    loss_functions: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    param_counts: Optional[List[str]] = None,
    metric: str = "aic",
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Run the evaluate_model_fits.py script with specified parameters"""
    logger = logging.getLogger(__name__)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/evaluate_model_fits.py",
        "--plot-heatmaps",
        "--metric", metric,
    ]
    
    # Add optional arguments
    if experiments:
        cmd.extend(["--experiments"] + experiments)
    
    if versions:
        cmd.extend(["--versions"] + versions)
    
    if agents:
        cmd.extend(["--agents"] + agents)
    
    if domains:
        cmd.extend(["--domains"] + domains)
    
    if prompt_categories:
        cmd.extend(["--prompt-categories"] + prompt_categories)
    
    if learning_rates:
        cmd.extend(["--learning-rates"] + [str(lr) for lr in learning_rates])
    
    if optimizers:
        cmd.extend(["--optimizers"] + optimizers)
    
    if loss_functions:
        cmd.extend(["--loss-functions"] + loss_functions)
    
    if models:
        cmd.extend(["--models"] + models)
    
    if param_counts:
        cmd.extend(["--param-counts"] + param_counts)
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    # Note: evaluate_model_fits.py doesn't have --verbose flag, so we don't pass it
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            agent_str = f"agents={agents}" if agents else "all_agents"
            model_str = f"models={models}" if models else "all_models"
            logger.info(f"✅ Successfully generated heatmaps for {agent_str}, {model_str}, metric={metric}")
            if verbose:
                logger.debug(f"Command output: {result.stdout}")
            return True
        else:
            agent_str = f"agents={agents}" if agents else "all_agents"
            model_str = f"models={models}" if models else "all_models"
            logger.error(f"❌ Failed to generate heatmaps for {agent_str}, {model_str}, metric={metric}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout generating heatmaps for agents={agents}, models={models}, metric={metric}")
        return False
    except Exception as e:
        logger.error(f"❌ Exception generating heatmaps for agents={agents}, models={models}, metric={metric}: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Batch generate model fit evaluation heatmaps for all combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all heatmaps for specific experiment and version
  python scripts/batch_evaluate_model_fits.py --experiments rw17_indep_causes --versions 2

  # Generate heatmaps for specific agents only
  python scripts/batch_evaluate_model_fits.py --experiments rw17_indep_causes --versions 2 --agents gpt-4o humans

  # Generate heatmaps for specific models and metrics
  python scripts/batch_evaluate_model_fits.py --experiments rw17_indep_causes --versions 2 --models logistic --metrics aic loss r2

  # Generate heatmaps for specific loss functions
  python scripts/batch_evaluate_model_fits.py --experiments rw17_indep_causes --versions 2 --loss-functions huber --metrics loocv_rmse

  # Compare MSE vs Huber loss across all models
  python scripts/batch_evaluate_model_fits.py --experiments rw17_indep_causes --versions 2 --loss-functions mse huber --metrics aic loocv_rmse

  # Generate heatmaps for all agents individually plus combined
  python scripts/batch_evaluate_model_fits.py --experiments rw17_indep_causes --versions 2 --individual-agents --combined-agents

  # Custom output directory with verbose logging
  python scripts/batch_evaluate_model_fits.py --experiments rw17_indep_causes --versions 2 --output-dir custom_heatmaps --verbose
        """,
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Experiments to include (e.g., 'rw17_indep_causes')",
    )

    parser.add_argument(
        "--versions",
        nargs="+",
        help="Version numbers to include (e.g., '2')",
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        help="Specific agents to process (default: all available agents)",
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domains to include",
    )

    parser.add_argument(
        "--prompt-categories",
        nargs="+",
        help="Prompt categories to include",
    )

    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to include",
    )

    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["lbfgs", "adam"],
        help="Optimizers to include",
    )

    parser.add_argument(
        "--loss-functions",
        nargs="+",
        choices=["mse", "huber"],
        help="Loss functions to include (e.g., mse huber)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic", "noisy_or"],
        help="Model types to include (default: both logistic and noisy_or)",
    )

    parser.add_argument(
        "--param-counts",
        nargs="+",
        choices=["3", "4", "5"],
        help="Parameter counts to include (default: all)",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["loss", "aic", "bic", "rmse", "mae", "r2", "loocv_rmse", "loocv_mae", "loocv_r2", "loocv_consistency", "loocv_calibration"],
        default=["aic", "loss"],
        help="Metrics to generate heatmaps for (default: aic, loss)",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for heatmaps (default: results/modelfits)",
    )

    parser.add_argument(
        "--individual-agents",
        action="store_true",
        help="Generate heatmaps for each agent individually",
    )

    parser.add_argument(
        "--combined-agents",
        action="store_true",
        help="Generate heatmaps combining all agents",
    )

    parser.add_argument(
        "--individual-models",
        action="store_true",
        help="Generate heatmaps for each model type individually",
    )

    parser.add_argument(
        "--combined-models",
        action="store_true",
        help="Generate heatmaps combining all model types",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize paths
        paths = PathManager()

        # Determine experiments to process
        if not args.experiments:
            logger.error("--experiments is required")
            sys.exit(1)

        # Discover model fitting directories
        model_fitting_base = paths.base_dir / "results" / "model_fitting"
        available_agents = set()
        
        for experiment in args.experiments:
            experiment_dir = model_fitting_base / experiment
            if not experiment_dir.exists():
                logger.warning(f"Model fitting directory not found: {experiment_dir}")
                continue
            
            exp_agents = discover_available_agents(experiment_dir)
            available_agents.update(exp_agents)

        if not available_agents:
            logger.error("No agents with model fits found")
            sys.exit(1)

        # Determine which agents to process
        if args.agents:
            # Expand GPT-5 bases to variant labels if present in index
            target_agents = expand_gpt5_variants(sorted(list(available_agents)), args.agents)
            if not target_agents:
                logger.error(f"None of the specified agents {args.agents} have model fits")
                sys.exit(1)
        else:
            target_agents = sorted(list(available_agents))

        # Determine which models to process
        if args.models:
            target_models = args.models
        else:
            target_models = ["logistic", "noisy_or"]

        # Determine processing strategy
        process_individual_agents = args.individual_agents or not (args.combined_agents or args.individual_models or args.combined_models)
        process_combined_agents = args.combined_agents or not (args.individual_agents or args.individual_models or args.combined_models)
        process_individual_models = args.individual_models
        process_combined_models = args.combined_models or not (args.individual_models)

        # Track results
        total_runs = 0
        successful_runs = 0

        logger.info("\n🔄 Starting batch evaluation runs...")
        logger.info(f"Experiments: {args.experiments}")
        logger.info(f"Versions: {args.versions}")
        logger.info(f"Target agents: {target_agents}")
        logger.info(f"Target models: {target_models}")
        logger.info(f"Metrics: {args.metrics}")

        # Process each metric
        for metric in args.metrics:
            logger.info(f"\n📊 Processing metric: {metric}")

            # Individual agents
            if process_individual_agents:
                for agent in target_agents:
                    if process_individual_models:
                        # Individual agent, individual models
                        for model in target_models:
                            total_runs += 1
                            success = run_evaluate_model_fits(
                                experiments=args.experiments,
                                versions=args.versions,
                                agents=[agent],
                                domains=args.domains,
                                prompt_categories=args.prompt_categories,
                                learning_rates=args.learning_rates,
                                optimizers=args.optimizers,
                                loss_functions=args.loss_functions,
                                models=[model],
                                param_counts=args.param_counts,
                                metric=metric,
                                output_dir=args.output_dir,
                                verbose=args.verbose,
                            )
                            if success:
                                successful_runs += 1
                    
                    if process_combined_models:
                        # Individual agent, combined models
                        total_runs += 1
                        success = run_evaluate_model_fits(
                            experiments=args.experiments,
                            versions=args.versions,
                            agents=[agent],
                            domains=args.domains,
                            prompt_categories=args.prompt_categories,
                            learning_rates=args.learning_rates,
                            optimizers=args.optimizers,
                            loss_functions=args.loss_functions,
                            models=target_models,
                            param_counts=args.param_counts,
                            metric=metric,
                            output_dir=args.output_dir,
                            verbose=args.verbose,
                        )
                        if success:
                            successful_runs += 1

            # Combined agents
            if process_combined_agents:
                if process_individual_models:
                    # Combined agents, individual models
                    for model in target_models:
                        total_runs += 1
                        success = run_evaluate_model_fits(
                            experiments=args.experiments,
                            versions=args.versions,
                            agents=target_agents,
                            domains=args.domains,
                            prompt_categories=args.prompt_categories,
                            learning_rates=args.learning_rates,
                            optimizers=args.optimizers,
                            loss_functions=args.loss_functions,
                            models=[model],
                            param_counts=args.param_counts,
                            metric=metric,
                            output_dir=args.output_dir,
                            verbose=args.verbose,
                        )
                        if success:
                            successful_runs += 1
                
                if process_combined_models:
                    # Combined agents, combined models
                    total_runs += 1
                    success = run_evaluate_model_fits(
                        experiments=args.experiments,
                        versions=args.versions,
                        agents=target_agents,
                        domains=args.domains,
                        prompt_categories=args.prompt_categories,
                        learning_rates=args.learning_rates,
                        optimizers=args.optimizers,
                        loss_functions=args.loss_functions,
                        models=target_models,
                        param_counts=args.param_counts,
                        metric=metric,
                        output_dir=args.output_dir,
                        verbose=args.verbose,
                    )
                    if success:
                        successful_runs += 1

        # Summary
        logger.info("\n📊 Batch processing complete!")
        logger.info(f"Total runs attempted: {total_runs}")
        logger.info(f"Successful runs: {successful_runs}")
        logger.info(f"Failed runs: {total_runs - successful_runs}")

        if args.output_dir:
            output_path = Path(args.output_dir)
        else:
            output_path = paths.base_dir / "results" / "modelfits"

        logger.info(f"Evaluation outputs saved under: {output_path}")

        if successful_runs < total_runs:
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
