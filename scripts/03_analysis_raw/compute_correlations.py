#!/usr/bin/env python3
# TODO: resolve bugs
"""
Correlation Analysis CLI Script (canonical)

Command-line interface for computing correlations between categories or subjects.
Supports all correlation parameters and automatic visualization with adaptive sizing.

Usage:
    # Basic domain correlations for all subjects
    python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column domain

    # Model-domain pair correlations
    python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column subject domain

    # Subject correlations with specific filtering
    python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_subjects --subjects gpt-4o claude-3-opus humans --temperature 0.0

    # Grouped analysis by prompt category
    python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column domain --group-by prompt_category

    # Complex analysis with pooling
    python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column domain --pool-columns reasoning_type cntbl_cond --group-by prompt_category
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

# Add project root to path (repo_root) before importing project modules
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.causalign.analysis.statistics.correlation import (  # noqa: E402
    compute_correlations,
    compute_model_domain_correlations,
)
from src.causalign.analysis.visualization.correlation_plots import (  # noqa: E402
    plot_correlation_heatmap,
)
from src.causalign.config.paths import PathManager  # noqa: E402


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_data(
    paths: PathManager,
    version: Optional[str] = None,
    experiment_name: Optional[str] = None,
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
    pipeline_mode: str = "llm_with_humans",
    input_file: Optional[str] = None,
) -> pd.DataFrame:
    """Load data with automatic path construction and fallback strategies"""
    logger = logging.getLogger(__name__)

    if input_file:
        logger.warning(
            "\n⚠️  Using direct input file, bypassing automatic path construction"
        )
        logger.info(f"Base directory that would have been used: {paths.base_dir}")
        if version or experiment_name:
            logger.warning(
                f"Note: Ignoring provided version ({version}) and experiment ({experiment_name})"
            )

        data_path = Path(input_file)
        logger.info(f"Loading data from specified file:\n{data_path.absolute()}")
    else:
        if not version or not experiment_name:
            raise ValueError(
                "Either --input-file or both --version and --experiment must be provided"
            )

        # Base processed directory structure
        processed_base = paths.base_dir / "data" / "processed"
        logger.info("\nConstructing data path from parameters:")
        logger.info(f"Base directory: {processed_base}")
        logger.info(f"Pipeline mode: {pipeline_mode}")
        logger.info(f"Version: {version}")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Graph type: {graph_type}")
        logger.info(f"Use roman numerals: {use_roman_numerals}")
        logger.info(f"Use aggregated: {use_aggregated}")

        # Determine base directory based on pipeline mode
        if pipeline_mode == "humans":
            experiment_dir = processed_base / "humans" / "rw17"
        elif pipeline_mode == "llm":
            experiment_dir = processed_base / "llm" / "rw17" / experiment_name
        else:  # "llm_with_humans" (default)
            experiment_dir = (
                processed_base / "llm_with_humans" / "rw17" / experiment_name
            )

        logger.info(f"Experiment directory: {experiment_dir}")

        # Generate version string
        version_str = f"{version}_v_" if version else ""

        if pipeline_mode == "humans":
            # Human-only data files
            data_path = experiment_dir / f"rw17_{graph_type}_humans_processed.csv"
        elif pipeline_mode == "llm":
            # LLM-only data files
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
                # Load the Roman numerals version (has reasoning types and aggregated humans)
                data_path = (
                    experiment_dir
                    / "reasoning_types"
                    / f"{version_str}{graph_type}_cleaned_data_roman.csv"
                )
            elif use_aggregated:
                # Load the aggregated version (balanced human sample sizes)
                data_path = (
                    experiment_dir
                    / f"{version_str}humans_avg_equal_sample_size_cogsci.csv"
                )
            else:
                # Load the main processed file
                data_path = (
                    experiment_dir / f"{version_str}{graph_type}_cleaned_data.csv"
                )

        logger.info(f"\nResolved data path:\n{data_path.absolute()}")

    # Try loading with fallback strategies
    try:
        data = pd.read_csv(data_path)
        logger.info(f"✅ Successfully loaded {len(data)} rows from primary path")
        return data
    except FileNotFoundError:
        if not input_file:  # Only try fallbacks for automatic path construction
            logger.warning("Primary data file not found, trying fallbacks...")

            # Try without Roman numerals
            if use_roman_numerals:
                try:
                    data = load_data(
                        paths,
                        version=version,
                        experiment_name=experiment_name,
                        graph_type=graph_type,
                        use_roman_numerals=False,
                        use_aggregated=use_aggregated,
                        pipeline_mode=pipeline_mode,
                    )
                    logger.info("⚠️ Loaded data without Roman numerals")
                    return data
                except Exception:
                    pass

            # Try without aggregation
            if use_aggregated:
                try:
                    data = load_data(
                        paths,
                        version=version,
                        experiment_name=experiment_name,
                        graph_type=graph_type,
                        use_roman_numerals=False,
                        use_aggregated=False,
                        pipeline_mode=pipeline_mode,
                    )
                    logger.info("⚠️ Loaded data without aggregation or Roman numerals")
                    return data
                except Exception:
                    pass

        # If all attempts failed
        raise FileNotFoundError(f"Could not load data from: {data_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Compute correlations between categories or subjects with flexible options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic domain correlations for all subjects
  python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column domain

  # Model-domain pair correlations (combined categories)
  python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column subject domain

  # Subject correlations with filtering
  python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_subjects --subjects gpt-4o claude-3-opus humans --temperature 0.0

  # Grouped analysis by prompt category
  python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column domain --group-by prompt_category

  # Complex analysis with pooling
  python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column domain --pool-columns reasoning_type cntbl_cond --group-by prompt_category

  # Use convenience function for model-domain correlations
  python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --use-model-domain-convenience

  # Direct file input (bypasses automatic path construction)
  python scripts/03_analysis_raw/compute_correlations.py --input-file data.csv --correlation-type between_categories --category-column domain

  # Custom correlation method and output
  python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column domain --method pearson --output-dir custom_correlations --save-data

  # Large matrix with custom adaptive sizing
  python scripts/03_analysis_raw/compute_correlations.py --version 2 --experiment pilot_study --correlation-type between_categories --category-column domain --pool-columns reasoning_type cntbl_cond --group-by prompt_category --figsize-base 0.6 --figsize-max 30.0
        """,
    )

    # Data loading arguments
    parser.add_argument(
        "--version",
        "-v",
        help="Version number (e.g., '2'). Required if --input-file not provided",
    )

    parser.add_argument(
        "--experiment",
        "-e",
        help="Experiment name (e.g., 'pilot_study'). Required if --input-file not provided",
    )

    parser.add_argument(
        "--input-file",
        help="Path to input CSV file (bypasses automatic path construction)",
    )

    parser.add_argument(
        "--pipeline-mode",
        choices=["llm_with_humans", "llm", "humans"],
        default="llm_with_humans",
        help="Pipeline mode to load data from (default: llm_with_humans)",
    )

    parser.add_argument(
        "--graph-type",
        choices=["collider", "fork", "chain"],
        default="collider",
        help="Graph type (default: collider)",
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

    # Correlation analysis arguments
    parser.add_argument(
        "--correlation-type",
        choices=["between_categories", "between_subjects"],
        required=True,
        help="Type of correlation to compute",
    )

    parser.add_argument(
        "--category-column",
        nargs="+",
        help="Column name(s) containing categories (required for between_categories). Can specify multiple columns for combined categories (e.g., 'subject domain')",
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        help="List of subjects to include in analysis (optional filter)",
    )

    parser.add_argument(
        "--group-by",
        nargs="+",
        help="List of columns to group by before computing correlations",
    )

    parser.add_argument(
        "--pool-columns",
        nargs="+",
        help="List of columns to pool/aggregate over before correlation",
    )

    parser.add_argument(
        "--y-column",
        default="likelihood",
        help="Name of the column containing values to correlate (default: likelihood)",
    )

    parser.add_argument(
        "--method",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Correlation method (default: spearman)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature value to filter by",
    )

    # Convenience functions
    parser.add_argument(
        "--use-model-domain-convenience",
        action="store_true",
        help="Use the convenience function for model-domain correlations (overrides category-column)",
    )

    # Visualization arguments
    parser.add_argument(
        "--output-dir",
        help="Output directory for plots and data (default: results/correlations)",
    )

    parser.add_argument(
        "--title-prefix",
        help="Optional prefix for plot titles",
    )

    parser.add_argument(
        "--cmap",
        default="RdYlBu_r",
        help="Colormap for heatmap (default: RdYlBu_r)",
    )

    parser.add_argument(
        "--filename-suffix",
        help="Custom suffix to add to filenames",
    )

    # Adaptive sizing arguments
    parser.add_argument(
        "--figsize-base",
        type=float,
        default=0.8,
        help="Base size multiplier for adaptive figure sizing (default: 0.8)",
    )

    parser.add_argument(
        "--figsize-min",
        type=float,
        default=6.0,
        help="Minimum figure size (default: 6.0)",
    )

    parser.add_argument(
        "--figsize-max",
        type=float,
        default=20.0,
        help="Maximum figure size (default: 20.0)",
    )

    parser.add_argument(
        "--no-auto-figsize",
        action="store_true",
        help="Disable automatic figure sizing (use fixed size)",
    )

    # Output control arguments
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save correlation results to CSV file",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't create plots (only compute correlations)",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots interactively",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize paths
        paths = PathManager()

        # Load data
        data = load_data(
            paths,
            version=args.version,
            experiment_name=args.experiment,
            graph_type=args.graph_type,
            use_roman_numerals=not args.no_roman_numerals,
            use_aggregated=not args.no_aggregated,
            pipeline_mode=args.pipeline_mode,
            input_file=args.input_file,
        )

        # Display data info
        logger.info(f"\n📊 Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        logger.info(f"Subjects: {list(data['subject'].unique())}")
        if "reasoning_type" in data.columns:
            logger.info(f"Reasoning types: {list(data['reasoning_type'].unique())}")
        if "task" in data.columns:
            logger.info(f"Tasks: {sorted(data['task'].unique())}")
        if "domain" in data.columns:
            logger.info(f"Domains: {list(data['domain'].unique())}")
        if "prompt_category" in data.columns:
            logger.info(f"Prompt categories: {list(data['prompt_category'].unique())}")

        # Compute correlations
        logger.info("\n🔄 Computing correlations...")

        if args.use_model_domain_convenience:
            logger.info("Using model-domain convenience function")
            correlations = compute_model_domain_correlations(
                df=data,
                y=args.y_column,
                method=args.method,
                temperature_filter=args.temperature,
                group_by=args.group_by,
                pool_columns=args.pool_columns,
            )
        else:
            if (
                args.correlation_type == "between_categories"
                and not args.category_column
            ):
                raise ValueError(
                    "--category-column is required for between_categories analysis"
                )

            correlations = compute_correlations(
                df=data,
                correlation_type=args.correlation_type,
                category_column=args.category_column,
                subjects=args.subjects,
                group_by=args.group_by,
                pool_columns=args.pool_columns,
                y=args.y_column,
                method=args.method,
                temperature_filter=args.temperature,
            )

        logger.info(f"✅ Computed {len(correlations)} correlation pairs")

        # Display results summary
        if len(correlations) > 0:
            logger.info("\n📈 Correlation Results Summary:")
            logger.info(f"Mean correlation: {correlations['correlation'].mean():.3f}")
            logger.info(f"Std correlation: {correlations['correlation'].std():.3f}")
            logger.info(f"Min correlation: {correlations['correlation'].min():.3f}")
            logger.info(f"Max correlation: {correlations['correlation'].max():.3f}")

            # Show top correlations
            top_corr = correlations.nlargest(5, "correlation")
            logger.info("\nTop 5 correlations:")
            for _, row in top_corr.iterrows():
                if args.correlation_type == "between_categories":
                    logger.info(
                        f"  {row['category_1']} ↔ {row['category_2']}: {row['correlation']:.3f} (p={row['p_value']:.3f})"
                    )
                else:
                    logger.info(
                        f"  {row['subject_1']} ↔ {row['subject_2']}: {row['correlation']:.3f} (p={row['p_value']:.3f})"
                    )

        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = paths.base_dir / "results" / "correlations"
            if args.experiment:
                output_dir = output_dir / args.experiment

        # Save correlation data if requested
        if args.save_data:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create filename based on analysis type
            components = ["correlations", args.correlation_type]
            if args.category_column:
                if isinstance(args.category_column, list):
                    components.append("_".join(args.category_column))
                else:
                    components.append(args.category_column)
            if args.group_by:
                components.append(f"grouped_by_{'_'.join(args.group_by)}")
            if args.temperature is not None:
                components.append(f"temp{args.temperature}")
            if args.filename_suffix:
                components.append(args.filename_suffix)

            data_filename = f"{'_'.join(components)}.csv"
            data_path = output_dir / data_filename

            correlations.to_csv(data_path, index=False)
            logger.info(f"💾 Correlation data saved to: {data_path}")

        # Create plots if requested
        if not args.no_plot and len(correlations) > 0:
            logger.info("\n📊 Creating correlation heatmap...")

            plot_correlation_heatmap(
                corr_df=correlations,
                correlation_type=args.correlation_type,
                output_dir=output_dir,
                group_by=args.group_by,
                show=not args.no_show,
                title_prefix=args.title_prefix,
                cmap=args.cmap,
                filename_suffix=args.filename_suffix,
                figsize_base=args.figsize_base,
                figsize_min=args.figsize_min,
                figsize_max=args.figsize_max,
                auto_figsize=not args.no_auto_figsize,
            )

            logger.info(f"✅ Plots saved to: {output_dir}")

        logger.info("\n🎉 Correlation analysis complete!")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
