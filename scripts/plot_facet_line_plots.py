#!/usr/bin/env python3
"""
Facet Plot - Advanced Visualization Tool

Command-line interface for creating faceted line plots with flexible options.
Supports multiple faceting dimensions, overlays, and customization options.

Usage:
    # Using version and experiment (auto-constructs data path)
    python scripts/facet_plot.py --version 1 --experiment rw17_indep_causes

    # Direct file input (bypasses automatic path construction)
    python scripts/facet_plot.py --input-file data.csv --facet-by prompt_category

    # Multiple faceting dimensions with version-based path
    python scripts/facet_plot.py --version 1 --experiment pilot_study --facet-by domain temperature

    # Plot confidence values with jittering
    python scripts/facet_plot.py --version 1 --experiment pilot_study --plot-confidence --confidence-jitter 0.1
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.causalign.analysis.visualization.facet_lineplot import create_facet_line_plot  # noqa: E402
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
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load data with automatic path construction and fallback strategies

    Args:
        paths: PathManager instance
        version: Version string (e.g., "1")
        experiment_name: Experiment name (e.g., "pilot_study")
        graph_type: Graph type (e.g., "collider")
        use_roman_numerals: Whether to load the Roman numerals version
        use_aggregated: Whether to load the aggregated human responses version
        pipeline_mode: Pipeline mode ("llm_with_humans", "llm", "humans")
        input_file: Optional path to directly load a specific CSV file
        models: Optional list of model names to load (for efficiency)
    """
    logger = logging.getLogger(__name__)

    if input_file:
        # Warn about bypassing automatic path construction
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
    data = None
    errors = []

    try:
        data = pd.read_csv(data_path)
        logger.info(f"✅ Successfully loaded {len(data)} rows from primary path")

        # Filter to specific models if requested
        if models is not None and "subject" in data.columns:
            available_models = set(data["subject"].unique())
            requested_models = set(models)
            missing_models = requested_models - available_models

            if missing_models:
                logger.warning(
                    f"Models not found in data: {missing_models}. Available models: {available_models}"
                )

            # Filter to requested models that exist in the data
            valid_models = requested_models & available_models
            if valid_models:
                data = data[data["subject"].isin(valid_models)].copy()
                logger.info(
                    f"Filtered to {len(valid_models)} models: {sorted(valid_models)}"
                )
            else:
                logger.warning("No valid models found, keeping all data")

        return data
    except FileNotFoundError as e:
        errors.append(str(e))
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
                except Exception as e:
                    errors.append(str(e))

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
                except Exception as e:
                    errors.append(str(e))

    # If all attempts failed
    error_msg = "\nFailed to load data. Attempted paths:"
    for error in errors:
        error_msg += f"\n- {error}"
    raise FileNotFoundError(error_msg)


def _normalize_variant_value(s: str) -> str:
    s = str(s).strip().lower()
    synonyms = {
        "min": "minimal",
        "minimal": "minimal",
        "low": "low",
        "mid": "medium",
        "med": "medium",
        "medium": "medium",
        "high": "high",
        "max": "high",
        "n/a": "n/a",
        "unspecified": "unspecified",
    }
    return synonyms.get(s, s)


def ensure_gpt5_agent_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Create agent_variant for GPT-5 family: subject + -v_<verbosity>-r_<reasoning_effort>.
    Missing columns are filled with 'n/a'. Known synonyms normalized to avoid duplicate labels.
    """
    df = df.copy()
    if "verbosity" not in df.columns:
        df["verbosity"] = "n/a"
    if "reasoning_effort" not in df.columns:
        df["reasoning_effort"] = "n/a"
    df["verbosity"] = df["verbosity"].apply(_normalize_variant_value)
    df["reasoning_effort"] = df["reasoning_effort"].apply(_normalize_variant_value)

    subj = df["subject"].astype(str)
    is_gpt5 = subj.str.startswith("gpt-5")
    already_variant = subj.str.contains(r"-v_.*-r_.*", regex=True)
    has_meta = ~(
        df["verbosity"].isin(["n/a", "unspecified"]) & df["reasoning_effort"].isin(["n/a", "unspecified"])  # noqa: E501
    )
    df["agent_variant"] = subj
    df.loc[is_gpt5 & has_meta & ~already_variant, "agent_variant"] = (
        df.loc[is_gpt5 & has_meta, "subject"].astype(str)
        + "-v_" + df.loc[is_gpt5 & has_meta, "verbosity"].astype(str)
        + "-r_" + df.loc[is_gpt5 & has_meta, "reasoning_effort"].astype(str)
    )
    return df


def expand_subjects_to_variants(subjects: Optional[List[str]], present_subjects: List[str]) -> Optional[List[str]]:
    """Expand base names (e.g., 'gpt-5') to matching variant labels present in the data."""
    if not subjects:
        return subjects
    present_set = set(present_subjects)
    expanded: List[str] = []
    for s in subjects:
        if s in present_set:
            expanded.append(s)
            continue
        matches = [p for p in present_subjects if p.startswith(s)]
        if matches:
            expanded.extend(matches)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for s in expanded:
        if s not in seen:
            result.append(s)
            seen.add(s)
    return result or subjects


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create faceted line plots with flexible options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using version and experiment (auto-constructs data path)
  python scripts/facet_plot.py --version 1 --experiment rw17_indep_causes --facet-by prompt_category

  # Direct file input (bypasses automatic path construction)
  python scripts/facet_plot.py --input-file data.csv --facet-by domain

  # Multiple faceting dimensions with version-based path
  python scripts/facet_plot.py --version 1 --experiment pilot_study --facet-by domain temperature

  # Group all subjects in same subplot
  python scripts/facet_plot.py --version 1 --experiment pilot_study --group-subjects

  # Plot with confidence values as scatter dots
  python scripts/facet_plot.py --version 1 --experiment pilot_study --facet-by prompt_category --plot-confidence

  # Custom confidence plotting with more jitter
  python scripts/facet_plot.py --version 1 --experiment pilot_study --plot-confidence --confidence-jitter 0.1 --confidence-alpha 0.8

  # Load only specific models (for efficiency)
  python scripts/facet_plot.py --version 1 --experiment pilot_study --models gpt-4o claude-3-opus

  # Filter to specific subjects only (from loaded data)
  python scripts/facet_plot.py --version 1 --experiment pilot_study --subjects gpt-4o humans

  # Single subject analysis
  python scripts/facet_plot.py --version 1 --experiment pilot_study --subjects claude-3-opus --facet-by domain

  # Disable uncertainty bands (enabled by default)
  python scripts/facet_plot.py --version 1 --experiment pilot_study --no-uncertainty

  # Custom uncertainty visualization (uncertainty enabled by default)
  python scripts/facet_plot.py --version 1 --experiment pilot_study --uncertainty-type se --uncertainty-alpha 0.3

  # Custom output directory and title
  python scripts/facet_plot.py --version 1 --experiment pilot_study --facet-by domain --output-dir plots/faceted --title "Domain Analysis"
        """,
    )

    parser.add_argument(
        "--version",
        "-v",
        help="Version number (e.g., '1'). Required if --input-file not provided",
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

    parser.add_argument(
        "--models",
        nargs="+",
        help="List of model names to load from data (for efficiency, loads only specified models)",
    )

    parser.add_argument(
        "--facet-by",
        nargs="+",
        help="Column(s) to facet by (e.g., 'prompt_category' or 'domain temperature')",
    )

    parser.add_argument(
        "--overlay-by",
        help="Column to use for overlaying in same subplot (e.g., 'prompt_category')",
    )

    parser.add_argument(
        "--group-subjects",
        action="store_true",
        help="Plot all subjects in the same subplot",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature filter for LLM responses",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for plots (default: results/plots/faceted)",
    )

    parser.add_argument(
        "--title",
        help="Title prefix for the plots",
    )

    parser.add_argument(
        "--x-column",
        default="task",
        help="Column to use for x-axis (default: task)",
    )

    parser.add_argument(
        "--y-column",
        default="likelihood-rating",
        help="Column to use for y-axis (default: likelihood-rating)",
    )

    parser.add_argument(
        "--no-inference-groups",
        action="store_true",
        help="Don't show inference group labels",
    )

    parser.add_argument(
        "--filename-suffix",
        help="Custom suffix to add to the filename (will be prefixed with underscore)",
    )

    parser.add_argument(
        "--plot-confidence",
        action="store_true",
        help="Plot confidence values as scatter dots on the same axis",
    )

    parser.add_argument(
        "--confidence-column",
        default="confidence",
        help="Column name for confidence values (default: confidence)",
    )

    parser.add_argument(
        "--confidence-alpha",
        type=float,
        default=0.7,
        help="Alpha transparency for confidence scatter dots (default: 0.7)",
    )

    parser.add_argument(
        "--confidence-jitter",
        type=float,
        default=0.05,
        help="Amount of horizontal jitter for confidence dots to handle overlapping (default: 0.05)",
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        help="List of subjects to include in the plot (e.g., 'gpt-4o humans' or 'claude-3-opus')",
    )

    parser.add_argument(
        "--show-uncertainty",
        action="store_true",
        default=True,
        help="Show uncertainty bands around line plots (default: True)",
    )

    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Disable uncertainty bands around line plots",
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
            models=args.models,
        )

        # Display data info
        logger.info(f"\n📊 Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        logger.info(f"Subjects (raw): {list(data['subject'].unique())}")
        if "reasoning_type" in data.columns:
            logger.info(f"Reasoning types: {list(data['reasoning_type'].unique())}")
        if "task" in data.columns:
            logger.info(f"Tasks: {sorted(data['task'].unique())}")
        if "domain" in data.columns:
            logger.info(f"Domains: {list(data['domain'].unique())}")

        # Build GPT-5 agent variants and replace subject with variant label so plots separate by v/r
        data = ensure_gpt5_agent_variants(data)
        data["subject"] = data["agent_variant"]
        variant_subjects = sorted(data["subject"].dropna().unique().tolist())
        logger.info(f"Subjects (with GPT-5 variants): {variant_subjects}")

        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = paths.base_dir / "results" / "plots" / "faceted"
            if args.experiment:
                output_dir = output_dir / args.experiment

        # Determine uncertainty setting
        show_uncertainty = args.show_uncertainty and not args.no_uncertainty

        # Expand requested subjects (if any) to their variant labels present in data
        expanded_subjects = expand_subjects_to_variants(args.subjects, variant_subjects)

        # Create plot
        logger.info("\n📈 Creating faceted line plot...")
        create_facet_line_plot(
            df=data,
            facet_by=args.facet_by,
            overlay_by=args.overlay_by,
            group_subjects=args.group_subjects,
            output_dir=output_dir,
            temperature_filter=args.temperature,
            title_prefix=args.title,
            x=args.x_column,
            y=args.y_column,
            show_inference_groups=not args.no_inference_groups,
            filename_suffix=args.filename_suffix,
            plot_confidence=args.plot_confidence,
            confidence_column=args.confidence_column,
            confidence_alpha=args.confidence_alpha,
            confidence_jitter=args.confidence_jitter,
            subjects=expanded_subjects,
            show_uncertainty=show_uncertainty,
            uncertainty_type=args.uncertainty_type,
            uncertainty_level=args.uncertainty_level,
            uncertainty_alpha=args.uncertainty_alpha,
        )

        logger.info(f"✅ Plot saved to: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
