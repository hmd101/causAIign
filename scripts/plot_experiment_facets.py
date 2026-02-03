#!/usr/bin/env python3
"""
Script to generate facet line plots for all domains in a given experiment.

This script will:
1. Load data from the processed llm_with_humans directory
2. Generate plots for each domain in the experiment
3. Save plots with proper naming convention and folder structure
4. Replicate the input folder structure in the output directory

Usage:
    python scripts/plot_experiment_facets.py --experiment abstract_reasoning --version 1
    python scripts/plot_experiment_facets.py --experiment pilot_study --version 2 --output-dir custom_plots
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causalign.analysis.visualization.facet_lineplot import create_facet_line_plot


def discover_domains(df: pd.DataFrame) -> List[str]:
    """Discover all unique domains in the dataset."""
    if "domain" not in df.columns:
        raise ValueError("Dataset must contain a 'domain' column")

    domains = sorted(df["domain"].unique())
    print(f"Found {len(domains)} domains: {domains}")
    return domains


def discover_models(df: pd.DataFrame) -> List[str]:
    """Discover all unique models/subjects in the dataset."""
    if "subject" not in df.columns:
        raise ValueError("Dataset must contain a 'subject' column")

    # Create GPT-5 variant labels when possible
    df = df.copy()
    if "verbosity" not in df.columns:
        df["verbosity"] = "n/a"
    if "reasoning_effort" not in df.columns:
        df["reasoning_effort"] = "n/a"
    for c in ["verbosity", "reasoning_effort"]:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"": "n/a", "nan": "n/a"})
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

    models = sorted(df["agent_variant"].unique())
    print(f"Found {len(models)} models: {models}")
    return models


def load_experiment_data(
    experiment_name: str,
    version: int,
    base_dir: Path = Path("data/processed/llm_with_humans/rw17"),
) -> pd.DataFrame:
    """Load experiment data from the processed directory."""

    # Try different possible file patterns
    experiment_dir = base_dir / experiment_name

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Look for the main data file
    possible_files = [
        f"{version}_v_collider_cleaned_data.csv",
        f"{version}_v_humans_avg_equal_sample_size_cogsci.csv",
    ]

    data_file = None
    for filename in possible_files:
        candidate = experiment_dir / filename
        if candidate.exists():
            data_file = candidate
            break

    if data_file is None:
        # List available files for debugging
        available_files = list(experiment_dir.glob("*.csv"))
        raise FileNotFoundError(
            f"No suitable data file found in {experiment_dir}. "
            f"Available files: {[f.name for f in available_files]}"
        )

    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    # Validate required columns
    required_columns = ["domain", "subject", "task", "likelihood"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Handle different likelihood column names
    if "likelihood-rating" not in df.columns and "likelihood" in df.columns:
        df["likelihood-rating"] = df["likelihood"]

    # Check for confidence column availability
    has_confidence = "confidence" in df.columns
    if has_confidence:
        print(
            f"Confidence data available: {df['confidence'].notna().sum()} non-null values"
        )
    else:
        print("No confidence column found in data")

    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def create_output_structure(
    base_output_dir: Path,
    experiment_name: str,
    version: int,
    models: List[str],
    with_confidence: bool = False,
) -> Dict[str, Path]:
    """Create the output directory structure."""

    # Main experiment directory
    exp_dir = base_output_dir / experiment_name

    # Add confidence subfolder if plotting confidence
    if with_confidence:
        exp_dir = exp_dir / "with_confidence"

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create model-specific directories
    model_dirs = {}
    for model in models:
        model_dir = exp_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dirs[model] = model_dir

    return model_dirs


def generate_plots_for_domain(
    df: pd.DataFrame,
    domain: str,
    experiment_name: str,
    version: int,
    model_dirs: Dict[str, Path],
    plot_config: Dict,
) -> None:
    """Generate plots for a specific domain."""

    # Filter data for this domain
    domain_df = df[df["domain"] == domain].copy()

    if domain_df.empty:
        print(f"Warning: No data found for domain '{domain}'")
        return

    print(f"Generating plots for domain: {domain} ({len(domain_df)} rows)")

    # Build variant labels
    if "verbosity" not in domain_df.columns:
        domain_df["verbosity"] = "n/a"
    if "reasoning_effort" not in domain_df.columns:
        domain_df["reasoning_effort"] = "n/a"
    for c in ["verbosity", "reasoning_effort"]:
        domain_df[c] = domain_df[c].astype(str).str.strip().str.lower().replace({"": "n/a", "nan": "n/a"})
    subj_str = domain_df["subject"].astype(str)
    is_gpt5 = subj_str.str.startswith("gpt-5")
    already_variant = subj_str.str.contains(r"-v_.*-r_.*", regex=True)
    has_meta = ~(
        domain_df["verbosity"].isin(["n/a", "unspecified"]) & domain_df["reasoning_effort"].isin(["n/a", "unspecified"])  # noqa: E501
    )
    domain_df["agent_variant"] = domain_df["subject"].astype(str)
    domain_df.loc[is_gpt5 & has_meta & ~already_variant, "agent_variant"] = (
        domain_df.loc[is_gpt5 & has_meta, "subject"].astype(str)
        + "-v_" + domain_df.loc[is_gpt5 & has_meta, "verbosity"].astype(str)
        + "-r_" + domain_df.loc[is_gpt5 & has_meta, "reasoning_effort"].astype(str)
    )

    # Get unique models in this domain (variants)
    domain_models = sorted(domain_df["agent_variant"].unique())

    # Generate plots for each model
    for model in domain_models:
        if model not in model_dirs:
            print(f"Warning: No output directory for model '{model}'")
            continue

        model_df = domain_df[domain_df["agent_variant"] == model].copy()
        if model_df.empty:
            print(f"Warning: No data for model '{model}' in domain '{domain}'")
            continue

        # Create the plot
        output_dir = model_dirs[model]
        filename_suffix = f"{version}_v_{domain}_{experiment_name}"

        # Title includes domain and experiment info
        title = f"{domain.title()} Domain - {experiment_name.replace('_', ' ').title()}"

        try:
            create_facet_line_plot(
                df=model_df,
                output_dir=output_dir,
                title_prefix=title,
                filename_suffix=filename_suffix,
                subjects=[model],  # Filter to just this model (variant label)
                show=False,  # Don't show plots in batch mode
                **plot_config,
            )

            print(f"  ✓ Generated plot for {model} in {domain}")

        except Exception as e:
            print(f"  ✗ Error generating plot for {model} in {domain}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate facet line plots for all domains in an experiment"
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name (e.g., abstract_reasoning, pilot_study)",
    )

    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Experiment version number (e.g., 1, 2)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/line_plots"),
        help="Base output directory (default: results/line_plots)",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/llm_with_humans/rw17"),
        help="Base input directory (default: data/processed/llm_with_humans/rw17)",
    )

    parser.add_argument(
        "--temperature-filter",
        type=float,
        help="Filter to specific temperature (e.g., 0.0)",
    )

    parser.add_argument(
        "--overlay-by",
        default="prompt_category",
        help="Column to use for overlaying (default: prompt_category)",
    )

    parser.add_argument(
        "--plot-confidence",
        action="store_true",
        help="Plot confidence values as scatter dots",
    )

    parser.add_argument(
        "--no-uncertainty", action="store_true", help="Disable uncertainty bands"
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
        help="Uncertainty level (default: 95)",
    )

    parser.add_argument(
        "--domains", nargs="+", help="Specific domains to plot (default: all domains)"
    )

    parser.add_argument(
        "--models", nargs="+", help="Specific models to plot (default: all models)"
    )

    args = parser.parse_args()

    try:
        # Load data
        print(f"Loading experiment: {args.experiment} v{args.version}")
        df = load_experiment_data(args.experiment, args.version, args.input_dir)

        # Discover domains and models
        all_domains = discover_domains(df)
        all_models = discover_models(df)

        # Filter domains and models if specified
        domains_to_plot = args.domains if args.domains else all_domains
        models_to_plot = args.models if args.models else all_models

        # Validate requested domains and models
        invalid_domains = set(domains_to_plot) - set(all_domains)
        if invalid_domains:
            print(f"Warning: Invalid domains requested: {invalid_domains}")
            domains_to_plot = [d for d in domains_to_plot if d in all_domains]

        invalid_models = set(models_to_plot) - set(all_models)
        if invalid_models:
            print(f"Warning: Invalid models requested: {invalid_models}")
            models_to_plot = [m for m in models_to_plot if m in all_models]

        print(
            f"Will plot {len(domains_to_plot)} domains for {len(models_to_plot)} models"
        )

        # Create output directory structure
        model_dirs = create_output_structure(
            args.output_dir,
            args.experiment,
            args.version,
            models_to_plot,
            with_confidence=args.plot_confidence,
        )

        # Check if confidence plotting is requested but data unavailable
        has_confidence = "confidence" in df.columns
        if args.plot_confidence and not has_confidence:
            print(
                "Warning: Confidence plotting requested but no confidence column found in data"
            )
            print("Disabling confidence plotting...")
            plot_confidence = False
        else:
            plot_confidence = args.plot_confidence

        # Configure plot settings
        plot_config = {
            "overlay_by": args.overlay_by,
            "temperature_filter": args.temperature_filter,
            "plot_confidence": plot_confidence,
            "show_uncertainty": not args.no_uncertainty,
            "uncertainty_type": args.uncertainty_type,
            "uncertainty_level": args.uncertainty_level,
            "show_inference_groups": True,
            "category_order": {
                "prompt_category": [
                    "numeric",
                    "numeric-conf",
                    "CoT",
                ]
            },
        }

        # Generate plots for each domain
        for domain in domains_to_plot:
            generate_plots_for_domain(
                df=df,
                domain=domain,
                experiment_name=args.experiment,
                version=args.version,
                model_dirs=model_dirs,
                plot_config=plot_config,
            )

        # Determine output directory for summary
        output_path = args.output_dir / args.experiment
        if args.plot_confidence:
            output_path = output_path / "with_confidence"

        print(f"\n✓ Completed! Plots saved to: {output_path}")

        # Print summary
        total_plots = len(domains_to_plot) * len(models_to_plot)
        confidence_status = (
            "with confidence dots" if plot_confidence else "without confidence"
        )
        print(
            f"Generated {total_plots} plots ({len(domains_to_plot)} domains × {len(models_to_plot)} models) {confidence_status}"
        )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
