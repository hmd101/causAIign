#!/usr/bin/env python3
"""
Plot Results - Pipeline Output Visualization

Command-line interface for creating plots from data processing pipeline output.
Supports multiple plot types, experiments, and configuration options.

Usage:
    python scripts/plot_results.py --version 8 --experiment pilot_study
    python scripts/plot_results.py --version 8 --experiment pilot_study --graph-type collider --output-dir custom/plots/
    python scripts/plot_results.py --list-available
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.causalign.config.paths import PathManager

# Mapping of tasks to their conditional probabilities
TASK_CONDITIONAL_PROB = {
    "VI": "p(Ci=1|E=1, Cj=1)",  # a
    "VII": "p(Ci=1|E=1)",  # b
    "VIII": "p(Ci=1|E=1, Cj=0)",  # c
    "IV": "p(Ci=1|Cj=1)",  # d
    "V": "p(Ci=1|Cj=0)",  # e
    "IX": "p(Ci=1|E=0, Cj=1)",  # f
    "X": "p(Ci=1|E=0)",  # g
    "XI": "p(Ci=1|E=0, Cj=0)",  # h
    "I": "p(E=1|Ci=0, Cj=0)",  # i
    "II": "p(E=1|Ci=0, Cj=1)",  # j
    "III": "p(E=1|Ci=1, Cj=1)",  # k
}

# Define reasoning groups for line segmentation
REASONING_GROUPS = {
    "Predictive": ["I", "II", "III"],
    "Independence": ["IV", "V"],
    "Effect-Present": ["VI", "VII", "VIII"],
    "Effect-Absent": ["IX", "X", "XI"],
}

# Define Roman numeral order
ROMAN_ORDER = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_processed_data(
    paths: PathManager,
    version: str,
    experiment_name: str = "pilot_study",
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
    pipeline_mode: str = "llm_with_humans",
    input_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load processed data from the pipeline output or a specific file

    Args:
        paths: PathManager instance
        version: Version string (e.g., "8")
        experiment_name: Experiment name (e.g., "pilot_study")
        graph_type: Graph type (e.g., "collider")
        use_roman_numerals: Whether to load the Roman numerals version
        use_aggregated: Whether to load the aggregated human responses version
        pipeline_mode: Pipeline mode ("llm_with_humans", "llm", "humans")
        input_file: Optional path to directly load a specific CSV file
    """
    logger = logging.getLogger(__name__)

    if input_file:
        # Use directly specified file
        data_path = Path(input_file)
        logger.info(f"\nLoading data from specified file:\n{data_path.absolute()}")
    else:
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
        experiment_dir = processed_base / "llm_with_humans" / "rw17" / experiment_name

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
            data_path = experiment_dir / f"{version_str}{graph_type}_cleaned_data.csv"

    logger.info(f"\nResolved data path:\n{data_path.absolute()}")

    if not data_path.exists():
        available_files = list(data_path.parent.glob("*.csv"))
        error_msg = f"\nData file not found: {data_path}\n"
        if available_files:
            error_msg += "\nAvailable files in the same directory:"
            for file in available_files:
                error_msg += f"\n- {file.name}"
        else:
            error_msg += f"\nNo CSV files found in: {data_path.parent}"
        raise FileNotFoundError(error_msg)

    # Load the data
    data = pd.read_csv(data_path)
    logger.info(f"Successfully loaded {len(data)} rows")

    return data


def create_line_plot(
    df: pd.DataFrame,
    output_dir: Path = None,
    version: str = "8",
    graph_type: str = "collider",
    experiment_name: str = "pilot_study",
    plot_type: str = "reasoning",
    temperature_filter: float = 0.0,
) -> None:
    """Create line plots from processed data"""
    logger = logging.getLogger(__name__)

    # Standardize column names
    if "likelihood" in df.columns and "likelihood-rating" not in df.columns:
        df["likelihood-rating"] = df["likelihood"]

    # Subject colors for consistent plotting
    subject_colors = {
        "humans": "#E7298A",
        "gpt-3.5-turbo": "#6BAED6",
        "gpt-4o": "#08306B",
        "claude-3-opus": "#66C2A5",
        "claude-3-opus-20240229": "#66C2A5",
        "claude-3-sonnet-20240229": "#A6CEE3",
        "claude-3-haiku-20240307": "#B2DF8A",
        "gemini-1.5-pro": "#A6D854",
        "gemini-1.5-pro copy": "#A6D854",  # Handle copy directory name
        "gemini-2.0-pro": "#FD8D3C",
    }

    # Filter to specified temperature for LLMs (or NaN for humans)
    plot_df = df[
        (df["temperature"] == temperature_filter)
        | ((df["subject"] == "humans") & df["temperature"].isna())
    ].copy()

    logger.info(f"Filtered to {len(plot_df)} rows (temperature={temperature_filter})")

    # Ensure Roman numerals are properly ordered
    if "task" in plot_df.columns:
        plot_df["task"] = pd.Categorical(
            plot_df["task"], categories=ROMAN_ORDER, ordered=True
        )

        # Add reasoning group information for line segmentation
        plot_df["reasoning_group"] = None
        for group, tasks in REASONING_GROUPS.items():
            plot_df.loc[plot_df["task"].isin(tasks), "reasoning_group"] = group

        plot_df = plot_df.sort_values(["reasoning_group", "task"])

    if plot_type == "reasoning" and "reasoning_type" in df.columns:
        # Plot by reasoning type
        reasoning_types = sorted(plot_df["reasoning_type"].unique())
        logger.info(f"Creating reasoning type plots for: {reasoning_types}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, reasoning_type in enumerate(reasoning_types[:4]):  # Max 4 plots
            if idx < len(axes):
                ax = axes[idx]
                subset = plot_df[plot_df["reasoning_type"] == reasoning_type]

                # Plot each subject's reasoning groups separately
                for subject in subset["subject"].unique():
                    subject_df = subset[subset["subject"] == subject]

                    for group in subject_df["reasoning_group"].unique():
                        group_df = subject_df[subject_df["reasoning_group"] == group]

                sns.lineplot(
                            data=group_df,
                    x="task",
                    y="likelihood-rating",
                            color=subject_colors.get(subject, "#333333"),
                            ax=ax,
                            label=subject
                            if group == subject_df["reasoning_group"].iloc[0]
                            else "_nolegend_",
                    marker="o",
                    linewidth=2,
                    markersize=6,
                )

                ax.set_title(f"{reasoning_type}", fontsize=12, fontweight="bold")
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)

                # Add conditional probabilities to x-axis labels
                x_ticks = ax.get_xticks()
                x_labels = [t.get_text() for t in ax.get_xticklabels()]
                new_labels = [
                    f"{l}\n{TASK_CONDITIONAL_PROB.get(l, '')}" for l in x_labels
                ]
                ax.set_xticklabels(new_labels, rotation=45, ha="right")

                if idx == 0:
                    ax.set_ylabel("Likelihood Rating", fontsize=11)
                else:
                    ax.set_ylabel("")

                if idx >= 2:
                    ax.set_xlabel("Task", fontsize=11)
                else:
                    ax.set_xlabel("")

                # Adjust legend
                if idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                else:
                    ax.get_legend().remove()

        # Hide unused subplots
        for idx in range(len(reasoning_types), len(axes)):
            axes[idx].set_visible(False)

        plot_title = "Likelihood Ratings by Reasoning Type"
        filename_suffix = "reasoning_types"

    else:
        # Simple plot by task
        logger.info("Creating simple task plot")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each subject's reasoning groups separately
        for subject in plot_df["subject"].unique():
            subject_df = plot_df[plot_df["subject"] == subject]

            for group in subject_df["reasoning_group"].unique():
                group_df = subject_df[subject_df["reasoning_group"] == group]

        sns.lineplot(
                    data=group_df,
            x="task",
            y="likelihood-rating",
                    color=subject_colors.get(subject, "#333333"),
                    ax=ax,
                    label=subject
                    if group == subject_df["reasoning_group"].iloc[0]
                    else "_nolegend_",
            marker="o",
            linewidth=2,
            markersize=6,
        )

        ax.set_title("Likelihood Ratings by Task", fontsize=14, fontweight="bold")
        ax.set_ylabel("Likelihood Rating", fontsize=12)
        ax.set_xlabel("Task", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Add conditional probabilities to x-axis labels
        x_ticks = ax.get_xticks()
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        new_labels = [f"{l}\n{TASK_CONDITIONAL_PROB.get(l, '')}" for l in x_labels]
        ax.set_xticklabels(new_labels, rotation=45, ha="right")

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plot_title = "Likelihood Ratings by Task"
        filename_suffix = "tasks"

    plt.suptitle(
        f"{plot_title} (v{version}, {experiment_name}, temp={temperature_filter})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if output_dir:
        # Create descriptive filename
        filename = f"v{version}_{experiment_name}_{graph_type}_{filename_suffix}_temp{temperature_filter}.pdf"
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to: {output_path}")

        # Also save as PNG for quick viewing
        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
        logger.info(f"Plot also saved as: {png_path}")
    else:
        logger.info("No output directory specified, displaying plot only")

    plt.show()


def list_available_data(paths: PathManager) -> None:
    """List all available processed data files in new directory structure"""
    logger = logging.getLogger(__name__)

    processed_base = paths.base_dir / "data" / "processed"

    if not processed_base.exists():
        logger.warning(f"Processed data directory not found: {processed_base}")
        return

    print("📊 Available Processed Data Files:")
    print("=" * 60)

    # Check each pipeline mode directory
    modes = {
        "humans": "Human-only Data",
        "llm": "LLM-only Data",
        "llm_with_humans": "Combined LLM+Human Data",
    }

    found_any = False

    for mode_dir, mode_name in modes.items():
        mode_path = processed_base / mode_dir / "rw17"

        if not mode_path.exists():
            continue

        print(f"\n📁 {mode_name} ({mode_dir})")
        print("-" * 40)

        if mode_dir == "humans":
            # Human data - direct files in rw17/
            csv_files = list(mode_path.glob("*.csv"))
            if csv_files:
                for file in sorted(csv_files):
                    print(f"    - {file.name}")
                found_any = True
            else:
                print("    ❌ No data files found")
        else:
            # LLM and combined data - experiment directories
            experiment_dirs = [d for d in mode_path.iterdir() if d.is_dir()]

            if not experiment_dirs:
                print("    ❌ No experiment directories found")
                continue

            for exp_dir in sorted(experiment_dirs):
                print(f"  📂 Experiment: {exp_dir.name}")

                # Find CSV files in experiment directory
                csv_files = list(exp_dir.glob("*.csv"))
                reasoning_files = (
                    list((exp_dir / "reasoning_types").glob("*.csv"))
                    if (exp_dir / "reasoning_types").exists()
                    else []
                )

                if csv_files:
                    print("    🔧 Main Files:")
                    for file in sorted(csv_files):
                        print(f"      - {file.name}")

                if reasoning_files:
                    print("    🔤 Reasoning Types Files:")
                    for file in sorted(reasoning_files):
                        print(f"      - {file.name}")

                if not csv_files and not reasoning_files:
                    print("    ❌ No data files found")
                else:
                    found_any = True

    if not found_any:
        print("\n❌ No processed data files found")
        print("💡 Run the data pipeline first: python scripts/run_data_pipeline.py")
    else:
        print(f"\n📁 Base directory: {processed_base}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create plots from data processing pipeline output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic plotting with version and experiment
  python scripts/plot_results.py --version 8 --experiment pilot_study

  # Plot with custom output directory
  python scripts/plot_results.py --version 8 --experiment pilot_study --output-dir results/plots/

  # Plot specific graph type and temperature
  python scripts/plot_results.py --version 8 --experiment pilot_study --graph-type collider --temperature 0.7

  # Use non-aggregated human data
  python scripts/plot_results.py --version 8 --experiment pilot_study --no-aggregated

  # Simple task plot instead of reasoning types
  python scripts/plot_results.py --version 8 --experiment pilot_study --plot-type task

  # Load data directly from a specific file
  python scripts/plot_results.py --input-file path/to/your/data.csv

  # List all available data files
  python scripts/plot_results.py --list-available
        """,
    )

    parser.add_argument("--version", "-v", help="Version number to plot (e.g., '8')")

    parser.add_argument(
        "--experiment",
        "-e",
        default="pilot_study",
        help="Experiment name (default: pilot_study)",
    )

    parser.add_argument(
        "--input-file",
        help="Path to directly load a specific CSV file (bypasses other data loading options)",
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
        "--plot-type",
        choices=["reasoning", "task"],
        default="reasoning",
        help="Plot type: reasoning types or simple tasks (default: reasoning)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature filter for LLM responses (default: 0.0)",
    )

    parser.add_argument(
        "--no-roman-numerals",
        action="store_true",
        help="Don't use Roman numerals version (skip reasoning types)",
    )

    parser.add_argument(
        "--no-aggregated",
        action="store_true",
        help="Don't use aggregated human responses",
    )

    parser.add_argument(
        "--output-dir", help="Output directory for plots (default: results/plots/)"
    )

    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List all available processed data files",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize paths
        paths = PathManager()

        if args.list_available:
            list_available_data(paths)
            return

        if not args.version and not args.input_file:
            parser.error(
                "Either --version or --input-file is required (or use --list-available)"
            )

        # Set output directory with experiment-based organization
        if args.output_dir:
            output_dir = Path(args.output_dir) / args.experiment
        else:
            output_dir = paths.base_dir / "results" / "plots" / args.experiment

        logger.info(
            f"🔍 Loading data for version {args.version}, experiment {args.experiment}"
        )

        # Load data with fallback strategy
        data = None
        use_roman_numerals = not args.no_roman_numerals
        use_aggregated = not args.no_aggregated

        try:
            # Try preferred version first
            data = load_processed_data(
                paths,
                version=args.version,
                experiment_name=args.experiment,
                graph_type=args.graph_type,
                use_roman_numerals=use_roman_numerals,
                use_aggregated=use_aggregated,
                pipeline_mode=args.pipeline_mode,
                input_file=args.input_file,
            )
            logger.info("✅ Loaded preferred data version")

        except FileNotFoundError:
            logger.warning("Preferred data file not found, trying fallbacks...")

            # Try without Roman numerals
            if use_roman_numerals:
                try:
                    data = load_processed_data(
                        paths,
                        version=args.version,
                        experiment_name=args.experiment,
                        graph_type=args.graph_type,
                        use_roman_numerals=False,
                        use_aggregated=use_aggregated,
                        pipeline_mode=args.pipeline_mode,
                        input_file=args.input_file,
                    )
                    logger.info("⚠️ Loaded data without Roman numerals")
                    use_roman_numerals = False
                except FileNotFoundError:
                    pass

            # Try without aggregation
            if data is None and use_aggregated:
                try:
                    data = load_processed_data(
                        paths,
                        version=args.version,
                        experiment_name=args.experiment,
                        graph_type=args.graph_type,
                        use_roman_numerals=False,
                        use_aggregated=False,
                        pipeline_mode=args.pipeline_mode,
                        input_file=args.input_file,
                    )
                    logger.info("⚠️ Loaded data without aggregation or Roman numerals")
                    use_roman_numerals = False
                    use_aggregated = False
                except FileNotFoundError:
                    pass

        if data is None:
            logger.error(f"❌ No data found for version {args.version}")
            logger.error("💡 Check available files with: --list-available")
            logger.error(
                "💡 Run data pipeline first: python scripts/run_data_pipeline.py"
            )
            sys.exit(1)

        # Display data info
        logger.info(f"📊 Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        logger.info(f"Subjects: {list(data['subject'].unique())}")

        if "reasoning_type" in data.columns:
            logger.info(f"Reasoning types: {list(data['reasoning_type'].unique())}")
        if "task" in data.columns:
            logger.info(f"Tasks: {sorted(data['task'].unique())}")
        if "domain" in data.columns:
            logger.info(f"Domains: {list(data['domain'].unique())}")

        # Create plots
        logger.info(f"📈 Creating {args.plot_type} plots...")

        create_line_plot(
            data,
            output_dir=output_dir,
            version=args.version,
            graph_type=args.graph_type,
            experiment_name=args.experiment,
            plot_type=args.plot_type,
            temperature_filter=args.temperature,
        )

        # Show response counts
        logger.info("📋 Response counts by subject:")
        if "reasoning_type" in data.columns and args.plot_type == "reasoning":
            counts = (
                data.groupby(["subject", "reasoning_type"]).size().unstack(fill_value=0)
            )
            print(counts)
        else:
            counts = data["subject"].value_counts()
            print(counts)

        logger.info("✅ Plotting completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.verbose:
            import traceback

            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
