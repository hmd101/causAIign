#!/usr/bin/env python3
"""
Plot Pipeline Output - Example Script

This script demonstrates how to load and visualize data from the
data processing pipeline output files.

Usage:
    python src/causalign/analysis/visualization/plot_pipeline_output.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

try:
    from causalign.config.paths import PathManager
except ImportError:
    # Fallback: direct import if package not properly installed
    sys.path.insert(0, str(src_dir / "causalign"))
    from config.paths import PathManager

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


def load_processed_data(
    paths: PathManager,
    version: str = "6",
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
) -> pd.DataFrame:
    """
    Load processed data from the pipeline output

    Args:
        paths: PathManager instance
        version: Version string (e.g., "6")
        graph_type: Graph type (e.g., "collider")
        use_roman_numerals: Whether to load the Roman numerals version
        use_aggregated: Whether to load the aggregated human responses version
    """

    if use_roman_numerals and use_aggregated:
        # Load the Roman numerals version (has reasoning types and aggregated humans)
        data_path = (
            paths.rw17_processed_llm_dir
            / "cleaned_data_combined_subjects"
            / "cleaned_data_combined_subjects_reasoning_types"
            / f"{version}_v_{graph_type}_cleaned_data_roman.csv"
        )

    elif use_aggregated:
        # Load the aggregated version (balanced human sample sizes)
        data_path = (
            paths.rw17_processed_llm_dir
            / "cleaned_data_combined_subjects"
            / f"{version}_v_humans_avg_equal_sample_size_cogsci.csv"
        )

    else:
        # Load the main processed file
        data_path = (
            paths.rw17_processed_llm_dir
            / "cleaned_data_combined_subjects"
            / f"{version}_v_{graph_type}_cleaned_data.csv"
        )

    print(f"Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Try different separators (comma first, then semicolon)
    try:
        df = pd.read_csv(data_path, sep=",")
        print(f"Successfully loaded {len(df)} rows with comma separator")
    except Exception:
        try:
            df = pd.read_csv(data_path, sep=";")
            print(f"Successfully loaded {len(df)} rows with semicolon separator")
        except Exception as e:
            raise Exception(f"Failed to load data with both separators: {e}")

    return df


def create_simple_line_plot(
    df,
    output_dir=None,
    version="6",
    graph_type="collider",
    experiment_name="pilot_study",
):
    """Create a simple line plot by reasoning type"""

    # Standardize column names
    if "likelihood" in df.columns and "likelihood-rating" not in df.columns:
        df["likelihood-rating"] = df["likelihood"]

    # Subject colors for consistent plotting
    subject_colors = {
        "humans": "#E7298A",
        "gpt-3.5-turbo": "#6BAED6",
        "gpt-4o": "#08306B",
        "claude-3-opus": "#66C2A5",
        "gemini-1.5-pro": "#A6D854",
        "gemini-2.0-pro": "#FD8D3C",
    }

    # Filter to temperature 0.0 for LLMs (or NaN for humans)
    plot_df = df[
        (df["temperature"] == 0.0)
        | ((df["subject"] == "humans") & df["temperature"].isna())
    ].copy()

    if "reasoning_type" in df.columns:
        # Plot by reasoning type
        reasoning_types = plot_df["reasoning_type"].unique()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, reasoning_type in enumerate(reasoning_types[:4]):  # Max 4 plots
            if idx < len(axes):
                ax = axes[idx]
                subset = plot_df[plot_df["reasoning_type"] == reasoning_type]

                sns.lineplot(
                    data=subset,
                    x="task",
                    y="likelihood-rating",
                    hue="subject",
                    palette=subject_colors,
                    marker="o",
                    ax=ax,
                )

                ax.set_title(reasoning_type)
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
                    ax.set_ylabel("Likelihood Rating")
                if idx >= 2:
                    ax.set_xlabel("Task")

        # Hide unused subplots
        for idx in range(len(reasoning_types), len(axes)):
            axes[idx].set_visible(False)

    else:
        # Simple plot by task
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.lineplot(
            data=plot_df,
            x="task",
            y="likelihood-rating",
            hue="subject",
            palette=subject_colors,
            marker="o",
            ax=ax,
        )

        ax.set_title("Likelihood Ratings by Task")
        ax.set_ylabel("Likelihood Rating")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Add conditional probabilities to x-axis labels
        x_ticks = ax.get_xticks()
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        new_labels = [f"{l}\n{TASK_CONDITIONAL_PROB.get(l, '')}" for l in x_labels]
        ax.set_xticklabels(new_labels, rotation=45, ha="right")

    plt.tight_layout()

    if output_dir:
        # Create descriptive filename with version and experiment info
        if "reasoning_type" in df.columns:
            filename = f"pipeline_reasoning_types_v{version}_{experiment_name}_{graph_type}_temp0.0.pdf"
        else:
            filename = (
                f"pipeline_tasks_v{version}_{experiment_name}_{graph_type}_temp0.0.pdf"
            )
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    """Main function to demonstrate pipeline output loading and plotting"""

    print("🔍 Loading data from pipeline output...")

    # Initialize paths
    paths = PathManager()

    # Configuration
    version = "1"
    graph_type = "collider"

    try:
        # Try loading the Roman numerals version first (preferred)
        data = load_processed_data(
            paths,
            version=version,
            graph_type=graph_type,
            use_roman_numerals=True,
            use_aggregated=True,
        )
        print("✅ Loaded Roman numerals version with reasoning types")

    except FileNotFoundError:
        try:
            # Fallback to aggregated version
            data = load_processed_data(
                paths,
                version=version,
                graph_type=graph_type,
                use_roman_numerals=False,
                use_aggregated=True,
            )
            print("⚠️ Loaded aggregated version (no Roman numerals)")

        except FileNotFoundError:
            # Final fallback to main file
            data = load_processed_data(
                paths,
                version=version,
                graph_type=graph_type,
                use_roman_numerals=False,
                use_aggregated=False,
            )
            print("⚠️ Loaded main processed file (no aggregation, no Roman numerals)")

    # Display data info
    print("\n📊 Data Information:")
    print(f"Shape: {data.shape}")
    print(f"Subjects: {data['subject'].unique()}")
    print(f"Columns: {list(data.columns)}")

    if "reasoning_type" in data.columns:
        print(f"Reasoning types: {data['reasoning_type'].unique()}")

    if "task" in data.columns:
        print(f"Tasks: {sorted(data['task'].unique())}")

    if "domain" in data.columns:
        print(f"Domains: {data['domain'].unique()}")

    # Create plots
    print("\n📈 Creating plots...")
    output_dir = paths.base_dir / "results" / "line_plots"

    # Configuration for filenames
    experiment_name = "pilot_study"

    create_simple_line_plot(
        data,
        output_dir=output_dir,
        version=version,
        graph_type=graph_type,
        experiment_name=experiment_name,
    )

    # Show response counts
    print("\n📋 Response counts by subject:")
    if "reasoning_type" in data.columns:
        counts = (
            data.groupby(["subject", "reasoning_type"]).size().unstack(fill_value=0)
        )
        print(counts)
    else:
        counts = data["subject"].value_counts()
        print(counts)


if __name__ == "__main__":
    main()
