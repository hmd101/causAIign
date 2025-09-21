#!/usr/bin/env python3
"""
Demo script showing how to use the facet line plot with category renaming.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.causalign.analysis.visualization.facet_lineplot import create_facet_line_plot
from src.causalign.config.paths import PathManager


def main():
    """Main function demonstrating plot creation with category renaming"""

    # Initialize paths
    paths = PathManager()

    # Load your data
    data_path = (
        paths.base_dir
        / "data"
        / "processed"
        / "llm_with_humans"
        / "rw17"
        / "abstract_reasoning"
        / "1_v_collider_cleaned_data.csv"
    )
    df = pd.read_csv(data_path)

    # Define category renaming
    rename_map = {"prompt_category": {"numeric_certainty": "numeric_confidence"}}

    # Create output directory
    output_dir = paths.base_dir / "results" / "plots" / "abstract_reasoning"

    # Create plot with faceting by prompt category
    create_facet_line_plot(
        df,
        facet_by="prompt_category",
        temperature_filter=0.0,
        title_prefix="Abstract Reasoning Results",
        output_dir=output_dir,
        rename_map=rename_map,
    )

    # Create plot with overlays
    create_facet_line_plot(
        df,
        overlay_by="prompt_category",
        temperature_filter=0.0,
        title_prefix="Abstract Reasoning Results (Overlaid)",
        output_dir=output_dir,
        rename_map=rename_map,
    )


if __name__ == "__main__":
    main()
