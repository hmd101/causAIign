from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    correlation_type: Literal["between_categories", "between_subjects"],
    output_dir: Optional[Union[str, Path]] = None,
    group_by: Optional[List[str]] = None,
    show: bool = True,
    title_prefix: Optional[str] = None,
    cmap: str = "RdYlBu_r",
    category_order: Optional[Dict[str, List[str]]] = None,
    category_column: Optional[str] = None,
    filename_suffix: Optional[str] = None,
    figsize_base: float = 0.8,
    figsize_min: float = 6.0,
    figsize_max: float = 20.0,
    auto_figsize: bool = True,
    **kwargs,
) -> None:
    """
    Create heatmap visualization of correlation results.

    Args:
        corr_df: DataFrame from compute_correlations()
        correlation_type: Type of correlation analysis performed
        output_dir: Directory to save plots
        group_by: Columns used for grouping in correlation analysis
        show: Whether to display plots
        title_prefix: Optional prefix for plot titles
        cmap: Colormap for heatmap
        category_order: Dict mapping column names to ordered list of categories
        category_column: Name of the column containing categories (required for between_categories)
        filename_suffix: Custom suffix to add to the filename (will be prefixed with underscore)
        figsize_base: Base size multiplier for adaptive figure sizing (default: 0.8)
        figsize_min: Minimum figure size (default: 6.0)
        figsize_max: Maximum figure size (default: 20.0)
        auto_figsize: Whether to automatically scale figure size based on matrix dimensions (default: True)
        **kwargs: Additional kwargs for sns.heatmap
    """
    # Get filter information from the correlation results
    filter_info = {}
    if "temperature" in corr_df.columns:
        filter_info["temperature"] = corr_df["temperature"].iloc[0]
    if "subjects" in corr_df.columns:
        filter_info["subjects"] = corr_df["subjects"].iloc[0]

    # Create filter info string for titles and filenames
    filter_str = ""
    if filter_info:
        filter_parts = []
        if "temperature" in filter_info:
            filter_parts.append(f"temp={filter_info['temperature']}")
        if "subjects" in filter_info:
            filter_parts.append(f"subj={filter_info['subjects']}")
        filter_str = f" ({', '.join(filter_parts)})"

    # Handle grouping with ordering
    if group_by:
        # Get unique combinations with ordered categories if specified
        unique_values = {}
        for col in group_by:
            if category_order and col in category_order:
                unique_values[col] = [
                    cat for cat in category_order[col] if cat in corr_df[col].unique()
                ]
            else:
                unique_values[col] = sorted(corr_df[col].unique())

        # Create all combinations of grouping values in the specified order
        from itertools import product

        combinations = list(product(*[unique_values[col] for col in group_by]))
        group_combinations = {}
        for combo in combinations:
            mask = pd.Series(True, index=corr_df.index)
            for col, val in zip(group_by, combo):
                mask &= corr_df[col] == val
            if mask.any():  # Only include combinations that exist in the data
                group_name = combo[0] if len(combo) == 1 else combo
                group_combinations[group_name] = corr_df[mask].index
    else:
        group_combinations = {"all": corr_df.index}

    def calculate_adaptive_figsize(matrix_size: int, base_size: float = 0.8) -> float:
        """Calculate adaptive figure size based on matrix dimensions"""
        if not auto_figsize:
            return 6.0  # Default fixed size

        # Scale based on matrix size with diminishing returns
        adaptive_size = base_size * (matrix_size + 2)  # +2 for axis labels

        # Apply constraints
        adaptive_size = max(figsize_min, min(figsize_max, adaptive_size))

        return adaptive_size

    # Set up subplot grid with adaptive sizing
    n_plots = len(group_combinations)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    # Estimate matrix size for adaptive sizing (will be refined per subplot)
    if correlation_type == "between_categories":
        if "category_1" in corr_df.columns:
            estimated_categories = set(
                list(corr_df["category_1"].unique())
                + list(corr_df["category_2"].unique())
            )
            estimated_matrix_size = len(estimated_categories)
        else:
            estimated_matrix_size = 10  # Fallback estimate
    else:  # between_subjects
        if "subject_1" in corr_df.columns:
            estimated_subjects = set(
                list(corr_df["subject_1"].unique())
                + list(corr_df["subject_2"].unique())
            )
            estimated_matrix_size = len(estimated_subjects)
        else:
            estimated_matrix_size = 6  # Fallback estimate

    # Calculate base figure size for the grid
    subplot_size = calculate_adaptive_figsize(estimated_matrix_size, figsize_base)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(subplot_size * n_cols, subplot_size * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    print(
        f"DEBUG: Matrix size estimate: {estimated_matrix_size}, subplot size: {subplot_size:.1f}, total figsize: ({subplot_size * n_cols:.1f}, {subplot_size * n_rows:.1f})"
    )

    for idx, (group_name, group_indices) in enumerate(group_combinations.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        group_df = corr_df.loc[group_indices].copy()

        if correlation_type == "between_categories":
            # Get categories in specified order if available
            if category_order and "category_1" in corr_df.columns:
                categories = [
                    cat
                    for cat in category_order.get(category_column, [])
                    if cat
                    in set(
                        list(group_df["category_1"].unique())
                        + list(group_df["category_2"].unique())
                    )
                ]
            else:
                categories = sorted(
                    set(
                        list(group_df["category_1"].unique())
                        + list(group_df["category_2"].unique())
                    )
                )

            corr_matrix = pd.DataFrame(
                np.eye(len(categories)), index=categories, columns=categories
            )

            # Fill matrix with correlations
            for _, row in group_df.iterrows():
                corr_matrix.loc[row["category_1"], row["category_2"]] = row[
                    "correlation"
                ]
                corr_matrix.loc[row["category_2"], row["category_1"]] = row[
                    "correlation"
                ]

            # Adaptive settings for large matrices
            matrix_size = len(categories)

            # Adjust annotation and font settings based on matrix size
            if matrix_size <= 10:
                annot = True
                fmt = ".2f"
                annot_kws = {"size": 10}
                cbar_kws = {"shrink": 0.8}
            elif matrix_size <= 20:
                annot = True
                fmt = ".1f"
                annot_kws = {"size": 8}
                cbar_kws = {"shrink": 0.6}
            else:
                annot = False  # No annotations for very large matrices
                fmt = ".2f"
                annot_kws = {"size": 6}
                cbar_kws = {"shrink": 0.5}

            # Allow kwargs to override adaptive settings
            heatmap_kwargs = {
                "annot": annot,
                "fmt": fmt,
                "annot_kws": annot_kws,
                "cbar_kws": cbar_kws,
                "cmap": cmap,
                "center": 0,
                "vmin": -1,
                "vmax": 1,
                "square": True,  # Make cells square for better readability
                **kwargs,  # User overrides
            }

            # Plot heatmap
            sns.heatmap(corr_matrix, ax=ax, **heatmap_kwargs)

            # Adjust tick labels for large matrices
            if matrix_size > 15:
                ax.tick_params(axis="both", which="major", labelsize=8)
                # Rotate labels if they're long
                max_label_length = max(len(str(cat)) for cat in categories)
                if max_label_length > 10:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            elif matrix_size > 10:
                ax.tick_params(axis="both", which="major", labelsize=9)
                # Rotate x labels if they're long
                max_label_length = max(len(str(cat)) for cat in categories)
                if max_label_length > 15:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            subtitle = f"Between Categories{filter_str}"

        else:  # between_subjects
            # Get subjects in specified order if available
            if category_order and "subject_1" in corr_df.columns:
                subjects = [
                    subj
                    for subj in category_order.get("subject", [])
                    if subj
                    in set(
                        list(group_df["subject_1"].unique())
                        + list(group_df["subject_2"].unique())
                    )
                ]
            else:
                subjects = sorted(
                    set(
                        list(group_df["subject_1"].unique())
                        + list(group_df["subject_2"].unique())
                    )
                )

            corr_matrix = pd.DataFrame(
                np.eye(len(subjects)), index=subjects, columns=subjects
            )

            # Fill matrix with correlations
            for _, row in group_df.iterrows():
                corr_matrix.loc[row["subject_1"], row["subject_2"]] = row["correlation"]
                corr_matrix.loc[row["subject_2"], row["subject_1"]] = row["correlation"]

            # Adaptive settings for large matrices
            matrix_size = len(subjects)

            # Adjust annotation and font settings based on matrix size
            if matrix_size <= 10:
                annot = True
                fmt = ".2f"
                annot_kws = {"size": 10}
                cbar_kws = {"shrink": 0.8}
            elif matrix_size <= 20:
                annot = True
                fmt = ".1f"
                annot_kws = {"size": 8}
                cbar_kws = {"shrink": 0.6}
            else:
                annot = False  # No annotations for very large matrices
                fmt = ".2f"
                annot_kws = {"size": 6}
                cbar_kws = {"shrink": 0.5}

            # Allow kwargs to override adaptive settings
            heatmap_kwargs = {
                "annot": annot,
                "fmt": fmt,
                "annot_kws": annot_kws,
                "cbar_kws": cbar_kws,
                "cmap": cmap,
                "center": 0,
                "vmin": -1,
                "vmax": 1,
                "square": True,  # Make cells square for better readability
                **kwargs,  # User overrides
            }

            # Plot heatmap
            sns.heatmap(corr_matrix, ax=ax, **heatmap_kwargs)

            # Adjust tick labels for large matrices
            if matrix_size > 15:
                ax.tick_params(axis="both", which="major", labelsize=8)
                # Rotate labels if they're long
                max_label_length = max(len(str(subj)) for subj in subjects)
                if max_label_length > 10:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            elif matrix_size > 10:
                ax.tick_params(axis="both", which="major", labelsize=9)
                # Rotate x labels if they're long
                max_label_length = max(len(str(subj)) for subj in subjects)
                if max_label_length > 15:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            subtitle = f"Between Subjects{filter_str}"

        # Set title
        if isinstance(group_name, tuple):
            group_title = ", ".join(
                [f"{col}: {val}" for col, val in zip(group_by, group_name)]
            )
        elif group_by:
            group_title = f"{group_by[0]}: {group_name}"
        else:
            group_title = ""

        if title_prefix:
            if group_title:
                ax.set_title(f"{title_prefix}\n{subtitle}\n{group_title}")
            else:
                ax.set_title(f"{title_prefix}\n{subtitle}")
        elif group_title:
            ax.set_title(f"{subtitle}\n{group_title}")
        else:
            ax.set_title(subtitle)

    # Hide unused axes
    for idx in range(len(group_combinations), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with filter information
        components = ["correlation_heatmap", correlation_type]
        if group_by:
            components.extend(group_by)

        # Add filter info to filename
        if filter_info:
            filter_filename_parts = []
            if "temperature" in filter_info:
                filter_filename_parts.append(f"temp{filter_info['temperature']}")
            if "subjects" in filter_info:
                filter_filename_parts.append(
                    str(filter_info["subjects"]).replace(" ", "-")
                )
            components.extend(filter_filename_parts)

        # Create base filename
        base_filename = "_".join(components)

        # Add custom suffix if provided
        if filename_suffix:
            if not filename_suffix.startswith("_"):
                filename_suffix = f"_{filename_suffix}"
            base_filename = f"{base_filename}{filename_suffix}"

        filename = f"{base_filename}.pdf"
        plt.savefig(output_dir / filename, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_dir / filename}")

    if show:
        plt.show()
    else:
        plt.close()
