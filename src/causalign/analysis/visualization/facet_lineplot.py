from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def rename_categories(
    df: pd.DataFrame, column: str, mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Rename categories in a DataFrame column using a mapping dictionary.

    Args:
        df: Input DataFrame
        column: Column name to rename categories in
        mapping: Dictionary mapping old names to new names

    Returns:
        DataFrame with renamed categories
    """
    return df.assign(**{column: df[column].replace(mapping)})


def create_facet_line_plot(
    df: pd.DataFrame,
    facet_by: Optional[
        Union[str, List[str]]
    ] = None,  # e.g., 'prompt_category' or ['domain', 'temperature']
    overlay_by: Optional[
        str
    ] = None,  # Column to use for overlaying in same subplot with different styles
    group_subjects: bool = False,  # If True, all subjects will be plotted in the same subplot
    output_dir: Optional[Union[str, Path]] = None,
    x: str = "task",
    y: str = "likelihood-rating",
    temperature_filter: Optional[float] = None,  # e.g., 0.0
    subject_colors: Optional[Dict[str, str]] = None,
    overlay_styles: Optional[
        Dict[str, Dict[str, any]]
    ] = None,  # Style mappings for overlay categories
    title_prefix: Optional[str] = None,
    savefig_kwargs: Optional[dict] = None,
    show: bool = True,
    legend_kwargs: Optional[dict] = None,  # Additional kwargs for legend customization
    category_order: Optional[
        Dict[str, List[str]]
    ] = None,  # Ordering for categorical columns
    rename_map: Optional[
        Dict[str, Dict[str, str]]
    ] = None,  # Mapping for renaming categories
    show_inference_groups: bool = True,  # Whether to show inference group labels
    filename_suffix: Optional[
        str
    ] = None,  # Custom suffix to add to the filename (will be prefixed with underscore)
    plot_confidence: bool = False,  # Whether to plot confidence values as scatter dots
    confidence_column: str = "confidence",  # Column name for confidence values
    confidence_alpha: float = 0.3,  # Alpha transparency for confidence scatter dots
    confidence_jitter: float = 0.05,  # Amount of horizontal jitter for confidence dots
    subjects: Optional[
        List[str]
    ] = None,  # List of subjects to include (None = all subjects)
    show_uncertainty: bool = True,  # Whether to show uncertainty bands around lines
    uncertainty_type: str = "ci",  # Type of uncertainty: 'ci', 'se', 'sd', 'pi'
    uncertainty_level: float = 95,  # Confidence level for CI or percentile level for PI
    uncertainty_alpha: float = 0.2,  # Alpha transparency for uncertainty bands
    title_fontsize: Optional[int] = None,  # Custom font size for titles
    legend_position: str = "bottom",  # Legend position: 'right', 'bottom', 'top', 'left'
    **kwargs,
):
    """
    Create line plots with flexible faceting and overlay options.

    Args:
        df: DataFrame containing the data.
        facet_by: Column(s) to facet by (str or list of str).
        overlay_by: Column to use for overlaying in same subplot with different styles.
        group_subjects: If True, plot all subjects in same subplot.
        output_dir: Directory to save the plot (optional).
        x: Column for x-axis (default: 'task').
        y: Column for y-axis (default: 'likelihood-rating').
        temperature_filter: If set, filter LLMs to this temperature, but always include humans.
        subject_colors: Dict mapping subject names to colors.
        overlay_styles: Dict mapping overlay categories to style dicts.
        title_prefix: Optional prefix for plot titles.
        savefig_kwargs: Additional kwargs for plt.savefig.
        show: Whether to show the plot (default: True).
        legend_kwargs: Additional kwargs for legend customization.
        category_order: Dict mapping column names to ordered list of categories.
        rename_map: Dict mapping column names to {old_name: new_name} mappings.
        show_inference_groups: Whether to show inference group labels.
        filename_suffix: Custom suffix to add to the filename (will be prefixed with underscore).
        plot_confidence: Whether to plot confidence values as scatter dots on the same axis.
        confidence_column: Column name for confidence values (default: 'confidence').
        confidence_alpha: Alpha transparency for confidence scatter dots (default: 0.7).
        confidence_jitter: Amount of horizontal jitter for confidence dots to handle overlapping (default: 0.05).
        subjects: List of subjects to include (None = all subjects).
        show_uncertainty: Whether to show uncertainty bands around lines.
        uncertainty_type: Type of uncertainty: 'ci', 'se', 'sd', 'pi'.
        uncertainty_level: Confidence level for CI or percentile level for PI.
        uncertainty_alpha: Alpha transparency for uncertainty bands.
        title_fontsize: Custom font size for plot titles (default: matplotlib default).
        legend_position: Position of the legend: 'right' (default), 'bottom', 'top', 'left'.
        **kwargs: Additional arguments for seaborn.lineplot.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Set up uncertainty parameters for seaborn
    def get_errorbar_config():
        """Configure errorbar parameter for seaborn lineplot based on uncertainty settings"""
        if not show_uncertainty:
            return None
        elif uncertainty_type == "ci":
            return ("ci", uncertainty_level)
        elif uncertainty_type == "pi":
            return ("pi", uncertainty_level)
        elif uncertainty_type in ["se", "sd"]:
            return uncertainty_type
        else:
            raise ValueError(
                f"Unknown uncertainty_type: {uncertainty_type}. Must be one of: 'ci', 'se', 'sd', 'pi'"
            )

    errorbar_config = get_errorbar_config()

    # Filter to specific subjects if requested
    if subjects is not None:
        if "subject" not in df.columns:
            raise ValueError(
                "subjects parameter provided but 'subject' column not found in dataframe"
            )

        available_subjects = set(df["subject"].unique())
        requested_subjects = set(subjects)
        missing_subjects = requested_subjects - available_subjects

        if missing_subjects:
            raise ValueError(
                f"Subjects not found in dataframe: {missing_subjects}. Available subjects: {available_subjects}"
            )

        df = df[df["subject"].isin(subjects)].copy()
        print(f"Filtered to {len(subjects)} subjects: {subjects}")

    # Rename categories if specified
    if rename_map:
        for col, mapping in rename_map.items():
            if col in df.columns:
                df = rename_categories(df, col, mapping)

    # Standardize column names
    if "likelihood" in df.columns and "likelihood-rating" not in df.columns:
        df["likelihood-rating"] = df["likelihood"]

    # Filter by temperature, but always include humans
    if temperature_filter is not None and "temperature" in df.columns:
        df = df[
            (df["temperature"] == temperature_filter)
            | ((df["subject"] == "humans") & (df["temperature"].isna()))
        ]

    # Set up default subject colors if not provided
    if subject_colors is None:
        subject_colors = {
            "humans": "#E7298A",
            "gpt-3.5-turbo": "#6BAED6",
            "gpt-4o": "#08306B",
            "claude-3-opus": "#66C2A5",
            "claude-3-opus-20240229": "#66C2A5",
            "gemini-1.5-pro": "#A6D854",
            "gemini-2.0-pro": "#FD8D3C",
        }

    # Ensure Roman numerals are properly ordered
    if x == "task" and "task" in df.columns:
        roman_order = [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
        ]
        df["task"] = pd.Categorical(df["task"], categories=roman_order, ordered=True)

        # Add reasoning group information for line segmentation
        reasoning_groups = {
            "Predictive": ["I", "II", "III"],
            "Independence": ["IV", "V"],
            "Effect-Present": ["VI", "VII", "VIII"],
            "Effect-Absent": ["IX", "X", "XI"],
        }

        # Add group column for line segmentation
        df["reasoning_group"] = None
        for group, tasks in reasoning_groups.items():
            df.loc[df["task"].isin(tasks), "reasoning_group"] = group

        df = df.sort_values(["reasoning_group", "task"])

    # Set up default overlay styles if not provided
    if overlay_by and overlay_styles is None:
        # Get categories in specified order if available
        if category_order and overlay_by in category_order:
            unique_categories = [
                cat
                for cat in category_order[overlay_by]
                if cat in df[overlay_by].unique()
            ]
        else:
            unique_categories = sorted(df[overlay_by].unique())

        overlay_styles = {
            cat: {"linestyle": style, "alpha": 0.8, "marker": marker}
            for cat, (style, marker) in zip(
                unique_categories,
                [
                    ("-", "o"),
                    ("--", "s"),
                    (":", "^"),
                    ("-.", "D"),
                    ("-", "v"),
                    ("--", "p"),
                ],
            )
        }

    # Set up default legend kwargs if not provided
    if legend_kwargs is None:
        # Configure legend position based on legend_position parameter
        if legend_position == "right":
            legend_kwargs = {
                "bbox_to_anchor": (1.05, 1),
                "loc": "upper left",
                "borderaxespad": 0,
            }
        elif legend_position == "bottom":
            legend_kwargs = {
                "bbox_to_anchor": (0.5, -0.5),  # Even more space (was -0.375, now -0.5)
                "loc": "upper center",
                "borderaxespad": 0,
                "ncol": 1,  # Single column for bottom legend
            }
        elif legend_position == "top":
            legend_kwargs = {
                "bbox_to_anchor": (0.5, 1.15),
                "loc": "lower center",
                "borderaxespad": 0,
                "ncol": 1,  # Single column for top legend
            }
        elif legend_position == "left":
            legend_kwargs = {
                "bbox_to_anchor": (-0.05, 1),
                "loc": "upper right",
                "borderaxespad": 0,
            }
        else:
            # Default to right if invalid position specified
            legend_kwargs = {
                "bbox_to_anchor": (1.05, 1),
                "loc": "upper left",
                "borderaxespad": 0,
            }

    # Determine subplot structure based on faceting and grouping options
    if group_subjects:
        # Force single subplot when grouping subjects
        n_rows, n_cols = 1, 1
        fig, axes = plt.subplots(1, 1, figsize=(10, 6), squeeze=False)
        axes = axes.flatten()
        plot_combinations = [{}]  # Single empty dict for one plot
    elif facet_by is not None:
        # Multiple subplots based on faceting
        if isinstance(facet_by, str):
            facet_by = [facet_by]

        # Get unique combinations with ordered categories if specified
        unique_values = {}
        for col in facet_by:
            if category_order and col in category_order:
                unique_values[col] = [
                    cat for cat in category_order[col] if cat in df[col].unique()
                ]
            else:
                unique_values[col] = sorted(df[col].unique())

        # Create all combinations of faceting values in the specified order
        from itertools import product

        combinations = list(product(*[unique_values[col] for col in facet_by]))
        plot_combinations = [dict(zip(facet_by, combo)) for combo in combinations]

        n_plots = len(plot_combinations)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
        )
        axes = axes.flatten()
    else:
        # Single subplot, no faceting
        n_rows, n_cols = 1, 1
        fig, axes = plt.subplots(1, 1, figsize=(10, 6), squeeze=False)
        axes = axes.flatten()
        plot_combinations = [{}]

    # Create plots
    for idx, plot_params in enumerate(plot_combinations):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Apply faceting filters
        mask = np.ones(len(df), dtype=bool)
        for col, val in plot_params.items():
            # Handle NaN values in temperature column
            if col == "temperature" and pd.isna(val):
                mask &= df[col].isna()
            else:
                mask &= df[col] == val
        plot_df = df[mask]

        if overlay_by:
            # Get categories in specified order if available
            if category_order and overlay_by in category_order:
                categories = [
                    cat
                    for cat in category_order[overlay_by]
                    if cat in plot_df[overlay_by].unique()
                ]
            else:
                categories = sorted(plot_df[overlay_by].unique())

            # Plot each overlay category separately
            for category in categories:
                category_df = plot_df[plot_df[overlay_by] == category]

                # Get style for this category
                style = overlay_styles.get(category, {})

                # Plot each subject with the category style
                for subject in category_df["subject"].unique():
                    subject_df = category_df[category_df["subject"] == subject]

                    # Plot each reasoning group separately to break lines between groups
                    if "reasoning_group" in subject_df.columns:
                        for group in subject_df["reasoning_group"].unique():
                            group_df = subject_df[
                                subject_df["reasoning_group"] == group
                            ]

                            # Combine subject color with category style
                            plot_style = {
                                "color": subject_colors.get(subject, "#333333"),
                                "errorbar": errorbar_config,
                                **style,
                                **kwargs,
                            }

                            # Add uncertainty band styling if enabled
                            if show_uncertainty:
                                plot_style["err_kws"] = {"alpha": uncertainty_alpha}

                            sns.lineplot(
                                data=group_df,
                                x=x,
                                y=y,
                                ax=ax,
                                label=f"{subject} ({category})"
                                if group == subject_df["reasoning_group"].iloc[0]
                                else "_nolegend_",
                                **plot_style,
                            )

                            # Plot confidence as scatter dots if requested
                            if (
                                plot_confidence
                                and confidence_column in group_df.columns
                            ):
                                confidence_data = group_df[
                                    group_df[confidence_column].notna()
                                ]
                                if not confidence_data.empty:
                                    # Add horizontal jitter to prevent overlapping dots
                                    # Create consistent mapping for all possible x values
                                    if x == "task":
                                        # Use Roman numeral ordering for tasks
                                        roman_order = [
                                            "I",
                                            "II",
                                            "III",
                                            "IV",
                                            "V",
                                            "VI",
                                            "VII",
                                            "VIII",
                                            "IX",
                                            "X",
                                            "XI",
                                        ]
                                        x_map = {
                                            task: pos
                                            for pos, task in enumerate(roman_order)
                                        }
                                    else:
                                        # For other x columns, use all unique values in the full dataframe
                                        all_x_values = sorted(df[x].unique())
                                        x_map = {
                                            val: pos
                                            for pos, val in enumerate(all_x_values)
                                        }

                                    x_numeric = [
                                        x_map[task] for task in confidence_data[x]
                                    ]
                                    x_jittered = x_numeric + np.random.normal(
                                        0, confidence_jitter, len(x_numeric)
                                    )

                                    scatter_style = {
                                        "color": subject_colors.get(subject, "#333333"),
                                        "marker": style.get("marker", "o"),
                                        "alpha": confidence_alpha,
                                        "s": 30,  # scatter point size
                                        "zorder": 5,  # ensure dots appear on top
                                    }
                                    ax.scatter(
                                        x_jittered,
                                        confidence_data[confidence_column],
                                        **scatter_style,
                                        label="_nolegend_",  # Don't add to legend
                                    )
                    else:
                        # If no reasoning groups, plot normally
                        plot_style = {
                            "color": subject_colors.get(subject, "#333333"),
                            "errorbar": errorbar_config,
                            **style,
                            **kwargs,
                        }

                        # Add uncertainty band styling if enabled
                        if show_uncertainty:
                            plot_style["err_kws"] = {"alpha": uncertainty_alpha}

                        sns.lineplot(
                            data=subject_df,
                            x=x,
                            y=y,
                            ax=ax,
                            label=f"{subject} ({category})",
                            **plot_style,
                        )

                        # Plot confidence as scatter dots if requested
                        if plot_confidence and confidence_column in subject_df.columns:
                            confidence_data = subject_df[
                                subject_df[confidence_column].notna()
                            ]
                            if not confidence_data.empty:
                                # Add horizontal jitter to prevent overlapping dots
                                # Create consistent mapping for all possible x values
                                if x == "task":
                                    # Use Roman numeral ordering for tasks
                                    roman_order = [
                                        "I",
                                        "II",
                                        "III",
                                        "IV",
                                        "V",
                                        "VI",
                                        "VII",
                                        "VIII",
                                        "IX",
                                        "X",
                                        "XI",
                                    ]
                                    x_map = {
                                        task: pos
                                        for pos, task in enumerate(roman_order)
                                    }
                                else:
                                    # For other x columns, use all unique values in the full dataframe
                                    all_x_values = sorted(df[x].unique())
                                    x_map = {
                                        val: pos for pos, val in enumerate(all_x_values)
                                    }

                                x_numeric = [x_map[task] for task in confidence_data[x]]
                                x_jittered = x_numeric + np.random.normal(
                                    0, confidence_jitter, len(x_numeric)
                                )

                                scatter_style = {
                                    "color": subject_colors.get(subject, "#333333"),
                                    "marker": style.get("marker", "o"),
                                    "alpha": confidence_alpha,
                                    "s": 30,  # scatter point size
                                    "zorder": 5,  # ensure dots appear on top
                                }
                                ax.scatter(
                                    x_jittered,
                                    confidence_data[confidence_column],
                                    **scatter_style,
                                    label="_nolegend_",  # Don't add to legend
                                )
        else:
            # Plot each subject's reasoning groups separately
            if "reasoning_group" in plot_df.columns:
                for subject in plot_df["subject"].unique():
                    subject_df = plot_df[plot_df["subject"] == subject]

                    for group in subject_df["reasoning_group"].unique():
                        group_df = subject_df[subject_df["reasoning_group"] == group]

                        # Create plot style for this group
                        group_plot_style = {
                            "color": subject_colors.get(subject, "#333333"),
                            "errorbar": errorbar_config,
                            "marker": "o",
                            **kwargs,
                        }

                        # Add uncertainty band styling if enabled
                        if show_uncertainty:
                            group_plot_style["err_kws"] = {"alpha": uncertainty_alpha}

                        sns.lineplot(
                            data=group_df,
                            x=x,
                            y=y,
                            ax=ax,
                            label=subject
                            if group == subject_df["reasoning_group"].iloc[0]
                            else "_nolegend_",
                            **group_plot_style,
                        )

                        # Plot confidence as scatter dots if requested
                        if plot_confidence and confidence_column in group_df.columns:
                            confidence_data = group_df[
                                group_df[confidence_column].notna()
                            ]
                            if not confidence_data.empty:
                                # Add horizontal jitter to prevent overlapping dots
                                # Create consistent mapping for all possible x values
                                if x == "task":
                                    # Use Roman numeral ordering for tasks
                                    roman_order = [
                                        "I",
                                        "II",
                                        "III",
                                        "IV",
                                        "V",
                                        "VI",
                                        "VII",
                                        "VIII",
                                        "IX",
                                        "X",
                                        "XI",
                                    ]
                                    x_map = {
                                        task: pos
                                        for pos, task in enumerate(roman_order)
                                    }
                                else:
                                    # For other x columns, use all unique values in the full dataframe
                                    all_x_values = sorted(df[x].unique())
                                    x_map = {
                                        val: pos for pos, val in enumerate(all_x_values)
                                    }

                                x_numeric = [x_map[task] for task in confidence_data[x]]
                                x_jittered = x_numeric + np.random.normal(
                                    0, confidence_jitter, len(x_numeric)
                                )

                                ax.scatter(
                                    x_jittered,
                                    confidence_data[confidence_column],
                                    color=subject_colors.get(subject, "#333333"),
                                    marker="o",
                                    alpha=confidence_alpha,
                                    s=30,
                                    zorder=5,
                                    label="_nolegend_",
                                )
            else:
                # If no reasoning groups, plot normally
                final_plot_style = {
                    "hue": "subject",
                    "palette": subject_colors,
                    "marker": "o",
                    "errorbar": errorbar_config,
                    **kwargs,
                }

                # Add uncertainty band styling if enabled
                if show_uncertainty:
                    final_plot_style["err_kws"] = {"alpha": uncertainty_alpha}

                sns.lineplot(
                    data=plot_df,
                    x=x,
                    y=y,
                    ax=ax,
                    **final_plot_style,
                )

                # Plot confidence as scatter dots if requested
                if plot_confidence and confidence_column in plot_df.columns:
                    for subject in plot_df["subject"].unique():
                        subject_df = plot_df[plot_df["subject"] == subject]
                        confidence_data = subject_df[
                            subject_df[confidence_column].notna()
                        ]
                        if not confidence_data.empty:
                            # Add horizontal jitter to prevent overlapping dots
                            # Create consistent mapping for all possible x values
                            if x == "task":
                                # Use Roman numeral ordering for tasks
                                roman_order = [
                                    "I",
                                    "II",
                                    "III",
                                    "IV",
                                    "V",
                                    "VI",
                                    "VII",
                                    "VIII",
                                    "IX",
                                    "X",
                                    "XI",
                                ]
                                x_map = {
                                    task: pos for pos, task in enumerate(roman_order)
                                }
                            else:
                                # For other x columns, use all unique values in the full dataframe
                                all_x_values = sorted(df[x].unique())
                                x_map = {
                                    val: pos for pos, val in enumerate(all_x_values)
                                }

                            x_numeric = [x_map[task] for task in confidence_data[x]]
                            x_jittered = x_numeric + np.random.normal(
                                0, confidence_jitter, len(x_numeric)
                            )

                            ax.scatter(
                                x_jittered,
                                confidence_data[confidence_column],
                                color=subject_colors.get(subject, "#333333"),
                                marker="o",
                                alpha=confidence_alpha,
                                s=30,
                                zorder=5,
                                label="_nolegend_",
                            )

        # Set title and axes labels
        if plot_params:
            # Format title differently for temperature
            if "temperature" in plot_params:
                temp_val = plot_params["temperature"]
                if pd.isna(temp_val):
                    subtitle = "No Temperature (humans)"
                else:
                    subtitle = f"Temperature: {temp_val}"
            else:
                subtitle = ", ".join([f"{val}" for val in plot_params.values()])

            title_kwargs = {'pad': 10}
            if title_fontsize:
                title_kwargs['fontsize'] = title_fontsize
                
            if title_prefix:
                # Create two-line title with prefix on top
                ax.set_title(f"{title_prefix}\n{subtitle}", **title_kwargs)
            else:
                ax.set_title(subtitle, **title_kwargs)
        elif title_prefix:
            title_kwargs = {}
            if title_fontsize:
                title_kwargs['fontsize'] = title_fontsize
            ax.set_title(title_prefix, **title_kwargs)

        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Set axis labels only where needed
        if idx == 0 or idx % n_cols == 0:
            ax.set_ylabel(y)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel(x)

        # Handle legend
        if idx == 0:
            ax.legend(**legend_kwargs)
        else:
            # Only try to remove legend if it exists
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # After creating the main plot in the loop over axes
        if (
            show_inference_groups
            and x == "task"
            and "reasoning_group" in plot_df.columns
        ):
            # Define group spans and labels
            group_info = {
                "Predictive": {
                    "tasks": ["I", "II", "III"],
                    "label": "Predictive\nInference",
                    "color": "#E3F2FD",  # Light blue
                },
                "Independence": {
                    "tasks": ["IV", "V"],
                    "label": "Conditional\nIndependence",
                    "color": "#F3E5F5",  # Light purple
                },
                "Effect-Present": {
                    "tasks": ["VI", "VII", "VIII"],
                    "label": "Effect-Present\nDiagnostic",
                    "color": "#E8F5E9",  # Light green
                },
                "Effect-Absent": {
                    "tasks": ["IX", "X", "XI"],
                    "label": "Effect-Absent\nDiagnostic",
                    "color": "#FFF3E0",  # Light orange
                },
            }

            # Get axis limits
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min

            # Add background regions and labels for each group
            for group, info in group_info.items():
                # Find the x-coordinates for the group's tasks
                group_tasks = info["tasks"]

                # Add subtle background region
                ax.axvspan(
                    group_tasks[0],
                    group_tasks[-1],
                    alpha=0.15,
                    color=info["color"],
                    zorder=0,  # Ensure it's behind the data
                )

                # Add text label above the plot
                mid_task = group_tasks[
                    len(group_tasks) // 2
                ]  # Middle task for label placement
                ax.text(
                    mid_task,  # x position (middle of group)
                    y_max + y_range * 0.05,  # y position (slightly above plot)
                    info["label"],
                    ha="center",  # horizontally centered
                    va="bottom",  # vertically aligned to bottom
                    fontsize=8,
                    fontweight="bold",
                    color="#666666",
                )

            # Adjust the plot layout to make room for labels
            ax.set_ylim(y_min, y_max + y_range * 0.15)

            # Add conditional probabilities to x-axis labels
            x_ticks = ax.get_xticks()
            x_labels = [t.get_text() for t in ax.get_xticklabels()]
            new_labels = [f"{l}\n{TASK_CONDITIONAL_PROB.get(l, '')}" for l in x_labels]
            ax.set_xticklabels(new_labels, rotation=45, ha="right")

    # Hide unused axes
    for idx in range(len(plot_combinations), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save plot if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create descriptive filename
        components = []
        if facet_by:
            components.extend(facet_by if isinstance(facet_by, list) else [facet_by])
        if overlay_by:
            components.append(f"overlay_{overlay_by}")
        if group_subjects:
            components.append("grouped")

        # Add subjects to filename if specific subjects were selected
        if subjects is not None:
            if len(subjects) == 1:
                components.append(f"subject_{subjects[0]}")
            else:
                # Create abbreviated subject list for filename
                subject_str = "_".join(sorted(subjects))
                components.append(f"subjects_{subject_str}")

        # Base filename
        base_filename = (
            f"facet_lineplot{'_' + '_'.join(components) if components else ''}"
        )

        # Add custom suffix if provided
        if filename_suffix:
            if not filename_suffix.startswith("_"):
                filename_suffix = f"_{filename_suffix}"
            base_filename = f"{base_filename}{filename_suffix}"

        filename = f"{base_filename}.pdf"

        save_kwargs = dict(format="pdf", dpi=300, bbox_inches="tight")
        if savefig_kwargs:
            save_kwargs.update(savefig_kwargs)

        plt.savefig(output_dir / filename, **save_kwargs)
        print(f"Plot saved to: {output_dir / filename}")

    if show:
        plt.show()
    else:
        plt.close()
