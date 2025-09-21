from typing import List, Literal, Optional, Union

import pandas as pd
from scipy import stats


def compute_model_domain_correlations(
    df: pd.DataFrame,
    y: str = "likelihood-rating",
    method: str = "spearman",
    temperature_filter: Optional[float] = None,
    group_by: Optional[List[str]] = None,
    pool_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to compute correlations between model-domain pairs.

    This creates categories like "gpt-4o_economy", "claude-3-opus_weather", etc.
    and computes correlations between all pairs.

    Args:
        df: Input DataFrame containing the data
        y: Column containing values to correlate
        method: Correlation method ("spearman" or "pearson")
        temperature_filter: Optional temperature value to filter by
        group_by: List of columns to group by before computing correlations
        pool_columns: List of columns to pool/aggregate over

    Returns:
        DataFrame with correlation results between model-domain pairs
    """
    return compute_correlations(
        df=df,
        correlation_type="between_categories",
        category_column=["subject", "domain"],
        y=y,
        method=method,
        temperature_filter=temperature_filter,
        group_by=group_by,
        pool_columns=pool_columns,
    )


def compute_correlations(
    df: pd.DataFrame,
    correlation_type: Literal["between_categories", "between_subjects"],
    category_column: Optional[Union[str, List[str]]] = None,
    subjects: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
    pool_columns: Optional[List[str]] = None,
    y: str = "likelihood",
    method: str = "spearman",
    temperature_filter: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute correlations between categories or subjects.

    Args:
        df: Input DataFrame containing the data
        correlation_type: Type of correlation to compute ("between_categories" or "between_subjects")
        category_column: Name of the column containing categories (required for between_categories)
                        Can be a string for single column or list of strings for combined categories
                        Example: ["subject", "domain"] creates subject-domain pair categories
        subjects: List of subjects to include in analysis (optional filter)
        group_by: List of columns to group by before computing correlations
        pool_columns: List of columns to pool/aggregate over
        y: Name of the column containing values to correlate
        method: Correlation method ("spearman" or "pearson")
        temperature_filter: Optional temperature value to filter by

    Returns:
        DataFrame containing correlation results with columns:
        - category_1/subject_1: First category/subject
        - category_2/subject_2: Second category/subject
        - correlation: Correlation coefficient
        - p_value: Statistical significance
        - n_samples: Number of samples used
        - Additional columns for filters and grouping information
    """
    df = df.copy()

    print(f"\nDEBUG: Initial data shape: {df.shape}")

    # Store filter information
    filter_info = {}

    # Filter by temperature
    if temperature_filter is not None:
        df = df[df["temperature"] == temperature_filter]
        filter_info["temperature"] = temperature_filter
        print(f"DEBUG: After temperature filter shape: {df.shape}")

    # Filter by subjects if specified
    if subjects:
        df = df[df["subject"].isin(subjects)]
        filter_info["subjects"] = subjects if len(subjects) > 1 else subjects[0]
        print(f"DEBUG: After subject filter shape: {df.shape}")

    # Handle multi-column categories
    if isinstance(category_column, list):
        # Create combined category column
        combined_category_name = "_".join(category_column)
        df[combined_category_name] = df[category_column].apply(
            lambda row: "_".join(row.astype(str)), axis=1
        )
        category_col_to_use = combined_category_name
        print(
            f"DEBUG: Created combined category column '{combined_category_name}' from {category_column}"
        )
        print(
            f"DEBUG: Sample combined categories: {df[combined_category_name].unique()[:5]}"
        )
    else:
        category_col_to_use = category_column

    results = []

    # Handle grouping
    groups = [df] if not group_by else [group for _, group in df.groupby(group_by)]
    print(f"DEBUG: Number of groups: {len(groups)}")

    for group_idx, group_df in enumerate(groups):
        print(f"\nDEBUG: Processing group {group_idx + 1}/{len(groups)}")
        print(f"DEBUG: Group shape: {group_df.shape}")

        # For each category pair
        categories = sorted(group_df[category_col_to_use].unique())
        print(f"DEBUG: Categories found: {categories}")

        for i, cat1 in enumerate(categories):
            for cat2 in categories[i + 1 :]:
                print(f"\nDEBUG: Processing category pair: {cat1} vs {cat2}")
                # Get data for each category
                cat1_data = group_df[group_df[category_col_to_use] == cat1]
                cat2_data = group_df[group_df[category_col_to_use] == cat2]

                print(f"DEBUG: Category 1 shape: {cat1_data.shape}")
                print(f"DEBUG: Category 2 shape: {cat2_data.shape}")

                # If pooling, aggregate by remaining columns
                if pool_columns:
                    # Only keep task for correlation if it's not in pool_columns
                    essential_cols = ["task"] if "task" not in pool_columns else []

                    # Add any group_by columns if they exist
                    if group_by:
                        essential_cols.extend(
                            [col for col in group_by if col not in pool_columns]
                        )

                    print(f"DEBUG: Essential columns for pooling: {essential_cols}")

                    cat1_data = (
                        cat1_data.groupby(essential_cols)[y].mean().reset_index()
                    )
                    cat2_data = (
                        cat2_data.groupby(essential_cols)[y].mean().reset_index()
                    )

                    print(
                        f"DEBUG: After pooling shapes - Cat1: {cat1_data.shape}, Cat2: {cat2_data.shape}"
                    )

                # Merge on common columns
                merge_cols = [col for col in cat1_data.columns if col != y]
                merged = pd.merge(
                    cat1_data, cat2_data, on=merge_cols, suffixes=("_1", "_2")
                )

                print(f"DEBUG: Merged shape: {merged.shape}")
                if len(merged) > 0:
                    print(f"DEBUG: Sample of merged data:\n{merged.head()}")

                if len(merged) > 1:  # Need at least 2 points for correlation
                    if method == "spearman":
                        corr, p_value = stats.spearmanr(
                            merged[f"{y}_1"], merged[f"{y}_2"]
                        )
                    else:
                        corr, p_value = stats.pearsonr(
                            merged[f"{y}_1"], merged[f"{y}_2"]
                        )

                    result = {
                        "category_1": cat1,
                        "category_2": cat2,
                        "correlation": corr,
                        "p_value": p_value,
                        "n_samples": len(merged),
                        **filter_info,  # Add filter information to results
                    }

                    # Add grouping information
                    if group_by:
                        for col in group_by:
                            result[col] = group_df[col].iloc[0]

                    results.append(result)
                    print(f"DEBUG: Added result: {result}")
                else:
                    print(
                        "DEBUG: Not enough data points after merging to compute correlation"
                    )

    print(f"\nDEBUG: Total results collected: {len(results)}")
    return pd.DataFrame(results)
