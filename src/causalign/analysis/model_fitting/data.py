"""
Data loading and filtering utilities for model fitting.

This mirrors the interface used by scripts/plot_results.py so that users can
specify version/experiment/graph_type/pipeline_mode and optional filters
such as agents, domains, temperature, reasoning types, and tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import logging
import pandas as pd

from ...config.paths import PathManager


logger = logging.getLogger(__name__)


def _normalize_response_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is a numeric response column in [0,1] named 'response'.

    - If 'likelihood-rating' exists, divide by 100 to map to [0,1].
    - Else if 'likelihood' exists, assume already scaled in [0,1] unless >1
      then divide by 100.
    - Raises ValueError if no recognizable response column is present.
    """
    df = df.copy()
    if "likelihood-rating" in df.columns:
        df["response"] = df["likelihood-rating"].astype(float) / 100.0
        return df
    if "likelihood" in df.columns:
        vals = df["likelihood"].astype(float)
        if (vals > 1).any():
            vals = vals / 100.0
        df["response"] = vals
        return df
    raise ValueError("No response column found. Expected 'likelihood-rating' or 'likelihood'.")


def _filter_by_values(df: pd.DataFrame, column: str, allowed: Optional[Sequence[str]]) -> pd.DataFrame:
    """Filter DataFrame by allowed values in a column if provided."""
    if allowed is None:
        return df
    allowed_set = set(allowed)
    return df[df[column].isin(allowed_set)].copy()


def _coerce_iterable(arg: Optional[str | Iterable[str]]) -> Optional[List[str]]:
    """Turn comma-separated string or iterable into list of strings."""
    if arg is None:
        return None
    if isinstance(arg, str):
        return [x.strip() for x in arg.split(",") if x.strip()]
    return [str(x) for x in arg]


def resolve_processed_data_path(
    paths: PathManager,
    version: Optional[str],
    experiment_name: str = "pilot_study",
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
    pipeline_mode: str = "llm_with_humans",
    input_file: Optional[str] = None,
) -> Path:
    """Resolve the CSV path using the same semantics as scripts/plot_results.py.

    Parameters mirror plotting script to provide a consistent UX.
    """
    if input_file:
        return Path(input_file)

    processed_base = paths.base_dir / "data" / "processed"

    if pipeline_mode == "humans":
        experiment_dir = processed_base / "human" / "rw17"
    elif pipeline_mode == "llm":
        experiment_dir = processed_base / "llm" / "rw17" / experiment_name
    else:
        experiment_dir = processed_base / "llm_with_humans" / "rw17" / experiment_name

    version_str = f"{version}_v_" if version else ""

    if pipeline_mode == "humans":
        data_path = experiment_dir / f"rw17_{graph_type}_humans_processed.csv"
    elif pipeline_mode == "llm":
        if use_roman_numerals:
            data_path = experiment_dir / "reasoning_types" / f"{version_str}{graph_type}_llm_only_roman.csv"
        else:
            data_path = experiment_dir / f"{version_str}{graph_type}_llm_only.csv"
    else:
        if use_roman_numerals and use_aggregated:
            data_path = experiment_dir / "reasoning_types" / f"{version_str}{graph_type}_cleaned_data_roman.csv"
        elif use_aggregated:
            data_path = experiment_dir / f"{version_str}humans_avg_equal_sample_size_cogsci.csv"
        else:
            data_path = experiment_dir / f"{version_str}{graph_type}_cleaned_data.csv"

    return data_path


def load_processed_data(
    paths: PathManager,
    version: Optional[str],
    experiment_name: str = "pilot_study",
    graph_type: str = "collider",
    use_roman_numerals: bool = True,
    use_aggregated: bool = True,
    pipeline_mode: str = "llm_with_humans",
    input_file: Optional[str] = None,
) -> pd.DataFrame:
    """Load processed data with identical logic to plotting, raising helpful errors."""
    data_path = resolve_processed_data_path(
        paths=paths,
        version=version,
        experiment_name=experiment_name,
        graph_type=graph_type,
        use_roman_numerals=use_roman_numerals,
        use_aggregated=use_aggregated,
        pipeline_mode=pipeline_mode,
        input_file=input_file,
    )

    # Friendly fallback: when non-aggregated human rows are needed, some repos provide
    # a file named '*_cleaned_data_indiv_humans.csv'. If the computed non-aggregated path
    # is missing, try that alternative before erroring out.
    if not data_path.exists():
        parent = data_path.parent
        name = data_path.name
        alt_candidates: list[Path] = []
        try:
            if name.endswith("_cleaned_data.csv"):
                alt_candidates.append(parent / name.replace("_cleaned_data.csv", "_cleaned_data_indiv_humans.csv"))
        except Exception:
            pass
        # Use the first existing alternative
        for alt in alt_candidates:
            if alt.exists():
                logger.info(f"Resolved missing data file by using alternative: {alt}")
                data_path = alt
                break

    if not data_path.exists():
        available = list(data_path.parent.glob("*.csv"))
        msg = f"Data file not found: {data_path}\n"
        if available:
            msg += "Available files in directory:\n" + "\n".join(f"- {p.name}" for p in available)
        else:
            msg += f"No CSV files found in: {data_path.parent}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(data_path)
    return df


def prepare_dataset(
    df: pd.DataFrame,
    agents: Optional[Sequence[str]] = None,
    domains: Optional[Sequence[str]] = None,
    temperature: Optional[float] = None,
    reasoning_types: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    prompt_categories: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Apply standard filters and normalize response to [0,1].

    - agents: filter by `subject` column
    - domains: filter by `domain` column
    - temperature: select exact match for LLM rows; allow NaN for humans
    - reasoning_types: filter by `reasoning_type`
    - tasks: filter by `task` (expects Roman numerals when Roman variant is loaded)
    - prompt_categories: filter by `prompt_category`
    """
    df = _normalize_response_column(df)

    if agents is not None:
        df = _filter_by_values(df, "subject", agents)
    if domains is not None and "domain" in df.columns:
        df = _filter_by_values(df, "domain", domains)
    if temperature is not None and "temperature" in df.columns:
        # Permit human variants with NaN temperature to pass the filter.
        # We consider any subject labeled 'humans', 'human', or prefixed with 'human-' / 'humans-' as human-like.
        subj_lower = df["subject"].astype(str).str.strip().str.lower()
        is_humanish = (
            subj_lower.eq("humans")
            | subj_lower.eq("human")
            | subj_lower.eq("humans-pooled")
            | subj_lower.str.startswith("human-")
            | subj_lower.str.startswith("humans-")
        )
        df = df[(df["temperature"] == temperature) | (is_humanish & df["temperature"].isna())]
    if reasoning_types is not None and "reasoning_type" in df.columns:
        df = _filter_by_values(df, "reasoning_type", reasoning_types)
    if tasks is not None and "task" in df.columns:
        df = _filter_by_values(df, "task", tasks)
    if prompt_categories is not None and "prompt_category" in df.columns:
        df = _filter_by_values(df, "prompt_category", prompt_categories)

    # Drop rows missing essentials
    required = ["subject", "task", "response"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["response", "task", "subject"]).copy()
    return df


