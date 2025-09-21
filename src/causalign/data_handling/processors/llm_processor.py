"""
LLM Response Processor

This module handles loading, cleaning, and processing LLM responses from the output directory.
It replaces the functionality from 0_response_loader.ipynb with a more robust script-based approach.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...config.paths import PathManager
from ..validators.data_validator import LLMResponseValidator


@dataclass
class ProcessingStats:
    """Statistics from LLM response processing"""

    xml_parsed: int = 0
    cot_parsed: int = 0
    numeric_parsed: int = 0
    dropped: int = 0
    quotation_stripped: int = 0
    total_files: int = 0


class LLMResponseProcessor:
    """
    Processes LLM responses from output directory and standardizes them.

    This replaces the functionality from the 0_response_loader.ipynb notebook
    with a more robust, scriptable approach.
    """

    def __init__(self, paths: Optional[PathManager] = None):
        self.paths = paths or PathManager()
        self.validator = LLMResponseValidator()
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processor"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def parse_llm_response(self, response_str: str) -> Dict[str, str]:
        """Parse LLM response in XML format"""
        if not isinstance(response_str, str):
            return {}

        response_str = response_str.strip().strip('"').strip()
        match = re.search(
            r"<response>(.*?)</response>", response_str, re.DOTALL | re.IGNORECASE
        )
        if not match:
            return {}

        inner = match.group(1)
        entries = re.findall(r"<(\w+?)>(.*?)</\1>", inner, re.DOTALL)
        parsed_data = {tag.strip(): value.strip() for tag, value in entries}

        # Rename explanation to reasoning for consistency
        if "explanation" in parsed_data:
            parsed_data["reasoning"] = parsed_data.pop("explanation")

        return parsed_data

    def parse_numeric_response(self, response_str: str) -> Dict[str, str]:
        """Parse plain text numeric response format"""
        if not isinstance(response_str, str):
            return {}

        response_str = response_str.strip().strip('"').strip()

        # Look for a standalone number between 0-100
        match = re.search(r"\b(\d{1,3})\b", response_str)
        if not match:
            return {}

        number = int(match.group(1))
        # Validate range
        if 0 <= number <= 100:
            return {"likelihood": str(number)}

        return {}

    def parse_fallback_response(self, response_str: str) -> Dict[str, str]:
        """Parse fallback CoT response format"""
        if not isinstance(response_str, str):
            return {}

        response_str = response_str.strip().strip('"')
        match = re.search(
            r"YOUR_STEP_BY_STEP_REASONING[:\s]*(.*?)\s*YOUR_NUMERIC_RESPONSE_HERE[:\s]*(\d+)\s*YOUR_CONFIDENCE_SCORE_HERE[:\s]*(\d+)",
            response_str,
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return {}

        explanation, likelihood, confidence = match.groups()
        return {
            "reasoning": explanation.strip(),
            "likelihood": likelihood.strip(),
            "confidence": confidence.strip(),
        }

    def is_valid_llm_response(self, response_str: str) -> bool:
        """Check if response is valid LLM response (XML format)"""
        if not isinstance(response_str, str):
            return False
        if "error" in response_str.lower():
            return False
        if "<response>" not in response_str or "</response>" not in response_str:
            return False
        return True

    def is_valid_numeric_response(self, response_str: str) -> bool:
        """Check if response is valid plain text numeric response"""
        if not isinstance(response_str, str):
            return False
        if "error" in response_str.lower():
            return False

        response_str = response_str.strip().strip('"').strip()

        # Check if it looks like a simple numeric response
        # Should be mostly just a number, possibly with minimal extra text
        match = re.search(r"\b(\d{1,3})\b", response_str)
        if not match:
            return False

        number = int(match.group(1))
        if not (0 <= number <= 100):
            return False

        # If the response is very short and contains the number, it's likely numeric-only
        # Allow some flexibility for responses like "85" or "My answer is 85"
        if len(response_str) <= 50 and number <= 100:
            return True

        return False

    def process_single_file(
        self, file_path: Path, model_name: str, experiment_name: str
    ) -> Tuple[pd.DataFrame, ProcessingStats]:
        """Process a single LLM response file"""

        self.logger.info(f"Processing file: {file_path}")

        # Extract temperature from filename
        temperature = self._extract_temperature(file_path.name)

        # Load CSV with fallback for different separators
        # Force response column to be read as string to prevent auto-conversion
        try:
            # First try semicolon separator (common for LLM outputs)
            df = pd.read_csv(file_path, sep=";", dtype={"response": "str"})
        except Exception:
            try:
                # Fallback to comma separator
                df = pd.read_csv(file_path, sep=",", dtype={"response": "str"})
            except Exception as e:
                self.logger.error(
                    f"Failed to read {file_path} with both separators: {e}"
                )
                return pd.DataFrame(), ProcessingStats()

        if df.empty:
            self.logger.warning(f"Empty file: {file_path}")
            return pd.DataFrame(), ProcessingStats()

        # Ensure presence of optional metadata columns introduced by GPT-5 Responses API
        for meta_col in ["reasoning_effort", "verbosity", "content_category"]:
            if meta_col not in df.columns:
                df[meta_col] = "unspecified" if meta_col == "content_category" else "N/A"
            else:
                df[meta_col] = df[meta_col].fillna(
                    "unspecified" if meta_col == "content_category" else "N/A"
                )

        # Process responses
        stats = ProcessingStats(total_files=1)
        valid_rows = []
        parsed_columns = set()

        for i, row in df.iterrows():
            raw_response = row.get("response", "")

            # Handle quotation marks
            if (
                isinstance(raw_response, str)
                and raw_response.strip().startswith('"')
                and raw_response.strip().endswith('"')
            ):
                stats.quotation_stripped += 1

            # Parse response - try different formats in order of preference
            parsed = {}

            # Try XML format first
            if self.is_valid_llm_response(raw_response):
                parsed = self.parse_llm_response(raw_response)
                if parsed:
                    stats.xml_parsed += 1

            # Try plain text numeric format if XML failed
            if not parsed and self.is_valid_numeric_response(raw_response):
                parsed = self.parse_numeric_response(raw_response)
                if parsed:
                    stats.numeric_parsed += 1

            # Fallback to CoT parsing if both XML and numeric failed
            if not parsed:
                parsed = self.parse_fallback_response(raw_response)
                if parsed:
                    stats.cot_parsed += 1

            if not parsed:
                stats.dropped += 1
                continue

            # Add parsed fields to dataframe
            for key, val in parsed.items():
                df.at[i, key] = val
                parsed_columns.add(key)
            valid_rows.append(i)

        # Keep only valid rows
        df = df.loc[valid_rows].copy()

        # Add metadata
        df["subject"] = model_name
        df["temperature"] = temperature
        df["experiment_name"] = experiment_name

        # Normalize meta columns to lowercase canonical values where applicable
        # Allowed: effort {minimal, low, medium, high}, verbosity {low, medium, high}
        for col in ["reasoning_effort", "verbosity", "content_category"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                if col == "content_category":
                    df[col] = df[col].replace({
                        "": "unspecified",
                        "nan": "unspecified",
                        "none": "unspecified",
                        "n/a": "unspecified",
                    })
                else:
                    df[col] = df[col].replace({"none": "n/a", "": "n/a", "nan": "n/a"})

        # Normalize prompt category
        df = self._normalize_prompt_category(df, file_path.name)

        # Convert numeric columns
        numeric_columns = [
            "likelihood",
            "confidence",
        ]  # Only convert actual numeric columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Variant-aware subject labeling for GPT-5 family (verbosity/reasoning_effort)
        try:
            if isinstance(model_name, str) and model_name.startswith("gpt-5"):
                # Ensure lowercase strings for consistency
                if "verbosity" in df.columns:
                    v = df["verbosity"].astype(str).str.strip().str.lower()
                else:
                    v = pd.Series(["n/a"] * len(df))
                if "reasoning_effort" in df.columns:
                    r = df["reasoning_effort"].astype(str).str.strip().str.lower()
                else:
                    r = pd.Series(["n/a"] * len(df))

                # Determine which rows have meaningful variant metadata
                has_variant = ~(
                    v.fillna("n/a").isin(["n/a", "unspecified"]) &
                    r.fillna("n/a").isin(["n/a", "unspecified"])
                )

                # Build variant label only where metadata is present
                variant_label = model_name + "-v_" + v + "-r_" + r
                df.loc[has_variant, "subject"] = variant_label[has_variant]
        except Exception:
            # Non-fatal: default to base model_name if anything goes wrong
            pass

        # Validate the processed data
        validation_errors = self.validator.validate_llm_dataframe(df)
        if validation_errors:
            self.logger.warning(
                f"Validation errors for {file_path}: {validation_errors}"
            )

        self.logger.info(
            f"[{file_path.name}] XML: {stats.xml_parsed} | "
            f"Numeric: {stats.numeric_parsed} | "
            f"CoT: {stats.cot_parsed} | Dropped: {stats.dropped}"
        )

        return df, stats

    def process_experiment(
        self,
        experiment_name: str,
        version: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Process all LLM responses for a given experiment

        CONFIGURATION MAPPING:
        - experiment_name: Maps to data/output_llm/{experiment_name}/
        - version: Filters files by prefix (e.g., "6" → "6_v_...csv")
        - models: Limits processing to specific models (None = all models)

        Args:
            experiment_name: Directory name under data/output_llm/
                           Examples: "pilot_study", "temp_test", "experiment_2024_batch_1"
            version: Filter files by version prefix (e.g., "6" for "6_v_...")
                    None = process ALL versions (can be slow)
            models: List of specific model names to include
                   None = process ALL models found in directory
                   Examples: ["claude-3-opus"], ["claude-3-opus", "gpt-4o"]

        Returns:
            pd.DataFrame: Combined LLM responses with standardized columns
        """

        experiment_dir = self.paths.output_llm_dir / experiment_name
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        all_data = []
        total_stats = ProcessingStats()

        for model_dir in experiment_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            # MODEL FILTERING: Only process specified models to reduce processing time
            # or focus analysis on specific model comparisons
            if models and model_name not in models:
                continue

            self.logger.info(f"Processing model: {model_name}")

            # Check if there are temperature subdirectories or files directly in model dir
            csv_files_direct = list(model_dir.glob("*.csv"))
            temp_subdirs = [d for d in model_dir.iterdir() if d.is_dir()]

            if csv_files_direct:
                # Files directly in model directory
                self.logger.info(
                    f"Found {len(csv_files_direct)} CSV files directly in {model_name}"
                )
                for csv_file in csv_files_direct:
                    # VERSION FILTERING: Only process files with specified version prefix
                    # This prevents processing old/experimental versions when focusing on specific data
                    if version and not csv_file.name.startswith(f"{version}_"):
                        continue

                    df, stats = self.process_single_file(
                        csv_file, model_name, experiment_name
                    )
                    if not df.empty:
                        all_data.append(df)
                        total_stats.xml_parsed += stats.xml_parsed
                        total_stats.numeric_parsed += stats.numeric_parsed
                        total_stats.cot_parsed += stats.cot_parsed
                        total_stats.dropped += stats.dropped
                        total_stats.quotation_stripped += stats.quotation_stripped
                        total_stats.total_files += stats.total_files

            elif temp_subdirs:
                # Temperature subdirectories exist
                self.logger.info(
                    f"Found {len(temp_subdirs)} temperature subdirectories in {model_name}"
                )
                for temp_dir in temp_subdirs:
                    if not temp_dir.is_dir():
                        continue

                    for csv_file in temp_dir.glob("*.csv"):
                        # VERSION FILTERING: Only process files with specified version prefix
                        # This prevents processing old/experimental versions when focusing on specific data
                        if version and not csv_file.name.startswith(f"{version}_"):
                            continue

                        df, stats = self.process_single_file(
                            csv_file, model_name, experiment_name
                        )
                        if not df.empty:
                            all_data.append(df)
                            total_stats.xml_parsed += stats.xml_parsed
                            total_stats.numeric_parsed += stats.numeric_parsed
                            total_stats.cot_parsed += stats.cot_parsed
                            total_stats.dropped += stats.dropped
                            total_stats.quotation_stripped += stats.quotation_stripped
                            total_stats.total_files += stats.total_files
            else:
                self.logger.warning(
                    f"No CSV files or temperature subdirectories found in {model_name}"
                )

        if not all_data:
            self.logger.warning("No valid data found")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Standardize column names
        if (
            "response" in combined_df.columns
            and "likelihood" not in combined_df.columns
        ):
            combined_df.rename(columns={"response": "likelihood"}, inplace=True)
        elif "response" in combined_df.columns and "likelihood" in combined_df.columns:
            combined_df["likelihood"] = combined_df["likelihood"].combine_first(
                combined_df["response"]
            )
            combined_df.drop(columns=["response"], inplace=True)

        # Handle both explanation and reasoning tags, converting to reasoning for consistency
        if "explanation" in combined_df.columns:
            if "reasoning" in combined_df.columns:
                # If both exist, merge explanation into reasoning
                combined_df["reasoning"] = combined_df["reasoning"].combine_first(
                    combined_df["explanation"]
                )
            else:
                # If only explanation exists, rename it
                combined_df.rename(columns={"explanation": "reasoning"}, inplace=True)
            # Drop explanation column if it still exists
            if "explanation" in combined_df.columns:
                combined_df.drop(columns=["explanation"], inplace=True)

        self.logger.info(f"Processing complete. Total stats: {total_stats}")

        return combined_df

    def _extract_temperature(self, filename: str) -> Optional[float]:
        """Extract temperature from filename"""
        for temp in ["0.0", "0.3", "0.5", "0.7", "1.0"]:
            if f"_{temp}_" in filename or filename.endswith(f"_{temp}.csv"):
                return float(temp)
        return None

    def _normalize_prompt_category(
        self, df: pd.DataFrame, filename: str
    ) -> pd.DataFrame:
        """Normalize prompt category based on filename"""
        if "prompt_category" not in df.columns:
            return df

        prompt_counts = df["prompt_category"].value_counts(dropna=False)

        # Check if we have any valid prompt categories
        if prompt_counts.empty:
            self.logger.warning(f"No valid prompt categories found in {filename}")
            # Check if dataframe is empty
            if df.empty:
                self.logger.warning(f"Empty dataframe for {filename}")
                return df
            # Try to infer from filename
            if "numeric-conf" in filename:
                df["prompt_category"] = "numeric-conf"
                inferred_category = "numeric-conf"
            elif "CoT" in filename:
                df["prompt_category"] = "CoT"
                inferred_category = "CoT"
            elif "numeric" in filename:
                df["prompt_category"] = "numeric"
                inferred_category = "numeric"
            else:
                df["prompt_category"] = "unknown"
                inferred_category = "unknown"
            self.logger.info(
                f"Inferred prompt category from filename: {inferred_category}"
            )
            return df

    # Try to match prompt category with filename (supports legacy and short names)
        for category in prompt_counts.index:
            if isinstance(category, str) and f"_{category}_" in filename:
                df["prompt_category"] = category
                self.logger.info(f"Matched prompt category: {category}")
                return df

        # Handle short-name filenames (e.g., _numeric_, _numeric-conf_, _CoT_)
        short_map = {
            "_numeric_conf_": "numeric-conf",  # guard if underscore variant appears
            "_numeric-conf_": "numeric-conf",
            "_numeric_": "numeric",
            "_CoT_": "CoT",
        }
        for needle, mapped in short_map.items():
            if needle in filename:
                df["prompt_category"] = mapped
                self.logger.info(f"Matched short prompt category: {mapped}")
                return df

        # Fallback to most frequent
        fallback_val = prompt_counts.idxmax()
        df["prompt_category"] = fallback_val
        self.logger.info(f"Used fallback prompt category: {fallback_val}")

        return df

    def save_processed_data(
        self, df: pd.DataFrame, output_name: str, experiment_name: str
    ) -> Path:
        """Save processed data to the processed directory"""
        output_dir = (
            self.paths.rw17_processed_llm_dir / "cleaned_data_combined_subjects"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / output_name
        df.to_csv(output_path, index=False)

        self.logger.info(f"Saved processed data to: {output_path}")
        return output_path
