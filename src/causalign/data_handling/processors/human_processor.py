"""
Human Data Processor

This module handles loading, cleaning, and processing human response data.
It manages the challenge of assigning unique IDs to match with LLM responses.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...config.paths import PathManager
from ..validators.data_validator import HumanResponseValidator


class HumanDataProcessor:
    """
    Processes human response data and assigns proper IDs for merging with LLM data.

    This handles the challenge of processing raw human data from data/raw/human/rw17
    and assigning unique identifiers that match the prompt IDs used in LLM experiments.
    """

    def __init__(self, paths: Optional[PathManager] = None):
        self.paths = paths or PathManager()
        self.validator = HumanResponseValidator()
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

    def load_raw_human_data(self, filename: str) -> pd.DataFrame:
        """Load raw human data from data/raw/human/rw17/"""
        file_path = self.paths.get_human_raw_data_path(filename)

        if not file_path.exists():
            raise FileNotFoundError(f"Human data file not found: {file_path}")

        self.logger.info(f"Loading raw human data: {file_path}")

        # Try different separators (semicolon first, then comma)
        try:
            df = pd.read_csv(file_path, sep=";")
        except Exception:
            try:
                df = pd.read_csv(file_path, sep=",")
            except Exception as e:
                self.logger.error(
                    f"Failed to read human data with both separators: {e}"
                )
                raise

        self.logger.info(f"Loaded {len(df)} rows of human data")
        return df

    def assign_prompt_ids(
        self, df: pd.DataFrame, prompt_mapping_file: str
    ) -> pd.DataFrame:
        """
        Load prompt mapping file and match human responses to prompt IDs.

        This method follows the ground truth logic where prompt IDs are assigned
        to prompts first, then human data is matched based on experimental conditions.

        Args:
            df: Human data DataFrame (must contain: domain, task, cntbl_cond, graph)
            prompt_mapping_file: CSV file with prompt mappings in data/input_llm/rw17/

        Returns:
            DataFrame: Human data with matched prompt IDs

        Raises:
            ValueError: If required columns are missing or no matches found
        """
        self.logger.info(f"Loading prompt mapping file: {prompt_mapping_file}")

        # Load prompt mapping file
        mapping_path = self.paths.rw17_input_llm_dir / prompt_mapping_file
        if not mapping_path.exists():
            raise FileNotFoundError(f"Prompt mapping file not found: {mapping_path}")

        try:
            prompt_data = pd.read_csv(mapping_path)
        except Exception as e:
            self.logger.error(f"Failed to load prompt mapping file: {e}")
            raise

        self.logger.info(f"Loaded prompt data with {len(prompt_data)} prompts")

        # Match human data to prompt IDs using the helper method
        return self._match_human_to_prompts(df, prompt_data)

    def _match_human_to_prompts(
        self, df: pd.DataFrame, prompt_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Match human responses to prompt IDs based on experimental conditions.

        This follows the notebook logic: merge on domain, task, cntbl_cond, graph
        """
        self.logger.info(
            "Matching human data to prompt IDs based on experimental conditions"
        )

        # Check required columns in human data
        human_required_cols = ["domain", "task", "cntbl_cond", "graph"]
        missing_human_cols = [
            col for col in human_required_cols if col not in df.columns
        ]
        if missing_human_cols:
            raise ValueError(
                f"Missing required columns in human data: {missing_human_cols}"
            )

        # Check required columns in prompt data
        prompt_required_cols = ["id", "domain", "task", "cntbl_cond", "graph"]
        missing_prompt_cols = [
            col for col in prompt_required_cols if col not in prompt_data.columns
        ]
        if missing_prompt_cols:
            raise ValueError(
                f"Missing required columns in prompt data: {missing_prompt_cols}"
            )

        # Get unique experimental conditions
        human_conditions = df[human_required_cols].drop_duplicates()
        prompt_conditions = prompt_data[human_required_cols].drop_duplicates()

        self.logger.info(
            f"Human data: {len(human_conditions)} unique experimental conditions"
        )
        self.logger.info(
            f"Prompt data: {len(prompt_conditions)} unique experimental conditions"
        )

        # Merge human data with prompt data on experimental conditions
        merged_df = pd.merge(
            df,
            prompt_data[["id"] + human_required_cols],
            on=human_required_cols,
            how="left",
        )

        # Check for unmatched human responses
        unmatched = merged_df[merged_df["id"].isna()]
        if len(unmatched) > 0:
            self.logger.warning(
                f"Found {len(unmatched)} human responses that could not be matched to prompts"
            )
            unmatched_conditions = unmatched[human_required_cols].drop_duplicates()
            self.logger.warning(
                f"Example unmatched conditions:\n{unmatched_conditions.head()}"
            )

        # Remove unmatched rows
        matched_df = merged_df.dropna(subset=["id"])

        if len(matched_df) == 0:
            raise ValueError("No human responses could be matched to prompt IDs")

        # Convert id to integer
        matched_df["id"] = matched_df["id"].astype(int)

        self.logger.info(
            f"Successfully matched {len(matched_df)} human responses to prompt IDs"
        )
        self.logger.info(
            f"Matched responses span {matched_df['id'].nunique()} unique prompt IDs"
        )

        return matched_df

    def clean_human_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize human response data following the notebook logic.

        This method transforms raw human data to match the expected format:
        - Renames columns to match experiment structure
        - Maps domain values (society → sociology)
        - Converts response values to numeric
        - Adds required metadata columns
        """
        df = df.copy()
        self.logger.info("Starting human data cleaning following notebook logic")

        # Step 0: Handle column name conflicts
        # The raw data has both 'letter.type' (task letters: a,b,c,etc.) and 'task' (only 'c' values)
        # We need the task letters, so drop the original 'task' column first
        if "task" in df.columns and "letter.type" in df.columns:
            self.logger.info("Dropping original 'task' column to avoid naming conflict")
            df.drop(columns=["task"], inplace=True)

        # Step 1: Handle column renaming following notebook transformations
        column_mappings = {
            "letter.type": "task",  # letter task labels (a, b, c, etc.)
            "attr.polarity": "cntbl_cond",  # counterbalance condition (ppp, pmm, etc.)
            "y": "response",  # human likelihood rating
            "s": "human_subj_id",  # subject identifier
        }

        # Apply column mappings if columns exist
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
                self.logger.info(f"Renamed column: {old_col} → {new_col}")

        # Step 2: Convert task labels to lowercase (following notebook)
        if "task" in df.columns:
            df["task"] = df["task"].str.lower()
            self.logger.info("Converted task labels to lowercase")

        # Step 3: Handle domain mapping (society → sociology)
        if "domain" in df.columns:
            original_domains = df["domain"].unique()
            df["domain"] = df["domain"].replace({"society": "sociology"})
            new_domains = df["domain"].unique()
            self.logger.info(f"Domain mapping: {original_domains} → {new_domains}")

        # Step 4: Clean response values (convert comma decimals to dots, then to float)
        if "response" in df.columns:
            # Handle comma decimal separators (European format)
            df["response"] = (
                df["response"].astype(str).str.replace(",", ".", regex=False)
            )
            df["response"] = pd.to_numeric(df["response"], errors="coerce")
            # Rename to likelihood for consistency with LLM data
            df.rename(columns={"response": "likelihood"}, inplace=True)
            self.logger.info("Converted response values to numeric likelihood")
        elif "y" in df.columns:
            # Direct handling if y column exists
            df["y"] = df["y"].astype(str).str.replace(",", ".", regex=False)
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df.rename(columns={"y": "likelihood"}, inplace=True)
            self.logger.info("Converted y column to numeric likelihood")

        # Step 5: Add required metadata columns
        df["subject"] = "humans"
        df["temperature"] = np.nan
        df["experiment_name"] = "pilot_study"

        # Add prompt_category to match LLM data structure
        if "prompt_category" not in df.columns:
            df["prompt_category"] = "single_numeric_response"

        # Ensure human_subj_id exists
        if "human_subj_id" not in df.columns:
            df["human_subj_id"] = df.get("participant_id", np.nan)

        # Step 6: Filter to keep only essential columns (following notebook)
        essential_cols = [
            "likelihood",
            "task",
            "human_subj_id",
            "cntbl_cond",
            "domain",
            "subject",
            "temperature",
            "experiment_name",
            "prompt_category",
        ]

        # Keep only columns that exist in the dataframe
        existing_essential_cols = [col for col in essential_cols if col in df.columns]

        # Keep id column if it exists (added during prompt matching)
        if "id" in df.columns:
            existing_essential_cols.append("id")

        # Keep graph column if it exists
        if "graph" in df.columns:
            existing_essential_cols.append("graph")

        df = df[existing_essential_cols]

        # Step 7: Validate the cleaned data
        validation_errors = self.validator.validate_human_dataframe(df)
        if validation_errors:
            self.logger.warning(f"Validation errors: {validation_errors}")

        self.logger.info(
            f"Human data cleaning complete: {len(df)} records with columns {list(df.columns)}"
        )
        return df

    def process_human_data(
        self, raw_filename: str, prompt_mapping_file: str, graph_type: str = "collider"
    ) -> pd.DataFrame:
        """
        Process human data and assign prompt IDs to match LLM data

        PROCESS ORDER (Updated to follow notebook logic):
        1. Load raw human data
        2. Clean and transform data (column renaming, domain mapping)
        3. Add graph type
        4. Assign prompt IDs by matching to prompt data

        Args:
            raw_filename: Filename in data/raw/human/rw17/
                         Examples: "rw17_collider_ce.csv", "rw17_fork_cc.csv"
            prompt_mapping_file: CSV file with prompts and IDs in data/input_llm/rw17/
                               Contains prompts with assigned IDs for matching
            graph_type: Graph structure ("collider", "fork", "chain")

        Returns:
            pd.DataFrame: Processed human data with matched prompt IDs
        """

        # Step 1: Load raw data
        df = self.load_raw_human_data(raw_filename)

        # Step 2: Clean and transform data (must be done before ID assignment)
        # This handles column renaming: letter.type→task, attr.polarity→cntbl_cond, etc.
        df = self.clean_human_responses(df)

        # Step 3: Add graph type (needed for prompt matching)
        df["graph"] = graph_type

        # Step 4: Assign prompt IDs by matching to prompt data
        # This requires cleaned column names: domain, task, cntbl_cond, graph
        df = self.assign_prompt_ids(df, prompt_mapping_file)

        self.logger.info(f"Human data processing complete: {len(df)} records")
        return df

    def save_processed_human_data(self, df: pd.DataFrame, output_filename: str) -> Path:
        """Save processed human data"""
        output_path = self.paths.get_human_processed_data_path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed human data: {output_path}")

        return output_path

    def aggregate_human_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate human responses by averaging per prompt ID.

        PURPOSE: Balance sample sizes between humans and LLMs for fair statistical comparison

        PROBLEM:
        - Humans: Multiple responses per prompt ID (5440 responses → many per ID)
        - LLMs: One response per prompt ID (720 responses → one per ID)
        - Unbalanced comparison affects statistical tests

        SOLUTION:
        - Average human responses per prompt ID (5440 → 336 responses)
        - Keep LLM responses unchanged (720 responses)
        - Final balanced dataset: ~1056 rows total

        CONFIGURATION IMPACT:
        - aggregate_human_responses=True: Enables this balancing (recommended)
        - aggregate_human_responses=False: Keeps all individual responses

        Args:
            df: Combined dataframe with both human and LLM data

        Returns:
            pd.DataFrame: Dataset with aggregated human responses
        """

        if "likelihood" not in df.columns:
            raise ValueError("DataFrame must contain 'likelihood' column")

        human_data = df[df["subject"] == "humans"].copy()

        if human_data.empty:
            self.logger.warning("No human data found for aggregation")
            return df

        # Calculate mean likelihood per ID
        human_means = (
            human_data.groupby("id", as_index=False)["likelihood"]
            .mean()
            .rename(columns={"likelihood": "mean_likelihood"})
        )

        # Get representative row per ID (preserve metadata)
        representative_humans = (
            human_data.sort_values(by="id").groupby("id", as_index=False).first()
        )

        # Replace likelihood with mean
        representative_humans = representative_humans.drop(columns=["likelihood"])
        representative_humans = representative_humans.merge(human_means, on="id")
        representative_humans = representative_humans.rename(
            columns={"mean_likelihood": "likelihood"}
        )

        # Combine with non-human data
        non_human_data = df[df["subject"] != "humans"]
        final_df = pd.concat([non_human_data, representative_humans], ignore_index=True)

        original_human_count = len(human_data)
        final_human_count = len(representative_humans)

        self.logger.info(
            f"Aggregated human responses: {original_human_count} → {final_human_count} rows"
        )

        return final_df.sort_values(by="id").reset_index(drop=True)
