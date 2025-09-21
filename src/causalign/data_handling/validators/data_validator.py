"""
Data Validators

This module provides validation classes for ensuring data quality and consistency
across the processing pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Mapping

import pandas as pd


class BaseDataValidator(ABC):
    """Base class for data validators"""

    @abstractmethod
    def validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Validate a dataframe and return list of error messages"""
        pass

    def _check_required_columns(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> List[str]:
        """Check if dataframe has required columns"""
        errors = []
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        return errors

    def _check_data_types(
        self, df: pd.DataFrame, expected_types: Mapping[str, object]
    ) -> List[str]:
        """Check if columns have expected data types"""
        errors = []
        for col, expected_type in expected_types.items():
            if col in df.columns:
                if expected_type == "numeric":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"Column '{col}' should be numeric")
                elif expected_type == "string":
                    if not pd.api.types.is_string_dtype(
                        df[col]
                    ) and not pd.api.types.is_object_dtype(df[col]):
                        errors.append(f"Column '{col}' should be string/object type")
        return errors

    def _check_value_ranges(
        self, df: pd.DataFrame, value_ranges: Dict[str, tuple]
    ) -> List[str]:
        """Check if numeric columns are within expected ranges"""
        errors = []
        for col, (min_val, max_val) in value_ranges.items():
            if col in df.columns:
                numeric_values = pd.to_numeric(df[col], errors="coerce").dropna()
                if not numeric_values.empty:
                    if numeric_values.min() < min_val or numeric_values.max() > max_val:
                        errors.append(
                            f"Column '{col}' has values outside range [{min_val}, {max_val}]"
                        )
        return errors


class LLMResponseValidator(BaseDataValidator):
    """Validator for LLM response data"""

    def __init__(self):
        self.required_columns = ["id", "subject", "likelihood", "temperature"]
        self.optional_columns = [
            "confidence",
            "explanation",
            "prompt_category",
            "content_category",
            "graph",
            "domain",
            "task",
            "query",
            "cntbl_cond",
            "reasoning_effort",
            "verbosity",
        ]
        self.expected_types: Mapping[str, object] = {
            "id": "numeric",
            "likelihood": "numeric",
            "confidence": "numeric",
            "temperature": "numeric",
            "subject": "string",
            "domain": "string",
            "task": "string",
            "graph": "string",
            "content_category": "string",
            "reasoning_effort": "string",
            "verbosity": "string",
        }
        self.value_ranges = {
            "likelihood": (0, 100),
            "confidence": (0, 100),
            "temperature": (0.0, 1.0),
        }

    def validate_llm_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Validate LLM response dataframe"""
        errors = []

        if df.empty:
            errors.append("DataFrame is empty")
            return errors

        # Check required columns
        errors.extend(self._check_required_columns(df, self.required_columns))

        # Check data types
        errors.extend(self._check_data_types(df, self.expected_types))

        # Check value ranges
        errors.extend(self._check_value_ranges(df, self.value_ranges))

        # Check for duplicate IDs within same subject
        if "id" in df.columns and "subject" in df.columns:
            duplicates = (
                df.groupby("subject")["id"].apply(lambda x: x.duplicated().sum()).sum()
            )
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate IDs within subjects")

        # Check for missing likelihood values
        if "likelihood" in df.columns:
            missing_likelihood = df["likelihood"].isna().sum()
            if missing_likelihood > 0:
                errors.append(f"Found {missing_likelihood} missing likelihood values")

        return errors

    def validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Alias for validate_llm_dataframe for compatibility"""
        return self.validate_llm_dataframe(df)


class HumanResponseValidator(BaseDataValidator):
    """Validator for human response data"""

    def __init__(self):
        self.required_columns = ["id", "subject", "likelihood"]
        self.optional_columns = [
            "human_subj_id",
            "domain",
            "task",
            "query",
            "cntbl_cond",
            "graph",
        ]
        self.expected_types = {
            "id": "numeric",
            "likelihood": "numeric",
            "human_subj_id": "numeric",
            "subject": "string",
            "domain": "string",
            "task": "string",
            "graph": "string",
        }
        self.value_ranges = {
            "likelihood": (0, 100),
            "human_subj_id": (0, 1000),  # Reasonable range for participant IDs
        }

    def validate_human_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Validate human response dataframe"""
        errors = []

        if df.empty:
            errors.append("DataFrame is empty")
            return errors

        # Check required columns
        errors.extend(self._check_required_columns(df, self.required_columns))

        # Check data types
        errors.extend(self._check_data_types(df, self.expected_types))

        # Check value ranges
        errors.extend(self._check_value_ranges(df, self.value_ranges))

        # Check that subject is 'humans'
        if "subject" in df.columns:
            non_human_subjects = df[df["subject"] != "humans"]["subject"].unique()
            if len(non_human_subjects) > 0:
                errors.append(
                    f"Found non-human subjects in human data: {non_human_subjects}"
                )

        # Check for missing likelihood values
        if "likelihood" in df.columns:
            missing_likelihood = df["likelihood"].isna().sum()
            if missing_likelihood > 0:
                errors.append(f"Found {missing_likelihood} missing likelihood values")

        return errors

    def validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Alias for validate_human_dataframe for compatibility"""
        return self.validate_human_dataframe(df)


class CombinedDataValidator(BaseDataValidator):
    """Validator for combined LLM and human data"""

    def __init__(self):
        self.required_columns = ["id", "subject", "likelihood"]
        self.expected_subjects = ["humans"]  # Will be extended with LLM model names

    def validate_combined_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Validate combined LLM and human dataframe"""
        errors = []

        if df.empty:
            errors.append("DataFrame is empty")
            return errors

        # Check required columns
        errors.extend(self._check_required_columns(df, self.required_columns))

        # Check that we have both human and LLM data
        if "subject" in df.columns:
            subjects = df["subject"].unique()
            has_humans = "humans" in subjects
            has_llms = any(subj != "humans" for subj in subjects)

            if not has_humans:
                errors.append("No human data found in combined dataset")
            if not has_llms:
                errors.append("No LLM data found in combined dataset")

        # Check ID consistency across subjects
        if "id" in df.columns and "subject" in df.columns:
            id_counts_by_subject = df.groupby("subject")["id"].nunique()
            unique_counts = id_counts_by_subject.unique()

            # All subjects should have the same number of unique IDs (ideally)
            if len(unique_counts) > 1:
                errors.append(
                    f"Inconsistent number of unique IDs across subjects: {dict(id_counts_by_subject)}"
                )

        return errors

    def validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Alias for validate_combined_dataframe for compatibility"""
        return self.validate_combined_dataframe(df)
