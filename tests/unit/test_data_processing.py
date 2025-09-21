"""
Unit tests for data processing and validation functions.
"""

import pandas as pd
import pytest

# Note: These imports may need adjustment based on actual module structure
# from causalign.data.processing import validate_csv_format, load_experiment_data


class TestDataValidation:
    """Test data validation functionality."""

    def test_valid_csv_format(self, sample_input_data):
        """Test validation of properly formatted CSV data."""
        # This would test a validation function if it exists
        required_columns = [
            "id",
            "prompt",
            "prompt_category",
            "graph",
            "domain",
            "cntbl_cond",
            "task",
        ]

        # Check all required columns are present
        for col in required_columns:
            assert col in sample_input_data.columns, f"Missing required column: {col}"

        # Check no empty values in critical columns
        assert not sample_input_data["id"].isnull().any(), (
            "ID column contains null values"
        )
        assert not sample_input_data["prompt"].isnull().any(), (
            "Prompt column contains null values"
        )

    def test_invalid_csv_missing_columns(self):
        """Test validation fails with missing required columns."""
        invalid_data = pd.DataFrame(
            [
                {"id": "1", "prompt": "test prompt"}  # Missing other required columns
            ]
        )

        required_columns = [
            "id",
            "prompt",
            "prompt_category",
            "graph",
            "domain",
            "cntbl_cond",
            "task",
        ]
        missing_columns = set(required_columns) - set(invalid_data.columns)

        assert len(missing_columns) > 0, "Should have missing columns"
        assert "prompt_category" in missing_columns
        assert "graph" in missing_columns

    def test_empty_dataframe_validation(self):
        """Test validation of empty dataframe."""
        empty_df = pd.DataFrame()

        assert empty_df.empty, "DataFrame should be empty"
        assert len(empty_df) == 0, "DataFrame should have zero rows"

    def test_duplicate_ids_detection(self):
        """Test detection of duplicate IDs in input data."""
        data_with_duplicates = pd.DataFrame(
            [
                {
                    "id": "duplicate_001",
                    "prompt": "First prompt",
                    "prompt_category": "test",
                    "graph": "A -> B",
                    "domain": "test",
                    "cntbl_cond": "none",
                    "task": "test",
                },
                {
                    "id": "duplicate_001",  # Duplicate ID
                    "prompt": "Second prompt",
                    "prompt_category": "test",
                    "graph": "C -> D",
                    "domain": "test",
                    "cntbl_cond": "none",
                    "task": "test",
                },
            ]
        )

        # Check for duplicates
        duplicates = data_with_duplicates["id"].duplicated()
        assert duplicates.any(), "Should detect duplicate IDs"
        assert duplicates.sum() == 1, "Should find exactly one duplicate"

    def test_special_characters_in_prompts(self):
        """Test handling of special characters in prompts."""
        special_char_data = pd.DataFrame(
            [
                {
                    "id": "special_001",
                    "prompt": 'Does "smoking" cause lung cancer? (Yes/No)',
                    "prompt_category": "special_chars",
                    "graph": "smoking -> cancer",
                    "domain": "health",
                    "cntbl_cond": "none",
                    "task": "causal_reasoning",
                },
                {
                    "id": "special_002",
                    "prompt": "What about A → B & C → D?",
                    "prompt_category": "unicode",
                    "graph": "A -> B, C -> D",
                    "domain": "test",
                    "cntbl_cond": "none",
                    "task": "causal_reasoning",
                },
            ]
        )

        # Verify data integrity is maintained
        assert (
            special_char_data.iloc[0]["prompt"]
            == 'Does "smoking" cause lung cancer? (Yes/No)'
        )
        assert "→" in special_char_data.iloc[1]["prompt"]
        assert "&" in special_char_data.iloc[1]["prompt"]


class TestCSVFileHandling:
    """Test CSV file loading and saving functionality."""

    def test_csv_loading(self, sample_csv_file):
        """Test loading CSV file."""
        df = pd.read_csv(sample_csv_file)

        assert not df.empty, "Loaded dataframe should not be empty"
        assert "id" in df.columns, "Should contain id column"
        assert "prompt" in df.columns, "Should contain prompt column"

    def test_csv_saving(self, tmp_path, sample_input_data):
        """Test saving dataframe to CSV."""
        output_file = tmp_path / "test_output.csv"

        sample_input_data.to_csv(output_file, index=False)

        assert output_file.exists(), "Output CSV file should be created"

        # Verify content
        loaded_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(sample_input_data, loaded_df)

    def test_csv_with_different_encodings(self, tmp_path):
        """Test handling CSV files with different text encodings."""
        # Create CSV with special characters
        data_with_unicode = pd.DataFrame(
            [
                {
                    "id": "unicode_001",
                    "prompt": "Does café consumption → better mood?",
                    "prompt_category": "unicode_test",
                    "graph": "café → mood",
                    "domain": "psychology",
                    "cntbl_cond": "none",
                    "task": "causal_reasoning",
                }
            ]
        )

        # Save with UTF-8 encoding
        utf8_file = tmp_path / "utf8_test.csv"
        data_with_unicode.to_csv(utf8_file, index=False, encoding="utf-8")

        # Load and verify
        loaded_df = pd.read_csv(utf8_file, encoding="utf-8")
        assert "→" in loaded_df.iloc[0]["prompt"]
        assert "café" in loaded_df.iloc[0]["prompt"]

    @pytest.mark.parametrize("separator", [",", ";", "\t"])
    def test_different_csv_separators(self, tmp_path, separator):
        """Test handling CSV files with different separators."""
        data = pd.DataFrame(
            [
                {
                    "id": "sep_test_001",
                    "prompt": "Test prompt",
                    "prompt_category": "separator_test",
                    "graph": "A -> B",
                    "domain": "test",
                    "cntbl_cond": "none",
                    "task": "test",
                }
            ]
        )

        # Save with different separator
        sep_file = tmp_path / f"sep_test_{separator.replace('/', '_')}.csv"
        data.to_csv(sep_file, index=False, sep=separator)

        # Load with correct separator
        loaded_df = pd.read_csv(sep_file, sep=separator)
        assert len(loaded_df) == 1
        assert loaded_df.iloc[0]["id"] == "sep_test_001"


class TestDataTransformations:
    """Test data transformation functions."""

    def test_add_metadata_columns(self, sample_input_data):
        """Test adding metadata columns to experimental data."""
        # Simulate adding response metadata
        enhanced_data = sample_input_data.copy()
        enhanced_data["response"] = "Test response"
        enhanced_data["subject"] = enhanced_data["id"]
        enhanced_data["temperature"] = 0.7

        # Verify new columns
        assert "response" in enhanced_data.columns
        assert "subject" in enhanced_data.columns
        assert "temperature" in enhanced_data.columns

        # Verify data integrity
        assert all(enhanced_data["subject"] == enhanced_data["id"])
        assert all(enhanced_data["temperature"] == 0.7)

    def test_data_filtering(self, sample_input_data):
        """Test filtering data based on criteria."""
        # Filter by domain
        health_data = sample_input_data[sample_input_data["domain"] == "health"]

        assert len(health_data) == 1  # Based on our sample data
        assert health_data.iloc[0]["domain"] == "health"

    def test_data_grouping(self, sample_input_data):
        """Test grouping data by categories."""
        # Group by domain
        grouped = sample_input_data.groupby("domain")

        assert "health" in grouped.groups
        assert "economics" in grouped.groups

        # Verify group sizes
        domain_counts = sample_input_data["domain"].value_counts()
        assert domain_counts["health"] >= 1
        assert domain_counts["economics"] >= 1
