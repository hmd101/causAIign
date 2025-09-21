"""
Unit tests for experiment runner functionality.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from causalign.experiment.api.api_runner import ExperimentRunner
from causalign.experiment.api.client import LLMConfig


class TestExperimentRunner:
    """Test ExperimentRunner functionality."""

    def test_init_with_valid_params(self):
        """Test runner initialization with valid parameters."""
        # Create mock provider configs as expected by the actual constructor
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        runner = ExperimentRunner(
            provider_configs=provider_configs,
            version="test_version",
            cot=False,
            n_times=2,
        )

        assert runner.version == "test_version"
        assert runner.cot == False
        assert runner.n_times == 2
        assert runner.provider_configs == provider_configs

    def test_init_with_default_params(self):
        """Test runner initialization with default parameters."""
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        runner = ExperimentRunner(provider_configs)

        # Check defaults
        assert runner.version == "2_v"
        assert runner.cot == False
        assert runner.n_times == 4
        assert runner.combine_cnt_cond == False

    def test_setup_results_folder(self):
        """Test results folder setup."""
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        with patch("os.makedirs") as mock_makedirs:
            runner = ExperimentRunner(provider_configs, version="test_v")

            # Should create directories
            mock_makedirs.assert_called()

    def test_load_or_create_log_new_file(self):
        """Test log creation when file doesn't exist."""
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        with patch("os.path.exists", return_value=False):
            runner = ExperimentRunner(provider_configs)

            # Should create empty log DataFrame
            assert isinstance(runner.init_log, pd.DataFrame)
            assert len(runner.init_log) == 0

    def test_load_or_create_log_existing_file(self):
        """Test log loading when file exists."""
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        mock_df = pd.DataFrame(
            [{"file_name": "test.csv", "init_response": "test response"}]
        )

        with patch("os.path.exists", return_value=True), patch(
            "pandas.read_csv", return_value=mock_df
        ):
            runner = ExperimentRunner(provider_configs)

            # Should load existing log
            assert len(runner.init_log) == 1
            assert runner.init_log.iloc[0]["file_name"] == "test.csv"

    @patch("causalign.experiment.api.client.OpenAIClient")
    def test_process_file_success(self, mock_client_class, tmp_path):
        """Test successful file processing."""
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_client.generate_response.return_value = mock_response
        mock_client.model_name = "gpt-3.5-turbo"

        # Create test CSV file
        test_data = pd.DataFrame(
            [
                {
                    "id": "test_001",
                    "prompt": "Test prompt",
                    "prompt_category": "test",
                    "graph": "A -> B",
                    "domain": "test",
                    "task": "test",
                    "cntbl_cond": "none",
                }
            ]
        )

        subfolder = "test_subfolder"
        input_file = "test_input.csv"

        # Create directory structure
        input_dir = tmp_path / subfolder
        input_dir.mkdir()
        test_csv_path = input_dir / input_file
        test_data.to_csv(test_csv_path, index=False)

        # Setup runner
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        with patch.object(
            ExperimentRunner, "_setup_results_folder"
        ) as mock_setup, patch.object(
            ExperimentRunner, "_load_or_create_log"
        ) as mock_load_log, patch("os.makedirs"):
            mock_setup.return_value = str(tmp_path / "results")
            mock_load_log.return_value = pd.DataFrame()

            runner = ExperimentRunner(
                provider_configs, input_path=str(tmp_path), n_times=1
            )

            # Mock the log response and save results methods
            with patch.object(runner, "_log_response"), patch.object(
                runner, "_save_results"
            ) as mock_save:
                runner.process_file(input_file, subfolder, mock_client, 0.7)

                # Verify methods were called
                mock_save.assert_called_once()
                mock_client.generate_response.assert_called()

    def test_log_response(self, tmp_path):
        """Test response logging functionality."""
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        with patch.object(ExperimentRunner, "_setup_results_folder"), patch.object(
            ExperimentRunner, "_load_or_create_log"
        ) as mock_load_log, patch("pandas.DataFrame.to_csv") as mock_to_csv:
            mock_load_log.return_value = pd.DataFrame()

            runner = ExperimentRunner(provider_configs)

            runner._log_response(
                input_file="test.csv",
                init_response="Test response",
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                subfolder="test",
                prompt_type="test",
            )

            # Should save log to CSV
            mock_to_csv.assert_called_once()

            # Check log entry was added
            assert len(runner.init_log) == 1
            assert runner.init_log.iloc[0]["file_name"] == "test.csv"

    def test_save_results(self, tmp_path):
        """Test results saving functionality."""
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        results = [
            {
                "id": "test_001",
                "response": "Test response",
                "prompt_category": "test",
                "graph": "A -> B",
                "domain": "test",
                "task": "test",
                "cntbl_cond": "none",
            }
        ]

        with patch.object(
            ExperimentRunner, "_setup_results_folder"
        ) as mock_setup, patch.object(ExperimentRunner, "_load_or_create_log"):
            mock_setup.return_value = str(tmp_path)

            runner = ExperimentRunner(provider_configs)

            with patch("os.makedirs"), patch("pandas.DataFrame.to_csv") as mock_to_csv:
                runner._save_results(
                    results=results,
                    cnt_cond=None,
                    model_name="gpt-3.5-turbo",
                    subfolder="test",
                    temperature=0.7,
                    input_file="test.csv",
                )

                # Should save results to CSV
                mock_to_csv.assert_called_once()

    def test_run_validation_error(self):
        """Test run method raises error for missing parameters."""
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        with patch.object(ExperimentRunner, "_setup_results_folder"), patch.object(
            ExperimentRunner, "_load_or_create_log"
        ):
            runner = ExperimentRunner(provider_configs)

            # Should raise ValueError when required parameters are missing
            with pytest.raises(
                ValueError,
                match="sub_folder_xs and temperature_value_xs must be provided",
            ):
                runner.run()

    def test_cot_flag_affects_results_folder(self):
        """Test that CoT flag affects results folder name."""
        mock_config = LLMConfig(
            provider="openai", api_key="test-key", model_name="gpt-3.5-turbo"
        )
        provider_configs = {"openai": [mock_config]}

        with patch("os.makedirs") as mock_makedirs, patch.object(
            ExperimentRunner, "_load_or_create_log"
        ):
            runner = ExperimentRunner(provider_configs, cot=True)

            # Results folder should include "_cot" suffix
            assert "_cot" in runner.results_folder
