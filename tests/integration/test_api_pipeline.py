"""
Integration tests for the full API pipeline.
"""

from unittest.mock import Mock, patch

import pandas as pd

from causalign.experiment.api.api_runner import ExperimentRunner


class TestAPIIntegration:
    """Integration tests for the full API pipeline."""

    @patch("causalign.experiment.api.llm_clients.OpenAIClient")
    def test_openai_full_pipeline(self, mock_client_class, sample_input_data, tmp_path):
        """Test full pipeline with OpenAI client."""
        # Setup mock client
        mock_client = Mock()
        mock_client.generate_response.return_value = "OpenAI test response"
        mock_client_class.return_value = mock_client

        runner = ExperimentRunner(
            model_name="gpt-3.5-turbo",
            experiment_name="test_openai",
            temperature=0.7,
            api_key="test-openai-key",
        )

        # Mock the output directory creation
        with patch.object(runner, "create_output_directory") as mock_create_dir:
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            mock_create_dir.return_value = output_dir

            # Run the experiment
            results = runner.run_experiment(sample_input_data)

        # Verify results
        assert len(results) == len(sample_input_data)
        for i, result in enumerate(results):
            assert result["response"] == "OpenAI test response"
            assert result["subject"] == sample_input_data.iloc[i]["id"]
            assert result["temperature"] == 0.7
            assert result["id"] == sample_input_data.iloc[i]["id"]

        # Verify output file was created
        output_files = list(output_dir.glob("*.csv"))
        assert len(output_files) == 1

        # Verify file contents
        df = pd.read_csv(output_files[0])
        assert len(df) == len(sample_input_data)
        assert "response" in df.columns
        assert all(df["response"] == "OpenAI test response")

    @patch("causalign.experiment.api.llm_clients.AnthropicClient")
    def test_anthropic_full_pipeline(
        self, mock_client_class, sample_input_data, tmp_path
    ):
        """Test full pipeline with Anthropic client."""
        # Setup mock client
        mock_client = Mock()
        mock_client.generate_response.return_value = "Anthropic test response"
        mock_client_class.return_value = mock_client

        runner = ExperimentRunner(
            model_name="claude-3-sonnet-20240229",
            experiment_name="test_anthropic",
            temperature=0.5,
            api_key="test-anthropic-key",
        )

        # Mock the output directory creation
        with patch.object(runner, "create_output_directory") as mock_create_dir:
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            mock_create_dir.return_value = output_dir

            # Run the experiment
            results = runner.run_experiment(sample_input_data)

        # Verify results
        assert len(results) == len(sample_input_data)
        for result in results:
            assert result["response"] == "Anthropic test response"
            assert result["temperature"] == 0.5

        # Verify output file was created and has correct content
        output_files = list(output_dir.glob("*.csv"))
        assert len(output_files) == 1

        df = pd.read_csv(output_files[0])
        assert all(df["response"] == "Anthropic test response")

    @patch("causalign.experiment.api.llm_clients.GeminiClient")
    def test_gemini_full_pipeline(self, mock_client_class, sample_input_data, tmp_path):
        """Test full pipeline with Gemini client."""
        # Setup mock client
        mock_client = Mock()
        mock_client.generate_response.return_value = "Gemini test response"
        mock_client_class.return_value = mock_client

        runner = ExperimentRunner(
            model_name="gemini-pro",
            experiment_name="test_gemini",
            temperature=0.9,
            api_key="test-google-key",
        )

        # Mock the output directory creation
        with patch.object(runner, "create_output_directory") as mock_create_dir:
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            mock_create_dir.return_value = output_dir

            # Run the experiment
            results = runner.run_experiment(sample_input_data)

        # Verify results
        assert len(results) == len(sample_input_data)
        for result in results:
            assert result["response"] == "Gemini test response"
            assert result["temperature"] == 0.9

    @patch("causalign.experiment.api.llm_clients.OpenAIClient")
    def test_pipeline_error_handling(
        self, mock_client_class, sample_input_data, tmp_path
    ):
        """Test pipeline handles API errors gracefully."""
        # Setup mock client that fails on second call
        mock_client = Mock()
        mock_client.generate_response.side_effect = [
            "Success response",
            Exception("API rate limit exceeded"),
            "Another success response",
        ]
        mock_client_class.return_value = mock_client

        runner = ExperimentRunner(
            model_name="gpt-4", experiment_name="test_errors", api_key="test-key"
        )

        # Mock the output directory creation
        with patch.object(runner, "create_output_directory") as mock_create_dir:
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            mock_create_dir.return_value = output_dir

            # Run the experiment
            results = runner.run_experiment(sample_input_data)

        # Verify error handling
        assert len(results) == len(sample_input_data)
        assert results[0]["response"] == "Success response"
        assert "ERROR:" in results[1]["response"]
        assert "API rate limit exceeded" in results[1]["response"]
        # Note: Third result might not exist if we only have 2 input rows

    def test_csv_input_output_format_consistency(self, tmp_path):
        """Test that CSV input/output format is consistent."""
        # Create test input CSV
        input_data = pd.DataFrame(
            [
                {
                    "id": "format_test_001",
                    "prompt": "Does A cause B?",
                    "prompt_category": "basic_causal",
                    "graph": "A -> B",
                    "domain": "test",
                    "cntbl_cond": "none",
                    "task": "causal_reasoning",
                }
            ]
        )

        input_file = tmp_path / "input_test.csv"
        input_data.to_csv(input_file, index=False)

        # Mock the API client
        with patch(
            "causalign.experiment.api.llm_clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.generate_response.return_value = "Format test response"
            mock_client_class.return_value = mock_client

            runner = ExperimentRunner(
                model_name="gpt-3.5-turbo",
                experiment_name="format_test",
                api_key="test-key",
            )

            # Mock the output directory creation
            with patch.object(runner, "create_output_directory") as mock_create_dir:
                output_dir = tmp_path / "output"
                output_dir.mkdir()
                mock_create_dir.return_value = output_dir

                # Load data and run experiment
                input_df = pd.read_csv(input_file)
                results = runner.run_experiment(input_df)

        # Verify output file format
        output_files = list(output_dir.glob("*.csv"))
        assert len(output_files) == 1

        output_df = pd.read_csv(output_files[0])

        # Check all required columns are present
        required_columns = [
            "id",
            "prompt",
            "prompt_category",
            "graph",
            "domain",
            "cntbl_cond",
            "task",
            "response",
            "subject",
            "temperature",
        ]

        for col in required_columns:
            assert col in output_df.columns, f"Missing column: {col}"

        # Check data integrity
        assert output_df.iloc[0]["id"] == "format_test_001"
        assert output_df.iloc[0]["response"] == "Format test response"
        assert output_df.iloc[0]["subject"] == "format_test_001"

    @patch("time.sleep")  # Speed up tests by mocking delays
    def test_rate_limiting_behavior(self, mock_sleep, sample_input_data, tmp_path):
        """Test that rate limiting delays are applied."""
        with patch(
            "causalign.experiment.api.llm_clients.OpenAIClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.generate_response.return_value = "Rate limit test"
            mock_client_class.return_value = mock_client

            runner = ExperimentRunner(
                model_name="gpt-3.5-turbo",
                experiment_name="rate_test",
                api_key="test-key",
                delay=2.0,  # 2 second delay
            )

            # Mock the output directory creation
            with patch.object(runner, "create_output_directory") as mock_create_dir:
                output_dir = tmp_path / "output"
                output_dir.mkdir()
                mock_create_dir.return_value = output_dir

                # Run experiment
                results = runner.run_experiment(sample_input_data)

            # Verify sleep was called for rate limiting
            # Should be called len(sample_input_data) - 1 times (no delay after last request)
            expected_calls = len(sample_input_data) - 1
            assert mock_sleep.call_count == expected_calls

            # Verify delay amount
            if expected_calls > 0:
                mock_sleep.assert_called_with(2.0)
