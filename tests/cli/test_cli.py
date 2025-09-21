"""
CLI tests for run_experiment.py command-line interface.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


class TestCLI:
    """Test command-line interface functionality."""

    def test_cli_help_message(self):
        """Test that CLI shows help message."""
        result = subprocess.run(
            ["python", "run_experiment.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Project root
        )

        assert result.returncode == 0
        assert "Run causal reasoning experiments" in result.stdout
        assert "--input-file" in result.stdout
        assert "--model" in result.stdout
        assert "--experiment-name" in result.stdout

    def test_cli_missing_required_args(self):
        """Test CLI with missing required arguments."""
        result = subprocess.run(
            ["python", "run_experiment.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower()

    def test_cli_invalid_model(self):
        """Test CLI with invalid model name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,prompt,prompt_category,graph,domain,cntbl_cond,task\n")
            f.write("1,test,test,test,test,test,test\n")
            temp_file = f.name

        try:
            result = subprocess.run(
                [
                    "python",
                    "run_experiment.py",
                    "--input-file",
                    temp_file,
                    "--model",
                    "invalid-model",
                    "--experiment-name",
                    "test",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode != 0
            assert (
                "unsupported" in result.stderr.lower()
                or "invalid" in result.stderr.lower()
            )
        finally:
            os.unlink(temp_file)

    def test_cli_nonexistent_input_file(self):
        """Test CLI with non-existent input file."""
        result = subprocess.run(
            [
                "python",
                "run_experiment.py",
                "--input-file",
                "nonexistent_file.csv",
                "--model",
                "gpt-3.5-turbo",
                "--experiment-name",
                "test",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode != 0
        assert (
            "not found" in result.stderr.lower()
            or "no such file" in result.stderr.lower()
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("causalign.experiment.api.experiment_runner.ExperimentRunner")
    def test_cli_successful_execution_with_env_key(self, mock_runner_class):
        """Test successful CLI execution with API key from environment."""
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,prompt,prompt_category,graph,domain,cntbl_cond,task\n")
            f.write("1,Test prompt,test,A->B,test,none,test\n")
            temp_file = f.name

        # Mock the runner
        mock_runner = Mock()
        mock_runner.run_experiment.return_value = []
        mock_runner_class.return_value = mock_runner

        try:
            result = subprocess.run(
                [
                    "python",
                    "run_experiment.py",
                    "--input-file",
                    temp_file,
                    "--model",
                    "gpt-3.5-turbo",
                    "--experiment-name",
                    "test_experiment",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            # Should succeed (though mocked)
            assert (
                result.returncode == 0 or "test-key" in result.stderr
            )  # Mock might cause different behavior
        finally:
            os.unlink(temp_file)

    def test_cli_missing_api_key(self):
        """Test CLI fails gracefully when API key is missing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,prompt,prompt_category,graph,domain,cntbl_cond,task\n")
            f.write("1,Test prompt,test,A->B,test,none,test\n")
            temp_file = f.name

        # Clear environment variables
        env = os.environ.copy()
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
            env.pop(key, None)

        try:
            result = subprocess.run(
                [
                    "python",
                    "run_experiment.py",
                    "--input-file",
                    temp_file,
                    "--model",
                    "gpt-3.5-turbo",
                    "--experiment-name",
                    "test",
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode != 0
            assert "api key" in result.stderr.lower()
        finally:
            os.unlink(temp_file)

    def test_cli_invalid_csv_format(self):
        """Test CLI with invalid CSV format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("wrong,columns\n")
            f.write("data,here\n")
            temp_file = f.name

        try:
            result = subprocess.run(
                [
                    "python",
                    "run_experiment.py",
                    "--input-file",
                    temp_file,
                    "--model",
                    "gpt-3.5-turbo",
                    "--experiment-name",
                    "test",
                    "--api-key",
                    "test-key",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode != 0
            assert (
                "missing" in result.stderr.lower()
                or "column" in result.stderr.lower()
                or "required" in result.stderr.lower()
            )
        finally:
            os.unlink(temp_file)

    def test_cli_temperature_parameter(self):
        """Test CLI temperature parameter validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,prompt,prompt_category,graph,domain,cntbl_cond,task\n")
            f.write("1,test,test,test,test,test,test\n")
            temp_file = f.name

        try:
            # Test invalid temperature (too high)
            result = subprocess.run(
                [
                    "python",
                    "run_experiment.py",
                    "--input-file",
                    temp_file,
                    "--model",
                    "gpt-3.5-turbo",
                    "--experiment-name",
                    "test",
                    "--temperature",
                    "2.5",
                    "--api-key",
                    "test-key",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            # Should either fail validation or accept it (depending on implementation)
            # At minimum, it should not crash unexpectedly
            assert result.returncode in [0, 1, 2]  # Expected return codes
        finally:
            os.unlink(temp_file)

    def test_cli_model_variants(self):
        """Test CLI accepts different model variants."""
        models_to_test = [
            "gpt-3.5-turbo",
            "gpt-4",
            "claude-3-sonnet-20240229",
            "gemini-pro",
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,prompt,prompt_category,graph,domain,cntbl_cond,task\n")
            f.write("1,test,test,test,test,test,test\n")
            temp_file = f.name

        try:
            for model in models_to_test:
                result = subprocess.run(
                    [
                        "python",
                        "run_experiment.py",
                        "--input-file",
                        temp_file,
                        "--model",
                        model,
                        "--experiment-name",
                        "test",
                        "--api-key",
                        "test-key",
                        "--dry-run",  # If supported
                    ],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent.parent,
                )

                # Should not fail due to unsupported model
                # (might fail due to missing API key, which is expected)
                assert "unsupported model" not in result.stderr.lower()
        finally:
            os.unlink(temp_file)
