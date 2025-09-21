"""
Pytest configuration and shared fixtures for causalign tests.
"""

import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest


@pytest.fixture
def sample_input_data():
    """Sample input data for testing."""
    return pd.DataFrame(
        [
            {
                "id": "test_001",
                "prompt": "Does smoking cause lung cancer?",
                "prompt_category": "basic_causal",
                "graph": "smoking -> lung_cancer",
                "domain": "health",
                "cntbl_cond": "none",
                "task": "causal_reasoning",
            },
            {
                "id": "test_002",
                "prompt": "What is the effect of education on income?",
                "prompt_category": "economic_causal",
                "graph": "education -> income",
                "domain": "economics",
                "cntbl_cond": "controlling_for_age",
                "task": "causal_reasoning",
            },
        ]
    )


@pytest.fixture
def sample_csv_file(sample_input_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_input.csv"
    sample_input_data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response object."""
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "Yes, smoking is a well-established cause of lung cancer."
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 25
    return mock_response


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response object."""
    mock_response = Mock()
    mock_content = Mock()
    mock_content.text = "Yes, smoking is a well-established cause of lung cancer."
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response object."""
    mock_response = Mock()
    mock_response.text = "Yes, smoking is a well-established cause of lung cancer."
    return mock_response


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for API keys."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GOOGLE_API_KEY": "test-google-key",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    return {
        "experiment_name": "test_experiment",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 500,
        "delay": 1.0,
    }


@pytest.fixture
def expected_output_columns():
    """Expected columns in the output CSV."""
    return [
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
