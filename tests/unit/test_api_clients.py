"""
Unit tests for LLM API clients.
"""

from unittest.mock import Mock, patch

import pytest

from causalign.experiment.api.client import (
    ClaudeClient,  # Note: actual class name is ClaudeClient, not AnthropicClient
    GeminiClient,
    OpenAIClient,
)


class TestOpenAIClient:
    """Test OpenAI client functionality."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = OpenAIClient(api_key="test-key")
        assert client.model_name == "gpt-3.5-turbo"  # default model
        assert client.client is not None

    def test_init_with_custom_model(self):
        """Test client initialization with custom model."""
        client = OpenAIClient(api_key="test-key", model_name="gpt-4")
        assert client.model_name == "gpt-4"

    @patch("openai.OpenAI")
    def test_generate_response_success(self, mock_openai_class, mock_openai_response):
        """Test successful response generation."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client

        client = OpenAIClient(api_key="test-key")
        response = client.generate_response(prompt="Test prompt", temperature=0.7)

        assert (
            response.content
            == "Yes, smoking is a well-established cause of lung cancer."
        )
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_generate_response_api_error(self, mock_openai_class):
        """Test API error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        client = OpenAIClient(api_key="test-key")

        with pytest.raises(Exception, match="OpenAI API error"):
            client.generate_response(prompt="Test prompt")


class TestClaudeClient:
    """Test Claude client functionality."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = ClaudeClient(api_key="test-key")
        assert client.model_name == "claude-3-opus-20240229"  # default model
        assert client.client is not None

    def test_init_with_custom_model(self):
        """Test client initialization with custom model."""
        client = ClaudeClient(api_key="test-key", model_name="claude-3-sonnet-20240229")
        assert client.model_name == "claude-3-sonnet-20240229"

    @patch("causalign.experiment.api.client.Anthropic")
    def test_generate_response_success(
        self, mock_anthropic_class, mock_anthropic_response
    ):
        """Test successful response generation."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient(api_key="test-key")
        response = client.generate_response(prompt="Test prompt", temperature=0.7)

        assert (
            response.content
            == "Yes, smoking is a well-established cause of lung cancer."
        )
        mock_client.messages.create.assert_called_once()

    @patch("causalign.experiment.api.client.Anthropic")
    def test_generate_response_api_error(self, mock_anthropic_class):
        """Test API error handling."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient(api_key="test-key")

        with pytest.raises(Exception, match="Claude API error"):
            client.generate_response(prompt="Test prompt")


class TestGeminiClient:
    """Test Gemini client functionality."""

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_init_with_api_key(self, mock_model_class, mock_configure):
        """Test client initialization with API key."""
        client = GeminiClient(api_key="test-key")
        assert client.model_name == "gemini-pro"  # default model
        mock_configure.assert_called_once_with(api_key="test-key")

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_init_with_custom_model(self, mock_model_class, mock_configure):
        """Test client initialization with custom model."""
        client = GeminiClient(api_key="test-key", model_name="gemini-1.5-pro")
        assert client.model_name == "gemini-1.5-pro"

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_generate_response_success(
        self, mock_model_class, mock_configure, mock_gemini_response
    ):
        """Test successful response generation."""
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model

        client = GeminiClient(api_key="test-key")
        response = client.generate_response(prompt="Test prompt", temperature=0.7)

        assert (
            response.content
            == "Yes, smoking is a well-established cause of lung cancer."
        )
        mock_model.generate_content.assert_called_once()

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_generate_response_api_error(self, mock_model_class, mock_configure):
        """Test API error handling."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model

        client = GeminiClient(api_key="test-key")

        with pytest.raises(Exception, match="Gemini API error"):
            client.generate_response(prompt="Test prompt")
