"""Backward compatibility shim for test imports.

Re-exports client implementations so tests referencing
`causalign.experiment.api.llm_clients.OpenAIClient` succeed.
"""

from .client import (
    OpenAIClient,
    GeminiClient,
    ClaudeClient,
    LLMConfig,
    create_llm_clients,
)

# Backwards compatibility: some tests or external code may reference AnthropicClient
AnthropicClient = ClaudeClient  # alias

__all__ = [
    "OpenAIClient",
    "GeminiClient",
    "ClaudeClient",
    "LLMConfig",
    "create_llm_clients",
    "AnthropicClient",
]
