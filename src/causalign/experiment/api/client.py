# src/causalign/experiment/api/client.py

"""
This module provides a standardized interface for interacting with various Large Language Model (LLM) providers such as OpenAI, Gemini, and Claude. It includes abstract base classes, specific client implementations for each provider, and a factory class to create the appropriate client based on the provider name.

Classes:
    LLMResponse: A dataclass to standardize the response format across different LLM providers.
    BaseLLMClient: An abstract base class defining the interface for LLM clients.
    OpenAIClient: A client implementation for interacting with OpenAI's GPT models.
    GeminiClient: A client implementation for interacting with Gemini's generative models.
    ClaudeClient: A client implementation for interacting with Claude's models.
    LLMClientFactory: A factory class to create appropriate LLM client instances based on the provider.
    LLMConfig: A configuration class to hold provider-specific settings.

Functions:
    create_llm_client: Creates an LLM client instance based on the provided configuration.
"""
import abc
from dataclasses import dataclass
from typing import Any, Optional
import os

import openai
from anthropic import Anthropic


@dataclass
class LLMResponse:
    """Standardized response format across different LLM providers"""

    content: str
    model_name: str
    raw_response: Any  # Store the original response for debugging
    # Optional metadata for models that support reasoning knobs (e.g., GPT-5 Responses API)
    reasoning_effort: Optional[str] = None  # one of {minimal, low, medium, high} or None/N/A
    verbosity: Optional[str] = None  # one of {low, medium, high} or None/N/A


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM clients"""

    @abc.abstractmethod
    def generate_response(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        """Generate a response from the LLM"""
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        # Define reasoning models that have different API requirements
        self.reasoning_models = {
            "o1-preview",
            "o1-mini",
            "o1",
            "o3-mini",
            "o3",
            "o3-high",
        }
        # Responses API models (GPT-5 family)
        self.gpt5_prefixes = ("gpt-5", "gpt-5-mini", "gpt-5-nano")
        self.is_gpt5 = model_name.lower().startswith(self.gpt5_prefixes)
        self.is_reasoning_model = any(
            reasoning_model in model_name.lower()
            for reasoning_model in self.reasoning_models
        )

    def generate_response(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        try:
            # GPT-5 uses the Responses API with optional reasoning and verbosity controls
            if self.is_gpt5:
                # Defaults: medium effort and verbosity unless configured via env vars
                effort = os.getenv("OPENAI_GPT5_EFFORT", "medium")
                verbosity = os.getenv("OPENAI_GPT5_VERBOSITY", "medium")
                # Build payload; GPT-5 may not accept 'temperature' -> omit it
                payload = {
                    "model": self.model_name,
                    "reasoning": {"effort": effort},
                    "text": {"verbosity": verbosity},
                }
                # Ensure non-empty input for initialization calls
                effective_input = prompt if (isinstance(prompt, str) and prompt.strip()) else "Hello"
                payload["input"] = effective_input
                response = self.client.responses.create(**payload)
                # Prefer the convenience text accessor if present
                content = getattr(response, "output_text", None)
                if not content:
                    # Fallback: concatenate text segments from structured output
                    try:
                        parts = []
                        for item in getattr(response, "output", []) or []:
                            for c in getattr(item, "content", []) or []:
                                txt = getattr(c, "text", None)
                                if txt:
                                    parts.append(txt)
                        content = "".join(parts) if parts else str(response)
                    except Exception:
                        content = str(response)
                return LLMResponse(
                    content=content,
                    model_name=self.model_name,
                    raw_response=response,
                    reasoning_effort=effort or "medium",
                    verbosity=verbosity or "medium",
                )
            else:
                # Chat Completions API for non-GPT-5 models
                api_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                # Reasoning (o1/o3) models often ignore temperature
                if not self.is_reasoning_model:
                    api_params["temperature"] = temperature
                response = self.client.chat.completions.create(**api_params)
                content = response.choices[0].message.content
                # Optionally include reasoning trace if available (provider-dependent)
                if self.is_reasoning_model and hasattr(
                    response.choices[0].message, "reasoning"
                ):
                    reasoning_trace = getattr(
                        response.choices[0].message, "reasoning", None
                    )
                    if reasoning_trace:
                        content = f"REASONING:\n{reasoning_trace}\n\nFINAL ANSWER:\n{content}"

                return LLMResponse(
                    content=content,
                    model_name=self.model_name,
                    raw_response=response,
                    reasoning_effort="N/A",
                    verbosity="N/A",
                )
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        # Import lazily to avoid hard dependency during type checks
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)  # type: ignore[attr-defined]
            self.model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
            self._genai_ok = True
        except Exception:
            # Defer failures to call time with a clearer error
            self.model = None
            self._genai_ok = False
        self.model_name = model_name

    def generate_response(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        try:
            if not self._genai_ok or self.model is None:
                raise RuntimeError(
                    "google.generativeai is not available or not initialized."
                )
            # Re-import types locally to appease linters
            try:
                import google.generativeai as genai  # type: ignore
                generation_config = genai.types.GenerationConfig(temperature=temperature)  # type: ignore[attr-defined]
            except Exception:
                generation_config = None
            if generation_config is not None:
                response = self.model.generate_content(  # type: ignore[call-arg]
                    prompt,
                    generation_config=generation_config,
                )
            else:
                response = self.model.generate_content(prompt)  # type: ignore[call-arg]
            return LLMResponse(
                content=getattr(response, "text", str(response)),
                model_name=self.model_name,
                raw_response=response,
            )
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,  # Added max_tokens parameter
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract first text block robustly
            content_text = None
            for block in getattr(response, "content", []) or []:
                text_val = getattr(block, "text", None)
                if text_val:
                    content_text = text_val
                    break
            if content_text is None:
                content_text = str(response)
            return LLMResponse(
                content=content_text,
                model_name=self.model_name,
                raw_response=response,
            )
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")


# class LLMClientFactory:
#     """Factory class to create appropriate LLM client based on provider"""

#     @staticmethod
#     def create_client(
#         provider: str, api_key: str, model_name: Optional[str] = None
#     ) -> BaseLLMClient:
#         if provider.lower() == "openai":
#             return OpenAIClient(api_key, model_name or "gpt-3.5-turbo")
#         elif provider.lower() == "gemini":
#             return GeminiClient(api_key, model_name or "gemini-pro")
#         elif provider.lower() == "claude":
#             return ClaudeClient(api_key, model_name or "claude-3-opus-20240229")
#         else:
#             raise ValueError(f"Unsupported LLM provider: {provider}")


class LLMClientFactory:
    """Factory class to create appropriate LLM clients for multiple models per provider"""

    @staticmethod
    def create_clients(provider: str, api_key: str, model_names: list) -> list:
        """Creates multiple LLM client instances for a given provider"""
        clients = []
        for model_name in model_names:
            if provider.lower() == "openai":
                clients.append(OpenAIClient(api_key, model_name))
            elif provider.lower() == "gemini":
                clients.append(GeminiClient(api_key, model_name))
            elif provider.lower() == "claude":
                clients.append(ClaudeClient(api_key, model_name))
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        return clients  # Return a list of clients


# Example configuration class
class LLMConfig:
    def __init__(self, provider: str, api_key: str, model_name: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name


# def create_llm_client(config: LLMConfig) -> BaseLLMClient:
#     return LLMClientFactory.create_client(
#         provider=config.provider, api_key=config.api_key, model_name=config.model_name
#     )


# supports multiple models per provider
def create_llm_clients(config: LLMConfig, model_names: list) -> list:
    """Creates multiple LLM clients for a given provider and its list of models"""
    return LLMClientFactory.create_clients(config.provider, config.api_key, model_names)
