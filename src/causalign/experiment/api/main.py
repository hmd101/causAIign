import os

"""
This script sets up and runs an experiment using various language model providers.

It loads API keys from a .env file, configures the providers (OpenAI, Gemini, and Claude) 
based on the available keys, and then runs the experiment using the specified configurations.

Classes:
    ExperimentRunner: Manages the execution of the experiment.
    LLMConfig: Holds configuration details for each language model provider.

Functions:
    load_dotenv: Loads environment variables from a .env file.

Usage:
    Ensure that the .env file contains the necessary API keys for the desired providers.
    Run the script to execute the experiment with the configured providers.
"""

from dotenv import load_dotenv

# New imports
from causalign.experiment.api.api_runner import ExperimentRunner
from causalign.experiment.api.client import LLMConfig

# Old imports
# from analysis.causalign.src.causalign.experiments.call_apis import ExperimentRunner
# from causalign.experiments.call_apis import ExperimentRunner
# from causalign.llm.client import LLMConfig

# from causalign.experiments.call_apis import ExperimentRunner
# from causalign.llm.client import LLMConfig


# Load environment variables from .env file
load_dotenv()


#####
# multiple models per api provider
if __name__ == "__main__":
    import os

    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    # Configure providers with multiple models
    provider_configs = {}

    # Define model lists for each provider
    openai_models = [
        "gpt-4o",  # uncomment!
        "gpt-3.5-turbo",
    ]  # "gpt-4.5"  Error code: 404
    google_models = [
        "gemini-1.5-pro",
        # "gemini-2.0-pro-exp-02-05",  # uncomment!
    ]
    anthropic_models = [
        # "claude-3-sonnet", # Claude API error: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'messages.0: all messages must have non-empty content except for the optional final assistant message'}}
        # "claude-3-haiku",  # Claude API error: Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-3-haiku'}}
        "claude-3-opus-20240229",
    ]

    # Add OpenAI if key exists
    if openai_key:
        provider_configs["openai"] = [
            LLMConfig(provider="openai", api_key=openai_key, model_name=model)
            for model in openai_models
        ]

    # Add Gemini if key exists
    if google_key:
        provider_configs["gemini"] = [
            LLMConfig(provider="gemini", api_key=google_key, model_name=model)
            for model in google_models
        ]

    # Add Claude if key exists
    if anthropic_key:
        provider_configs["claude"] = [
            LLMConfig(provider="claude", api_key=anthropic_key, model_name=model)
            for model in anthropic_models
        ]

    if not provider_configs:
        raise ValueError(
            "No API keys found. Please set at least one API key in .env file"
        )

    # Attention: Do not flatten provider_configs
    runner = ExperimentRunner(
        provider_configs=provider_configs, version="6_v", cot=False, n_times=1
    )

    runner.run(
        input_path="datasets/17_rw",
        # input_path="datasets/abstract_collider_prompts",
        output_path="../../results/17_rw",
        sub_folder_xs=["prompts_for_LLM_api"],
        temperature_value_xs=[0.0],  # 1.0
    )
