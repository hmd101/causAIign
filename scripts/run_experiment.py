#!/usr/bin/env python3
"""
Causal Alignment Experiment Runner

A user-friendly command-line tool for running causal reasoning experiments with LLMs.

Usage:
    # Auto-discovery (recommended)
    python run_experiment.py --version 10 --experiment pilot_study --model gpt-4o --temperature 0.0
    
    # Direct dataset path (legacy)
    python run_experiment.py --dataset data/input_llm/rw17/pilot_study/10_v_numeric_LLM_prompting_collider.csv \
                            --model gpt-4o --experiment pilot_study --temperature 0.0

Environment Variables:
    Set API keys in .env file or environment:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY  
    - GOOGLE_API_KEY

Output Structure:
    data/output_llm/experiment-name/model-name/{version}_{prompt_category}_{experiment}_{model}_{temperature}_temp.csv
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the experiment components
from src.causalign.experiment.api.api_runner import ExperimentRunner  # noqa: E402
from src.causalign.experiment.api.client import LLMConfig  # noqa: E402

# Load environment variables
load_dotenv()


class CausalExperimentCLI:
    """User-friendly CLI for causal alignment experiments."""

    SUPPORTED_MODELS = {
        # OpenAI models
        "gpt-4o": "openai",
        "gpt-4.1": "openai",
        "gpt-4.1-mini": "openai",
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
    # OpenAI GPT-5 (Responses API)
    "gpt-5": "openai",
    "gpt-5-mini": "openai",
    "gpt-5-nano": "openai",
        # OpenAI reasoning models (no temperature support)
        "o1-mini": "openai",
        "o1": "openai",
        "o3-mini": "openai",
        "o3": "openai",
        # Anthropic models
        # "claude-3-opus-20240229": "claude",
        # "claude-opus-4-1-20250805": "claude",
        # "claude-3-haiku-20240307": "claude",
        # "claude-opus-4-20250514": "claude",
        # "claude-sonnet-4-20250514": "claude",
        # "claude-3-7-sonnet-20250219": "claude",
        # "claude-3-5-haiku-20241022": "claude",
        "claude-3-opus-20240229": "claude",
        "claude-opus-4-20250514": "claude",
        "claude-opus-4-1-20250805": "claude",
        "claude-3-haiku-20240307": "claude",
        "claude-sonnet-4-20250514": "claude",
        "claude-3-7-sonnet-20250219": "claude",
        "claude-3-5-haiku-20241022": "claude",
        # Google models
        "gemini-1.5-pro": "gemini",
        "gemini-2.5-pro": "gemini",  # note: thinking on by default!
        "gemini-2.5-flash": "gemini",  # note: thinking on by default!
        "gemini-2.5-flash-lite": "gemini",  #
    }

    def __init__(self):
        self.args = self._parse_arguments()
        self._validate_inputs()

    def _parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Run causal reasoning experiments with LLMs",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Auto-discovery using version + experiment (recommended)
  python run_experiment.py --version 10 --experiment pilot_study --model gpt-4o

  # Auto-discovery with custom settings
  python run_experiment.py --version 9 --experiment xml_format_study --model claude-3-opus-20240229 --temperature 0.7

  # Using reasoning models (note: temperature is ignored)
  python run_experiment.py --version 10 --experiment pilot_study --model o1-preview --temperature 0.0
  python run_experiment.py --version 10 --experiment pilot_study --model o3-mini

  # Multiple models including reasoning models
  python run_experiment.py --version 10 --experiment pilot_study --model gpt-4o o1-mini --temperature 0.0

  # Direct dataset path (legacy method)
  python run_experiment.py --dataset data/input_llm/rw17/pilot_study/10_v_numeric_LLM_prompting_collider.csv --model gpt-4o --experiment pilot_study

  # Custom base directory
  python run_experiment.py --version 8 --experiment pilot_study --model gpt-4o --base-dir /custom/path/input_llm/rw17

Available models: {models}

Output filename format: {{version}}_{{prompt_category}}_{{experiment}}_{{model}}_{{temperature}}_temp.csv
            """.format(models=", ".join(self.SUPPORTED_MODELS.keys())),
        )

        parser.add_argument(
            "--dataset",
            "-d",
            help="Path to input CSV file (alternative to using version + experiment auto-discovery)",
        )

        parser.add_argument(
            "--version",
            "-V",
            type=int,
            help="Version number (0-1000) to auto-discover input files",
        )

        parser.add_argument(
            "--base-dir",
            default="data/input_llm/rw17",
            help="Base directory for auto-discovery (default: data/input_llm/rw17)",
        )
        # pass multiple models like this: --model gpt-4o claude-3-opus-20240229
        parser.add_argument(
            "--model",
            "-m",
            nargs="+",
            choices=list(self.SUPPORTED_MODELS.keys()) + ["all"],
            help=(
                "LLM model(s) to use (overrides experiment config). "
                "Use 'all' to run every supported model."
            ),
        )

        parser.add_argument(
            "--experiment",
            "-e",
            required=True,
            help="Experiment name (used for output folder and auto-discovery)",
        )

        parser.add_argument(
            "--temperature",
            "-t",
            type=float,
            default=0.0,
            help="Temperature for LLM generation (default: 0.0)",
        )

        parser.add_argument(
            "--api-key", help="API key (overrides environment variables)"
        )

        parser.add_argument(
            "--output-dir",
            default="data/output_llm",
            help="Base output directory (default: data/output_llm)",
        )

        parser.add_argument(
            "--runs",
            "-n",
            type=int,
            default=1,
            help="Number of times to repeat each prompt (default: 1)",
        )

        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output"
        )

        # GPT-5 specific knobs (Responses API). If provided, forwarded via env vars.
        parser.add_argument(
            "--gpt5-effort",
            nargs="+",
            choices=["minimal", "low", "medium", "high"],
            help=(
                "GPT-5 reasoning effort (Responses API). "
                "Pass one or more values to run all combinations with --gpt5-verbosity."
            ),
        )
        parser.add_argument(
            "--gpt5-verbosity",
            nargs="+",
            choices=["low", "medium", "high"],
            help=(
                "GPT-5 output verbosity (Responses API). "
                "Pass one or more values to run all combinations with --gpt5-effort."
            ),
        )
        parser.add_argument(
            "--list-gpt5-options",
            action="store_true",
            help="Print available GPT-5 effort/verbosity options and exit",
        )

        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List all supported model IDs and exit",
        )

        return parser.parse_args()

    def _validate_inputs(self):
        """Validate input arguments and setup."""
        # Validate that either dataset OR version is provided
        if not self.args.dataset and self.args.version is None:
            print("❌ Error: Must provide either --dataset or --version")
            print(
                "Use --dataset for direct file path, or --version + --experiment for auto-discovery"
            )
            sys.exit(1)

        if self.args.dataset and self.args.version is not None:
            print(
                "❌ Error: Cannot use both --dataset and --version. Choose one approach."
            )
            sys.exit(1)

        # Validate version range if provided
        if self.args.version is not None and not (0 <= self.args.version <= 1000):
            print("❌ Error: Version must be between 0 and 1000")
            sys.exit(1)

        # Optionally list GPT-5 options for user discovery
        if self.args.list_gpt5_options:
            print("GPT-5 Responses API options:")
            print("  reasoning_effort: minimal | low | medium | high (default: medium)")
            print("  verbosity: low | medium | high (default: medium)")
            sys.exit(0)

        # Optionally list supported models
        if getattr(self.args, "list_models", False):
            print("Supported models:")
            for m in sorted(self.SUPPORTED_MODELS.keys()):
                print(f"  - {m} ({self.SUPPORTED_MODELS[m]})")
            sys.exit(0)

        # Auto-discover input files or validate provided dataset
        if self.args.version is not None:
            self.input_files = self._discover_input_files()
        else:
            # Validate provided dataset file
            if not Path(self.args.dataset).exists():
                print(f"❌ Error: Dataset file not found: {self.args.dataset}")
                sys.exit(1)
            self.input_files = [Path(self.args.dataset)]

        # Determine which models to use
        self.models = self._determine_models()

        # Check for reasoning models and temperature conflicts
        self._validate_reasoning_models()

        # Validate that all models have API keys
        self._validate_api_keys()

    # Note: GPT-5 effort/verbosity env vars are set per combination in run_experiment()

        # Create output directory structure (no temperature folder)
        # Use the first model for the output directory name, or a generic name if multiple models
        if self.models and len(self.models) == 1:
            model_dir = self.models[0]
        else:
            model_dir = "multiple_models"

        self.output_path = Path(self.args.output_dir) / self.args.experiment / model_dir
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.args.verbose:
            print(f"📁 Output directory: {self.output_path}")

    def _discover_input_files(self):
        """Auto-discover input files based on version and experiment name."""
        base_dir = Path(self.args.base_dir)
        experiment_dir = base_dir / self.args.experiment

        if not experiment_dir.exists():
            print(f"❌ Error: Experiment directory not found: {experiment_dir}")
            print("💡 Suggestions:")
            print("   1. Use --dataset to provide direct file path")
            print("   2. Update --base-dir to correct location")
            print(f"   3. Check experiment name: {self.args.experiment}")
            sys.exit(1)

        # Look for files starting with version number
        pattern = f"{self.args.version}_v_*.csv"
        matching_files = list(experiment_dir.glob(pattern))

        if not matching_files:
            print(
                f"❌ Error: No files found matching pattern '{pattern}' in {experiment_dir}"
            )
            print("💡 Suggestions:")
            print(f"   1. Check available files: ls {experiment_dir}")
            print("   2. Use --dataset to provide direct file path")
            print("   3. Update --base-dir if files are in different location")
            print(f"   4. Verify version number: {self.args.version}")
            sys.exit(1)

        if self.args.verbose:
            print(f"🔍 Found {len(matching_files)} matching files:")
            for file in matching_files:
                print(f"   - {file.name}")

        return matching_files

    def _rename_output_files(self):
        """Rename output files to match desired format: {version}_{prompt_category}_{experiment}_{model}_{temperature}_temp.csv"""
        import re
        
        # Determine which directories to scan for outputs:
        # - If a single model is selected, files live in self.output_path (…/<experiment>/<model>/)
        # - If multiple models, ExperimentRunner places files in per-model subfolders under
        #   self.output_path.parent (…/<experiment>/<model>/). Scan those model folders.
        if self.models and len(self.models) == 1:
            dirs_to_scan = [self.output_path]
        else:
            base_parent = self.output_path.parent
            dirs_to_scan = [(base_parent / m) for m in self.models]

        pattern = r"^(\d+)_v_(.+?)_(.+?)_(\d+\.?\d*)_temp\.csv$"

        for out_dir in dirs_to_scan:
            if not out_dir.exists():
                continue

            for old_file in out_dir.glob("*.csv"):
                old_name = old_file.name

                match = re.match(pattern, old_name)
                if not match:
                    if self.args.verbose:
                        print(f"⚠️  Could not parse filename format: {old_name}")
                    continue

                version, prompt_category, model, temperature = match.groups()

                # Optional GPT-5 verbosity/effort suffix to avoid overwrites across runs
                gpt5_suffix = ""
                if model.startswith("gpt-5"):
                    v = getattr(self.args, "gpt5_verbosity", None)
                    e = getattr(self.args, "gpt5_effort", None)
                    if v:
                        gpt5_suffix += f"_v_{v}"
                    if e:
                        gpt5_suffix += f"_e_{e}" # reasoning efffort

                # Create new filename: {version}_{prompt_category}_{experiment}_{model}_{temperature}[gpt5_suffix]_temp.csv
                new_name = (
                    f"{version}_{prompt_category}_{self.args.experiment}_{model}_{temperature}{gpt5_suffix}_temp.csv"
                )
                new_file = old_file.parent / new_name

                # Skip if already correctly named
                if old_file.name == new_name:
                    continue

                # Rename the file
                old_file.rename(new_file)

                if self.args.verbose:
                    print(f"📝 Renamed: {old_name} → {new_name}")

    def _determine_models(self):
        """Determine which models to use based on experiment config and user input."""
        from causalign.experiment.config.experiment_config import (
            get_experiment_config,
        )

        # Check if experiment exists in config
        try:
            experiment_config = get_experiment_config(self.args.experiment)
            config_models = experiment_config.get_models()
            print(f"📋 Found experiment config: {experiment_config.name}")
            print(f"📋 Config models: {config_models}")
        except ValueError:
            # Experiment not found in config, require user to specify models
            if not self.args.model:
                print(
                    "❌ Error: Experiment not found in config and no models specified"
                )
                print("💡 Use --model to specify which model(s) to use")
                sys.exit(1)
            config_models = []

        # If user specified models, use those (override config)
        if self.args.model:
            user_models = self.args.model
            # Expand special token 'all' (case-insensitive) to all supported models
            if any(m.lower() == "all" for m in user_models):
                expanded = list(self.SUPPORTED_MODELS.keys())
                print("🎯 User specified models: all")
                print("✅ Expanding 'all' to all supported models:")
                print(f"   {expanded}")
                print("⚠️  Warning: User models override experiment config")
                return expanded
            print(f"🎯 User specified models: {user_models}")
            print("⚠️  Warning: User models override experiment config")
            return user_models

        # Otherwise use models from config
        if config_models:
            print(f"✅ Using models from experiment config: {config_models}")
            return config_models

        # Fallback: require user to specify models
        print("❌ Error: No models specified and experiment config has no models")
        print("💡 Use --model to specify which model(s) to use")
        sys.exit(1)

    def _validate_reasoning_models(self):
        """Validate reasoning model usage and warn about temperature."""
        reasoning_models = {"o1-preview", "o1-mini", "o1", "o3-mini", "o3", "o3-high"}

        used_reasoning_models = [
            model
            for model in self.models
            if any(
                reasoning_model in model.lower() for reasoning_model in reasoning_models
            )
        ]

        if used_reasoning_models and self.args.temperature != 0.0:
            print("⚠️  Warning: Reasoning models don't support temperature parameter!")
            print(f"   Reasoning models detected: {used_reasoning_models}")
            print(
                f"   Temperature {self.args.temperature} will be ignored for these models"
            )
            print("   💡 Tip: Use --temperature 0.0 to avoid this warning")

    def _validate_api_keys(self):
        """Validate that API keys are available for all specified models."""
        missing_keys = []

        for model in self.models:
            provider = self.SUPPORTED_MODELS[model]
            api_key = self._get_api_key_for_provider(provider)

            if not api_key:
                missing_keys.append((model, provider))

        if missing_keys:
            print("❌ Error: Missing API keys for the following models:")
            for model, provider in missing_keys:
                env_var_map = {
                    "openai": "OPENAI_API_KEY",
                    "claude": "ANTHROPIC_API_KEY",
                    "gemini": "GOOGLE_API_KEY",
                }
                env_var = env_var_map.get(provider, f"{provider.upper()}_API_KEY")
                print(f"   - {model} (requires {env_var})")
            print("💡 Set the required API keys in .env file or use --api-key")
            sys.exit(1)

    def _get_api_key_for_provider(self, provider):
        """Get API key for a specific provider."""
        if self.args.api_key:
            return self.args.api_key

        # Map provider to environment variable
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",  # Claude models use Anthropic API key
            "gemini": "GOOGLE_API_KEY",
        }
        env_var = env_var_map.get(provider)
        return os.getenv(env_var) if env_var else None


    def run_experiment(self):
        """Run the causal alignment experiment."""
        print("🚀 Starting Causal Alignment Experiment")
        if self.args.version is not None:
            print(f"🔢 Version: {self.args.version}")
            print(f"📂 Base directory: {self.args.base_dir}")
            print(f"📊 Found {len(self.input_files)} input files")
        else:
            print(f"📊 Dataset: {self.args.dataset}")
        print(f"🤖 Models: {self.models}")
        print(f"🔬 Experiment: {self.args.experiment}")
        print(f"🌡️  Temperature: {self.args.temperature}")
        print(f"📁 Output: {self.output_path}")
        print("-" * 50)

        # Setup experiment configuration for each model
        provider_configs = {}
        for model in self.models:
            provider = self.SUPPORTED_MODELS[model]
            api_key = self._get_api_key_for_provider(provider)
            # _validate_api_keys ensures api_key is present for all models
            assert api_key is not None
            config = LLMConfig(provider=provider, api_key=api_key, model_name=model)

            if provider not in provider_configs:
                provider_configs[provider] = []
            provider_configs[provider].append(config)

        # Create experiment runner with updated filename logic
        runner = ExperimentRunner(
            provider_configs=provider_configs,
            version=str(self.args.version)
            if self.args.version is not None
            else "custom",
            cot=False,
            n_times=self.args.runs,
        )

        # Process each input file for all combinations of GPT-5 knobs
        import shutil
        import tempfile

        # Build combinations; if not provided, run once with None
        efforts = self.args.gpt5_effort if self.args.gpt5_effort else [None]
        verbosities = self.args.gpt5_verbosity if self.args.gpt5_verbosity else [None]

        # If any GPT-5 model is present, summarize planned parameter sweep
        has_gpt5 = any(m.startswith("gpt-5") for m in self.models)
        if has_gpt5:
            eff_vals = [e if e is not None else "default" for e in efforts]
            verb_vals = [v if v is not None else "default" for v in verbosities]
            total = len(efforts) * len(verbosities)
            print("🧪 GPT-5 parameter sweep:")
            print(f"   efforts: {eff_vals}")
            print(f"   verbosity: {verb_vals}")
            print(f"   total combinations: {total}")

        original_efforts = self.args.gpt5_effort
        original_verbosities = self.args.gpt5_verbosity

        try:
            for effort in efforts:
                for verbosity in verbosities:
                    if has_gpt5:
                        print(
                            f"➡️  Running GPT-5 combo: effort={effort or 'default'}, verbosity={verbosity or 'default'}"
                        )
                    # Set per-combination env vars for GPT-5 clients
                    if effort is not None:
                        os.environ["OPENAI_GPT5_EFFORT"] = effort
                    else:
                        os.environ.pop("OPENAI_GPT5_EFFORT", None)
                    if verbosity is not None:
                        os.environ["OPENAI_GPT5_VERBOSITY"] = verbosity
                    else:
                        os.environ.pop("OPENAI_GPT5_VERBOSITY", None)

                    # Temporarily set args to current scalar values for renaming suffix
                    self.args.gpt5_effort = effort
                    self.args.gpt5_verbosity = verbosity

                    temp_dirs = []
                    try:
                        for input_file in self.input_files:
                            print(f"\n📝 Processing: {input_file.name}")

                            # Create a temporary subdirectory structure that the runner expects
                            temp_input_dir = Path(tempfile.mkdtemp(prefix="causal_exp_"))
                            temp_dirs.append(temp_input_dir)
                            temp_subfolder = temp_input_dir / "prompts"
                            temp_subfolder.mkdir(parents=True, exist_ok=True)

                            # Copy the dataset to the expected location
                            temp_dataset = temp_subfolder / input_file.name
                            shutil.copy2(input_file, temp_dataset)

                            # Run the experiment for this file
                            # Pass the parent directory so ExperimentRunner can create model folder inside
                            output_parent = self.output_path.parent
                            output_parent.mkdir(parents=True, exist_ok=True)

                            runner.run(
                                input_path=str(temp_input_dir),
                                output_path=str(output_parent),  # data/output_llm/pilot_study
                                sub_folder_xs=["prompts"],
                                temperature_value_xs=[self.args.temperature],
                            )

                        # Rename output files to match desired format (with GPT-5 suffix)
                        self._rename_output_files()

                    finally:
                        # Clean up temporary directories for this combo
                        for temp_dir in temp_dirs:
                            if temp_dir.exists():
                                shutil.rmtree(temp_dir)

            print("\n✅ Experiment completed successfully!")
            print(f"📁 Results saved to: {self.output_path}")

            # List output files
            output_files = list(self.output_path.glob("*.csv"))
            if output_files:
                print("📄 Generated files:")
                for file in output_files:
                    print(f"   - {file.name}")

        except Exception as e:
            print(f"❌ Error during experiment: {e}")
            raise
        finally:
            # Restore original args values
            self.args.gpt5_effort = original_efforts
            self.args.gpt5_verbosity = original_verbosities


def main():
    """Main entry point."""
    # Show available datasets if no arguments provided
    if len(sys.argv) == 1:
        print("📊 Available datasets:")
        input_dir = Path("data/input_llm")
        if input_dir.exists():
            for csv_file in input_dir.rglob("*.csv"):
                print(f"   - {csv_file}")
        else:
            print(f"❌ Input directory not found: {input_dir}")
        print("\nUse --help for usage information")
        return

    # Parse arguments and run experiment
    try:
        cli = CausalExperimentCLI()
        cli.run_experiment()
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
