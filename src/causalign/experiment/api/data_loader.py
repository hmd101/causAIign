# src/causalign/experiment/api/data_loader.py
"""
This module provides functionality for loading and managing causal inference experimental data.

The module contains the CausalExperimentLoader class which handles loading various types of
experimental data including:
- Base prompt component data
- Human experimental responses
- Model-generated responses across different:
  - Models (e.g. GPT-3, GPT-4)
  - Prompt types
  - Temperature settings
  - Experimental conditions

Classes:
    CausalExperimentLoader: Main class for loading and managing experimental data

Methods:
    parse_filename: Extracts metadata from experiment result filenames
    load_base_data: Loads the base prompt components data
    load_human_data: Loads human experimental response data
    discover_available_data: Discovers what model data is available
    load_model_responses: Loads specific model response data
    load_data: Main method to load and combine all requested data

The loader expects a specific directory structure and file naming convention:
    base_path/
    ├── {version}/
    │   ├── {version}_prompt_components_only.csv
    │   └── {version}_{condition}_{model}_type_{prompt_type}_responses_temp_{temperature}.csv
    └── human_data_for_analysis.csv
"""

import pathlib
import re
from typing import Dict, List, Optional, Tuple, Union

#         return {col: sorted(data[col].unique().tolist()) for col in data.columns}
import pandas as pd


class CausalExperimentLoader:
    """Loader for causal inference experimental data with flexible model handling."""

    def __init__(self, base_path: Union[str, pathlib.Path]):
        """
        Initialize loader with base path to data directory.

        Parameters
        ----------
        base_path : Union[str, pathlib.Path]
            Path to the root data directory
        """
        self.base_path = pathlib.Path(base_path)
        self.version = None
        self.prompt_components = None
        self.human_data = None

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse experiment filename to extract metadata."""
        pattern = r"(?P<version>\d+_v)_(?P<condition>[a-z]{3})_(?P<model>[\w\-\.]+)_type_(?P<prompt_type>\w+)_responses_temp_(?P<temperature>[\d\.]+)\.csv"
        match = re.match(pattern, filename)
        return match.groupdict() if match else {}

    def load_base_data(self, version: str = "1_v") -> pd.DataFrame:
        """Load prompt components data."""
        if self.prompt_components is None or self.version != version:
            self.version = version
            path = self.base_path / version / f"{version}_prompt_components_only.csv"

            if not path.exists():
                raise FileNotFoundError(
                    f"Base data file not found at {path}. "
                    f"Please ensure the file exists and the path is correct."
                )

            self.prompt_components = pd.read_csv(path, sep=";")
        return self.prompt_components

    def load_human_data(self) -> pd.DataFrame:
        """Load human experimental data."""
        if self.human_data is None:
            path = self.base_path / "human_data_for_analysis.csv"

            if not path.exists():
                raise FileNotFoundError(
                    f"Human data file not found at {path}. "
                    f"Please ensure the file exists and the path is correct."
                )

            self.human_data = pd.read_csv(path, sep=";")
            self.human_data["subject"] = "humans"
            self.human_data = self.human_data.rename(columns={"y": "response"})
        return self.human_data

    def load_model_responses(
        self,
        version: str,
        model: str,
        prompt_type: str,
        condition: str,
        temperature: str,
    ) -> Optional[pd.DataFrame]:
        """Load specific model responses."""
        file_path = (
            self.base_path
            / "1_v"
            / model
            / prompt_type
            / f"{version}_{condition}_{model}_type_{prompt_type}_responses_temp_{temperature}.csv"
        )

        try:
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return None

            result = pd.read_csv(file_path, sep=";")
            # Clean response column
            result["response"] = result["response"].apply(
                lambda x: x[8:-9]
                if isinstance(x, str) and x.startswith("<answer>")
                else x
            )
            result["response"] = pd.to_numeric(result["response"], errors="coerce")
            return result
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def discover_available_data(self) -> Dict[str, List[str]]:
        """Discover available models, temperatures, and prompt types."""
        available_data = {
            "models": set(),
            "temperatures": set(),
            "prompt_types": set(),
            "conditions": set(),
        }

        for file_path in self.base_path.rglob("*.csv"):
            metadata = self.parse_filename(file_path.name)
            if metadata:
                available_data["models"].add(metadata["model"])
                available_data["temperatures"].add(metadata["temperature"])
                available_data["prompt_types"].add(metadata["prompt_type"])
                available_data["conditions"].add(metadata["condition"])

        return {k: sorted(list(v)) for k, v in available_data.items()}

    def load_experimental_data(
        self,
        version: str = "1_v",
        models: Optional[List[str]] = None,
        prompt_types: Optional[List[str]] = None,
        temperatures: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        include_human: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Load and combine all experimental data.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, List[str]]]
            Combined DataFrame and dictionary of what data was actually loaded
        """
        # Load base data
        base_data = self.load_base_data(version)
        loaded_data = []

        # Load human data if requested
        if include_human:
            human_data = self.load_human_data()
            human_data_subset = human_data[["id", "subject", "response", "y_length"]]
            merged_human = pd.merge(base_data, human_data_subset, on="id", how="inner")
            loaded_data.append(merged_human)

        # Discover available data if not specified
        available = self.discover_available_data()
        models = models or available["models"]
        prompt_types = prompt_types or available["prompt_types"]
        temperatures = temperatures or available["temperatures"]
        conditions = conditions or available["conditions"]

        # Track what was actually loaded
        loaded_metadata = {
            "models": set(),
            "prompt_types": set(),
            "temperatures": set(),
            "conditions": set(),
        }

        # Load model data
        for model in models:
            for prompt_type in prompt_types:
                for condition in conditions:
                    for temp in temperatures:
                        result = self.load_model_responses(
                            version, model, prompt_type, condition, temp
                        )
                        if result is not None:
                            merged_result = pd.merge(
                                base_data, result, on="id", how="inner"
                            )
                            loaded_data.append(merged_result)

                            # Update loaded metadata
                            loaded_metadata["models"].add(model)
                            loaded_metadata["prompt_types"].add(prompt_type)
                            loaded_metadata["temperatures"].add(temp)
                            loaded_metadata["conditions"].add(condition)

        # Combine all data
        combined_data = pd.concat(loaded_data, axis=0, ignore_index=True)

        # Set y_length for non-human subjects
        combined_data.loc[combined_data["subject"] != "humans", "y_length"] = 1

        return combined_data, {k: sorted(list(v)) for k, v in loaded_metadata.items()}
