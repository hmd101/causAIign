"""
Base Prompt Generator

Abstract base class for all prompt generators. This extracts the core logic
from Jupyter notebooks into reusable, testable components.
"""

import abc
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..core.constants import (
    graph_structures,
    inference_tasks_rw17,
    rw_17_domain_components,
)
from ..core.domain import create_domain_dict
from ..core.processing import generate_prompt_dataframe
from ..core.utils import append_dfs
from ..styles.base_style import BasePromptStyle


class BasePromptGenerator(abc.ABC):
    """
    Abstract base class for prompt generators.

    This class defines the interface for generating prompts for LLM experiments.
    Different implementations can handle:
    - Human-study-based prompts (match existing human data)
    - Abstract prompts (no human baseline required)
    - Different graph topologies (collider, fork, chain)
    """

    def __init__(self, version: str, output_dir: Optional[Path] = None):
        """
        Initialize the prompt generator.

        Args:
            version: Version identifier (e.g., "6", "7") for file naming
            output_dir: Directory to save generated prompts (default: data/input_llm/rw17/)
        """
        self.version = version
        self.output_dir = output_dir
        self.domain_components = rw_17_domain_components
        self.graph_structures = graph_structures
        self.inference_tasks = inference_tasks_rw17

    @abc.abstractmethod
    def get_domains(self) -> List[str]:
        """Return list of domains to generate prompts for."""
        pass

    @abc.abstractmethod
    def get_graph_types(self) -> List[str]:
        """Return list of graph types to generate prompts for."""
        pass

    @abc.abstractmethod
    def get_counterbalance_conditions(self) -> List[str]:
        """Return list of counterbalance conditions to include."""
        pass

    def generate_domain_prompts(
        self,
        domain_name: str,
        graph_type: str,
        prompt_style: "BasePromptStyle",
        indep_causes_collider: bool = False,
    ) -> pd.DataFrame:
        """
        Generate prompts for a specific domain and graph type.

        Args:
            domain_name: Domain name (economy, sociology, weather)
            graph_type: Graph structure (collider, fork, chain)
            prompt_style: Prompt style object defining response format

        Returns:
            DataFrame with generated prompts for this domain
        """
        # Create domain dictionary
        domain_dict = create_domain_dict(
            domain_name=domain_name,
            introduction=self.domain_components[domain_name]["introduction"],
            variables_config=self.domain_components[domain_name]["variables"],
            graph_type=graph_type,
        )

        # Generate prompts using the style's formatting
        prompts_df = generate_prompt_dataframe(
            domain_dict=domain_dict,
            inference_tasks=self.inference_tasks,
            graph_type=graph_type,
            graph_structures=self.graph_structures,
            counterbalance_enabled=True,
            prompt_category=prompt_style.get_category(),
            prompt_type=prompt_style.get_prompt_instructions(),
            indep_causes_collider=indep_causes_collider,
        )

        return prompts_df

    def generate_all_prompts(
        self, prompt_style: "BasePromptStyle", indep_causes_collider: bool = False
    ) -> pd.DataFrame:
        """
        Generate prompts for all specified domains and graph types.

        Args:
            prompt_style: Prompt style object defining response format

        Returns:
            Combined DataFrame with all generated prompts
        """
        all_dataframes = []

        for domain in self.get_domains():
            for graph_type in self.get_graph_types():
                domain_df = self.generate_domain_prompts(
                    domain, graph_type, prompt_style, indep_causes_collider
                )
                all_dataframes.append(domain_df)

        # Combine all dataframes
        if not all_dataframes:
            raise ValueError(
                "No prompts generated - check domain and graph type configurations"
            )

        combined_df = append_dfs(*all_dataframes)

        # Assign sequential IDs (following notebook ground truth logic)
        combined_df["id"] = range(1, len(combined_df) + 1)

        # Filter for specified counterbalance conditions
        selected_conditions = self.get_counterbalance_conditions()
        if selected_conditions:
            combined_df = combined_df[
                combined_df["cntbl_cond"].isin(selected_conditions)
            ]

        return combined_df

    def save_prompts(
        self,
        prompts_df: pd.DataFrame,
    prompt_style: "BasePromptStyle",
    graph_type: Optional[str] = None,
        content_category: Optional[str] = None,
    ) -> Path:
        """
        Save generated prompts to CSV file following naming convention.

        Args:
            prompts_df: DataFrame with generated prompts
            prompt_style: Prompt style object
            graph_type: Graph type for filename (if None, uses first graph type)

        Returns:
            Path to saved file
        """
        if graph_type is None:
            graph_type = (
                prompts_df["graph"].iloc[0] if len(prompts_df) > 0 else "collider"
            )

        # Follow existing naming convention: {version}_v_{category}_LLM_prompting_{graph}.csv
        filename = f"{self.version}_v_{prompt_style.get_category()}_LLM_prompting_{graph_type}.csv"

        if self.output_dir:
            output_path = self.output_dir / filename
        else:
            # Default to data/input_llm/rw17/
            from ...config.paths import PathManager

            paths = PathManager()
            output_path = paths.rw17_input_llm_dir / filename

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Select columns for LLM prompting (following notebook logic)
        llm_columns = [
            "id",
            "prompt",
            "prompt_category",
            "content_category",
            "graph",
            "domain",
            "cntbl_cond",
            "task",
        ]
        # Copy required columns and attach content category annotation
        llm_prompts_df = prompts_df[[c for c in llm_columns if c in prompts_df.columns]].copy()
        # Ensure content_category column exists and is populated
        cc_value = (content_category or "unspecified").strip()
        llm_prompts_df["content_category"] = cc_value

        # Warn if overwriting existing file
        if output_path.exists():
            print(
                f"⚠️  Overwriting existing prompt file: {output_path} with content_category='{llm_prompts_df['content_category'].iloc[0]}'"
            )

        # Save to CSV
        llm_prompts_df.to_csv(output_path, index=False)

        print(f"Saved {len(llm_prompts_df)} prompts to: {output_path}")
        return output_path

    def generate_and_save(
        self,
        prompt_style: "BasePromptStyle",
        indep_causes_collider: bool = False,
        content_category: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Complete workflow: generate prompts and save to file.

        Args:
            prompt_style: Prompt style object defining response format

        Returns:
            Tuple of (generated_prompts_df, saved_file_path)
        """
        prompts_df = self.generate_all_prompts(prompt_style, indep_causes_collider)
        saved_path = self.save_prompts(
            prompts_df, prompt_style, content_category=content_category
        )

        return prompts_df, saved_path
