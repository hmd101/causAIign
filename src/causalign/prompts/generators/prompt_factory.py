"""
Prompt Factory
==============

This module centralizes prompt generation for experiments:

- Styles (PROMPT_STYLES):
    * numeric, numeric_xml, numeric-conf, CoT (+ variants)
- Generators (GENERATORS):
    * "rw17": RW17Generator
    * "abstract": AbstractGenerator

Key API used by scripts/generate_experiment_prompts.py
-----------------------------------------------------
- PromptFactory.create_style(style_name): returns a BasePromptStyle
- PromptFactory.create_generator(generator_type, version, output_dir): returns a BasePromptGenerator
- PromptFactory.generate_experiment_prompts(experiment_config, version, output_dir, indep_causes_collider, content_category):
    Orchestrates RW17 generation across the experiment's graph types and styles. Applies optional overlays from
    experiment_config.custom_settings (rw17_overlays / rw17_overlays_file) when provided, and honors configured domains.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from ...experiment.config.experiment_config import ExperimentConfig
from ..styles import (
    BasePromptStyle,
    ChainOfThoughtStyle,
    ConfidenceStyle,
    NumericOnlyStyle,
)
from .abstract_generator import AbstractGenerator
from .base_generator import BasePromptGenerator
from .rw17_generator import RW17Generator


class PromptFactory:
    """
    Factory for creating prompt generators and styles.

    Provides a unified interface for generating prompts for different experiment types.
    """

    # Registry of available prompt styles
    PROMPT_STYLES = {
        "numeric": NumericOnlyStyle,
        "numeric_xml": lambda: NumericOnlyStyle(use_xml=True),
        "numeric-conf": ConfidenceStyle,
        "CoT": ChainOfThoughtStyle,
        "CoT-brief": lambda: ChainOfThoughtStyle(
            max_reasoning_words=50, max_reasoning_tokens=67
        ),
        "CoT-moderate": lambda: ChainOfThoughtStyle(
            max_reasoning_words=80, max_reasoning_tokens=107
        ),
    }

    # Registry of available generators
    GENERATORS = {
        "rw17": RW17Generator,
        "abstract": AbstractGenerator,
        # Can add more generators here: "abstract": AbstractGenerator, etc.
    }

    @classmethod
    def create_style(cls, style_name: str) -> BasePromptStyle:
        """
        Create a prompt style instance.

        Args:
            style_name: Name of the prompt style

        Returns:
            Prompt style instance

        Raises:
            ValueError: If style name not recognized
        """
        if style_name not in cls.PROMPT_STYLES:
            available = list(cls.PROMPT_STYLES.keys())
            raise ValueError(
                f"Unknown prompt style '{style_name}'. Available: {available}"
            )

        style_class_or_lambda = cls.PROMPT_STYLES[style_name]
        if (
            callable(style_class_or_lambda)
            and hasattr(style_class_or_lambda, "__name__")
            and style_class_or_lambda.__name__ == "<lambda>"
        ):
            # It's a lambda function, call it directly
            return cast(BasePromptStyle, style_class_or_lambda())
        else:
            # It's a class, instantiate it
            return cast(BasePromptStyle, style_class_or_lambda())

    @classmethod
    def create_generator(
    cls, generator_type: str, version: str, output_dir: Optional[Path] = None
    ) -> BasePromptGenerator:
        """
        Create a prompt generator instance.

        Args:
            generator_type: Type of generator ("rw17", "abstract", etc.)
            version: Version identifier for file naming
            output_dir: Directory to save prompts

        Returns:
            Prompt generator instance

        Raises:
            ValueError: If generator type not recognized
        """
        if generator_type not in cls.GENERATORS:
            available = list(cls.GENERATORS.keys())
            raise ValueError(
                f"Unknown generator type '{generator_type}'. Available: {available}"
            )

        generator_class = cls.GENERATORS[generator_type]
        return generator_class(version, output_dir)

    @classmethod
    def generate_experiment_prompts(
        cls,
        experiment_config: ExperimentConfig,
        version: str,
    output_dir: Optional[Path] = None,
        indep_causes_collider: bool = False,
    content_category: Optional[str] = None,
    ) -> Dict[str, List[Tuple[pd.DataFrame, Path]]]:
        """
        Generate all prompts for an experiment configuration.

        Args:
            experiment_config: Experiment configuration
            version: Version identifier
            output_dir: Directory to save prompts

        Returns:
            Dict mapping prompt_style -> [(dataframe, saved_path), ...]
        """
        results = {}

        # Create appropriate generator based on experiment type
        if experiment_config.human_data_match:
            generator: Any = cls.create_generator("rw17", version, output_dir)
            # Configure graph types for this experiment
            generator = generator.with_graph_types(experiment_config.graph_types)
            # If domains are specified in config, honor them (supports overloaded names)
            if experiment_config.domains:
                try:
                    generator = generator.with_domains(experiment_config.domains)
                except Exception:
                    pass
            # Apply optional RW17 overlays from custom_settings
            cs = experiment_config.custom_settings or {}
            if cs:
                try:
                    from ..core.overlays import create_overloaded_domains
                    overlays_list = []
                    if "rw17_overlays" in cs and isinstance(cs["rw17_overlays"], list):
                        overlays_list = cs["rw17_overlays"]
                    if "rw17_overlays_file" in cs and cs["rw17_overlays_file"]:
                        from pathlib import Path as _P

                        import yaml
                        p = _P(cs["rw17_overlays_file"]).expanduser()
                        print(f"📄 Overlays file: {p} (exists={p.exists()})")
                        if p.exists():
                            with open(p, "r") as f:
                                data = yaml.safe_load(f) or []
                                if isinstance(data, list):
                                    overlays_list.extend(data)
                                else:
                                    print(f"⚠️ Overlays YAML did not parse as a list, got {type(data)}; ignoring.")
                        else:
                            print("⚠️ Overlays file not found; skipping overlays.")
                    if overlays_list:
                        print(f" Loaded {len(overlays_list)} overlay entries.")
                        generator.domain_components = create_overloaded_domains(
                            generator.domain_components, overlays_list
                        )
                        # If experiment didn't explicitly set domains, default to the new clone names
                        if not experiment_config.domains:
                            # Prefer 'new_name' from overlays; fallback to "{domain}_overloaded"
                            clone_names: List[str] = []
                            for ov in overlays_list:
                                nn = ov.get("new_name")
                                if nn and nn in generator.domain_components:
                                    clone_names.append(nn)
                                else:
                                    d = ov.get("domain") or ov.get("domain_name")
                                    if d:
                                        fallback = f"{d}_overloaded"
                                        if fallback in generator.domain_components:
                                            clone_names.append(fallback)
                            # If nothing matched (e.g., strict membership check failed),
                            # fall back to any keys that look like overloaded clones.
                            if not clone_names:
                                for k in generator.domain_components.keys():
                                    if "_ovl_" in k or "=ctl" in k or "=econ" in k or "=soc" in k or "=weath" in k:
                                        clone_names.append(k)
                            if not clone_names:
                                print(" No obvious clone-name keys found in domain_components; keys sample:")
                                try:
                                    sample_keys = list(generator.domain_components.keys())[:10]
                                    print(sample_keys)
                                except Exception as _e:
                                    print(f"(could not sample keys: {_e})")
                            # Deduplicate and keep order
                            seen: set[str] = set()
                            ordered: List[str] = []
                            for name in clone_names:
                                if name not in seen:
                                    seen.add(name)
                                    ordered.append(name)
                            if ordered:
                                print(f" Auto-selecting overlay clone domains: {ordered}")
                                generator = generator.with_domains(ordered)
                            else:
                                print(" No overlay clone domains detected; using default RW17 domains.")
                except Exception:
                    # Non-fatal: continue without overlays, but surface the error for debugging.
                    import traceback
                    print("Overlay application failed; continuing without overlays. Details:")
                    traceback.print_exc()
        else:
            # For abstract experiments, support overlays via custom_settings
            cs = experiment_config.custom_settings or {}
            overlays_yaml = None
            if cs:
                if "rw17_overlays_file" in cs and cs["rw17_overlays_file"]:
                    overlays_yaml = cs["rw17_overlays_file"]
            # Pass overlays_yaml to AbstractGenerator if present
            from .abstract_generator import AbstractGenerator
            generator = AbstractGenerator(
                version,
                output_dir,
                custom_domains=experiment_config.get_domains(),
                overlays_yaml=overlays_yaml,
            )
            generator = generator.with_domains(experiment_config.get_domains())
            generator = generator.with_graph_types(experiment_config.graph_types)

        # Generate prompts for each style
        for style_name in experiment_config.prompt_styles:
            style = cls.create_style(style_name)
            style_results = []

            # Generate prompts for each graph type
            for graph_type in experiment_config.graph_types:
                # Use the correct generator (already configured above)
                single_graph_generator = generator.with_graph_types([graph_type])

                # Generate and save prompts
                prompts_df, saved_path = single_graph_generator.generate_and_save(
                    style,
                    indep_causes_collider,
                    content_category=content_category,
                )
                style_results.append((prompts_df, saved_path))

                print(
                    f"Generated {len(prompts_df)} prompts for {style_name} + {graph_type}"
                )

            results[style_name] = style_results

        return results

    @classmethod
    def list_available_styles(cls) -> List[str]:
        """List all available prompt styles."""
        return list(cls.PROMPT_STYLES.keys())

    @classmethod
    def list_available_generators(cls) -> List[str]:
        """List all available generators."""
        return list(cls.GENERATORS.keys())

    @classmethod
    def get_style_description(cls, style_name: str) -> str:
        """Get description of a prompt style."""
        style = cls.create_style(style_name)
        return style.get_description()
