"""
Experiment Configuration

Defines experiment configurations using Option 2: Research Questions approach.
Each experiment represents a research question and can include multiple prompt styles.

AVAILABLE LLM-AGENTS:
gpt-4o gpt-4.1 gpt-4.1-mini gpt-4 gpt-3.5-turbo
gpt-5
gpt-5-mini
gpt-5-nano
o1-mini o1 o3-mini o3
claude-3-opus-20240229 claude-opus-4-20250514 claude-opus-4-1-20250805 claude-3-haiku-20240307 claude-opus-4-20250514 claude-sonnet-4-20250514 claude-3-7-sonnet-20250219 claude-3-5-haiku-20241022
gemini-1.5-pro gemini-2.5-pro gemini-2.5-flash gemini-2.5-flash-lite


RW17 OVERLAYS: HOW TO CREATE AND USE OVERLOADED PROMPTS
-------------------------------------------------------
Overview
- RW17 overlays let you create new domain variants ("overloaded" clones) by appending or overriding
  parts of the RW17 domain text. The engine can:
  - append from literal strings (e.g., introduction_append)
  - append from other base RW17 domains (e.g., introduction_append_from, explanations_append_from)
  - override value dictionaries (p_value/m_value) and explanations
  - optionally append length-matched neutral filler to explanations for content-control

Two ready-to-use overlay YAML sources
1) Explanation-only + matched filler: src/causalign/prompts/custom_domains/rw17_overlays_expl_filler.yaml
    - Focuses only on explanations (X/Y/Z) with optional matched-length filler
    - Good for content-vs-length controls
2) Systematic generator output: src/causalign/prompts/custom_domains/rw17_overlays_generated.yaml
    - Produced by scripts/generate_rw17_overlays.py
    - Systematically appends content across intro/variables from other domains

Naming scheme (generator default)
- domain_overloaded_xeN_yeN_zeN where N is 1 or 2 = number of source domains used for that variable’s explanations.
- Variables are X, Y, Z; fields are xe/ye/ze for explanations.
- Only variables with appended explanations are listed in the name.

Typical workflow to run with overlays
1) Choose overlays YAML:
    - Set ExperimentConfig.custom_settings["rw17_overlays_file"] to one of the files above.
    - To regenerate the systematic YAML with the latest naming, run:
      python3 scripts/generate_rw17_overlays.py
2) Decide domain selection:
    - If ExperimentConfig.domains is None or [], all created clones will be included.
    - To run specific clones only, put their exact new_name tokens in ExperimentConfig.domains
      (e.g., ["economy_overloaded_xe1_ye2", "sociology_overloaded_ze1"]).
3) Generate prompt CSVs for the experiment:
    - Example (zsh):
      python3 scripts/generate_experiment_prompts.py --version 3 --experiment rw17_overloaded --prompt-content realistic
4) Outputs:
    - CSVs saved under the versioned prompts output directory; logs print their paths.

Notes
- YAML anchors (&id) and aliases (*id) inside YAML are harmless; they just deduplicate repeated lists.
- If a YAML overlay lacks a unique new_name, the fallback is "{domain}_overloaded" which can overwrite
  previous clones. Prefer explicit new_name or use the generator’s naming scheme.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for a specific experiment.

    Each experiment represents a research question and defines:
    - Which prompt styles to use
    - Which graph types to test
    - Whether human data matching is required
    - Specific domains (if different from default RW17)
    """

    name: str
    description: str
    prompt_styles: List[str]
    graph_types: List[str]
    human_data_match: bool = True
    domains: Optional[List[str]] = None  # None = use RW17 defaults
    counterbalance_conditions: Optional[List[str]] = None  # None = use RW17 defaults
    models: Optional[List[str]] = None  # None = use all available
    temperatures: Optional[List[float]] = None  # None = use default [0.0]
    custom_settings: Optional[Dict[str, Any]] = None

    def get_domains(self) -> List[str]:
        """Get domains for this experiment."""
        return self.domains or ["economy", "sociology", "weather"]

    def get_counterbalance_conditions(self) -> List[str]:
        """Get counterbalance conditions for this experiment."""
        return self.counterbalance_conditions or ["ppp", "pmm", "mmp", "mpm"]

    def get_models(self) -> List[str]:
        """Get models for this experiment."""
        return self.models or ["gpt-4o", "claude-3-opus-20240229", "gemini-1.5-pro"]

    def get_temperatures(self) -> List[float]:
        """Get temperatures for this experiment."""
        return self.temperatures or [0.0]

    def requires_human_data(self) -> bool:
        """Check if this experiment requires human data for comparison."""
        return self.human_data_match

    def get_human_data_files(self) -> Dict[str, str]:
        """Get mapping of graph types to human data files."""
        if not self.human_data_match:
            return {}

        # Default mapping for RW17 study
        return {
            "collider": "rw17_collider_ce.csv",
            "fork": "rw17_fork_cc.csv",
            # Add other graph types as needed
        }


# Predefined experiment configurations (Research Questions)
EXPERIMENT_CONFIGS = {
    "pilot_study": ExperimentConfig(
        name="pilot_study",
        description="Initial pilot study comparing basic numeric responses with confidence ratings on collider tasks",
        prompt_styles=["numeric"],
        graph_types=["collider"],
        human_data_match=True,
        models=["gpt-4o", "claude-3-opus-20240229",], 
        temperatures=[0.0],
    ),
    "abstract_reasoning": ExperimentConfig(
        name="abstract_reasoning",
        description="Test causal reasoning with abstract domains (no human baseline)",
        prompt_styles=[
            "numeric",
            "numeric-conf",
            "CoT",
        ],
        graph_types=["collider"],
        human_data_match=False,
        domains=[
            # "abstract_abc",
            # "abstract_xyz",
            # "abstract_triplets",
            # "abstract_triplets_symb",
            # "abstract_quintuplets",
        ],
        models=[
            "gpt-4o",
            # "claude-3-opus-20240229",
        ],  # out of credit for "gemini-1.5-pro" ['claude-3-haiku-20240307', 'claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219', 'claude-3-5-haiku-20241022', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4'
        temperatures=[0.0],
    ),

    "random_abstract": ExperimentConfig(
        name="random_abstract",
        description="Test causal reasoning with randomly generated abstract domains (no human baseline)",
        prompt_styles=[
            "numeric",
            "numeric-conf",
            "CoT",
        ],
        graph_types=["collider"],
        human_data_match=False,
        domains=[
            "abs_alnum_10",
            "abs_num_symb_10",
            "abs_all_10"
        ],
        models=[
            "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4", "gpt-3.5-turbo",
            "gpt-5", "gpt-5-mini", "gpt-5-nano",
             "o1-mini", "o1", "o3-mini", "o3", 
            "claude-3-opus-20240229", "claude-opus-4-20250514", "claude-opus-4-1-20250805", "claude-3-haiku-20240307", "claude-opus-4-20250514",
            "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022", "gemini-1.5-pro",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
        ],  
        temperatures=[0.0],
        ),





    "rw17_indep_causes": ExperimentConfig(
        name="rw17_indep_causes",
        description="Test causal reasoning with explicitly stating that the causes are independent",
        prompt_styles=[
            "numeric",
            "numeric-conf",
            "CoT",
        ],
        graph_types=["collider"],
        human_data_match=True,
        domains=["economy", "sociology", "weather"],
        models=[
            "gpt-4o",
            # "claude-3-opus-20240229",
        ],  # out of credit for "gemini-1.5-pro" ['claude-3-haiku-20240307', 'claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219', 'claude-3-5-haiku-20241022', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4'
        temperatures=[0.0],
    ),
    "rw17_overloaded_d": ExperimentConfig(
        name="rw17_overloaded_d",
        description="RW17 prompts with additional overloaded text in detailed variable  description fields (d)",
        prompt_styles=[
            "numeric",
            "numeric-conf",
            "CoT",
        ],
        graph_types=["collider"],
        human_data_match=True,
        # Domains here can include overloaded clones produced by overlays.
        # - To include ALL clones created by the overlays YAML, set this to [] or None.
        # - To include a SUBSET, list the exact new_name tokens created by your YAML, e.g.:
        #   ["economy_overloaded_xe1_ye2", "sociology_overloaded_ze1"].
        # - If you keep legacy placeholders like below, ensure they exist in your chosen YAML.
        # domains=["economy_overloaded", "sociology_overloaded", "weather_overloaded"],
        models=[
            "gpt-4o",
        ],
        temperatures=[0.0],
        custom_settings={
            # You can either inline overlays via 'rw17_overlays' or point to a YAML file with a list.
            # Switch between these files as needed:
            # - Systematic generator output:
            #   "rw17_overlays_file": "src/causalign/prompts/custom_domains/rw17_overlays_generated.yaml",
            # - Explanation-only + matched filler (content control):
            "rw17_overlays_file": "src/causalign/prompts/custom_domains/rw17_overlays_append_d.yaml",
        },
    ),


    "rw17_overloaded_e": ExperimentConfig(
        name="rw17_overloaded_e",
        description="RW17 prompts with additional overloaded text in detailed variable  causal mechanism, ie graph edges ( explanations)",
        prompt_styles=[
            "numeric",
            "numeric-conf",
            "CoT",
        ],
        graph_types=["collider"],
        human_data_match=True,
        
        models=[
            "gpt-4o",
        ],
        temperatures=[0.0],
        custom_settings={
            
            "rw17_overlays_file": "src/causalign/prompts/custom_domains/rw17_overlays_append_e.yaml",
        },
    ),

    "rw17_overloaded_de": ExperimentConfig(
        name="rw17_overloaded_de",
        description="RW17 prompts with additional overloaded text in detailed variable  description fields + causak mechanism, ie graph edges (detailed + explanations)",
        prompt_styles=[
            "numeric",
            "numeric-conf",
            "CoT",
        ],
        graph_types=["collider"],
        human_data_match=True,
        
        models=[
            "gpt-4o",
        ],
        temperatures=[0.0],
        custom_settings={
            
            "rw17_overlays_file": "src/causalign/prompts/custom_domains/rw17_overlays_append_de.yaml",
        },
    ),
############
# ABSTRACT OVERLOADED
"abstract_overloaded_rw17_de": ExperimentConfig(
    name="abstract_overloaded_rw17_de",
    description="Abstract domain prompts with overloaded text in detailed variable description fields (d)  matches rw17_indep_causes average length.",
    prompt_styles=[
        "numeric",
        "numeric-conf",
        "CoT",
    ],
    graph_types=["collider"],
    human_data_match=False,
    domains=[
        "abs_alnum_10",
        "abs_num_symb_10",
        "abs_all_10"
    ],
    models=[
        "gpt-4o",
    ],
    temperatures=[0.0],
    custom_settings={
        "rw17_overlays_file": "src/causalign/prompts/custom_domains/overlay_fillers_rw17_indep_causes_de_overlays.yaml",
    },
),
"abstract_overloaded_lorem_de": ExperimentConfig(
    name="abstract_overloaded_lorem_de",
    description="Abstract domain prompts with overloaded text in both detailed and explanation fields (de) matches rw17_indep_causes average length.",
    prompt_styles=[
        "numeric",
        "numeric-conf",
        "CoT",
    ],
    graph_types=["collider"],
    human_data_match=False,
    domains=[
        "abs_alnum_10",
        "abs_num_symb_10",
        "abs_all_10"
    ],
    models=[
        "gpt-4o",
    ],
    temperatures=[0.0],
    custom_settings={
        "rw17_overlays_file": "src/causalign/prompts/custom_domains/overlay_fillers_lorem_ipsum_de_overlays.yaml",
    },
),

    

#########

    # fork is not yet supported
    "graph_comparison": ExperimentConfig(
        name="graph_comparison",
        description="Compare LLM performance across different causal graph structures",
        prompt_styles=["numeric"],
        graph_types=["collider", "fork"],  # Can extend to ["collider", "fork", "chain"]
        human_data_match=True,
        models=["gpt-4o"],
        temperatures=[0.0],
    ),
    "reasoning_methods": ExperimentConfig(
        name="reasoning_methods",
        description="Compare different reasoning approaches: basic vs confidence vs chain-of-thought",
        prompt_styles=[
            "numeric",
            "numeric-conf",
            "CoT",
        ],
        graph_types=["collider"],
        human_data_match=True,
        models=["gpt-4o"],
        temperatures=[0.0],
    ),
    "temperature_study": ExperimentConfig(
        name="temperature_study",
        description="Study the effect of temperature on causal reasoning consistency",
        prompt_styles=["numeric"],
        graph_types=["collider"],
        human_data_match=True,  # Use default RW17 domains: economy, sociology, weather
        models=["gpt-4o"],
        temperatures=[0.0, 0.5, 1.0],
    ),
    "xml_format_study": ExperimentConfig(
        name="xml_format_study",
        description="Test single numeric responses using XML format for structured output",
        prompt_styles=["numeric_xml"],
        graph_types=["collider"],
        human_data_match=True,
        models=["gpt-4o"],
        temperatures=[0.0],
    ),
}


def get_experiment_config(experiment_name: str) -> ExperimentConfig:
    """
    Get experiment configuration by name.

    Args:
        experiment_name: Name of the experiment

    Returns:
        ExperimentConfig object

    Raises:
        ValueError: If experiment name not found
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        available = list(EXPERIMENT_CONFIGS.keys())
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. Available: {available}"
        )

    return EXPERIMENT_CONFIGS[experiment_name]


def list_available_experiments() -> List[str]:
    """List all available experiment names."""
    return list(EXPERIMENT_CONFIGS.keys())


def print_experiment_summary():
    """Print a summary of all available experiments."""
    print("Available Experiments:")
    print("=" * 50)

    for name, config in EXPERIMENT_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.description}")
        print(f"  Prompt Styles: {config.prompt_styles}")
        print(f"  Graph Types: {config.graph_types}")
        print(f"  Human Data: {'Yes' if config.human_data_match else 'No'}")
        print(f"  Models: {config.get_models()}")
        print(f"  Temperatures: {config.get_temperatures()}")
