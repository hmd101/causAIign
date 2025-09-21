"""
Data Processing Pipeline

This module provides a robust pipeline for processing LLM experiment outputs,
cleaning and merging them with human baseline data, and preparing analysis-ready datasets.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.causalign.config.paths import PathManager
from src.causalign.data_handling.processors.human_processor import HumanDataProcessor
from src.causalign.data_handling.processors.llm_processor import LLMResponseProcessor
from src.causalign.data_handling.validators.data_validator import (
    CombinedDataValidator,
)


class PipelineMode(Enum):
    """Pipeline execution modes"""

    COMBINED = "llm_with_humans"  # LLM + Human data combined
    LLM_ONLY = "llm"  # LLM data only
    HUMAN_ONLY = "humans"  # Human data only


@dataclass
class PipelineConfig:
    """
    Configuration for the data processing pipeline

    This class defines all configurable parameters for the data processing pipeline.
    Each parameter controls a specific aspect of how data is loaded, processed, and output.
    """

    # EXPERIMENT SELECTION
    experiment_name: str = "pilot_study"
    """
    Specifies which experiment directory to process.
    Maps to: data/output_llm/{experiment_name}/
    
    Examples:
    - "pilot_study" → data/output_llm/pilot_study/
    - "temp_test" → data/output_llm/temp_test/
    - "experiment_2024_batch_1" → data/output_llm/experiment_2024_batch_1/
    """

    version: Optional[str] = None
    """
    Filters which version of prompt files to process.
    Only processes files starting with "{version}_v_"
    
    Examples:
    - "6" → processes files like "6_v_claude-3-opus_...csv"
    - "5" → processes files like "5_v_gpt-4o_...csv"  
    - None → processes ALL version files (can be slow!)
    """

    graph_type: str = "collider"
    """
    Specifies the causal graph structure being studied.
    Used for file naming, human data matching, output organization.
    
    Valid values:
    - "collider" → X → Z ← Y (common effect structure)
    - "fork" → X ← Z → Y (confounding structure)  
    - "chain" → X → Z → Y (mediation structure)
    """

    # MODEL SELECTION
    models: Optional[List[str]] = None
    """
    Specifies which AI models to include in processing.
    
    Values:
    - None → process all available models in experiment directory
    - ["claude-3-opus"] → process only Claude Opus
    - ["claude-3-opus", "gpt-4o"] → process Claude and GPT-4o
    - ["gemini-1.5-pro", "claude-3-opus", "gpt-4o"] → process all three
    
    Note: Model names are automatically cleaned (e.g., "claude-3-opus-20240229" → "claude-3-opus")
    """

    # HUMAN DATA CONFIGURATION
    human_raw_file: str = "rw17_collider_ce.csv"
    """
    Filename of raw human data in data/raw/human/rw17/
    
    Examples:
    - "rw17_collider_ce.csv" → for collider experiments
    - "rw17_fork_cc.csv" → for fork experiments
    - "rw16_ce.csv" → for older experiment versions
    """

    prompt_mapping_file: str = "6_v_single_numeric_response_LLM_prompting_collider.csv"
    """
    CSV file containing prompt ID mappings in data/input_llm/rw17/
    Used for assigning proper IDs to human responses to match LLM data.
    
    This file provides the mapping between experimental conditions
    (cntbl_cond, domain, task, graph) and unique prompt IDs.
    """

    # PROCESSING OPTIONS
    add_reasoning_types: bool = True
    """
    Adds cognitive reasoning type annotations to tasks.
    
    True: 
    - Maps letter tasks (a,b,c,...,k) to cognitive reasoning categories
    - Converts tasks to Roman numerals (a→VI, b→VII, etc.)
    - Adds columns: reasoning_type, RW17_label, task (as Roman numerals)
    - Reasoning categories:
      * a,b,c → "Effect-Present Diagnostic Inference" 
      * f,g,h → "Effect-Absent Diagnostic Inference"
      * d,e → "Conditional Independence"
      * i,j,k → "Predictive Inference"
    
    False:
    - Keeps original letter task labels (a,b,c,d,e,f,g,h,i,j,k)
    - No reasoning type annotations added
    """

    aggregate_human_responses: bool = True
    """
    Averages multiple human responses per prompt ID to balance sample sizes.
    
    True (recommended for statistical analysis):
    - Averages human responses per prompt ID
    - Balances sample sizes: 5440 → 336 human responses
    - Final dataset: ~1056 rows (720 LLM + 336 aggregated human)
    - Better for model comparisons and statistical tests
    
    False (for response variability analysis):
    - Keeps all individual human responses
    - Preserves response variability: 5440 human responses
    - Final dataset: ~6160 rows (720 LLM + 5440 individual human)
    - Good for studying individual differences and response distributions
    """

    save_intermediate: bool = True
    """
    Whether to save processed output files.
    
    True:
    - Saves main processed CSV file
    - Saves Roman numerals version (if add_reasoning_types=True)
    - Saves aggregated version (if aggregate_human_responses=True) 
    - Saves processing summary and logs
    
    False:
    - Returns processed data in memory only
    - No files saved to disk
    - Useful for testing or custom analysis workflows
    """

    output_dir: Optional[str] = None
    """
    Custom output directory for processed data files.
    
    None (default):
    - Uses mode-specific directories:
      - LLM+Human: data/processed/llm_with_humans/rw17/{experiment_name}/
      - LLM-only: data/processed/llm/rw17/{experiment_name}/  
      - Human-only: data/processed/humans/rw17/
    
    Custom path:
    - Saves to: {output_dir}/{experiment_name}/
    - Creates directory structure if it doesn't exist
    - Useful for custom analysis workflows or different storage locations
    """

    pipeline_mode: PipelineMode = PipelineMode.COMBINED
    """
    Pipeline execution mode that determines output directory structure.
    
    PipelineMode.COMBINED (default):
    - Processes both LLM and human data, combines them
    - Saves to: data/processed/llm_with_humans/rw17/{experiment_name}/
    
    PipelineMode.LLM_ONLY:
    - Processes only LLM data
    - Saves to: data/processed/llm/rw17/{experiment_name}/
    
    PipelineMode.HUMAN_ONLY:
    - Processes only human data
    - Saves to: data/processed/humans/rw17/
    """


class DataProcessingPipeline:
    """
    Main data processing pipeline orchestrator.

    This class coordinates the complete data processing workflow:
    1. Process LLM responses from output directory
    2. Process human data and assign IDs
    3. Combine and standardize data
    4. Add reasoning type annotations
    5. Aggregate human responses for balanced samples
    6. Validate and save processed data
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        paths: Optional[PathManager] = None,
    ):
        self.config = config or PipelineConfig()
        self.paths = paths or PathManager()

        # Initialize processors
        self.llm_processor = LLMResponseProcessor(self.paths)
        self.human_processor = HumanDataProcessor(self.paths)
        self.validator = CombinedDataValidator()

        # Setup logging
        self.logger = self._setup_logging()

        # Reasoning type mappings based on cognitive psychology literature
        # These map the original experimental tasks to cognitive reasoning categories
        self.reasoning_types = {
            # EFFECT-PRESENT DIAGNOSTIC INFERENCE (Explaining Away)
            # When an effect is observed, infer probability of causes
            # Classic example: fever observed, what caused it?
            "a": "Effect-Present Diagnostic Inference",  # p(Ci=1|E=1, Cj=1)
            "b": "Effect-Present Diagnostic Inference",  # p(Ci=1|E=1)
            "c": "Effect-Present Diagnostic Inference",  # p(Ci=1|E=1, Cj=0)
            # EFFECT-ABSENT DIAGNOSTIC INFERENCE
            # When an effect is NOT observed, infer probability of causes
            # Example: no fever observed, what does this tell us about causes?
            "f": "Effect-Absent Diagnostic Inference",  # p(Ci=1|E=0, Cj=1)
            "g": "Effect-Absent Diagnostic Inference",  # p(Ci=1|E=0)
            "h": "Effect-Absent Diagnostic Inference",  # p(Ci=1|E=0, Cj=0)
            # CONDITIONAL INDEPENDENCE
            # Probability of one cause given another cause (should be independent in collider)
            # Tests understanding of collider structure: causes are independent
            "d": "Conditional Independence",  # p(Ci=1|Cj=1)
            "e": "Conditional Independence",  # p(Ci=1|Cj=0)
            # PREDICTIVE INFERENCE
            # Given causes, predict the effect
            # Forward reasoning from causes to effects
            "i": "Predictive Inference",  # p(E=1|Ci=0, Cj=0)
            "j": "Predictive Inference",  # p(E=1|Ci=0, Cj=1) or p(E=1|Ci=1, Cj=0)
            "k": "Predictive Inference",  # p(E=1|Ci=1, Cj=1)
        }

        # Roman numeral mappings for presentation
        # Orders tasks by reasoning complexity (predictive → independence → diagnostic)
        self.roman_numerals = {
            # Predictive Inference (I-III): Forward reasoning, typically easier
            "i": "I",
            "j": "II",
            "k": "III",
            # Conditional Independence (IV-V): Tests structural understanding
            "d": "IV",
            "e": "V",
            # Effect-Present Diagnostic (VI-VIII): Backward reasoning with effect
            "a": "VI",
            "b": "VII",
            "c": "VIII",
            # Effect-Absent Diagnostic (IX-XI): Backward reasoning without effect, typically hardest
            "f": "IX",
            "g": "X",
            "h": "XI",
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run_complete_pipeline(self) -> pd.DataFrame:
        """
        Run the complete data processing pipeline

        Returns:
            pd.DataFrame: Final processed and validated dataset
        """

        self.logger.info("Starting complete data processing pipeline")
        self.logger.info(f"Configuration: {self.config}")

        try:
            # Step 1: Process LLM responses
            self.logger.info("Step 1: Processing LLM responses")
            llm_data = self.llm_processor.process_experiment(
                experiment_name=self.config.experiment_name,
                version=self.config.version,
                models=self.config.models,
            )

            if llm_data.empty:
                raise ValueError("No LLM data was processed")

            self.logger.info(f"Processed {len(llm_data)} LLM responses")

            # Step 2: Process human data
            self.logger.info("Step 2: Processing human data")
            human_data = self.human_processor.process_human_data(
                raw_filename=self.config.human_raw_file,
                prompt_mapping_file=self.config.prompt_mapping_file,
                graph_type=self.config.graph_type,
            )

            if human_data.empty:
                raise ValueError("No human data was processed")

            self.logger.info(f"Processed {len(human_data)} human responses")

            # Step 3: Combine datasets
            self.logger.info("Step 3: Combining LLM and human data")
            combined_data = self._combine_datasets(llm_data, human_data)

            # Step 4: Add reasoning types and Roman numerals (optional)
            if self.config.add_reasoning_types:
                self.logger.info("Step 4: Adding reasoning types and Roman numerals")
                combined_data = self._add_reasoning_annotations(combined_data)

            # Step 5: Aggregate human responses (optional)
            if self.config.aggregate_human_responses:
                self.logger.info("Step 5: Aggregating human responses")
                combined_data = self.human_processor.aggregate_human_responses(
                    combined_data
                )

            # Step 6: Final validation
            self.logger.info("Step 6: Final validation")
            validation_errors = self.validator.validate_combined_dataframe(
                combined_data
            )
            if validation_errors:
                self.logger.warning(f"Validation warnings: {validation_errors}")

            # Step 7: Save processed data
            if self.config.save_intermediate:
                self.logger.info("Step 7: Saving processed data")
                self._save_processed_data(combined_data)

            self.logger.info(
                f"Pipeline complete! Final dataset: {len(combined_data)} rows, "
                f"{combined_data['subject'].nunique()} subjects"
            )

            return combined_data

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def _combine_datasets(
        self, llm_data: pd.DataFrame, human_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine LLM and human datasets"""

        # Ensure common columns
        common_columns = set(llm_data.columns) & set(human_data.columns)
        self.logger.info(f"Common columns: {common_columns}")

        # Add missing columns to each dataset
        for col in llm_data.columns:
            if col not in human_data.columns:
                human_data[col] = np.nan

        for col in human_data.columns:
            if col not in llm_data.columns:
                llm_data[col] = np.nan

        # Combine datasets
        combined_data = pd.concat([llm_data, human_data], ignore_index=True)

        # Standardize likelihood column name
        # Since human processor now correctly maps 'y' to 'likelihood',
        # both datasets should have likelihood data in 'likelihood' column
        if (
            "response" in combined_data.columns
            and "likelihood" not in combined_data.columns
        ):
            combined_data.rename(columns={"response": "likelihood"}, inplace=True)
        elif (
            "response" in combined_data.columns
            and "likelihood" in combined_data.columns
        ):
            # For any remaining response data, fill missing likelihood values
            # This preserves both human and LLM likelihood data
            combined_data["likelihood"] = combined_data["likelihood"].fillna(
                combined_data["response"]
            )
            combined_data.drop(columns=["response"], inplace=True)

        # Clean up subject names (from notebooks)
        combined_data["subject"] = combined_data["subject"].replace(
            {
                "claude-3-opus-20240229": "claude-3-opus",
                "gemini-2.0-pro-exp-02-05": "gemini-2.0-pro",
            }
        )

        self.logger.info(f"Combined dataset: {len(combined_data)} rows")
        return combined_data

    def _add_reasoning_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add reasoning types and Roman numeral task labels"""

        if "task" not in df.columns:
            self.logger.warning(
                "No 'task' column found, skipping reasoning annotations"
            )
            return df

        # Add reasoning types
        df["reasoning_type"] = df["task"].map(self.reasoning_types)

        # Store original task labels
        df["RW17_label"] = df["task"]

        # Convert to Roman numerals
        df["task"] = df["task"].map(self.roman_numerals)

        # Convert to categorical with proper ordering
        roman_order = [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
        ]
        df["task"] = pd.Categorical(df["task"], categories=roman_order, ordered=True)

        # Sort by task
        df = df.sort_values("task").reset_index(drop=True)

        self.logger.info("Added reasoning type annotations and Roman numerals")
        return df

    def _save_processed_data(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Save processed data using mode-specific directory structure"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Create mode-specific directory structure
        if self.config.output_dir:
            # Use custom output directory with mode organization
            if self.config.pipeline_mode == PipelineMode.HUMAN_ONLY:
                experiment_base_dir = Path(self.config.output_dir) / "rw17"
            else:
                experiment_base_dir = (
                    Path(self.config.output_dir) / "rw17" / self.config.experiment_name
                )
        else:
            # Use default mode-specific directory structure
            processed_base = self.paths.base_dir / "data" / "processed"

            if self.config.pipeline_mode == PipelineMode.HUMAN_ONLY:
                experiment_base_dir = processed_base / "humans" / "rw17"
            elif self.config.pipeline_mode == PipelineMode.LLM_ONLY:
                experiment_base_dir = (
                    processed_base / "llm" / "rw17" / self.config.experiment_name
                )
            else:  # COMBINED
                experiment_base_dir = (
                    processed_base
                    / "llm_with_humans"
                    / "rw17"
                    / self.config.experiment_name
                )

        experiment_base_dir.mkdir(parents=True, exist_ok=True)

        # Generate mode-specific filenames
        version_str = f"{self.config.version}_v_" if self.config.version else ""

        if self.config.pipeline_mode == PipelineMode.HUMAN_ONLY:
            main_filename = f"rw17_{self.config.graph_type}_humans_processed.csv"
        elif self.config.pipeline_mode == PipelineMode.LLM_ONLY:
            main_filename = f"{version_str}{self.config.graph_type}_llm_only.csv"
        else:  # COMBINED
            main_filename = f"{version_str}{self.config.graph_type}_cleaned_data.csv"

        # Main processed file
        main_path = experiment_base_dir / main_filename
        df.to_csv(main_path, index=False)
        saved_files["main"] = main_path
        self.logger.info(f"Saved main processed file: {main_path}")

        # Additional files only for combined and LLM-only modes
        if self.config.pipeline_mode != PipelineMode.HUMAN_ONLY:
            # Roman numerals version (if reasoning types were added)
            if self.config.add_reasoning_types:
                if self.config.pipeline_mode == PipelineMode.LLM_ONLY:
                    roman_filename = (
                        f"{version_str}{self.config.graph_type}_llm_only_roman.csv"
                    )
                else:  # COMBINED
                    roman_filename = (
                        f"{version_str}{self.config.graph_type}_cleaned_data_roman.csv"
                    )

                # Save to reasoning types subdirectory
                reasoning_dir = experiment_base_dir / "reasoning_types"
                reasoning_dir.mkdir(parents=True, exist_ok=True)

                roman_path = reasoning_dir / roman_filename
                df.to_csv(roman_path, index=False)
                saved_files["roman"] = roman_path
                self.logger.info(f"Saved Roman numerals version: {roman_path}")

        # Aggregated human version (only for combined mode)
        if (
            self.config.pipeline_mode == PipelineMode.COMBINED
            and self.config.aggregate_human_responses
        ):
            agg_filename = f"{version_str}humans_avg_equal_sample_size_cogsci.csv"
            agg_path = experiment_base_dir / agg_filename
            df.to_csv(agg_path, index=False)
            saved_files["aggregated"] = agg_path
            self.logger.info(f"Saved aggregated version: {agg_path}")

        return saved_files

    def run_llm_only_pipeline(self) -> pd.DataFrame:
        """Run pipeline for LLM data only (useful for testing)"""
        self.logger.info("Running LLM-only pipeline")

        llm_data = self.llm_processor.process_experiment(
            experiment_name=self.config.experiment_name,
            version=self.config.version,
            models=self.config.models,
        )

        if self.config.add_reasoning_types:
            llm_data = self._add_reasoning_annotations(llm_data)

        # Save processed data if requested
        if self.config.save_intermediate:
            self.logger.info("Saving LLM-only processed data")
            self._save_processed_data(llm_data)

        self.logger.info(
            f"LLM-only pipeline complete! Final dataset: {len(llm_data)} rows, "
            f"{llm_data['subject'].nunique()} subjects"
        )

        return llm_data

    def run_human_only_pipeline(self) -> pd.DataFrame:
        """Run pipeline for human data only (useful for testing)"""
        self.logger.info("Running human-only pipeline")

        human_data = self.human_processor.process_human_data(
            raw_filename=self.config.human_raw_file,
            prompt_mapping_file=self.config.prompt_mapping_file,
            graph_type=self.config.graph_type,
        )

        if self.config.add_reasoning_types:
            human_data = self._add_reasoning_annotations(human_data)

        # Save processed data if requested
        if self.config.save_intermediate:
            self.logger.info("Saving human-only processed data")
            self._save_processed_data(human_data)

        self.logger.info(
            f"Human-only pipeline complete! Final dataset: {len(human_data)} rows, "
            f"{human_data['subject'].nunique()} subjects"
        )

        return human_data
