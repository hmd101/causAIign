#!/usr/bin/env python3
"""
Data Processing Pipeline Runner

This script provides a command-line interface for running the robust data processing pipeline.
It replaces the manual execution of Jupyter notebooks with an automated, configurable script.

Usage:
    python scripts/run_data_pipeline.py --experiment pilot_study --version 6
    python scripts/run_data_pipeline.py --llm-only --models claude-3-opus gpt-4o
    python scripts/run_data_pipeline.py --human-only --human-file rw17_collider_ce.csv
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.causalign.config.paths import PathManager
from src.causalign.data_handling.processors.pipeline import (
    DataProcessingPipeline,
    PipelineConfig,
    PipelineMode,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("data_pipeline.log")],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for pilot study, version 6
  python scripts/run_data_pipeline.py --experiment pilot_study --version 6
  
  # Run only LLM processing for specific models
  python scripts/run_data_pipeline.py --llm-only --models claude-3-opus gpt-4o
  
  # Run only human data processing
  python scripts/run_data_pipeline.py --human-only --human-file rw17_collider_ce.csv
  
  # Run pipeline without aggregating human responses
  python scripts/run_data_pipeline.py --no-aggregate
  
  # Run pipeline without reasoning type annotations
  python scripts/run_data_pipeline.py --no-reasoning-types
        """,
    )

    # Pipeline mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--llm-only", action="store_true", help="Process only LLM data"
    )
    mode_group.add_argument(
        "--human-only", action="store_true", help="Process only human data"
    )

    # Data configuration
    parser.add_argument(
        "--experiment",
        default="pilot_study",
        help="Experiment name (default: pilot_study)",
    )
    parser.add_argument(
        "--version", default="6", help="Version filter for data files (default: 6)"
    )
    parser.add_argument(
        "--graph-type", default="collider", help="Graph type (default: collider)"
    )
    parser.add_argument(
        "--models", nargs="+", help="Specific models to process (default: all models)"
    )
    parser.add_argument(
        "--human-file",
        default="rw17_collider_ce.csv",
        help="Human data raw file (default: rw17_collider_ce.csv)",
    )
    parser.add_argument(
        "--prompt-mapping",
        default="6_v_numeric_LLM_prompting_collider.csv",
        help="Prompt mapping file for human ID assignment",
    )

    # Processing options
    parser.add_argument(
        "--no-reasoning-types",
        action="store_true",
        help="Skip adding reasoning type annotations",
    )
    parser.add_argument(
        "--no-aggregate", action="store_true", help="Skip aggregating human responses"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving intermediate files"
    )

    # Output options
    parser.add_argument(
        "--output-dir", help="Custom output directory (default: use path manager)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Determine pipeline mode based on CLI arguments
        if args.llm_only:
            pipeline_mode = PipelineMode.LLM_ONLY
        elif args.human_only:
            pipeline_mode = PipelineMode.HUMAN_ONLY
        else:
            pipeline_mode = PipelineMode.COMBINED

        # Create pipeline configuration with detailed parameter explanations
        config = PipelineConfig(
            # EXPERIMENT_NAME: Specifies which experiment directory to process
            # Maps to: data/output_llm/{experiment_name}/
            # Examples: "pilot_study", "temp_test", "experiment_2024_batch_1"
            experiment_name=args.experiment,
            # VERSION: Filters which version of prompt files to process
            # Only processes files starting with "{version}_v_"
            # Examples: "6" → processes "6_v_claude-3-opus_...csv", None → processes ALL versions
            version=args.version,
            # GRAPH_TYPE: Specifies the causal graph structure being studied
            # Used for file naming, human data matching, output organization
            # Examples: "collider" (X→Z←Y), "fork" (X←Z→Y), "chain" (X→Z→Y)
            graph_type=args.graph_type,
            # MODELS: Specifies which AI models to include in processing
            # None = process all available models, List = process only specified models
            # Examples: None, ["claude-3-opus"], ["claude-3-opus", "gpt-4o"]
            models=args.models,
            # HUMAN_RAW_FILE: Filename of raw human data in data/raw/human/rw17/
            # Examples: "rw17_collider_ce.csv", "rw17_fork_cc.csv"
            human_raw_file=args.human_file,
            # PROMPT_MAPPING_FILE: CSV file containing prompt ID mappings in data/input_llm/rw17/
            # Used for assigning proper IDs to human responses to match LLM data
            prompt_mapping_file=args.prompt_mapping,
            # ADD_REASONING_TYPES: Adds cognitive reasoning type annotations to tasks
            # True = Maps letter tasks (a,b,c...) to reasoning categories + Roman numerals
            # False = Keeps original letter task labels (a,b,c,d,e,f,g,h,i,j,k)
            # Reasoning types: Effect-Present/Absent Diagnostic, Conditional Independence, Predictive
            add_reasoning_types=not args.no_reasoning_types,
            # AGGREGATE_HUMAN_RESPONSES: Averages multiple human responses per prompt ID
            # True = Balances sample sizes (5440→336 human responses, better for statistics)
            # False = Keeps all individual responses (5440 human responses, good for variability analysis)
            aggregate_human_responses=not args.no_aggregate,
            # SAVE_INTERMEDIATE: Whether to save processed output files
            # True = Saves main CSV, Roman numerals version, aggregated version, summary
            # False = Returns data in memory only, no files saved
            save_intermediate=not args.no_save,
            # OUTPUT_DIR: Custom output directory for processed files
            # None = Uses default structure: data/processed/{mode}/rw17/{experiment}/
            # Custom = Saves to: {output_dir}/rw17/{experiment}/
            output_dir=args.output_dir,
            # PIPELINE_MODE: Determines directory structure and processing mode
            # COMBINED = data/processed/llm_with_humans/rw17/{experiment}/
            # LLM_ONLY = data/processed/llm/rw17/{experiment}/
            # HUMAN_ONLY = data/processed/humans/rw17/
            pipeline_mode=pipeline_mode,
        )

        logger.info(f"Starting data pipeline with configuration: {config}")
        logger.info(f"Pipeline mode: {pipeline_mode.value}")

        # Initialize pipeline
        paths = PathManager()
        pipeline = DataProcessingPipeline(config, paths)

        # Run appropriate pipeline mode
        if args.llm_only:
            logger.info("Running LLM-only pipeline")
            result_df = pipeline.run_llm_only_pipeline()
        elif args.human_only:
            logger.info("Running human-only pipeline")
            result_df = pipeline.run_human_only_pipeline()
        else:
            logger.info("Running complete pipeline")
            result_df = pipeline.run_complete_pipeline()

        # Print summary
        logger.info("=" * 50)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total rows processed: {len(result_df)}")
        logger.info(f"Subjects: {result_df['subject'].unique()}")
        logger.info(f"Number of unique IDs: {result_df['id'].nunique()}")

        if "task" in result_df.columns:
            logger.info(f"Tasks: {sorted(result_df['task'].unique())}")

        if "reasoning_type" in result_df.columns:
            reasoning_counts = result_df["reasoning_type"].value_counts()
            logger.info(f"Reasoning types: {dict(reasoning_counts)}")

        logger.info("Pipeline completed successfully!")

        # Optionally save a summary file
        if not args.no_save:
            # Use mode-specific directory structure for summary files
            if args.output_dir:
                if pipeline_mode == PipelineMode.HUMAN_ONLY:
                    summary_dir = Path(args.output_dir) / "rw17"
                else:
                    summary_dir = Path(args.output_dir) / "rw17" / args.experiment
            else:
                # Use default mode-specific directory structure
                processed_base = paths.base_dir / "data" / "processed"

                if pipeline_mode == PipelineMode.HUMAN_ONLY:
                    summary_dir = processed_base / "humans" / "rw17"
                elif pipeline_mode == PipelineMode.LLM_ONLY:
                    summary_dir = processed_base / "llm" / "rw17" / args.experiment
                else:  # COMBINED
                    summary_dir = (
                        processed_base / "llm_with_humans" / "rw17" / args.experiment
                    )

            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = (
                summary_dir / f"pipeline_summary_{args.version}_{args.experiment}.txt"
            )

            with open(summary_path, "w") as f:
                f.write("Data Processing Pipeline Summary\n")
                f.write("=" * 40 + "\n")
                f.write(f"Configuration: {config}\n")
                f.write(f"Total rows: {len(result_df)}\n")
                f.write(f"Subjects: {list(result_df['subject'].unique())}\n")
                f.write(f"Unique IDs: {result_df['id'].nunique()}\n")
                if "reasoning_type" in result_df.columns:
                    f.write(
                        f"Reasoning types: {dict(result_df['reasoning_type'].value_counts())}\n"
                    )

            logger.info(f"Summary saved to: {summary_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
