#!/usr/bin/env python3
"""
Example usage of the new data processing pipeline

This script demonstrates how to use the refactored data processing pipeline
both programmatically and with different configurations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.causalign.config.paths import PathManager
from src.causalign.data_handling.processors.pipeline import (
    DataProcessingPipeline,
    PipelineConfig,
)


def example_complete_pipeline():
    """Example: Run the complete pipeline with default settings"""
    print("Running complete pipeline example...")

    # Create configuration
    config = PipelineConfig(
        experiment_name="pilot_study",
        version="6",
        graph_type="collider",
        models=None,  # Process all models
        add_reasoning_types=True,
        aggregate_human_responses=True,
        save_intermediate=True,
    )

    # Initialize and run pipeline
    pipeline = DataProcessingPipeline(config)
    result_df = pipeline.run_complete_pipeline()

    print(f"✅ Complete pipeline finished: {len(result_df)} rows processed")
    return result_df


def example_llm_only_pipeline():
    """Example: Process only specific LLM models"""
    print("Running LLM-only pipeline example...")

    config = PipelineConfig(
        experiment_name="pilot_study",
        version="6",
        models=["claude-3-opus-20240229", "gpt-4o"],  # Only these models
        add_reasoning_types=True,
        save_intermediate=False,  # Don't save intermediate files
    )

    pipeline = DataProcessingPipeline(config)
    result_df = pipeline.run_llm_only_pipeline()

    print(f"✅ LLM-only pipeline finished: {len(result_df)} rows processed")
    print(f"Models processed: {result_df['subject'].unique()}")
    return result_df


def example_human_only_pipeline():
    """Example: Process only human data"""
    print("Running human-only pipeline example...")

    config = PipelineConfig(
        human_raw_file="rw17_collider_ce.csv",
        prompt_mapping_file="6_v_numeric_LLM_prompting_collider.csv",
        graph_type="collider",
        save_intermediate=False,
    )

    pipeline = DataProcessingPipeline(config)
    result_df = pipeline.run_human_only_pipeline()

    print(f"✅ Human-only pipeline finished: {len(result_df)} rows processed")
    return result_df


def example_custom_processing():
    """Example: Custom processing with individual processors"""
    print("Running custom processing example...")

    from src.causalign.data_handling.processors.human_processor import (
        HumanDataProcessor,
    )
    from src.causalign.data_handling.processors.llm_processor import (
        LLMResponseProcessor,
    )

    paths = PathManager()

    # Process LLM data for specific experiment
    llm_processor = LLMResponseProcessor(paths)
    llm_data = llm_processor.process_experiment(
        experiment_name="pilot_study",
        version="6",
        models=["gpt-4o"],  # Only GPT-4O
    )

    # Save processed LLM data
    if not llm_data.empty:
        output_path = llm_processor.save_processed_data(
            llm_data, "custom_gpt4o_processed.csv", "pilot_study"
        )
        print(f"✅ Custom LLM processing finished: {output_path}")

    # Process human data separately
    human_processor = HumanDataProcessor(paths)
    human_data = human_processor.process_human_data(
        raw_filename="rw17_collider_ce.csv",
        prompt_mapping_file="6_v_numeric_LLM_prompting_collider.csv",
        graph_type="collider",
    )

    # Save processed human data
    if not human_data.empty:
        human_path = human_processor.save_processed_human_data(
            human_data, "custom_human_processed.csv"
        )
        print(f"✅ Custom human processing finished: {human_path}")

    return llm_data, human_data


def main():
    """Run all examples"""
    print("🚀 Data Processing Pipeline Examples")
    print("=" * 50)

    try:
        # Example 1: Complete pipeline
        print("\n1. Complete Pipeline Example")
        print("-" * 30)
        complete_data = example_complete_pipeline()

        # Example 2: LLM-only pipeline
        print("\n2. LLM-Only Pipeline Example")
        print("-" * 30)
        llm_data = example_llm_only_pipeline()

        # Example 3: Human-only pipeline
        print("\n3. Human-Only Pipeline Example")
        print("-" * 30)
        human_data = example_human_only_pipeline()

        # Example 4: Custom processing
        print("\n4. Custom Processing Example")
        print("-" * 30)
        custom_llm, custom_human = example_custom_processing()

        print("\n✅ All examples completed successfully!")

        # Print summary
        print("\n📊 SUMMARY")
        print("=" * 50)
        print(f"Complete pipeline: {len(complete_data)} rows")
        print(f"LLM-only pipeline: {len(llm_data)} rows")
        print(f"Human-only pipeline: {len(human_data)} rows")
        print(f"Custom LLM processing: {len(custom_llm)} rows")
        print(f"Custom human processing: {len(custom_human)} rows")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
