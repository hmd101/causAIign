# import sys
# #sys.path.append('../../../')
# from causalign.config.paths import PathManager

# # Initialize PathManager
# paths = PathManager()

from pathlib import Path


class PathManager:
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Default to project root (two levels up from this file)
            self.base_dir = Path(__file__).parent.parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

        # Main directories at project root
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        self.publication_dir = self.base_dir / "publication"

        # Data subdirectories
        self.input_llm_dir = self.data_dir / "input_llm"
        self.output_llm_dir = self.data_dir / "output_llm"  # Added for LLM outputs
        self.merged_with_human_dir = self.data_dir / "merged_with_human"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"

        # RW17 specific directories
        self.rw17_input_llm_dir = self.input_llm_dir / "rw17"
        self.rw17_merged_dir = self.merged_with_human_dir / "rw17"
        self.rw17_raw_human_dir = self.raw_data_dir / "human" / "rw17"
        self.rw17_processed_human_dir = self.processed_data_dir / "human" / "rw17"
        self.rw17_processed_llm_dir = self.processed_data_dir / "llm" / "17_rw"

    # Data paths
    def get_human_raw_data_path(self, filename):
        """Get path for raw human data files"""
        return self.rw17_raw_human_dir / filename

    def get_human_processed_data_path(self, filename):
        """Get path for processed human data files"""
        return self.rw17_processed_human_dir / filename

    def get_llm_output_path(self, experiment_name, model_name, filename):
        """Get path for LLM output files"""
        return self.output_llm_dir / experiment_name / model_name / filename

    def get_llm_processed_path(self, model_name, filename):
        """Get path for processed LLM data files"""
        return self.rw17_processed_llm_dir / model_name / filename

    def get_llm_prompt_path(self, filename):
        """Get path for LLM prompt files"""
        return self.raw_data_dir / "llm" / "prompts" / filename

    def get_llm_response_path(self, filename):
        """Get path for LLM response files"""
        return self.raw_data_dir / "llm" / "responses" / filename

    def get_merged_data_path(self, version, graph_type):
        """Get path for merged data files"""
        return self.rw17_merged_dir / f"{version}_{graph_type}_merged.csv"

    def get_prompt_path(self, version, prompt_category, graph_type):
        """Get path for prompt files"""
        return (
            self.rw17_input_llm_dir
            / f"{version}_v_{prompt_category}_LLM_prompting_{graph_type}.csv"
        )

    def get_human_prompts_coll_path(self):
        """Get path for human prompts collated file"""
        return self.rw17_merged_dir / "humans_prompts_coll.csv"

    # Results paths
    def get_model_fitting_path(self, filename):
        """Get path for model fitting results"""
        return self.results_dir / "model_fitting" / filename

    def get_line_plot_path(self, filename):
        """Get path for line plot results"""
        return self.results_dir / "line_plots" / filename

    def get_significance_path(self, filename):
        """Get path for significance test results"""
        return self.results_dir / "significance" / filename

    # Publication paths
    def get_paper_path(self, filename):
        """Get path for paper-related files"""
        return self.publication_dir / "paper" / filename

    def get_thesis_path(self, filename):
        """Get path for thesis-related files"""
        return self.publication_dir / "thesis" / filename

    def get_graph_cartoon_path(self, filename):
        """Get path for graph cartoon files"""
        return self.publication_dir / "graph_cartoons" / filename
