"""
RW17 Prompt Generator

Human-study-based prompt generator that matches existing human experimental data.
This replicates the logic from the Jupyter notebooks.
"""

from typing import List

from .base_generator import BasePromptGenerator


class RW17Generator(BasePromptGenerator):
    """
    RW17 human-study-based prompt generator.
    
    Generates prompts that match the experimental conditions from the RW17 human study.
    Uses the exact domains, tasks, and counterbalance conditions from the original study.
    """
    
    def __init__(self, version: str, output_dir=None):
        super().__init__(version, output_dir)
        # Optional attrs populated by with_* methods
        self._graph_types = ["collider"]
        self._domains = ["economy", "sociology", "weather"]
    
    def get_domains(self) -> List[str]:
        """Return domains (default to RW17 trio unless overridden)."""
        return self._domains
        
    def get_graph_types(self) -> List[str]:
        """Return configured graph types."""
        return self._graph_types
        
    def get_counterbalance_conditions(self) -> List[str]:
        """Return the four counterbalance conditions used in RW17."""
        return ["ppp", "pmm", "mmp", "mpm"]
        
    def with_graph_types(self, graph_types: List[str]) -> "RW17Generator":
        """
        Create a new generator with different graph types.
        
        This allows using the same RW17 domains and conditions
        but for different graph topologies (collider, fork, chain).
        
        Args:
            graph_types: List of graph types to generate
            
        Returns:
            New RW17Generator with updated graph types
        """
        new_generator = RW17Generator(self.version, self.output_dir)
        new_generator._graph_types = list(graph_types)
        new_generator._domains = list(self._domains)
        # Preserve any modified domain components (e.g., overlays applied)
        new_generator.domain_components = self.domain_components
        new_generator.graph_structures = self.graph_structures
        new_generator.inference_tasks = self.inference_tasks
        return new_generator
        
    def get_description(self) -> str:
        """Return description of this generator."""
        return (
            f"RW17 human-study-based generator: "
            f"domains={self.get_domains()}, "
            f"graphs={self.get_graph_types()}, "
            f"conditions={len(self.get_counterbalance_conditions())} counterbalance"
        )

    def with_domains(self, domains: List[str]) -> "RW17Generator":
        """Create a new generator with specific domains."""
        new_generator = RW17Generator(self.version, self.output_dir)
        new_generator._graph_types = list(self._graph_types)
        new_generator._domains = list(domains)
        # Preserve any modified domain components (e.g., overlays applied)
        new_generator.domain_components = self.domain_components
        new_generator.graph_structures = self.graph_structures
        new_generator.inference_tasks = self.inference_tasks
        return new_generator
