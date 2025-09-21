"""
Abstract Prompt Generator

Generates prompts for abstract domains without human baseline data.
Supports both predefined and custom user-defined domains.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.abstract_domains import ABSTRACT_DOMAINS
from ..custom_domains.domain_loader import CustomDomainLoader
from .base_generator import BasePromptGenerator


class AbstractGenerator(BasePromptGenerator):
    """
    Abstract prompt generator for non-human-study experiments.

    Creates prompts for abstract/fantasy domains that don't have
    corresponding human experimental data.

    Supports both predefined domains and custom user-defined domains.
    """

    def __init__(
        self,
        version: str,
        output_dir: Optional[Path] = None,
        custom_domains: Optional[List[str]] = None,
        custom_domain_files: Optional[List[str]] = None,
        custom_domains_dir: Optional[Path] = None,
        overlays_yaml: Optional[str] = None,
    ):
        super().__init__(version, output_dir)

        # Set up domain components
        self.domain_components = ABSTRACT_DOMAINS.copy()
        self._graph_types = ["collider"]

        # Overlay support for abstract domains
        if overlays_yaml:
            import yaml
            from ..core.overlays import create_overloaded_domains
            try:
                with open(overlays_yaml, "r") as f:
                    overlays = yaml.safe_load(f)
                overloaded_domains = create_overloaded_domains(self.domain_components, overlays)
                self.domain_components.update(overloaded_domains)
                print(f"✅ Loaded {len(overloaded_domains)} overloaded abstract domains from overlays: {list(overloaded_domains.keys())}")
            except Exception as e:
                print(f"❌ Failed to load overlays YAML {overlays_yaml}: {e}")

        # Load custom domains if specified
        if custom_domain_files or custom_domains_dir:
            loader = CustomDomainLoader(custom_domains_dir)

            if custom_domain_files:
                for domain_file in custom_domain_files:
                    try:
                        custom_domain = loader.load_domain_from_yaml(domain_file)
                        domain_name = custom_domain["domain_name"]
                        self.domain_components[domain_name] = custom_domain
                        print(f"✅ Loaded custom domain: {domain_name}")
                    except Exception as e:
                        print(f"❌ Failed to load {domain_file}: {e}")
            else:
                custom_domains_dict = loader.load_all_custom_domains()
                self.domain_components.update(custom_domains_dict)
                if custom_domains_dict:
                    print(f"✅ Loaded {len(custom_domains_dict)} custom domains: {list(custom_domains_dict.keys())}")

        # Set default domains
        self.custom_domains = custom_domains or self._get_available_domains()

    def _get_available_domains(self) -> List[str]:
        """Get all available domain names (predefined + custom)."""
        return list(self.domain_components.keys())

    def get_domains(self) -> List[str]:
        """Return custom abstract domains."""
        return self.custom_domains

    def get_graph_types(self) -> List[str]:
        """Return graph types for abstract experiments."""
        return getattr(self, "_graph_types", ["collider"])

    def get_counterbalance_conditions(self) -> List[str]:
        """Return counterbalance conditions for abstract experiments."""
        return ["ppp", "pmm", "mmp", "mpm"]  # Keep all for consistency

    def with_domains(self, domains: List[str]) -> "AbstractGenerator":
        """Create generator with custom domains."""
        # Validate that all requested domains are available
        available_domains = self._get_available_domains()
        missing_domains = [d for d in domains if d not in available_domains]
        if missing_domains:
            raise ValueError(
                f"Unknown domains: {missing_domains}. Available: {available_domains}"
            )

        new_generator = AbstractGenerator(self.version, self.output_dir, domains)
        new_generator.domain_components = self.domain_components
        new_generator._graph_types = getattr(self, "_graph_types", ["collider"])
        return new_generator

    def with_graph_types(self, graph_types: List[str]) -> "AbstractGenerator":
        """Create generator with different graph types."""
        new_generator = AbstractGenerator(
            self.version, self.output_dir, self.custom_domains
        )
        new_generator.domain_components = self.domain_components
        new_generator._graph_types = graph_types
        return new_generator

    def generate_domain_prompts(
        self,
        domain_name: str,
        graph_type: str,
        prompt_style,
        indep_causes_collider: bool = False,
    ):
        """
        Override to handle abstract domains that may not exist in components.
        """
        if domain_name not in self.domain_components:
            available = list(self.domain_components.keys())
            raise ValueError(
                f"Abstract domain '{domain_name}' not found. Available: {available}"
            )

        return super().generate_domain_prompts(
            domain_name, graph_type, prompt_style, indep_causes_collider
        )

    def list_available_domains(self) -> List[str]:
        """List all available domains (predefined + loaded custom)."""
        return self._get_available_domains()

    def get_domain_info(self, domain_name: str) -> Dict[str, Any]:
        """Get information about a specific domain."""
        if domain_name not in self.domain_components:
            raise ValueError(f"Domain '{domain_name}' not found")
        return self.domain_components[domain_name]

    def get_description(self) -> str:
        """Return description of this generator."""
        return (
            f"Abstract generator: "
            f"domains={self.get_domains()}, "
            f"graphs={self.get_graph_types()}, "
            f"no human baseline required"
        )


def create_custom_abstract_generator(
    version: str,
    custom_domain_files: Optional[List[str]] = None,
    custom_domains_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    overlays_yaml: Optional[str] = None,
) -> AbstractGenerator:
    """
    Convenience function to create an AbstractGenerator with custom domains.

    Args:
        version: Version identifier
        custom_domain_files: List of YAML files with domain definitions
        custom_domains_dir: Directory containing custom domain YAML files
        output_dir: Output directory for generated prompts
        overlays_yaml: overlays YAML file for overloaded domains

    Returns:
        AbstractGenerator configured with custom domains

    Example:
        generator = create_custom_abstract_generator(
            version="9",
            custom_domain_files=["my_robotics.yaml", "my_biology.yaml"],
            output_dir="my_prompts/",
            overlays_yaml="overlays.yaml"
        )
    """
    output_path = Path(output_dir) if output_dir else None
    domains_path = Path(custom_domains_dir) if custom_domains_dir else None

    return AbstractGenerator(
        version=version,
        output_dir=output_path,
        custom_domain_files=custom_domain_files,
        custom_domains_dir=domains_path,
        overlays_yaml=overlays_yaml,
    )
