"""
Custom Domain Loader

Loads user-defined custom domains from YAML files.
Allows users to create their own abstract experiments with custom variables.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging


class CustomDomainLoader:
    """
    Loads custom domain definitions from YAML files.
    
    Allows users to define their own abstract domains for experiments
    without having to modify the core codebase.
    """
    
    from typing import Optional
    def __init__(self, custom_domains_dir: 'Optional[Path]' = None):
        """
        Initialize the domain loader.
        
        Args:
            custom_domains_dir: Directory containing custom domain YAML files
                               Default: src/causalign/prompts/custom_domains/
        """
        if custom_domains_dir is None:
            custom_domains_dir = Path(__file__).parent
        self.custom_domains_dir = Path(custom_domains_dir)
        self.logger = logging.getLogger(__name__)
        
    def load_domain_from_yaml(self, yaml_file: str) -> Dict[str, Any]:
        """
        Load a custom domain from a YAML file.
        
        Args:
            yaml_file: Name of YAML file (e.g., "my_domain.yaml") 
                      or full path to YAML file
                      
        Returns:
            Dict: Domain dictionary in the format expected by create_domain_dict()
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML structure is invalid
        """
        # Handle both filename and full path
        if Path(yaml_file).is_absolute():
            yaml_path = Path(yaml_file)
        else:
            yaml_path = self.custom_domains_dir / yaml_file
            
        if not yaml_path.exists():
            raise FileNotFoundError(f"Custom domain file not found: {yaml_path}")
            
        self.logger.info(f"Loading custom domain from: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML file {yaml_path}: {e}")
            
        # Validate and convert YAML structure
        domain_dict = self._convert_yaml_to_domain_dict(yaml_data, yaml_path)
        
        self.logger.info(f"Successfully loaded custom domain: {domain_dict['domain_name']}")
        return domain_dict
        
    def _convert_yaml_to_domain_dict(self, yaml_data: Dict, yaml_path: Path) -> Dict[str, Any]:
        """
        Convert YAML data to the domain dictionary format expected by create_domain_dict().
        
        Args:
            yaml_data: Parsed YAML data
            yaml_path: Path to YAML file (for error messages)
            
        Returns:
            Dict: Domain dictionary compatible with create_domain_dict()
        """
        required_fields = ["domain_name", "introduction", "variables"]
        missing_fields = [field for field in required_fields if field not in yaml_data]
        if missing_fields:
            raise ValueError(f"Missing required fields in {yaml_path}: {missing_fields}")
            
        domain_name = yaml_data["domain_name"]
        introduction = yaml_data["introduction"]
        variables = yaml_data["variables"]
        
        # Validate variables structure
        required_vars = ["X", "Y", "Z"]
        missing_vars = [var for var in required_vars if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables in {yaml_path}: {missing_vars}")
            
        # Convert variables to the format expected by create_domain_dict()
        # create_domain_dict() expects: {var_key: {name, detailed, p_value, m_value}}
        converted_variables = {}
        for var_key in ["X", "Y", "Z"]:
            var_data = variables[var_key]
            
            # Validate variable structure
            required_var_fields = ["name", "detailed", "p_values", "m_values"]
            missing_var_fields = [field for field in required_var_fields if field not in var_data]
            if missing_var_fields:
                raise ValueError(f"Variable {var_key} missing fields in {yaml_path}: {missing_var_fields}")
                
            # Convert to the format expected by create_domain_dict
            converted_variables[var_key] = {
                "name": var_data["name"],
                "detailed": var_data["detailed"],
                "p_value": var_data["p_values"],
                "m_value": var_data["m_values"]
            }
            
        return {
            "domain_name": domain_name,
            "introduction": introduction,
            "variables": converted_variables
        }
        
    def load_all_custom_domains(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all custom domains from YAML files in the custom domains directory.
        
        Returns:
            Dict: Mapping of domain_name -> domain_dict for all found domains
        """
        custom_domains = {}
        
        if not self.custom_domains_dir.exists():
            self.logger.warning(f"Custom domains directory not found: {self.custom_domains_dir}")
            return custom_domains
            
        # Find all YAML files
        yaml_files = list(self.custom_domains_dir.glob("*.yaml")) + list(self.custom_domains_dir.glob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                domain_dict = self.load_domain_from_yaml(yaml_file)
                domain_name = domain_dict["domain_name"]
                custom_domains[domain_name] = domain_dict
                self.logger.info(f"Loaded custom domain: {domain_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load domain from {yaml_file}: {e}")
                
        self.logger.info(f"Loaded {len(custom_domains)} custom domains")
        return custom_domains
        
    def list_available_domains(self) -> List[str]:
        """List names of all available custom domains."""
        custom_domains = self.load_all_custom_domains()
        return list(custom_domains.keys())
        
    def create_domain_template(self, domain_name: str, output_file: str = None) -> Path:
        """
        Create a template YAML file for defining a new custom domain.
        
        Args:
            domain_name: Name for the new domain
            output_file: Output filename (default: {domain_name}.yaml)
            
        Returns:
            Path: Path to created template file
        """
        if output_file is None:
            output_file = f"{domain_name}.yaml"
            
        output_path = self.custom_domains_dir / output_file
        
        template = f"""# Custom Domain: {domain_name}
# Copy and modify this template to create your own abstract domain

domain_name: "{domain_name}"

introduction: >
  [Describe the domain context here. This will appear at the start of every prompt.
   Example: "Researchers in {domain_name} study the relationships between different 
   factors that influence outcomes in this field."]

variables:
  X:
    name: "[first variable name]"
    detailed: "[Detailed description of the first variable]"
    p_values:
      "1": "[positive value label]"
      "0": "[neutral/negative value label]"
    m_values:
      "1": "[opposite positive value label]" 
      "0": "[opposite neutral/negative value label]"
      
  Y:
    name: "[second variable name]"
    detailed: "[Detailed description of the second variable]"
    p_values:
      "1": "[positive value label]"
      "0": "[neutral/negative value label]"
    m_values:
      "1": "[opposite positive value label]"
      "0": "[opposite neutral/negative value label]"
      
  Z:
    name: "[outcome variable name]"
    detailed: "[Detailed description of the outcome variable]"
    p_values:
      "1": "[positive outcome label]"
      "0": "[neutral/negative outcome label]"
    m_values:
      "1": "[opposite positive outcome label]"
      "0": "[opposite neutral/negative outcome label]"

# Optional: Add custom explanations for causal relationships
# explanations:
#   X_causes_Z:
#     p_p: "When X is positive and Z is positive, explain the relationship..."
#     m_m: "When X is negative and Z is negative, explain the relationship..."
"""
        
        with open(output_path, 'w') as f:
            f.write(template)
            
        self.logger.info(f"Created domain template: {output_path}")
        return output_path
