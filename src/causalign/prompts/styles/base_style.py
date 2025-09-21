"""
Base Prompt Style

Abstract base class defining the interface for prompt styles.
Each style specifies how LLMs should format their responses.
"""

import abc


class BasePromptStyle(abc.ABC):
    """
    Abstract base class for prompt styles.
    
    Each prompt style defines:
    1. Response format (XML, plain text, etc.)
    2. Instructions for the LLM
    3. Category identifier for file naming
    """
    
    @abc.abstractmethod
    def get_category(self) -> str:
        """
        Return the category identifier for this style.
        
        Used in file naming: {version}_v_{category}_LLM_prompting_{graph}.csv
        
        Examples:
        - "numeric"
        - "numeric-conf" 
        - "CoT"
        """
        pass
        
    @abc.abstractmethod
    def get_prompt_instructions(self) -> str:
        """
        Return the prompt instructions that tell the LLM how to respond.
        
        This is appended to the domain content and inference task.
        Should include:
        - Response format specification
        - Examples if needed
        - Constraints and requirements
        """
        pass
        
    @abc.abstractmethod
    def get_response_parser_config(self) -> dict:
        """
        Return configuration for parsing LLM responses.
        
        Used by the data processing pipeline to extract structured data
        from LLM responses in this format.
        
        Returns:
            dict: Configuration with parsing instructions
        """
        pass
        
    def get_description(self) -> str:
        """
        Return a human-readable description of this style.
        
        Used for documentation and experiment tracking.
        """
        return f"Prompt style: {self.get_category()}"
