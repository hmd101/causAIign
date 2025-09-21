"""
Prompt Styles Module

This module contains different prompt styles that define how LLMs should respond.
Each style specifies the response format and instructions.
"""

from .base_style import BasePromptStyle
from .chain_of_thought import ChainOfThoughtStyle
from .confidence import ConfidenceStyle
from .numeric_only import NumericOnlyStyle

__all__ = [
    "BasePromptStyle",
    "NumericOnlyStyle",
    "ConfidenceStyle",
    "ChainOfThoughtStyle",
]
