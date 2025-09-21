"""
Chain of Thought Prompt Style

Combines reasoning explanation with confidence ratings in XML format.
Supports configurable reasoning length limits.
"""

from .base_style import BasePromptStyle


class ChainOfThoughtStyle(BasePromptStyle):
    """
    Chain of thought style with XML response format and optional reasoning length limits.

    LLM provides step-by-step reasoning followed by likelihood and confidence:
    <response>
        <reasoning>Step by step explanation...</reasoning>
        <likelihood>75</likelihood>
        <confidence>80</confidence>
    </response>

    Args:
        max_reasoning_words: Maximum words allowed in reasoning (default: None = unlimited)
        max_reasoning_tokens: Maximum tokens allowed in reasoning (default: None = unlimited)
    """

    def __init__(
        self, max_reasoning_words: int = None, max_reasoning_tokens: int = None
    ):
        """
        Initialize the chain of thought style with optional reasoning constraints.

        Args:
            max_reasoning_words: Maximum words in reasoning (None = unlimited)
            max_reasoning_tokens: Maximum tokens in reasoning (None = unlimited)
        """
        self.max_reasoning_words = max_reasoning_words
        self.max_reasoning_tokens = max_reasoning_tokens

    def get_category(self) -> str:
        """Return category identifier for file naming."""
        if self.max_reasoning_words:
            return f"CoT-{self.max_reasoning_words}w"
        elif self.max_reasoning_tokens:
            return f"CoT-{self.max_reasoning_tokens}t"
        return "CoT"

    def get_prompt_instructions(self) -> str:
        """Return instructions for chain-of-thought responses."""
        xml_format = "<response><explanation>YOUR_STEP_BY_STEP_REASONING</explanation><likelihood>YOUR_NUMERIC_RESPONSE_HERE</likelihood><confidence>YOUR_CONFIDENCE_SCORE_HERE</confidence></response>"

        # Build reasoning constraints instruction
        reasoning_constraint = ""
        if self.max_reasoning_words and self.max_reasoning_tokens:
            reasoning_constraint = f" (maximum {self.max_reasoning_words} words or {self.max_reasoning_tokens} tokens)"
        elif self.max_reasoning_words:
            reasoning_constraint = f" (maximum {self.max_reasoning_words} words)"
        elif self.max_reasoning_tokens:
            reasoning_constraint = f" (maximum {self.max_reasoning_tokens} tokens)"

        xml_explanation = (
            f"Replace YOUR_STEP_BY_STEP_REASONING with your concise reasoning process{reasoning_constraint}. "
            "Replace YOUR_NUMERIC_RESPONSE_HERE with your likelihood estimate between 0 (very unlikely) and 100 (very likely). "
            "Replace YOUR_CONFIDENCE_SCORE_HERE with a number between 0 (very uncertain about your likelihood estimate) and 100 (very certain about your likelihood estimate). "
            "DO NOT include any other information, explanation, or formatting outside the XML. "
            "DO NOT use Markdown, code blocks, quotation marks, or special characters."
        )

        return (
            "First, think through this step by step and explain your reasoning. "
            "Then provide your likelihood estimate and confidence level about your likelihood estimate (how certain you are about your likelihood estimate between 0 and 100 with 0 being very uncertain and 100 being very certain about your likelihood estimate). "
            "Return your response as raw text in one single line using this exact XML format: "
            + xml_format
            + " "
            + xml_explanation
        )

    def get_response_parser_config(self) -> dict:
        """Return parser configuration for CoT XML responses."""
        config = {
            "response_type": "xml_cot",
            "extract_method": "xml_parsing",
            "xml_tags": {
                "explanation": {"type": "text", "required": True},
                "likelihood": {"type": "numeric", "range": (0, 100)},
                "confidence": {"type": "numeric", "range": (0, 100)},
            },
            "fallback_method": "regex",
            "fallback_patterns": {
                "explanation": r"<explanation>(.*?)</explanation>",
                "likelihood": r"<likelihood>(\d{1,3})</likelihood>",
                "confidence": r"<confidence>(\d{1,3})</confidence>",
            },
        }

        # Add length constraints to reasoning tag if specified
        if self.max_reasoning_words:
            config["xml_tags"]["reasoning"]["max_words"] = self.max_reasoning_words
        if self.max_reasoning_tokens:
            config["xml_tags"]["reasoning"]["max_tokens"] = self.max_reasoning_tokens

        return config

    def get_description(self) -> str:
        """Return description of this style."""
        description = (
            "Chain of thought: Reasoning + likelihood + confidence in XML format"
        )

        if self.max_reasoning_words and self.max_reasoning_tokens:
            description += f" (≤{self.max_reasoning_words} words or {self.max_reasoning_tokens} tokens)"
        elif self.max_reasoning_words:
            description += f" (≤{self.max_reasoning_words} words)"
        elif self.max_reasoning_tokens:
            description += f" (≤{self.max_reasoning_tokens} tokens)"

        return description
