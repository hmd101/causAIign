"""
Numeric Only Prompt Style

Simple numeric response format - LLM provides just a number between 0-100.
Supports both plain text and XML formats for flexibility.
"""

from .base_style import BasePromptStyle


class NumericOnlyStyle(BasePromptStyle):
    """
    Numeric-only response style.

    LLM responds with a single number between 0 (very unlikely) and 100 (very likely).
    This can be either plain text or XML format.

    Args:
        use_xml: If True, use XML format <response><likelihood>VALUE</likelihood></response>
                If False, use plain text format (just the number)
    """

    def __init__(self, use_xml: bool = False):
        """
        Initialize the numeric-only style.

        Args:
            use_xml: Whether to use XML format (default: False for backward compatibility)
        """
        self.use_xml = use_xml

    def get_category(self) -> str:
        """Return category identifier for file naming."""
        if self.use_xml:
            return "numeric_xml"
        return "numeric"

    def get_prompt_instructions(self) -> str:
        """Return instructions for numeric-only responses."""
        if self.use_xml:
            xml_format = "<response><likelihood>YOUR_NUMERIC_RESPONSE_HERE</likelihood></response>"
            xml_explanation = (
                "Replace YOUR_NUMERIC_RESPONSE_HERE with your likelihood estimate between 0 (very unlikely) and 100 (very likely). "
                "DO NOT include any other information, explanation, or formatting in your response. "
                "DO NOT use Markdown, code blocks, quotation marks, or special characters."
            )
            return (
                "Return your response as raw text in one single line using this exact XML format: "
                + xml_format
                + " "
                + xml_explanation
            )
        else:
            return (
                "Please provide your answer as a single number between 0 and 100, "
                "where 0 means very unlikely and 100 means very likely. "
                "Do not include any explanations or additional text."
            )

    def get_response_parser_config(self) -> dict:
        """Return parser configuration for numeric responses."""
        if self.use_xml:
            return {
                "response_type": "xml_numeric",
                "extract_method": "xml_parsing",
                "xml_tags": {"likelihood": {"type": "numeric", "range": (0, 100)}},
                "fallback_method": "regex",
                "fallback_patterns": {
                    "likelihood": r"<likelihood>(\d{1,3})</likelihood>"
                },
            }
        else:
            return {
                "response_type": "numeric",
                "extract_method": "regex",
                "regex_pattern": r"\b(\d{1,3})\b",
                "expected_range": (0, 100),
                "fallback_method": "string_search",
            }

    def get_description(self) -> str:
        """Return description of this style."""
        format_type = "XML format" if self.use_xml else "Plain text"
        return f"Numeric-only responses: Single number between 0-100 ({format_type})"
