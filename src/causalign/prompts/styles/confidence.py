"""
Confidence Rating Prompt Style

XML-based response format where LLM provides both likelihood and confidence ratings.
"""

from .base_style import BasePromptStyle


class ConfidenceStyle(BasePromptStyle):
    """
    Confidence rating style with XML response format.

    LLM responds with both likelihood estimate and confidence level in XML format:
    <response>
        <likelihood>75</likelihood>
        <confidence>80</confidence>
    </response>
    """

    def get_category(self) -> str:
        """Return category identifier for file naming."""
        return "numeric-conf"

    def get_prompt_instructions(self) -> str:
        """Return instructions for confidence-rating responses."""
        xml_format = "<response><likelihood>YOUR_NUMERIC_RESPONSE_HERE</likelihood><confidence>YOUR_CONFIDENCE_SCORE_HERE</confidence></response>"

        xml_explanation = (
            "Replace YOUR_CONFIDENCE_SCORE_HERE with a number between 0 (very uncertain about your likelihood estimate) and 100 (very certain about your likelihood estimate)."
            "DO NOT include any other information, explanation, or formatting in your response. "
            "DO NOT use Markdown, code blocks, quotation marks, or special characters."
        )

        return (
            "Provide your likelihood estimate and confidence level about your likelihood estimate (how certain you are about your likelihood estimate between 0 and 100 with 0 being very uncertain and 100 being very certain about your likelihood estimate). "
            "Return your response as raw text in one single line using this exact XML format: "
            + xml_format
            + " "
            + xml_explanation
        )

    def get_response_parser_config(self) -> dict:
        """Return parser configuration for XML responses."""
        return {
            "response_type": "xml",
            "extract_method": "xml_parsing",
            "xml_tags": {
                "reasoning": {"type": "text", "required": True},
                "likelihood": {"type": "numeric", "range": (0, 100)},
                "confidence": {"type": "numeric", "range": (0, 100)},
            },
            "fallback_method": "regex",
            "fallback_patterns": {
                "reasoning": r"<reasoning>(.*?)</reasoning>",
                "likelihood": r"<likelihood>(\d{1,3})</likelihood>",
                "confidence": r"<confidence>(\d{1,3})</confidence>",
            },
        }

    def get_description(self) -> str:
        """Return description of this style."""
        return "Confidence ratings: XML format with likelihood + confidence scores"
