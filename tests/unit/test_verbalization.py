"""
Unit tests for verbalization and capitalization functionality.

This test suite validates the automatic capitalization fixes that ensure
proper grammar in generated prompts. The main focus is on the
capitalize_after_periods() function which addresses the issue where
sense variables (like "large", "high", "low") from domain dictionaries
were appearing in lowercase after periods, violating standard grammar rules.

Test coverage includes:
- Basic capitalization after periods
- Capitalization of sense variables in explanations
- Edge cases and error handling
- Integration with all verbalization functions
- Consistency across different domains and counterbalance conditions
- Full prompt generation with proper capitalization

This addresses the original issue where prompts contained text like:
"High interest rates cause low retirement savings. the good economic times..."
which is now corrected to:
"High interest rates cause low retirement savings. The good economic times..."
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from causalign.prompts.core.constants import (
    graph_structures,
    inference_tasks_rw17,
    rw_17_domain_components,
)
from causalign.prompts.core.processing import generate_prompt_dataframe
from causalign.prompts.core.verbalization import (
    capitalize_after_periods,
    verbalize_causal_mechanism,
    verbalize_domain_intro,
    verbalize_inference_task,
    verbalize_variables_section,
)


class TestCapitalizeAfterPeriods:
    """Test the capitalize_after_periods utility function."""

    def test_basic_capitalization(self):
        """Test basic capitalization after periods."""
        test_cases = [
            (
                "This is a test. this should be capitalized.",
                "This is a test. This should be capitalized.",
            ),
            (
                "First sentence. second sentence. third sentence.",
                "First sentence. Second sentence. Third sentence.",
            ),
            ("No periods here", "No periods here"),
            ("End with period. ", "End with period. "),
            (
                "Multiple spaces after.   this should work.",
                "Multiple spaces after. This should work.",
            ),
            ("Already correct. This is fine.", "Already correct. This is fine."),
        ]

        for input_text, expected in test_cases:
            result = capitalize_after_periods(input_text)
            assert result == expected, f"Failed for input: '{input_text}'"

    def test_capitalization_with_sense_variables(self):
        """Test capitalization with lowercase sense variables."""
        test_input = "large trade deficits causes low retirement savings. the loss of local manufacturing jobs means that there are people out of work."
        expected = "Large trade deficits causes low retirement savings. The loss of local manufacturing jobs means that there are people out of work."

        result = capitalize_after_periods(test_input)
        assert result == expected

    def test_edge_cases(self):
        """Test edge cases for capitalization."""
        test_cases = [
            ("", ""),  # Empty string
            (None, None),  # None input
            (".", "."),  # Single period
            ("a.", "A."),  # Single letter
            ("a. b. c.", "A. B. C."),  # Single letters
            ("First. ", "First. "),  # Period with space at end
        ]

        for input_text, expected in test_cases:
            result = capitalize_after_periods(input_text)
            assert result == expected, f"Failed for input: '{input_text}'"

    def test_non_string_input(self):
        """Test function handles non-string input gracefully."""
        assert capitalize_after_periods(123) == 123
        assert capitalize_after_periods([]) == []
        assert capitalize_after_periods({"key": "value"}) == {"key": "value"}

    def test_first_word_capitalization(self):
        """Test that the very first word is capitalized."""
        test_cases = [
            (
                "low interest rates cause high savings.",
                "Low interest rates cause high savings.",
            ),
            ("small deficits help economy.", "Small deficits help economy."),
            (
                "high unemployment affects savings. low rates help.",
                "High unemployment affects savings. Low rates help.",
            ),
        ]

        for input_text, expected in test_cases:
            result = capitalize_after_periods(input_text)
            assert result == expected, f"Failed for input: '{input_text}'"


class TestVerbalizationFunctions:
    """Test individual verbalization functions."""

    @pytest.fixture
    def sample_domain(self):
        """Sample domain for testing."""
        return rw_17_domain_components["economy"]

    @pytest.fixture
    def sample_row(self):
        """Sample dataframe row for testing."""
        from causalign.prompts.core.processing import expand_domain_to_dataframe

        domain = rw_17_domain_components["economy"]
        df = expand_domain_to_dataframe(domain)
        return df.iloc[0]  # First row

    def test_verbalize_domain_intro_capitalization(self, sample_domain):
        """Test domain introduction has proper capitalization."""
        intro = verbalize_domain_intro(sample_domain)

        # Check that intro is a string and not empty
        assert isinstance(intro, str)
        assert len(intro) > 0

        # Check that sentences start with capital letters
        sentences = intro.split(". ")
        for sentence in sentences:
            if sentence.strip():
                first_char = sentence.strip()[0]
                assert first_char.isupper(), (
                    f"Sentence should start with uppercase: '{sentence.strip()[:30]}...'"
                )

    def test_verbalize_causal_mechanism_capitalization(self, sample_domain, sample_row):
        """Test causal mechanism text has proper capitalization."""
        mechanism = verbalize_causal_mechanism(
            sample_domain,
            sample_row,
            "collider",
            graph_structures,
            indep_causes_collider=False,
        )

        # Check that mechanism is a string
        assert isinstance(mechanism, str)

        # Check sentences start with capital letters
        sentences = mechanism.split(". ")
        for sentence in sentences:
            if sentence.strip():
                first_char = sentence.strip()[0]
                assert first_char.isupper(), (
                    f"Sentence should start with uppercase: '{sentence.strip()[:30]}...'"
                )

    def test_verbalize_inference_task_capitalization(self, sample_domain, sample_row):
        """Test inference task text has proper capitalization."""
        # Add required fields to the row for inference task
        row_with_task = sample_row.copy()
        row_with_task["observation"] = "X=1, Y=1"
        row_with_task["query_node"] = "Z=1"
        row_with_task["X_sense"] = "low"
        row_with_task["Y_sense"] = "small"
        row_with_task["Z_sense"] = "high"

        task = verbalize_inference_task(
            row_with_task, sample_domain, "Please provide only a numeric response."
        )

        # Check that task is a string
        assert isinstance(task, str)

        # Check sentences start with capital letters
        sentences = task.split(". ")
        for sentence in sentences:
            if (
                sentence.strip() and not sentence.strip()[0].isdigit()
            ):  # Skip sentences starting with numbers
                first_char = sentence.strip()[0]
                assert first_char.isupper(), (
                    f"Sentence should start with uppercase: '{sentence.strip()[:30]}...'"
                )

    def test_verbalize_variables_section_capitalization(
        self, sample_domain, sample_row
    ):
        """Test variables section has proper capitalization."""
        variables = verbalize_variables_section(sample_domain, sample_row)

        # Check that variables is a string
        assert isinstance(variables, str)

        # Check sentences start with capital letters
        sentences = variables.split(". ")
        for sentence in sentences:
            if sentence.strip():
                first_char = sentence.strip()[0]
                assert first_char.isupper(), (
                    f"Sentence should start with uppercase: '{sentence.strip()[:30]}...'"
                )


class TestFullPromptGeneration:
    """Test complete prompt generation with capitalization."""

    def test_generated_prompt_capitalization(self):
        """Test that generated prompts have proper capitalization throughout."""
        # Generate a sample prompt
        economy_domain = rw_17_domain_components["economy"]

        df = generate_prompt_dataframe(
            economy_domain,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="numeric",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Test first few prompts
        for i in range(min(3, len(df))):
            prompt = df.iloc[i]["prompt"]

            # Check that prompt is a string
            assert isinstance(prompt, str)
            assert len(prompt) > 0

            # Split into sentences and check capitalization
            sentences = prompt.split(". ")

            for j, sentence in enumerate(sentences):
                if sentence.strip():
                    first_char = sentence.strip()[0]

                    # Skip sentences starting with numbers (like "0 means...")
                    if not first_char.isdigit():
                        assert first_char.isupper(), (
                            f"Sentence {j + 1} in prompt {i + 1} should start with uppercase: "
                            f"'{sentence.strip()[:50]}...'"
                        )

    def test_specific_problematic_cases(self):
        """Test specific cases that were problematic before the fix."""
        economy_domain = rw_17_domain_components["economy"]

        df = generate_prompt_dataframe(
            economy_domain,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="numeric",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Look for prompts with 'mmm' condition which should have explanations starting with sense variables
        mmm_rows = df[df["cntbl_cond"] == "mmm"]

        if len(mmm_rows) > 0:
            prompt = mmm_rows.iloc[0]["prompt"]

            # Check specifically for the causal relationships section
            parts = prompt.split("Here are the causal relationships:")
            if len(parts) > 1:
                explanations_part = parts[1].split("Suppose that")[0]

                # Should not contain lowercase explanations after periods
                sentences = explanations_part.split(". ")
                for sentence in sentences:
                    if sentence.strip() and not sentence.strip()[0].isdigit():
                        first_char = sentence.strip()[0]
                        assert first_char.isupper(), (
                            f"Explanation sentence should start with uppercase: "
                            f"'{sentence.strip()[:50]}...'"
                        )

    def test_capitalization_preserves_meaning(self):
        """Test that capitalization doesn't change the meaning of the text."""
        # Test with a known domain and configuration
        weather_domain = rw_17_domain_components["weather"]

        df = generate_prompt_dataframe(
            weather_domain,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="numeric",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        prompt = df.iloc[0]["prompt"]

        # Check that the prompt contains expected content
        assert "weather" in prompt.lower()
        assert (
            "humidity" in prompt.lower()
            or "ozone" in prompt.lower()
            or "pressure" in prompt.lower()
        )

        # Check that capitalization is consistent
        # Every sentence should start with capital letter
        sentences = prompt.split(". ")
        capitalized_count = 0
        total_sentences = 0

        for sentence in sentences:
            if sentence.strip() and not sentence.strip()[0].isdigit():
                total_sentences += 1
                if sentence.strip()[0].isupper():
                    capitalized_count += 1

        # All non-numeric sentences should be capitalized
        if total_sentences > 0:
            capitalization_ratio = capitalized_count / total_sentences
            assert capitalization_ratio >= 0.95, (
                f"Expected at least 95% of sentences to be capitalized, "
                f"got {capitalization_ratio:.2%} ({capitalized_count}/{total_sentences})"
            )


class TestCapitalizationConsistency:
    """Test capitalization consistency across different domains and conditions."""

    @pytest.mark.parametrize("domain_name", ["economy", "sociology", "weather"])
    def test_all_domains_capitalization(self, domain_name):
        """Test capitalization works for all domains."""
        domain = rw_17_domain_components[domain_name]

        df = generate_prompt_dataframe(
            domain,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="numeric",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Test a few prompts from each domain
        for i in range(min(2, len(df))):
            prompt = df.iloc[i]["prompt"]
            sentences = prompt.split(". ")

            for sentence in sentences:
                if sentence.strip() and not sentence.strip()[0].isdigit():
                    first_char = sentence.strip()[0]
                    assert first_char.isupper(), (
                        f"Domain {domain_name}, sentence should start with uppercase: "
                        f"'{sentence.strip()[:50]}...'"
                    )

    @pytest.mark.parametrize("cntbl_condition", ["ppp", "pmm", "mmp", "mmm"])
    def test_all_counterbalance_conditions_capitalization(self, cntbl_condition):
        """Test capitalization works for all counterbalance conditions."""
        economy_domain = rw_17_domain_components["economy"]

        df = generate_prompt_dataframe(
            economy_domain,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="numeric",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Filter to specific counterbalance condition
        condition_rows = df[df["cntbl_cond"] == cntbl_condition]

        if len(condition_rows) > 0:
            prompt = condition_rows.iloc[0]["prompt"]
            sentences = prompt.split(". ")

            for sentence in sentences:
                if sentence.strip() and not sentence.strip()[0].isdigit():
                    first_char = sentence.strip()[0]
                    assert first_char.isupper(), (
                        f"Condition {cntbl_condition}, sentence should start with uppercase: "
                        f"'{sentence.strip()[:50]}...'"
                    )
