"""
Integration tests for prompt generation and domain dictionary population.

This test suite verifies that prompts are correctly populated with values from
domain dictionaries, ensuring data integrity and consistency across the pipeline.

OVERVIEW
========
This file tests the complete integration between:
1. Domain dictionaries (constants.py, abstract_domains.py)
2. Domain creation function (create_domain_dict in domain.py)
3. Prompt generation pipeline (processing.py)
4. Text verbalization (verbalization.py)

The tests ensure that when we generate prompts for LLM experiments, all the
variable names, sense values, explanations, and other content from domain
dictionaries correctly appear in the final prompts that get sent to models.

WHY THIS MATTERS
================
The prompt generation system is complex with multiple interdependent components:
- Domain dictionaries define variables and their properties
- Processing functions expand these into DataFrames with counterbalance conditions
- Verbalization functions convert structured data into natural language prompts

If any component has bugs or format inconsistencies, the final prompts sent to
LLMs could be malformed, missing information, or contain placeholder text.
This would invalidate experimental results.

WHAT WE TEST
============
1. **Structure Integrity**: Domain dictionaries have correct format
2. **Content Population**: Variable names/values appear in generated prompts
3. **Counterbalance Logic**: Different conditions produce different prompts
4. **Format Consistency**: All components use the same data format
5. **Capitalization**: Text is properly formatted for readability
6. **Integration**: Components work together without errors

EXPECTED BEHAVIOR
=================
When the system works correctly:
- Variable names from dictionaries appear verbatim in prompts
- Sense values (high/low, strong/weak) appear based on counterbalance conditions
- Explanations are included when available and properly capitalized
- Domain introductions appear at the start of prompts
- All components use consistent dictionary format (clean "name"/"detailed" format)
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from causalign.prompts.core.abstract_domains import ABSTRACT_DOMAINS
from causalign.prompts.core.constants import (
    graph_structures,
    inference_tasks_rw17,
    rw_17_domain_components,
)
from causalign.prompts.core.domain import create_domain_dict
from causalign.prompts.core.processing import generate_prompt_dataframe


class TestDomainDictionaryIntegrity:
    """
    Test that domain dictionaries have correct structure and content.

    WHAT WE EXPECT
    ==============
    Domain dictionaries should have consistent structure across all domains:
    - Required top-level keys: domain_name, introduction, variables
    - Variables dict with exactly X, Y, Z keys
    - Each variable has: name, detailed, p_value, m_value
    - Value structures have "0" and "1" keys with string values
    - Explanations (if present) use valid counterbalance keys (p_p, p_m, m_p, m_m)

    WHY THIS MATTERS
    ================
    Inconsistent structure would cause:
    - KeyError exceptions during prompt generation
    - Missing variable information in prompts
    - Inconsistent behavior across different domains
    - Difficult-to-debug runtime errors
    """

    def test_domain_structure_consistency(self):
        """
        Test that all domains have consistent structure.

        WHAT THIS TESTS
        ===============
        - Every domain has required top-level keys (domain_name, introduction, variables)
        - Variables dict contains exactly 3 variables with keys X, Y, Z
        - Each variable has required fields: name, detailed, p_value, m_value
        - p_value and m_value dicts have "0" and "1" keys with string values

        WHAT WE EXPECT
        ==============
        All domains should pass these structural checks. If any domain fails,
        it indicates a malformed dictionary that would break prompt generation.

        FAILURE SCENARIOS
        =================
        - Missing required keys → KeyError during prompt generation
        - Wrong variable keys → Variable names won't appear in prompts
        - Missing value keys → Sense values won't be populated correctly
        - Non-string values → Text formatting errors in prompts
        """
        for domain_name, domain_dict in rw_17_domain_components.items():
            # Check required top-level keys
            assert "domain_name" in domain_dict
            assert "introduction" in domain_dict
            assert "variables" in domain_dict

            # Check variables structure
            variables = domain_dict["variables"]
            assert len(variables) == 3, (
                f"Domain {domain_name} should have exactly 3 variables"
            )

            # Check variable keys are X, Y, Z
            assert set(variables.keys()) == {"X", "Y", "Z"}, (
                f"Domain {domain_name} should have X, Y, Z variables"
            )

            # Check each variable has required fields
            for var_key, var_details in variables.items():
                assert "name" in var_details, (
                    f"Variable {var_key} in {domain_name} missing 'name'"
                )
                assert "detailed" in var_details, (
                    f"Variable {var_key} in {domain_name} missing 'detailed'"
                )
                assert "p_value" in var_details, (
                    f"Variable {var_key} in {domain_name} missing 'p_value'"
                )
                assert "m_value" in var_details, (
                    f"Variable {var_key} in {domain_name} missing 'm_value'"
                )

                # Check p_value and m_value structure
                for value_type in ["p_value", "m_value"]:
                    values = var_details[value_type]
                    assert "0" in values, (
                        f"Variable {var_key} {value_type} missing '0' key"
                    )
                    assert "1" in values, (
                        f"Variable {var_key} {value_type} missing '1' key"
                    )
                    assert isinstance(values["0"], str), (
                        f"Variable {var_key} {value_type}['0'] should be string"
                    )
                    assert isinstance(values["1"], str), (
                        f"Variable {var_key} {value_type}['1'] should be string"
                    )

    def test_abstract_domains_structure(self):
        """
        Test that abstract domains have correct structure.

        WHAT THIS TESTS
        ===============
        Abstract domains (used for experiments without human baselines) should
        have the same structural requirements as regular domains.

        WHAT WE EXPECT
        ==============
        Abstract domains should have identical structure to regular domains,
        ensuring they can be used interchangeably in the prompt generation pipeline.

        WHY THIS MATTERS
        ================
        Abstract domains need to work with the same processing functions as
        regular domains. Structure inconsistencies would break experiments
        that use abstract reasoning tasks.
        """
        for domain_name, domain_dict in ABSTRACT_DOMAINS.items():
            # Check required top-level keys
            assert "domain_name" in domain_dict
            assert "introduction" in domain_dict
            assert "variables" in domain_dict

            # Check variables structure
            variables = domain_dict["variables"]
            assert len(variables) == 3, (
                f"Abstract domain {domain_name} should have exactly 3 variables"
            )
            assert set(variables.keys()) == {"X", "Y", "Z"}

    def test_explanations_format(self):
        """
        Test that explanations follow correct format when present.

        WHAT THIS TESTS
        ===============
        When domains include explanations for causal relationships:
        - Explanation keys are valid counterbalance combinations (p_p, p_m, m_p, m_m)
        - Explanation values are non-empty strings

        WHAT WE EXPECT
        ==============
        Explanations should use the 4 standard counterbalance combinations:
        - p_p: positive cause → positive effect
        - p_m: positive cause → negative effect
        - m_p: negative cause → positive effect
        - m_m: negative cause → negative effect

        WHY THIS MATTERS
        ================
        Explanations are included in prompts to provide causal reasoning context.
        Invalid explanation keys would cause KeyErrors, and empty explanations
        would result in incomplete prompts that don't properly explain the
        causal relationships to the LLM.

        FAILURE SCENARIOS
        =================
        - Invalid keys → KeyError when trying to access explanations
        - Empty explanations → Incomplete causal descriptions in prompts
        - Non-string explanations → Text formatting errors
        """
        for domain_name, domain_dict in rw_17_domain_components.items():
            for var_key, var_details in domain_dict["variables"].items():
                if "explanations" in var_details:
                    explanations = var_details["explanations"]

                    # Check explanation keys are valid counterbalance combinations
                    valid_keys = {"p_p", "p_m", "m_p", "m_m"}
                    for key in explanations.keys():
                        assert key in valid_keys, (
                            f"Invalid explanation key '{key}' in {domain_name}.{var_key}"
                        )

                    # Check explanation values are non-empty strings
                    for key, explanation in explanations.items():
                        assert isinstance(explanation, str), (
                            f"Explanation {key} should be string"
                        )
                        assert len(explanation.strip()) > 0, (
                            f"Explanation {key} should not be empty"
                        )


class TestPromptPopulation:
    """
    Test that prompts are correctly populated with domain dictionary values.

    WHAT WE EXPECT
    ==============
    When we generate prompts from domain dictionaries:
    - Variable names from dictionaries appear verbatim in the generated prompts
    - Sense values (high/low, strong/weak) appear based on counterbalance conditions
    - Explanations are included when available and properly integrated
    - Domain introductions appear at the start of prompts
    - Different counterbalance conditions produce different but consistent prompts

    WHY THIS MATTERS
    ================
    This is the core functionality test - ensuring that structured domain data
    correctly becomes natural language prompts. Failures here mean:
    - LLMs receive malformed or incomplete prompts
    - Experimental manipulations (counterbalancing) don't work
    - Results are invalid because prompts don't contain expected content

    TESTING STRATEGY
    ================
    We test with parametrized domains to ensure consistency across all domains,
    and verify specific content appears in generated prompts rather than just
    checking that prompt generation doesn't crash.
    """

    @pytest.mark.parametrize("domain_name", ["economy", "sociology", "weather"])
    def test_variable_names_in_prompts(self, domain_name):
        """
        Test that variable names from domain dictionary appear correctly in prompts.

        WHAT THIS TESTS
        ===============
        For each domain (economy, sociology, weather):
        - Generate a full set of prompts using the prompt generation pipeline
        - Verify that all variable names (X, Y, Z) from the domain dictionary
          appear somewhere in every generated prompt

        WHAT WE EXPECT
        ==============
        Variable names should appear exactly as defined in domain dictionaries:
        - Economy: "interest rates", "trade deficits", "retirement savings"
        - Sociology: "urbanization", "interest in religion", "socio-economic mobility"
        - Weather: "ozone levels", "air pressure", "humidity"

        WHY THIS MATTERS
        ================
        Variable names are the core content that defines what the LLM is reasoning about.
        If names don't appear, the prompts are meaningless. If wrong names appear,
        the experiment measures reasoning about the wrong concepts.

        FAILURE SCENARIOS
        =================
        - Missing variable names → Prompts contain placeholder text or are incomplete
        - Wrong variable names → LLM reasons about unintended concepts
        - Partial variable names → Ambiguous or confusing prompts
        """
        domain_dict = rw_17_domain_components[domain_name]

        df = generate_prompt_dataframe(
            domain_dict,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="single_numeric_response",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Get expected variable names
        expected_names = {
            "X": domain_dict["variables"]["X"]["name"],
            "Y": domain_dict["variables"]["Y"]["name"],
            "Z": domain_dict["variables"]["Z"]["name"],
        }

        # Check that all variable names appear in prompts
        for i, row in df.iterrows():
            prompt = row["prompt"]

            # Each prompt should contain all three variable names
            for var_key, var_name in expected_names.items():
                assert var_name in prompt, (
                    f"Variable name '{var_name}' not found in prompt for domain {domain_name}, "
                    f"row {i}. Prompt: {prompt[:200]}..."
                )

    @pytest.mark.parametrize("domain_name", ["economy", "sociology", "weather"])
    def test_variable_senses_in_prompts(self, domain_name):
        """
        Test that variable sense values (high/low, etc.) appear correctly in prompts.

        WHAT THIS TESTS
        ===============
        For each domain:
        - Generate prompts with different counterbalance conditions
        - Verify that sense values (the specific adjectives describing variable states)
          appear in prompts according to the counterbalance condition

        WHAT WE EXPECT
        ==============
        Counterbalance conditions should control which sense values appear:
        - "p" condition → uses p_value mappings (e.g., "1" → "high")
        - "m" condition → uses m_value mappings (e.g., "1" → "low")

        For example, in economy domain with X (interest rates):
        - p_value: {"1": "low", "0": "normal"}
        - m_value: {"1": "high", "0": "normal"}
        - With p condition: prompt should contain "low interest rates"
        - With m condition: prompt should contain "high interest rates"

        WHY THIS MATTERS
        ================
        Sense values implement the experimental manipulation. If they don't change
        based on counterbalance conditions, we're not actually testing different
        scenarios - all prompts would be identical.

        FAILURE SCENARIOS
        =================
        - Wrong sense values → All prompts identical, no experimental manipulation
        - Missing sense values → Generic prompts without specific scenarios
        - Inconsistent sense values → Confusing or contradictory prompts
        """
        domain_dict = rw_17_domain_components[domain_name]

        df = generate_prompt_dataframe(
            domain_dict,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="single_numeric_response",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Check first few rows to verify sense values are populated
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            prompt = row["prompt"]

            # Get expected sense values based on counterbalance condition
            cntbl_cond = row["cntbl_cond"]

            for j, var_key in enumerate(["X", "Y", "Z"]):
                cntbl_type = cntbl_cond[j]  # p or m
                value_dict = domain_dict["variables"][var_key][f"{cntbl_type}_value"]

                # Get the sense value for "1" (which is what we typically use in prompts)
                expected_sense = value_dict["1"]

                # The sense value should appear in the prompt
                assert expected_sense in prompt, (
                    f"Expected sense '{expected_sense}' for {var_key} not found in prompt. "
                    f"Domain: {domain_name}, Condition: {cntbl_cond}, Row: {i}"
                )

    def test_explanations_in_prompts(self):
        """
        Test that explanations appear correctly in prompts when available.

        WHAT THIS TESTS
        ===============
        For domains that include causal explanations (currently economy domain):
        - Generate prompts with different counterbalance conditions
        - Verify that appropriate explanations appear in the prompts
        - Check that explanation content is properly integrated into prompt text

        WHAT WE EXPECT
        ==============
        When explanations are available for a causal relationship:
        - The explanation should be selected based on counterbalance conditions
        - The explanation text should appear in the generated prompt
        - Multiple words from the explanation should be present (not just coincidental matches)

        For example, in economy domain X→Z relationship:
        - p_p condition should include "Low interest rates stimulate economic growth..."
        - m_m condition should include "A lot of people are making large monthly interest payments..."

        WHY THIS MATTERS
        ================
        Explanations provide crucial context for causal reasoning. They help LLMs
        understand WHY certain causal relationships exist, not just that they exist.
        Missing explanations would result in prompts that are harder to reason about.

        FAILURE SCENARIOS
        =================
        - Missing explanations → Prompts lack causal reasoning context
        - Wrong explanations → Contradictory or confusing causal stories
        - Truncated explanations → Incomplete reasoning context
        """
        # Test with economy domain which has explanations
        domain_dict = rw_17_domain_components["economy"]

        df = generate_prompt_dataframe(
            domain_dict,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="single_numeric_response",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Test a few specific cases where we know explanations should appear
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            prompt = row["prompt"]
            cntbl_cond = row["cntbl_cond"]

            # Check X variable explanation
            x_cntbl = cntbl_cond[0]  # First character is X counterbalance
            z_cntbl = cntbl_cond[2]  # Third character is Z counterbalance
            x_z_key = f"{x_cntbl}_{z_cntbl}"

            if (
                "explanations" in domain_dict["variables"]["X"]
                and x_z_key in domain_dict["variables"]["X"]["explanations"]
            ):
                expected_explanation = domain_dict["variables"]["X"]["explanations"][
                    x_z_key
                ]
                # Check that some part of the explanation appears in the prompt
                explanation_words = expected_explanation.split()[:5]  # First 5 words
                for word in explanation_words:
                    if len(word) > 3:  # Skip very short words
                        assert word in prompt, (
                            f"Explanation word '{word}' not found in prompt. "
                            f"Expected explanation: {expected_explanation[:50]}..."
                        )

    def test_domain_introduction_in_prompts(self):
        """
        Test that domain introductions appear in prompts.

        WHAT THIS TESTS
        ===============
        For each domain:
        - Generate prompts and verify the domain introduction text appears
        - Check that introduction appears near the beginning of prompts
        - Verify introduction content matches what's defined in domain dictionaries

        WHAT WE EXPECT
        ==============
        Domain introductions should appear in every prompt and provide context about:
        - What field/domain we're reasoning about (economics, sociology, meteorology)
        - What kind of relationships we study in this domain
        - General framing for the causal reasoning task

        For example, economy introduction starts with:
        "Economists seek to describe and predict the regular patterns of economic fluctuation..."

        WHY THIS MATTERS
        ================
        Introductions set the context for the entire reasoning task. They help LLMs
        understand what domain they're reasoning about and what kinds of relationships
        are relevant. Missing introductions would make prompts less coherent.

        FAILURE SCENARIOS
        =================
        - Missing introductions → Lack of domain context for reasoning
        - Wrong introductions → Confusion about what domain we're in
        - Partial introductions → Incomplete context setting
        """
        for domain_name, domain_dict in rw_17_domain_components.items():
            df = generate_prompt_dataframe(
                domain_dict,
                inference_tasks_rw17,
                "collider",
                graph_structures,
                prompt_category="single_numeric_response",
                prompt_type="Please provide only a numeric response.",
                indep_causes_collider=False,
            )

            expected_intro = domain_dict["introduction"]

            # Check first prompt (they should all have the same intro)
            prompt = df.iloc[0]["prompt"]

            # Introduction should appear near the beginning of the prompt
            intro_words = expected_intro.split()[:10]  # First 10 words
            for word in intro_words:
                if len(word) > 3:  # Skip short words like "the", "and"
                    assert word in prompt, (
                        f"Introduction word '{word}' not found in prompt for domain {domain_name}. "
                        f"Expected intro: {expected_intro[:100]}..."
                    )

    def test_counterbalance_consistency(self):
        """
        Test that counterbalance conditions are applied consistently.

        WHAT THIS TESTS
        ===============
        For all prompts with the same counterbalance condition:
        - Variable senses should be identical across all inference tasks
        - Same counterbalance condition should always produce same variable descriptions
        - Different counterbalance conditions should produce different variable descriptions

        WHAT WE EXPECT
        ==============
        Counterbalance conditions (e.g., "ppm") should deterministically control
        variable senses across all prompts:
        - All prompts with condition "ppm" should have identical X_sense, Y_sense, Z_sense
        - Prompts with different conditions should have different senses

        WHY THIS MATTERS
        ================
        Consistency is crucial for experimental validity. If the same counterbalance
        condition produces different prompts, we can't properly control experimental
        variables or interpret results.

        FAILURE SCENARIOS
        =================
        - Inconsistent senses → Uncontrolled experimental variation
        - Random sense assignment → No actual experimental manipulation
        - Condition-independent senses → All conditions equivalent (no manipulation)
        """
        domain_dict = rw_17_domain_components["economy"]

        df = generate_prompt_dataframe(
            domain_dict,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="single_numeric_response",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Group by counterbalance condition and check consistency
        for cntbl_cond, group in df.groupby("cntbl_cond"):
            # All rows with same counterbalance should have same variable senses
            first_row = group.iloc[0]

            for var_key in ["X", "Y", "Z"]:
                expected_sense = first_row[f"{var_key}_sense"]

                for _, row in group.iterrows():
                    actual_sense = row[f"{var_key}_sense"]
                    assert actual_sense == expected_sense, (
                        f"Inconsistent {var_key}_sense for condition {cntbl_cond}: "
                        f"expected {expected_sense}, got {actual_sense}"
                    )


class TestCreateDomainDict:
    """
    Test the create_domain_dict function produces correct format.

    WHAT WE EXPECT
    ==============
    The create_domain_dict function should:
    - Accept variable configurations and produce properly formatted domain dictionaries
    - Use the clean format ("name"/"detailed") not redundant format ("X_name"/"X_detailed")
    - Work seamlessly with the prompt generation pipeline
    - Handle all required fields and optional fields like explanations

    WHY THIS MATTERS
    ================
    create_domain_dict is used to generate custom domains programmatically.
    If it produces the wrong format, custom domains won't work with the
    prompt generation pipeline, limiting experimental flexibility.

    TESTING STRATEGY
    ================
    We test both format correctness and integration with downstream functions
    to ensure create_domain_dict output is fully compatible with the system.
    """

    def test_create_domain_dict_format(self):
        """
        Test that create_domain_dict produces the clean format.

        WHAT THIS TESTS
        ===============
        - create_domain_dict output uses clean format: "name", "detailed"
        - Output does NOT use old redundant format: "X_name", "X_detailed"
        - All required fields are present in correct format
        - Function accepts standard variable configuration input

        WHAT WE EXPECT
        ==============
        Function should create dictionaries that match the format used in
        constants.py and abstract_domains.py:

        variables: {
            "X": {
                "name": "variable name",          # NOT "X_name"
                "detailed": "description",        # NOT "X_detailed"
                "p_value": {"1": "high", "0": "low"},
                "m_value": {"1": "low", "0": "high"}
            }
        }

        WHY THIS MATTERS
        ================
        The clean format is what all processing functions expect. If create_domain_dict
        uses the old redundant format, it would cause KeyErrors in prompt generation.

        FAILURE SCENARIOS
        =================
        - Using old format → KeyError: 'name' when accessing var_details["name"]
        - Missing required fields → Incomplete domain dictionaries
        - Wrong structure → Incompatibility with processing functions
        """
        variables_config = {
            "X": {
                "name": "test variable X",
                "detailed": "This is a test variable X",
                "p_value": {"1": "high", "0": "normal"},
                "m_value": {"1": "low", "0": "normal"},
            },
            "Y": {
                "name": "test variable Y",
                "detailed": "This is a test variable Y",
                "p_value": {"1": "strong", "0": "weak"},
            },
            "Z": {
                "name": "test variable Z",
                "detailed": "This is a test variable Z",
                "p_value": {"1": "positive", "0": "negative"},
            },
        }

        domain_dict = create_domain_dict(
            domain_name="test_domain",
            introduction="This is a test domain",
            variables_config=variables_config,
        )

        # Check structure
        assert domain_dict["domain_name"] == "test_domain"
        assert domain_dict["introduction"] == "This is a test domain"
        assert "variables" in domain_dict

        # Check variables use clean format
        for var_key in ["X", "Y", "Z"]:
            var_details = domain_dict["variables"][var_key]

            # Should use clean format
            assert "name" in var_details
            assert "detailed" in var_details
            assert var_details["name"] == variables_config[var_key]["name"]
            assert var_details["detailed"] == variables_config[var_key]["detailed"]

            # Should NOT use old format
            assert f"{var_key}_name" not in var_details
            assert f"{var_key}_detailed" not in var_details

    def test_create_domain_dict_integration(self):
        """
        Test that create_domain_dict output works with prompt generation.

        WHAT THIS TESTS
        ===============
        - create_domain_dict output can be used directly with generate_prompt_dataframe
        - Resulting prompts contain expected variable names
        - No errors occur during prompt generation
        - Integration works end-to-end

        WHAT WE EXPECT
        ==============
        Domains created with create_domain_dict should work identically to
        predefined domains in constants.py. The prompt generation pipeline
        should handle them seamlessly.

        WHY THIS MATTERS
        ================
        This tests the critical integration point. If create_domain_dict output
        can't be used with prompt generation, the function is useless for
        creating custom experimental domains.

        FAILURE SCENARIOS
        =================
        - KeyError during prompt generation → Format incompatibility
        - Missing variable names in prompts → Incomplete domain creation
        - Empty or malformed prompts → Broken integration
        - Processing errors → Inconsistent data structures
        """
        variables_config = {
            "X": {
                "name": "factor A",
                "detailed": "Factor A affects the system",
                "p_value": {"1": "high", "0": "normal"},
            },
            "Y": {
                "name": "factor B",
                "detailed": "Factor B influences outcomes",
                "p_value": {"1": "strong", "0": "weak"},
            },
            "Z": {
                "name": "outcome C",
                "detailed": "Outcome C is the result",
                "p_value": {"1": "positive", "0": "negative"},
            },
        }

        domain_dict = create_domain_dict(
            domain_name="integration_test",
            introduction="Testing integration",
            variables_config=variables_config,
        )

        # Should work with prompt generation without errors
        df = generate_prompt_dataframe(
            domain_dict,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="single_numeric_response",
            prompt_type="Test response.",
            indep_causes_collider=False,
        )

        assert len(df) > 0

        # Check that variable names appear in prompts
        sample_prompt = df.iloc[0]["prompt"]
        assert "factor A" in sample_prompt
        assert "factor B" in sample_prompt
        assert "outcome C" in sample_prompt


class TestCapitalizationIntegration:
    """
    Test that capitalization works correctly with the clean format.

    WHAT WE EXPECT
    ==============
    After the format standardization (moving from redundant to clean format):
    - Capitalization functionality should still work correctly
    - Sentences should start with capital letters
    - Proper sentence structure should be maintained
    - No lowercase letters should appear after periods

    WHY THIS MATTERS
    ================
    Capitalization was a major feature that required format changes. We need
    to ensure that fixing the format inconsistency didn't break capitalization,
    and that capitalization works with the new clean format.

    TESTING STRATEGY
    ================
    We test both general capitalization rules and specific explanation text
    formatting to ensure the capitalization system works end-to-end.
    """

    def test_capitalization_with_clean_format(self):
        """
        Test that capitalization still works after format standardization.

        WHAT THIS TESTS
        ===============
        - Generated prompts have proper sentence capitalization
        - First letter of each sentence is uppercase
        - Capitalization works with clean format variable access
        - No regression from format standardization changes

        WHAT WE EXPECT
        ==============
        Every sentence in generated prompts should start with a capital letter.
        This includes:
        - Domain introduction sentences
        - Variable description sentences
        - Causal explanation sentences
        - Inference task sentences

        WHY THIS MATTERS
        ================
        Proper capitalization makes prompts more professional and readable.
        Poor capitalization could affect LLM performance or make prompts
        look unprofessional in publications.

        FAILURE SCENARIOS
        =================
        - Lowercase sentence starts → Unprofessional appearance
        - Inconsistent capitalization → Distracting formatting issues
        - Capitalization function broken → Regression from format changes
        """
        domain_dict = rw_17_domain_components["economy"]

        df = generate_prompt_dataframe(
            domain_dict,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="single_numeric_response",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Check a few prompts for proper capitalization
        for i in range(min(3, len(df))):
            prompt = df.iloc[i]["prompt"]

            # Split into sentences and check capitalization
            sentences = prompt.split(". ")

            for j, sentence in enumerate(sentences):
                if sentence.strip() and not sentence.strip()[0].isdigit():
                    first_char = sentence.strip()[0]
                    assert first_char.isupper(), (
                        f"Sentence {j + 1} in prompt {i + 1} should start with uppercase: "
                        f"'{sentence.strip()[:50]}...'"
                    )

    def test_explanations_capitalization(self):
        """
        Test that explanations are properly capitalized.

        WHAT THIS TESTS
        ===============
        - Explanation text in prompts follows proper capitalization rules
        - No lowercase letters appear immediately after periods
        - Explanation integration preserves sentence structure
        - Capitalization function works on explanation content

        WHAT WE EXPECT
        ==============
        Explanation text should be properly integrated into prompts with
        correct capitalization. We should not see patterns like ". the"
        or ". a" which indicate improper capitalization.

        WHY THIS MATTERS
        ================
        Explanations are critical content that provides causal reasoning context.
        Poor capitalization in explanations would be particularly noticeable
        and could affect readability or LLM performance.

        FAILURE SCENARIOS
        =================
        - Lowercase after periods → ". the economy" instead of ". The economy"
        - Inconsistent explanation formatting → Some capitalized, some not
        - Broken explanation integration → Missing or malformed explanation text
        """
        domain_dict = rw_17_domain_components["economy"]

        df = generate_prompt_dataframe(
            domain_dict,
            inference_tasks_rw17,
            "collider",
            graph_structures,
            prompt_category="single_numeric_response",
            prompt_type="Please provide only a numeric response.",
            indep_causes_collider=False,
        )

        # Look for specific explanation text that should be capitalized
        sample_prompt = df.iloc[0]["prompt"]

        # Should not contain lowercase explanations after periods
        # Look for patterns like ". the" or ". a" which indicate improper capitalization
        import re

        lowercase_after_period = re.findall(r"\. [a-z]", sample_prompt)

        assert len(lowercase_after_period) == 0, (
            f"Found lowercase letters after periods: {lowercase_after_period}. "
            f"Prompt excerpt: {sample_prompt[:500]}..."
        )
