"""
token_utils.py: Utility functions for tokenization and token counting in prompts.
"""
from typing import List, Union

def count_tokens_whitespace(prompt: Union[str, List[str]]) -> Union[int, List[int]]:
    """
    Count tokens in a prompt or list of prompts using whitespace split.
    Returns the token count (int) or list of counts (List[int]).
    """
    if isinstance(prompt, str):
        return len(prompt.split())
    elif isinstance(prompt, list):
        return [len(p.split()) for p in prompt]
    else:
        raise TypeError("Input must be a string or list of strings.")


def count_tokens_per_prompt(prompts: List[str]) -> List[int]:
    """
    Count tokens for each prompt in a list using whitespace split.
    Returns a list of token counts.
    """
    return [len(p.split()) for p in prompts]


def average_token_count(prompts: List[str]) -> float:
    """
    Compute the average token count for a list of prompts.
    """
    counts = count_tokens_per_prompt(prompts)
    return sum(counts) / len(counts) if counts else 0.0
