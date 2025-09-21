"""
Generate random vocab filler blocks to match target token lengths for overloaded prompts.
"""
from __future__ import annotations

import random
from typing import Dict

import yaml

# Example token lengths from your stats (adjust as needed)
TOKEN_LENGTHS = {
    "rw17_indep_causes": 268,  # base
    "rw17_overloaded_de": 368,  # overloaded
}

# Overlay types and their insertion locations
OVERLAY_LOCATIONS = {
    "basic": [],
    "d": ["detailed"],
    "e": ["explanations"],
    "de": ["detailed", "explanations"],
}

# How many fillers per location for each overlay type
FILLER_SPLIT = {
    "d": [1],
    "e": [1],
    "de": [0.5, 0.5],  # split evenly between detailed and explanations
}

LOREM_IPSUM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua "
    "ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor "
    "in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident "
    "sunt in culpa qui officia deserunt mollit anim id est laborum"
).split()


def generate_lorem_vocab_block(num_tokens: int) -> str:
    # Generate a block of words from lorem ipsum vocab
    words = [random.choice(LOREM_IPSUM_WORDS) for _ in range(num_tokens)]
    return " ".join(words)


def compute_filler_lengths(base_len: int, target_len: int, overlay_type: str) -> Dict[str, int]:
    # Compute how many tokens to insert at each location
    extra = target_len - base_len
    split = FILLER_SPLIT.get(overlay_type, [])
    locations = OVERLAY_LOCATIONS.get(overlay_type, [])
    if not locations or extra <= 0:
        return {}
    # Distribute extra tokens according to split
    filler_lengths = {loc: int(extra * frac) for loc, frac in zip(locations, split)}
    # Adjust for rounding
    total = sum(filler_lengths.values())
    if total < extra:
        # Add remainder to first location
        if locations:
            filler_lengths[locations[0]] += extra - total
    return filler_lengths


def main():
    output = {}
    for overlay_type in ["d", "e", "de"]:
        base_len = TOKEN_LENGTHS["rw17_indep_causes"]
        target_len = TOKEN_LENGTHS["rw17_overloaded_de"]
        filler_lengths = compute_filler_lengths(base_len, target_len, overlay_type)
        output[overlay_type] = {}
        for loc, num_tokens in filler_lengths.items():
            block = generate_lorem_vocab_block(num_tokens)
            output[overlay_type][loc] = block

    # Write to YAML file
    with open("filler_blocks.yaml", "w") as f:
        yaml.safe_dump(output, f, sort_keys=False, allow_unicode=True)
    print("Wrote filler blocks to filler_blocks.yaml")


if __name__ == "__main__":
    main()
    main()
