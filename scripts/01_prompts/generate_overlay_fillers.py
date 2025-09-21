"""
Generate reproducible overlay filler blocks for each overlay type, variable, and field.
"""
from __future__ import annotations

import random
from typing import Dict

import yaml

# Fixed random seed for reproducibility
SEED = 42
random.seed(SEED)

# Example token lengths from your stats (adjust as needed)
TOKEN_LENGTHS = {
    "rw17_indep_causes": 268,  # base
    "rw17_overloaded_de": 368,  # overloaded
}

# Overlay types and their insertion locations
OVERLAY_TYPES = ["d", "e", "de"]
VARIABLES = ["X", "Y", "Z"]
FIELDS = ["detailed", "explanations"]

# How many fillers per location for each overlay type
FILLER_SPLIT = {
    "d": {"detailed": 1, "explanations": 0},
    "e": {"detailed": 0, "explanations": 1},
    "de": {"detailed": 0.5, "explanations": 0.5},
}

LOREM_IPSUM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua "
    "ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor "
    "in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident "
    "sunt in culpa qui officia deserunt mollit anim id est laborum"
).split()


def generate_lorem_vocab_block(num_tokens: int) -> str:
    words = [random.choice(LOREM_IPSUM_WORDS) for _ in range(num_tokens)]
    return " ".join(words)


def compute_filler_lengths(base_len: int, target_len: int, overlay_type: str) -> Dict[str, int]:
    extra = target_len - base_len
    split = FILLER_SPLIT.get(overlay_type, {})
    lengths = {field: int(extra * frac) for field, frac in split.items() if frac > 0}
    # Adjust for rounding
    total = sum(lengths.values())
    if total < extra:
        for field in split:
            if split[field] > 0:
                lengths[field] += extra - total
                break
    return lengths


def main():
    base_len = TOKEN_LENGTHS["rw17_indep_causes"]
    target_len = TOKEN_LENGTHS["rw17_overloaded_de"]
    output = {}
    for overlay_type in OVERLAY_TYPES:
        output[overlay_type] = {}
        filler_lengths = compute_filler_lengths(base_len, target_len, overlay_type)
        for var in VARIABLES:
            output[overlay_type][var] = {}
            for field in FIELDS:
                num_tokens = filler_lengths.get(field, 0)
                block = generate_lorem_vocab_block(num_tokens) if num_tokens > 0 else ""
                output[overlay_type][var][field] = block
    with open("overlay_fillers.yaml", "w") as f:
        yaml.safe_dump(output, f, sort_keys=False, allow_unicode=True)
    print("Wrote overlay fillers to overlay_fillers.yaml")


if __name__ == "__main__":
    main()
    main()
