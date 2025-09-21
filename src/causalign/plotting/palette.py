"""
Global color constants and helpers for plotting across causalalign.

Usage:
    from causalign.plotting.palette import PROMPT_CATEGORY_COLORS, canon_prompt_category
    color = PROMPT_CATEGORY_COLORS[canon_prompt_category(label)]

Update the colors here to change them everywhere.
"""
from __future__ import annotations

from typing import Dict

# Central, canonical labels for prompt categories used in plots
NUMERIC_LABEL = "numeric"
COT_LABEL = "CoT"

# Default color scheme (RGB 0..1 tuples) — colorblind-friendly and print-safe
    # NUMERIC_COLOR = (0.85, 0.60, 0.55)  # muted red
    # COT_COLOR = (0.00, 0.20, 0.55)      # deep blue


# CAblue: RGB (10, 80, 110)
CAblue = (10/255, 80/255, 110/255)       # (0.039, 0.314, 0.431)

# CAlightblue: RGB (58, 160, 171)
CAlightblue = (58/255, 160/255, 171/255) # (0.227, 0.627, 0.671)

NUMERIC_COLOR = CAlightblue  
COT_COLOR = CAblue      

PROMPT_CATEGORY_COLORS: Dict[str, tuple[float, float, float]] = {
    NUMERIC_LABEL: NUMERIC_COLOR,
    COT_LABEL: COT_COLOR,
}

# Synonyms mapping to canonical labels
_NUMERIC_SYNS = {
    "numeric", "pcnum", "num", "single_numeric", "single_numeric_response",
}
_COT_SYNS = {
    "cot", "pccot", "chain_of_thought", "chain-of-thought", "cot_stepwise", "CoT",
}

def canon_prompt_category(label: str) -> str:
    """Return the canonical prompt-category label used for palette lookup.

    Normalizes different pipeline spellings to {"numeric", "CoT"}.
    """
    t = str(label).strip()
    tl = t.lower()
    if tl in _NUMERIC_SYNS:
        return NUMERIC_LABEL
    if tl in _COT_SYNS:
        return COT_LABEL
    # pass through if already canonical (e.g., "CoT") or unknown
    return t

__all__ = [
    "PROMPT_CATEGORY_COLORS",
    "NUMERIC_COLOR",
    "COT_COLOR",
    "NUMERIC_LABEL",
    "COT_LABEL",
    "canon_prompt_category",
]
