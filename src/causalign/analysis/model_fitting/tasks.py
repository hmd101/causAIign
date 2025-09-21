from __future__ import annotations

"""
Task probability evaluators for collider graph.

Implements A - K (letters) and maps them to Roman numerals I - XI used in processed
datasets. Supports two link functions: Logistic and Noisy-OR.

This module exposes small pure functions that compute the predicted probabilities
per task, which are differentiable when used with torch tensors.
"""

from typing import Dict, Tuple

import torch


# Roman <-> letter mapping used in the pipeline (RW17 collider)
ROMAN_TO_LETTER: Dict[str, str] = {
    "I": "i",
    "II": "j",
    "III": "k",
    "IV": "d",
    "V": "e",
    "VI": "a",
    "VII": "b",
    "VIII": "c",
    "IX": "f",
    "X": "g",
    "XI": "h",
}

LETTER_TO_ROMAN: Dict[str, str] = {v: k for k, v in ROMAN_TO_LETTER.items()}


def logistic_sigma(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return numerator / torch.clamp(denominator, min=eps)


def logistic_e_given_c(w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, c1: int, c2: int) -> torch.Tensor:
    """p(E=1|C1, C2) under logistic link: \sigma(w0 + w1*C1 + w2*C2)."""
    return logistic_sigma(w0 + w1 * float(c1) + w2 * float(c2))


def logistic_Z(w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, pC1: torch.Tensor, pC2: torch.Tensor) -> torch.Tensor:
    """Evidence p(E=1) for logistic link (see my LaTeX derivations for thesis)."""
    term11 = logistic_e_given_c(w0, w1, w2, 1, 1) * pC1 * pC2
    term10 = logistic_e_given_c(w0, w1, w2, 1, 0) * pC1 * (1 - pC2)
    term01 = logistic_e_given_c(w0, w1, w2, 0, 1) * (1 - pC1) * pC2
    term00 = logistic_e_given_c(w0, w1, w2, 0, 0) * (1 - pC1) * (1 - pC2)
    return term11 + term10 + term01 + term00


def noisy_or_e_given_c(b: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor, c1: int, c2: int) -> torch.Tensor:
    """p(E=1|C1,C2) under noisy-OR: 1 - (1-b) * (1-m1)^C1 * (1-m2)^C2

    Matches the LaTeX derivation in task_probabilities.tex where Task I equals b.
    """
    return 1 - (1 - b) * torch.pow(1 - m1, float(c1)) * torch.pow(1 - m2, float(c2))


def noisy_or_Z(b: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor, pC1: torch.Tensor, pC2: torch.Tensor) -> torch.Tensor:
    """Evidence p(E=1) for noisy-OR (per derivation)."""
    term11 = noisy_or_e_given_c(b, m1, m2, 1, 1) * pC1 * pC2
    term10 = noisy_or_e_given_c(b, m1, m2, 1, 0) * pC1 * (1 - pC2)
    term01 = noisy_or_e_given_c(b, m1, m2, 0, 1) * (1 - pC1) * pC2
    term00 = noisy_or_e_given_c(b, m1, m2, 0, 0) * (1 - pC1) * (1 - pC2)
    return term11 + term10 + term01 + term00


def logistic_task_probability(letter: str, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, pC1: torch.Tensor, pC2: torch.Tensor) -> torch.Tensor:
    """Compute predicted probability for task letter a - k under logistic link.

    Implements formulas from the provided LaTeX derivations.
    """
    letter = letter.lower()
    if letter == "a":  # p(C1=1 | E=1, C2=1)
        num = logistic_e_given_c(w0, w1, w2, 1, 1) * pC1
        den = num + logistic_e_given_c(w0, w1, w2, 0, 1) * (1 - pC1)
        return _safe_div(num, den)
    if letter == "b":  # p(C1=1 | E=1)
        Z = logistic_Z(w0, w1, w2, pC1, pC2)
        Z = torch.clamp(Z, min=1e-8)
        num = pC1 * (pC2 * logistic_e_given_c(w0, w1, w2, 1, 1) + (1 - pC2) * logistic_e_given_c(w0, w1, w2, 1, 0))
        return num / Z
    if letter == "c":  # p(C1=1 | E=1, C2=0)
        num = logistic_e_given_c(w0, w1, w2, 1, 0) * pC1
        den = num + logistic_e_given_c(w0, w1, w2, 0, 0) * (1 - pC1)
        return _safe_div(num, den)
    if letter == "d":  # p(C1=1 | C2=1)
        return pC1
    if letter == "e":  # p(C1=1 | C2=0)
        return pC1
    if letter == "f":  # p(C1=1 | E=0, C2=1)
        num = (1 - logistic_e_given_c(w0, w1, w2, 1, 1)) * pC1
        den = num + (1 - logistic_e_given_c(w0, w1, w2, 0, 1)) * (1 - pC1)
        return _safe_div(num, den)
    if letter == "g":  # p(C1=1 | E=0)
        Z = logistic_Z(w0, w1, w2, pC1, pC2)
        denom = torch.clamp(1 - Z, min=1e-8)
        num = pC1 * (pC2 * (1 - logistic_e_given_c(w0, w1, w2, 1, 1)) + (1 - pC2) * (1 - logistic_e_given_c(w0, w1, w2, 1, 0)))
        return num / denom
    if letter == "h":  # p(C1=1 | E=0, C2=0)
        num = (1 - logistic_e_given_c(w0, w1, w2, 1, 0)) * pC1
        den = num + (1 - logistic_e_given_c(w0, w1, w2, 0, 0)) * (1 - pC1)
        return _safe_div(num, den)
    if letter == "i":  # p(E=1 | C1=0, C2=0)
        return logistic_e_given_c(w0, w1, w2, 0, 0)
    if letter == "j":  # p(E=1 | C1=0, C2=1) by default
        return logistic_e_given_c(w0, w1, w2, 0, 1)
    if letter == "k":  # p(E=1 | C1=1, C2=1)
        return logistic_e_given_c(w0, w1, w2, 1, 1)
    raise ValueError(f"Unknown task letter: {letter}")


def noisy_or_task_probability(letter: str, b: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor, pC1: torch.Tensor, pC2: torch.Tensor) -> torch.Tensor:
    """Compute predicted probability for task letter a - k under noisy-OR link.

    Implements formulas from the provided LaTeX derivations.
    """
    letter = letter.lower()
    if letter == "a":  # p(C1=1 | E=1, C2=1)
        num = noisy_or_e_given_c(b, m1, m2, 1, 1) * pC1
        den = num + noisy_or_e_given_c(b, m1, m2, 0, 1) * (1 - pC1)
        return _safe_div(num, den)
    if letter == "b":  # p(C1=1 | E=1)
        Z = noisy_or_Z(b, m1, m2, pC1, pC2)
        Z = torch.clamp(Z, min=1e-8)
        num = pC1 * (pC2 * noisy_or_e_given_c(b, m1, m2, 1, 1) + (1 - pC2) * noisy_or_e_given_c(b, m1, m2, 1, 0))
        return num / Z
    if letter == "c":  # p(C1=1 | E=1, C2=0)
        num = noisy_or_e_given_c(b, m1, m2, 1, 0) * pC1
        den = num + noisy_or_e_given_c(b, m1, m2, 0, 0) * (1 - pC1)
        return _safe_div(num, den)
    if letter == "d":
        return pC1
    if letter == "e":
        return pC1
    if letter == "f":  # p(C1=1 | E=0, C2=1)
        num = (1 - noisy_or_e_given_c(b, m1, m2, 1, 1)) * pC1
        den = num + (1 - noisy_or_e_given_c(b, m1, m2, 0, 1)) * (1 - pC1)
        return _safe_div(num, den)
    if letter == "g":  # p(C1=1 | E=0)
        Z = noisy_or_Z(b, m1, m2, pC1, pC2)
        denom = torch.clamp(1 - Z, min=1e-8)
        num = pC1 * (pC2 * (1 - noisy_or_e_given_c(b, m1, m2, 1, 1)) + (1 - pC2) * (1 - noisy_or_e_given_c(b, m1, m2, 1, 0)))
        return num / denom
    if letter == "h":  # p(C1=1 | E=0, C2=0)
        num = (1 - noisy_or_e_given_c(b, m1, m2, 1, 0)) * pC1
        den = num + (1 - noisy_or_e_given_c(b, m1, m2, 0, 0)) * (1 - pC1)
        return _safe_div(num, den)
    if letter == "i":  # p(E=1|0,0)
        return noisy_or_e_given_c(b, m1, m2, 0, 0)
    if letter == "j":  # p(E=1|0,1)
        return noisy_or_e_given_c(b, m1, m2, 0, 1)
    if letter == "k":  # p(E=1|1,1)
        return noisy_or_e_given_c(b, m1, m2, 1, 1)
    raise ValueError(f"Unknown task letter: {letter}")


def roman_task_to_probability(
    roman_task: str,
    link: str,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute p for a Roman-numeral task under the specified link.

    - link: 'logistic' or 'noisy_or'
    - params:
      * logistic: {w0,w1,w2,pC1,pC2}
      * noisy_or: {b,m1,m2,pC1,pC2}
    """
    letter = ROMAN_TO_LETTER[roman_task]
    if link == "logistic":
        return logistic_task_probability(letter, params["w0"], params["w1"], params["w2"], params["pC1"], params["pC2"])
    if link == "noisy_or":
        return noisy_or_task_probability(letter, params["b"], params["m1"], params["m2"], params["pC1"], params["pC2"])
    raise ValueError(f"Unsupported link: {link}")


