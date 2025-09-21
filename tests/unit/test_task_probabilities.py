from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch

# Ensure package imports work regardless of editable install
_proj_root = Path(__file__).resolve().parents[2]
sys.path.append(str(_proj_root))            # allow 'scripts' and other roots
sys.path.append(str(_proj_root / "src"))   # allow 'causalign' from src layout

from causalign.analysis.model_fitting.tasks import (  # type: ignore
    logistic_task_probability,
    noisy_or_task_probability,
    ROMAN_TO_LETTER,
)


LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return num / torch.clamp(den, min=eps)

def _sigma(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def enumerated_logistic(letter: str, w0, w1, w2, pC1, pC2) -> float:
    # Build basic terms
    def pE(c1, c2):
        return _sigma(w0 + w1 * float(c1) + w2 * float(c2))

    # Priors and joints
    p_c1 = pC1
    p_c2 = pC2
    p_c1c2 = {
        (0, 0): (1 - p_c1) * (1 - p_c2),
        (0, 1): (1 - p_c1) * p_c2,
        (1, 0): p_c1 * (1 - p_c2),
        (1, 1): p_c1 * p_c2,
    }

    # Evidence
    Z = sum(pE(c1, c2) * p for (c1, c2), p in p_c1c2.items())
    Z = torch.clamp(Z, min=1e-12, max=1 - 1e-12)

    # Compute per letter
    l = letter.lower()
    if l == "a":
        num = pE(1, 1) * p_c1
        den = num + pE(0, 1) * (1 - p_c1)
        return float(_safe_div(num, den).item())
    if l == "b":
        num = p_c1 * (p_c2 * pE(1, 1) + (1 - p_c2) * pE(1, 0))
        return float(_safe_div(num, Z).item())
    if l == "c":
        num = pE(1, 0) * p_c1
        den = num + pE(0, 0) * (1 - p_c1)
        return float(_safe_div(num, den).item())
    if l in ("d", "e"):
        return float(p_c1.item())
    if l == "f":
        num = (1 - pE(1, 1)) * p_c1
        den = num + (1 - pE(0, 1)) * (1 - p_c1)
        return float(_safe_div(num, den).item())
    if l == "g":
        num = p_c1 * (p_c2 * (1 - pE(1, 1)) + (1 - p_c2) * (1 - pE(1, 0)))
        return float(_safe_div(num, 1 - Z).item())
    if l == "h":
        num = (1 - pE(1, 0)) * p_c1
        den = num + (1 - pE(0, 0)) * (1 - p_c1)
        return float(_safe_div(num, den).item())
    if l == "i":
        return float(pE(0, 0).item())
    if l == "j":
        return float(pE(0, 1).item())
    if l == "k":
        return float(pE(1, 1).item())
    raise ValueError(l)


def enumerated_noisy_or(letter: str, b, m1, m2, pC1, pC2) -> float:
    def pE(c1, c2):
        return 1 - (1 - b) * torch.pow(1 - m1, float(c1)) * torch.pow(1 - m2, float(c2))

    p_c1 = pC1
    p_c2 = pC2
    p_c1c2 = {
        (0, 0): (1 - p_c1) * (1 - p_c2),
        (0, 1): (1 - p_c1) * p_c2,
        (1, 0): p_c1 * (1 - p_c2),
        (1, 1): p_c1 * p_c2,
    }

    Z = sum(pE(c1, c2) * p for (c1, c2), p in p_c1c2.items())
    Z = torch.clamp(Z, min=1e-12, max=1 - 1e-12)

    l = letter.lower()
    if l == "a":
        num = pE(1, 1) * p_c1
        den = num + pE(0, 1) * (1 - p_c1)
        return float(_safe_div(num, den).item())
    if l == "b":
        num = p_c1 * (p_c2 * pE(1, 1) + (1 - p_c2) * pE(1, 0))
        return float(_safe_div(num, Z).item())
    if l == "c":
        num = pE(1, 0) * p_c1
        den = num + pE(0, 0) * (1 - p_c1)
        return float(_safe_div(num, den).item())
    if l in ("d", "e"):
        return float(p_c1.item())
    if l == "f":
        num = (1 - pE(1, 1)) * p_c1
        den = num + (1 - pE(0, 1)) * (1 - p_c1)
        return float(_safe_div(num, den).item())
    if l == "g":
        num = p_c1 * (p_c2 * (1 - pE(1, 1)) + (1 - p_c2) * (1 - pE(1, 0)))
        return float(_safe_div(num, 1 - Z).item())
    if l == "h":
        num = (1 - pE(1, 0)) * p_c1
        den = num + (1 - pE(0, 0)) * (1 - p_c1)
        return float(_safe_div(num, den).item())
    if l == "i":
        return float(pE(0, 0).item())
    if l == "j":
        return float(pE(0, 1).item())
    if l == "k":
        return float(pE(1, 1).item())
    raise ValueError(l)


def _rand_prob(lo=0.05, hi=0.95) -> torch.Tensor:
    return torch.tensor(random.uniform(lo, hi), dtype=torch.float32)


def _rand_weight(bound=2.5) -> torch.Tensor:
    return torch.tensor(random.uniform(-bound, bound), dtype=torch.float32)


def test_logistic_task_probabilities_enumeration_agree():
    random.seed(0)
    torch.manual_seed(0)

    for _ in range(10):
        pC1 = _rand_prob()
        pC2 = _rand_prob()
        w0 = _rand_weight()
        w1 = _rand_weight()
        w2 = _rand_weight()

        for letter in LETTERS:
            # closed form
            cf = logistic_task_probability(letter, w0, w1, w2, pC1, pC2).item()
            # enumerated
            en = enumerated_logistic(letter, w0, w1, w2, pC1, pC2)
            assert abs(cf - en) < 1e-6


def test_noisy_or_task_probabilities_enumeration_agree():
    random.seed(0)
    torch.manual_seed(0)

    for _ in range(10):
        pC1 = _rand_prob()
        pC2 = _rand_prob()
        b = _rand_prob()
        m1 = _rand_prob()
        m2 = _rand_prob()

        for letter in LETTERS:
            cf = noisy_or_task_probability(letter, b, m1, m2, pC1, pC2).item()
            en = enumerated_noisy_or(letter, b, m1, m2, pC1, pC2)
            assert abs(cf - en) < 1e-6


