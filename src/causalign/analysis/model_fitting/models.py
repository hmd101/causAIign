from __future__ import annotations

"""
PyTorch modules for collider models with different link functions and parameter tying.

This defines a small parameter container with unconstrained parameters and
deterministic transforms to appropriate domains (probabilities in [0,1],
weight ranges for logistic, etc.).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


def device_from_string(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


@dataclass
class ParameterTying:
    """Configuration for tying parameters across symmetric causes.

    - three_param: pC1==pC2, w1==w2 (or m1==m2), separate w0/b
    - four_param: pC1==pC2, w1 and w2 free, w0/b free
    - five_param: all free
    """

    num_params: int = 3  # 3, 4, or 5

    @property
    def tie_priors(self) -> bool:
        # 3- and 4-parameter variants tie priors; 5-parameter leaves them free
        return self.num_params in (3, 4)

    @property
    def tie_strengths(self) -> bool:
        return self.num_params == 3


class ColliderLogisticParameters(nn.Module):
    """Unconstrained parameters for the logistic collider model.

    Exposes transformed properties:
    - pC1, pC2 in [0,1]
    - w0, w1, w2 in [-3, 3] via tanh transform
    Tying is applied in the forward properties.
    """

    def __init__(self, tying: ParameterTying):
        super().__init__()
        self.tying = tying

        # Unconstrained parameters
        self.theta_pC = nn.Parameter(torch.zeros(()))  # shared prior base
        self.theta_pC2 = nn.Parameter(torch.zeros(()))  # only used if untied
        # Logistic regression weights: w0 (bias), w1 (effect of C1), w2 (effect of C2)
        self.theta_w0 = nn.Parameter(torch.zeros(()))
        self.theta_w1 = nn.Parameter(torch.zeros(()))
        self.theta_w2 = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _to_prob(theta: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(theta)

    @staticmethod
    def _to_bounded_weight(theta: torch.Tensor, bound: float = 3.0) -> torch.Tensor:
        return bound * torch.tanh(theta)

    def get_params(self) -> Dict[str, torch.Tensor]:
        pC1 = self._to_prob(self.theta_pC)
        if self.tying.tie_priors:
            pC2 = pC1
        else:
            pC2 = self._to_prob(self.theta_pC2)

        w0 = self._to_bounded_weight(self.theta_w0)
        if self.tying.tie_strengths:
            w1 = self._to_bounded_weight(self.theta_w1)
            w2 = w1
        else:
            w1 = self._to_bounded_weight(self.theta_w1)
            w2 = self._to_bounded_weight(self.theta_w2)

        return {"pC1": pC1, "pC2": pC2, "w0": w0, "w1": w1, "w2": w2}


class ColliderNoisyOrParameters(nn.Module):
    """Unconstrained parameters for the noisy-OR collider model.

    Exposes transformed properties:
    - pC1, pC2, b, m1, m2 in [0,1]
    Tying is applied for m1==m2 when 3-parameter config is used.
    
    We use sigmoid transforms to map unconstrained parameters to [0,1] because:
    - All noisy-OR parameters are probabilities (pC1, pC2, b) or probability-like
      causal strengths (m1, m2) that must be bounded in [0,1]
    - Sigmoid provides smooth, differentiable mapping from R to (0,1) with good
      gradient properties for optimization
    - Unlike clipping, sigmoid preserves gradients everywhere and avoids boundary issues
    """

    def __init__(self, tying: ParameterTying):
        super().__init__()
        self.tying = tying
        # priors for C1 and C2
        self.theta_pC = nn.Parameter(torch.zeros(()))
        self.theta_pC2 = nn.Parameter(torch.zeros(()))

        # Noisy-OR parameters for collider graph:
        # b: base rate (P(Y=1) when both C1=0 and C2=0)
        # m1: strength of C1 -> Y causal link
        # m2: strength of C2 -> Y causal link
        self.theta_b = nn.Parameter(torch.zeros(()))
        self.theta_m1 = nn.Parameter(torch.zeros(()))
        self.theta_m2 = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _to_prob(theta: torch.Tensor) -> torch.Tensor:
        """Transform unconstrained parameter to probability via sigmoid.
        
        Sigmoid maps R -> (0,1), ensuring all noisy-OR parameters stay in valid
        probability range while maintaining differentiability for gradient-based
        optimization.
        """
        return torch.sigmoid(theta)

    def get_params(self) -> Dict[str, torch.Tensor]:
        pC1 = self._to_prob(self.theta_pC)
        if self.tying.tie_priors:
            pC2 = pC1
        else:
            pC2 = self._to_prob(self.theta_pC2)

        b = self._to_prob(self.theta_b)
        m1 = self._to_prob(self.theta_m1)
        if self.tying.tie_strengths:
            m2 = m1
        else:
            m2 = self._to_prob(self.theta_m2)

        return {"pC1": pC1, "pC2": pC2, "b": b, "m1": m1, "m2": m2}


def create_parameter_module(link: str, num_params: int) -> nn.Module:
    """Factory to create parameter modules given link and tying size."""
    tying = ParameterTying(num_params=num_params)
    if link == "logistic":
        return ColliderLogisticParameters(tying)
    if link == "noisy_or":
        return ColliderNoisyOrParameters(tying)
    raise ValueError(f"Unsupported link: {link}")


