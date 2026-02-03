#!/usr/bin/env python3
"""
Canonical location for validate_task_probs (moved from scripts/validate_task_probs.py).
"""
from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root))  # allow importing modules at repo root

import torch  # noqa: E402

from causalign.analysis.model_fitting.tasks import (  # noqa: E402
    ROMAN_TO_LETTER,
    logistic_task_probability,
    noisy_or_task_probability,
)


def _load_task_conditional_map() -> dict:
    # Try regular import if scripts is a package (unlikely)
    try:
        from scripts.plot_results import TASK_CONDITIONAL_PROB  # type: ignore
        return TASK_CONDITIONAL_PROB  # noqa: F401
    except Exception:
        pass

    # Fallback: import plot_results.py by path
    plot_path = project_root / "scripts" / "plot_results.py"
    if plot_path.exists():
        spec = spec_from_file_location("plot_results", str(plot_path))
        if spec and spec.loader:
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return getattr(mod, "TASK_CONDITIONAL_PROB", {})
    return {}


TASK_CONDITIONAL_PROB = _load_task_conditional_map()


def build_parser():
    p = argparse.ArgumentParser(description="Validate closed-form task probabilities")
    p.add_argument("--model", choices=["logistic", "noisy_or"], default="logistic")
    p.add_argument("--pC1", type=float, default=0.5)
    p.add_argument("--pC2", type=float, default=0.5)
    # logistic parameters for collider graph: p(E=1|C1,C2) = σ(w0 + w1*C1 + w2*C2)
    p.add_argument("--w0", type=float, default=0.0)  # baseline log-odds when C1=0, C2=0
    p.add_argument("--w1", type=float, default=3.0)  # effect of C1 on E (causal strength)
    p.add_argument("--w2", type=float, default=3.0)  # effect of C2 on E (causal strength)
    # noisy-or parameters for collider graph: p(E=1|C1,C2) = 1 - (1-b) * (1-m1)^C1 * (1-m2)^C2
    p.add_argument("--b", type=float, default=0.1)   # baseline probability when C1=0, C2=0
    p.add_argument("--m1", type=float, default=0.5)  # causal strength of C1 on E
    p.add_argument("--m2", type=float, default=0.5)  # causal strength of C2 on E
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    roman_order = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]

    pC1 = torch.tensor(args.pC1, dtype=torch.float32)
    pC2 = torch.tensor(args.pC2, dtype=torch.float32)
    # print the initial paramter values and the model
    print(f"Initial parameters: pC1: {pC1.item()}, pC2: {pC2.item()}, w0: {args.w0}, w1: {args.w1}, w2: {args.w2}, b: {args.b}, m1: {args.m1}, m2: {args.m2}")
    print(f"Model: {args.model}")

    print("Task  Roman  Letter  Conditional                       Predicted")
    print("-" * 72)
    for idx, roman in enumerate(roman_order, 1):
        letter = ROMAN_TO_LETTER[roman]
        cond = TASK_CONDITIONAL_PROB.get(roman, "")
        if args.model == "logistic":
            w0 = torch.tensor(args.w0, dtype=torch.float32)
            w1 = torch.tensor(args.w1, dtype=torch.float32)
            w2 = torch.tensor(args.w2, dtype=torch.float32)
            val = logistic_task_probability(letter, w0, w1, w2, pC1, pC2).item()
        else:
            b = torch.tensor(args.b, dtype=torch.float32)
            m1 = torch.tensor(args.m1, dtype=torch.float32)
            m2 = torch.tensor(args.m2, dtype=torch.float32)
            val = noisy_or_task_probability(letter, b, m1, m2, pC1, pC2).item()
        print(f"{idx:>2}    {roman:<4}   {letter}      {cond:<35}  {val:0.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
    raise SystemExit(main())
