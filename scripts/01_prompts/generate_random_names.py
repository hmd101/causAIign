"""
Generate random variable names for abstract domains.

This is a thin wrapper around `causalign.prompts.core.utils.random_variable_names`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Sequence


def _import_utils():
    """Import utils after ensuring the repo's `src` is on sys.path."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from causalign.prompts.core.utils import (  # type: ignore
        CHAR_POOLS,
        random_variable_names,
    )

    return CHAR_POOLS, random_variable_names


def build_parser(known_pools):
    parser = argparse.ArgumentParser(
        description=(
            "Generate random variable names using predefined or custom character pools.\n\n"
            "Pass pool name (letters, digits, alnum, symbols, lower, upper, letters_symbols, digits_symbols, all) or a literal set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", "-c", type=int, default=3, help="How many names to generate")
    parser.add_argument("--length", "-l", type=int, default=5, help="Length of each name")
    parser.add_argument("--chars", default="letters_symbols", help="Pool name or literal chars")
    parser.add_argument("--unique", action="store_true", help="Ensure all names are unique")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--snippet", action="store_true", help="Print paste-ready abstract domain snippet (requires count=3)")
    return parser


def print_names(names):
    print("Generated names:")
    for n in names:
        print(f"- {n}")


def print_snippet(names):
    if len(names) != 3:
        print("[snippet skipped] Snippet output requires exactly 3 names (X, Y, Z).")
        return
    x, y, z = names
    obj = {
        "domain_name": "systems",
        "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic variables.",
        "variables": {
            "X": {"name": x, "detailed": "", "p_value": {"1": "high", "0": "low"}, "m_value": {"1": "low", "0": "high"}},
            "Y": {"name": y, "detailed": "", "p_value": {"1": "strong", "0": "weak"}, "m_value": {"1": "weak", "0": "strong"}},
            "Z": {"name": z, "detailed": "", "p_value": {"1": "powerful", "0": "weak"}, "m_value": {"1": "weak", "0": "powerful"}},
        },
    }
    print("\nPaste-ready snippet (abstract domain skeleton):\n")
    print(json.dumps(obj, indent=4))


def main(argv: Sequence[str] | None = None) -> int:
    CHAR_POOLS, random_variable_names = _import_utils()
    parser = build_parser(list(CHAR_POOLS.keys()))
    args = parser.parse_args(argv)

    names = random_variable_names(
        count=args.count,
        length=args.length,
        chars=args.chars,
        unique=args.unique,
        seed=args.seed,
    )

    print_names(names)
    if args.snippet:
        print_snippet(names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

if __name__ == "__main__":
    raise SystemExit(main())
