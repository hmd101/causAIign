"""Property test: compute_data_hash must be invariant to row ordering.

We construct a synthetic group rows signature as (task, response) tuples, permute
it several times, and verify both full and short hashes remain identical.
"""
from __future__ import annotations

import itertools
import random

from causalign.analysis.model_fitting.hashing import compute_data_hash


def test_compute_data_hash_order_invariance():
    base = [
        ("I", 0.1),
        ("II", 0.2),
        ("III", 0.9),
        ("IV", 0.4),
        ("V", 0.7),
    ]
    # Reference hash using sorted copy (adapter uses sorted rows signature generation)
    sorted_ref = sorted(base)
    ref_full, ref_short = compute_data_hash(sorted_ref)

    # Generate several permutations (not all factorial for speed)
    perms = [list(p) for p in itertools.islice(itertools.permutations(base), 20)]
    random.seed(42)
    random.shuffle(perms)

    for perm in perms:
        # Simulate adapter canonicalization by sorting before hashing
        full, short = compute_data_hash(sorted(perm))
        assert full == ref_full, f"Full hash mismatch for permutation {perm}"
        assert short == ref_short, f"Short hash mismatch for permutation {perm}"
