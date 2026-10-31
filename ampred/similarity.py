"""Sequence similarity between test and training peptides (homology check).

We use k-mer Jaccard similarity: a fast, alignment-free proxy for
"how similar is this test peptide to its closest training peptide?".
"""

from __future__ import annotations

import numpy as np


def kmers(seq: str, k: int = 3) -> set[str]:
    """Set of all overlapping k-mers in ``seq``."""
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """|A & B| / |A | B| (0 if both empty)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def max_similarity_to_reference(
    queries: list[str], reference: list[str], k: int = 3
) -> np.ndarray:
    """For each query, the highest Jaccard similarity to any reference peptide."""
    ref_sets = [kmers(s, k) for s in reference]
    index: dict[str, list[int]] = {}
    for j, ks in enumerate(ref_sets):
        for km in ks:
            index.setdefault(km, []).append(j)
    out = np.zeros(len(queries))
    for i, q in enumerate(queries):
        qs = kmers(q, k)
        candidates = {j for km in qs for j in index.get(km, [])}
        out[i] = max((jaccard(qs, ref_sets[j]) for j in candidates), default=0.0)
    return out
