"""Hand-crafted peptide features: composition and simple physico-chemistry."""

from __future__ import annotations

from itertools import product

import numpy as np

from .data import AMINO_ACIDS

# Kyte-Doolittle hydropathy scale (Kyte & Doolittle, 1982).
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
DIPEPTIDES = ["".join(p) for p in product(AMINO_ACIDS, repeat=2)]
_DIPEP_INDEX = {d: i for i, d in enumerate(DIPEPTIDES)}
PHYSCHEM_NAMES = [
    "length", "net_charge", "frac_cationic", "frac_anionic", "frac_hydrophobic",
    "frac_aromatic", "mean_hydropathy", "hydrophobic_moment",
]


def aa_composition(seq: str) -> np.ndarray:
    """Fraction of each of the 20 amino acids (sums to 1)."""
    counts = np.array([seq.count(a) for a in AMINO_ACIDS], dtype=float)
    return counts / max(len(seq), 1)


def dipeptide_composition(seq: str) -> np.ndarray:
    """Fraction of each of the 400 amino-acid pairs (sums to 1 if len >= 2)."""
    vec = np.zeros(len(DIPEPTIDES))
    for i in range(len(seq) - 1):
        vec[_DIPEP_INDEX[seq[i : i + 2]]] += 1
    total = vec.sum()
    return vec / total if total else vec


def net_charge(seq: str) -> float:
    """Approximate charge at pH 7: +1 for K/R, -1 for D/E (His ignored)."""
    return float(seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"))


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """Eisenberg hydrophobic moment for an ideal alpha-helix (100 deg per residue).

    A large value means hydrophobic residues sit on one face of the helix:
    the classic "amphipathic" shape many AMPs use to punch holes in membranes.
    """
    if not seq:
        return 0.0
    angles = np.deg2rad(angle_deg) * np.arange(len(seq))
    h = np.array([KYTE_DOOLITTLE[a] for a in seq])
    return float(np.hypot((h * np.cos(angles)).sum(), (h * np.sin(angles)).sum()) / len(seq))


def physchem(seq: str) -> np.ndarray:
    """8 interpretable physico-chemical features (see PHYSCHEM_NAMES)."""
    n = max(len(seq), 1)
    return np.array([
        len(seq),
        net_charge(seq),
        sum(seq.count(a) for a in "KR") / n,
        sum(seq.count(a) for a in "DE") / n,
        sum(seq.count(a) for a in "AILMFVW") / n,
        sum(seq.count(a) for a in "FWY") / n,
        float(np.mean([KYTE_DOOLITTLE[a] for a in seq])) if seq else 0.0,
        hydrophobic_moment(seq),
    ])


def featurize(seqs: list[str], kind: str = "aac+physchem") -> np.ndarray:
    """Build a feature matrix. ``kind`` is a '+'-joined list of
    'aac', 'dpc', 'physchem' or 'length'."""
    blocks = []
    for part in kind.split("+"):
        if part == "aac":
            blocks.append(np.array([aa_composition(s) for s in seqs]))
        elif part == "dpc":
            blocks.append(np.array([dipeptide_composition(s) for s in seqs]))
        elif part == "physchem":
            blocks.append(np.array([physchem(s) for s in seqs]))
        elif part == "length":
            blocks.append(np.array([[len(s)] for s in seqs], dtype=float))
        else:
            raise ValueError(f"unknown feature block: {part}")
    return np.hstack(blocks)
