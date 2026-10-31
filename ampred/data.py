"""Download and parse the AMP Scanner v2 benchmark (Veltri et al., 2018).

The FASTA files are fetched from the authors' public GitHub repository at a
pinned commit, so no Kaggle login is needed and the data never changes.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

COMMIT = "16d48ef78d150853bd4bd8a3b18b50b03a357a85"
BASE_URL = (
    "https://raw.githubusercontent.com/dan-veltri/amp-scanner-v2/"
    f"{COMMIT}/original-dataset/"
)
SPLITS = {"train": "tr", "eval": "eval", "test": "te"}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class Split:
    """One data split: sequences, labels (1 = AMP, 0 = non-AMP) and IDs."""

    ids: list[str]
    seqs: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.seqs)


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into (id, sequence) pairs. Handles multi-line records."""
    records: list[tuple[str, str]] = []
    name, chunks = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks).upper()))
            name, chunks = line[1:].split()[0] if len(line) > 1 else "", []
        else:
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks).upper()))
    return records


def is_valid_peptide(seq: str) -> bool:
    """True if the sequence only uses the 20 standard amino acids."""
    return len(seq) > 0 and all(c in AMINO_ACIDS for c in seq)


def download(data_dir: str | Path = "data/raw", force: bool = False) -> Path:
    """Download the 6 FASTA files into ``data_dir`` (skips files already present)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("AMP", "DECOY"):
        for short in SPLITS.values():
            fname = f"{kind}.{short}.fa"
            out = data_dir / fname
            if out.exists() and not force:
                continue
            with urllib.request.urlopen(BASE_URL + fname, timeout=60) as resp:
                out.write_bytes(resp.read())
    return data_dir


def load_split(name: str, data_dir: str | Path = "data/raw") -> Split:
    """Load one split ("train", "eval" or "test"), AMPs first then decoys."""
    short = SPLITS[name]
    data_dir = Path(data_dir)
    ids, seqs, labels = [], [], []
    for kind, label in (("AMP", 1), ("DECOY", 0)):
        text = (data_dir / f"{kind}.{short}.fa").read_text()
        for rid, seq in parse_fasta(text):
            if is_valid_peptide(seq):
                ids.append(rid)
                seqs.append(seq)
                labels.append(label)
    return Split(ids, seqs, labels)
