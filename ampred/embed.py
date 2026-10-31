"""ESM-2 protein language model embeddings (optional: needs torch + transformers).

We use the smallest ESM-2 model (8M parameters, 6 layers) so everything runs
on a laptop CPU in a few minutes. Each peptide becomes one 320-number vector:
the average of the per-residue embeddings (mean pooling, special tokens removed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"


def esm_embed(
    seqs: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
    cache: str | Path | None = None,
) -> np.ndarray:
    """Mean-pooled last-layer ESM-2 embeddings, shape (n_seqs, hidden_size)."""
    if cache is not None and Path(cache).exists():
        arr = np.load(cache)
        if arr.shape[0] == len(seqs):
            return arr
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            # drop <cls> (first) and <eos> (last real token) from the average
            mask[:, 0] = 0
            lengths = enc["attention_mask"].sum(1)
            mask[torch.arange(len(batch)), lengths - 1] = 0
            m = mask.unsqueeze(-1).float()
            out.append(((hidden * m).sum(1) / m.sum(1)).numpy())
    arr = np.vstack(out)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
    return arr
