"""ESM-2 embeddings for the hemolysis head.

The encoder is vendored under `checkpoint/esm2_t12_35M/` so generation fetches
nothing. Weights are stored as float16 to stay under GitHub's file size limit and
are cast to float32 on load, which is both faster on CPU and what the heads were
fitted against.

Embeddings are the mean over residue positions, excluding the leading CLS token
and the trailing EOS token. Including them shifts short peptides most, because the
two special positions are a larger share of a 12-mer than of a 40-mer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MAX_LEN = 52  # CLS + 50 residues + EOS


def load(directory: Path):
    """Return (tokenizer, model) read from a local directory, never the network."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(directory), local_files_only=True, dtype=torch.float32
    )
    model.eval()
    return tok, model


@torch.no_grad()
def embed(sequences: list[str], encoder, batch: int = 256) -> np.ndarray:
    """Mean-pooled residue embeddings, shape (n, hidden)."""
    tok, model = encoder
    out = []
    for i in range(0, len(sequences), batch):
        chunk = sequences[i : i + batch]
        enc = tok(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
        )
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0
        lengths = mask.sum(1)
        for j, L in enumerate(lengths):
            mask[j, L] = 0
        mask = mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.numpy())
    return np.vstack(out).astype(np.float32) if out else np.zeros((0, 1), np.float32)


def score(
    sequences: list[str], bundle: dict, encoder, descriptors: np.ndarray
) -> dict[str, np.ndarray]:
    """Run the ESM hemolysis heads over `sequences`."""
    X = np.hstack([descriptors, embed(sequences, encoder)]).astype(np.float32)
    out = {}
    for name in ("hem_esm:log2_hc50", "hem_esm:safe"):
        models = bundle[name]
        classify = name.endswith("safe")
        acc = np.zeros(len(sequences), dtype=np.float64)
        for m in models:
            acc += m.predict_proba(X)[:, 1] if classify else m.predict(X)
        out[name] = (acc / len(models)).astype(np.float32)
    return out
