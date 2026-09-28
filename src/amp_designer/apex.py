"""APEX-pathogen inference, vendored.

APEX-pathogen (Wan et al., de la Fuente lab, *Nature Microbiology* 2025, MIT
licensed) predicts MIC in µM against an 11-pathogen panel, and every one of those
11 is on this competition's 20-strain panel: all 5 Gram-positive strains, 6 of the
15 Gram-negative, and 4 of the 8 MDR isolates. It is also the model that lab used
to score the AMP-Diffusion baseline, and that lab runs Phase 2.

It earns its place by being independent, not by being better. On the 47 HydrAMP
peptides with prospective wet-lab MIC values, APEX reaches Spearman 0.50 and AUROC
0.80 for active at <= 32 µM. This repository's own MIC ensemble appears to score
higher there, but 32 of those 47 sequences are in its training data, so that
comparison is leakage; its honest number is the grouped-CV AUROC of 0.80. Two
independent models of about equal strength are worth ensembling, and they agree
only at Spearman 0.69.

Four of the eight released checkpoints are shipped. Against the full eight they
rank 4,000 peptides at Spearman 0.983, for 81 MB instead of 220 MB.

The checkpoints are pickled `nn.Module` objects rather than state dicts, so
loading them runs pickle and needs the defining class importable under its
original name. Both are handled below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from . import apex_models

# The checkpoints were pickled from a script whose module was named APEX_models.
sys.modules.setdefault("APEX_models", apex_models)

MAX_LEN = 52  # start token + 50 residues + end token
AA = "ACDEFGHIKLMNPQRSTVWY"

PATHOGENS = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus ATCC BAA-1556 (MRSA)",
    "E. faecalis ATCC 700802 (VRE)",
    "E. faecium ATCC 700221 (VRE)",
]
GRAM_NEGATIVE = [0, 1, 2, 3, 4, 5, 6]
GRAM_POSITIVE = [7, 8, 9, 10]
# The MDR isolates among the eleven: E. coli AIC222 (CRE), MRSA, and the two VRE.
MDR = [3, 8, 9, 10]


def _vocab() -> dict[str, int]:
    word2idx = {"0": 0, "1": 1, "2": 2}
    for i, aa in enumerate(AA, start=3):
        word2idx[aa] = i
    return word2idx


VOCAB = _vocab()


def encode(sequences: list[str]) -> np.ndarray:
    """Index encoding with start and end tokens, zero-padded to MAX_LEN."""
    out = np.zeros((len(sequences), MAX_LEN), dtype=np.int64)
    for i, s in enumerate(sequences):
        out[i, 0] = 1
        j = 0
        for j, c in enumerate(s[: MAX_LEN - 2]):
            out[i, j + 1] = VOCAB.get(c, 0)
        out[i, min(len(s), MAX_LEN - 2) + 1] = 2
    return out


def load(directory: Path) -> list:
    models = []
    for path in sorted(directory.glob("APEX_*")):
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        models.append(model)
    if not models:
        raise FileNotFoundError(f"no APEX checkpoints under {directory}")
    return models


@torch.no_grad()
def predict(sequences: list[str], models: list, batch: int = 2048) -> np.ndarray:
    """Return an (n, 11) array of predicted MIC in µM, averaged over the ensemble.

    The training target was -log10(MIC / 1e6), so the inverse is 10 ** (6 - y).
    Averaging happens in µM to match the released inference script exactly.
    """
    X = encode(sequences)
    total = np.zeros((len(sequences), len(PATHOGENS)), dtype=np.float64)
    for model in models:
        parts = []
        for i in range(0, len(sequences), batch):
            chunk = torch.from_numpy(X[i : i + batch])
            parts.append(10.0 ** (6.0 - model(chunk).numpy()))
        total += np.concatenate(parts) if parts else 0.0
    return total / len(models)


def summary(mic: np.ndarray) -> dict[str, np.ndarray]:
    """Collapse the 11 predictions into log2 µM quantities for ranking."""
    log2 = np.log2(np.clip(mic, 1e-3, None))
    return {
        "apex_log2_median": np.median(log2, axis=1),
        "apex_log2_gram_neg": log2[:, GRAM_NEGATIVE].mean(axis=1),
        "apex_log2_gram_pos": log2[:, GRAM_POSITIVE].mean(axis=1),
        "apex_log2_mdr": log2[:, MDR].mean(axis=1),
        "apex_log2_best": log2.min(axis=1),
        "apex_log2_worst": log2.max(axis=1),
    }
