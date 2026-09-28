"""Inference for the shipped MIC, hemolysis, and AMP-classifier ensembles.

Training lives in `training/train_scorers.py`. This module rebuilds nothing and
only runs the fitted scikit-learn ensembles stored in `checkpoint/scorers.pkl`.

Predictions are made at `amidated=0`. Most potent entries in DBAASP are
C-terminally amidated and amidation typically buys several-fold potency, but the
competition requires free termini, so asking the model for the amidated case
would systematically overstate what these designs can do.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np

from . import features

MIC_CEILING = 64.0
HC50_CEILING = 128.0
POTENCY_THRESHOLD = 16.0


def featurize(sequences: list[str], amidated=None) -> np.ndarray:
    d = features.descriptors(sequences)
    k2 = features.kmer_counts(sequences, k=2)
    if amidated is None:
        amidated = np.zeros(len(sequences), dtype=np.float32)
    return np.hstack(
        [d, k2, np.asarray(amidated, dtype=np.float32)[:, None]]
    ).astype(np.float32)


def load(path: Path) -> dict:
    """Load the ensembles. Gzip cuts the checkpoint from 70 MB to 23 MB and costs
    a tenth of a second, which is worth it in a repository the validator clones."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        return pickle.load(fh)


# Heads that take ESM embeddings alongside the descriptors. They expect a wider
# feature matrix than the rest, so `score` leaves them to `esm.score`.
ESM_PREFIX = "hem_esm:"


def _is_probability(name: str) -> bool:
    return name.endswith("safe") or name == "amp:clf"


def score(
    sequences: list[str], bundle: dict, amidated=None, batch: int = 50_000
) -> dict[str, np.ndarray]:
    X = featurize(sequences, amidated)
    out: dict[str, np.ndarray] = {}
    for name in sorted(bundle):
        if name.startswith(ESM_PREFIX):
            continue
        models = bundle[name]
        classify = _is_probability(name)
        acc = np.zeros(len(sequences), dtype=np.float64)
        for m in models:
            parts = []
            for i in range(0, len(sequences), batch):
                chunk = X[i : i + batch]
                parts.append(
                    m.predict_proba(chunk)[:, 1] if classify else m.predict(chunk)
                )
            acc += np.concatenate(parts) if parts else 0.0
        out[name] = (acc / len(models)).astype(np.float32)
    return out


def panel_summary(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Collapse per-species predictions into the quantities Phase 2 scores.

    The panel is 15 Gram-negative and 5 Gram-positive strains, so broad-spectrum
    success rate is dominated by Gram-negative potency. MIC50 is approximated by
    the panel-weighted mean of the per-species predictions in log2 space, which is
    the space the models were fitted in.
    """
    species_neg = ["mic:sp_ecoli", "mic:sp_paeru", "mic:sp_kpneu", "mic:sp_abaum"]
    species_pos = ["mic:sp_saure", "mic:sp_efaec", "mic:sp_bsubt"]

    def stack(names):
        cols = [scores[n] for n in names if n in scores]
        return np.vstack(cols) if cols else None

    neg, pos = stack(species_neg), stack(species_pos)
    out: dict[str, np.ndarray] = {}
    out["log2_mic_neg"] = neg.mean(axis=0) if neg is not None else scores["mic:gram_neg"]
    out["log2_mic_pos"] = pos.mean(axis=0) if pos is not None else scores["mic:gram_pos"]
    out["log2_mic50"] = 0.75 * out["log2_mic_neg"] + 0.25 * out["log2_mic_pos"]
    # The hardest Gram-negative species is what separates broad-spectrum peptides
    # from ones that only clear E. coli.
    out["log2_mic_worst_neg"] = neg.max(axis=0) if neg is not None else out["log2_mic_neg"]

    log2_hc50 = np.minimum(scores["hem:log2_hc50"], np.log2(HC50_CEILING))
    out["log2_hc50"] = log2_hc50
    out["p_safe"] = scores["hem:safe"]
    out["log2_safety_window"] = log2_hc50 - np.maximum(
        out["log2_mic50"], np.log2(0.5)
    )
    return out
