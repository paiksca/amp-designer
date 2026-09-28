"""Rebuild `checkpoint/controls.npz` without retraining the language model.

The control table is the distribution sampling draws its conditioning from. It is
derived from the training corpus, not from the model weights, so the tilt toward
potency and hemolytic safety can be changed on its own.

    python -m training.build_controls --potency 0.55 --safety 0.70
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from amp_designer import features, lm

from . import train_scorers

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AMP_TRAINING_DATA", ROOT / "data" / "training"))
CHECKPOINT = ROOT / "checkpoint"


def corpus_controls() -> np.ndarray:
    seqs = (DATA / "amp_positives.txt").read_text().split()
    seqs = [s for s in seqs if lm.MIN_LEN <= len(s) <= lm.MAX_LEN]

    bundle = train_scorers.load(CHECKPOINT / "scorers.pkl.gz")
    wanted = {"mic:gram_neg", "hem:safe"}
    raw = train_scorers.score(seqs, {k: v for k, v in bundle.items() if k in wanted})

    charge = features.net_charge(seqs)
    lengths = np.array([len(s) for s in seqs])
    controls = np.zeros((len(seqs), 4), dtype=np.int64)
    controls[:, 0] = [lm.bin_of(v, lm.LEN_EDGES) for v in lengths]
    controls[:, 1] = [lm.bin_of(v, lm.CHARGE_EDGES) for v in charge]
    controls[:, 2] = lm.quantile_bin(-raw["mic:gram_neg"], lm.POTENCY_BINS)
    controls[:, 3] = lm.quantile_bin(raw["hem:safe"], lm.SAFETY_BINS)
    return controls


def build(potency: float, safety: float) -> None:
    controls = corpus_controls()
    combos, counts = np.unique(controls, axis=0, return_counts=True)
    tilt = np.exp(potency * combos[:, 2] + safety * combos[:, 3])
    weights = counts.astype(np.float64) * tilt
    weights /= weights.sum()
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    np.savez(
        CHECKPOINT / "controls.npz",
        combos=combos.astype(np.int64),
        weights=weights.astype(np.float64),
    )
    top_safety = weights[combos[:, 3] == lm.SAFETY_BINS - 1].sum()
    top_potency = weights[combos[:, 2] == lm.POTENCY_BINS - 1].sum()
    print(f"{len(combos)} control combinations")
    print(f"  mass on the top safety quartile  {top_safety:.3f}")
    print(f"  mass on the top potency quintile {top_potency:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--potency", type=float, default=0.55)
    ap.add_argument("--safety", type=float, default=0.35)
    args = ap.parse_args()
    build(args.potency, args.safety)
