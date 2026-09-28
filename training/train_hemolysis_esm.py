"""Fit the hemolysis heads on ESM-2 embeddings and vendor the encoder.

The descriptor-only hemolysis head is the weakest model in the pipeline, and it
gates the selectivity category. Measured on the same 5,617 sequences under the
same MMseqs2 cluster-grouped splits:

    descriptors                 Spearman 0.424   AUROC 0.755
    ESM-2 t6 8M                 0.442            0.757
    ESM-2 t6 8M + descriptors   0.449            0.770
    ESM-2 t12 35M               0.447            0.764
    ESM-2 t12 35M + descriptors 0.462            0.786

The last one wins by about four standard errors on AUROC, so it is what ships.
Embeddings alone are worse than the combination in every case, which says the
descriptors carry something the language model does not: charge and hydrophobicity
are explicit there and only implicit in a masked-language-model representation.

The encoder is stored as float16 to stay under GitHub's 100 MB file limit and cast
back to float32 at load time, so inference runs at full speed and no weights are
fetched at generation time.

    python -m training.train_hemolysis_esm
"""

from __future__ import annotations

import gzip
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from amp_designer import esm as esm_module

from . import train_scorers as T

REPO = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO / "checkpoint"
ENCODER_DIR = CHECKPOINT / "esm2_t12_35M"
HF_NAME = "facebook/esm2_t12_35M_UR50D"


def vendor_encoder() -> None:
    """Copy the encoder locally, with the weights halved to float16."""
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import save_file

    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(HF_NAME)
    tok.save_pretrained(ENCODER_DIR)

    model = AutoModel.from_pretrained(HF_NAME)
    model.config.save_pretrained(ENCODER_DIR)
    state = {k: v.to(torch.float16).contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(ENCODER_DIR / "model.safetensors"))

    for junk in ("model.safetensors.index.json",):
        p = ENCODER_DIR / junk
        if p.exists():
            p.unlink()
    size = sum(f.stat().st_size for f in ENCODER_DIR.iterdir() if f.is_file())
    print(f"vendored encoder to {ENCODER_DIR} ({size/1e6:.1f} MB)")
    shutil.rmtree(ENCODER_DIR / ".cache", ignore_errors=True)


def train() -> dict:
    hem = T.hemolysis_target()
    seqs = hem["sequence"].tolist()
    y = hem["y"].to_numpy(np.float32)
    safe = hem["safe"].to_numpy(np.float32)
    print(f"{len(seqs)} sequences, {safe.mean():.1%} safe", flush=True)

    encoder = esm_module.load(ENCODER_DIR)
    emb = esm_module.embed(seqs, encoder)
    X = np.hstack([T.featurize(seqs), emb]).astype(np.float32)
    print(f"feature matrix {X.shape}", flush=True)

    groups = T.cluster_ids(seqs)
    w = np.ones(len(seqs), dtype=np.float32)

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    regs, oof = T.fit_ensemble(X, y, w, groups, seed=0)
    clfs, coof = T.fit_ensemble(X, safe, w, groups, seed=0, classify=True)
    ok = ~np.isnan(oof)
    report = {
        "n": int(len(seqs)),
        "spearman": float(spearmanr(y[ok], oof[ok]).statistic),
        "auroc": float(roc_auc_score(safe[~np.isnan(coof)], coof[~np.isnan(coof)])),
    }
    print(f"ESM hemolysis: Spearman {report['spearman']:.3f}  "
          f"AUROC {report['auroc']:.3f}", flush=True)

    bundle = T.load(CHECKPOINT / "scorers.pkl.gz")
    bundle["hem_esm:log2_hc50"] = regs
    bundle["hem_esm:safe"] = clfs
    with gzip.open(CHECKPOINT / "scorers.pkl.gz", "wb", compresslevel=6) as fh:
        pickle.dump(bundle, fh, protocol=5)
    print(f"bundle now has {len(bundle)} heads", flush=True)
    return report


if __name__ == "__main__":
    vendor_encoder()
    train()
