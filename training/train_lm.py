"""Train the property-conditioned peptide LM and build its control table.

Run after `build_data.py` and `train_scorers.py`:

    python -m training.train_lm

Writes `checkpoint/peptide_lm.pt` and `checkpoint/controls.npz`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from amp_designer import features, lm

from . import train_scorers

ROOT = Path(__file__).resolve().parent.parent
import os
DATA = Path(os.environ.get("AMP_TRAINING_DATA", ROOT / "data" / "training"))
CHECKPOINT = ROOT / "checkpoint"

# How hard sampling leans toward the potent, non-hemolytic corner of the corpus.
# Both are per-bin log-weights, so 0.55 makes the top potency bin e^2.2 more
# likely than the bottom one while leaving every observed combination reachable.
POTENCY_TILT = 0.55
SAFETY_TILT = 0.35


def device_of() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_corpus() -> tuple[list[str], np.ndarray, np.ndarray]:
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

    mic = pd.read_csv(DATA / "mic.csv.gz")
    median = mic.groupby("sequence")["mic_uM"].median()
    potent = set(median[median <= 16.0].index)
    weight = np.array([2.0 if s in potent else 1.0 for s in seqs], dtype=np.float32)
    return seqs, controls, weight


def build_control_table(controls: np.ndarray) -> None:
    """Save the observed control combinations with potency- and safety-tilted mass.

    Sampling only ever asks for a combination the corpus actually contains, which
    keeps the conditioning in distribution. The tilt shifts mass inside that set.
    """
    combos, counts = np.unique(controls, axis=0, return_counts=True)
    tilt = np.exp(POTENCY_TILT * combos[:, 2] + SAFETY_TILT * combos[:, 3])
    weights = counts.astype(np.float64) * tilt
    weights /= weights.sum()
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    np.savez(
        CHECKPOINT / "controls.npz",
        combos=combos.astype(np.int64),
        weights=weights.astype(np.float64),
    )
    print(f"control table: {len(combos)} combinations")


def train(
    steps: int = 5000,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 0,
    val_frac: float = 0.03,
) -> dict:
    torch.manual_seed(seed)
    dev = device_of()
    seqs, controls, weight = build_corpus()
    print(f"corpus {len(seqs)} sequences, device {dev}", flush=True)
    build_control_table(controls)

    tokens = lm.encode_batch(seqs, controls)
    n = tokens.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])

    model = lm.PeptideLM().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {n_params/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95)
    )

    tokens_dev = tokens.to(dev)
    w = torch.from_numpy(weight)[tr_idx]
    probs = (w / w.sum()).numpy()

    history, t0 = [], time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lm.cosine_lr(step, steps, lr, warmup=250)
        pick = rng.choice(len(tr_idx), size=batch_size, p=probs)
        batch = tokens_dev[tr_idx[pick]]
        logits = model(batch[:, :-1])
        target = batch[:, 1:].clone()
        target[:, : lm.N_CONTROL] = -100
        target[target == lm.PAD] = -100
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vb = tokens_dev[val_idx[: min(2048, n_val)]]
                vlogits = model(vb[:, :-1])
                vtarget = vb[:, 1:].clone()
                vtarget[:, : lm.N_CONTROL] = -100
                vtarget[vtarget == lm.PAD] = -100
                vloss = float(
                    torch.nn.functional.cross_entropy(
                        vlogits.reshape(-1, vlogits.size(-1)),
                        vtarget.reshape(-1),
                        ignore_index=-100,
                    )
                )
            model.train()
            history.append({"step": step, "train": float(loss), "val": vloss})
            print(
                f"step {step:5d} train {float(loss):.4f} val {vloss:.4f} "
                f"ppl {np.exp(vloss):.2f} ({time.time()-t0:.0f}s)",
                flush=True,
            )

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu")
    torch.save(
        {"state_dict": model.state_dict(), "config": lm.Config().__dict__},
        CHECKPOINT / "peptide_lm.pt",
    )
    report = {
        "n_params": int(n_params),
        "corpus": len(seqs),
        "steps": steps,
        "final_val_loss": history[-1]["val"],
        "final_val_ppl": float(np.exp(history[-1]["val"])),
        "history": history,
    }
    (CHECKPOINT / "lm_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
