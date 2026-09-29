"""MIC, hemolysis, and AMP-classifier ensembles, and their grouped-CV report.

Gradient boosting beat an MLP on this tabular, few-thousand-row
data, most of all for hemolysis (Spearman 0.52 against 0.42, AUROC 0.77 against
0.67). The implementation is scikit-learn's histogram boosting
and not xgboost, because xgboost and torch load duplicate OpenMP runtimes and
segfault in one process on macOS, and the entry point needs both.

Splits are grouped by MMseqs2 cluster at 50% identity. DBAASP is dense with
analogue series. On the Gram-negative head an ungrouped split reports
Spearman 0.691 against the grouped 0.557, and AUROC 0.844 against 0.777.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from amp_designer import features

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AMP_TRAINING_DATA", REPO / "data" / "training"))
MODELS = REPO / "checkpoint"

MIC_CEILING = 64.0
HC50_CEILING = 128.0
POTENCY_THRESHOLD = 16.0

PANEL = {
    "gram_neg": [
        "Acinetobacter baumannii", "Enterobacter cloacae", "Escherichia coli",
        "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "Salmonella enterica",
        "Salmonella typhimurium",
    ],
    "gram_pos": [
        "Bacillus subtilis", "Staphylococcus aureus", "Enterococcus faecalis",
        "Enterococcus faecium",
    ],
}
SPECIES_TARGETS = {
    "sp_ecoli": "Escherichia coli",
    "sp_paeru": "Pseudomonas aeruginosa",
    "sp_saure": "Staphylococcus aureus",
    "sp_kpneu": "Klebsiella pneumoniae",
    "sp_abaum": "Acinetobacter baumannii",
    "sp_efaec": "Enterococcus faecalis",
    "sp_bsubt": "Bacillus subtilis",
}

REG_PARAMS = dict(
    max_iter=500, learning_rate=0.05, max_depth=6, max_leaf_nodes=31,
    min_samples_leaf=15, l2_regularization=1.0, max_features=0.5,
    early_stopping=False,
)
CLF_PARAMS = dict(REG_PARAMS)


def featurize(sequences: list[str], amidated=None) -> np.ndarray:
    d = features.descriptors(sequences)
    k2 = features.kmer_counts(sequences, k=2)
    if amidated is None:
        amidated = np.zeros(len(sequences), dtype=np.float32)
    return np.hstack(
        [d, k2, np.asarray(amidated, dtype=np.float32)[:, None]]
    ).astype(np.float32)


def cluster_ids(sequences: list[str], min_seq_id: float = 0.5) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fa = tmp / "in.fasta"
        fa.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(sequences)))
        subprocess.run(
            ["mmseqs", "easy-cluster", str(fa), str(tmp / "res"), str(tmp / "tmp"),
             "--min-seq-id", str(min_seq_id), "-c", "0.5", "--cov-mode", "1",
             "-s", "6.0", "--cluster-mode", "1", "-v", "0"],
            check=True, capture_output=True,
        )
        rep_of = {}
        for line in (tmp / "res_cluster.tsv").read_text().splitlines():
            rep, member = line.split("\t")
            rep_of[member] = rep
    reps = sorted(set(rep_of.values()))
    index = {r: i for i, r in enumerate(reps)}
    out = np.full(len(sequences), -1, dtype=np.int64)
    for i in range(len(sequences)):
        r = rep_of.get(f"s{i}")
        if r is not None:
            out[i] = index[r]
    nxt = len(reps)
    for i in np.where(out < 0)[0]:
        out[i] = nxt
        nxt += 1
    return out


def fit_ensemble(
    X, y, w, groups, *, seed: int = 0, n_folds: int = 5, classify: bool = False
):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.integers(0, n_folds, size=len(uniq))))
    folds = np.array([fold_of[g] for g in groups])

    oof = np.full(len(y), np.nan, dtype=np.float32)
    models = []
    for f in range(n_folds):
        tr, va = folds != f, folds == f
        if va.sum() == 0 or tr.sum() < 64:
            continue
        cls = HistGradientBoostingClassifier if classify else HistGradientBoostingRegressor
        params = CLF_PARAMS if classify else REG_PARAMS
        m = cls(random_state=seed * 100 + f, **params)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        oof[va] = m.predict_proba(X[va])[:, 1] if classify else m.predict(X[va])
        models.append(m)
    return models, oof


def _report(y, oof, classify: bool) -> dict:
    from scipy.stats import spearmanr

    ok = ~np.isnan(oof)
    out = {"n": int(len(y))}
    if classify:
        from sklearn.metrics import roc_auc_score

        out["auroc"] = float(roc_auc_score(y[ok], oof[ok]))
    else:
        out["spearman"] = float(spearmanr(y[ok], oof[ok]).statistic)
        out["rmse"] = float(np.sqrt(np.mean((y[ok] - oof[ok]) ** 2)))
        out["r2"] = float(1 - np.mean((y[ok] - oof[ok]) ** 2) / np.var(y[ok]))
    return out


def mic_targets() -> dict[str, pd.DataFrame]:
    mic = pd.read_csv(DATA / "mic.csv.gz")
    mic = mic[mic["mic_uM"].notna()].copy()
    mic["amidated"] = mic["amidated"].replace(-1, 0)
    agg = mic.groupby(["sequence", "species", "amidated"], as_index=False).agg(
        mic_uM=("mic_uM", "median"), n=("mic_uM", "size")
    )
    agg["y"] = np.log2(np.clip(agg["mic_uM"], 0.05, MIC_CEILING))

    out = {}
    for name, species in PANEL.items():
        sel = agg[agg["species"].isin(species)]
        out[name] = sel.groupby(["sequence", "amidated"], as_index=False).agg(
            y=("y", "mean"), n=("n", "sum")
        )
    for key, sp in SPECIES_TARGETS.items():
        sel = agg[agg["species"] == sp]
        if len(sel) >= 400:
            out[key] = sel[["sequence", "amidated", "y", "n"]].reset_index(drop=True)
    out["all"] = agg.groupby(["sequence", "amidated"], as_index=False).agg(
        y=("y", "mean"), n=("n", "sum")
    )
    return out


def hemolysis_target() -> pd.DataFrame:
    frames = []
    hemopi = Path(os.environ.get("AMP_HEMOPI", REPO.parent / "work" / "data" / "HemoPI2" / "Dataset"))
    for name in ("cross_val_dataset.csv", "independent_dataset.csv"):
        df = pd.read_csv(hemopi / name).rename(
            columns={"SEQUENCE": "sequence", "μM": "hc50_uM"}
        )
        frames.append(df[["sequence", "hc50_uM"]])
    hyd = pd.read_csv(
        Path(os.environ.get("AMP_UPSTREAM", REPO.parent / "work" / "upstream"))
        / "hydramp-starter-kit" / "experimental" / "hemolysis.csv"
    )
    frames.append(hyd[["sequence", "hc50_uM"]])

    # Human-erythrocyte HC50 from the direct DBAASP harvest. It adds 370 sequences
    # the HemoPI2 release does not carry, and the two agree at Spearman 0.916 on
    # the 1,306 they share, which is the check that they are measuring the same
    # thing before they are pooled.
    dbaasp = Path(
        os.environ.get(
            "AMP_DBAASP", REPO.parent / "work" / "data" / "dbaasp" / "dbaasp_flat.csv"
        )
    )
    if dbaasp.exists():
        d = pd.read_csv(dbaasp)
        d = d[d["human_hc50_uM"].notna()].drop_duplicates("sequence")
        # A censored value is a lower bound. One written ">150" is safely above the
        # 128 µM ceiling and can be clipped to it. One written ">100" could be
        # anywhere above 100 and is dropped, never guessed.
        ambiguous = d["hc50_censored"] & (d["human_hc50_uM"] < HC50_CEILING)
        d = d[~ambiguous].copy()
        d.loc[d["hc50_censored"], "human_hc50_uM"] = HC50_CEILING
        frames.append(
            d[["sequence", "human_hc50_uM"]].rename(
                columns={"human_hc50_uM": "hc50_uM"}
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    df = df[df["sequence"].map(lambda s: bool(s) and not (set(s) - aa))]
    # The descriptor encoder is sized to the competition's 50-residue ceiling, and
    # anything outside 8 to 50 is out of scope for this submission anyway.
    df = df[df["sequence"].str.len().between(8, 50)]
    df = df[df["hc50_uM"].notna() & (df["hc50_uM"] > 0)]
    df["log2_hc50"] = np.log2(df["hc50_uM"])
    agg = df.groupby("sequence", as_index=False).agg(
        y=("log2_hc50", "mean"), n=("log2_hc50", "size")
    )
    agg["safe"] = (agg["y"] >= np.log2(HC50_CEILING)).astype(np.float32)
    return agg


def amp_classifier_target(seed: int = 0, cap: int = 30000) -> pd.DataFrame:
    pos = (DATA / "amp_positives.txt").read_text().split()
    neg = (DATA / "amp_negatives.txt").read_text().split()
    df = pd.DataFrame(
        {"sequence": pos + neg, "y": [1.0] * len(pos) + [0.0] * len(neg)}
    )
    return df.sample(n=min(cap, len(df)), random_state=seed).reset_index(drop=True)


def train_all(seed: int = 0) -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report, bundle = {}, {}

    targets = mic_targets()
    all_seqs = sorted({s for df in targets.values() for s in df["sequence"]})
    cluster_of = dict(zip(all_seqs, cluster_ids(all_seqs)))

    for name, df in targets.items():
        if len(df) < 300:
            continue
        X = featurize(df["sequence"].tolist(), df["amidated"].to_numpy())
        y = df["y"].to_numpy(dtype=np.float32)
        w = np.log1p(df["n"].to_numpy(dtype=np.float32))
        groups = np.array([cluster_of[s] for s in df["sequence"]])
        models, oof = fit_ensemble(X, y, w, groups, seed=seed)
        rep = _report(y, oof, classify=False)
        ok = ~np.isnan(oof)
        hit = y <= np.log2(POTENCY_THRESHOLD)
        if hit[ok].sum() > 10 and (~hit[ok]).sum() > 10:
            from sklearn.metrics import roc_auc_score

            rep["auroc_active_16uM"] = float(roc_auc_score(hit[ok], -oof[ok]))
        report[f"mic:{name}"] = rep
        bundle[f"mic:{name}"] = models
        print(f"mic:{name:10s} n={rep['n']:6d} rho={rep['spearman']:.3f} "
              f"rmse={rep['rmse']:.2f} "
              f"auroc={rep.get('auroc_active_16uM', float('nan')):.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    hem = hemolysis_target()
    X = featurize(hem["sequence"].tolist())
    groups = cluster_ids(hem["sequence"].tolist())
    w = np.ones(len(hem), dtype=np.float32)
    models, oof = fit_ensemble(X, hem["y"].to_numpy(np.float32), w, groups, seed=seed)
    report["hem:log2_hc50"] = _report(hem["y"].to_numpy(np.float32), oof, False)
    bundle["hem:log2_hc50"] = models
    cmodels, coof = fit_ensemble(
        X, hem["safe"].to_numpy(np.float32), w, groups, seed=seed, classify=True
    )
    report["hem:safe"] = _report(hem["safe"].to_numpy(np.float32), coof, True)
    bundle["hem:safe"] = cmodels
    print(f"hem            n={len(hem):6d} rho={report['hem:log2_hc50']['spearman']:.3f} "
          f"auroc={report['hem:safe']['auroc']:.3f} [{time.time()-t0:.0f}s]", flush=True)

    amp = amp_classifier_target(seed)
    X = featurize(amp["sequence"].tolist())
    groups = cluster_ids(amp["sequence"].tolist())
    w = np.ones(len(amp), dtype=np.float32)
    models, oof = fit_ensemble(
        X, amp["y"].to_numpy(np.float32), w, groups, seed=seed, classify=True
    )
    report["amp:clf"] = _report(amp["y"].to_numpy(np.float32), oof, True)
    bundle["amp:clf"] = models
    print(f"amp:clf        n={len(amp):6d} auroc={report['amp:clf']['auroc']:.3f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    import pickle

    import gzip

    with gzip.open(MODELS / "scorers.pkl.gz", "wb", compresslevel=6) as fh:
        pickle.dump(bundle, fh, protocol=5)
    (MODELS / "scorer_report.json").write_text(json.dumps(report, indent=2))
    return report


def load(path: Path | None = None) -> dict:
    import pickle

    import gzip

    path = path or (MODELS / "scorers.pkl.gz")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as fh:
        return pickle.load(fh)


def score(sequences: list[str], bundle: dict, amidated=None) -> dict[str, np.ndarray]:
    X = featurize(sequences, amidated)
    out = {}
    for name, models in bundle.items():
        classify = name.endswith("safe") or name == "amp:clf"
        acc = np.zeros(len(sequences), dtype=np.float64)
        for m in models:
            acc += m.predict_proba(X)[:, 1] if classify else m.predict(X)
        out[name] = (acc / len(models)).astype(np.float32)
    return out


if __name__ == "__main__":
    train_all()
