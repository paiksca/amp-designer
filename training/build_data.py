"""Assemble the public peptide datasets this project trains on.

Every source is a public repository or database; `data/training/README.md` lists
them with their licences. Clone them first (see that file), then point
`AMP_UPSTREAM` at the directory holding the clones and run:

    AMP_UPSTREAM=/path/to/clones python -m training.build_data

The assembled tables are written to `data/training/`, which is what the
competition's training-data disclosure requirement refers to.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
UP = Path(os.environ.get("AMP_UPSTREAM", REPO.parent / "work" / "upstream"))
CHALLENGE = UP / "amp-challenge-2027"
BATTLE = UP / "battleamp-snakemake"
HYDRA = UP / "hydramp-starter-kit"
OUT = Path(os.environ.get("AMP_TRAINING_DATA", REPO / "data" / "training"))

AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 8, 50

# Average residue masses, Da. Peptide MW = sum(residues) + water.
RESIDUE_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
WATER = 18.0153


def molecular_weight(seq: str) -> float:
    return sum(RESIDUE_MASS[c] for c in seq) + WATER


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records, header, parts = [], None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header, parts = line[1:], []
        else:
            parts.append(line.upper())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def clean(seqs) -> list[str]:
    """Keep canonical sequences inside the competition length window, deduplicated."""
    seen, out = set(), []
    for s in seqs:
        if not isinstance(s, str):
            continue
        s = s.strip().upper()
        if not s or set(s) - AA:
            continue
        if not (MIN_LEN <= len(s) <= MAX_LEN):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# --------------------------------------------------------------------------- corpora


def challenge_reference() -> list[str]:
    """The 39,448-sequence reference set the validator checks against."""
    return [s.upper() for _, s in read_fasta(CHALLENGE / "data" / "antibacterial.fasta")]


def challenge_reference_meta() -> pd.DataFrame:
    rows = []
    for header, seq in read_fasta(CHALLENGE / "data" / "antibacterial.fasta"):
        m = re.search(r"activity=(\S+)", header)
        rows.append(
            {
                "sequence": seq.upper(),
                "activity": m.group(1) if m else "",
                "id": header.split()[0],
            }
        )
    return pd.DataFrame(rows)


def amp_positives() -> list[str]:
    """Known antibacterial peptides pooled across the local public sources."""
    seqs: list[str] = []
    seqs += [s for _, s in read_fasta(CHALLENGE / "data" / "antibacterial.fasta")]
    seqs += [s for _, s in read_fasta(BATTLE / "data" / "amp_positive.fasta")]
    for name in ("broad", "gramplus", "gramminus"):
        seqs += [s for _, s in read_fasta(BATTLE / "data" / "activity" / f"{name}_positive.fasta")]
    df = pd.read_csv(HYDRA / "data" / "training" / "unlabelled_positive.csv")
    col = "sequence" if "sequence" in df.columns else df.columns[-1]
    seqs += df[col].tolist()
    ds = pd.read_csv(BATTLE / "data" / "dbaasp" / "dbaasp_sequences.csv")
    seqs += ds["sequence"].tolist()
    return clean(seqs)


def amp_negatives() -> list[str]:
    seqs: list[str] = []
    seqs += [s for _, s in read_fasta(BATTLE / "data" / "amp_negative.fasta")]
    for name in ("broad", "gramplus", "gramminus"):
        seqs += [s for _, s in read_fasta(BATTLE / "data" / "activity" / f"{name}_negative.fasta")]
    df = pd.read_csv(HYDRA / "data" / "training" / "unlabelled_negative.csv")
    col = "sequence" if "sequence" in df.columns else df.columns[-1]
    seqs += df[col].tolist()
    return clean(seqs)


# ----------------------------------------------------------------------------- MIC

GRAM_POSITIVE_GENERA = {
    "Staphylococcus", "Streptococcus", "Enterococcus", "Bacillus", "Listeria",
    "Clostridium", "Corynebacterium", "Micrococcus", "Lactobacillus",
    "Mycobacterium", "Propionibacterium", "Cutibacterium", "Nocardia",
    "Rhodococcus", "Actinomyces", "Peptostreptococcus", "Lactococcus",
    "Leuconostoc", "Sarcina", "Kocuria", "Paenibacillus", "Geobacillus",
}

# The Phase-2 panel, mapped onto the species labels DBAASP uses.
PANEL_GRAM_NEG = [
    "Acinetobacter baumannii", "Enterobacter cloacae", "Escherichia coli",
    "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "Salmonella enterica",
]
PANEL_GRAM_POS = [
    "Bacillus subtilis", "Staphylococcus aureus", "Enterococcus faecalis",
    "Enterococcus faecium",
]


def _gram_of(species: str) -> str:
    genus = str(species).split()[0] if isinstance(species, str) and species else ""
    return "+" if genus in GRAM_POSITIVE_GENERA else "-"


def _parse_concentration(raw) -> tuple[float | None, bool]:
    """Return (value, censored). '>256' means the assay never reached inhibition."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, False
    s = str(raw).strip()
    censored = s.startswith(">") or s.startswith("≥")
    s = s.lstrip("><≥≤=~ ")
    m = re.match(r"^(\d+(?:\.\d+)?)", s.replace(",", "."))
    if not m:
        return None, censored
    return float(m.group(1)), censored


def mic_table() -> pd.DataFrame:
    """One row per (sequence, target species) with MIC in µM.

    Keeps the C-terminal amidation flag, because amidation typically improves MIC
    several-fold and our designs must be free acid.
    """
    seqs = pd.read_csv(BATTLE / "data" / "dbaasp" / "dbaasp_sequences.csv")
    seqs = seqs.rename(columns={"Unnamed: 0": "row"})
    seqs["amidated"] = seqs["cTerminus"].fillna("").str.contains("AMD").astype(int)
    seqs = seqs[["id", "sequence", "amidated"]].drop_duplicates("id")

    act = pd.read_csv(BATTLE / "data" / "dbaasp" / "dbaasp_activity.csv")
    act = act.rename(columns={"Unnamed: 0": "kind"})
    df = act.merge(seqs, on="id", how="inner")

    parsed = df["concentration"].map(_parse_concentration)
    df["value"] = [p[0] for p in parsed]
    df["censored"] = [p[1] for p in parsed]
    df = df[df["value"].notna() & (df["value"] > 0)]

    df["sequence"] = df["sequence"].str.upper().str.strip()
    ok = df["sequence"].map(lambda s: isinstance(s, str) and s and not (set(s) - AA))
    df = df[ok]
    df = df[df["sequence"].str.len().between(MIN_LEN, MAX_LEN)]

    mw = df["sequence"].map(molecular_weight)
    is_mass = df["unit"].astype(str).str.contains("g/ml", na=False)
    df["mic_uM"] = np.where(is_mass, df["value"] * 1000.0 / mw, df["value"])

    df["species"] = df["targetSpecies"].astype(str).str.split().str[:2].str.join(" ")
    df["gram"] = df["species"].map(_gram_of)
    df = df[df["mic_uM"].between(0.01, 4096)]
    return df[
        ["sequence", "amidated", "targetSpecies", "species", "gram", "mic_uM", "censored"]
    ].reset_index(drop=True)


def dbaasp_harvest() -> pd.DataFrame:
    """MIC rows from a direct harvest of the DBAASP detail API.

    The DBAASP slice redistributed inside battleamp-snakemake is an older snapshot.
    A fresh harvest of all 25,542 records adds 5,134 sequences with MIC that the
    snapshot does not carry, 4,218 of them on panel species, which is a 65%
    increase over the assembled table. `work/data/dbaasp/harvest.py` fetches the
    records and `parse.py` flattens them; set `AMP_DBAASP` to the flat CSV.
    """
    path = Path(
        os.environ.get(
            "AMP_DBAASP", REPO.parent / "work" / "data" / "dbaasp" / "dbaasp_flat.csv"
        )
    )
    if not path.exists():
        return pd.DataFrame(
            columns=["sequence", "amidated", "targetSpecies", "species", "gram",
                     "mic_uM", "censored"]
        )
    df = pd.read_csv(path)
    df = df[df["mic_uM"].notna()].copy()
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    ok = df["sequence"].map(lambda s: bool(s) and not (set(s) - AA))
    df = df[ok]
    df = df[df["sequence"].str.len().between(MIN_LEN, MAX_LEN)]
    df = df[df["mic_uM"].between(0.01, 4096)]

    df["amidated"] = (
        df["c_term"].fillna("").astype(str).str.contains("AMD").astype(int)
    )
    df["targetSpecies"] = df["species"].astype(str)
    df["species"] = df["targetSpecies"].str.split().str[:2].str.join(" ")
    df["gram"] = df["species"].map(_gram_of)
    df["censored"] = df["raw_conc"].astype(str).str.startswith(">")
    return df[
        ["sequence", "amidated", "targetSpecies", "species", "gram", "mic_uM", "censored"]
    ].reset_index(drop=True)


def grampa_mic() -> pd.DataFrame:
    """HydrAMP's shipped GRAMPA slice: log10(MIC / µM) against E. coli."""
    df = pd.read_csv(HYDRA / "data" / "training" / "mic_data.csv")
    df = df.rename(columns={df.columns[0]: "row"})
    df["sequence"] = df["sequence"].str.upper().str.strip()
    ok = df["sequence"].map(lambda s: isinstance(s, str) and s and not (set(s) - AA))
    df = df[ok]
    df = df[df["sequence"].str.len().between(MIN_LEN, MAX_LEN)]
    df["mic_uM"] = 10.0 ** df["value"]
    df["species"] = "Escherichia coli"
    df["gram"] = "-"
    df["censored"] = False
    df["amidated"] = -1  # unknown in this release
    df["targetSpecies"] = "Escherichia coli"
    return df[
        ["sequence", "amidated", "targetSpecies", "species", "gram", "mic_uM", "censored"]
    ].reset_index(drop=True)


def hydramp_experimental() -> tuple[pd.DataFrame, pd.DataFrame]:
    mic = pd.read_csv(HYDRA / "experimental" / "mic.csv")
    hem = pd.read_csv(HYDRA / "experimental" / "hemolysis.csv")
    return mic, hem


def slay() -> pd.DataFrame:
    """SLAY display screen against E. coli; lfcMLE is a growth-inhibition log ratio."""
    pos = pd.read_csv(BATTLE / "data" / "slay" / "slay_positives.csv")
    neg = pd.read_csv(BATTLE / "data" / "slay" / "slay_negatives.csv")
    df = pd.concat([pos, neg], ignore_index=True)
    df["sequence"] = df["Sequence"].str.upper().str.strip()
    ok = df["sequence"].map(lambda s: isinstance(s, str) and s and not (set(s) - AA))
    df = df[ok]
    return df[["sequence", "lfcMLE", "class"]].reset_index(drop=True)


def build(verbose: bool = True) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}

    pos = amp_positives()
    neg = amp_negatives()
    neg = [s for s in neg if s not in set(pos)]
    (OUT / "amp_positives.txt").write_text("\n".join(pos))
    (OUT / "amp_negatives.txt").write_text("\n".join(neg))
    manifest["amp_positives"] = len(pos)
    manifest["amp_negatives"] = len(neg)

    ref = challenge_reference()
    (OUT / "challenge_reference.txt").write_text("\n".join(ref))
    manifest["challenge_reference"] = len(ref)

    mic = pd.concat([mic_table(), dbaasp_harvest(), grampa_mic()], ignore_index=True)
    mic.to_csv(OUT / "mic.csv.gz", index=False)
    manifest["mic_rows"] = int(len(mic))
    manifest["mic_sequences"] = int(mic["sequence"].nunique())

    sl = slay()
    sl.to_csv(OUT / "slay.csv.gz", index=False)
    manifest["slay_rows"] = int(len(sl))

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if verbose:
        print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    build()
