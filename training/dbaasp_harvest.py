"""Fetch every DBAASP peptide record and flatten it into one table.

The DBAASP slice redistributed inside battleamp-snakemake is an older snapshot.
Harvesting the detail API directly adds 5,134 sequences with MIC and 370 with
human-erythrocyte HC50 that the snapshot does not carry, which is a 65% increase
in distinct MIC sequences.

    python -m training.dbaasp_harvest --out data/dbaasp
    AMP_DBAASP=data/dbaasp/dbaasp_flat.csv python -m training.build_data

The fetch is resumable: a record already on disk is skipped, so re-running it
continues rather than restarting. The server throttles above about eight workers,
so raising the worker count does not help. A full pass takes a few hours.

DBAASP is CC BY 4.0. Cite Pirtskhalava et al., Nucleic Acids Research 49:D288
(2021).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time

NUMBER = re.compile(r"^\s*[><=~]*\s*(\d+(?:\.\d+)?)")
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_ID = 30_000
CANONICAL = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")

# Average residue masses, for converting µg/ml to µM.
AVG_MW = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14, "E": 129.12,
    "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16, "L": 113.16, "K": 128.17,
    "M": 131.19, "F": 147.18, "P": 97.12, "S": 87.08, "T": 101.10, "W": 186.21,
    "Y": 163.18, "V": 99.13,
}


def molecular_weight(seq: str) -> float:
    return sum(AVG_MW[c] for c in seq) + 18.02


def to_micromolar(value, unit, seq: str) -> float | None:
    """Concentration in µM. `unit` is sometimes a bare string and sometimes a dict."""
    if value is None:
        return None
    if isinstance(unit, dict):
        unit = unit.get("name") or unit.get("value")
    # Concentrations are written as "32", ">1000" or "2.1-4.2"; take the leading
    # number and keep the censoring flag separately.
    match = NUMBER.match(str(value))
    if not match:
        return None
    value = float(match.group(1))
    u = str(unit or "").lower().replace("µ", "u")
    if "um" in u:
        return float(value)
    if "g/ml" in u or "g/l" in u:  # µg/ml is the same as mg/L
        return float(value) * 1000.0 / molecular_weight(seq)
    return None


def fetch_one(raw_dir: Path, peptide_id: int) -> None:
    path = raw_dir / f"{peptide_id}.json"
    if path.exists() and path.stat().st_size > 200:
        return
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                f"https://dbaasp.org/peptides/{peptide_id}",
                headers={"User-Agent": "amp-designer/1.0", "Accept": "application/json"},
            )
            body = urllib.request.urlopen(request, timeout=45).read()
            if len(body) > 200:
                path.write_bytes(body)
            return
        except Exception:
            time.sleep(1 + attempt * 2)


def harvest(raw_dir: Path, workers: int = 8) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(
            pool.map(lambda i: fetch_one(raw_dir, i), range(1, MAX_ID + 1))
        ):
            if n % 2000 == 0:
                print(f"  {n} records", flush=True)
    print(f"harvested {len(list(raw_dir.glob('*.json')))} records", flush=True)


def _name_of(field):
    """Several DBAASP fields arrive either as a bare string or as {"name": ...}.

    Unwrapping every one of them matters: leaving `targetSpecies` wrapped turns the
    species column into a dict repr, every panel match fails silently, and the
    harvest contributes nothing while appearing to load fine.
    """
    if isinstance(field, dict):
        return field.get("name") or field.get("value")
    return field


def parse(raw_dir: Path, out: Path) -> int:
    rows = []
    for path in glob.glob(str(raw_dir / "*.json")):
        try:
            record = json.load(open(path))
        except Exception:
            continue
        seq = (record.get("sequence") or "").upper()
        if not seq or not CANONICAL.fullmatch(seq):
            continue

        n_term = _name_of(record.get("nTerminus"))
        c_term = _name_of(record.get("cTerminus"))
        complexity = _name_of(record.get("complexity")) or ""
        unmodified = not record.get("unusualAminoAcids") and not record.get(
            "intrachainBonds"
        ) and not record.get("interchainBonds")

        # Human-erythrocyte HC50. A value written ">100" is a lower bound, not a
        # measurement, and most non-hemolytic peptides are recorded that way, so
        # the flag has to travel with the number. Taking the minimum across
        # records is the conservative reading when a peptide has several.
        hc50, hc50_censored = None, False
        for entry in record.get("hemoliticCytotoxicActivities") or []:
            target = str(_name_of(entry.get("targetCell")) or "").lower()
            # Phase 2 measures human red blood cells. Sheep, rabbit and horse
            # erythrocytes differ in sensitivity, so pooling species would repeat
            # the mistake that pooling MBC with MIC made.
            if "erythrocyte" not in target or "human" not in target:
                continue
            raw = str(entry.get("concentration") or "")
            value = to_micromolar(raw, entry.get("unit"), seq)
            if value is None:
                continue
            censored = raw.strip().startswith(">")
            if hc50 is None or value < hc50:
                hc50, hc50_censored = value, censored

        for entry in record.get("targetActivities") or []:
            # DBAASP records several potency measures under one list. MBC, IC50,
            # MFC, LC, LD50 and EC50 are different quantities on different scales,
            # and MIC50 and MIC90 are panel statistics rather than per-strain
            # values. Pooling them cost 0.05 AUROC when it was tried.
            if _name_of(entry.get("activityMeasureGroup")) != "MIC":
                continue
            mic = to_micromolar(entry.get("concentration"), entry.get("unit"), seq)
            rows.append(
                {
                    "dbaasp_id": record.get("id"),
                    "sequence": seq,
                    "length": len(seq),
                    "species": _name_of(entry.get("targetSpecies")),
                    "mic_uM": mic,
                    "measure": "MIC",
                    "raw_conc": entry.get("concentration"),
                    "unit": _name_of(entry.get("unit")),
                    "human_hc50_uM": hc50,
                    "hc50_censored": hc50_censored,
                    "n_term": n_term,
                    "c_term": c_term,
                    "free_termini": not n_term and not c_term,
                    "unmodified": unmodified,
                    "monomer": complexity.lower() == "monomer",
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} rows -> {out}", flush=True)
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/dbaasp"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()

    raw = args.out / "raw"
    if not args.skip_fetch:
        harvest(raw, args.workers)
    parse(raw, args.out / "dbaasp_flat.csv")
