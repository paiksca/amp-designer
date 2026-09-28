"""Synthesis, solubility, and the measured hemolysis design envelope.

Phase 2 never replaces a peptide that fails synthesis or comes out insoluble, and
a team's score is the arithmetic mean over its 25 tested peptides, so a failure
enters that mean at its worst possible value. Avoided risk is therefore worth
about as much as predicted potency, and these rules are cheap.

Percentages below are how often a published AMP trips each rule, measured over
9,942 unique canonical sequences of length 8-50 in the assembled MIC table. They
say what each filter costs in diversity.

Hard rejects, with the chemistry behind each:
  Cys (19.0%)      free thiols oxidise and scramble; no cyclisation is allowed here
  Met (13.8%)      oxidises to the sulfoxide during cleavage and storage
  D-[GASTCRDN]     aspartimide formation under Fmoc base; Asp-Gly is worst (7.2%)
  N-G (3.1%)       deamidates through the same succinimide intermediate
  6+ AILMFVWCY     on-resin beta-sheet aggregation, the classic difficult peptide (4.6%)
  GRAVY > 1.0      needs DMSO to dissolve at assay stock concentration (9.5%)
  charge < +2      precipitates in assay buffer near neutral (12.4%)
  N-terminal Q     cyclises to pyroglutamate; free termini leave no capping fix (1.0%)
  QQ or NN (4.9%)  amyloid-like aggregation and deamidation
  4+ identical     a single deletion is then invisible by mass (2.5% at 5+)

The envelope in `envelope_score` is separate: a graded preference, not a rule.
"""

from __future__ import annotations

import numpy as np

from . import features

HYDROPHOBIC = set("AILMFWVY")
BETA_PRONE = set("AILMFVWCY")
SHEET_FORMERS = set("IVTYFW")
POSITIVE = set("KR")
AROMATIC = set("FWY")

# Asp followed by any of these forms aspartimide under repeated Fmoc deprotection.
ASPARTIMIDE_PARTNERS = set("GASTCRDN")


def _max_run(seq: str, alphabet: set[str]) -> int:
    run = best = 0
    for c in seq:
        run = run + 1 if c in alphabet else 0
        if run > best:
            best = run
    return best


def _max_identical_run(seq: str) -> int:
    if not seq:
        return 0
    run = best = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        if run > best:
            best = run
    return best


def _max_positive_in_window(seq: str, window: int = 5) -> int:
    flags = [1 if c in POSITIVE else 0 for c in seq]
    if len(flags) <= window:
        return sum(flags)
    run = sum(flags[:window])
    best = run
    for j in range(window, len(flags)):
        run += flags[j] - flags[j - window]
        if run > best:
            best = run
    return best


def _has_aspartimide(seq: str) -> bool:
    for a, b in zip(seq, seq[1:]):
        if a == "D" and b in ASPARTIMIDE_PARTNERS:
            return True
        if a == "N" and b == "G":
            return True
    return False


def _has_amide_repeat(seq: str) -> bool:
    return any(a == b and a in "QN" for a, b in zip(seq, seq[1:]))


def library_flags(sequences: list[str]) -> np.ndarray:
    """Loose rules for the 50,000-sequence library.

    Phase 1 scores the *rate* of sequences meeting synthesizability constraints,
    so the library is pushed toward them without discarding the distribution tails
    that the realism metrics need. Only the three cheapest rejects apply.
    """
    out = np.ones(len(sequences), dtype=bool)
    for i, s in enumerate(sequences):
        if "C" in s or _max_identical_run(s) > 4 or _max_run(s, BETA_PRONE) > 6:
            out[i] = False
    return out


def strict_flags(sequences: list[str]) -> np.ndarray:
    """Every rule, for candidates that may actually be synthesised."""
    charge = features.net_charge(sequences)
    idx, lengths = features.encode(sequences)
    comp = features.composition(idx, lengths)
    gravy = comp @ features.KD_V
    sheet = sum(comp[:, features.AA_INDEX[a]] for a in SHEET_FORMERS)

    out = np.ones(len(sequences), dtype=bool)
    for i, s in enumerate(sequences):
        if "C" in s or "M" in s:
            out[i] = False
        elif _has_aspartimide(s) or _has_amide_repeat(s):
            out[i] = False
        elif s.startswith("Q"):
            out[i] = False
        elif _max_identical_run(s) > 3:
            out[i] = False
        elif _max_run(s, BETA_PRONE) > 5:
            out[i] = False
        elif _max_positive_in_window(s, 5) > 4:
            out[i] = False
        elif sheet[i] > 0.42:
            out[i] = False
        elif charge[i] < 2.0 or gravy[i] > 1.0:
            out[i] = False
        elif not (11 <= len(s) <= 26):
            # Below 11 residues potency is rare; above 26 crude purity and cost
            # both worsen, and the measured activity envelope ends around 22.
            out[i] = False
    return out


def risk_score(sequences: list[str]) -> np.ndarray:
    """A graded 0-3 penalty used to rank rather than exclude."""
    charge = features.net_charge(sequences)
    idx, lengths = features.encode(sequences)
    comp = features.composition(idx, lengths)
    gravy = comp @ features.KD_V

    risk = np.zeros(len(sequences), dtype=np.float32)
    for i, s in enumerate(sequences):
        r = 0.35 * s.count("C") + 0.10 * s.count("M")
        r += 0.15 * _has_aspartimide(s) + 0.15 * _has_amide_repeat(s)
        r += 0.12 * max(0, _max_identical_run(s) - 3)
        r += 0.12 * max(0, _max_run(s, BETA_PRONE) - 4)
        r += 0.08 * max(0, _max_run(s, {"R"}) - 2)   # poly-Arg couples slowly
        r += 0.08 * max(0, _max_run(s, {"G"}) - 2)
        r += 0.20 * max(0.0, float(gravy[i]) - 0.5)
        r += 0.20 * max(0.0, 2.0 - float(charge[i]))
        r += 0.03 * max(0, len(s) - 26)
        risk[i] = r
    return np.clip(risk, 0.0, 3.0)


def _band(x: np.ndarray, lo: float, hi: float, width: float) -> np.ndarray:
    """1 inside [lo, hi], decaying linearly to 0 over `width` on either side."""
    below = np.clip((x - (lo - width)) / width, 0.0, 1.0)
    above = np.clip(((hi + width) - x) / width, 0.0, 1.0)
    return np.minimum(below, above)


def envelope_score(sequences: list[str]) -> np.ndarray:
    """How well a sequence sits in the measured potent-and-safe envelope, 0 to 1.

    Derived from a Mann-Whitney comparison of hemolytic against safe peptides
    within the potent subset (median MIC <= 8 µM, n = 498; hemolytic HC50 <= 32 µM,
    n = 196; safe HC50 >= 128 µM, n = 125). Only descriptors that actually
    separated the two groups are used:

        Eisenberg mean hydrophobicity  p = 1e-08   safe -0.03, hemolytic +0.18
        GRAVY                          p = 1e-07   safe +0.10, hemolytic +0.57
        hydrophobic fraction           p = 1e-06   safe 0.50,  hemolytic 0.58
        charge density                 p = 7e-05   safe 0.31,  hemolytic 0.25
        Leu fraction                   p = 0.002   safe 0.14,  hemolytic 0.19
        Lys fraction                   p = 0.010   safe 0.23,  hemolytic 0.17

    Hydrophobic moment (p = 0.23), net charge (p = 0.25), Trp (p = 0.30), Arg
    (p = 0.10) and Pro (p = 0.74) did not separate them, so none of them appears
    here however often the folklore invokes them. Net charge is represented only
    through charge density, which does separate: +5 on 16 residues is safe, +5 on
    26 is not.

    The Arg-share term is the one exception to "only separators", and it comes from
    a different cut: among cationic peptides (K+R >= 3, n = 908), the safety window
    peaks when Arg is 0 to 50% of the cationic residues. All-Lys peptides had the
    worst median window and all-Arg the lowest HC50.

    This is a prior, not a solution. Applying the best of these filters to the
    potent subset moved P(HC50 >= 128 µM) only from 0.25 to 0.31, which is why the
    ranked list is also spread across clusters rather than concentrated on whatever
    scores highest.
    """
    idx, lengths = features.encode(sequences)
    comp = features.composition(idx, lengths)
    charge = features.net_charge(sequences)
    lengths_f = lengths.astype(np.float64)

    eisenberg = comp @ features.EIS_V
    gravy = comp @ features.KD_V
    hyd = sum(comp[:, features.AA_INDEX[a]] for a in HYDROPHOBIC)
    aro = sum(comp[:, features.AA_INDEX[a]] for a in AROMATIC)
    leu = comp[:, features.AA_INDEX["L"]]
    lys = comp[:, features.AA_INDEX["K"]]
    arg = comp[:, features.AA_INDEX["R"]]
    arg_share = arg / np.maximum(lys + arg, 1e-9)
    charge_density = charge / lengths_f

    terms = [
        (2.0, _band(eisenberg, -0.40, 0.10, 0.25)),
        (2.0, _band(gravy, -1.20, 0.30, 0.50)),
        (1.5, _band(hyd, 0.33, 0.50, 0.10)),
        (1.5, _band(charge_density, 0.28, 0.55, 0.10)),
        (1.0, _band(leu, 0.0, 0.15, 0.08)),
        (1.0, _band(lys, 0.18, 0.60, 0.08)),
        (0.8, _band(arg_share, 0.0, 0.50, 0.25)),
        (0.6, _band(aro, 0.0, 0.20, 0.10)),
        (1.2, _band(lengths_f, 12, 22, 4)),
    ]
    total = sum(w for w, _ in terms)
    return (sum(w * v for w, v in terms) / total).astype(np.float32)
