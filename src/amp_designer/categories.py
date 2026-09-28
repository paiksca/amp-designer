"""The five competition categories and the objective each one ranks by.

Phase 2 scores five categories against different slices of the 20-strain panel,
so the ranked list that wins one is not the list that wins another. Both official
baselines expose only `generate_broad_spectrum`; this submission ranks separately
for all five.

Every objective is expected value over the whole list, not best case, because 25
peptides are drawn uniformly from the 100 and the team score is their arithmetic
mean. Each term is in log2 µM unless noted, so a unit is a two-fold change.
"""

from __future__ import annotations

import numpy as np

HC50_CEILING_LOG2 = 7.0  # log2(128 µM), the reported ceiling
MIC_CEILING_LOG2 = 6.0   # log2(64 µM)

# The Phase-2 panel: 15 Gram-negative strains and 5 Gram-positive.
PANEL_COUNTS = {
    "mic:sp_ecoli": 5,        # ATCC 11775, AIC221, AIC222, BAA-3170, K-12 BW25113
    "mic:sp_paeru": 3,        # PAO1, PA14, BAA-3197
    "mic:sp_kpneu": 2,        # ATCC 13883, BAA-2342
    "mic:sp_abaum": 2,        # ATCC 19606, BAA-1605
    "mic:sp_saure": 2,        # ATCC 12600, BAA-1556
    "mic:sp_efaec": 2,        # E. faecalis 700802, E. faecium 700221
    "mic:sp_bsubt": 1,        # ATCC 23857
}
# E. cloacae and the two S. enterica strains have no species head of their own;
# their three panel slots fall back to the pooled Gram-negative model.
GRAM_NEG_FALLBACK = 3

MDR_COUNTS = {
    # The 8 multi-drug-resistant isolates, by the species head that covers each.
    "mic:sp_abaum": 1,   # BAA-1605
    "mic:sp_ecoli": 2,   # AIC222 (CRE), BAA-3170 (CRE)
    "mic:sp_kpneu": 1,   # BAA-2342 (EIRK)
    "mic:sp_paeru": 1,   # BAA-3197 (FBCRP)
    "mic:sp_saure": 1,   # BAA-1556 (MRSA)
    "mic:sp_efaec": 2,   # E. faecalis VRE, E. faecium VRE
}

GRAM_POS_HEADS = ["mic:sp_saure", "mic:sp_efaec", "mic:sp_bsubt"]
GRAM_NEG_HEADS = ["mic:sp_ecoli", "mic:sp_paeru", "mic:sp_kpneu", "mic:sp_abaum"]


def _weighted(scores: dict[str, np.ndarray], counts: dict[str, int],
              fallback: str | None = None, fallback_n: int = 0) -> np.ndarray:
    total = 0
    acc = None
    for head, n in counts.items():
        if head not in scores:
            continue
        acc = scores[head] * n if acc is None else acc + scores[head] * n
        total += n
    if fallback and fallback_n and fallback in scores:
        acc = scores[fallback] * fallback_n if acc is None else acc + scores[fallback] * fallback_n
        total += fallback_n
    return acc / max(total, 1)


def _worst(scores: dict[str, np.ndarray], heads: list[str]) -> np.ndarray:
    cols = [scores[h] for h in heads if h in scores]
    return np.vstack(cols).max(axis=0) if cols else None


def broad_spectrum(scores, panel, envelope, risk, amp) -> np.ndarray:
    """Success rate over all 20 strains, tie-broken by MIC90.

    The panel weighting is the real strain census, so Gram-negative potency
    carries three quarters of the score. The worst-species term stands in for
    MIC90: a peptide that clears E. coli but not P. aeruginosa has a low success
    rate however good its mean looks.
    """
    mean_mic = _weighted(scores, PANEL_COUNTS, "mic:gram_neg", GRAM_NEG_FALLBACK)
    worst = _worst(scores, GRAM_NEG_HEADS + GRAM_POS_HEADS)
    return (
        -1.40 * mean_mic
        - 0.70 * worst
        + 0.90 * panel["p_safe"]
        + 0.60 * amp
        + 1.10 * envelope
        - 1.10 * risk
    )


def gram_pos(scores, panel, envelope, risk, amp) -> np.ndarray:
    """Five Gram-positive strains: two S. aureus, two enterococci, B. subtilis."""
    mean_mic = _weighted(scores, {h: PANEL_COUNTS[h] for h in GRAM_POS_HEADS})
    worst = _worst(scores, GRAM_POS_HEADS)
    return (
        -1.80 * mean_mic
        - 0.60 * worst
        + 0.80 * panel["p_safe"]
        + 0.50 * amp
        + 0.90 * envelope
        - 1.10 * risk
    )


def gram_neg(scores, panel, envelope, risk, amp) -> np.ndarray:
    """Fifteen Gram-negative strains, where the outer membrane is the barrier."""
    mean_mic = _weighted(
        scores, {h: PANEL_COUNTS[h] for h in GRAM_NEG_HEADS},
        "mic:gram_neg", GRAM_NEG_FALLBACK,
    )
    worst = _worst(scores, GRAM_NEG_HEADS)
    return (
        -1.80 * mean_mic
        - 0.70 * worst
        + 0.80 * panel["p_safe"]
        + 0.50 * amp
        + 0.90 * envelope
        - 1.10 * risk
    )


def mdr(scores, panel, envelope, risk, amp) -> np.ndarray:
    """The eight MDR ESKAPE isolates.

    Resistance to conventional antibiotics comes from efflux, beta-lactamases and
    target mutation, none of which defends against membrane disruption, so the
    usable signal is species-level potency over the ESKAPE set with the hardest
    Gram-negatives weighted as they appear on the panel.
    """
    mean_mic = _weighted(scores, MDR_COUNTS)
    worst = _worst(scores, list(MDR_COUNTS))
    return (
        -1.70 * mean_mic
        - 0.80 * worst
        + 0.80 * panel["p_safe"]
        + 0.50 * amp
        + 1.00 * envelope
        - 1.20 * risk
    )


def therapeutic(scores, panel, envelope, risk, amp) -> np.ndarray:
    """Optimal Selectivity: the safety window HC50 / MIC50.

    HC50 is reported only up to 128 µM, so once a peptide is confidently past that
    ceiling a further drop in hemolysis earns nothing, while MIC still spans 0.5 to
    64 µM. The window is therefore won on potency subject to clearing the ceiling,
    not by minimising hemolysis. The qualifying rule reinforces it: a peptide needs
    MIC <= 16 µM on at least one strain to enter the category at all.

    `p_safe` enters through a saturating term rather than linearly, which pays for
    reaching high confidence of clearing 128 µM and stops paying past it. The
    potency term then does the rest of the work.
    """
    mean_mic = _weighted(scores, PANEL_COUNTS, "mic:gram_neg", GRAM_NEG_FALLBACK)
    best = np.vstack(
        [scores[h] for h in GRAM_NEG_HEADS + GRAM_POS_HEADS if h in scores]
    ).min(axis=0)
    clears_ceiling = np.sqrt(np.clip(panel["p_safe"], 0.0, 1.0))
    return (
        -1.10 * mean_mic
        - 0.60 * np.clip(best, np.log2(0.5), None)   # the qualifying strain
        + 2.60 * clears_ceiling
        + 1.90 * envelope
        + 0.30 * amp
        - 1.30 * risk
    )


OBJECTIVES = {
    "generate_broad_spectrum": broad_spectrum,
    "generate_gram_pos": gram_pos,
    "generate_gram_neg": gram_neg,
    "generate_mdr": mdr,
    "generate_therapeutic": therapeutic,
}
# The challenge template's own single entry point, kept so either validator passes.
DEFAULT = "generate_broad_spectrum"


def objective_for(entry_point: str):
    return OBJECTIVES.get(entry_point, OBJECTIVES[DEFAULT])


# --------------------------------------------------------------- APEX re-ranking

def rank_pct(values: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1], ties broken by index so the result is stable."""
    order = np.lexsort((np.arange(len(values)), values))
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(len(values)) / max(len(values) - 1, 1)
    return out


def apex_broad(a) -> np.ndarray:
    return -(0.65 * a["apex_log2_gram_neg"] + 0.35 * a["apex_log2_gram_pos"]) - 0.4 * a["apex_log2_worst"]


def apex_gram_pos(a) -> np.ndarray:
    return -a["apex_log2_gram_pos"]


def apex_gram_neg(a) -> np.ndarray:
    return -a["apex_log2_gram_neg"]


def apex_mdr(a) -> np.ndarray:
    return -a["apex_log2_mdr"]


def apex_therapeutic(a) -> np.ndarray:
    # The window needs one strain under 16 µM to qualify, then depends on MIC50.
    return -(0.6 * a["apex_log2_median"] + 0.4 * a["apex_log2_best"])


APEX_OBJECTIVES = {
    "generate_broad_spectrum": apex_broad,
    "generate_gram_pos": apex_gram_pos,
    "generate_gram_neg": apex_gram_neg,
    "generate_mdr": apex_mdr,
    "generate_therapeutic": apex_therapeutic,
}

# APEX and this repository's MIC ensemble are about equally accurate on prospective
# peptides (AUROC 0.80 each, honestly measured) and agree only at Spearman 0.69, so
# the blend is close to even, tilted to the in-house models because they cover all
# 20 panel strains rather than 11 and carry a hemolysis head APEX does not have.
APEX_WEIGHT = 0.45


def blend(model_quality: np.ndarray, apex_quality: np.ndarray,
          weight: float = APEX_WEIGHT) -> np.ndarray:
    """Rank-blend two scores whose raw scales are not comparable."""
    return (
        (1.0 - weight) * rank_pct(model_quality) + weight * rank_pct(apex_quality)
    ).astype(np.float32)
