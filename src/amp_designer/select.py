"""Library selection: turn a large candidate pool into the 50,000-sequence library.

Phase 1 scores four families at once and they pull against each other. Novelty and
diversity want sequences far from known AMPs and from each other; distributional
similarity and property conformity want the library to look exactly like known
AMPs; surrogate activity wants the potent corner of that distribution.

The selection resolves this by fixing the *distribution* and spending the freedom
inside it:

  - Quotas over a (length, charge, hydrophobic moment) grid are copied from the
    reference AMP set with no reweighting. Those three axes are the ones Phase 1
    names for property conformity, and copying rather than tilting them is what
    keeps KL divergence, precision and recall from paying for potency.
  - Inside every cell, candidates are taken one per sequence cluster in descending
    quality, so coverage is spent before any cluster contributes a second member.
    Potency, safety and synthesizability enter here, where they move which
    sequences are picked without moving the marginals.
  - A small floor keeps every cell the reference occupies from emptying, because
    `KLDivergence` is measured as KL(reference || generated) and so punishes a
    missing mode far harder than an over-represented one.
"""

from __future__ import annotations

import numpy as np

from . import features, lm

# Eisenberg hydrophobic-moment bin edges, chosen at the reference set's quartiles.
MU_EDGES = [0.22, 0.34, 0.47]


def minhash_signature(
    sequences: list[str], k: int = 3, n_hash: int = 4, n_buckets: int = 1 << 22
) -> np.ndarray:
    """Banded MinHash over k-mers: a locality-sensitive cluster key.

    A digest computed from character codes, not Python's `hash`, because `hash`
    on strings is salted per process and the entry point has to be reproducible
    across separate runs.
    """
    primes = [
        (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9),
        (0x94D049BB133111EB, 0xD6E8FEB86659FD93),
        (0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9),
        (0x27D4EB2F165667C5, 0x85EBCA77C2B2AE63),
    ]
    mask = 0xFFFFFFFFFFFFFFFF
    out = np.empty(len(sequences), dtype=np.int64)
    for i, s in enumerate(sequences):
        grams = [s[j : j + k] for j in range(len(s) - k + 1)] if len(s) >= k else [s]
        codes = []
        for km in grams:
            c = 0
            for ch in km:
                c = (c * 31 + ord(ch)) & mask
            codes.append(c)
        acc = 1469598103934665603
        for a, b in primes[:n_hash]:
            mn = min(((c * a) ^ b) & mask for c in codes)
            acc = ((acc ^ mn) * 1099511628211) & mask
        out[i] = acc % n_buckets
    return out


def property_cells(sequences: list[str]) -> list[tuple[int, int, int]]:
    """Bin sequences on the three axes Phase 1 names for property conformity."""
    charge = features.net_charge(sequences)
    idx, lengths = features.encode(sequences)
    mu = features.hydrophobic_moment(idx, lengths)
    return [
        (
            lm.bin_of(int(l), lm.LEN_EDGES),
            lm.bin_of(float(c), lm.CHARGE_EDGES),
            lm.bin_of(float(m), MU_EDGES),
        )
        for l, c, m in zip(lengths, charge, mu)
    ]


def target_quotas(
    reference: list[str], total: int, floor: float = 0.0004
) -> dict[tuple[int, int, int], int]:
    """Counts per property cell, copied from the reference set's own histogram."""
    cells = property_cells(reference)
    grid: dict[tuple[int, int, int], float] = {}
    for cell in cells:
        grid[cell] = grid.get(cell, 0.0) + 1.0

    mass = sum(grid.values())
    share = {k: max(v / mass, floor) for k, v in grid.items()}
    norm = sum(share.values())

    quotas = {k: int(round(total * v / norm)) for k, v in share.items()}
    drift = total - sum(quotas.values())
    order = sorted(quotas, key=lambda k: (-quotas[k], k))
    i = 0
    while drift != 0 and order:
        k = order[i % len(order)]
        step = 1 if drift > 0 else -1
        if quotas[k] + step >= 0:
            quotas[k] += step
            drift -= step
        i += 1
    return quotas


def select(
    sequences: list[str],
    quality: np.ndarray,
    quotas: dict[tuple[int, int, int], int],
    cluster: np.ndarray,
    total: int,
    forced: list[int] | None = None,
) -> np.ndarray:
    """Pick `total` indices, filling each quota cell cluster-first by quality.

    Deterministic by construction: every ordering is an explicit sort with the
    candidate index as the final tiebreak, so nothing depends on set or dict
    iteration order over strings.
    """
    cell_of = property_cells(sequences)
    order = np.lexsort((np.arange(len(sequences)), -quality))

    by_cell: dict[tuple[int, int, int], list[int]] = {}
    for i in order:
        by_cell.setdefault(cell_of[i], []).append(int(i))

    taken = np.zeros(len(sequences), dtype=bool)
    chosen: list[int] = []
    used: dict[int, int] = {}

    def take(i: int) -> None:
        taken[i] = True
        chosen.append(i)
        c = int(cluster[i])
        used[c] = used.get(c, 0) + 1

    for i in forced or []:
        if not taken[i]:
            take(int(i))

    max_rounds = 64
    for cell in sorted(quotas):
        want = quotas[cell]
        pool = by_cell.get(cell)
        if want <= 0 or not pool:
            continue
        got = sum(1 for i in chosen if cell_of[i] == cell)
        local: dict[int, int] = {}
        for r in range(1, max_rounds + 1):
            if got >= want:
                break
            for i in pool:
                if got >= want:
                    break
                if taken[i]:
                    continue
                c = int(cluster[i])
                if local.get(c, 0) >= r:
                    continue
                local[c] = local.get(c, 0) + 1
                take(i)
                got += 1

    # Cells the candidate pool could not fill leave a shortfall; make it up
    # globally, still spreading over clusters before repeating any.
    if len(chosen) < total:
        for r in range(1, max_rounds + 1):
            if len(chosen) >= total:
                break
            for i in order:
                i = int(i)
                if len(chosen) >= total:
                    break
                if taken[i] or used.get(int(cluster[i]), 0) >= r:
                    continue
                take(i)

    if len(chosen) > total:
        keep = set(int(i) for i in (forced or []))
        trimmed = [i for i in chosen if i in keep]
        for i in chosen:
            if len(trimmed) >= total:
                break
            if i not in keep:
                trimmed.append(i)
        chosen = trimmed
    return np.array(chosen, dtype=np.int64)


def composition_gap(sequences: list[str], reference: list[str]) -> float:
    """L2 distance between mean amino-acid compositions.

    Reported rather than optimised directly. With a small protein language model
    most of the Frechet distance is carried by composition, so this is the cheapest
    early warning that the library has drifted off the reference distribution.
    """
    a_idx, a_len = features.encode(sequences)
    b_idx, b_len = features.encode(reference)
    a = features.composition(a_idx, a_len).mean(axis=0)
    b = features.composition(b_idx, b_len).mean(axis=0)
    return float(np.linalg.norm(a - b))
