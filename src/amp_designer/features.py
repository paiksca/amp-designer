"""Physicochemical descriptors for short linear peptides.

Every function is vectorised over a list of sequences and returns float32 arrays.
Scales are the published ones, cited at each table, so the numbers are comparable
with the AMP literature.
"""

from __future__ import annotations

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Kyte & Doolittle 1982, J Mol Biol 157:105.
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Eisenberg consensus scale, 1984, Ann Rev Biochem 53:595. Used for the
# hydrophobic moment so that muH is on the scale the AMP literature reports.
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08,
}

# Fauchere & Pliska 1983 octanol-water pi values.
FP = {
    "A": 0.31, "R": -1.01, "N": -0.60, "D": -0.77, "C": 1.54, "Q": -0.22,
    "E": -0.64, "G": 0.00, "H": 0.13, "I": 1.80, "L": 1.70, "K": -0.99,
    "M": 1.23, "F": 1.79, "P": 0.72, "S": -0.04, "T": 0.26, "W": 2.25,
    "Y": 0.96, "V": 1.22,
}

# Boman 1996 protein-binding potential (kcal/mol), sign flipped as Boman reports it.
BOMAN = {
    "A": -1.81, "R": 14.92, "N": 6.64, "D": 8.72, "C": -1.28, "Q": 5.54,
    "E": 6.81, "G": -0.94, "H": 4.66, "I": -4.92, "L": -4.92, "K": 5.55,
    "M": -2.35, "F": -2.98, "P": 0.00, "S": 3.40, "T": 2.57, "W": -2.33,
    "Y": -0.14, "V": -4.04,
}

# Pace & Scholtz 1998 helix propensity, kcal/mol, lower means more helical.
HELIX = {
    "A": 0.0, "L": 0.21, "R": 0.21, "M": 0.24, "K": 0.26, "Q": 0.39, "E": 0.40,
    "I": 0.41, "W": 0.49, "S": 0.50, "Y": 0.53, "F": 0.54, "H": 0.61, "V": 0.61,
    "N": 0.65, "T": 0.66, "C": 0.68, "D": 0.69, "G": 1.0, "P": 3.16,
}

# Side-chain pKa, Nozaki & Tanford / Thurlkill 2006 values.
PKA_SIDE = {"D": 3.65, "E": 4.25, "H": 6.00, "C": 8.18, "Y": 10.07, "K": 10.53, "R": 12.48}
PKA_NTERM = 8.0
PKA_CTERM = 3.65

POSITIVE = set("KR")
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")

# Guruprasad 1990 instability index dipeptide weights, the 20x20 DIWV table.
_DIWV_ROWS = """
A 1.0 44.94 7.49 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 20.26 1.0 1.0 1.0 1.0 1.0 1.0 1.0
C 1.0 1.0 20.26 1.0 1.0 33.6 1.0 1.0 1.0 20.26 33.6 1.0 20.26 -6.54 1.0 1.0 33.6 24.68 1.0 1.0
D 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 -7.49 1.0 1.0 1.0 1.0 1.0 -6.54 20.26 -14.03 1.0 1.0 1.0
E 1.0 44.94 20.26 33.6 1.0 1.0 -6.54 20.26 20.26 1.0 1.0 1.0 20.26 20.26 1.0 20.26 1.0 1.0 -14.03 1.0
F 1.0 1.0 13.34 1.0 1.0 1.0 1.0 1.0 -14.03 1.0 1.0 1.0 20.26 1.0 1.0 1.0 1.0 1.0 1.0 33.601
G -7.49 1.0 1.0 -6.54 1.0 13.34 1.0 -7.49 -7.49 1.0 1.0 -7.49 1.0 1.0 1.0 1.0 -7.49 1.0 13.34 -7.49
H 1.0 1.0 1.0 1.0 -9.37 -9.37 1.0 44.94 24.68 1.0 1.0 24.68 -1.88 1.0 1.0 1.0 -6.54 1.0 -1.88 44.94
I 1.0 1.0 1.0 44.94 1.0 1.0 13.34 1.0 -7.49 20.26 1.0 1.0 -1.88 1.0 1.0 1.0 1.0 -7.49 1.0 1.0
K 1.0 1.0 1.0 1.0 1.0 -7.49 1.0 -7.49 1.0 -7.49 33.6 1.0 -6.54 24.68 33.6 1.0 1.0 -7.49 1.0 1.0
L 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 -7.49 1.0 1.0 1.0 20.26 33.6 20.26 1.0 1.0 1.0 24.68 1.0
M 13.34 1.0 1.0 1.0 1.0 1.0 58.28 1.0 1.0 1.0 -1.88 1.0 44.94 -6.54 -6.54 44.94 -1.88 1.0 1.0 24.68
N 1.0 -1.88 1.0 1.0 -14.03 -14.03 1.0 44.94 24.68 1.0 1.0 1.0 -1.88 -6.54 1.0 1.0 -7.49 -1.88 -9.37 1.0
P 20.26 -6.54 -6.54 18.38 20.26 1.0 1.0 1.0 1.0 1.0 -6.54 1.0 20.26 20.26 -6.54 20.26 1.0 20.26 -1.88 1.0
Q 1.0 -6.54 20.26 20.26 -6.54 1.0 1.0 1.0 1.0 1.0 1.0 1.0 20.26 20.26 1.0 44.94 1.0 -6.54 1.0 -6.54
R 1.0 1.0 1.0 1.0 1.0 -7.49 20.26 1.0 1.0 1.0 1.0 13.34 20.26 20.26 58.28 44.94 1.0 1.0 58.28 -6.54
S 1.0 33.6 1.0 20.26 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 44.94 20.26 20.26 20.26 1.0 1.0 1.0 1.0
T 1.0 1.0 1.0 20.26 13.34 -7.49 1.0 1.0 1.0 1.0 1.0 -14.03 1.0 -6.54 1.0 1.0 1.0 1.0 -14.03 1.0
V 1.0 1.0 -14.03 1.0 1.0 -7.49 1.0 1.0 -1.88 1.0 1.0 1.0 20.26 1.0 1.0 1.0 -7.49 1.0 1.0 -6.54
W -14.03 1.0 1.0 1.0 1.0 -9.37 24.68 1.0 1.0 13.34 24.68 13.34 1.0 1.0 1.0 1.0 -14.03 -7.49 1.0 1.0
Y 24.68 1.0 24.68 -6.54 1.0 -7.49 13.34 1.0 1.0 1.0 44.94 1.0 13.34 1.0 -15.91 1.0 -7.49 1.0 -9.37 13.34
"""
_DIWV_COLS = list("ACDEFGHIKLMNPQRSTVWY")


def _build_diwv() -> np.ndarray:
    table = np.ones((20, 20), dtype=np.float64)
    for line in _DIWV_ROWS.strip().splitlines():
        parts = line.split()
        i = AA_INDEX[parts[0]]
        for j, v in enumerate(parts[1:]):
            table[i, AA_INDEX[_DIWV_COLS[j]]] = float(v)
    return table


DIWV = _build_diwv()


def encode(sequences: list[str], max_len: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return an (n, max_len) int8 index matrix with -1 padding, and the lengths."""
    n = len(sequences)
    idx = np.full((n, max_len), -1, dtype=np.int8)
    lengths = np.empty(n, dtype=np.int32)
    for i, s in enumerate(sequences):
        lengths[i] = len(s)
        for j, c in enumerate(s):
            idx[i, j] = AA_INDEX[c]
    return idx, lengths


def _scale_vector(scale: dict[str, float]) -> np.ndarray:
    return np.array([scale[a] for a in AA], dtype=np.float64)


KD_V = _scale_vector(KD)
EIS_V = _scale_vector(EISENBERG)
FP_V = _scale_vector(FP)
BOMAN_V = _scale_vector(BOMAN)
HELIX_V = _scale_vector(HELIX)


def composition(idx: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Fractional amino-acid composition, shape (n, 20)."""
    n = idx.shape[0]
    counts = np.zeros((n, 20), dtype=np.float64)
    valid = idx >= 0
    rows = np.repeat(np.arange(n), valid.sum(axis=1))
    np.add.at(counts, (rows, idx[valid]), 1.0)
    return counts / lengths[:, None]


def net_charge(sequences: list[str], ph: float = 7.4) -> np.ndarray:
    """Net charge from Henderson-Hasselbalch over side chains plus free termini.

    Our peptides are linear with free termini, so both termini are titratable.
    """
    out = np.empty(len(sequences), dtype=np.float64)
    for i, s in enumerate(sequences):
        q = 1.0 / (1.0 + 10.0 ** (ph - PKA_NTERM))
        q -= 1.0 / (1.0 + 10.0 ** (PKA_CTERM - ph))
        for c in s:
            pka = PKA_SIDE.get(c)
            if pka is None:
                continue
            if c in "KRH":
                q += 1.0 / (1.0 + 10.0 ** (ph - pka))
            else:
                q -= 1.0 / (1.0 + 10.0 ** (pka - ph))
        out[i] = q
    return out


def hydrophobic_moment(
    idx: np.ndarray, lengths: np.ndarray, window: int = 11, angle: float = 100.0
) -> np.ndarray:
    """Eisenberg's muH: the largest moment over any window of `window` residues.

    For sequences shorter than the window the whole sequence is used. The 100 deg
    step is the alpha-helical periodicity; 180 deg would give the beta-strand moment.
    """
    n, max_len = idx.shape
    rad = np.deg2rad(angle) * np.arange(max_len)
    cosv, sinv = np.cos(rad), np.sin(rad)

    h = np.where(idx >= 0, EIS_V[np.clip(idx, 0, 19)], 0.0)
    hc = h * cosv[None, :]
    hs = h * sinv[None, :]
    # Prefix sums let every window be read in constant time.
    pc = np.concatenate([np.zeros((n, 1)), np.cumsum(hc, axis=1)], axis=1)
    ps = np.concatenate([np.zeros((n, 1)), np.cumsum(hs, axis=1)], axis=1)

    best = np.zeros(n, dtype=np.float64)
    for start in range(max_len):
        w = np.minimum(window, lengths - start)
        active = w > 0
        if not active.any():
            break
        end = start + np.clip(w, 1, None)
        rows = np.arange(n)
        # A window rotated by start*angle has the same magnitude, so no correction.
        c = pc[rows, end] - pc[rows, start]
        s = ps[rows, end] - ps[rows, start]
        mu = np.sqrt(c * c + s * s) / np.clip(w, 1, None)
        best = np.where(active, np.maximum(best, mu), best)
    return best


def max_run(sequences: list[str], alphabet: set[str]) -> np.ndarray:
    """Longest run of consecutive residues drawn from `alphabet`."""
    out = np.zeros(len(sequences), dtype=np.float64)
    for i, s in enumerate(sequences):
        run = best = 0
        for c in s:
            run = run + 1 if c in alphabet else 0
            if run > best:
                best = run
        out[i] = best
    return out


def max_identical_run(sequences: list[str]) -> np.ndarray:
    out = np.zeros(len(sequences), dtype=np.float64)
    for i, s in enumerate(sequences):
        run = best = 1
        for a, b in zip(s, s[1:]):
            run = run + 1 if a == b else 1
            if run > best:
                best = run
        out[i] = best if s else 0
    return out


def instability_index(sequences: list[str]) -> np.ndarray:
    out = np.empty(len(sequences), dtype=np.float64)
    for i, s in enumerate(sequences):
        if len(s) < 2:
            out[i] = 0.0
            continue
        total = 0.0
        for a, b in zip(s, s[1:]):
            total += DIWV[AA_INDEX[a], AA_INDEX[b]]
        out[i] = 10.0 / len(s) * total
    return out


def aliphatic_index(comp: np.ndarray) -> np.ndarray:
    a = comp[:, AA_INDEX["A"]]
    v = comp[:, AA_INDEX["V"]]
    il = comp[:, AA_INDEX["I"]] + comp[:, AA_INDEX["L"]]
    return 100.0 * (a + 2.9 * v + 3.9 * il)


def isoelectric_point(sequences: list[str]) -> np.ndarray:
    lo, hi = np.zeros(len(sequences)), np.full(len(sequences), 14.0)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        # net_charge is cheap enough that a bisection over 40 steps stays fast.
        q = np.array([_charge_at(s, p) for s, p in zip(sequences, mid)])
        hi = np.where(q > 0, hi, mid)
        lo = np.where(q > 0, mid, lo)
    return (lo + hi) / 2.0


def _charge_at(seq: str, ph: float) -> float:
    q = 1.0 / (1.0 + 10.0 ** (ph - PKA_NTERM)) - 1.0 / (1.0 + 10.0 ** (PKA_CTERM - ph))
    for c in seq:
        pka = PKA_SIDE.get(c)
        if pka is None:
            continue
        if c in "KRH":
            q += 1.0 / (1.0 + 10.0 ** (ph - pka))
        else:
            q -= 1.0 / (1.0 + 10.0 ** (pka - ph))
    return q


FEATURE_NAMES = (
    [f"frac_{a}" for a in AA]
    + [
        "length",
        "log_length",
        "charge",
        "charge_density",
        "gravy",
        "eisenberg_mean",
        "fp_mean",
        "boman",
        "helix_penalty",
        "mu_h",
        "mu_h_beta",
        "amphipathicity",
        "frac_positive",
        "frac_negative",
        "frac_hydrophobic",
        "frac_aromatic",
        "frac_small",
        "max_hydrophobic_run",
        "max_identical_run",
        "max_positive_run",
        "instability",
        "aliphatic",
        "pi",
        "kr_ratio",
        "n_cys",
        "n_met",
        "n_pro",
        "n_gly",
        "max_pos_in_5",
        "hydrophobic_face_frac",
    ]
)


def descriptors(sequences: list[str]) -> np.ndarray:
    """Return the (n, len(FEATURE_NAMES)) descriptor matrix."""
    idx, lengths = encode(sequences)
    comp = composition(idx, lengths)
    lengths_f = lengths.astype(np.float64)

    gravy = comp @ KD_V
    eis = comp @ EIS_V
    fp = comp @ FP_V
    boman = comp @ BOMAN_V
    helix = comp @ HELIX_V
    mu = hydrophobic_moment(idx, lengths, window=11, angle=100.0)
    mu_beta = hydrophobic_moment(idx, lengths, window=7, angle=180.0)
    charge = net_charge(sequences)

    pos = comp[:, AA_INDEX["K"]] + comp[:, AA_INDEX["R"]]
    neg = comp[:, AA_INDEX["D"]] + comp[:, AA_INDEX["E"]]
    hyd = sum(comp[:, AA_INDEX[a]] for a in HYDROPHOBIC)
    aro = sum(comp[:, AA_INDEX[a]] for a in AROMATIC)
    small = sum(comp[:, AA_INDEX[a]] for a in "AGST")

    k = comp[:, AA_INDEX["K"]]
    r = comp[:, AA_INDEX["R"]]

    cols = [
        comp,
        lengths_f[:, None],
        np.log(lengths_f)[:, None],
        charge[:, None],
        (charge / lengths_f)[:, None],
        gravy[:, None],
        eis[:, None],
        fp[:, None],
        boman[:, None],
        helix[:, None],
        mu[:, None],
        mu_beta[:, None],
        (mu * np.maximum(eis, 0))[:, None],
        pos[:, None],
        neg[:, None],
        hyd[:, None],
        aro[:, None],
        small[:, None],
        max_run(sequences, HYDROPHOBIC)[:, None],
        max_identical_run(sequences)[:, None],
        max_run(sequences, POSITIVE)[:, None],
        instability_index(sequences)[:, None],
        aliphatic_index(comp)[:, None],
        isoelectric_point(sequences)[:, None],
        (k / np.maximum(k + r, 1e-9))[:, None],
        (comp[:, AA_INDEX["C"]] * lengths_f)[:, None],
        (comp[:, AA_INDEX["M"]] * lengths_f)[:, None],
        (comp[:, AA_INDEX["P"]] * lengths_f)[:, None],
        (comp[:, AA_INDEX["G"]] * lengths_f)[:, None],
        max_positive_in_window(sequences, 5)[:, None],
        hydrophobic_face_fraction(sequences)[:, None],
    ]
    return np.hstack(cols).astype(np.float32)


def max_positive_in_window(sequences: list[str], window: int = 5) -> np.ndarray:
    out = np.zeros(len(sequences), dtype=np.float64)
    for i, s in enumerate(sequences):
        best = 0
        flags = [1 if c in POSITIVE else 0 for c in s]
        run = sum(flags[:window])
        best = run
        for j in range(window, len(flags)):
            run += flags[j] - flags[j - window]
            if run > best:
                best = run
        out[i] = best
    return out


def hydrophobic_face_fraction(sequences: list[str], angle: float = 100.0) -> np.ndarray:
    """Fraction of residues that fall on the hydrophobic half of the helical wheel.

    A clean amphipathic helix puts most of its hydrophobic residues in one arc.
    The value is the share of residues lying within 90 deg of the moment vector
    that are hydrophobic, which separates amphipathic designs from evenly mixed ones.
    """
    out = np.zeros(len(sequences), dtype=np.float64)
    rad = np.deg2rad(angle)
    for i, s in enumerate(sequences):
        if not s:
            continue
        h = np.array([EISENBERG[c] for c in s])
        ang = rad * np.arange(len(s))
        vx = float((h * np.cos(ang)).sum())
        vy = float((h * np.sin(ang)).sum())
        if vx == 0.0 and vy == 0.0:
            continue
        theta = np.arctan2(vy, vx)
        delta = np.abs(np.angle(np.exp(1j * (ang - theta))))
        on_face = delta <= np.pi / 2
        if on_face.sum() == 0:
            continue
        hydro = np.array([1.0 if c in HYDROPHOBIC else 0.0 for c in s])
        out[i] = hydro[on_face].mean()
    return out


def kmer_counts(sequences: list[str], k: int = 2) -> np.ndarray:
    """Normalised k-mer frequency matrix, 20**k columns."""
    dim = 20**k
    out = np.zeros((len(sequences), dim), dtype=np.float32)
    for i, s in enumerate(sequences):
        if len(s) < k:
            continue
        total = 0
        for j in range(len(s) - k + 1):
            code = 0
            for c in s[j : j + k]:
                code = code * 20 + AA_INDEX[c]
            out[i, code] += 1.0
            total += 1
        if total:
            out[i] /= total
    return out
