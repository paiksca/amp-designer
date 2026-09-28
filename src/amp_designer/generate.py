"""Entry point: write `generate/library.fasta` and `generate/top.fasta`.

Pipeline
--------
1. Sample a large candidate pool from the property-conditioned peptide LM. Control
   vectors are drawn from a table of combinations observed in the training corpus,
   reweighted toward predicted potency and predicted hemolytic safety, so the pool
   sits in the potent corner of the known-AMP distribution, not outside it.
2. Drop anything that breaks a competition rule, duplicates another candidate, or
   matches the reference antibacterial set exactly.
3. Score every survivor with the MIC, hemolysis, and AMP-classifier ensembles.
4. Pick the ranked top-100 first, under the strict synthesis rules, an exact
   Levenshtein novelty check, and a cap per sequence cluster.
5. Fill the 50,000-sequence library around them, matching the reference set's
   (length, charge) distribution cell by cell and taking one candidate per cluster
   per pass so coverage is spent before any cluster repeats.

Both files are byte-reproducible: every ordering that reaches the output is fixed
by an explicit sort, never by iteration over a set or dict, whose order over
strings changes with the per-process hash seed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rapidfuzz import process
from rapidfuzz.distance import Indel

from . import apex, categories, esm, features, lm, novelty, scoring, select, synthesis

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

DEFAULT_LIBRARY = 50_000
DEFAULT_TOP_K = 100
DEFAULT_SEED = 42
DEFAULT_LENGTH = 50

CANDIDATE_MULTIPLIER = 6
MAX_CANDIDATES = 300_000
SAMPLE_BATCH = 8_192
# The validator rejects a ranked sequence above Levenshtein.ratio 0.80 against the
# reference set, while the proposal states the same rule as MMseqs2 alignment
# identity. The two are different measures, so the ranked lists are held to 0.75
# and the margin covers the gap between them.
NOVELTY_THRESHOLD = 0.75
TOP_CLUSTER_CAP = 3
# No two ranked sequences may exceed this Levenshtein ratio to each other. The
# MinHash cluster key only catches near-identical sequences, and the point of the
# cap is coarser than that: 25 peptides are drawn from the 100 and averaged, so a
# list built from one scaffold risks every draw failing for the same reason. The
# models are far better at telling active from inactive than at ranking among the
# active, which makes spread across scaffolds worth more than the ranking it costs.
TOP_MAX_PAIRWISE = 0.65
# No ranked sequence may share an exact substring this long with the reference set.
LONG_SUBSTRING = 10

# The heads that run over every candidate. The per-species MIC models cost four
# times as much and only change the ranked list, so they run on the shortlist.
COARSE_TARGETS = ("amp:clf", "mic:gram_neg", "hem:safe", "surrogate:mbc")
# Shortlist size as a multiple of the library, spread across cells by quota share.
SHORTLIST_SIZE = 90_000
# How many leading candidates per category go to APEX. It costs about a second per
# hundred sequences, so the pool is kept to the region where the ranked list is
# actually decided.
APEX_POOL_PER_CATEGORY = 2_000


def _resolve(*parts: str) -> Path:
    """Find a shipped file whether the process runs from the repo root or elsewhere."""
    rel = Path(*parts)
    for base in (Path.cwd(), REPO_ROOT, PACKAGE_ROOT):
        p = base / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"{rel} not found (cwd: {os.getcwd()})")


def _set_determinism() -> None:
    # Two runs on one machine must agree byte for byte. A fixed thread count keeps
    # every reduction in the same order. The seeds are set per call site.
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_fasta(path: Path) -> list[str]:
    seqs, parts, started = [], [], False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if started:
                seqs.append("".join(parts))
            parts, started = [], True
        else:
            parts.append(line.upper())
    if started:
        seqs.append("".join(parts))
    return seqs


def load_model() -> lm.PeptideLM:
    blob = torch.load(_resolve("checkpoint", "peptide_lm.pt"), map_location="cpu",
                      weights_only=False)
    model = lm.PeptideLM(lm.Config(**blob["config"]))
    model.load_state_dict({k: v.float() for k, v in blob["state_dict"].items()})
    model.eval()
    return model


def load_controls() -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(_resolve("checkpoint", "controls.npz"))
    return blob["combos"], blob["weights"]


def sample_pool(
    model: lm.PeptideLM,
    combos: np.ndarray,
    weights: np.ndarray,
    n_target: int,
    max_len: int,
    seed: int,
    reference: frozenset[str],
) -> list[str]:
    """Sample until `n_target` distinct valid candidates exist, then stop."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    probs = weights / weights.sum()

    seen: set[str] = set()
    pool: list[str] = []
    # Three temperatures widen the pool: the cool pass supplies high-likelihood
    # sequences, the warm pass supplies the diversity the coverage metric rewards.
    temperatures = (0.85, 1.0, 1.15)
    rounds = 0
    while len(pool) < n_target and rounds < 400:
        temp = temperatures[rounds % len(temperatures)]
        pick = rng.choice(len(combos), size=SAMPLE_BATCH, p=probs)
        controls = combos[pick].astype(np.int64)
        batch = lm.sample(
            model, controls, gen, temperature=temp, top_p=0.95,
            min_len=lm.MIN_LEN, max_len=max_len,
        )
        for s in batch:
            if not (lm.MIN_LEN <= len(s) <= max_len):
                continue
            if s in seen or s in reference:
                continue
            seen.add(s)
            pool.append(s)
        rounds += 1
    return pool


def coarse_quality(
    sequences: list[str], scores: dict[str, np.ndarray], reuse: np.ndarray
) -> np.ndarray:
    """Cheap within-cell ranking over the whole pool.

    `surrogate:mbc` is a head distilled from MBC-Attention, one of the three
    surrogates the proposal names for its activity family. It ranks the library
    within a cell only. The top-100 objectives leave it out, because it is
    demonstrably wrong in places and a wet-lab slot is too expensive to spend on
    a model that scores poly-glutamate at 1.8 µM.

    `reuse` is the share of a sequence's 6-mers that also occur in the reference
    set. Phase 1 scores novelty against known AMPs by normalized alignment
    bit-score over the whole library, and a long shared substring is what drives
    an alignment score, so penalizing k-mer reuse pushes the library away from the
    reference in sequence space without moving its composition.
    """
    risk = synthesis.risk_score(sequences)
    ok = synthesis.library_flags(sequences).astype(np.float32)
    return (
        1.20 * scores["amp:clf"]
        - 0.45 * scores["mic:gram_neg"]
        - 0.90 * scores["surrogate:mbc"]
        + 0.55 * scores["hem:safe"]
        + 0.40 * ok
        - 0.30 * risk
        - 0.90 * reuse
    ).astype(np.float32)


def library_quality(
    sequences: list[str],
    scores: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    reuse: np.ndarray,
) -> np.ndarray:
    """Rank within a quota cell: AMP-like, potent, non-hemolytic, novel, buildable."""
    risk = synthesis.risk_score(sequences)
    ok = synthesis.library_flags(sequences).astype(np.float32)
    return (
        1.20 * scores["amp:clf"]
        - 0.45 * panel["log2_mic50"]
        - 0.90 * scores["surrogate:mbc"]
        + 0.55 * panel["p_safe"]
        + 0.40 * ok
        - 0.30 * risk
        - 0.90 * reuse
    ).astype(np.float32)


def pick_top(
    sequences: list[str],
    quality: np.ndarray,
    cluster: np.ndarray,
    reference: list[str],
    top_k: int,
    long_kmers: frozenset[str] | None = None,
    safety: np.ndarray | None = None,
    cluster_cap: int = TOP_CLUSTER_CAP,
    max_pairwise: float = TOP_MAX_PAIRWISE,
) -> list[int]:
    """Choose the ranked list: strict rules, exact novelty, capped per cluster.

    The cap is a hedge, not a metric. The models here are right about which
    peptides are active far more often than they are right about how active, so a
    list of near-copies of one scaffold risks all 25 drawn peptides failing
    together. Several families fail independently.
    """
    strict = synthesis.strict_flags(sequences)
    if safety is not None:
        strict = strict & (safety >= categories.SAFETY_FLOOR)
    if long_kmers:
        strict = strict & ~novelty.shares_long_substring(
            sequences, long_kmers, k=LONG_SUBSTRING
        )
    order = np.lexsort((np.arange(len(sequences)), -quality))
    eligible = [int(i) for i in order if strict[i]]
    if len(eligible) < top_k * 4:
        # Too few survivors to fill the list, so fall back to ranking every candidate.
        eligible = [int(i) for i in order]

    chosen: list[int] = []
    per_cluster: dict[int, int] = {}
    # Novelty is checked in blocks, because the exact Levenshtein pass against
    # 39,448 references is the expensive step and most candidates never need it.
    block = 400
    cursor = 0
    while len(chosen) < top_k and cursor < len(eligible):
        window = eligible[cursor : cursor + block]
        cursor += block
        ratios = novelty.max_levenshtein_ratio(
            [sequences[i] for i in window], reference, threshold=NOVELTY_THRESHOLD
        )
        for i, r in zip(window, ratios):
            if len(chosen) >= top_k:
                break
            if r > NOVELTY_THRESHOLD:
                continue
            c = int(cluster[i])
            if per_cluster.get(c, 0) >= cluster_cap:
                continue
            if chosen and max_pairwise < 1.0:
                near = process.cdist(
                    [sequences[i]], [sequences[j] for j in chosen],
                    scorer=Indel.normalized_similarity, workers=-1,
                ).max()
                if near > max_pairwise:
                    continue
            per_cluster[c] = per_cluster.get(c, 0) + 1
            chosen.append(i)

    # Relax the spacing, because the validator rejects a short list.
    if len(chosen) < top_k and max_pairwise < 1.0:
        return pick_top(
            sequences, quality, cluster, reference, top_k, long_kmers=long_kmers,
            safety=safety, cluster_cap=cluster_cap + 2, max_pairwise=min(1.0, max_pairwise + 0.15),
        )
    return chosen


def _quantile_map(values: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Re-express `values` on the scale of `onto`, preserving their ordering."""
    order = np.lexsort((np.arange(len(values)), values))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return np.sort(onto)[ranks].astype(np.float32)


def shortlist_by_cell(
    sequences: list[str], quality: np.ndarray, keep_per_cell: dict, cap: int
) -> list[int]:
    """Keep the best candidates inside each property cell, never across cells.

    Pruning globally by quality would quietly re-shape the library, because the
    high-scoring end of the pool is more cationic and more amphipathic than the
    reference set. Pruning cell by cell leaves every marginal exactly where the
    quota grid put it.
    """
    cells = select.property_cells(sequences)
    order = np.lexsort((np.arange(len(sequences)), -quality))
    counts: dict[tuple, int] = {}
    keep: list[int] = []
    for i in order:
        i = int(i)
        cell = cells[i]
        budget = max(int(cap * keep_per_cell.get(cell, 0)), 40)
        if counts.get(cell, 0) >= budget:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        keep.append(i)
    return sorted(keep)


class _Build:
    """One full pipeline run, cached so `generate` and `score` agree.

    Scoring is staged. The cheap ensemble runs over every candidate and decides
    the library, where a within-cell ranking only needs to be roughly right. The
    full per-species ensemble runs over a shortlist and decides the top-100, where
    each entry commits a wet-lab slot.
    """

    def __init__(self, n_sequences: int, length: int, seed: int, top_k: int):
        _set_determinism()
        t0 = time.time()

        def step(message: str) -> None:
            print(f"[{time.time() - t0:6.0f}s] {message}", flush=True)

        max_len = int(min(max(length, lm.MIN_LEN), lm.MAX_LEN))
        reference = read_fasta(_resolve("data", "antibacterial.fasta"))
        reference_set = frozenset(reference)

        model = load_model()
        combos, weights = load_controls()
        n_target = min(
            MAX_CANDIDATES, max(n_sequences * CANDIDATE_MULTIPLIER, n_sequences + 1000)
        )
        step(f"sampling up to {n_target} candidates from the language model")
        pool = sample_pool(
            model, combos, weights, n_target, max_len, seed, reference_set
        )
        step(f"sampled {len(pool)} distinct candidates")
        if len(pool) < n_sequences:
            raise RuntimeError(
                f"only {len(pool)} distinct candidates for a library of {n_sequences}"
            )

        bundle = scoring.load(_resolve("checkpoint", "scorers.pkl.gz"))
        coarse = {k: v for k, v in bundle.items() if k in COARSE_TARGETS}
        step("scoring the pool with the coarse ensemble")
        raw = scoring.score(pool, coarse)

        ref_kmers = novelty.reference_kmers(reference, k=6)
        long_kmers = novelty.reference_kmers(reference, k=LONG_SUBSTRING)
        reuse = novelty.kmer_hit_fraction(pool, ref_kmers, k=6)
        cluster = select.minhash_signature(pool)
        lib_q = coarse_quality(pool, raw, reuse)

        quotas = select.target_quotas(reference, n_sequences)
        shares = {cell: q / max(n_sequences, 1) for cell, q in quotas.items()}
        short = shortlist_by_cell(pool, lib_q, shares, SHORTLIST_SIZE)
        step(f"shortlisted {len(short)} candidates across {len(quotas)} property cells")

        short_seqs = [pool[i] for i in short]
        full = scoring.score(short_seqs, bundle)
        step("scored the shortlist with the full per-species ensemble")
        panel = scoring.panel_summary(full)
        envelope = synthesis.envelope_score(short_seqs)
        risk = synthesis.risk_score(short_seqs)
        short_cluster = cluster[np.asarray(short, dtype=np.int64)]

        # Every category ranks the same shortlist by its own objective. The five
        # ranked lists are computed together so all of them can be forced into the
        # one shared library, which the validator requires of every top sequence.
        model_q = {
            name: fn(full, panel, envelope, risk, full["amp:clf"]).astype(np.float32)
            for name, fn in sorted(categories.OBJECTIVES.items())
        }

        # Stage 3: APEX-pathogen re-ranks the region where the list is decided.
        # Only candidates that already clear the strict synthesis rules and the
        # safety gate go in, so none of that budget is spent on sequences no
        # category would rank anyway.
        safe_enough = full["hem:safe"] >= categories.SAFETY_FLOOR
        eligible = np.flatnonzero(synthesis.strict_flags(short_seqs) & safe_enough)
        pool_idx: set[int] = set()
        for q in model_q.values():
            ranked = eligible[np.argsort(-q[eligible], kind="stable")]
            pool_idx.update(int(i) for i in ranked[:APEX_POOL_PER_CATEGORY])
        apex_idx = np.array(sorted(pool_idx), dtype=np.int64)

        apex_seqs = [short_seqs[i] for i in apex_idx]
        step(f"running APEX-pathogen on {len(apex_seqs)} leading candidates")
        apex_sum = apex.summary(
            apex.predict(apex_seqs, apex.load(_resolve("checkpoint", "apex")))
        )
        apex_cluster = short_cluster[apex_idx]

        # The descriptor hemolysis head ranks the pool fast. The ESM head is
        # more accurate (AUROC 0.786 against 0.755 on cluster-grouped splits) and
        # runs here, over the few thousand candidates that actually become the
        # ranked lists. The gate is re-applied on the better estimate.
        step(f"re-scoring hemolysis with ESM-2 on {len(apex_seqs)} candidates")
        encoder = esm.load(_resolve("checkpoint", "esm2_t12_35M"))
        hem_esm = esm.score(
            apex_seqs, bundle, encoder, scoring.featurize(apex_seqs)
        )
        apex_panel = {k: v[apex_idx] for k, v in panel.items()}
        apex_panel["p_safe"] = hem_esm["hem_esm:safe"]
        apex_panel["log2_hc50"] = np.minimum(
            hem_esm["hem_esm:log2_hc50"], np.log2(scoring.HC50_CEILING)
        )
        apex_panel["log2_safety_window"] = apex_panel["log2_hc50"] - np.maximum(
            apex_panel["log2_mic50"], np.log2(0.5)
        )
        apex_scores = {k: v[apex_idx] for k, v in full.items()}
        apex_env = envelope[apex_idx]
        apex_risk = risk[apex_idx]

        tops: dict[str, list[int]] = {}
        for name in sorted(categories.OBJECTIVES):
            # Re-rank on the ESM hemolysis estimate, which supersedes the coarse head.
            refined_q = categories.OBJECTIVES[name](
                apex_scores, apex_panel, apex_env, apex_risk, apex_scores["amp:clf"]
            ).astype(np.float32)
            q = categories.blend(
                refined_q, categories.APEX_OBJECTIVES[name](apex_sum)
            )
            local = pick_top(
                apex_seqs, q, apex_cluster, reference, top_k,
                long_kmers=long_kmers, safety=hem_esm["hem_esm:safe"],
            )
            tops[name] = [short[int(apex_idx[i])] for i in local]

        self.apex_index = apex_idx
        self.apex_summary = apex_sum

        forced = sorted({i for idx_list in tops.values() for i in idx_list})

        # The shortlist gets a better score from the full ensemble, but that score
        # is not on the same scale as the coarse one the rest of the pool carries,
        # and one ranking mixes them. Quantile-mapping the refined ordering back
        # onto the coarse scores of those same candidates keeps the improvement in
        # ordering without letting a scale offset promote or demote the shortlist
        # as a block.
        short_arr = np.asarray(short, dtype=np.int64)
        better = library_quality(short_seqs, full, panel, reuse[short_arr])
        refined = lib_q.copy()
        refined[short_arr] = _quantile_map(better, lib_q[short_arr])
        step(f"ranked {len(tops)} categories; selecting the library")
        centroids, sub_quotas = select.subcluster_quotas(reference, quotas)
        sub_assign = select.assign_subcluster(
            pool, select.property_cells(pool), centroids
        )
        step(f"split {len(sub_quotas)} cells into reference sub-regions")
        target_comp = select.mean_composition(reference)
        lib_idx, prices = select.select_matching_composition(
            pool, refined, quotas, cluster, n_sequences, target_comp,
            forced=forced, sub_quotas=sub_quotas, sub_assign=sub_assign,
        )
        gap = np.linalg.norm(
            select.mean_composition([pool[i] for i in lib_idx]) - target_comp
        )
        step(f"library selected; composition gap to the reference {gap:.4f}")
        self.composition_prices = prices

        self.pool = pool
        self.shortlist = short
        self.scores = full
        self.panel = panel
        self.library = [pool[i] for i in lib_idx]
        self.tops = {name: [pool[i] for i in idx] for name, idx in tops.items()}
        self.library_quality = refined

    def top(self, entry_point: str) -> list[str]:
        return list(self.tops.get(entry_point, self.tops[categories.DEFAULT]))


_CACHE: dict[tuple, _Build] = {}


def _build(n_sequences: int, length: int, seed: int, top_k: int) -> _Build:
    key = (n_sequences, length, seed, top_k)
    if key not in _CACHE:
        _CACHE[key] = _Build(n_sequences, length, seed, top_k)
    return _CACHE[key]


def generate(
    n_sequences: int = DEFAULT_LIBRARY,
    *,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Return `n_sequences` designed peptides.

    `length` is the maximum residue count, capped at the competition's 50. The
    library spans 8 to that bound, following the reference set's length profile.
    """
    return list(_build(n_sequences, length, seed, DEFAULT_TOP_K).library)


def score(sequences: list[str], category: str = categories.DEFAULT) -> list[float]:
    """Score sequences for ranking. Higher is better.

    This is the broad-spectrum objective by default, the same one that orders
    `generate/top.fasta`. It leaves out the APEX term, which only ever runs on a
    shortlist, so a score from here is the model-side component alone.
    """
    seqs = list(sequences)
    _set_determinism()
    bundle = scoring.load(_resolve("checkpoint", "scorers.pkl.gz"))
    raw = scoring.score(seqs, bundle)
    panel = scoring.panel_summary(raw)
    objective = categories.objective_for(category)
    values = objective(
        raw,
        panel,
        synthesis.envelope_score(seqs),
        synthesis.risk_score(seqs),
        raw["amp:clf"],
    )
    return [float(v) for v in values]


def _write_fasta(sequences: list[str], path: Path, prefix: str) -> None:
    with open(path, "w") as fh:
        for i, seq in enumerate(sequences, start=1):
            fh.write(f">{prefix}{i}\n{seq}\n")


def main() -> None:
    """Entry point for `generate` and for each `generate_<category>` script.

    The category is read from the invoked script's name, and output goes to a
    directory of that same name. The challenge template's validator runs
    `uv run generate` and reads `generate/`. The starter kits' validator runs one
    of the five category scripts and reads the matching directory. Exposing all
    six satisfies either one, and ranks separately for every category that Phase 2
    scores.
    """
    entry_point = Path(sys.argv[0]).stem or "generate"

    parser = argparse.ArgumentParser(
        description="Generate the AMP Challenge 2027 library and ranked top-100."
    )
    parser.add_argument("--n-sequences", type=int, default=DEFAULT_LIBRARY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--category",
        default=None,
        choices=sorted(categories.OBJECTIVES),
        help="Override the category inferred from the script name.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    category = args.category or (
        entry_point if entry_point in categories.OBJECTIVES else categories.DEFAULT
    )
    out_dir = args.out_dir or Path(entry_point)
    out_dir.mkdir(parents=True, exist_ok=True)

    build = _build(args.n_sequences, args.length, args.seed, args.top_k)

    library_path = out_dir / "library.fasta"
    _write_fasta(build.library, library_path, "seq")
    print(f"Generated {len(build.library)} sequences -> {library_path}")

    top = build.top(category)
    top_path = out_dir / "top.fasta"
    _write_fasta(top, top_path, "top")
    print(f"Top {len(top)} sequences for {category} -> {top_path}")


if __name__ == "__main__":
    main()
