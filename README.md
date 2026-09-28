# amp-designer

A submission for [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027):
a property-conditioned peptide language model whose output is selected to match the
known-AMP distribution cell by cell, while spending every remaining degree of freedom
on diversity, predicted potency, predicted low hemolysis, and synthesis risk.

```bash
uv run generate                     # template validator: writes generate/
uv run generate_broad_spectrum      # starter-kit validator: writes generate_broad_spectrum/
uv run generate_gram_pos
uv run generate_gram_neg
uv run generate_mdr
uv run generate_therapeutic
```

Each writes `library.fasta` (50,000 sequences) and `top.fasta` (100 ranked candidates)
into a directory named after the script. Every file is byte-identical on repeated runs.

The challenge template's validator runs `uv run generate` and reads `generate/`; both
starter kits ship a validator that runs one of five `generate_<category>` scripts and
reads the matching directory. All six entry points are exposed, so either validator
passes.

The library is the same across all six, because Phase 1 scores diversity, novelty,
distributional similarity and property conformity, none of which is category-specific.
The ranked lists differ: Phase 2 scores five categories against different slices of the
20-strain panel, and the list that wins one does not win another. Both official baselines
expose only `generate_broad_spectrum`; this submission ranks separately for all five.

| entry point | ranked on |
|---|---|
| `generate_broad_spectrum` | success rate over all 20 strains, weighted 15:5 Gram-negative to Gram-positive, with a worst-species term standing in for MIC90 |
| `generate_gram_pos` | the 5 Gram-positive strains |
| `generate_gram_neg` | the 15 Gram-negative strains |
| `generate_mdr` | the 8 MDR ESKAPE isolates, by the species head covering each |
| `generate_therapeutic` | safety window HC50/MIC50, as potency subject to clearing the 128 µM HC50 ceiling |

## Abstract

A 1.8M-parameter decoder-only transformer is trained on 44,585 known antibacterial
peptides drawn from public databases. Each training sequence carries a four-token
control prefix giving its length bin, net-charge bin, predicted Gram-negative potency
quintile, and predicted hemolytic-safety quartile; the last two come from gradient-boosted
ensembles fitted in this repository on DBAASP, GRAMPA, and HemoPI2 assay data, so the
conditioning signal is self-distilled from measured activity rather than hand-specified.

Sampling draws 300,000 candidates using only control combinations the training corpus
actually contains, reweighted toward the potent and non-hemolytic end of that set.
Candidates are then selected, not filtered: quotas over a (length, net charge, hydrophobic
moment) grid are copied without reweighting from the reference antibacterial set, and
inside each cell candidates are taken one per sequence cluster in descending quality.
Fixing the distribution and spending the freedom inside it is what lets one library
satisfy Phase 1's four metric families at once, which otherwise pull against each other.

The ranked top-100 is chosen on the quantity Phase 2 scores: 25 peptides are drawn
uniformly from it and the team score is their arithmetic mean, so the objective is
expected value across all 100 rather than the best few. Gram-negative potency carries
double weight because 15 of the 20 panel strains are Gram-negative; hemolysis enters as a
constraint rather than an objective because HC50 is reported only up to 128 µM, past which
being less hemolytic earns nothing; and synthesis risk is penalised directly because a
peptide that fails synthesis is never replaced and enters the mean at its worst value.
Every entry clears a strict synthesis rule set, an exact `Levenshtein.ratio <= 0.8` check
against all 39,448 reference sequences, and a cap of three per sequence cluster, so a
single mistaken scaffold cannot take the whole draw down with it.

No manual curation or hand-selection is applied at any stage.

## How it works

### 1. Conditioned generation

`src/amp_designer/lm.py` is a 4-layer, 192-dimensional, 6-head decoder-only transformer
(1.81M parameters) over the 20 amino acids, with key/value caching for decoding. Every
sequence is prefixed with four control tokens:

| control | bins | source |
|---|---|---|
| length | 7 | residue count |
| net charge | 7 | Henderson-Hasselbalch at pH 7.4, free termini |
| Gram-negative potency | 5 | quintile of this repo's MIC ensemble |
| hemolytic safety | 4 | quartile of this repo's P(HC50 >= 128 µM) |

`checkpoint/controls.npz` holds the 687 control combinations the corpus contains, with
mass reweighted by `exp(0.55 * potency_bin + 0.35 * safety_bin)`. Sampling never requests
a combination the model has not seen, which keeps the conditioning in distribution.
Candidates are drawn at three temperatures (0.85, 1.0, 1.15) under nucleus sampling at
`top_p=0.95`: the cool pass supplies high-likelihood sequences, the warm pass supplies
diversity.

### 2. Scoring

`checkpoint/scorers.pkl` holds gradient-boosted ensembles fitted in
`training/train_scorers.py`. Accuracy is measured with five-fold cross-validation grouped
by MMseqs2 cluster at 50% identity, because DBAASP is dense with analogue series and an
ungrouped split reports roughly twice the real accuracy.

| head | n | Spearman | AUROC |
|---|---|---|---|
| MIC, Gram-negative panel | 8,351 | 0.60 | 0.80 (active at <= 16 µM) |
| MIC, Gram-positive panel | 4,268 | 0.54 | 0.77 |
| MIC, *E. coli* | 7,958 | 0.59 | 0.79 |
| MIC, *P. aeruginosa* | 2,992 | 0.59 | 0.80 |
| MIC, *S. aureus* | 4,093 | 0.52 | 0.77 |
| MIC, *K. pneumoniae* | 1,321 | 0.53 | 0.76 |
| MIC, *A. baumannii* | 1,131 | 0.50 | 0.77 |
| MIC, *E. faecalis* | 980 | 0.57 | 0.79 |
| MIC, *B. subtilis* | 1,030 | 0.45 | 0.77 |
| log2 HC50 | 1,957 | 0.51 | 0.77 (HC50 >= 128 µM) |
| AMP classifier | 30,000 | | 0.92 |

For context, BattleAMP's 2026 holdout puts published AMP classifiers at AUROC 0.70-0.75,
so these are in a plausible range rather than the product of a leaky split.

Predictions are made at `amidated=0`. Most potent DBAASP entries are C-terminally
amidated and amidation typically buys several-fold potency, but this competition requires
free termini, so the amidation state is a model input and is held at free acid. Scoring
otherwise overstates what these designs can do.

#### APEX-pathogen

`checkpoint/apex/` vendors four of the eight released APEX-pathogen checkpoints (Wan et al.,
de la Fuente lab, *Nature Microbiology* 2025, MIT licensed). APEX predicts MIC in µM against
an 11-pathogen panel, and all 11 are on this competition's 20-strain panel: every Gram-positive
strain, 6 of the 15 Gram-negative, and 4 of the 8 MDR isolates. It is also the model that lab
used to score the AMP-Diffusion baseline, and that lab runs Phase 2.

It is here because it is independent, not because it is better. On the 47 HydrAMP peptides
with prospective wet-lab MIC values, APEX reaches Spearman 0.50 and AUROC 0.80 for active at
<= 32 µM. The in-house ensemble appears to beat it there, but 32 of those 47 sequences are in
its training data, so that comparison is leakage; its honest figure is the grouped-CV AUROC of
0.80. The two agree at Spearman 0.69 only, so the ranked lists use a rank blend weighted 55:45
toward the in-house models, which cover all 20 strains rather than 11 and carry a hemolysis
head APEX does not have.

Four checkpoints rank 4,000 peptides at Spearman 0.983 against the full eight, for 81 MB
instead of 220 MB.

Scoring runs in three stages. Three heads run over all 300,000 candidates and decide the
library, where a within-cell ranking only needs to be roughly right. The full per-species
ensemble runs over a 40,000-sequence shortlist and decides the ranked list, where each
entry commits a wet-lab slot. APEX then re-ranks the
roughly 6,000 candidates that lead any category and already clear the strict synthesis rules,
which is the only region where the ranked list is actually decided.

The shortlist is taken cell by cell rather than globally, because the high-scoring end of the
pool is more cationic than the reference set and a global cut would quietly re-shape the
library.

### 3. Selection

`src/amp_designer/select.py`. Quotas over the (length, charge, hydrophobic moment) grid
come from the reference set's own histogram with no reweighting, plus a small floor on
every occupied cell. The floor matters because Phase 1 measures KL(reference || generated),
which punishes a missing mode far harder than an over-represented one.

Within a cell, candidates are taken one per MinHash cluster per pass, so the library
spreads over as many distinct families as the quota allows before taking a second member
of any. Cluster keys are a digest of character codes rather than Python's `hash`, which is
salted per process and would break reproducibility.

### 4. Ranked top-100

`generate.py:pick_top`. Candidates must pass the strict synthesis rules in
`synthesis.py` (no Cys or Met, no aspartimide-prone motif, no run over two identical or
four hydrophobic residues, at most four basic residues in any five-residue window, net
charge at least +2, GRAVY at most 1.2, 10 to 30 residues), then an exact
`Levenshtein.ratio` check against every one of the 39,448 reference sequences, then a cap
of three per cluster.

## Reproducing the checkpoints

The entry point needs none of this; it runs from the committed checkpoints.

```bash
git clone https://github.com/szczurek-lab/amp-challenge-2027       ../amp-challenge-2027
git clone https://github.com/szczurek-lab/battleamp-snakemake      ../battleamp-snakemake
git clone https://github.com/szczurek-lab/hydramp-starter-kit      ../hydramp-starter-kit
git clone https://github.com/raghavagps/HemoPI2                    ../HemoPI2

uv sync --extra training
AMP_UPSTREAM=.. uv run python -m training.build_data     # writes data/training/
uv run python -m training.train_scorers                  # writes checkpoint/scorers.pkl
uv run python -m training.train_lm                       # writes checkpoint/peptide_lm.pt
```

`train_scorers.py` needs MMseqs2 on `PATH` for the grouped cross-validation splits. The
entry point does not.

## Training data

Every source is public. `data/training/README.md` lists each one with its licence, row
count, and what it contributes. No proprietary or non-public data is used.

## Layout

```
src/amp_designer/
  lm.py          conditioned transformer, KV-cached decoding
  features.py    physicochemical descriptors
  scoring.py     MIC / hemolysis / AMP-classifier inference
  select.py      quota grid and cluster-first selection
  synthesis.py   synthesis and solubility risk rules
  novelty.py     Levenshtein and MMseqs2 novelty checks
  generate.py    entry point
training/        offline: data assembly, scorer and LM training
checkpoint/      committed model weights
data/            challenge reference set, assembled training data
scripts/         the challenge validator, copied verbatim
```

## Options

| flag | default | meaning |
|---|---|---|
| `--n-sequences` | 50000 | library size |
| `--top-k` | 100 | ranked list size |
| `--length` | 50 | maximum residue count |
| `--seed` | 42 | random seed |

## Use of AI assistants

Per the NeurIPS Main Track Handbook, this is disclosed: the code in this repository was
written with Claude (Anthropic) under human direction. The method design, the data
sources, and the submitted content are the author's responsibility.

## License

MIT, see [LICENSE](LICENSE).
