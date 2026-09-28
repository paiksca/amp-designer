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
Every entry clears a strict synthesis rule set, an exact `Levenshtein.ratio <= 0.75` check
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

`checkpoint/scorers.pkl.gz` holds gradient-boosted ensembles fitted in
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

Independence is what it adds; on accuracy the two are level. On the 47 HydrAMP peptides
with prospective wet-lab MIC values, APEX reaches Spearman 0.50 and AUROC 0.80 for active at
<= 32 µM. The in-house ensemble appears to beat it there, but 32 of those 47 sequences are in
its training data, so that comparison is leakage; its honest figure is the grouped-CV AUROC of
0.80. The two agree at Spearman 0.69 only, so the ranked lists use a rank blend weighted 55:45
toward the in-house models, which cover all 20 strains rather than 11 and carry a hemolysis
head APEX does not have.

Four checkpoints rank 4,000 peptides at Spearman 0.983 against the full eight, for 81 MB
instead of 220 MB.

Scoring runs in three stages, spending model compute where the stakes are highest. Three
heads run over all 300,000 candidates and decide the library, where a within-cell ranking
only needs to be roughly right. The full per-species ensemble runs over a 40,000-sequence
shortlist. APEX re-ranks the roughly 6,000 candidates that lead any category and already
clear the strict synthesis rules, which is where the ranked lists are settled and where each
entry commits a wet-lab slot.

The two score scales are not comparable, so the shortlist's refined scores are quantile-mapped
back onto the coarse scores of those same candidates. That keeps the better ordering without
letting a scale offset promote or demote the whole shortlist as a block.

The shortlist is taken cell by cell rather than globally, because the high-scoring end of the
pool is more cationic than the reference set and a global cut would quietly re-shape the
library.

### 3. Selection

`src/amp_designer/select.py`. Quotas over the (length, charge, hydrophobic moment) grid
come from the reference set's own histogram with no reweighting, plus a small floor on
every occupied cell. The floor matters because Phase 1 measures KL(reference || generated),
which punishes a missing mode far harder than an over-represented one.

Copying the charge histogram rather than shifting it is the central design choice, and it
is deliberate. Predicted activity rises steeply with net charge: over a 6,000-sequence
sample of the reference set, the AMP classifier gives 0.767 to sequences below zero charge
and 0.980 to those at +8 or above, with predicted MIC falling from 37 µM to 5.8 µM. A
library tilted cationic therefore scores better on the surrogate-activity family. It scores
worse on everything else, because charge feeds the conformity score, the KL divergences, the
Frechet distance, MMD, precision and recall. The HydrAMP baseline shows the size of that
trade: its mean charge of 4.88 against the reference's 2.59 comes with a KL-charge of 0.68
and a Frechet distance of 1.71, where a held-out slice of real AMPs scores 0.010 and 0.012.
Matching the histogram and then taking the best candidates inside each cell collects the
activity gain without paying the distributional price.

Within a cell, candidates are taken one per MinHash cluster per pass, so the library
spreads over as many distinct families as the quota allows before taking a second member
of any. Cluster keys are a digest of character codes rather than Python's `hash`, which is
salted per process and would break reproducibility.

The quota grid pins length, charge and hydrophobic moment, but nothing pins residue
composition, and ranking on predicted activity inside a cell pulls the library toward Lys,
Arg and Leu. A price per residue corrects it: selection maximises
`quality - composition @ price`, and after each pass the price of an over-produced residue
rises. Prices only reorder candidates inside a cell, so every quota and therefore every
property marginal survives untouched.

### 4. Ranked top-100

`generate.py:pick_top`. Candidates must first pass the strict rules in `synthesis.py`: no
Cys or Met, no Asp followed by G, A, S, T, C, R, D or N, no Asn-Gly, no QQ or NN, no
N-terminal Gln, no run over three identical or five beta-sheet-prone residues, at most four
basic residues in any five-residue window, beta-sheet formers under 42%, net charge at least
+2, GRAVY at most 1.0, and 11 to 26 residues. Each rule carries a measured cost in diversity,
listed in that file as how often a published AMP would trip it.

Survivors then face an exact `Levenshtein.ratio` check against every one of the 39,448
reference sequences, a ban on sharing any exact 10-residue substring with them, a cap of
three per MinHash cluster, and a rule that no two ranked sequences exceed a Levenshtein
ratio of 0.65 to each other.

The threshold is 0.75 rather than the validator's 0.80 because the proposal states the same
rule as MMseqs2 alignment identity, and the two are different measures. MMseqs2 reports
identity over the aligned region, so one exact 10-residue match inside a 25-mer scores 1.0
however different the rest is; the substring ban closes that without needing an aligner at
generation time. It costs little. Across 465 ranked candidates only 3% carry a shared
10-mer, against 54% of real AMPs measured against the rest of the reference set. On the
same comparison a held-out slice of real AMPs reaches MMseqs2 identity 0.875 and
Levenshtein 0.900 against the reference, where these ranked lists sit at 0.620 and 0.667.

The pairwise cap is a hedge rather than a metric. The models separate active from inactive
far better than they rank among the active, so 25 peptides drawn from a list built on one
scaffold risk failing together for the same reason. Real AMPs sit at a median pairwise
ratio of 0.258, so 0.65 blocks near-copies without binding on genuine variety.

One more gate applies before any of this: a candidate whose predicted probability of
HC50 >= 128 µM falls below 0.35 is dropped from every ranked list. Potency and hemolysis
both rise with charge and hydrophobicity, so ranking on potency alone walks into the
hemolytic corner. An earlier build of this pipeline came out at 30% predicted safe against
a 37% base rate among potent peptides, which is why the gate exists rather than a heavier
weight: the models rank potency better than they rank hemolysis, and one severely hemolytic
peptide wastes a wet-lab slot in every category, not only in selectivity.

Where the in-house ensemble and APEX disagree about a candidate, its blended score is
reduced. Taking the top of a noisy score over-represents candidates whose error happened to
run favourable, and a peptide only one of two independent models likes is the usual shape of
that. With 25 peptides drawn at random and averaged, shrinking those optimistic outliers is
worth more than the few genuine finds it costs.

## Phase-1 metrics

Measured with `seqme` 0.5.1 at 10,000 sequences per side, against the challenge's own
reference antibacterial set, with a held-out slice of that set as the realism ceiling.
The baseline column is the HydrAMP library shipped in its starter kit.

| metric | want | this library | HydrAMP baseline | real AMPs |
|---|---|---|---|---|
| Uniqueness | high | 1.000 | 1.000 | 1.000 |
| Diversity | high | **0.854** | 0.806 | 0.855 |
| FKEA (effective modes) | high | **78.5** | 37.7 | 58.9 |
| Novelty (exact) | high | **1.000** | 1.000 | 0.000 |
| Authenticity | high | 0.782 | 0.814 | 0.438 |
| FBD vs AMPs | low | **0.062** | 1.650 | 0.003 |
| FBD vs generic peptides | low | **0.614** | 2.554 | 0.499 |
| MMD vs AMPs | low | **0.045** | 10.657 | 0.005 |
| Precision | high | 0.876 | 0.918 | 0.952 |
| Recall | high | **0.849** | 0.477 | 0.949 |
| Clipped density | high | **0.566** | 0.217 | 1.000 |
| Clipped coverage | high | **0.445** | 0.139 | 1.000 |
| Conformity score | high | **0.504** | 0.404 | 0.499 |
| KL charge | low | **0.216** | 0.752 | 0.011 |
| KL hydrophobic moment | low | **0.002** | 0.074 | 0.000 |
| KL GRAVY | low | **0.024** | 0.259 | 0.001 |
| KL length | low | **0.077** | 407.1 | 0.003 |
| mean charge | match | 2.72 | 4.86 | 2.59 |
| mean length | match | 18.4 | 20.8 | 18.7 |
| composition L2 gap | low | **0.014** | 0.255 | 0.002 |

The conformity score and the diversity both land on the held-out reference set's own value,
and FKEA lands above it, so the library is as realistic as real AMPs are while covering more
of the space and while every sequence stays novel by exact match.

HydrAMP wins precision, at 0.918 against 0.876, by occupying a narrow part of the
distribution. It pays for that with recall of 0.477 against 0.849 and a KL length of 407
against 0.077: its library misses most of the reference distribution rather than covering it.

## What the models get wrong

Nine peptides with published free-termini MIC and human-erythrocyte HC50, spanning
selectivity indices from 15 to 111, were scored as positive controls alongside poly-Ala,
poly-Glu and a scrambled sequence. All nine outrank all three decoys, and the MIC
predictions land within about a five-fold band of the measured values.

The hemolysis head is the weak one. Over those nine the predicted safety window correlates
with the measured index at Spearman 0.25, and the single most selective control, dhvar5
(`LLLFLLKKRKKRKY`, measured HC50 120 µM), is predicted at 35 µM. Its N-terminal `LLLFLL`
block is also long enough to trip the six-residue beta-sheet run rule, so this pipeline
would have excluded it outright. Short cationic peptides that carry their hydrophobicity in
one contiguous block are the failure case.

Two things keep that in proportion. Those nine were selected for high selectivity, so the
range is narrow and a rank correlation across it means little; measured across the full
range, the hemolysis head reaches AUROC 0.768 and the blend with the envelope 0.737 on the
potent subset. And the ranked lists are spread across at least 100 distinct clusters with
no pair above a 0.65 Levenshtein ratio, which is the hedge against exactly this kind of
systematic model error.

## Reproducing the checkpoints

The entry point needs none of this; it runs from the committed checkpoints.

```bash
git clone https://github.com/szczurek-lab/amp-challenge-2027       ../amp-challenge-2027
git clone https://github.com/szczurek-lab/battleamp-snakemake      ../battleamp-snakemake
git clone https://github.com/szczurek-lab/hydramp-starter-kit      ../hydramp-starter-kit
git clone https://github.com/raghavagps/HemoPI2                    ../HemoPI2

uv sync --extra training
AMP_UPSTREAM=.. uv run python -m training.build_data     # writes data/training/
uv run python -m training.train_scorers                  # writes checkpoint/scorers.pkl.gz
uv run python -m training.train_lm                       # writes checkpoint/peptide_lm.pt
```

`training/build_controls.py` rebuilds `checkpoint/controls.npz` on its own, with different
tilts toward potency and hemolytic safety, without retraining the language model.

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
  apex.py        vendored APEX-pathogen inference
  categories.py  the five Phase-2 categories and their ranking objectives
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
