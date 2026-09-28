# amp-designer

A submission for [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027):
we condition a peptide language model on properties, then select its output to match the
known-AMP distribution cell by cell and use the remaining freedom for diversity, predicted
potency, predicted low hemolysis, and synthesis risk.

```bash
uv run generate                     # template validator: writes generate/
uv run generate_broad_spectrum      # starter-kit validator: writes generate_broad_spectrum/
uv run generate_gram_pos
uv run generate_gram_neg
uv run generate_mdr
uv run generate_therapeutic
```

Each writes `library.fasta` (50,000 sequences) and `top.fasta` (100 ranked candidates) into
a directory named after the script, byte-identical on repeated runs.

The challenge template's validator runs `uv run generate` and reads `generate/`. Both
starter kits ship a validator that runs one of five `generate_<category>` scripts and reads
the matching directory. We expose all six, so either validator passes.

The six share one library because Phase 1 scores diversity, novelty, distributional
similarity and property conformity, which are not category-specific. The ranked lists differ
because Phase 2 scores five categories against different slices of the 20-strain panel and
the list that wins one does not win another. Both official baselines expose only
`generate_broad_spectrum`.

| entry point | ranked on |
|---|---|
| `generate_broad_spectrum` | success rate over the 20 strains, weighted 15:5 Gram-negative to Gram-positive, with a worst-species term standing in for MIC90 |
| `generate_gram_pos` | the 5 Gram-positive strains |
| `generate_gram_neg` | the 15 Gram-negative strains |
| `generate_mdr` | the 8 MDR ESKAPE isolates, by the species head that covers it |
| `generate_therapeutic` | safety window HC50/MIC50, as potency subject to clearing the 128 µM HC50 ceiling |

## Abstract

We train a 1.8M-parameter decoder-only transformer on 44,585 known antibacterial peptides
drawn from public databases. Each training sequence carries a four-token control prefix
giving its length bin, net-charge bin, predicted Gram-negative potency quintile, and
predicted hemolytic-safety quartile. We compute the last two with gradient-boosted ensembles
fitted here on DBAASP, GRAMPA and HemoPI2 assay data, so the conditioning signal is
self-distilled from measured activity rather than hand-specified.

We sample 300,000 candidates, using only control combinations the training corpus contains
and reweighting toward the potent, non-hemolytic end of that set. We then select rather than
filter: we copy quotas over a (length, net charge, hydrophobic moment) grid from the
reference antibacterial set without reweighting, and inside each cell we take one candidate
per sequence cluster in descending quality. Because we fix the distribution and use the
freedom inside it, one library satisfies Phase 1's four metric families at once, which
otherwise pull against each other.

We rank the top-100 on the quantity Phase 2 scores. Organizers draw 25 peptides uniformly
from the list and average them, so we optimize expected value across the 100 rather than the
best few. We weight Gram-negative potency double because 15 of the 20 panel strains are
Gram-negative. We treat hemolysis as a constraint rather than an objective because HC50 is
reported only up to 128 µM and past that ceiling a lower value scores the same. We penalize
synthesis risk because a peptide that fails synthesis is never replaced and enters the mean
at its worst value.

Every entry clears a strict synthesis rule set, a gate on predicted hemolysis, an exact
`Levenshtein.ratio <= 0.75` check against the 39,448 reference sequences, a ban on sharing
an exact 10-residue substring with them, and a cap of 0.65 on the Levenshtein ratio between
any two ranked sequences.

We bring in two outside signals. APEX-pathogen predicts micromolar MIC for 11 of the 20
panel strains and comes from the laboratory running Phase 2, so we vendor it and rank-blend
it with our ensembles at 45%, penalizing candidates the two disagree about. Hemolysis is our
weakest prediction and the one the selectivity category turns on, so we re-estimate it with
an ESM-2 head over the few thousand candidates that reach the lists.

We apply no manual curation.

## How it works

### 1. Conditioned generation

`src/amp_designer/lm.py` is a 4-layer, 192-dimensional, 6-head decoder-only transformer
(1.81M parameters) over the 20 amino acids, with key/value caching for decoding. We prefix
every sequence with four control tokens:

| control | bins | source |
|---|---|---|
| length | 7 | residue count |
| net charge | 7 | Henderson-Hasselbalch at pH 7.4, free termini |
| Gram-negative potency | 5 | quintile of this repo's MIC ensemble |
| hemolytic safety | 4 | quartile of this repo's P(HC50 >= 128 µM) |

`checkpoint/controls.npz` holds the 687 control combinations the corpus contains, with mass
reweighted by `exp(0.55 * potency_bin + 0.35 * safety_bin)`. We never request a combination
the model has not seen, which keeps the conditioning in distribution. We draw candidates at
three temperatures (0.85, 1.0, 1.15) under nucleus sampling at `top_p=0.95`: the cool pass
supplies high-likelihood sequences, the warm pass supplies diversity.

### 2. Scoring

`checkpoint/scorers.pkl.gz` holds gradient-boosted ensembles fitted in
`training/train_scorers.py`. We measure accuracy with five-fold cross-validation grouped by
MMseqs2 cluster at 50% identity, as DBAASP is dense with analogue series and an ungrouped
split reports roughly twice the real accuracy.

| head | n | Spearman | AUROC |
|---|---|---|---|
| MIC, Gram-negative panel | 13,269 | 0.55 | 0.78 (active at <= 16 µM) |
| MIC, Gram-positive panel | 10,278 | 0.48 | 0.74 |
| MIC, *E. coli* | 12,220 | 0.54 | 0.77 |
| MIC, *P. aeruginosa* | 7,132 | 0.48 | 0.74 |
| MIC, *S. aureus* | 9,614 | 0.48 | 0.75 |
| MIC, *K. pneumoniae* | 3,060 | 0.48 | 0.75 |
| MIC, *A. baumannii* | 2,591 | 0.56 | 0.78 |
| MIC, *E. faecalis* | 2,161 | 0.49 | 0.75 |
| MIC, *B. subtilis* | 3,058 | 0.51 | 0.77 |
| log2 HC50, descriptors | 5,617 | 0.42 | 0.76 (HC50 >= 128 µM) |
| log2 HC50, ESM-2 + descriptors | 5,617 | 0.46 | 0.79 |
| MBC-Attention surrogate | 59,466 | 0.80 | |
| AMP classifier | 30,000 | | 0.92 |

BattleAMP's 2026 holdout puts published AMP classifiers at AUROC 0.70 to 0.75, so these
numbers match what the literature reports on a clean split.

The hemolysis head takes ESM-2 t12 embeddings alongside the descriptors. Over the same 5,617
sequences and the same cluster-grouped splits, descriptors alone reach AUROC 0.755, ESM-2
alone 0.764, and the two together 0.786. Embeddings alone lose to the combination in the
variants we tried: the descriptors state charge and hydrophobicity outright, where a masked
language model only implies them. We vendor the encoder at float16 under
`checkpoint/esm2_t12_35M/`, so generation fetches nothing.

We predict at `amidated=0`. Most potent DBAASP entries are C-terminally amidated and
amidation raises potency several-fold, but this competition requires free termini, so we
made the amidation state a model input and hold it at free acid.

#### APEX-pathogen

We vendor four of the eight released APEX-pathogen checkpoints under `checkpoint/apex/` (Wan
et al., de la Fuente lab, *Nature Microbiology* 2025, MIT licensed). APEX predicts MIC in µM
against an 11-pathogen panel, and all 11 are on this competition's 20-strain panel: the 5
Gram-positive strains, 6 of the 15 Gram-negative, and 4 of the 8 MDR isolates. It is also
the model that lab used to score the AMP-Diffusion baseline, and that lab runs Phase 2.

It adds independence rather than accuracy: on the 47 HydrAMP peptides with prospective
wet-lab MIC values, APEX reaches Spearman 0.50 and AUROC 0.80 for active at <= 32 µM. Our
ensemble appears to beat it there, but 32 of those 47 sequences are in APEX's training data,
which makes that comparison leakage. The two agree at Spearman 0.69 only, so we blend ranks
55:45 toward our models, which cover all 20 strains rather than 11 and include a hemolysis
head APEX does not have.

Four checkpoints rank 4,000 peptides at Spearman 0.983 against the full eight, for 81 MB
instead of 220 MB.

We score in four stages, putting the slow models only where the stakes are highest. Four
fast heads run over all 300,000 candidates and choose the library, where a within-cell
ranking only needs to be roughly right. The full per-species ensemble runs over a
90,000-sequence shortlist. APEX re-ranks the 3,552 candidates that lead any category and
clear both the strict synthesis rules and the hemolysis gate, and the ESM hemolysis head
runs over the same 3,552, which settles the ranked lists.

We take the shortlist cell by cell rather than globally because the high-scoring end of the
pool is more cationic than the reference set and a global cut would re-shape the library.
The two score scales are not comparable, so we quantile-map the shortlist's refined scores
back onto the coarse scores of the same candidates, which keeps an offset from promoting the
whole shortlist as a block.

#### The distilled surrogate

`surrogate:mbc` reproduces MBC-Attention, one of the three activity surrogates the proposal
names. Running MBC-Attention itself takes 27 minutes per 300,000 candidates and pulls in
TensorFlow and a vendored model, so we distilled it into the same gradient-boosting form as
our other heads, fitting on 59,466 sequences spanning the library, the training corpus and
the reference set. It reproduces the real model at Spearman 0.801 on held-out clusters.

Measured directly on MBC-Attention at 4,000 sequences per side, this library has a median
predicted MIC of 19.2 µM against 27.4 µM for real AMPs and 14.5 µM for the HydrAMP baseline.
We lose to HydrAMP because of the charge choice described under Selection.

We use the distilled head for the library only and leave it out of the top-100 objectives.
MBC-Attention scores poly-glutamate, which cannot be an antimicrobial peptide, at 1.8 µM,
and we will not commit a wet-lab slot on a model with that failure mode.

### 3. Selection

`src/amp_designer/select.py`. We read quotas over the (length, charge, hydrophobic moment)
grid straight off the reference set's histogram, with no reweighting, and put a small floor
under every occupied cell. We add the floor because Phase 1 measures KL(reference ||
generated), which punishes a missing mode far harder than an over-represented one.

We copy the charge histogram rather than shifting it, which is the pipeline's largest
trade-off. Predicted activity rises with net charge: over a 6,000-sequence sample of the
reference set, the AMP classifier gives 0.767 to sequences below zero charge and 0.980 to
those at +8 or above, with predicted MIC falling from 37 µM to 5.8 µM. A cationic library
therefore scores better on predicted activity and worse on the four Phase-1 families,
because charge enters the conformity score, the KL divergences, the Frechet distance, MMD,
precision and recall. The HydrAMP baseline shows how large the trade-off is: its mean charge
of 4.88 against the reference's 2.59 comes with a KL-charge of 0.68 and a Frechet distance
of 1.71, where a held-out slice of real AMPs scores 0.010 and 0.012. We match the histogram
and take the best candidates inside each cell instead, so we get the higher activity and
still match the reference distribution.

Within a cell we take one candidate per MinHash cluster per pass, so the library spreads
over as many distinct families as the quota allows before we take a second member of any. We
build cluster keys from a digest of character codes rather than Python's `hash`, which is
salted per process and would break reproducibility.

The quota grid pins length, charge and hydrophobic moment, but nothing pins residue
composition, and ranking on predicted activity inside a cell pulls us toward Lys, Arg and
Leu. We correct that with a price per residue: selection maximizes `quality - composition @
price`, and after each pass we raise the price of any residue we over-produced. The prices
only reorder candidates inside a cell, so the quotas and the property marginals do not
change.

Cells are still wide: two sequences can share all three bins and sit far apart in
composition, and if we fill a cell from one corner we cover less of the reference
distribution than its quota suggests. So we k-means cluster the reference members of a cell
and split its quota across those sub-regions in proportion to how many reference sequences
they hold. That turns 53 cells into 453 sub-regions, with the sub-quotas summing back to the
cell quota.

The split raises clipped coverage from 0.512 to 0.544 and clipped density from 0.580 to
0.621, with recall, precision, conformity, the Frechet distance and KL-length all better. It
widens the gap in composition from 0.013 to 0.017 and MMD from 0.876 to 0.947 because a
tighter constraint leaves the residue prices fewer candidates to choose among.

### 4. Ranked top-100

`generate.py:pick_top`. Candidates must first pass the strict rules in `synthesis.py`: no
Cys or Met, no Asp followed by G, A, S, T, C, R, D or N, no Asn-Gly, no QQ or NN, no
N-terminal Gln, no run over three identical or five beta-sheet-prone residues, at most four
basic residues in any five-residue window, beta-sheet formers under 42%, net charge at least
+2, GRAVY at most 1.0, and 11 to 26 residues. For each rule that file records how often a
published AMP would trip it.

We then check survivors against the 39,448 reference sequences with an exact
`Levenshtein.ratio`, drop any that share an exact 10-residue substring with one, cap each
MinHash cluster at three, and drop any candidate within 0.65 Levenshtein of a sequence we
have already picked.

The threshold is 0.75 rather than the validator's 0.80 because the proposal states the same
rule as MMseqs2 alignment identity, and the two are different measures. MMseqs2 reports
identity over the aligned region, so one exact 10-residue match inside a 25-mer scores 1.0
however different the rest is. Banning shared 10-mers closes that gap without making us run
an aligner at generation time. The rule drops 3% of our ranked candidates, where it would
drop 54% of real AMPs measured against the rest of the reference set.

On that same comparison a held-out slice of real AMPs reaches MMseqs2 identity 0.875 and
Levenshtein 0.900 against the reference, where these ranked lists sit at 0.60 and 0.67. The
identity hits that remain above 0.80 are short local alignments, median 11 residues over
half the query, with 3 of 129 reaching 80% query coverage. The real-AMP hits cover the query
completely.

The pairwise cap is a hedge rather than a metric. The models separate active from inactive
far better than they rank among the active, so 25 peptides drawn from a list built on one
scaffold risk failing together for the same reason. Real AMPs sit at a median pairwise ratio
of 0.258, where a cap of 0.65 blocks near-copies without binding on genuine variety.

One gate applies before any of this: we drop from every ranked list any candidate whose
predicted probability of HC50 >= 128 µM falls below 0.50. Potency and hemolysis both rise
with charge and hydrophobicity, so ranking on potency alone drifts toward hemolytic
sequences. We use a gate rather than a heavier weight because the models rank potency better
than they rank hemolysis and one severely hemolytic peptide wastes a wet-lab slot in every
category, not only in selectivity.

Where our ensemble and APEX disagree about a candidate, we reduce its blended score. Taking
the top of a noisy score over-represents candidates whose error happened to run favorably,
and a peptide that only one of two independent models rates highly is usually one of them.
With 25 peptides drawn at random and averaged, the mean is better off without those
optimistic outliers, even though a few genuine finds go with them.

## Phase-1 metrics

We measure with `seqme` 0.5.1 against the challenge's reference antibacterial set, using a
held-out slice of that set to show what real AMPs score. The baseline column is the HydrAMP
library shipped in its starter kit. Embeddings are ESM-2 t6_8M, one of the two model
families the competition names, at 4,000 sequences per side.

| metric | want | this library | HydrAMP baseline | real AMPs |
|---|---|---|---|---|
| Uniqueness | high | 1.000 | 1.000 | 1.000 |
| Diversity | high | **0.851** | 0.805 | 0.855 |
| Novelty (exact) | high | **1.000** | 1.000 | 0.000 |
| Authenticity | high | 0.779 | 0.914 | 0.550 |
| FBD vs AMPs | low | **0.440** | 7.859 | 0.053 |
| FBD vs generic peptides | low | **3.872** | 13.201 | 3.116 |
| MMD vs AMPs | low | **0.947** | 48.717 | 0.031 |
| Precision | high | **0.894** | 0.648 | 0.947 |
| Recall | high | **0.857** | 0.402 | 0.937 |
| Clipped density | high | **0.621** | 0.105 | 1.000 |
| Clipped coverage | high | **0.544** | 0.129 | 1.000 |
| Conformity score | high | **0.512** | 0.408 | 0.496 |
| KL charge | low | **0.079** | 0.559 | 0.009 |
| KL hydrophobic moment | low | **0.005** | 0.066 | 0.000 |
| KL GRAVY | low | **0.022** | 0.223 | 0.000 |
| KL length | low | **0.037** | 257.5 | 0.009 |
| mean charge | match | 2.73 | 4.88 | 2.63 |
| mean length | match | 18.7 | 20.8 | 18.7 |
| composition L2 gap | low | **0.017** | 0.256 | 0.005 |

The conformity score and the diversity land on the held-out reference set's value, so the
library is as realistic as real AMPs are while every sequence stays novel by exact match.
Recall, clipped density and clipped coverage are four to five times the baseline's, as the
quota grid covers the reference distribution where the baseline occupies one corner of it,
and the KL divergences follow for the same reason.

We leave FKEA out of the table because at 4,000 sequences it sits near its sample-size
ceiling for the libraries we scored and separates nothing. The seqme authors report the same
instability.

We repeated the table with a 1-mer and 2-mer frequency embedding in place of ESM-2. No
ranking changed except precision, which reads 0.876 against the baseline's 0.918 there. That
reading is an artifact: our novelty term penalizes reuse of reference 6-mers, so a k-mer
embedding measures what that term suppresses.

## The ranked lists

The five lists hold 100 sequences each. Every entry is in the library and predicted active
at 16 µM or below, no entry shares an exact 10-mer with the reference set or sits above a
Levenshtein ratio of 0.75 to it, and no pair sits above 0.65 to another.

| list | predicted MIC50 | P(HC50 >= 128) | worst entry | predicted window | envelope |
|---|---|---|---|---|---|
| broad spectrum | 7.65 µM | 0.63 | 0.50 | 12.3 | 0.86 |
| Gram-negative | 7.23 µM | 0.63 | 0.50 | 12.6 | 0.87 |
| Gram-positive | 8.06 µM | 0.63 | 0.50 | 11.4 | 0.79 |
| MDR | 7.40 µM | 0.63 | 0.50 | 12.0 | 0.81 |
| selectivity | 8.06 µM | 0.64 | 0.50 | 12.7 | 0.89 |

Envelope is `synthesis.py:envelope_score`, which measures how well a sequence's descriptors
match measured peptides that are both potent and non-hemolytic. The hemolysis figures come
from the descriptor head while selection used the ESM head, so they are an independent check
rather than a restatement of the scores we selected on.

The five lists share between 28 and 71 sequences pairwise and together cover 241 distinct
peptides across the 500 slots. The selectivity list has both the highest predicted window
and the highest hemolytic safety.

## What the models get wrong

We scored nine peptides with published free-termini MIC and human-erythrocyte HC50, spanning
selectivity indices from 15 to 111, against poly-Ala, poly-Glu and a scrambled sequence. All
nine outrank the three decoys, and our MIC predictions land within five-fold of the measured
values.

Our hemolysis head remains the weak one even after the ESM-2 rebuild. Over those nine its
predicted safety window correlates with the measured index at Spearman 0.25, and we predict
50 µM for dhvar5 (`LLLFLLKKRKKRKY`), the most selective control, against a measured 120 µM.
Its N-terminal `LLLFLL` block is also long enough to trip the six-residue beta-sheet run
rule, so this pipeline would have excluded it outright. Short cationic peptides that carry
their hydrophobicity in one contiguous block are where the head fails. That 0.25 understates
it. We chose those nine for high selectivity, so the range is narrow, and across the full
range it reaches AUROC 0.786.

The ranked lists are also concentrated in one structural class. Every entry is a cationic
amphipathic sequence of the helix-forming kind, and across the selectivity list the residue
counts are Lys 481, Arg 276, Ile 189, Leu 162, Val 160 and Trp 136, with no Cys and no Met.
That is the best-validated AMP class and the one our training data is densest in, and with
25 peptides drawn at random and averaged, the class with the highest expected value is the
right choice for the mean. If that class fails systematically, every peptide in the draw
fails with it, so we spread each list of 100 over 99 or 100 distinct MinHash clusters with
no pair above a 0.65 Levenshtein ratio.

Amphipathic beta-sheet designs were the obvious second class to add. We tested the
beta-sheet hydrophobic moment against measured HC50 on 465 potent peptides, found the
relationship weak and not monotonic, and added none on that basis.

Arg makes up 43% of the cationic residues across the ranked lists, inside the 0 to 50% band
where the measured safety window peaks, and that comes from the envelope's Arg-share term
and the fitted models rather than a rule we set.

## Reproducing the checkpoints

The entry point runs from the committed checkpoints. This section rebuilds them.

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

`train_scorers.py` needs MMseqs2 on `PATH` for the grouped cross-validation splits, and the
entry point does not.

## Training data

Every source is public. `data/training/README.md` lists them with licenses, row counts, and
what they contribute.

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

The code in this repository was written with Claude (Anthropic) under human direction. The
method design, the data sources and the submitted content are the author's responsibility.

## License

MIT, see [LICENSE](LICENSE).
