# Antimicrobial peptide design by distribution matching

A submission for [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027):
we condition a peptide language model on properties. Quotas copied from the known-AMP
distribution set how many sequences go in each cell, and diversity, predicted potency,
predicted low hemolysis and synthesis risk decide which ones.

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
the matching directory. We expose all six entry points, so either validator passes.

The entry points share one library because Phase 1 scores diversity, novelty, distributional
similarity and property conformity, which are not category-specific. The ranked lists differ
because Phase 2 scores five categories against different slices of the 20-strain panel. Both
official baselines expose only `generate_broad_spectrum`.

| entry point | ranked on |
|---|---|
| `generate_broad_spectrum` | success rate over the 20 strains, weighted 15:5 Gram-negative to Gram-positive, with a worst-species term standing in for MIC90 |
| `generate_gram_pos` | the 5 Gram-positive strains |
| `generate_gram_neg` | the 15 Gram-negative strains |
| `generate_mdr` | the 8 MDR ESKAPE isolates, by the species head that covers it |
| `generate_therapeutic` | safety window HC50/MIC50, as potency subject to clearing the 128 µM HC50 ceiling |

## Abstract

We train a 1.8M-parameter decoder-only transformer on 44,585 known antibacterial peptides
drawn from public databases. Each training sequence has a four-token control prefix giving
its length bin, net-charge bin, predicted Gram-negative potency quintile, and predicted
hemolytic-safety quartile. We compute the last two with gradient-boosted ensembles fitted
here on DBAASP, GRAMPA and HemoPI2 assay data, so the conditioning signal comes from
measured activity.

We sample 300,000 candidates, using only control combinations the training corpus contains
and reweighting toward the potent, non-hemolytic end of that set. We then build the library
by copying the (length, net charge, hydrophobic moment) histogram of the reference
antibacterial set without reweighting, and inside each cell we take one candidate per
sequence cluster in descending quality. Because the quotas fix the distribution and we
choose only within each cell, one library satisfies Phase 1's four metric families at once,
which otherwise conflict.

We rank the top-100 on what Phase 2 scores. Organizers draw 25 peptides at random and
average them, so we optimize expected value across the list and hold quality flat along it.
We weight Gram-negative potency double because 15 of the 20 panel strains are Gram-negative.
We treat hemolysis as a constraint because HC50 is reported only up to 128 µM and past that
ceiling a lower value scores the same. We penalize synthesis risk because a peptide that
fails synthesis is never replaced and enters the mean at its worst value.

Ranked entries clear the strict synthesis rules, a gate on predicted hemolysis, an exact
`Levenshtein.ratio <= 0.75` check against the 39,448 reference sequences, a ban on sharing
an exact 10-residue substring with them, and a cap of 0.65 on the Levenshtein ratio between
any two ranked sequences.

We vendor APEX-pathogen because it predicts micromolar MIC for 11 of the 20 panel strains
and comes from the laboratory running Phase 2. We rank-blend APEX with our ensembles at 45%
and penalize candidates where they disagree. We re-estimate hemolysis with an ESM-2 head
over the 3,552 candidates that reach the lists because it is our weakest prediction and the
selectivity category depends on it.

We apply no manual curation.

## How it works

### 1. Conditioned generation

`src/amp_designer/lm.py` is a 4-layer, 192-dimensional, 6-head decoder-only transformer
(1.8M parameters) over the 20 amino acids, with key/value caching for decoding. We prefix
sequences with four control tokens:

| control | bins | source |
|---|---|---|
| length | 7 | residue count |
| net charge | 7 | Henderson-Hasselbalch at pH 7.4, free termini |
| Gram-negative potency | 5 | quintile of this repo's MIC ensemble |
| hemolytic safety | 4 | quartile of this repo's P(HC50 >= 128 µM) |

`checkpoint/controls.npz` contains the 687 control combinations that appear in the corpus,
with mass reweighted by `exp(0.55 * potency_bin + 0.35 * safety_bin)`. We never request a
combination the model has not seen. We draw candidates at three temperatures (0.85, 1.0,
1.15) under nucleus sampling at `top_p=0.95`. The cool pass supplies high-likelihood
sequences and the warm pass supplies diversity.

### 2. Scoring

`checkpoint/scorers.pkl.gz` contains gradient-boosted ensembles fitted in
`training/train_scorers.py`. We measure accuracy with five-fold cross-validation grouped by
MMseqs2 cluster at 50% identity, as DBAASP is dense with analogue series. On the
Gram-negative head an ungrouped split reports Spearman 0.691 against the grouped 0.557, and
AUROC 0.844 against 0.777.

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

The hemolysis head takes ESM-2 t12 embeddings alongside the descriptors. Over the same
sequences and the same cluster-grouped splits, descriptors alone reach AUROC 0.755, ESM-2
alone 0.764, and descriptors with ESM-2 0.786. Embeddings alone lose to the combination
because the descriptors state charge and hydrophobicity where a masked language model only
implies them. We vendor `facebook/esm2_t12_35M_UR50D` (Lin et al., *Science* 2023, MIT
licensed) at float16 under `checkpoint/esm2_t12_35M/`, so generation fetches nothing.

Most potent DBAASP entries are C-terminally amidated and amidation raises potency
several-fold, but this competition requires free termini, so we made the amidation state a
model input and predict at `amidated=0`.

#### APEX-pathogen

We vendor four of the eight released APEX-pathogen checkpoints under `checkpoint/apex/` (Wan
et al., de la Fuente lab, *Nature Microbiology* 2025, MIT licensed). APEX predicts MIC in µM
against an 11-pathogen panel, and all 11 strains are on this competition's 20-strain panel:
the 5 Gram-positive strains and 6 of the 15 Gram-negative, including 4 of the 8 MDR
isolates. It is also the model that lab used to score the AMP-Diffusion baseline, and that
lab runs Phase 2.

APEX was trained by another group on other data, so it gives us a second opinion. On the 47
HydrAMP peptides with prospective wet-lab MIC values, it reaches Spearman 0.50 and AUROC
0.80 for active at <= 32 µM. Our ensemble appears to beat it there, but 32 of those 47
sequences are in APEX's training data, which makes that comparison leakage. APEX and our
ensemble agree at only Spearman 0.69, so we blend ranks 55:45 toward our models, which cover
all 20 strains and include a hemolysis head APEX does not have.

Four checkpoints rank 4,000 peptides at Spearman 0.983 against the full set, at 81 MB
against 220 MB.

We score in four stages, running the slow models only on the candidates that matter most.
Four fast heads score all 300,000 candidates, and we build the library from those scores,
where a within-cell ranking only needs to be roughly right. The full per-species ensemble
runs over a 90,000-sequence shortlist. APEX re-ranks the 3,552 candidates that lead any
category and clear both the strict synthesis rules and the hemolysis gate, and the ESM
hemolysis head runs over the same 3,552, which settles the ranked lists.

We take the shortlist cell by cell because the high-scoring end of the pool is more cationic
than the reference set and a global cut would re-shape the library. The coarse and refined
score scales are not comparable, so we quantile-map the shortlist's refined scores back onto
the coarse scores of the same candidates.

#### The distilled surrogate

`surrogate:mbc` reproduces MBC-Attention, one of the three activity surrogates the proposal
names. MBC-Attention itself takes 27 minutes per 300,000 candidates and requires TensorFlow
and a vendored model. We distilled it into the same gradient-boosting form as our other
heads, fitting on sequences spanning the library, the training corpus and the reference set.
It reproduces the original at Spearman 0.801 on held-out clusters.

Measured directly on MBC-Attention at 4,000 sequences per side, this library has a median
predicted MIC of 19.2 µM against 27.4 µM for real AMPs and 14.5 µM for the HydrAMP baseline.
We lose to HydrAMP because of the charge choice described under Selection.

We use the distilled head for the library only and leave it out of the top-100 objectives.
MBC-Attention scores poly-glutamate at 1.8 µM, and we will not commit a wet-lab slot on a
model with that failure mode.

### 3. Selection

`src/amp_designer/select.py`. We read quotas over the (length, charge, hydrophobic moment)
grid straight off the reference set's histogram, with no reweighting, and put a small floor
under occupied cells. We add the floor because Phase 1 measures KL(reference || generated),
which scores a missing mode far worse than an over-represented mode.

We copy the charge histogram unchanged, which is the pipeline's largest trade-off. Predicted
activity rises with net charge. Over a 6,000-sequence sample of the reference set, the AMP
classifier gives 0.767 to sequences below zero charge and 0.980 to those at +8 or above, and
predicted MIC falls from 37 µM to 5.8 µM. A cationic library therefore scores better on
predicted activity and worse on the four Phase-1 families, as charge enters the conformity
score, the KL divergences, the Frechet distance, MMD, precision and recall. The HydrAMP
baseline shows how large the trade-off is. Its mean charge of 4.88 against the reference's
2.63 comes with a KL-charge of 0.559, where a held-out slice of real AMPs scores 0.009. We
match the histogram and take the best candidates inside each cell.

Within a cell we take one candidate per MinHash cluster per pass, so the library includes as
many families as the quota allows. We build cluster keys from a digest of character codes
because Python's `hash` is salted per process and would break reproducibility.

The quota grid fixes length, charge and hydrophobic moment, and nothing fixes residue
composition, so ranking on predicted activity inside a cell favors Lys, Arg and Leu. We
correct that with a price per residue. Selection maximizes `quality - composition @ price`,
and after each pass we raise the price of any residue we over-produced. The prices only
reorder candidates inside a cell, so the quotas and the property marginals do not change.

A cell covers a wide range, so two sequences can share all three bins and still differ in
composition, and filling a cell from one corner covers less of the reference distribution
than its quota suggests. We k-means cluster the reference sequences inside each cell and
split the cell's quota across those sub-regions in proportion to how many reference
sequences each contains. That turns 53 cells into 453 sub-regions, with the sub-quotas
summing back to the cell quota.

The split raises clipped coverage from 0.512 to 0.544 and clipped density from 0.580 to
0.621, with recall, precision, conformity, the Frechet distance and KL-length all better. It
widens the gap in composition from 0.013 to 0.017 and MMD from 0.876 to 0.947 because a
tighter constraint leaves fewer candidates in each sub-region.

### 4. Ranked top-100

`generate.py:pick_top`. Candidates must first pass the strict rules in `synthesis.py`: no
Cys or Met, no Asp followed by G, A, S, T, C, R, D or N, no Asn-Gly, no QQ or NN, no
N-terminal Gln, no run over three identical or five beta-sheet-prone residues, at most four
basic residues in any five-residue window, beta-sheet formers under 42%, net charge at least
+2, GRAVY at most 1.0, and 11 to 26 residues. That file records how often a published AMP
would trip each rule.

We then check survivors against the 39,448 reference sequences with an exact
`Levenshtein.ratio`, drop any that share an exact 10-residue substring with them, cap each
MinHash cluster at three, and drop any candidate within 0.65 Levenshtein of a sequence we
have already picked.

The validator checks Levenshtein ratio, and the proposal states the same rule as MMseqs2
alignment identity. Those two measures differ, so we set our threshold at 0.75, under the
validator's 0.80. MMseqs2 reports identity over the aligned region, so one exact 10-residue
match inside a 25-mer scores 1.0 however different the rest is. Banning shared 10-mers
closes that gap without making us run an aligner at generation time. The rule drops 3% of
our ranked candidates, where it would drop 54% of real AMPs measured against the rest of the
reference set.

On that same comparison a held-out slice of real AMPs reaches MMseqs2 identity 0.875 and
Levenshtein 0.900 against the reference, where these ranked lists are at 0.60 and 0.67. The
identity hits that remain above 0.80 are short local alignments, median 11 residues over
half the query, with 3 of 129 reaching 80% query coverage. The real-AMP hits cover the query
completely.

The pairwise cap guards against a shared failure. The models separate active from inactive
far better than they rank among the active, so 25 peptides drawn from a list built on one
scaffold can fail together for the same reason. Real AMPs have a median pairwise ratio of
0.258, so a cap of 0.65 only catches near-copies.

We drop from the ranked lists any candidate whose predicted probability of HC50 >= 128 µM
falls below 0.50. Ranking on potency alone favors hemolytic sequences because potency and
hemolysis both rise with charge and hydrophobicity. A hard cut is appropriate here because
the models rank potency better than they rank hemolysis, and a severely hemolytic peptide is
a wasted wet-lab slot in all five categories.

Where our ensemble and APEX disagree about a candidate, we reduce its blended score. Taking
the top of a noisy score over-represents candidates whose error happened to run favorably,
and a peptide that only one of two independent models rates highly is usually one of those.
Lowering their scores improves the average of the 25 peptides organizers draw, though it
also demotes some potent peptides.

## Phase-1 metrics

We measure with `seqme` 0.5.1 against the challenge's reference antibacterial set, using a
held-out slice of that set to show what real AMPs score. The baseline column is the HydrAMP
library shipped in its starter kit. Embeddings are ESM-2 t6_8M, one of the embedding models
the competition names, at 4,000 sequences per side.

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

The conformity score and the diversity land on the held-out reference set's value, while the
sequences stay novel by exact match. Recall, clipped density and clipped coverage are four
to five times the baseline's, as the quota grid covers the reference distribution where the
baseline occupies one corner of it, and the KL divergences follow for the same reason.

We leave FKEA out of the table because at 4,000 sequences it is near its sample-size ceiling
for the libraries we scored and separates nothing, an instability the seqme authors report.

We repeated the table with a 1-mer and 2-mer frequency embedding in place of ESM-2. No
ranking changed except precision, which reads 0.876 against the baseline's 0.918 there. That
reading is an artifact of our novelty term, which penalizes reuse of reference 6-mers, so a
k-mer embedding measures what the term suppresses.

## The ranked lists

Each of the five lists has 100 sequences, all in the library and predicted active at 16 µM
or below, none sharing an exact 10-mer with the reference set or standing above a
Levenshtein ratio of 0.75 to it, and no pair above 0.65 to another.

The proposal and the template say the 25 assayed peptides come from the top 100, and the FAQ
says the top 50. We hold quality flat across all 100 so the lists work under either reading.
The broad-spectrum list has a median predicted MIC of 7.0 µM over ranks 1 to 50 and 8.0 µM
over ranks 51 to 100, with predicted safety at 0.65 and 0.61.

| list | predicted MIC50 | P(HC50 >= 128) | worst entry | predicted window | envelope |
|---|---|---|---|---|---|
| broad spectrum | 7.65 µM | 0.63 | 0.50 | 12.3 | 0.86 |
| Gram-negative | 7.23 µM | 0.63 | 0.50 | 12.6 | 0.87 |
| Gram-positive | 8.06 µM | 0.63 | 0.50 | 11.4 | 0.79 |
| MDR | 7.40 µM | 0.63 | 0.50 | 12.0 | 0.81 |
| selectivity | 8.06 µM | 0.64 | 0.50 | 12.7 | 0.89 |

Envelope is `synthesis.py:envelope_score`, which measures how well a sequence's descriptors
match measured peptides that are both potent and non-hemolytic. The hemolysis figures come
from the descriptor head while selection used the ESM head, so they are an independent
check.

The five lists share between 31 and 64 sequences pairwise and together cover 225 distinct
peptides across the 500 slots. The selectivity list has both the highest predicted window
and the highest hemolytic safety.

One draw of 25 peptides is scored in all five categories, so only the list a validator reads
is assayed. `generate` and `generate_broad_spectrum` produce the same list, so that is the
one tested whichever validator runs. It sits within 4% of the best of the five on mean
predicted MIC50, safety window and hemolytic safety, so ranking per category does no harm to
the list that is tested.

## What the models get wrong

We scored nine peptides with published free-termini MIC and human-erythrocyte HC50, spanning
selectivity indices from 15 to 111, against poly-Ala, poly-Glu and a scrambled sequence. The
peptides outrank the decoys, and our MIC predictions land within five-fold of the measured
values.

Our hemolysis head remains the weakest after the ESM-2 rebuild. Over those nine its
predicted safety window correlates with the measured index at Spearman 0.25, and we predict
50 µM for dhvar5 (`LLLFLLKKRKKRKY`), the most selective control, against a measured 120 µM.
This pipeline would have excluded it because its N-terminal `LLLFLL` block trips the
six-residue beta-sheet run rule. Short cationic peptides that carry their hydrophobicity in
one contiguous block are where the head fails. That 0.25 understates the head because we
chose those peptides for high selectivity and their measured indices span a narrow range.
Across the full range the head reaches AUROC 0.786.

The ranked lists are also concentrated in one structural class. Entries are cationic
amphipathic sequences of the helix-forming kind, built from Lys and Arg with Ile, Leu, Val
and Trp, and no Cys or Met. That class is the best validated, and our training data is
densest in it. If it fails systematically, the peptides in the draw fail with it, so we
spread each list of 100 over 99 or 100 distinct MinHash clusters with no pair above a 0.65
Levenshtein ratio.

Amphipathic beta-sheet designs were the second class we considered. We tested the beta-sheet
hydrophobic moment against measured HC50 on 465 potent peptides, found the relationship weak
and not monotonic, and added none.

The envelope's Arg-share term and the fitted models put Arg at 43% of the cationic residues
across the ranked lists, inside the 0 to 50% band where the measured safety window peaks.

## Reproducing the checkpoints

The entry point runs from the committed checkpoints, and this section rebuilds them.

```bash
git clone https://github.com/szczurek-lab/amp-challenge-2027       ../amp-challenge-2027
git clone https://github.com/szczurek-lab/battleamp-snakemake      ../battleamp-snakemake
git clone https://github.com/szczurek-lab/hydramp-starter-kit      ../hydramp-starter-kit
git clone https://github.com/raghavagps/HemoPI2                    ../HemoPI2

uv sync --extra training
AMP_UPSTREAM=.. uv run python -m training.build_data     # writes data/training/
uv run python -m training.train_scorers                  # writes checkpoint/scorers.pkl.gz
uv run python -m training.train_hemolysis_esm            # adds the ESM-2 hemolysis heads
uv run python -m training.train_lm                       # writes checkpoint/peptide_lm.pt
```

`training/build_controls.py` rebuilds `checkpoint/controls.npz` on its own, with different
tilts toward potency and hemolytic safety, without retraining the language model.

`train_hemolysis_esm.py` adds the `hem_esm:` heads to the bundle `train_scorers.py` wrote,
so run it second. Both need MMseqs2 on `PATH` for the grouped cross-validation splits, and
the entry point does not.

## Training data

All sources are public, and `data/training/README.md` lists them with licenses, row counts
and what they contribute.

## Layout

```
src/amp_designer/
  lm.py          conditioned transformer, KV-cached decoding
  features.py    physicochemical descriptors
  scoring.py     MIC / hemolysis / AMP-classifier inference
  esm.py         vendored ESM-2 encoder for the hemolysis head
  apex.py        vendored APEX-pathogen inference
  apex_models.py the module APEX checkpoints were pickled against
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
| `--category` | from the script name | which Phase-2 objective ranks the top-k |
| `--out-dir` | named after the script | where to write the two FASTA files |

## Use of AI assistants

The code in this repository was written with Claude (Anthropic) under human direction. The
method design, the data sources and the submitted content are the author's responsibility.

## License

MIT, see [LICENSE](LICENSE).
