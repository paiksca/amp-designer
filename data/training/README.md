# Training data disclosure

All sources are public, so the competition's data rules ask for no additional release.
`build_data.py` assembles the files in this directory from the repositories listed below.
Re-run it to rebuild them.

## Assembled files

| file | rows | what it is |
|---|---|---|
| `amp_positives.txt` | 44,585 | known antibacterial peptides, 8-50 residues, canonical alphabet, deduplicated. Trains the language model. |
| `amp_negatives.txt` | 15,821 | sequences assumed non-antimicrobial, same length window, disjoint from the positives. Trains the AMP classifier. |
| `mic.csv.gz` | 137,786 | one row per (sequence, target species) with MIC converted to µM, a censoring flag for assay-ceiling values, and a C-terminal amidation flag. 12,996 distinct sequences. |
| `challenge_reference.txt` | 39,448 | the challenge's reference set, copied for convenience. Not training data. |
| `slay.csv.gz` | 438,484 | SLAY display-screen growth-inhibition ratios against *E. coli*. Assembled and committed, used by no shipped checkpoint. |

HemoPI2 is GPL-3.0 licensed, so we read its hemolysis data at training time and keep it out
of this repository. Its HC50 values come from DBAASP and Hemolytik, both public. We pool
them with the direct DBAASP harvest, which adds 370 sequences HemoPI2 does not have. The two
agree at Spearman 0.916 on the 1,306 they share, which is our check that they measure the
same thing.

The direct harvest is why the MIC table is 65% larger in distinct sequences than the
redistributed snapshot: 12,996 against 7,872, with 4,218 of the new ones on panel species.
`training/dbaasp_harvest.py` fetches and flattens the records. The flat table is 17 MB and
stays out of this repository. Rebuild it with that script and point `AMP_DBAASP` at the
result.

## Upstream sources

| source | license | used for |
|---|---|---|
| [DBAASP](https://dbaasp.org) (Pirtskhalava et al., *NAR* 49:D288, 2021), harvested directly from the detail API | CC BY 4.0 | MIC per species and human-erythrocyte HC50, with terminal modifications and chemistry flags. All 25,542 records. |
| [DBAASP v3](https://dbaasp.org) snapshot redistributed in `szczurek-lab/battleamp-snakemake` | CC BY 4.0 | the same database at an earlier snapshot, kept because its species labels are already normalized |
| [GRAMPA](https://github.com/zswitten/Antimicrobial-Peptides) (Witten & Witten), via `szczurek-lab/hydramp-starter-kit` | MIT | MIC against *E. coli*, as log10(MIC/µM) |
| [APD3 / APD6](https://aps.unmc.edu) (Wang et al., *NAR* 54:D363, 2026) | free for academic use | known-AMP sequences, via the challenge reference set and dbAMP pooling |
| [DRAMP](http://dramp.cpu-bioinfor.org) | free for academic use | known-AMP sequences |
| [dbAMP 3.0](https://awi.cuhk.edu.cn/dbAMP) (Yao et al., *NAR* 53:D364, 2025) | free for academic use | known-AMP sequences |
| [MarLys / MLAMP](https://doi.org/10.17632/w4hb5grjwb.3) (Marczak et al., 2026) | CC0 | the challenge reference set `data/antibacterial.fasta`, which aggregates thirteen primary AMP databases |
| [AMP Scanner v2](https://www.dveltri.com/ascan/) (Veltri et al., 2018), via `szczurek-lab/hydramp-starter-kit` | free for academic use | AMP / non-AMP classifier training set |
| [UniProt](https://www.uniprot.org) | CC BY 4.0 | non-AMP negatives |
| [HemoPI2](https://github.com/raghavagps/HemoPI2) (Rathore et al., *Commun Biol* 8:176, 2025) | GPL-3.0 (code), data from DBAASP and Hemolytik | HC50 values in µM |
| [SLAY](https://doi.org/10.1038/s41467-018-08181-y), via `szczurek-lab/battleamp-snakemake` | CC BY 4.0 | display-screen activity, assembled but unused |
| [ESM-2](https://huggingface.co/facebook/esm2_t12_35M_UR50D) `esm2_t12_35M_UR50D` (Lin et al., *Science* 379:1123, 2023) | MIT | protein language model embeddings for the hemolysis head, vendored under `checkpoint/esm2_t12_35M/` |
| [APEX-pathogen](https://doi.org/10.1038/s41564-024-01907-3) (Wan et al., *Nature Microbiology* 2025) | MIT | MIC prediction over 11 pathogens, four checkpoints vendored under `checkpoint/apex/` |
| `szczurek-lab/hydramp-starter-kit` `experimental/` | MIT | 31 HydrAMP peptides with measured HC50, added to the hemolysis set |

We copy the challenge reference set at `data/antibacterial.fasta` verbatim from the
challenge template and use it only for the overlap and novelty checks, never as training
data.

## Two label-quality traps in DBAASP

We measured both, and both lowered accuracy.

**Potency measures are not interchangeable.** A peptide's `targetActivities` list mixes MIC
with MBC, IC50, MFC, LC, LD50 and EC50, and with MIC50 and MIC90, which are statistics over
a panel where MIC is a per-strain value. In a 4,000-record sample the split was 22,803 MIC
against roughly 6,900 of everything else. Pooling them dropped the Gram-negative model from
AUROC 0.801 to 0.750 even though it added 59% more sequences. `dbaasp_harvest.py` keeps
`activityMeasureGroup == "MIC"` and nothing else.

**A censored HC50 is a lower bound, not a measurement.** Most non-hemolytic peptides are
recorded as ">100 µM" or similar, and dropping them throws away the safe class while
treating them as measurements understates safety. We clip values censored at or above the
competition's 128 µM ceiling to the ceiling, where the scale ends anyway, and drop values
censored below it as ambiguous. That leaves 5,747 sequences with a usable hemolysis label
against 1,957 from HemoPI2 alone.

We keep only human-erythrocyte hemolysis records. Sheep, rabbit and horse red cells differ
in sensitivity, and Phase 2 measures human cells.

## Preprocessing

- We upper-case sequences, restrict them to `ACDEFGHIKLMNPQRSTVWY`, and keep only 8 to 50
  residues, matching the competition's rules.
- We convert MIC concentrations given in µg/ml to µM using the molecular weight computed
  from the sequence with average residue masses plus water.
- We record a concentration written as `>X` at X with `censored = True`, meaning the assay
  never reached inhibition.
- We take the median of replicate measurements per (sequence, species, amidation) and cap
  MIC at the competition's 64 µM ceiling before the log2 transform.
- We average hemolysis values in log space per sequence and cap HC50 at the competition's
  128 µM ceiling.
