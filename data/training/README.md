# Training data disclosure

Every source is public. Nothing proprietary or non-public was used, so nothing additional
needs releasing under the competition's data rules. `build_data.py` assembles the files in
this directory from the repositories listed below. Re-run it to rebuild them.

## Assembled files

| file | rows | what it is |
|---|---|---|
| `amp_positives.txt` | 44,585 | known antibacterial peptides, 8-50 residues, canonical alphabet, deduplicated. Trains the language model. |
| `amp_negatives.txt` | 15,821 | sequences assumed non-antimicrobial, same length window, disjoint from the positives. Trains the AMP classifier. |
| `mic.csv.gz` | 137,819 | one row per (sequence, target species) with MIC converted to µM, a censoring flag for assay-ceiling values, and a C-terminal amidation flag. 12,993 distinct sequences. |
| `challenge_reference.txt` | 39,448 | the challenge's own reference set, copied for convenience. Not training data. |

`build_data.py` also assembles a 438,484-row table of SLAY display-screen growth-inhibition
ratios against *E. coli*. No shipped checkpoint uses it, so it is not committed here. Re-run
`build_data.py` to produce it.

Hemolysis data is read directly from the HemoPI2 release at training time and is not
redistributed here, because that repository is GPL-3.0 licensed. The underlying HC50
values come from DBAASP and Hemolytik, both public. Pooled with the direct DBAASP
harvest, which adds 370 sequences HemoPI2 does not carry. The two agree at Spearman 0.916
on the 1,306 they share, which is the check that they measure the same thing before being
pooled.

The direct harvest is why the MIC table is 65% larger in distinct sequences than the
redistributed snapshot: 12,993 against 7,872, with 4,218 of the new ones on panel species.
`work/data/dbaasp/harvest.py` fetches the records and `parse.py` flattens them. The flat
table is 17 MB and is not committed. Rebuild it with those two scripts and point
`AMP_DBAASP` at the result.

## Upstream sources

| source | licence | used for |
|---|---|---|
| [DBAASP](https://dbaasp.org) (Pirtskhalava et al., *NAR* 49:D288, 2021), harvested directly from the detail API | CC BY 4.0 | MIC per species and human-erythrocyte HC50, with terminal modifications and chemistry flags. All 25,542 records. |
| [DBAASP v3](https://dbaasp.org) snapshot redistributed in `szczurek-lab/battleamp-snakemake` | CC BY 4.0 | the same database at an earlier snapshot, kept because its species labels are already normalised |
| [GRAMPA](https://github.com/zswitten/Antimicrobial-Peptides) (Witten & Witten), via `szczurek-lab/hydramp-starter-kit` | MIT | MIC against *E. coli*, as log10(MIC/µM) |
| [APD3 / APD6](https://aps.unmc.edu) (Wang et al., *NAR* 54:D363, 2026) | free for academic use | known-AMP sequences, via the challenge reference set and dbAMP pooling |
| [DRAMP](http://dramp.cpu-bioinfor.org) | free for academic use | known-AMP sequences |
| [dbAMP 3.0](https://awi.cuhk.edu.cn/dbAMP) (Yao et al., *NAR* 53:D364, 2025) | free for academic use | known-AMP sequences |
| [MarLys / MLAMP](https://doi.org/10.17632/w4hb5grjwb.3) (Marczak et al., 2026) | CC0 | the challenge reference set `data/antibacterial.fasta`, which aggregates thirteen primary AMP databases |
| [AMP Scanner v2](https://www.dveltri.com/ascan/) (Veltri et al., 2018), via `szczurek-lab/hydramp-starter-kit` | free for academic use | AMP / non-AMP classifier training set |
| [UniProt](https://www.uniprot.org) | CC BY 4.0 | non-AMP negatives |
| [HemoPI2](https://github.com/raghavagps/HemoPI2) (Rathore et al., *Commun Biol* 8:176, 2025) | GPL-3.0 (code); data from DBAASP and Hemolytik | HC50 values in µM |
| [SLAY](https://doi.org/10.1038/s41467-018-08181-y), via `szczurek-lab/battleamp-snakemake` | CC BY 4.0 | display-screen activity, assembled but unused |
| `szczurek-lab/hydramp-starter-kit` `experimental/` | MIT | 31 HydrAMP peptides with measured HC50, added to the hemolysis set |

The challenge reference set at `data/antibacterial.fasta` is copied verbatim from the
challenge template and is used only for the overlap and novelty checks, not as training
data.

## Two label-quality traps in DBAASP

Both were measured rather than assumed, and both cost real accuracy when they were
present.

**Potency measures are not interchangeable.** A peptide's `targetActivities` list mixes MIC
with MBC, IC50, MFC, LC, LD50 and EC50, and with MIC50 and MIC90, which are panel
statistics rather than per-strain values. In a 4,000-record sample the split was 22,803 MIC
against roughly 6,900 of everything else. Pooling them dropped the Gram-negative model from
AUROC 0.801 to 0.750 even though it added 59% more sequences. `dbaasp_harvest.py` keeps
`activityMeasureGroup == "MIC"` and nothing else.

**A censored HC50 is a lower bound, not a measurement.** Most non-hemolytic peptides are
recorded as ">100 µM" or similar, and dropping them throws away the safe class while
treating them as measurements understates safety. Values censored at or above the
competition's 128 µM ceiling are clipped to it, since the ceiling is where the scale ends
anyway. Values censored below it are ambiguous and are dropped rather than guessed. That
leaves 5,747 sequences with a usable hemolysis label against 1,957 from HemoPI2 alone.

Hemolysis records are also restricted to human erythrocytes. Sheep, rabbit and horse red
cells differ in sensitivity, and Phase 2 measures human cells.

## Preprocessing

- Sequences are upper-cased, restricted to `ACDEFGHIKLMNPQRSTVWY`, and kept only at 8 to 50
  residues, matching the competition's own rules.
- MIC concentrations given in µg/ml are converted to µM using the molecular weight computed
  from the sequence with average residue masses plus water.
- A concentration written as `>X` is recorded at X with `censored = True`, meaning the
  assay never reached inhibition.
- Replicate measurements are reduced to the median per (sequence, species, amidation), and
  MIC is capped at the competition's 64 µM ceiling before the log2 transform.
- Hemolysis values are averaged in log space per sequence, and HC50 is capped at the
  competition's 128 µM ceiling.
