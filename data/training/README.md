# Training data disclosure

Every source is public. Nothing proprietary or non-public was used, so nothing additional
needs releasing under the competition's data rules. `build_data.py` assembles the files in
this directory from the repositories listed below; re-run it to rebuild them.

## Assembled files

| file | rows | what it is |
|---|---|---|
| `amp_positives.txt` | 44,585 | known antibacterial peptides, 8-50 residues, canonical alphabet, deduplicated. Trains the language model. |
| `amp_negatives.txt` | 15,821 | sequences assumed non-antimicrobial, same length window, disjoint from the positives. Trains the AMP classifier. |
| `mic.csv.gz` | 36,312 | one row per (sequence, target species) with MIC converted to µM, a censoring flag for assay-ceiling values, and a C-terminal amidation flag. 7,872 distinct sequences. |
| `slay.csv.gz` | 438,484 | SLAY display-screen growth-inhibition log ratios against *E. coli*. Assembled and disclosed; not used by the shipped checkpoints. |
| `challenge_reference.txt` | 39,448 | the challenge's own reference set, copied for convenience. Not training data. |

Hemolysis data is read directly from the HemoPI2 release at training time and is not
redistributed here, because that repository is GPL-3.0 licensed. The underlying HC50
values come from DBAASP and Hemolytik, both public. 1,957 sequences after cleaning.

## Upstream sources

| source | licence | used for |
|---|---|---|
| [DBAASP v3](https://dbaasp.org) (Pirtskhalava et al., *NAR* 49:D288, 2021), via `szczurek-lab/battleamp-snakemake` | CC BY 4.0 | MIC values per species and strain; peptide sequences and terminal modifications |
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

## Preprocessing

- Sequences are upper-cased, restricted to `ACDEFGHIKLMNPQRSTVWY`, and kept only at 8 to 50
  residues, matching the competition's own rules.
- MIC concentrations given in µg/ml are converted to µM using the molecular weight computed
  from the sequence with average residue masses plus water.
- A concentration written as `>X` is recorded at X with `censored = True`; it means the
  assay never reached inhibition.
- Replicate measurements are reduced to the median per (sequence, species, amidation), and
  MIC is capped at the competition's 64 µM ceiling before the log2 transform.
- Hemolysis values are averaged in log space per sequence, and HC50 is capped at the
  competition's 128 µM ceiling.
