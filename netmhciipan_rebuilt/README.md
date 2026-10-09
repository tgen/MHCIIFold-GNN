# NetMHCIIpan-rebuilt

A reimplementation of a NetMHCIIpan-style predictor of peptide binding to MHC
class II alleles, trained on in-house PepSeq binding data together with the
NetMHCIIpan 4.3 binding-affinity data, and evaluated against MHCIIFold-GNN,
FCM and NetMHCIIpan itself on a held-out panel of alleles.

A peptide is cut into overlapping 9-residue cores. Each core is scored on its
own from a BLOSUM62 encoding of the core, its flank lengths, and the allele's
34-residue pseudosequence; the peptide's score is the geometric mean over its
cores. Because the allele enters only as a pseudosequence, a trained model can
score alleles it never saw. See `saved_models/README.md` for the exact
architecture.

> **Score polarity:** training uses binder → 0, non-binder → 1, so model output
> is **low for binders** and high for non-binders. Every script here assumes
> that.

The released predictor is an **ensemble of 15 models** — three hidden widths
(20, 40, 60) × five random seeds — each taken at the epoch chosen on the
validation alleles.

## Layout

```
.
├── netmhciipan_code.py              training
├── netmhciipan_test.py              scoring with a trained checkpoint
├── select_epoch_from_validation.py  epoch selection (retraining only)
├── evaluate_on_test_set.py          held-out test evaluation
├── input_data/                      everything the scripts read   → DATA_FILES.md
├── output_data/                     per-allele tables, predictions, results → README.md
└── saved_models/                    the 15 released weight files  → README.md
```

Each folder has its own README describing every file in it.

## The pipeline

Run in this order. Stages 1–3 are only needed to retrain; the release ships
their results.

### 1. `netmhciipan_code.py` — training

```
python netmhciipan_code.py <model_index> <number_of_netmhciipan_hlas>
```

`model_index` (1–15) selects the hidden width and seed. The allele panel is 14
fixed PepSeq DRB1 alleles plus `number_of_netmhciipan_hlas` NetMHCIIpan
alleles; the released models use 20. A checkpoint is written every 10 epochs,
up to 500.

The flag `SKIP_MHCIIFOLD_GNN_BA_DATA` at the top of the file controls whether
`netmhciipan_ba_data_used_in_mhciifold_gnn_training.csv` contributes training
rows. `True` (the default) reproduces the released models.

### 2. `netmhciipan_test.py` — scoring

```
python netmhciipan_test.py <model_index> <number_of_netmhciipan_hlas>
python netmhciipan_test.py <saved_model.pt>
```

Scores the external validation/test peptides. The first form scores **every**
checkpoint of a run, writing one `*_epoch<E>_test_predictions.csv` per epoch;
the second scores a single checkpoint.

### 3. `select_epoch_from_validation.py` — epoch selection

```
python select_epoch_from_validation.py --model-index I --netmhciipan-n N [--run-dir DIR]
```

The only place validation labels influence a choice. Every checkpoint of one
run is evaluated on the validation alleles and the best epoch is printed, ready
to paste into `SELECTED_EPOCHS`. The per-epoch prediction CSVs are not part of
the release, so `--run-dir` points this at wherever stage 2 wrote them.

### 4. `evaluate_on_test_set.py` — held-out test evaluation

```
python evaluate_on_test_set.py
```

No arguments, and nothing is selected: the 15 models and their epochs are fixed
in `SELECTED_EPOCHS`. Their raw `predicted_score` values are averaged per
peptide, rank-normalised per allele, and evaluated on the held-out test alleles
alongside the comparison models. **This is the script that reproduces the
published numbers**, and it runs from the files already in the repo.

## How performance is measured

Not a pooled AUC. For each known epitope, the model ranks it against the decoy
peptides tiled across the same `Original` group:

```
ratio = (# peptides in the group scoring at least as well as the true epitope)
        / (group size)
```

reported as `1 - ratio`, so it reads like an AUC (higher is better), aggregated
across epitopes with a 95 % confidence interval.

Two corrections, both in `compute_ratios()`:

- other known epitopes of the same allele (from
  `input_data/known_epitopes_per_allele.csv`) are dropped from a group, so a
  real epitope is never counted as a decoy for another;
- a group holding more than one positive is split by
  `sequence_reconstructer()`, which chains the overlapping kmers back into
  contiguous stretches so each is ranked on its own.

The same measure is computed for NetMHCIIpan, FCM, MHCIIFold-GNN and the
MHCIIFold-GNN + NetMHCIIpan combination, so every number is reported next to
the baselines it is judged against.

