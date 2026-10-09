# `output_data/` — per-allele score tables, model predictions, and everything the scripts write

## Provided: the per-allele score tables

One file per allele, holding that allele's peptides and the comparison-model
scores they are judged against.

| File | Read by |
|---|---|
| `<HLA>_validation_final.csv` | `select_epoch_from_validation.py` |
| `<HLA>_test_final.csv` | `evaluate_on_test_set.py` |

| Column | Contents |
|---|---|
| `Peptide` | Peptide sequence. |
| `Original` | The epitope group the peptide belongs to: the true epitope plus the decoys tiled across it. Performance is measured as the rank of the true epitope within its own group. |
| `score` | 1 = true epitope, 0 = decoy. |
| `GeoMean_letter` | MHCIIFold-GNN score (amino-acid identity model). |
| `GeoMean_conf` | MHCIIFold-GNN score (confidence model). |
| `Geomean_FCM` | FCM comparison-model score. |
| `Rank` | NetMHCIIpan percentile rank (lower = stronger binder). |

The two panels are disjoint: the validation alleles choose the epochs, the test
alleles are never used for any choice.

**Score polarity:** model scores are **low for binders** and high for
non-binders (the network is trained against `1 - y`). The ranking in
`compute_ratios` assumes this throughout.

---

## Provided: the released models' predictions

One file per trained model, already at that model's selected epoch:

```
netmhciipan_model_<II>_14_pepseq_hlas_fixed_20netmhciipan_hlas_hd<HD>.csv
```

Fifteen files, `<II>` = 01..15. Written by `netmhciipan_test.py`, read by
`evaluate_on_test_set.py`, which averages their `predicted_score` into the
ensemble. See `../saved_models/README.md` for which epoch each one is and what
`<II>`/`<HD>` mean.

| Column | Contents |
|---|---|
| `Peptide` | Peptide sequence. |
| `real_HLA` | Allele name, before it was replaced by its pseudosequence. |
| `predicted_score` | Raw sigmoid output of the model. Not a percentile — the evaluation scripts rank-normalise it per allele. |
| `Original`, `score` | Carried over from the external validation/test set. |

These cover every allele; validation and test are split by `real_HLA`, not by
file.

---

## Written by the scripts

None of these has to be provided; they appear here when the pipeline is run.

| Output | Written by | Contents |
|---|---|---|
| `test_netmhciipan/test_results_netmhciipan_n20.pkl` | `evaluate_on_test_set.py` | The ensemble's summary table (mean and 95 % CI per model) and the per-epitope ratios behind it, on the held-out test alleles. Also records which models and epochs went into the ensemble. |

Retraining adds, per trained run:

| Output | Written by | Contents |
|---|---|---|
| `<run_name>/<run_name>_epoch<E>.pt` | `netmhciipan_code.py` | Whole pickled model (not a `state_dict`), every 10th epoch. |
| `<run_name>/<run_name>_epoch<E>_test_predictions.csv` | `netmhciipan_test.py` | Per-peptide `predicted_score` of one checkpoint, all alleles. |
| `validation_netmhciipan/validation_results_nmpmodel<II>_nmpn<N>_nmpepoch<E>.pkl` | `select_epoch_from_validation.py` | Summary table for one candidate epoch of one run. |
| `validation_netmhciipan/validation_everything_df_nmpmodel<II>_nmpn<N>_nmpepoch<E>.pkl` | `select_epoch_from_validation.py` | Per-epitope ratios behind that table. |

where

```
<run_name> = netmhciipan_model_<II>_14_pepseq_hlas_fixed_<N>netmhciipan_hlas_hd<HD>_seed<S>_new_split
```

The per-epoch checkpoints and prediction CSVs are **not** part of the release —
they are tens of files per run. `select_epoch_from_validation.py` takes
`--run-dir` so it can be pointed at wherever they were written.
