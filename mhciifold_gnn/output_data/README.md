# `output_data/` — per-allele score tables, and everything the scripts write

## Provided: the per-allele score tables

One file per allele, holding that allele's peptides and the comparison-model
scores they are judged against.

| File | Read by |
|---|---|
| `<HLA>_validation_final.csv` | `select_epochs_from_validation.py` |
| `<HLA>_test_final.csv` | `evaluate_on_test_set.py` |

| Column | Contents |
|---|---|
| `Peptide` | Peptide sequence. |
| `Original` | The epitope group the peptide belongs to: the true epitope plus the decoys tiled across it. Performance is measured as the rank of the true epitope within its own group. |
| `score` | 1 = true epitope, 0 = decoy. |
| `GeoMean_letter` | Earlier MHCIIFold-GNN score (amino-acid identity model). |
| `GeoMean_conf` | Earlier MHCIIFold-GNN score (confidence model). |
| `Geomean_FCM` | FCM comparison-model score. |
| `Rank` | NetMHCIIpan percentile rank (lower = stronger binder). |

**Score polarity:** model scores are **low for binders** and high for
non-binders (the network is trained against `1 - y`).

---

## Written by the scripts

None of these has to be provided; they appear here when the pipeline is run.

| Output | Written by | Contents |
|---|---|---|
| `<HLA>_epitope_test_released_<aa_set>.parquet` | `test_mhciifold_gnn.py` | Per-peptide score (`GeoMean`) of a released model on one allele. |
| `test_14_pepseq_hlas_fixed/test_results_nhlas<N>_seed<S>.pkl` | `evaluate_on_test_set.py` | Summary table (mean and 95 % CI per model) and per-epitope ratios on the held-out test alleles. |

Retraining adds, for each `aa_set` and seed:

| Output | Written by | Contents |
|---|---|---|
| `netmhciipan_train_included_..._seed<S>/train_GNN_model_epoch<E>_regular22_baseline_14_pepseq_hlas_fixed.pth` | `train_mhciifold_gnn.py` | Whole pickled model (not a `state_dict`), every 10th epoch. |
| `<same directory>/<HLA>_epitope_test_epoch<E>_regular22_baseline_<aa_set>.parquet` | `test_mhciifold_gnn.py --all-epochs` | Per-peptide score of one checkpoint on one allele. |
| `validation_14_pepseq_hlas_fixed/validation_results_nhlas<N>_seed<S>.pkl` | `select_epochs_from_validation.py` | Summary table for every (all_aa epoch, only_HLA_aa epoch) pair. |
| `validation_14_pepseq_hlas_fixed/validation_everything_df_nhlas<N>_seed<S>.pkl` | `select_epochs_from_validation.py` | Per-epitope ratios behind those tables. |
