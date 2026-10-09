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
| `GeoMean_letter` | MHCIIFold-GNN peptide agnostic score. |
| `GeoMean_conf` | MHCIIFold-GNN peptide aware score. |
| `Geomean_FCM` | FCM score. |
| `Rank` | NetMHCIIpan percentile rank (lower = stronger binder). |

**Score polarity:** every score here reads the same way — **low for binders**,
high for non-binders. The network is trained against `1 - y`, and
NetMHCIIpan's `Rank` is a percentile, so the comparison is consistent.

---

## Written by the scripts

None of these has to be provided; they appear here when the pipeline is run.

| Output | Written by | Contents |
|---|---|---|
| `<HLA>_epitope_test_released_<aa_set>.parquet` | `test_mhciifold_gnn.py` | One row per peptide: `AA` (the peptide), `GeoMean` (that model's score, low = binder), and the `HLA` / `Original` / `score` columns carried over from `external_validation_and_test_set.csv`. One file per allele per model. |
| `test_14_pepseq_hlas_fixed/test_results_nhlas<N>_seed<S>.pkl` | `evaluate_on_test_set.py` | A dict with `summary` (mean and 95 % CI per model), `everything` (the per-epitope ratios behind it), the alleles covered and the models used. The summary is printed as well. |

`<HLA>` in these names never carries the `DRB1_` prefix, which is how
`evaluate_on_test_set.py` looks them up.

Retraining adds, for each `aa_set` and seed:

| Output | Written by | Contents |
|---|---|---|
| `netmhciipan_train_included_..._seed<S>/train_GNN_model_epoch<E>_regular22_baseline_14_pepseq_hlas_fixed.pth` | `train_mhciifold_gnn.py` | Whole pickled model (not a `state_dict`), every 10th epoch. |
| `<same directory>/<HLA>_epitope_test_epoch<E>_regular22_baseline_<aa_set>.parquet` | `test_mhciifold_gnn.py --all-epochs` | Per-peptide score of one checkpoint on one allele. |
| `validation_14_pepseq_hlas_fixed/validation_results_nhlas<N>_seed<S>.pkl` | `select_epochs_from_validation.py` | Summary table for every (all_aa epoch, only_HLA_aa epoch) pair. |
| `validation_14_pepseq_hlas_fixed/validation_everything_df_nhlas<N>_seed<S>.pkl` | `select_epochs_from_validation.py` | Per-epitope ratios behind those tables. |
