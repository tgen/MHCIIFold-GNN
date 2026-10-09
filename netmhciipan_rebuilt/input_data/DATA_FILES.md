# Data files

Every file the scripts read. Nothing here is produced by the pipeline — these
are the inputs.

| File | Read by |
|---|---|
| `pepseq_drb1_binding_data.csv` | `netmhciipan_code.py` |
| `netmhciipan_ba_peptides_full.csv` | `netmhciipan_code.py` |
| `netmhciipan_ba_data_used_in_mhciifold_gnn_training.csv` | `netmhciipan_code.py` |
| `hla_pseudosequences_2023.dat` | `netmhciipan_code.py`, `netmhciipan_test.py` |
| `external_validation_and_test_set.csv` | `netmhciipan_test.py` |
| `known_epitopes_per_allele.csv` | `select_epoch_from_validation.py`, `evaluate_on_test_set.py` |

## What each file is

### `pepseq_drb1_binding_data.csv`

In-house PepSeq binding data (DRB1 alleles). Columns used: peptide, HLA, status (binder/nonbinder). The PepSeq portion of the training set.

### `netmhciipan_ba_peptides_full.csv`

NetMHCIIpan 4.3 binding-affinity training peptides, full set. Columns used: peptide, HLA, status. Source of every NetMHCIIpan allele in current runs and the file whose HLA column defines the DP/DQ panel.

### `netmhciipan_ba_data_used_in_mhciifold_gnn_training.csv`

NetMHCIIpan binding-affinity data used in MHCII-Fold GNN training. This dataset is read by the training code as full_data2.

### `hla_pseudosequences_2023.dat`

HLA -> 34-residue pseudosequence lookup (whitespace-separated, no header). Both scripts replace the allele name with its pseudosequence, allowing the model to generalize across HLA alleles.

### `external_validation_and_test_set.csv`

External validation and test dataset containing experimental peptides together with decoy negative peptides. Columns used: AA, HLA, Original, score.

### `known_epitopes_per_allele.csv`

Every known (epitope, allele) pair. Columns: `HLA`, `Original`.

Used only by the evaluation scripts, and only to decide what counts as a decoy.
Performance is measured as the rank of a true epitope within its own group of
tiled peptides; if another known epitope of the same allele happens to sit in
that group, it is removed first, so a real epitope is never counted as a decoy
for another. Changing this file changes the metric, so it is kept as an
explicit input rather than being inferred at run time.
