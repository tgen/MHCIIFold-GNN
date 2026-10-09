# `input_data/` — everything the scripts read

| File | Read by |
|---|---|
| `full_test_and_validation_set.tar` | — (extract first, see below) |
| `peptide_aware_model_train_weights.pth` | `test_mhciifold_gnn.py` (`all_aa`) |
| `peptide_agnostic_model_train_weights.pth` | `test_mhciifold_gnn.py` (`only_HLA_aa`) |
| `external_validation_and_test_set.csv` | `test_mhciifold_gnn.py` |
| `known_epitopes_per_allele.csv` | `select_epochs_from_validation.py`, `evaluate_on_test_set.py` |
| `pepseq_drb1_binding_data.csv` | `train_mhciifold_gnn.py` |
| `netmhciipan_ba_peptides_full.csv` | `train_mhciifold_gnn.py` |
| graph pickles (from the `.tar`) | `train_mhciifold_gnn.py`, `test_mhciifold_gnn.py` |

**Extract the archive before running anything:**

```bash
tar -xf input_data/full_test_and_validation_set.tar -C input_data/
```

---

## Released model weights

The two published models. The epoch of each was chosen on the validation set,
so they are the final models the paper reports.

| File | `aa_set` | Model |
|---|---|---|
| `peptide_aware_model_train_weights.pth` | `all_aa` | peptide residues keep their amino-acid one-hot |
| `peptide_agnostic_model_train_weights.pth` | `only_HLA_aa` | peptide amino-acid one-hot zeroed |

The final MHCIIFold-GNN score is the geometric mean of the two models'
rank-normalised scores.

---

## Graph pickles

Built from AlphaFold3 predictions by `graph_creation_from_AF3_outputs.py` and
merged per allele. The `5` in the names is the edge distance threshold in Å.

| File | Used by |
|---|---|
| `alpha_9mers_<HLA>_epitopes_AA_level_GNN_5_with_letters.pkl` | test |
| `alpha_9mers_<HLA>_new_data_AA_level_GNN_5_with_letters.pkl` | train (PepSeq DRB1 alleles) |
| `alpha_9mers_<HLA>_NetMHCIIPan_AA_level_GNN_5_with_letters.pkl` (plus an optional `..._2.pkl`) | train (NetMHCIIpan alleles) |

Each pickle is a dictionary `{9mer sequence: torch_geometric.data.Data}`. Every
`Data` object is one predicted peptide–MHC complex, trimmed to the binding
groove (HLA residues with an atom within 5 Å of the peptide, plus the full
peptide):

| Field | Shape | Contents |
|---|---|---|
| `x` | (n_residues, 7) | One row per residue, at its alpha carbon: `x, y, z, pLDDT (residue mean), molecule type (0 = HLA alpha, 1 = HLA beta, 2 = peptide), amino-acid code (1–20 standard, 21 = X, 22 = B), residue number within its chain`. The last 9 rows are the peptide. |
| `edge_index` | (2, n_edges) | Residue pairs with any two atoms within 5 Å, stored in both directions. |
| `edge_features` | (n_edges,) | AlphaFold3 contact probability of the residue pair. |
| `edge_lengths` | (n_edges,) | Distance between the two residues' alpha carbons, in Å. Can exceed 5 Å, because edges are decided on atom-to-atom distance. |

Peptides longer than 9 residues are scored from all of their overlapping 9mer
windows, each of which has its own entry.

---

## Peptide data

| File | Contents | Columns used |
|---|---|---|
| `external_validation_and_test_set.csv` | External validation and test peptides: experimentally identified epitopes together with decoy ("fake") peptides tiled across the same source proteins. | `AA` (peptide), `HLA` |
| `known_epitopes_per_allele.csv` | Every known (epitope, allele) pair, across validation and test. Keeps the other true epitopes of an allele out of each epitope's decoy group. | `Original` (epitope sequence), `HLA` |
| `pepseq_drb1_binding_data.csv` | In-house PepSeq binding data for the DRB1 alleles; the PepSeq portion of the training set. | `peptide`, `HLA` (e.g. `DRB1_0405`), `status` (`binder` / `nonbinder`) |
| `netmhciipan_ba_peptides_full.csv` | NetMHCIIpan 4.3 binding-affinity training peptides; source of the non-DRB1 (DRB3/4/5, DP, DQ) training alleles. | `peptide`, `HLA`, `status` |

In training, non-binders are down-sampled per peptide length to match the
binders' length distribution.
