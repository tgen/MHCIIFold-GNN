# `input_data/` — everything the scripts read

Files marked **released** are all you need to reproduce the published test
results; the rest are only used when retraining.

| File | Read by | |
|---|---|---|
| `alpha_9mers_<HLA>_epitopes_AA_level_GNN_5_with_letters.pkl` | `test_mhciifold_gnn.py` | released |
| `peptide_aware_model_train_weights.pth` | `test_mhciifold_gnn.py` (`all_aa`) | released |
| `peptide_agnostic_model_train_weights.pth` | `test_mhciifold_gnn.py` (`only_HLA_aa`) | released |
| `external_validation_and_test_set.csv` | `test_mhciifold_gnn.py` | released |
| `known_epitopes_per_allele.csv` | `select_epochs_from_validation.py`, `evaluate_on_test_set.py` | released |
| `pepseq_drb1_binding_data.csv` | `train_mhciifold_gnn.py` | retraining |
| `netmhciipan_ba_peptides_full.csv` | `train_mhciifold_gnn.py` | retraining |

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

Despite the file names, these hold **whole pickled model objects**, not
`state_dict`s, so `torch.load` needs `weights_only=False` and needs the `GNN`
and `GCNGraphClassifier` classes importable from `__main__`.
`test_mhciifold_gnn.py` satisfies both when run as a script.

---

## Graph pickles

Built from AlphaFold3 predictions by `graph_creation_from_AF3_outputs.py` and
merged per allele. The `5` in the names is the edge distance threshold in Å.

One pickle per allele, named

```
alpha_9mers_<HLA>_epitopes_AA_level_GNN_5_with_letters.pkl
```

`test_mhciifold_gnn.py` scores whichever of these it finds in this folder, so a
partial set of alleles is fine. `<HLA>` may carry the DRB1_ prefix
(`alpha_9mers_DRB1_0818_...`) or be the bare number (`alpha_9mers_0405_...`);
either way the parquet is written without the prefix, which is how the
evaluation scripts look it up.

**Training graphs are not part of the release.** Retraining additionally needs
`alpha_9mers_<HLA>_new_data_AA_level_GNN_5_with_letters.pkl` (PepSeq DRB1
alleles) and `alpha_9mers_<HLA>_NetMHCIIPan_AA_level_GNN_5_with_letters.pkl`
(NetMHCIIpan alleles, plus an optional `..._2.pkl` second part), placed in this
folder.

Each pickle is a dictionary `{9mer sequence: torch_geometric.data.Data}`. Every
`Data` object is one predicted peptide–MHC complex, trimmed to the binding
groove (HLA residues with an atom within 5 Å of the peptide, plus the full
peptide):

| Field | Shape | Contents |
|---|---|---|
| `x` | (n_residues, 6) | One row per residue, at its alpha carbon: `x`, `y`, `z`, pLDDT (mean over the residue's atoms), molecule type (0 = HLA alpha, 1 = HLA beta, 2 = peptide), amino-acid code (1–20 standard, 21 = X, 22 = B). The last 9 rows are the peptide. Graphs straight out of `graph_creation_from_AF3_outputs.py` carry a 7th column, the residue number within its chain, which the scripts drop. |
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
