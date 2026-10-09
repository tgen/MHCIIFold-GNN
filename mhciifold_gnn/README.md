# MHCIIFold-GNN

A graph neural network that predicts peptide binding to MHC class II alleles
from the **3D structure** of the peptide–MHC complex predicted by AlphaFold3,
rather than from sequence alone.

Each AlphaFold3 complex is converted into a graph whose nodes are the residues
of the HLA alpha chain, the HLA beta chain and the peptide:

| | |
|---|---|
| node features | pLDDT (1) · molecule-type one-hot (3) · amino-acid one-hot (20) |
| edges | residue pairs with any two atoms within 5 Å |
| edge features | AF3 contact probability · CA–CA distance |

Peptides longer than 9 residues are cut into overlapping 9mer windows (step 1);
every window is its own predicted structure. The windows of one peptide are
concatenated into a single graph, scored per window by a GATv2 network, and
combined into one peptide-level score by geometric mean.

> **Score polarity:** the network is trained against `1 - y`, so the output is
> **low for binders** and high for non-binders. Every script here assumes that.

The final MHCIIFold-GNN score is the geometric mean of two models: a
peptide-aware one (`all_aa`) and a peptide-agnostic one (`only_HLA_aa`, in
which the peptide residues' amino-acid identity is hidden). Both are released
in `input_data/`.

## Layout

```
mhciifold_gnn/
├── graph_creation_from_AF3_outputs.py   AlphaFold3 output -> graph pickle
├── train_mhciifold_gnn.py               training
├── test_mhciifold_gnn.py                scores peptides with a trained model
├── select_epochs_from_validation.py     epoch selection (retraining only)
├── evaluate_on_test_set.py              evaluation on the held-out test alleles
├── graph_config.json                    settings for graph creation
├── input_data/                          model weights and all input data
└── output_data/                         per-allele score tables; all outputs
```

Paths are resolved relative to the scripts, so they can be run from any
directory. `input_data/README.md` and `output_data/README.md` describe every
file in those folders.

## Reproducing the published test results

```bash
# 1. Score each test allele with both released models. The graphs are read
#    straight out of input_data/full_test_and_validation_set.tar, so there is
#    nothing to unpack. An allele can be named either way: 0404 or DRB1_0404.
python test_mhciifold_gnn.py 0404      all_aa
python test_mhciifold_gnn.py 0404      only_HLA_aa
python test_mhciifold_gnn.py DRB5_0101 all_aa
python test_mhciifold_gnn.py DRB5_0101 only_HLA_aa
# ... for every test allele

# 2. Evaluate.
python evaluate_on_test_set.py
```

Step 1 writes `output_data/<HLA>_epitope_test_released_<aa_set>.parquet`; step 2
reads them, combines the two models and prints the summary table.

The 43 test alleles are listed in `evaluate_on_test_set.py` (`test_HLAs` and
`test_other_HLAs`), so step 1 can be looped:

```bash
python - <<'PY' > score_all.sh
import re
src = open('evaluate_on_test_set.py').read()
hlas = re.findall(r"'([A-Z0-9_-]+)'", src.split('test_HLAs = [')[1].split('test_sqq')[0])
for h in hlas:
    for aa in ('all_aa', 'only_HLA_aa'):
        print(f'python test_mhciifold_gnn.py {h} {aa}')
PY
bash score_all.sh
```

## Retraining from scratch

```bash
# <hla_group_idx> picks how many NetMHCIIpan alleles join the 14 PepSeq DRB1
# alleles; group 14 = 18 alleles, i.e. a 32-allele panel.
python train_mhciifold_gnn.py all_aa      14
python train_mhciifold_gnn.py only_HLA_aa 14   # needs the training graphs, see below

# Score every validation and test allele with every 10th-epoch checkpoint.
python test_mhciifold_gnn.py 0404 all_aa      --all-epochs --n-hlas 32
python test_mhciifold_gnn.py 0404 only_HLA_aa --all-epochs --n-hlas 32
# ... for every allele

# Choose the epoch pair on the validation alleles, then evaluate it on the test
# alleles.
python select_epochs_from_validation.py --n-hlas 32
python evaluate_on_test_set.py --epochs --n-hlas 32
```

`--n-hlas` is the trained panel size (14 + the number of NetMHCIIpan alleles in
the training group) and must match across these steps. `--seed` (default 3)
selects the training replicate the same way.

**Retraining needs graphs that are not in this release.**
`full_test_and_validation_set.tar` holds the validation and test graphs only.
Training also needs `alpha_9mers_<HLA>_new_data_...pkl` and
`alpha_9mers_<HLA>_NetMHCIIPan_...pkl` for the training alleles; put them in
`input_data/` to retrain.

**The epoch sweep cannot be reproduced from this release.** It needs the
per-epoch checkpoints, which are not shipped; the epochs it chose are already
baked into the released weights. `select_epochs_from_validation.py` is included
for the record, and applies to a retrained model.

## The scripts

### `graph_creation_from_AF3_outputs.py`

Reads AlphaFold3's `model.cif` and `confidences.json` for one 9mer–MHC complex
and writes the graph as a pickle. HLA residues are kept only if they have an
atom within 5 Å of the peptide, which keeps the graphs small. Nodes are
residues, placed at their alpha carbon. Two residues are connected when any
pair of their atoms is within 5 Å; the edge stores the AF3 contact probability
and the CA–CA distance, which can therefore exceed 5 Å. The output folders in
`graph_config.json` are placeholders. The AlphaFold3 predictions themselves are
not part of this repository — the validation and test graphs the other scripts
need are in `input_data/full_test_and_validation_set.tar`.

### `train_mhciifold_gnn.py`

```
python train_mhciifold_gnn.py <aa_set> <hla_group_idx> [seed]
```

Training data is in-house PepSeq binding data for 14 DRB1 alleles plus
NetMHCIIpan 4.3 binding-affinity data for non-DRB1 alleles (DRB3/4/5, DP, DQ).
Non-binders are down-sampled per peptide length to match the binders, and the
loss is weighted for both class imbalance and data source. A checkpoint is
saved every 10 epochs.

### `test_mhciifold_gnn.py`

```
python test_mhciifold_gnn.py <HLA> <aa_set> [batch_size]
                             [--checkpoint FILE | --all-epochs]
```

Scores the allele's peptides and writes one parquet. By default it uses the
released weights for that `aa_set`; `--all-epochs` scores every 10th-epoch
checkpoint of a retrained model instead, and `--checkpoint` scores one given
file. `<aa_set>` must match the model — a mismatch would not change any feature
width, so it is checked explicitly. The allele's graphs are read directly from
`full_test_and_validation_set.tar`, or from an extracted `.pkl` in
`input_data/` when one is there. Batch size only affects memory, never the
scores.

### `select_epochs_from_validation.py`

```
python select_epochs_from_validation.py [--n-hlas N] [--seed S]
```

Model selection, for a retrained model: the only place validation labels
influence a choice. The two models need not peak at the same epoch, so every
(all_aa, only_HLA_aa) epoch pair in the 30 × 30 grid is evaluated and the best
pair reported.

Performance is measured per epitope. Each true epitope is ranked against the
decoy peptides of its own group:

```
ratio = (# peptides in the group scoring at least as well as the true epitope) / (group size)
```

and `1 - ratio` is reported, so it reads like an AUC (mean and 95 % CI over
epitopes). Other known true epitopes of the same allele
(`known_epitopes_per_allele.csv`) are kept out of a group's decoys, and groups
with more than one positive are split into contiguous stretches first.
NetMHCIIpan, FCM and the GNN + NetMHCIIpan combination are scored the same way
for comparison.

### `evaluate_on_test_set.py`

```
python evaluate_on_test_set.py [--epochs] [--n-hlas N] [--seed S]
```

Applies the released models (or, with `--epochs`, a retrained model at the
selected epoch pair) to the held-out test alleles and reports the same
per-epitope measure. Nothing is selected here.

## Requirements

Python 3, PyTorch, PyTorch Geometric, pandas, NumPy, SciPy, scikit-learn and
pyarrow (parquet). Biopython is needed only for graph creation. A GPU is used
when available; otherwise everything runs on the CPU.
