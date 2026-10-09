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

## What this repository lets you do

The two trained models are included, so the published test results can be
reproduced end to end:

```bash
# 1. Score every allele that has a graph pickle in input_data/, with each of
#    the two released models.
python test_mhciifold_gnn.py all_aa
python test_mhciifold_gnn.py only_HLA_aa

# 2. Evaluate.
python evaluate_on_test_set.py
```

Step 1 scores whichever `alpha_9mers_<HLA>_epitopes_...pkl` files are in
`input_data/` — a partial set is fine — and writes one
`output_data/<HLA>_epitope_test_released_<aa_set>.parquet` per allele. Step 2
reads them, combines the two models and prints the summary table, naming any
test allele it found no scores for.

The code that produced the models is included as well
(`graph_creation_from_AF3_outputs.py`, `train_mhciifold_gnn.py`,
`select_epochs_from_validation.py`), but it cannot be re-run here: the
AlphaFold3 predictions and the training binding data are not part of this
release. Those scripts are provided so the method can be read and checked.

## Layout

```
mhciifold_gnn/
├── test_mhciifold_gnn.py                scores peptides with a released model
├── evaluate_on_test_set.py              evaluation on the held-out test alleles
│
├── graph_creation_from_AF3_outputs.py   AlphaFold3 output -> graph pickle   ─┐
├── train_mhciifold_gnn.py               training                             ├ reference
├── select_epochs_from_validation.py     epoch selection                     ─┘
├── graph_config.json                    settings for graph creation
│
├── input_data/                          model weights and all input data
└── output_data/                         per-allele score tables; all outputs
```

Paths are resolved relative to the scripts, so they can be run from any
directory. Every data file is documented column by column in
[`input_data/README.md`](input_data/README.md) (what the scripts read) and
[`output_data/README.md`](output_data/README.md) (the per-allele score tables,
and everything the scripts write).

## How the pieces fit together

The model was built in four stages. The first three produced the released
weights; only the last two run from this repository.

| Stage | Script | Reads | Writes | Runs here |
|---|---|---|---|---|
| 1. Graphs | `graph_creation_from_AF3_outputs.py` | AlphaFold3 `model.cif` + `confidences.json` | one graph pickle per 9mer–MHC complex | no |
| 2. Training | `train_mhciifold_gnn.py` | training graph pickles and binding data | a checkpoint every 10 epochs | no |
| 3. Epoch choice | `select_epochs_from_validation.py` | validation scores + `<HLA>_validation_final.csv` | the chosen (all_aa, only_HLA_aa) epoch pair | no |
| 4. Scoring | `test_mhciifold_gnn.py` | a released model + the per-allele graph pickles + `external_validation_and_test_set.csv` | one parquet of per-peptide scores per allele | **yes** |
| 5. Evaluation | `evaluate_on_test_set.py` | the test parquets + `<HLA>_test_final.csv` + `known_epitopes_per_allele.csv` | the summary table and per-epitope ratios | **yes** |

Stage 3 is what the released weights already encode: each model is the
checkpoint at its selected epoch, so no epoch is chosen at scoring time.

`graph_config.json` holds the distance threshold and the output paths stage 1
uses; nothing else reads it.

### What a run produces

Stage 4 writes `output_data/<HLA>_epitope_test_released_<aa_set>.parquet`, one
per allele per model, with a `GeoMean` column (the model's score for that
peptide, low = binder) alongside the peptide, its epitope group and its label.

Stage 5 prints a table of the mean within-group epitope rank, with 95 %
confidence intervals, for the new model and each comparison model, and saves
the same table plus the per-epitope rows to
`output_data/test_14_pepseq_hlas_fixed/test_results_nhlas<N>_seed<S>.pkl`.

## The scripts

### `test_mhciifold_gnn.py`

```
python test_mhciifold_gnn.py <aa_set> [batch_size]
```

Scores every allele that has a graph pickle in `input_data/` and writes one
parquet each, so a partial set of alleles is fine. `<aa_set>` is `all_aa` or
`only_HLA_aa` and selects the released model; it must match, because a
mismatch would not change any feature width and would silently produce
meaningless scores, so it is checked explicitly. Batch size only affects
memory, never the scores.

### `evaluate_on_test_set.py`

```
python evaluate_on_test_set.py
```

Combines the two models' scores on the held-out test alleles and reports
performance per epitope. Each true epitope is ranked against the decoy
peptides of its own group:

```
ratio = (# peptides in the group scoring at least as well as the true epitope) / (group size)
```

and `1 - ratio` is reported, so it reads like an AUC (mean and 95 % CI over
epitopes). Other known true epitopes of the same allele
(`known_epitopes_per_allele.csv`) are kept out of a group's decoys, and groups
with more than one positive are split into contiguous stretches first.
NetMHCIIpan, FCM and the GNN + NetMHCIIpan combination are scored the same way
for comparison. Nothing is selected here.

## Reference code

These three produced the released models and are included so the method can be
read and checked. They need the AlphaFold3 predictions and the training
binding data, which are not part of this release.

### `graph_creation_from_AF3_outputs.py`

Reads AlphaFold3's `model.cif` and `confidences.json` for one 9mer–MHC complex
and writes the graph as a pickle. HLA residues are kept only if they have an
atom within 5 Å of the peptide, which keeps the graphs small. Nodes are
residues, placed at their alpha carbon. Two residues are connected when any
pair of their atoms is within 5 Å; the edge stores the AF3 contact probability
and the CA–CA distance, which can therefore exceed 5 Å. The output folders in
`graph_config.json` are placeholders.

### `train_mhciifold_gnn.py`

Training data is in-house PepSeq binding data for 14 DRB1 alleles plus
NetMHCIIpan 4.3 binding-affinity data for non-DRB1 alleles (DRB3/4/5, DP, DQ);
how many of the latter are included is the allele-panel sweep axis. Non-binders
are down-sampled per peptide length to match the binders, and the loss is
weighted for both class imbalance and data source. A checkpoint is saved every
10 epochs.

### `select_epochs_from_validation.py`

Model selection: the only place validation labels influence a choice. The
peptide-aware and peptide-agnostic models need not peak at the same epoch, so
the epoch is chosen as a pair, sweeping the full 30 × 30 grid on the validation
alleles with the same per-epitope measure used for the test set. The pair it
chose is what the released weights are.

## Requirements

Python 3, PyTorch, PyTorch Geometric, pandas, NumPy, SciPy, scikit-learn and
pyarrow (parquet). Biopython is needed only for graph creation. A GPU is used
when available; otherwise everything runs on the CPU.
