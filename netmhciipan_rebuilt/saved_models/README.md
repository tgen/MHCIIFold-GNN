# `saved_models/` — the released weights

Fifteen trained models, one per `model_index`. Together they are the released
netmhciipan-rebuilt predictor: `evaluate_on_test_set.py` averages all fifteen
into one ensemble score.

```
netmhciipan_model_<II>_14_pepseq_hlas_fixed_20netmhciipan_hlas_hd<HD>.pt
```

| part | meaning |
|---|---|
| `<II>` | `model_index`, 01..15 |
| `14_pepseq_hlas_fixed` | the PepSeq side of the training panel, always the same 14 DRB1 alleles |
| `20netmhciipan_hlas` | plus 20 NetMHCIIpan alleles |
| `<HD>` | hidden layer width |

## Which model is which

`model_index` fixes both the hidden width and the random seed, through the
table in `netmhciipan_code.py` (`hidden_dims = (20, 40, 60)` × `seeds = 0..4`).
The epoch is the one chosen on the validation alleles by
`select_epoch_from_validation.py`; it is not in the file name, so it is here.

| model_index | hidden_dim | seed | epoch |
|---|---|---|---|
| 1 | 20 | 0 | 430 |
| 2 | 20 | 1 | 10 |
| 3 | 20 | 2 | 10 |
| 4 | 20 | 3 | 40 |
| 5 | 20 | 4 | 10 |
| 6 | 40 | 0 | 300 |
| 7 | 40 | 1 | 150 |
| 8 | 40 | 2 | 160 |
| 9 | 40 | 3 | 120 |
| 10 | 40 | 4 | 110 |
| 11 | 60 | 0 | 260 |
| 12 | 60 | 1 | 250 |
| 13 | 60 | 2 | 60 |
| 14 | 60 | 3 | 310 |
| 15 | 60 | 4 | 10 |

The same table is hardcoded as `SELECTED_EPOCHS` in `evaluate_on_test_set.py`,
so the test evaluation selects nothing.

## What is inside one file

A **whole pickled `nn.Module`**, saved with `torch.save(model)` — not a
`state_dict`. Loading therefore needs the class importable and
`weights_only=False` on torch ≥ 2.6, which flipped that default:

```python
import torch
import netmhciipan_code          # defines NetMHCIIPanPeptideModel

model = torch.load(path, map_location='cpu', weights_only=False)
model.eval()
```

`netmhciipan_test.py` does exactly this, with a fallback for older torch.

## The architecture these weights belong to

`NetMHCIIPanPeptideModel` scores one 9-residue core at a time and combines the
cores of a peptide:

```
per 9mer window:  Linear(863 -> hidden_dim) -> ReLU -> Linear(hidden_dim, 1) -> sigmoid
per peptide:      geometric mean over that peptide's windows
```

The 863 input features per window are

| block | size | contents |
|---|---|---|
| peptide core | 9 × 20 = 180 | BLOSUM62 encoding of the 9-residue core |
| flanks | 3 | residues before the core, residues after it, full peptide length |
| HLA | 34 × 20 = 680 | BLOSUM62 encoding of the allele's pseudosequence |

Because the allele enters only as its pseudosequence, a model can score an
allele it never saw in training.

**Score polarity:** training uses binder → 0, non-binder → 1, so the output is
**low for binders**. Everything downstream assumes this.

## Size

Each file is 140–210 KB; the fifteen together are under 3 MB. The matching
prediction CSVs are in `../output_data/`.
