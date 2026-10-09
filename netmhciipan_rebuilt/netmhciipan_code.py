import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re


# --- PATHS ---------------------------------------------------------------
# Everything is resolved relative to THIS FILE, so the script runs from any
# working directory: input_data/ holds the data files it reads, output_data/
# receives the checkpoints and predictions it writes. Both sit next to this
# script in the repository.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'input_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_data')


# --- HLA PANEL SELECTION -------------------------------------------------
# The pepseq side is now FIXED at the first FIXED_PEPSEQ_HLAS pepseq HLAs, and
# only the netmhciipan side varies. The varying count is the 2nd CLI arg, AFTER
# the model index (argv[1]): it takes the first N HLAs of netmhciipan_order.
# That add-order is the one defined by daily_use_2.py's hla_groups -- every
# hla_groups entry is a prefix of the next, so the largest key holds the full
# ordered list and the first N are just its first N entries.
# N may be 0..20 (netmhciipan_order has 20 entries); N = 0 is the
# pepseq-only baseline. The sweep in daily_use_auto.sh submits
# N = 0, 1, 6, 10, 20.
FIXED_PEPSEQ_HLAS = 14

# Where the two netmhciipan peptide tables live. The "all_" one holds many more
# peptides per HLA; the other is the original, smaller set.
NETMHCIIPAN_BA_CSV     = os.path.join(
    INPUT_DIR, 'netmhciipan_ba_data_used_in_mhciifold_gnn_training.csv')
ALL_NETMHCIIPAN_BA_CSV = os.path.join(
    INPUT_DIR, 'netmhciipan_ba_peptides_full.csv')

# --- Is the MHCIIFold-GNN binding-affinity table part of the training set? ----
#   True  : it is NOT used. Every netmhciipan HLA is drawn from
#           ALL_NETMHCIIPAN_BA_CSV (full_data22) only, which has far more
#           peptides per HLA. This is what every run in the paper used.
#   False : it IS used. The same HLAs are additionally drawn from
#           NETMHCIIPAN_BA_CSV (full_data2) and both frames are concatenated
#           into the training set.
SKIP_MHCIIFOLD_GNN_BA_DATA = True

# --- N = 999: the "full DP/DQ" instance ---------------------------------------
# A special panel, NOT a count. It takes
#   (a) the original 20 netmhciipan HLAs, and
#   (b) EVERY HLA in ALL_NETMHCIIPAN_BA_CSV whose name contains 'DP' or 'DQ',
# and draws ALL of them from ALL_NETMHCIIPAN_BA_CSV (full_data22) rather than
# rather than from the smaller NETMHCIIPAN_BA_CSV, because that file has far
# more peptides per HLA. 999 is just an easy-to-spot tag in the run directory name.
FULL_DP_DQ_N = 999

if len(sys.argv) < 3:
    raise SystemExit(
        f"Usage: {sys.argv[0]} <model_index> <number_of_netmhciipan_hlas>\n"
        "  model_index                 : which model/seed config to train\n"
        f"  number_of_netmhciipan_hlas : N, how many netmhciipan HLAs to use\n"
        f"                               (0 = pepseq only, {FULL_DP_DQ_N} = the full\n"
        f"                               DP/DQ instance). The pepseq side is\n"
        f"                               fixed at the first {FIXED_PEPSEQ_HLAS}."
    )
number_of_netmhciipan_hlas = int(sys.argv[2])
# Kept as aliases for any older reference to these names.
number_of_hlas = number_of_netmhciipan_hlas
number_of_pepseq_hlas = FIXED_PEPSEQ_HLAS

pepseq_hla_list = ['DRB1_0405','DRB1_1201','DRB1_0301','DRB1_0901','DRB1_0701','DRB1_1001','DRB1_1301','DRB1_1101','DRB1_0406','DRB1_1503','DRB1_1602','DRB1_0804','DRB1_1401','DRB1_0102','DRB1_1102','DRB1_1302']

netmhciipan_order = [
    'DRB5_0101',
    'DRB3_0101',
    'DRB4_0101',
    'HLA-DQA10501-DQB10301',
    'DRB3_0202',
    'HLA-DQA10301-DQB10302',
    'HLA-DQA10101-DQB10501',
    'HLA-DQA10501-DQB10201',
    'HLA-DPA10103-DPB10401',
    'HLA-DQA10102-DQB10602',
    'HLA-DPA10301-DPB10402',
    'HLA-DPA10201-DPB10501',
    'DRB3_0301',
    'HLA-DQA10301-DQB10201',
    'DRB4_0103',
    'HLA-DQA10301-DQB10301',
    'HLA-DQA10501-DQB10302',
    'HLA-DQA10103-DQB10603',
    'HLA-DPA10103-DPB10201',
    'HLA-DQA10401-DQB10402',
    "DRB1_0101",
    "DRB1_0701",
    "DRB1_0401",
    "DRB1_1101",
    "DRB1_0301",
    "DRB1_1501",
    "DRB1_0802",
    "DRB1_1302",
    "DRB1_0901",
    "DRB1_0405",
    "DRB1_0404"
]

# The first N_ORIGINAL_NETMHCIIPAN entries are the original (non-DRB1) block;
# everything after that is the extension. BOTH now come from full_data22
# (netmhciipan_ba_peptides_full.csv), which has far more peptides per HLA than the
# older NETMHCIIPAN_BA_CSV. The run directory carries a '999' tag so these
# runs are not confused with the earlier ones that used the small file.
N_ORIGINAL_NETMHCIIPAN = 20
netmhciipan_original  = netmhciipan_order[:N_ORIGINAL_NETMHCIIPAN]
netmhciipan_extension = netmhciipan_order[N_ORIGINAL_NETMHCIIPAN:]

FULL_DP_DQ_RUN = (number_of_netmhciipan_hlas == FULL_DP_DQ_N)

if not FULL_DP_DQ_RUN and not 0 <= number_of_netmhciipan_hlas <= len(netmhciipan_order):
    raise SystemExit(
        f"number_of_netmhciipan_hlas must be between 0 and "
        f"{len(netmhciipan_order)} (or {FULL_DP_DQ_N} for the full DP/DQ "
        f"instance), got {number_of_netmhciipan_hlas}"
    )

print('number_of_pepseq_hlas (fixed):', FIXED_PEPSEQ_HLAS)
print('number_of_netmhciipan_hlas:', number_of_netmhciipan_hlas)

# Pepseq side: FIXED. NetMHCIIpan side: the varying one.
hla_list = pepseq_hla_list[:FIXED_PEPSEQ_HLAS]

# The BASE panel for every run is now the same: the original 20 netmhciipan HLAs
# plus EVERY DP/DQ HLA in the big file. N = 999 is that base on its own; any
# other N adds the first (N - 20) extension HLAs on top, so the panels are
# nested and 999 is the zero-extension point of the same curve.
_all_hlas = pd.read_csv(ALL_NETMHCIIPAN_BA_CSV, usecols=['HLA'])['HLA']
_all_hlas = _all_hlas.astype(str).str.strip().unique().tolist()

_dp_dq = sorted(h for h in _all_hlas if ('DP' in h) or ('DQ' in h))

# original 20 first (keeps the 6 non-DRB1 DR alleles), then the DP/DQ ones that
# are not already in it
netmhciipan_base = list(netmhciipan_original)
netmhciipan_base += [h for h in _dp_dq if h not in netmhciipan_base]

if FULL_DP_DQ_RUN:
    netmhciipan_extension_used = []
else:
    # the usual extension entries, netmhciipan_order[20:N]
    netmhciipan_extension_used = [h for h in netmhciipan_order[:number_of_netmhciipan_hlas]
                                  if h in netmhciipan_extension]

netmhciipan_hla_list = netmhciipan_base + [h for h in netmhciipan_extension_used
                                           if h not in netmhciipan_base]

# Which netmhciipan table(s) each HLA is drawn from, per the flag at the top of
# the file. full_data2 is NETMHCIIPAN_BA_CSV (the MHCIIFold-GNN training data),
# full_data22 is ALL_NETMHCIIPAN_BA_CSV (the full set). Both frames are
# concatenated later, so listing an HLA on both sides means its peptides come
# from both files.
if SKIP_MHCIIFOLD_GNN_BA_DATA:
    # the published behaviour: the big file only
    netmhciipan_from_full_data2  = []
else:
    netmhciipan_from_full_data2  = list(netmhciipan_hla_list)
netmhciipan_from_full_data22 = list(netmhciipan_hla_list)

print(f'SKIP_MHCIIFOLD_GNN_BA_DATA = {SKIP_MHCIIFOLD_GNN_BA_DATA} '
      f'-> MHCIIFold-GNN BA table is '
      f'{"NOT used" if SKIP_MHCIIFOLD_GNN_BA_DATA else "used"}')

_missing = [h for h in netmhciipan_hla_list if h not in set(_all_hlas)]
print(f'{len(_all_hlas)} HLAs in {os.path.basename(ALL_NETMHCIIPAN_BA_CSV)}, '
      f'{len(_dp_dq)} of them DP/DQ')
print(f'  base (original 20 + all DP/DQ): {len(netmhciipan_base)} HLAs')
print(f'  extensions added              : {len(netmhciipan_extension_used)} '
      f'{netmhciipan_extension_used}')
print(f'  netmhciipan total             : {len(netmhciipan_hla_list)} HLAs, '
      f'from full_data22'
      f'{"" if SKIP_MHCIIFOLD_GNN_BA_DATA else " + full_data2"}')
if _missing:
    print(f'  [warn] {len(_missing)} selected HLA(s) are NOT in that file and '
          f'will contribute no rows: {_missing}')

hla_list.extend(netmhciipan_hla_list)

print('I am using these HLAs: ', hla_list)
print('Number of HLAs: ', len(hla_list))
print(f'  netmhciipan from full_data2  ({len(netmhciipan_from_full_data2)}): '
      f'{netmhciipan_from_full_data2}')
print(f'  netmhciipan from full_data22 ({len(netmhciipan_from_full_data22)}): '
      f'{netmhciipan_from_full_data22}')
# -------------------------------------------------------------------------


full_data = pd.read_csv(os.path.join(INPUT_DIR, 'pepseq_drb1_binding_data.csv'))
full_data = full_data[full_data['HLA'].isin(hla_list)]
full_data2 = pd.read_csv(NETMHCIIPAN_BA_CSV)

full_data22 = pd.read_csv(ALL_NETMHCIIPAN_BA_CSV)
full_data22['length'] = full_data22['peptide'].str.len()

# Each side keeps only the HLAs that belong to it. N = 0 means the pepseq-only
# baseline, and then both lists are empty and no netmhciipan rows are kept.
full_data2  = full_data2[full_data2['HLA'].isin(netmhciipan_from_full_data2)]
full_data22 = full_data22[full_data22['HLA'].isin(netmhciipan_from_full_data22)]

full_data['Peptide'] = full_data['peptide']
full_data["score"] = full_data["status"].map({"binder": 0, "nonbinder": 1})
full_data = full_data[['Peptide','HLA','score']]


# NOTE the label convention: binder -> 0, nonbinder -> 1, the SAME mapping the
# pepseq data and full_data2 use. full_data22 is mapped the same way here.
full_data2['Peptide'] = full_data2['peptide']
full_data2["score"] = full_data2["status"].map({"binder": 0, "nonbinder": 1})
full_data2 = full_data2[['Peptide','HLA','score']]

full_data22['Peptide'] = full_data22['peptide']
full_data22["score"] = full_data22["status"].map({"binder": 0, "nonbinder": 1})
full_data22 = full_data22[['Peptide','HLA','score']]

# Both netmhciipan sides go into the SAME training frame, so everything after
# this point is unchanged.
print(f'rows: pepseq={len(full_data)}, full_data2={len(full_data2)}, '
      f'full_data22={len(full_data22)}')

_netmhciipan_parts = [d for d in (full_data2, full_data22) if len(d)]
if _netmhciipan_parts:
    full_data = pd.concat([full_data] + _netmhciipan_parts)
else:
    print('No netmhciipan rows kept (N=0): training on the pepseq data alone.')

hlas = pd.read_csv(os.path.join(INPUT_DIR, 'hla_pseudosequences_2023.dat'),
                   sep=r"\s+", header=None)
hlas = hlas.rename(columns={0: "HLA", 1: "Pseudosequence"})



full_data = full_data.merge(hlas, on="HLA", how="left")
full_data = full_data.drop(columns=["HLA"]).rename(columns={"Pseudosequence": "HLA"})

# define standard amino acids
pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYX]+$")

full_data = full_data[
    full_data["Peptide"].str.len().ge(9) &
    full_data["Peptide"].str.match(pattern)
].reset_index(drop=True)


def balance_length_distribution_by_hla(
    df,
    peptide_col="Peptide",
    hla_col="HLA",
    score_col="score",
    binder_label=0.0,
    nonbinder_label=1.0,
):
    balanced_groups = []

    for hla_value, hla_df in df.groupby(hla_col, sort=False):
        hla_df = hla_df.copy()
        hla_df["length"] = hla_df[peptide_col].str.len()

        df_binders = hla_df[hla_df[score_col] == binder_label].copy()
        df_nonbinders = hla_df[hla_df[score_col] == nonbinder_label].copy()

        if len(df_binders) == 0 or len(df_nonbinders) == 0:
            balanced_groups.append(hla_df.drop(columns=["length"]))
            continue

        length_counts_binders = df_binders["length"].value_counts().sort_index()
        scale_factor = len(df_nonbinders) / float(length_counts_binders.sum())
        length_allocation_nonbinders = (length_counts_binders * scale_factor).astype(int)

        trimmed_nonbinders = []

        for length, count in length_allocation_nonbinders.items():
            available_nonbinders = df_nonbinders[df_nonbinders["length"] >= length]
            sample_size = min(int(count), len(available_nonbinders))

            if sample_size == 0:
                continue

            selected_nonbinders = available_nonbinders.sample(sample_size, replace=False)

            for idx, row in selected_nonbinders.iterrows():
                peptide = row[peptide_col]
                if len(peptide) > length:
                    trim_start = random.choice([True, False])
                    if trim_start:
                        trimmed_peptide = peptide[-length:]
                    else:
                        trimmed_peptide = peptide[:length]
                else:
                    trimmed_peptide = peptide

                trimmed_row = row.copy()
                trimmed_row[peptide_col] = trimmed_peptide
                trimmed_row["length"] = length
                trimmed_nonbinders.append(trimmed_row)

            df_nonbinders = df_nonbinders.drop(selected_nonbinders.index)

        if trimmed_nonbinders:
            df_trimmed_nonbinders = pd.DataFrame(trimmed_nonbinders)
            hla_balanced = pd.concat([df_binders, df_trimmed_nonbinders], ignore_index=True)
        else:
            hla_balanced = df_binders.copy()

        hla_balanced = hla_balanced[hla_balanced["length"] >= 9]
        balanced_groups.append(hla_balanced.drop(columns=["length"]))

    balanced_df = pd.concat(balanced_groups, ignore_index=True)
    print(f"Length-balanced rows: {len(balanced_df)}")
    return balanced_df


full_data = balance_length_distribution_by_hla(full_data)


AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# Standard BLOSUM62 rows ordered by AMINO_ACIDS above.
BLOSUM62 = np.array(
    [
        [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    dtype=np.float32,
)

PEPTIDE_CORE_LENGTH = 9
PEPTIDE_FEATURE_DIM = PEPTIDE_CORE_LENGTH * 20
HLA_SEQUENCE_LENGTH = 34
HLA_FEATURE_DIM = HLA_SEQUENCE_LENGTH * 20
EXTRA_FEATURE_DIM = 3
TOTAL_FEATURE_DIM = PEPTIDE_FEATURE_DIM + EXTRA_FEATURE_DIM + HLA_FEATURE_DIM


def print_tensor_stats(name, tensor, max_entries=5):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    flat = tensor.detach().reshape(-1).float().cpu()
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    total_count = flat.numel()

    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"finite={finite_count}/{total_count}"
    )

    if finite_count > 0:
        finite_values = flat[finite_mask]
        print(
            f"{name} stats: min={finite_values.min().item():.6f}, "
            f"max={finite_values.max().item():.6f}, "
            f"mean={finite_values.mean().item():.6f}"
        )

    preview = flat[:max_entries].tolist()
    print(f"{name} preview: {preview}")

    if finite_count != total_count:
        bad_idx = torch.nonzero(~finite_mask, as_tuple=False).reshape(-1)[:max_entries].tolist()
        print(f"{name} non-finite indices preview: {bad_idx}")


def _validate_sequence(seq, label):
    seq = str(seq).strip().upper()
    invalid = sorted({aa for aa in seq if aa not in AA_TO_INDEX})
    if invalid:
        raise ValueError(f"{label} contains unsupported amino acids: {invalid}")
    return seq


def is_standard_sequence(seq):
    seq = str(seq).strip().upper()
    return len(seq) > 0 and all(aa in AA_TO_INDEX for aa in seq)


def encode_sequence_with_blosum(sequence, expected_length=None):
    sequence = _validate_sequence(sequence, label="Sequence")
    if expected_length is not None and len(sequence) != expected_length:
        raise ValueError(
            f"Expected sequence length {expected_length}, got {len(sequence)} for '{sequence}'"
        )

    encoded = np.stack([BLOSUM62[AA_TO_INDEX[aa]] for aa in sequence], axis=0)
    return encoded.reshape(-1).astype(np.float32)


def peptide_to_9mer_features(peptide, hla_sequence):
    peptide = _validate_sequence(peptide, label="Peptide")
    hla_sequence = _validate_sequence(hla_sequence, label="HLA")

    if len(peptide) < PEPTIDE_CORE_LENGTH:
        raise ValueError(
            f"Peptide '{peptide}' is shorter than {PEPTIDE_CORE_LENGTH} amino acids"
        )
    if len(hla_sequence) != HLA_SEQUENCE_LENGTH:
        raise ValueError(
            f"HLA sequence must be length {HLA_SEQUENCE_LENGTH}, got {len(hla_sequence)}"
        )

    hla_encoded = encode_sequence_with_blosum(
        hla_sequence, expected_length=HLA_SEQUENCE_LENGTH
    )

    window_features = []
    peptide_length = float(len(peptide))

    for start in range(len(peptide) - PEPTIDE_CORE_LENGTH + 1):
        end = start + PEPTIDE_CORE_LENGTH
        core_9mer = peptide[start:end]

        peptide_encoded = encode_sequence_with_blosum(
            core_9mer, expected_length=PEPTIDE_CORE_LENGTH
        )
        flank_features = np.array(
            [float(start), float(len(peptide) - end), peptide_length],
            dtype=np.float32,
        )

        features = np.concatenate([peptide_encoded, flank_features, hla_encoded]).astype(
            np.float32
        )
        window_features.append(features)

    return np.stack(window_features, axis=0)


def dataframe_to_window_dataset(
    df,
    peptide_col="Peptide",
    hla_col="HLA",
    score_col="score",
):
    x_list = []
    scores = []

    rows = list(_iterate_rows(df))
    if not rows:
        raise ValueError("Input dataframe is empty")

    required_columns = {peptide_col, hla_col, score_col}
    missing = required_columns.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for row_idx, row in enumerate(rows):
        try:
            x_list.append(peptide_to_9mer_features(row[peptide_col], row[hla_col]))
        except ValueError as exc:
            raise ValueError(f"Row {row_idx}: {exc}") from exc
        score = float(row[score_col])
        if score not in (0.0, 1.0):
            raise ValueError(f"Row {row_idx}: score must be binary 0/1, got {score}")
        scores.append(score)

    y = np.asarray(scores, dtype=np.float32)
    return x_list, y


def filter_standard_peptides(
    df,
    peptide_col="Peptide",
    hla_col="HLA",
    score_col="score",
    min_peptide_length=PEPTIDE_CORE_LENGTH,
    hla_length=HLA_SEQUENCE_LENGTH,
):
    rows = []
    removed_count = 0

    for row in _iterate_rows(df):
        peptide = str(row[peptide_col]).strip().upper()
        hla = str(row[hla_col]).strip().upper()
        score = row[score_col]

        keep = (
            is_standard_sequence(peptide)
            and is_standard_sequence(hla)
            and len(peptide) >= min_peptide_length
            and len(hla) == hla_length
            and float(score) in (0.0, 1.0)
        )

        if keep:
            cleaned_row = dict(row)
            cleaned_row[peptide_col] = peptide
            cleaned_row[hla_col] = hla
            cleaned_row[score_col] = float(score)
            rows.append(cleaned_row)
        else:
            removed_count += 1

    if hasattr(df, "columns") and hasattr(df, "loc"):
        filtered_df = pd.DataFrame(rows, columns=df.columns)
    else:
        filtered_df = rows

    print(f"Filtered out {removed_count} rows with non-standard amino acids or invalid lengths")
    return filtered_df


def stratified_sample_dataframe(df, n_samples, stratify_col="score", random_state=42):
    if not (hasattr(df, "groupby") and hasattr(df, "sample")):
        raise TypeError("stratified_sample_dataframe requires a pandas DataFrame")

    if n_samples >= len(df):
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    class_counts = df[stratify_col].value_counts().sort_index()
    raw_targets = class_counts / class_counts.sum() * n_samples
    target_counts = np.floor(raw_targets).astype(int)

    remainder = int(n_samples - target_counts.sum())
    if remainder > 0:
        fractional = (raw_targets - target_counts).sort_values(ascending=False)
        for label in fractional.index[:remainder]:
            target_counts[label] += 1

    sampled_parts = []
    for idx, (label, target_count) in enumerate(target_counts.items()):
        group = df[df[stratify_col] == label]
        sample_n = min(int(target_count), len(group))
        sampled_parts.append(group.sample(n=sample_n, random_state=random_state + idx))

    sampled_df = pd.concat(sampled_parts, axis=0)
    sampled_df = sampled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled_df


def _iterate_rows(df):
    if hasattr(df, "iterrows"):
        for _, row in df.reset_index(drop=True).iterrows():
            yield row
        return

    if isinstance(df, list):
        for row in df:
            if not isinstance(row, dict):
                raise TypeError("List inputs must contain dictionaries")
            yield row
        return

    raise TypeError("df must be a pandas DataFrame or a list of dictionaries")


def binary_auc_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    start = 0
    n = sorted_scores.shape[0]
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end

    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks

    sum_pos_ranks = full_ranks[pos_mask].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class PeptideWindowDataset(Dataset):
    def __init__(self, x_list, y):
        self.x_list = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
            for x in x_list
        ]
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

        if len(self.x_list) != len(self.y):
            raise ValueError("x_list and y must have the same length")

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.y[idx]


def peptide_collate_fn(batch):
    x_list, y_list = zip(*batch)

    flat_x = []
    segment_ids = []

    for peptide_idx_in_batch, x in enumerate(x_list):
        n_windows = x.shape[0]
        flat_x.append(x)
        segment_ids.append(torch.full((n_windows,), peptide_idx_in_batch, dtype=torch.long))

    flat_x = torch.cat(flat_x, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)
    y_batch = torch.stack(y_list, dim=0)

    return flat_x, segment_ids, y_batch


class NetMHCIIPanPeptideModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, score_activation="sigmoid"):
        super().__init__()
        self.row_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.score_activation = score_activation

    def _activate_window_scores(self, raw_scores):
        raw_scores = raw_scores.squeeze(1)

        if self.score_activation == "sigmoid":
            return torch.sigmoid(raw_scores).clamp_min(1e-8)
        if self.score_activation == "softplus":
            return torch.nn.functional.softplus(raw_scores) + 1e-8
        raise ValueError("score_activation must be 'sigmoid' or 'softplus'")

    def _geometric_mean_by_segment(self, window_scores, segment_ids, n_peptides):
        log_scores = torch.log(window_scores.clamp_min(1e-8))

        sums = torch.zeros(n_peptides, device=window_scores.device, dtype=window_scores.dtype)
        sums.index_add_(0, segment_ids, log_scores)

        counts = torch.zeros(n_peptides, device=window_scores.device, dtype=window_scores.dtype)
        counts.index_add_(0, segment_ids, torch.ones_like(window_scores))

        mean_logs = sums / counts.clamp_min(1.0)
        return torch.exp(mean_logs)

    def forward(self, flat_x, segment_ids, n_peptides=None):
        if n_peptides is None:
            n_peptides = int(segment_ids.max().item()) + 1

        raw_window_scores = self.row_net(flat_x)
        window_scores = self._activate_window_scores(raw_window_scores)
        peptide_scores = self._geometric_mean_by_segment(
            window_scores=window_scores,
            segment_ids=segment_ids,
            n_peptides=n_peptides,
        )
        return peptide_scores, window_scores


class PeptideLevelLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            class_weights = {0.0: 1.0, 1.0: 1.0}
        self.neg_weight = float(class_weights.get(0.0, 1.0))
        self.pos_weight = float(class_weights.get(1.0, 1.0))

    def forward(self, peptide_scores, peptide_targets):
        eps = torch.finfo(peptide_scores.dtype).eps
        peptide_scores = peptide_scores.clamp(eps, 1.0 - eps)
        weights = torch.where(
            peptide_targets > 0.5,
            torch.full_like(peptide_targets, self.pos_weight),
            torch.full_like(peptide_targets, self.neg_weight),
        )
        return F.binary_cross_entropy(peptide_scores, peptide_targets, weight=weights)


def save_model_checkpoint(model, checkpoint_dir, checkpoint_prefix, epoch_index):
    """Save `model` as <checkpoint_dir>/<checkpoint_prefix>_epoch<epoch_index>.pt

    The directory is the run directory (model index + HLA panel size + hidden
    dim + seed) and carries NO epoch; the epoch lives only in the file name,
    zero-padded to 3 digits so the checkpoints sort correctly.
    `epoch_index` is 1-based: epoch010.pt is the model after 10 epochs.
    """
    if checkpoint_dir is None:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    prefix = checkpoint_prefix or "model"
    path = os.path.join(checkpoint_dir, f"{prefix}_epoch{epoch_index:03d}.pt")
    torch.save(model, path)
    print(f"Saved checkpoint: {path}", flush=True)
    return path


def train_one_model(
    train_dataset,
    input_dim,
    hidden_dim,
    seed,
    class_weights=None,
    eval_x_list=None,
    eval_y=None,
    lr=0.01,
    epochs=500,
    burn_in=20,
    batch_size=32,
    score_activation="sigmoid",
    device="cpu",
    checkpoint_dir=None,
    checkpoint_prefix=None,
    checkpoint_every=10,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=peptide_collate_fn,
    )

    model = NetMHCIIPanPeptideModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        score_activation=score_activation,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = PeptideLevelLoss(class_weights=class_weights)

    model.train()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (flat_x, segment_ids, y_batch) in enumerate(loader):
            flat_x = flat_x.to(device)
            segment_ids = segment_ids.to(device)
            y_batch = y_batch.to(device)

            if epoch == 0 and batch_idx == 0:
                print(f"Debug epoch={epoch + 1} batch={batch_idx + 1}")
                print_tensor_stats("flat_x", flat_x)
                print_tensor_stats("segment_ids", segment_ids)
                print_tensor_stats("y_batch", y_batch)

            if not torch.isfinite(flat_x).all():
                print(f"Non-finite values detected in input features at epoch {epoch + 1}, batch {batch_idx + 1}")
                print_tensor_stats("flat_x", flat_x)
                raise ValueError("flat_x contains NaN or Inf")

            optimizer.zero_grad()
            peptide_scores, window_scores = model(
                flat_x=flat_x,
                segment_ids=segment_ids,
                n_peptides=y_batch.shape[0],
            )

            if epoch == 0 and batch_idx == 0:
                print_tensor_stats("window_scores", window_scores)
                print_tensor_stats("peptide_scores", peptide_scores)

            if not torch.isfinite(window_scores).all():
                print(f"Non-finite values detected in window_scores at epoch {epoch + 1}, batch {batch_idx + 1}")
                print_tensor_stats("window_scores", window_scores)
                raise ValueError("window_scores contains NaN or Inf")

            if not torch.isfinite(peptide_scores).all():
                print(f"Non-finite values detected in peptide_scores at epoch {epoch + 1}, batch {batch_idx + 1}")
                print_tensor_stats("peptide_scores", peptide_scores)
                raise ValueError("peptide_scores contains NaN or Inf")

            loss = criterion(peptide_scores, y_batch)

            if not torch.isfinite(loss):
                print(f"Non-finite loss detected at epoch {epoch + 1}, batch {batch_idx + 1}")
                print_tensor_stats("flat_x", flat_x)
                print_tensor_stats("y_batch", y_batch)
                print_tensor_stats("window_scores", window_scores)
                print_tensor_stats("peptide_scores", peptide_scores)
                print(f"class_weights={class_weights}")
                raise ValueError("loss is NaN or Inf")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch == burn_in - 1:
            print(f"Burn-in finished at epoch {burn_in}")

        if eval_x_list is not None and eval_y is not None:
            eval_preds = predict_dataset_single_model(
                model=model,
                x_list=eval_x_list,
                batch_size=batch_size,
                device=device,
            )
            eval_auc = binary_auc_score(eval_y, eval_preds)
            print(
                f"Epoch {epoch + 1}/{epochs} | loss={total_loss:.4f} | overall_auc={eval_auc:.6f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{epochs} | loss={total_loss:.4f}")

        # Save every `checkpoint_every`-th epoch, counting from 1: with
        # checkpoint_every=10 that is epoch 10, 20, 30, ... (epoch010.pt is
        # the model after 10 completed epochs).
        epoch_number = epoch + 1
        if checkpoint_dir is not None and epoch_number % checkpoint_every == 0:
            save_model_checkpoint(model, checkpoint_dir, checkpoint_prefix, epoch_number)

    # Always keep the final epoch, even when it is not a multiple of the interval.
    if checkpoint_dir is not None and epochs % checkpoint_every != 0:
        save_model_checkpoint(model, checkpoint_dir, checkpoint_prefix, epochs)

    return model


def train_ensemble(
    x_train_list,
    y_train,
    eval_x_list=None,
    eval_y=None,
    hidden_dims=(20, 40, 60),
    seeds=range(10),
    lr=0.05,
    epochs=500,
    burn_in=20,
    batch_size=32,
    score_activation="sigmoid",
    device="cpu",
    checkpoint_root=None,
    checkpoint_every=10,
):
    train_dataset = PeptideWindowDataset(x_train_list, y_train)
    input_dim = train_dataset[0][0].shape[1]
    class_weights = compute_simple_class_weights(y_train)

    ensemble = []

    for hidden_dim in hidden_dims:
        for seed in seeds:
            print(f"Training model: hidden_dim={hidden_dim}, seed={seed}")
            if checkpoint_root is None:
                member_dir = None
                member_prefix = None
            else:
                member_prefix = f"netmhciipan_model_hd{hidden_dim}_seed{seed}"
                member_dir = os.path.join(checkpoint_root, member_prefix)
            model = train_one_model(
                train_dataset=train_dataset,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seed=seed,
                class_weights=class_weights,
                eval_x_list=eval_x_list,
                eval_y=eval_y,
                lr=lr,
                epochs=epochs,
                burn_in=burn_in,
                batch_size=batch_size,
                score_activation=score_activation,
                device=device,
                checkpoint_dir=member_dir,
                checkpoint_prefix=member_prefix,
                checkpoint_every=checkpoint_every,
            )
            ensemble.append({"hidden_dim": hidden_dim, "seed": seed, "model": model})

    return ensemble


def compute_simple_class_weights(y):
    y = np.asarray(y, dtype=np.float32)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    if n0 == 0 or n1 == 0:
        return {0.0: 1.0, 1.0: 1.0}

    if n0 < n1:
        weights = {0.0: n1 / n0, 1.0: 1.0}
    elif n1 < n0:
        weights = {0.0: 1.0, 1.0: n0 / n1}
    else:
        weights = {0.0: 1.0, 1.0: 1.0}

    print(
        "Class balance:"
        f" n0={n0}, n1={n1},"
        f" w0={weights[0.0]:.4f}, w1={weights[1.0]:.4f}"
    )
    return weights


def get_model_config_from_index(model_index):
    hidden_dims = (20, 40, 60)
    seeds = range(5)
    model_configs = [
        {"hidden_dim": hidden_dim, "seed": seed}
        for hidden_dim in hidden_dims
        for seed in seeds
    ]

    if model_index < 1 or model_index > len(model_configs):
        raise ValueError(
            f"Model index must be between 1 and {len(model_configs)}, got {model_index}"
        )

    return model_configs[model_index - 1]


def train_ensemble_from_dataframe(
    df,
    peptide_col="Peptide",
    hla_col="HLA",
    score_col="score",
    hidden_dims=(20, 40, 60),
    seeds=range(10),
    lr=0.05,
    epochs=500,
    burn_in=20,
    batch_size=32,
    score_activation="sigmoid",
    device="cpu",
):
    x_train_list, y_train = dataframe_to_window_dataset(
        df=df,
        peptide_col=peptide_col,
        hla_col=hla_col,
        score_col=score_col,
    )
    return train_ensemble(
        x_train_list=x_train_list,
        y_train=y_train,
        eval_x_list=x_train_list,
        eval_y=y_train,
        hidden_dims=hidden_dims,
        seeds=seeds,
        lr=lr,
        epochs=epochs,
        burn_in=burn_in,
        batch_size=batch_size,
        score_activation=score_activation,
        device=device,
    )


@torch.no_grad()
def predict_dataset_single_model(model, x_list, batch_size=64, device="cpu"):
    dummy_y = np.zeros(len(x_list), dtype=np.float32)
    dataset = PeptideWindowDataset(x_list, dummy_y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=peptide_collate_fn,
    )

    model.eval()
    all_peptide_scores = []

    for flat_x, segment_ids, y_batch in loader:
        flat_x = flat_x.to(device)
        segment_ids = segment_ids.to(device)

        peptide_scores, _ = model(
            flat_x=flat_x,
            segment_ids=segment_ids,
            n_peptides=y_batch.shape[0],
        )
        all_peptide_scores.append(peptide_scores.cpu())

    return torch.cat(all_peptide_scores, dim=0).numpy()


@torch.no_grad()
def predict_ensemble(ensemble, x_list, batch_size=64, device="cpu"):
    preds = []

    for entry in ensemble:
        model = entry["model"]
        preds.append(
            predict_dataset_single_model(
                model=model,
                x_list=x_list,
                batch_size=batch_size,
                device=device,
            )
        )

    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0)


def predict_dataframe(
    ensemble,
    df,
    peptide_col="Peptide",
    hla_col="HLA",
    score_col="score",
    batch_size=64,
    device="cpu",
):
    x_list, _ = dataframe_to_window_dataset(
        df=df,
        peptide_col=peptide_col,
        hla_col=hla_col,
        score_col=score_col,
    )
    predictions = predict_ensemble(
        ensemble=ensemble,
        x_list=x_list,
        batch_size=batch_size,
        device=device,
    )

    if hasattr(df, "columns") and hasattr(df, "loc"):
        result = df.copy()
        result["predicted_score"] = predictions
        return result

    result = [dict(row) for row in df]
    for row, pred in zip(result, predictions):
        row["predicted_score"] = float(pred)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python netmhciipan_code.py <model_index: 1-15>")

    model_index = int(sys.argv[1])
    model_config = get_model_config_from_index(model_index)
    print(
        f"Selected model_index={model_index} -> "
        f"hidden_dim={model_config['hidden_dim']}, seed={model_config['seed']}"
    )

    full_data = filter_standard_peptides(full_data)
    # full_data = stratified_sample_dataframe(
    #     full_data,
    #     n_samples=5000,
    #     stratify_col="score",
    #     random_state=42,
    # )
    print("Sampled rows:", len(full_data))
    print("Sampled score counts:")
    print(full_data["score"].value_counts().sort_index())
    x_train_list, y_train = dataframe_to_window_dataset(full_data)
    #print("Per-peptide window counts:", [x.shape[0] for x in x_train_list])
    print("Feature dimension:", x_train_list[0].shape[1])
    print("Expected feature dimension:", TOTAL_FEATURE_DIM)
    print_tensor_stats("first_peptide_windows", x_train_list[0])
    print_tensor_stats("y_train", y_train)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = compute_simple_class_weights(y_train)
    train_dataset = PeptideWindowDataset(x_train_list, y_train)

    # Checkpoints go into output_data/<run_name>/, which is where
    # netmhciipan_test.py and select_epoch_from_validation.py look for them.
    save_dir = OUTPUT_DIR
    # Directory name: model index + the fixed pepseq panel + the varying
    # netmhciipan panel + hidden dim + seed. No epoch here -- every checkpoint of
    # this run lives in this one directory and is told apart by the _epoch<NNN>
    # suffix on the file name.
    # '999' marks a run whose netmhciipan peptides all come from the BIG file
    # (netmhciipan_ba_peptides_full.csv). N=999 is itself the full DP/DQ panel, so
    # it already carries the tag; every other N gets it appended after the count.
    _nmp_tag = (f"{number_of_netmhciipan_hlas}netmhciipan_hlas"
                if FULL_DP_DQ_RUN else
                f"{number_of_netmhciipan_hlas}netmhciipan_hlas_{FULL_DP_DQ_N}")
    run_name = (
        f"netmhciipan_model_{model_index:02d}"
        f"_{FIXED_PEPSEQ_HLAS}_pepseq_hlas_fixed"
        f"_{_nmp_tag}"
        f"_hd{model_config['hidden_dim']}"
        f"_seed{model_config['seed']}_new_split"
    )
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoint directory: {run_dir}")

    print(
        f"Training single model: hidden_dim={model_config['hidden_dim']}, "
        f"seed={model_config['seed']}"
    )
    model = train_one_model(
        train_dataset=train_dataset,
        input_dim=TOTAL_FEATURE_DIM,
        hidden_dim=model_config["hidden_dim"],
        seed=model_config["seed"],
        class_weights=class_weights,
        eval_x_list=x_train_list,
        eval_y=y_train,
        lr=0.05,
        epochs=500,
        burn_in=20,
        batch_size=32,
        score_activation="sigmoid",
        device=device,
        checkpoint_dir=run_dir,
        checkpoint_prefix=run_name,
        checkpoint_every=10,
    )

    print(f"Finished training. Checkpoints are in: {run_dir}")

    predictions = predict_dataset_single_model(
        model=model,
        x_list=x_train_list,
        batch_size=32,
        device=device,
    )
    prediction_df = full_data.copy()
    prediction_df["predicted_score"] = predictions
    print(prediction_df)
