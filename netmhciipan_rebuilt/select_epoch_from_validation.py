"""
NETMHCIIPAN-REBUILT -- VALIDATION / EPOCH SELECTION
===================================================
(cleaned from validation_run.py)

Stage 3 of the netmhciipan-rebuilt pipeline:

    netmhciipan_code.py              <- fits the model, one checkpoint every
                                        10 epochs
    netmhciipan_test.py              <- scores peptides with EVERY saved
                                        checkpoint
    select_epoch_from_validation.py  <- THIS FILE
    evaluate_on_test_set.py          <- applies the selected epochs to the
                                        held-out test alleles

ONLY NEEDED AFTER RETRAINING
----------------------------
This step chose the epochs of the released models, and its result is already
baked into the weights and prediction CSVs in the release. Reproducing it needs
every checkpoint's prediction CSV, which is not part of the release; this
script is here for the record and for anyone who retrains with
netmhciipan_code.py and then runs netmhciipan_test.py over all checkpoints.

WHAT THIS FILE IS FOR
---------------------
This is NOT simply "run the model on the validation set" -- that already
happened in netmhciipan_test.py, which wrote one prediction CSV per checkpoint.
This script is the MODEL-SELECTION step: it decides WHICH TRAINING EPOCH to
use, and it is the only place the validation labels influence a choice.

One job handles one trained run (one model_index, one netmhciipan allele
count). Every prediction CSV in that run's directory -- one per 10-epoch
checkpoint -- is evaluated on the validation alleles, and each epoch is written
out as its own result. The epoch with the best validation performance is the
one carried forward; the selected epoch is then applied unchanged to the
held-out test set in evaluate_on_test_set.py.

HOW AN EPOCH IS SCORED (the "frank" ratio)
------------------------------------------
Validation performance is not a plain AUC over all peptides. For each known
epitope (`Original`) the model scores that epitope together with the decoy /
background peptides of the same group, and what is recorded is the epitope's
rank WITHIN its own group:

    ratio = (# peptides in the group scoring at least as well as the true
              epitope) / (group size)

so a small ratio means the true epitope was ranked near the top. `1 - ratio` is
reported, making it read like an AUC (higher is better). This is computed by
compute_ratios(); mean_ci() then aggregates it across epitopes with a 95%
confidence interval.

Two complications handled in compute_ratios():
  * the other known true epitopes of the same allele (from
    known_epitopes_per_allele.csv) are excluded from a group, so a second true
    epitope cannot count as a decoy;
  * if a group contains more than one positive, sequence_reconstructer()
    stitches the overlapping kmers back into contiguous sequences and each
    reconstructed stretch is ranked separately.

The same ratio is computed for the comparison models, so every epoch is
reported alongside the baselines it is being judged against: NetMHCIIpan, FCM,
the MHCIIFold-GNN score, and the MHCIIFold-GNN + NetMHCIIpan combination
(`MODELS` / `custom_order`).

COMMAND LINE
------------
    python select_epoch_from_validation.py [--model-index I] [--netmhciipan-n N]

    --model-index I    which trained run, 1..15. The index fixes
                       (hidden_dim, seed) through the same table
                       netmhciipan_code.py uses. Default 1.
    --netmhciipan-n N  how many netmhciipan alleles that run was trained on
                       (the pepseq side is fixed at 14). Default 20.
    --run-dir DIR      where that run's per-epoch prediction CSVs are. Default
                       output_data/<run_name>/. The released repo does not ship
                       them (see ONLY NEEDED AFTER RETRAINING above), so point
                       this at wherever netmhciipan_test.py wrote them.

    There is deliberately no epoch flag: ALL of the run's prediction CSVs are
    evaluated in one job, so a single job covers the whole epoch axis of one
    run.

OUTPUT
------
Per netmhciipan test epoch, two pickles in OUT_DIR:

    validation_results_nmpmodel<II>_nmpn<N>_nmpepoch<NNN>.pkl
        the summary table (mean and 95% CI per model)
    validation_everything_df_nmpmodel<II>_nmpn<N>_nmpepoch<NNN>.pkl
        the underlying per-epitope rows

Each is written atomically, and the best epoch is printed at the end.

NOTE ON PATHS: known_epitopes_per_allele.csv is read from input_data/; the
per-allele <HLA>_validation_final.csv files, the netmhciipan_test.py prediction
CSVs and all outputs live in output_data/. Both folders sit next to this script
(see the README in each folder).
"""

import glob
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

# Inputs are read from input_data/; the per-allele tables, the prediction CSVs
# and all outputs live in output_data/. Both folders sit next to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'input_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_data')


def _pop_flag_value(flag, default):
    """Remove `flag VALUE` from sys.argv and return VALUE (or `default`)."""
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        val = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2]
        return val
    return default


# Which trained run to validate. The index fixes (hidden_dim, seed); the count
# is how many netmhciipan alleles the run was trained on.
MODEL_INDEX = int(_pop_flag_value('--model-index', 1))
N_NETMHCIIPAN = int(_pop_flag_value('--netmhciipan-n', 20))
RUN_DIR_OVERRIDE = _pop_flag_value('--run-dir', None)
FIXED_PEPSEQ_HLAS = 14

NMP_LABEL = 'Netmhciipan-rebuilt-new-train'   # df label of the rebuilt score


# =====================================================================================
#   1) VALIDATION PANEL
# =====================================================================================
# The validation alleles swept below.
sqq = ['DRB5_0101', 'HLA-DQA10301-DQB10302', 'HLA-DQA10102-DQB10602', 'DRB1_0404',
       'DRB1_0405', 'DRB1_0901', 'DRB1_1201', 'DRB1_0802', 'DRB1_1502', 'DRB1_1401',
       'DRB1_0801', 'DRB1_1104', 'DRB3_0101', 'DRB4_0103', 'HLA-DPA10103-DPB10401',
       'HLA-DQA10501-DQB10302']

# Every known (epitope, allele) pair. Used to keep the other true epitopes of
# an allele out of each epitope's decoy group (see compute_ratios).
known_epitopes = pd.read_csv(os.path.join(INPUT_DIR, 'known_epitopes_per_allele.csv'))


def known_epitopes_of(HLA):
    """All known true epitopes (`Original` sequences) of one allele."""
    hla_name = f'DRB1_{HLA}' if len(HLA) == 4 else HLA
    return known_epitopes.loc[known_epitopes['HLA'] == hla_name, 'Original'].unique()


# =====================================================================================
#   2) THE FRANK-RATIO MACHINERY
# =====================================================================================
# A group is one `Original` peptide: the true epitope plus the decoy/background
# peptides tiled across it. The metric is where the true epitope ranks inside
# its own group (see the module docstring).
def sequence_reconstructer(df, original):
    """Stitch overlapping kmers back into contiguous sequences, labelled 'class'.

    Needed when one group holds more than one positive: the kmers are chained
    by their overlap (forward and backward from each unused kmer) so each
    contiguous stretch becomes its own class and is ranked separately.
    Without this, two true epitopes in one group would each count as a decoy
    for the other.
    """
    used = set()
    groups = []
    df = df.reset_index(drop=True)

    for idx in range(len(df)):
        if idx in used:
            continue
        sequence = df.loc[idx, 'Peptide']
        current_group = [idx]
        used.add(idx)

        # Extend forward.
        last_kmer = sequence
        while True:
            suffix = last_kmer[1:]
            match = df[~df.index.isin(used) & df['Peptide'].str.startswith(suffix)]
            if match.empty:
                break
            next_idx = match.index[0]
            next_kmer = match.iloc[0]['Peptide']
            sequence += next_kmer[-1]
            current_group.append(next_idx)
            used.add(next_idx)
            last_kmer = next_kmer

        # Extend backward.
        first_kmer = df.loc[current_group[0], 'Peptide']
        while True:
            prefix = first_kmer[:-1]
            match = df[~df.index.isin(used) & df['Peptide'].str.endswith(prefix)]
            if match.empty:
                break
            prev_idx = match.index[0]
            prev_kmer = match.iloc[0]['Peptide']
            sequence = prev_kmer[0] + sequence
            current_group.insert(0, prev_idx)
            used.add(prev_idx)
            first_kmer = prev_kmer

        groups.append({'reconstructed_sequence': sequence, 'indices': current_group})

    # Class numbers start at 1.
    df['class'] = -1
    for class_id, group in enumerate(groups):
        for idx in group['indices']:
            df.at[idx, 'class'] = class_id + 1

    return df


def compute_ratios(grouped, score_col, df_label, HLA, known_epitopes, ratios):
    """Within-group frank ratio for one model (score_col) / label (df_label).

    `known_epitopes` are all true epitopes (`Original` sequences) of this
    allele; the others are removed from each group so a second true epitope
    cannot count as a decoy. Results are appended in place to `ratios`.
    """
    known_epitopes = pd.Series(known_epitopes)

    for original, group in grouped:
        score_1_row = group[group['score'] == 1]

        # Other true epitopes of this allele must not count as decoys.
        other_epitopes = known_epitopes[known_epitopes != original]

        # More than one positive: reconstruct the contiguous sequences and
        # score each reconstructed class separately.
        if len(score_1_row) > 1:
            group = sequence_reconstructer(group, original)
            for ccc, group2 in group.groupby('class'):
                score_1_row2 = group2[group2['score'] == 1]
                group2 = group2[~group2.Peptide.isin(other_epitopes)]

                if len(group2) == 0:
                    continue

                score_1_preds_value2 = score_1_row2[score_col].iloc[0]
                ratio2 = (group2[score_col] <= score_1_preds_value2).sum() / len(group2)
                ratios.append({'Label': HLA, 'Original': original + '_' + f'{ccc}',
                               'Ratio': ratio2, 'df': df_label})
            continue

        if len(score_1_row) == 0:
            continue

        score_1_preds_value = score_1_row[score_col].iloc[0]
        group = group[~group.Peptide.isin(other_epitopes)]

        if len(group) == 0:
            continue

        ratio = (group[score_col] <= score_1_preds_value).sum() / len(group)
        ratios.append({'Label': HLA, 'Original': original,
                       'Ratio': ratio, 'df': df_label})


def mean_ci(series, confidence=0.95):
    """Mean with a normal-approximation confidence interval."""
    n = len(series)
    mean = series.mean()
    sem = stats.sem(series)
    z = stats.norm.ppf((1 + confidence) / 2.)
    h = sem * z if n > 1 else 0
    return pd.Series({'mean': mean, 'ci_lower': mean - h, 'ci_upper': mean + h})


def rank_normalise(series):
    """Average rank, ascending, divided by the max rank."""
    r = series.rank(method='average', ascending=True)
    return r / r.max()


# =====================================================================================
#   3) THE RUN BEING VALIDATED, AND ITS PER-EPOCH PREDICTIONS
# =====================================================================================
def model_config_from_index(model_index):
    """Same index -> (hidden_dim, seed) table as netmhciipan_code.py."""
    configs = [{'hidden_dim': hd, 'seed': s}
               for hd in (20, 40, 60) for s in range(5)]
    if model_index < 1 or model_index > len(configs):
        raise ValueError(f'--model-index must be 1..{len(configs)}, got {model_index}')
    return configs[model_index - 1]


def run_name_of(model_index, n_netmhciipan, config):
    """The run directory name netmhciipan_code.py wrote."""
    return (f'netmhciipan_model_{model_index:02d}'
            f'_{FIXED_PEPSEQ_HLAS}_pepseq_hlas_fixed'
            f'_{n_netmhciipan}netmhciipan_hlas'
            f'_hd{config["hidden_dim"]}'
            f'_seed{config["seed"]}_new_split')


MODEL_CONFIG = model_config_from_index(MODEL_INDEX)
RUN_NAME = run_name_of(MODEL_INDEX, N_NETMHCIIPAN, MODEL_CONFIG)
# The per-epoch prediction CSVs are not part of the release, so --run-dir
# points this at wherever netmhciipan_test.py wrote them.
RUN_DIR = (os.path.abspath(os.path.expanduser(RUN_DIR_OVERRIDE))
           if RUN_DIR_OVERRIDE else os.path.join(OUTPUT_DIR, RUN_NAME))

if not os.path.isdir(RUN_DIR):
    raise FileNotFoundError(
        f'Run directory does not exist: {RUN_DIR}\n'
        'Check --model-index / --netmhciipan-n against the trained runs, or pass\n'
        '--run-dir with the directory holding that run\'s per-epoch\n'
        '*_test_predictions.csv files.')


def epoch_of(path):
    """The NNN in <run_name>_epoch<NNN>_test_predictions.csv."""
    m = re.search(r'_epoch(\d+)_test_predictions\.csv$', os.path.basename(path))
    return int(m.group(1)) if m else None


prediction_csv = {}
for path in glob.glob(os.path.join(RUN_DIR, '*_test_predictions.csv')):
    epoch = epoch_of(path)
    if epoch is not None:
        prediction_csv[epoch] = path
EPOCHS = sorted(prediction_csv)

if not EPOCHS:
    raise FileNotFoundError(
        f'No *_epoch<NNN>_test_predictions.csv in {RUN_DIR}\n'
        'Has netmhciipan_test.py been run over the checkpoints of this run?')

OUT_DIR = os.path.join(OUTPUT_DIR, 'validation_netmhciipan')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TAG = f'nmpmodel{MODEL_INDEX:02d}_nmpn{N_NETMHCIIPAN}'

print('=' * 70)
print(f'VALIDATION   netmhciipan model_index={MODEL_INDEX} '
      f'(hidden_dim={MODEL_CONFIG["hidden_dim"]}, seed={MODEL_CONFIG["seed"]})')
print(f'netmhciipan HLAs   : {N_NETMHCIIPAN}   pepseq HLAs: {FIXED_PEPSEQ_HLAS} (fixed)')
print(f'run directory      : {RUN_DIR}')
print(f'test epochs        : {len(EPOCHS)}  {EPOCHS[0]}..{EPOCHS[-1]}')
print(f'validation HLAs    : {len(sqq)}  {sqq}')
print(f'output pickles     : {2 * len(EPOCHS)} files in {OUT_DIR}')
print(f'                     validation_results_{OUT_TAG}_nmpepoch<NNN>.pkl')
print(f'                     validation_everything_df_{OUT_TAG}_nmpepoch<NNN>.pkl')
print('=' * 70, flush=True)

# epoch -> {HLA -> Series(Peptide -> predicted_score)}, validation alleles only.
# One pass over the CSVs, so no file is read twice and only the sqq alleles are
# held in memory.
score_maps = {}
for epoch in EPOCHS:
    path = prediction_csv[epoch]
    df = pd.read_csv(path)
    for col in ('Peptide', 'real_HLA', 'predicted_score'):
        if col not in df.columns:
            raise ValueError(f"{path} has no '{col}' column; found {list(df.columns)}")
    df = df[df['real_HLA'].isin(sqq)]
    per_hla = {}
    for hla, g in df.groupby('real_HLA', sort=False):
        g = g.drop_duplicates(subset='Peptide', keep='first').copy()
        g['Peptide'] = g['Peptide'].astype(str).str.strip()
        per_hla[hla] = g.set_index('Peptide')['predicted_score']
    score_maps[epoch] = per_hla
    print(f'[nmp] epoch{epoch:03d}: {len(df)} rows over {len(per_hla)} validation HLAs',
          flush=True)


def nmp_col(epoch):
    """Column of `a` holding this epoch's rank-normalised netmhciipan score."""
    return f'netmhciipan_rebuilt_ranked_ep{epoch:03d}'


# Epoch-independent comparison models: (score column, label).
MODELS = [
    ('min_value_combination',  'MHCIIFold-GNN+NetMHCIIPan'),
    ('min_value_GNN',          'MHCIIFold-GNN'),
    ('min_value_Original',     'Original'),
    ('Rank',                   ' NetMHCIIPan'),
    ('Geomean_FCM',            'FCM'),
]
# The rebuilt netmhciipan model is NOT in MODELS: its ratios depend on which
# test epoch is being written, so they are precomputed per (HLA, epoch) below.

# Every label must appear here: pivot_df casts 'df' to a Categorical over
# custom_order, so a missing label would silently become NaN.
custom_order = [' NetMHCIIPan', NMP_LABEL, 'FCM', 'MHCIIFold-GNN',
                'Original', 'MHCIIFold-GNN+NetMHCIIPan']


# =====================================================================================
#   4) Per-HLA preparation, done ONCE
# =====================================================================================
# Nothing here depends on the epoch except the netmhciipan columns, and those are
# one column per epoch, so the whole epoch axis is covered in this single pass.
base_ratios = {}   # HLA -> ratio dicts for the comparison MODELS
nmp_ratios  = {}   # HLA -> {epoch -> ratio dicts for NMP_LABEL}

for HLA in sqq:
    a = pd.read_csv(os.path.join(OUTPUT_DIR, f'{HLA}_validation_final.csv'))
    a = a.copy()

    a['Peptide'] = a['Peptide'].astype(str).str.strip()

    # Rebuilt netmhciipan: map (NOT merge, so no row fan-out) this run's
    # predicted_score onto `a` by peptide, one column per epoch. predicted_score
    # is a raw sigmoid output, not a percentile, so it is rank-normalised exactly
    # like the other raw scores.
    for epoch in EPOCHS:
        score_map = score_maps[epoch].get(HLA)
        if score_map is None:
            a[nmp_col(epoch)] = np.nan
        else:
            a[nmp_col(epoch)] = rank_normalise(a['Peptide'].map(score_map))

    a['GeoMean_letter_ranked'] = rank_normalise(a['GeoMean_letter'])
    a['GeoMean_conf_ranked']   = rank_normalise(a['GeoMean_conf'])
    a['Geomean_FCM_ranked']    = rank_normalise(a['Geomean_FCM'])

    a['min_value_Original']    = ((a['GeoMean_conf_ranked'] * a['GeoMean_letter_ranked'])) ** (1 / 2)
    # MHCIIFold-GNN score: geometric mean of the letter and conf ranks.
    a['min_value_GNN']         = (a['GeoMean_letter_ranked'] * a['GeoMean_conf_ranked']) ** (1 / 2)
    a['min_value_combination'] = ((a['GeoMean_conf_ranked'] * a['GeoMean_letter_ranked']) * a['Rank']) ** (1 / 3)

    first_col = nmp_col(EPOCHS[0])
    n_mapped = a[first_col].notna().sum()
    n_peptides = len(score_maps[EPOCHS[0]].get(HLA, []))
    print(f'{HLA}: rows={len(a)}, positives {len(a[a["score"] == 1])}, '
          f'netmhciipan mapped for {n_mapped}/{len(a)} rows '
          f'(CSV peptides for this HLA: {n_peptides})', flush=True)

    grouped = a.groupby('Original')

    ratios = []
    for score_col, df_label in MODELS:
        compute_ratios(grouped, score_col, df_label, HLA,
                       known_epitopes_of(HLA), ratios)
    base_ratios[HLA] = ratios

    nmp_ratios[HLA] = {}
    for epoch in EPOCHS:
        per_epoch = []
        compute_ratios(grouped, nmp_col(epoch), NMP_LABEL, HLA,
                       known_epitopes_of(HLA), per_epoch)
        nmp_ratios[HLA][epoch] = per_epoch


# =====================================================================================
#   5) One summary + one everything_df per test epoch
# =====================================================================================
def out_paths(epoch):
    """The two pickles for one netmhciipan test epoch."""
    tag = f'{OUT_TAG}_nmpepoch{epoch:03d}'
    return (os.path.join(OUT_DIR, f'validation_results_{tag}.pkl'),
            os.path.join(OUT_DIR, f'validation_everything_df_{tag}.pkl'))


def save_one(epoch, pivot_df, everything_df):
    """Write both pickles for one epoch atomically (temp file + rename)."""
    meta = {
        'hlas':                          list(sqq),
        'netmhciipan_rebuilt_label':     NMP_LABEL,
        'netmhciipan_model_index':       MODEL_INDEX,
        'netmhciipan_n_hlas':            N_NETMHCIIPAN,
        'netmhciipan_fixed_pepseq_hlas': FIXED_PEPSEQ_HLAS,
        'netmhciipan_test_epoch':        epoch,
        'netmhciipan_all_test_epochs':   EPOCHS,
        'netmhciipan_prediction_csv':    prediction_csv[epoch],
        'netmhciipan_run_dir':           RUN_DIR,
    }
    for path, obj in zip(out_paths(epoch),
                         (dict(meta, results=pivot_df),
                          dict(meta, results=everything_df))):
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)


t0 = time.time()
mean_per_epoch = {}

for epoch in EPOCHS:
    rows = []
    for HLA in sqq:
        rows.extend(base_ratios[HLA])
        rows.extend(nmp_ratios[HLA][epoch])

    if not rows:
        print(f'[warn] epoch{epoch:03d}: no ratios computed, skipping', flush=True)
        continue

    # Original and the GNN+NetMHCIIpan combination are left out of this table.
    # Report 1 - ratio so it reads like an AUC (higher is better).
    everything_df = pd.DataFrame(rows)
    everything_df = everything_df[
        ~everything_df['df'].isin(['Original', 'MHCIIFold-GNN+NetMHCIIPan'])].copy()
    everything_df['Ratio'] = 1 - everything_df['Ratio']

    summary_df = everything_df.groupby('df')['Ratio'].apply(mean_ci).reset_index()
    pivot_df = summary_df.pivot(index='df', columns='level_1', values='Ratio').reset_index()
    pivot_df['df'] = pd.Categorical(pivot_df['df'], categories=custom_order, ordered=True)
    pivot_df = pivot_df.sort_values('df').reset_index(drop=True)
    pivot_df = pivot_df[['df', 'mean', 'ci_lower', 'ci_upper']]
    pivot_df.columns.name = None

    everything_df = everything_df.reset_index(drop=True)

    nmp_mean = pivot_df.loc[pivot_df['df'] == NMP_LABEL, 'mean']
    nmp_mean = float(nmp_mean.iloc[0]) if len(nmp_mean) else float('nan')
    mean_per_epoch[epoch] = nmp_mean

    print(f'[epoch{epoch:03d}] {NMP_LABEL} mean AUC={nmp_mean:.4f}  '
          f'({len(everything_df) // everything_df["df"].nunique()} epitopes, '
          f'{(time.time() - t0) / 60:.1f} min)', flush=True)
    print(pivot_df.to_string(index=False), flush=True)

    save_one(epoch, pivot_df, everything_df)


# =====================================================================================
#   6) The selected epoch
# =====================================================================================
print('=' * 70)
print(f'DONE  model_index={MODEL_INDEX} n={N_NETMHCIIPAN}: {len(mean_per_epoch)} epoch(s), '
      f'{2 * len(mean_per_epoch)} pickles, {(time.time() - t0) / 60:.1f} min')
if mean_per_epoch:
    best_epoch = max(mean_per_epoch, key=lambda e: mean_per_epoch[e])
    print(f'best epoch for {NMP_LABEL}: {best_epoch} '
          f'(mean AUC={mean_per_epoch[best_epoch]:.4f})')
    print('this is the epoch to put in SELECTED_EPOCHS in evaluate_on_test_set.py, '
          f'as  {MODEL_INDEX}: {best_epoch},')
print(f'pickles written to : {OUT_DIR}')
print('=' * 70, flush=True)
