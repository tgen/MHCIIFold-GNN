"""
NETMHCIIPAN-REBUILT -- HELD-OUT TEST SET EVALUATION
===================================================

Final stage of the netmhciipan-rebuilt pipeline:

    netmhciipan_code.py              <- fits the model, one checkpoint every
                                        10 epochs
    netmhciipan_test.py              <- scores peptides with EVERY saved
                                        checkpoint
    select_epoch_from_validation.py  <- picks one epoch per trained run on the
                                        validation alleles
    evaluate_on_test_set.py          <- THIS FILE: scores the held-out test
                                        alleles with those epochs

WHAT THIS FILE DOES
-------------------
Applies the epochs chosen by select_epoch_from_validation.py, unchanged, to the
held-out test alleles. Nothing is selected here: the epochs are fixed in
SELECTED_EPOCHS below, so the test labels never influence a choice.

The netmhciipan-rebuilt score is an ENSEMBLE of the 15 trained models. Each
model_index contributes the predictions of its own selected epoch; the raw
predicted_score values are averaged across the 15 models per peptide, and the
mean is then rank-normalised per allele -- the same averaging netmhciipan_code.py
uses across replicates, and the same rank normalisation the other raw scores get.

Performance is the within-group "frank" ratio of each known epitope, reported as
1 - ratio with a 95% confidence interval, alongside the same comparison models
(NetMHCIIpan, FCM, the MHCIIFold-GNN score and MHCIIFold-GNN + NetMHCIIpan).
See select_epoch_from_validation.py for the full description of the metric.

COMMAND LINE
------------
    python evaluate_on_test_set.py

    No arguments. The models, their epochs and the panel are all fixed above
    so the published numbers are reproduced exactly.

INPUT
-----
One prediction CSV per model, already at its selected epoch, named after its
training run, in output_data/:

    netmhciipan_model_<II>_14_pepseq_hlas_fixed_<N>netmhciipan_hlas_hd<HD>.csv

Each holds Peptide, real_HLA and predicted_score for every allele, so the test
alleles are taken by filtering on real_HLA -- the same CSVs serve validation and
test, split by allele.

OUTPUT
------
    test_results_netmhciipan_n<N>.pkl in OUT_DIR, holding the model/epoch table,
    the summary table (mean and 95% CI per model) and the per-epitope rows.

NOTE ON PATHS: known_epitopes_per_allele.csv is read from input_data/; the
per-allele <HLA>_test_final.csv files, the prediction CSVs and all outputs live
in output_data/. Both folders sit next to this script (see the README in each
folder).
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats

# Inputs are read from input_data/; the per-allele tables, the prediction CSVs
# and all outputs live in output_data/. Both folders sit next to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'input_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_data')

# The per-model prediction CSVs live directly in output_data/, beside the
# per-allele <HLA>_test_final.csv tables.
PREDICTION_DIR = OUTPUT_DIR

N_NETMHCIIPAN = 20          # netmhciipan alleles the released models trained on
FIXED_PEPSEQ_HLAS = 14      # the pepseq side is fixed

NMP_LABEL = 'Netmhciipan-rebuilt-new-train'   # df label of the rebuilt score

# The epoch chosen on the validation alleles for each trained model, from
# select_epoch_from_validation.py. Fixed here so the test evaluation selects
# nothing: model_index -> epoch.
SELECTED_EPOCHS = {
    1: 430,
    2: 10,
    3: 10,
    4: 40,
    5: 10,
    6: 300,
    7: 150,
    8: 160,
    9: 120,
    10: 110,
    11: 260,
    12: 250,
    13: 60,
    14: 310,
    15: 10,
}

OUT_DIR = os.path.join(OUTPUT_DIR, 'test_netmhciipan')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PKL = os.path.join(OUT_DIR, f'test_results_netmhciipan_n{N_NETMHCIIPAN}.pkl')


# =====================================================================================
#   1) TEST PANEL
# =====================================================================================
# The held-out test alleles.
test_HLAs = ['DRB1_1202', 'DRB1_0301', 'DRB1_0408', 'DRB1_0803', 'DRB1_0818', 'DRB1_1103',
             'DRB1_1404', 'DRB1_1405', 'DRB1_0101', 'DRB1_0102', 'DRB1_0302', 'DRB1_0401',
             'DRB1_0402', 'DRB1_0403', 'DRB1_0701', 'DRB1_1101', 'DRB1_0407', 'DRB1_1001',
             'DRB1_1501', 'DRB1_1601', 'DRB1_1602', 'DRB1_1301', 'DRB1_1503', 'DRB1_1302',
             'DRB1_1303']
test_other_HLAs = ['DRB3_0202', 'DRB3_0301', 'DRB4_0101', 'DRB5_0101',
                   'HLA-DPA10103-DPB10201', 'HLA-DPA10201-DPB10501', 'HLA-DPA10301-DPB10402',
                   'HLA-DQA10101-DQB10201', 'HLA-DQA10101-DQB10501', 'HLA-DQA10103-DQB10601',
                   'HLA-DQA10103-DQB10603', 'HLA-DQA10301-DQB10201', 'HLA-DQA10301-DQB10301',
                   'HLA-DQA10301-DQB10302', 'HLA-DQA10401-DQB10402', 'HLA-DQA10501-DQB10201',
                   'HLA-DQA10501-DQB10202', 'HLA-DQA10501-DQB10301']
test_sqq = test_HLAs + test_other_HLAs

# Every known (epitope, allele) pair. Used to keep the other true epitopes of
# an allele out of each epitope's decoy group (see compute_ratios).
known_epitopes = pd.read_csv(os.path.join(INPUT_DIR, 'known_epitopes_per_allele.csv'))


def known_epitopes_of(HLA):
    """All known true epitopes (`Original` sequences) of one allele."""
    hla_name = f'DRB1_{HLA}' if len(HLA) == 4 else HLA
    return known_epitopes.loc[known_epitopes['HLA'] == hla_name, 'Original'].unique()


# =====================================================================================
#   2) THE FRANK-RATIO MACHINERY (identical to select_epoch_from_validation.py)
# =====================================================================================
def sequence_reconstructer(df, original):
    """Stitch overlapping kmers back into contiguous sequences, labelled 'class'.

    Needed when one group holds more than one positive: each contiguous
    stretch becomes its own class and is ranked separately.
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
#   3) THE ENSEMBLE: 15 models, each at its selected epoch
# =====================================================================================
def model_config_from_index(model_index):
    """Same index -> (hidden_dim, seed) table as netmhciipan_code.py."""
    configs = [{'hidden_dim': hd, 'seed': s}
               for hd in (20, 40, 60) for s in range(5)]
    if model_index < 1 or model_index > len(configs):
        raise ValueError(f'model_index must be 1..{len(configs)}, got {model_index}')
    return configs[model_index - 1]


def prediction_path(model_index):
    """This model's prediction CSV, already at its selected epoch."""
    config = model_config_from_index(model_index)
    name = (f'netmhciipan_model_{model_index:02d}'
            f'_{FIXED_PEPSEQ_HLAS}_pepseq_hlas_fixed'
            f'_{N_NETMHCIIPAN}netmhciipan_hlas'
            f'_hd{config["hidden_dim"]}.csv')
    return os.path.join(PREDICTION_DIR, name)


print('=' * 70)
print('TEST EVALUATION   netmhciipan-rebuilt')
print(f'ensemble          : {len(SELECTED_EPOCHS)} models, each at its '
      f'validation-selected epoch')
print(f'netmhciipan HLAs  : {N_NETMHCIIPAN}   pepseq HLAs: {FIXED_PEPSEQ_HLAS} (fixed)')
print(f'test HLAs         : {len(test_sqq)}')
print(f'output pickle     : {OUT_PKL}')
print('=' * 70, flush=True)

# Load the 15 prediction CSVs and average their predicted_score per peptide.
# Averaging the RAW scores (not the ranks) mirrors how netmhciipan_code.py
# combines replicates; the mean is rank-normalised per allele further down.
per_model = {}      # model_index -> DataFrame[Peptide, real_HLA, predicted_score]
missing_models = []

for model_index in sorted(SELECTED_EPOCHS):
    path = prediction_path(model_index)
    if not os.path.isfile(path):
        print(f'[warn] model {model_index:02d}: prediction CSV missing: {path}')
        missing_models.append((model_index, path))
        continue

    df = pd.read_csv(path)
    for col in ('Peptide', 'real_HLA', 'predicted_score'):
        if col not in df.columns:
            raise ValueError(f"{path} has no '{col}' column; found {list(df.columns)}")

    df = df[df['real_HLA'].isin(test_sqq)].copy()
    df['Peptide'] = df['Peptide'].astype(str).str.strip()
    # One row per (Peptide, allele): the same peptide can appear under several
    # Originals with the same score.
    df = df.drop_duplicates(subset=['real_HLA', 'Peptide'], keep='first')
    per_model[model_index] = df[['real_HLA', 'Peptide', 'predicted_score']]

    config = model_config_from_index(model_index)
    print(f'model {model_index:02d}  hd{config["hidden_dim"]:<3d} seed{config["seed"]}  '
          f'epoch {SELECTED_EPOCHS[model_index]:3d}  '
          f'{len(df)} test rows', flush=True)

if not per_model:
    raise FileNotFoundError(
        f'no prediction CSVs found in {PREDICTION_DIR}; expected one per model.')
if missing_models:
    print(f'[warn] ensembling {len(per_model)} of {len(SELECTED_EPOCHS)} models', flush=True)

# Mean predicted_score across the models, per (allele, peptide). A peptide
# missing from one model's CSV is averaged over the models that do have it,
# and the count is reported so a partial ensemble is visible.
stacked = pd.concat(per_model.values(), ignore_index=True)
ensemble = (stacked.groupby(['real_HLA', 'Peptide'])['predicted_score']
            .agg(['mean', 'count']).reset_index()
            .rename(columns={'mean': 'predicted_score', 'count': 'n_models'}))

n_full = int((ensemble['n_models'] == len(per_model)).sum())
print(f'ensemble rows     : {len(ensemble)}  '
      f'({n_full} averaged over all {len(per_model)} models)', flush=True)

# allele -> Series(Peptide -> ensemble predicted_score)
ensemble_of = {hla: g.set_index('Peptide')['predicted_score']
               for hla, g in ensemble.groupby('real_HLA', sort=False)}


MODELS = [
    ('min_value_combination',  'MHCIIFold-GNN+NetMHCIIPan'),
    ('min_value_GNN',          'MHCIIFold-GNN'),
    ('min_value_Original',     'Original'),
    ('Rank',                   ' NetMHCIIPan'),
    ('Geomean_FCM',            'FCM'),
    ('netmhciipan_rebuilt_ranked', NMP_LABEL),
]

custom_order = [' NetMHCIIPan', NMP_LABEL, 'FCM', 'MHCIIFold-GNN',
                'Original', 'MHCIIFold-GNN+NetMHCIIPan']


# =====================================================================================
#   4) Per-HLA evaluation
# =====================================================================================
ratios_per_hla = {}
missing_hlas = []

for HLA in test_sqq:
    csv_path = os.path.join(OUTPUT_DIR, f'{HLA}_test_final.csv')
    if not os.path.isfile(csv_path):
        print(f'[warn] {HLA}: no {os.path.basename(csv_path)}, skipping')
        missing_hlas.append((HLA, 'per-allele CSV missing'))
        continue

    score_map = ensemble_of.get(HLA)
    if score_map is None:
        print(f'[warn] {HLA}: no ensemble predictions for this allele, skipping')
        missing_hlas.append((HLA, 'no netmhciipan predictions'))
        continue

    a = pd.read_csv(csv_path)
    a = a.copy()

    a['Peptide'] = a['Peptide'].astype(str).str.strip()

    a['GeoMean_letter_ranked'] = rank_normalise(a['GeoMean_letter'])
    a['GeoMean_conf_ranked']   = rank_normalise(a['GeoMean_conf'])
    a['Geomean_FCM_ranked']    = rank_normalise(a['Geomean_FCM'])

    a['min_value_Original']    = ((a['GeoMean_conf_ranked'] * a['GeoMean_letter_ranked'])) ** (1 / 2)
    a['min_value_GNN']         = (a['GeoMean_letter_ranked'] * a['GeoMean_conf_ranked']) ** (1 / 2)
    a['min_value_combination'] = ((a['GeoMean_conf_ranked'] * a['GeoMean_letter_ranked']) * a['Rank']) ** (1 / 3)

    # Rebuilt netmhciipan: map (NOT merge, so no row fan-out) the ensemble score
    # onto `a` by peptide, then rank-normalise it like the other raw scores.
    a['netmhciipan_rebuilt'] = a['Peptide'].map(score_map)
    a['netmhciipan_rebuilt_ranked'] = rank_normalise(a['netmhciipan_rebuilt'])

    print(f'{HLA}: rows={len(a)}, positives {len(a[a["score"] == 1])}, '
          f'netmhciipan mapped for {a["netmhciipan_rebuilt"].notna().sum()}/{len(a)}',
          flush=True)

    grouped = a.groupby('Original')
    ratios = []
    for score_col, df_label in MODELS:
        compute_ratios(grouped, score_col, df_label, HLA,
                       known_epitopes_of(HLA), ratios)

    if not ratios:
        print(f'[warn] {HLA}: no ratios computed, skipping HLA', flush=True)
        missing_hlas.append((HLA, 'no ratios computed'))
        continue
    ratios_per_hla[HLA] = pd.DataFrame(ratios)

if missing_hlas:
    print(f'[warn] {len(missing_hlas)} test HLA(s) left out: {missing_hlas}', flush=True)
if not ratios_per_hla:
    raise RuntimeError('no test HLA could be evaluated')


# =====================================================================================
#   5) Summary and save
# =====================================================================================
combined_df = pd.concat(ratios_per_hla.values(), ignore_index=True)

# Original and the GNN+NetMHCIIpan combination are left out of the table, as in
# validation. Report 1 - ratio so it reads like an AUC (higher is better).
everything_df = combined_df[
    ~combined_df['df'].isin(['Original', 'MHCIIFold-GNN+NetMHCIIPan'])].copy()
everything_df['Ratio'] = 1 - everything_df['Ratio']

summary_df = everything_df.groupby('df')['Ratio'].apply(mean_ci).reset_index()
pivot_df = summary_df.pivot(index='df', columns='level_1', values='Ratio').reset_index()
pivot_df['df'] = pd.Categorical(pivot_df['df'], categories=custom_order, ordered=True)
pivot_df = pivot_df.sort_values('df').reset_index(drop=True)
pivot_df = pivot_df[['df', 'mean', 'ci_lower', 'ci_upper']]
pivot_df.columns.name = None

payload = {
    'netmhciipan_n_hlas':            N_NETMHCIIPAN,
    'netmhciipan_fixed_pepseq_hlas': FIXED_PEPSEQ_HLAS,
    'netmhciipan_rebuilt_label':     NMP_LABEL,
    'selected_epochs':               dict(SELECTED_EPOCHS),
    'models_ensembled':              sorted(per_model),
    'models_missing':                missing_models,
    'hlas':                          list(ratios_per_hla),
    'hlas_missing':                  missing_hlas,
    'summary':                       pivot_df,
    'everything':                    everything_df.reset_index(drop=True),
    'all_ratios':                    combined_df,
}
tmp = OUT_PKL + '.tmp'
with open(tmp, 'wb') as f:
    pickle.dump(payload, f)
os.replace(tmp, OUT_PKL)

print('=' * 70)
print(f'TEST RESULTS  (ensemble of {len(per_model)} models, {len(ratios_per_hla)} HLAs, '
      f'n={len(everything_df) // everything_df["df"].nunique()} epitopes)')
print(pivot_df.to_string(index=False))
print(f'pickle written to : {OUT_PKL}')
print('=' * 70, flush=True)
