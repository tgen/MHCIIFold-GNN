"""
MHCIIFold-GNN -- HELD-OUT TEST SET EVALUATION
=============================================

Final stage of the MHCIIFold-GNN pipeline:

    train_mhciifold_gnn.py            <- fits the model, one checkpoint every
                                         10 epochs
    test_mhciifold_gnn.py             <- scores peptides with EVERY saved
                                         checkpoint
    select_epochs_from_validation.py  <- picks the (all_aa, only_HLA_aa)
                                         epoch pair on the validation alleles
    evaluate_on_test_set.py           <- THIS FILE: scores the held-out test
                                         alleles with that one epoch pair

WHAT THIS FILE DOES
-------------------
Applies the epoch pair chosen by select_epochs_from_validation.py, unchanged,
to the held-out test alleles. Nothing is selected here: exactly one epoch pair
is evaluated, so the test labels never influence a choice.

The evaluation is identical to the validation step: the final MHCIIFold-GNN
score is the geometric mean of the rank-normalised peptide-aware ('all_aa') and
peptide-agnostic ('only_HLA_aa') scores, and performance is the within-group
"frank" ratio of each known epitope, reported as 1 - ratio with a 95%
confidence interval, alongside the same comparison models (NetMHCIIpan, FCM,
the original MHCIIFold-GNN score and MHCIIFold-GNN + NetMHCIIpan).
See select_epochs_from_validation.py for the full description of the metric.

COMMAND LINE
------------
    python evaluate_on_test_set.py [--epochs] [--n-hlas N] [--seed S]
                                   [--epoch-all-aa E1 --epoch-only-hla-aa E2]

    By default it reads the parquets written by test_mhciifold_gnn.py with the
    released weights (<HLA>_epitope_test_released_<aa_set>.parquet), which hold
    the scores of the published models. Nothing is selected here.

    --epochs              instead use the per-epoch parquets of a retrained
                          model, for one (all_aa, only_HLA_aa) epoch pair.
    --epoch-all-aa E1     epoch of the peptide-aware model (--epochs only).
    --epoch-only-hla-aa E2
                          epoch of the peptide-agnostic model (--epochs only).
                          If neither is given, the best pair is read from the
                          validation_results_nhlas<N>_seed<S>.pkl written by
                          select_epochs_from_validation.py.
    --n-hlas N            retrained allele-panel size (--epochs only). Default 32.
    --seed S              retrained replicate (--epochs only). Default 3.

OUTPUT
------
    test_results_nhlas<N>_seed<S>.pkl in OUT_DIR, holding the epoch pair, the
    summary table (mean and 95% CI per model) and the per-epitope rows.

NOTE ON PATHS: known_epitopes_per_allele.csv is read from input_data/; the
per-allele <HLA>_test_final.csv files, the test_mhciifold_gnn.py parquets
and all outputs live in output_data/. Both folders sit next to this script
(see the README in each folder).
"""

import os
import pickle
import sys

import pandas as pd
from scipy import stats

# Inputs are read from input_data/; the per-allele tables, the parquets and
# all outputs live in output_data/. Both folders sit next to this script.
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


def _pop_flag(flag):
    """Remove a valueless `flag` from sys.argv; return whether it was there."""
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


# Default: score from the released models' parquets. --epochs switches to the
# per-epoch parquets of a retrained model.
EPOCH_MODE = _pop_flag('--epochs')

N_HLAS = int(_pop_flag_value('--n-hlas', 32))
MODEL_SEED = int(_pop_flag_value('--seed', 3))
EPOCH_ALL_AA = _pop_flag_value('--epoch-all-aa', None)
EPOCH_ONLY_HLA_AA = _pop_flag_value('--epoch-only-hla-aa', None)

AA_SETS   = ('all_aa', 'only_HLA_aa')
NEW_LABEL = 'MHCIIFold-GNN-new'

VALIDATION_PKL = os.path.join(OUTPUT_DIR, 'validation_14_pepseq_hlas_fixed',
                              f'validation_results_nhlas{N_HLAS}_seed{MODEL_SEED}.pkl')
OUT_DIR = os.path.join(OUTPUT_DIR, 'test_14_pepseq_hlas_fixed')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PKL = os.path.join(OUT_DIR, f'test_results_nhlas{N_HLAS}_seed{MODEL_SEED}.pkl')


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
#   2) THE FRANK-RATIO MACHINERY (identical to select_epochs_from_validation.py)
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
#   3) THE EPOCH PAIR (chosen on the validation set, never here)
# =====================================================================================
if not EPOCH_MODE:
    if EPOCH_ALL_AA is not None or EPOCH_ONLY_HLA_AA is not None:
        raise ValueError('--epoch-all-aa / --epoch-only-hla-aa need --epochs')
    source = 'released weights'
else:
    if (EPOCH_ALL_AA is None) != (EPOCH_ONLY_HLA_AA is None):
        raise ValueError('give both --epoch-all-aa and --epoch-only-hla-aa, or neither')

    if EPOCH_ALL_AA is None:
        with open(VALIDATION_PKL, 'rb') as f:
            validation_results = pickle.load(f)['results']
        EPOCH_ALL_AA, EPOCH_ONLY_HLA_AA = max(
            validation_results,
            key=lambda k: float(validation_results[k].loc[
                validation_results[k]['df'] == NEW_LABEL, 'mean'].iloc[0]))
        source = f'best validation pair from {VALIDATION_PKL}'
    else:
        EPOCH_ALL_AA, EPOCH_ONLY_HLA_AA = int(EPOCH_ALL_AA), int(EPOCH_ONLY_HLA_AA)
        source = 'command line'

print('=' * 70)
print('TEST EVALUATION')
if EPOCH_MODE:
    print(f'models            : retrained, N_HLAS={N_HLAS} seed={MODEL_SEED}, '
          f'epochs all_aa={EPOCH_ALL_AA} only_HLA_aa={EPOCH_ONLY_HLA_AA} ({source})')
else:
    print(f'models            : {source}')
print(f'test HLAs         : {len(test_sqq)}  {test_sqq}')
print(f'output pickle     : {OUT_PKL}')
print('=' * 70, flush=True)


# =====================================================================================
#   4) THE NEW MODEL'S SCORES
# =====================================================================================
def gnn_result_dir(aa_set):
    """Where test_mhciifold_gnn.py wrote the parquets (mirrors its TRAIN_DIR)."""
    stem = os.path.join(
        OUTPUT_DIR,
        f'netmhciipan_train_included_only_peptide_conf_meansigmoid_new_order_new_set{N_HLAS}'
        f'_with_nethmciipan_weights_{aa_set}_and_yes_res_con_FCM_14_pepseq_hlas_fixed_seed{MODEL_SEED}',
    )
    d = f'{stem}/'
    if not os.path.isdir(d):
        print(f'[warn] no result directory for aa_set={aa_set}: {d}')
    return d


GNN_DIRS = {s: gnn_result_dir(s) for s in AA_SETS} if EPOCH_MODE else {}


def parquet_path(aa_set, hqq, epoch):
    """The test_mhciifold_gnn.py parquet holding this allele's scores."""
    if not EPOCH_MODE:
        return os.path.join(OUTPUT_DIR, f'{hqq}_epitope_test_released_{aa_set}.parquet')
    return os.path.join(GNN_DIRS[aa_set],
                        f'{hqq}_epitope_test_epoch{epoch}_regular22_baseline_{aa_set}.parquet')


def load_new_geomean(aa_set, hqq, epoch):
    """AA -> GeoMean Series from one test_mhciifold_gnn.py parquet, or None if missing."""
    path = parquet_path(aa_set, hqq, epoch)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)[['AA', 'GeoMean']].drop_duplicates()
    df['AA'] = df['AA'].astype(str).str.strip()
    n_conflict = df['AA'].duplicated().sum()
    if n_conflict:
        print(f'[warn] {os.path.basename(path)}: {n_conflict} AA rows with '
              f'differing GeoMean, keeping the first', flush=True)
        df = df.drop_duplicates(subset='AA', keep='first')
    return df.set_index('AA')['GeoMean']


MODELS = [
    ('min_value_combination',  'MHCIIFold-GNN+NetMHCIIPan'),
    ('min_value_GNN',          'MHCIIFold-GNN'),
    ('min_value_Original',     'Original'),
    ('Rank',                   ' NetMHCIIPan'),
    ('Geomean_FCM',            'FCM'),
    ('min_value_new',          NEW_LABEL),
]

custom_order = [' NetMHCIIPan', 'FCM', 'MHCIIFold-GNN', NEW_LABEL,
                'Original', 'MHCIIFold-GNN+NetMHCIIPan']


# =====================================================================================
#   5) Per-HLA evaluation
# =====================================================================================
ratios_per_hla = {}
missing = []

for HLA in test_sqq:
    hqq = HLA[5:] if HLA[:4] == 'DRB1' else HLA

    new_all  = load_new_geomean('all_aa',      hqq, EPOCH_ALL_AA)
    new_only = load_new_geomean('only_HLA_aa', hqq, EPOCH_ONLY_HLA_AA)
    if new_all is None or new_only is None:
        missing.append((HLA,
                        'all_aa' if new_all is None else '',
                        'only_HLA_aa' if new_only is None else ''))
        continue

    a = pd.read_csv(os.path.join(OUTPUT_DIR, f'{HLA}_test_final.csv'))

    a = a.copy()

    a['Peptide'] = a['Peptide'].astype(str).str.strip()

    a['GeoMean_letter_ranked'] = rank_normalise(a['GeoMean_letter'])
    a['GeoMean_conf_ranked']   = rank_normalise(a['GeoMean_conf'])
    a['Geomean_FCM_ranked']    = rank_normalise(a['Geomean_FCM'])

    a['min_value_Original']    = ((a['GeoMean_conf_ranked'] * a['GeoMean_letter_ranked'])) ** (1 / 2)
    a['min_value_GNN']         = (a['GeoMean_letter_ranked'] * a['GeoMean_conf_ranked']) ** (1 / 2)
    a['min_value_combination'] = ((a['GeoMean_conf_ranked'] * a['GeoMean_letter_ranked']) * a['Rank']) ** (1 / 3)

    # New model: geometric mean of the two rank-normalised scores.
    a['GeoMean_new_all_aa']      = a['Peptide'].map(new_all)
    a['GeoMean_new_only_HLA_aa'] = a['Peptide'].map(new_only)
    a['GeoMean_new_all_aa_ranked']      = rank_normalise(a['GeoMean_new_all_aa'])
    a['GeoMean_new_only_HLA_aa_ranked'] = rank_normalise(a['GeoMean_new_only_HLA_aa'])
    a['min_value_new'] = (a['GeoMean_new_all_aa_ranked'] *
                          a['GeoMean_new_only_HLA_aa_ranked']) ** (1 / 2)

    print(f"{HLA}: rows={len(a)}, positives {len(a[a['score'] == 1])}, "
          f"new score mapped for {a['min_value_new'].notna().sum()}/{len(a)}", flush=True)

    grouped = a.groupby('Original')
    ratios = []
    for score_col, df_label in MODELS:
        compute_ratios(grouped, score_col, df_label, HLA,
                       known_epitopes_of(HLA), ratios)

    if not ratios:
        print(f'[warn] {HLA}: no ratios computed, skipping HLA', flush=True)
        continue
    ratios_per_hla[HLA] = pd.DataFrame(ratios)

if missing:
    print(f'[warn] parquet missing, HLAs left out: {missing}', flush=True)
if not ratios_per_hla:
    raise RuntimeError('no test HLA could be evaluated')


# =====================================================================================
#   6) Summary and save
# =====================================================================================
combined_df = pd.concat(ratios_per_hla.values(), ignore_index=True)

# Original and the GNN+NetMHCIIpan combination are left out of the table,
# as in validation. Report 1 - ratio so it reads like an AUC (higher is better).
everything_df = combined_df[~combined_df['df'].isin(['Original', 'MHCIIFold-GNN+NetMHCIIPan'])].copy()
everything_df['Ratio'] = 1 - everything_df['Ratio']

summary_df = everything_df.groupby('df')['Ratio'].apply(mean_ci).reset_index()
pivot_df = summary_df.pivot(index='df', columns='level_1', values='Ratio').reset_index()
pivot_df['df'] = pd.Categorical(pivot_df['df'], categories=custom_order, ordered=True)
pivot_df = pivot_df.sort_values('df').reset_index(drop=True)
pivot_df = pivot_df[['df', 'mean', 'ci_lower', 'ci_upper']]
pivot_df.columns.name = None

payload = {
    'n_hlas':          N_HLAS,
    'seed':            MODEL_SEED,
    'models':          source,
    'epoch_pair':       (EPOCH_ALL_AA, EPOCH_ONLY_HLA_AA) if EPOCH_MODE else None,
    'hlas':            list(ratios_per_hla),
    'missing':         missing,
    'new_model_label': NEW_LABEL,
    'summary':         pivot_df,
    'everything':      everything_df.reset_index(drop=True),
    'all_ratios':      combined_df,
}
tmp = OUT_PKL + '.tmp'
with open(tmp, 'wb') as f:
    pickle.dump(payload, f)
os.replace(tmp, OUT_PKL)

print('=' * 70)
_models = (f'epochs all_aa={EPOCH_ALL_AA}, only_HLA_aa={EPOCH_ONLY_HLA_AA}'
           if EPOCH_MODE else 'released weights')
print(f'TEST RESULTS  ({_models}, {len(ratios_per_hla)} HLAs, '
      f'n={len(everything_df) // everything_df["df"].nunique()} epitopes)')
print(pivot_df.to_string(index=False))
print(f'pickle written to : {OUT_PKL}')
print('=' * 70, flush=True)
