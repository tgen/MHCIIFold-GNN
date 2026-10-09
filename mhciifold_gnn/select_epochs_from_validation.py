"""
MHCIIFold-GNN -- VALIDATION / EPOCH SELECTION
=============================================

Stage 3 of 3 in the MHCIIFold-GNN pipeline:

    train_mhciifold_gnn.py          <- fits the model, one checkpoint every
                                       10 epochs
    test_mhciifold_gnn.py           <- scores the validation peptides with
                                       EVERY saved checkpoint
    select_epochs_from_validation.py<- THIS FILE

ONLY NEEDED AFTER RETRAINING
----------------------------
This step chose the epochs of the published models, and its result is already
baked into the released weights in input_data/. Reproducing the sweep needs the
per-epoch checkpoints, which are not part of the release; this script is here
for the record and for anyone who retrains with train_mhciifold_gnn.py and then
runs test_mhciifold_gnn.py --all-epochs.

WHAT THIS FILE IS FOR
---------------------
This is NOT simply "run the model on the validation set" -- that already
happened in test_mhciifold_gnn.py, which wrote one parquet per (allele, epoch).
This script is the MODEL-SELECTION step: it decides WHICH TRAINING EPOCHS to
use, and it is the only place the validation labels influence a choice.

The final MHCIIFold-GNN score is the geometric mean of two separately trained
models -- the peptide-aware one ('all_aa') and the peptide-agnostic one
('only_HLA_aa'). Those two do not necessarily peak at the same epoch, so the
epoch is chosen as a PAIR. This script sweeps the full 30 x 30 grid
(10, 20, ..., 300 epochs for each of the two models = 900 combinations),
evaluates the combined score for every pair on the validation alleles, and
stores the result per pair. The pair with the best validation performance is
the one carried forward; the selected epochs are then applied unchanged to the
held-out test set elsewhere.

HOW A PAIR IS SCORED (the "frank" ratio)
----------------------------------------
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
  * if a group contains more than one positive, sequence_reconstructer() stitches
    the overlapping kmers back into contiguous sequences and each reconstructed
    stretch is ranked separately.

The same ratio is computed for the comparison models, so every epoch pair is
reported alongside the baselines it is being judged against: NetMHCIIpan,
FCM, the original MHCIIFold-GNN score, and the
MHCIIFold-GNN + NetMHCIIpan combination (`MODELS` / `custom_order`).

COMMAND LINE
------------
    python select_epochs_from_validation.py [--n-hlas N] [--seed S]

    --n-hlas N   trained allele-panel size -- picks which pair of
                 test_mhciifold_gnn.py output directories (all_aa /
                 only_HLA_aa) is read. Default 32.
    --seed S     which training replicate to validate. Default 3.

OUTPUT
------
Two pickles in OUT_DIR, keyed by (epoch_all_aa, epoch_only_HLA_aa):

    validation_results_nhlas<N>_seed<S>.pkl
        per pair: the summary table (mean and 95% CI per model)
    validation_everything_df_nhlas<N>_seed<S>.pkl
        per pair: the underlying per-epitope rows

Both are rewritten atomically every CHECKPOINT_EVERY iterations, so a
wall-clock kill mid-sweep does not lose the completed pairs. The best pair is
printed at the end.

STRUCTURE OF THE CODE BELOW
---------------------------
  1. The validation allele panel.
  2. compute_ratios() / sequence_reconstructer() -- the frank-ratio machinery.
  3. Per-allele preparation done ONCE: build frame `a`, rank-normalise every
     score, and compute the comparison models' ratios (they do not depend on
     the epoch pair).
  4. The 30 x 30 sweep: only the two new GeoMean columns and the new model's
     ratios are recomputed inside the loop.
  5. Save.

NOTE ON PATHS: known_epitopes_per_allele.csv is read from input_data/; the
per-allele <HLA>_validation_final.csv files, the test_mhciifold_gnn.py parquets
and all outputs live in output_data/. Both folders sit next to this script
(see the README in each folder).
"""

import os
import pickle
import sys
import time

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


# --n-hlas: which trained panel's test_mhciifold_gnn.py parquets to validate.
# --seed:   which trained replicate to validate.
N_HLAS = int(_pop_flag_value('--n-hlas', 32))
MODEL_SEED = int(_pop_flag_value('--seed', 3))


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
#   VALIDATION SWEEP over the test_mhciifold_gnn.py (Mode A) parquets
# =====================================================================================
# For every (epoch_all_aa, epoch_only_HLA_aa) pair -- 30 x 30 = 900 -- the two
# parquets of every sqq HLA are mapped onto data frame `a` by peptide (map, not
# merge, so no row fan-out), rank-normalised like the other scores, and
# combined into 'min_value_new' (geometric mean of the two ranks). That score
# is evaluated alongside the existing MODELS and the resulting pivot_df is
# stored under key (epoch_all_aa, epoch_only_HLA_aa).
#
# Everything that does not depend on the epoch pair is computed once per HLA
# before the sweep.
# =====================================================================================
EPOCHS    = list(range(10, 301, 10))         # 10, 20, ..., 300  -> 30 per aa_set
AA_SETS   = ('all_aa', 'only_HLA_aa')
NEW_LABEL = 'MHCIIFold-GNN-new'              # df label of the new score in MODELS

OUT_DIR = os.path.join(OUTPUT_DIR, 'validation_14_pepseq_hlas_fixed')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PKL = os.path.join(OUT_DIR, f'validation_results_nhlas{N_HLAS}_seed{MODEL_SEED}.pkl')
OUT_EVERYTHING_PKL = os.path.join(OUT_DIR, f'validation_everything_df_nhlas{N_HLAS}_seed{MODEL_SEED}.pkl')
CHECKPOINT_EVERY = 30

print('=' * 70)
print(f'VALIDATION SWEEP   N_HLAS={N_HLAS}  seed={MODEL_SEED}')
print(f'epochs             : {EPOCHS[0]}..{EPOCHS[-1]} step 10 '
      f'({len(EPOCHS)} x {len(EPOCHS)} = {len(EPOCHS) ** 2} iterations)')
print(f'validation HLAs    : {len(sqq)}  {sqq}')
print(f'output pickle      : {OUT_PKL}')
print(f'everything_df pkl  : {OUT_EVERYTHING_PKL}')
print('=' * 70, flush=True)


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


GNN_DIRS = {s: gnn_result_dir(s) for s in AA_SETS}
for s, d in GNN_DIRS.items():
    print(f'parquet dir {s:12s}: {d}')


def parquet_path(aa_set, hqq, epoch):
    return os.path.join(
        GNN_DIRS[aa_set],
        f'{hqq}_epitope_test_epoch{epoch}_regular22_baseline_{aa_set}.parquet')


def load_new_geomean(aa_set, hqq, epoch):
    """AA -> GeoMean Series from one test_mhciifold_gnn.py parquet, or None if missing.

    The parquet has one row per (AA, Original), so the same AA repeats with the
    same GeoMean; it is reduced to a unique AA index for use with .map().
    """
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


# Epoch-independent comparison models: (score column, label).
MODELS = [
    ('min_value_combination',  'MHCIIFold-GNN+NetMHCIIPan'),
    ('min_value_GNN',          'MHCIIFold-GNN'),
    ('min_value_Original',     'Original'),
    ('Rank',                   ' NetMHCIIPan'),
    ('Geomean_FCM',            'FCM'),
]
NEW_MODEL = ('min_value_new', NEW_LABEL)

custom_order = [' NetMHCIIPan', 'FCM', 'MHCIIFold-GNN', NEW_LABEL,
                'Original', 'MHCIIFold-GNN+NetMHCIIPan']


# =====================================================================================
#   3) Per-HLA preparation, done ONCE: build `a`, all old scores, old ratios
# =====================================================================================
base_a       = {}   # HLA -> prepared data frame `a` (before the new columns)
base_ratios  = {}   # HLA -> list of ratio dicts for the old MODELS
hqq_of       = {}   # HLA -> name used in the parquet file names

for HLA in sqq:
    a = pd.read_csv(os.path.join(OUTPUT_DIR, f'{HLA}_validation_final.csv'))

    hqq = HLA[5:] if HLA[:4] == 'DRB1' else HLA
    hqq_of[HLA] = hqq

    a = a.copy()

    a['Peptide'] = a['Peptide'].astype(str).str.strip()

    a['GeoMean_letter_ranked'] = rank_normalise(a['GeoMean_letter'])
    a['GeoMean_conf_ranked']   = rank_normalise(a['GeoMean_conf'])
    a['Geomean_FCM_ranked']    = rank_normalise(a['Geomean_FCM'])

    a['min_value_Original']    = ((a['GeoMean_conf_ranked'] * a['GeoMean_letter_ranked'])) ** (1 / 2)
    # MHCIIFold-GNN score: geometric mean of the letter and conf ranks.
    a['min_value_GNN']         = (a['GeoMean_letter_ranked'] * a['GeoMean_conf_ranked']) ** (1 / 2)
    a['min_value_combination'] = ((a['GeoMean_conf_ranked'] * a['GeoMean_letter_ranked']) * a['Rank']) ** (1 / 3)

    if a.isna().any().any():
        print(a[a.isna().any(axis=1)])

    print(f"{HLA}: rows={len(a)}, positives {len(a[a['score'] == 1])}", flush=True)

    # The old MODELS' frank ratios never change across the epoch sweep.
    grouped = a.groupby('Original')
    ratios = []
    for score_col, df_label in MODELS:
        compute_ratios(grouped, score_col, df_label, HLA,
                       known_epitopes_of(HLA), ratios)

    base_a[HLA]      = a
    base_ratios[HLA] = ratios

# Fail early if no parquets exist at all (e.g. a wrong N_HLAS).
_n_found = sum(os.path.exists(parquet_path(s, hqq_of[h], e))
               for s in AA_SETS for h in sqq for e in EPOCHS)
_n_total = len(AA_SETS) * len(sqq) * len(EPOCHS)
print(f'parquets available : {_n_found}/{_n_total}', flush=True)
if _n_found == 0:
    raise FileNotFoundError(
        f'no test_mhciifold_gnn.py parquets found for N_HLAS={N_HLAS} under {list(GNN_DIRS.values())}')


# =====================================================================================
#   4) The sweep: 30 x 30 epoch pairs
# =====================================================================================
validation_results = {}      # (epoch_all_aa, epoch_only_HLA_aa) -> pivot_df
everything_results = {}      # (epoch_all_aa, epoch_only_HLA_aa) -> everything_df
_coverage_reported = set()   # HLAs whose AA-mapping coverage has been printed
skipped_pairs      = {}      # (epoch_all_aa, epoch_only_HLA_aa) -> reason
t0 = time.time()
n_iter = 0


def save_results(final=False):
    """Write both result pickles atomically (temp file + rename)."""
    payload = {
        'n_hlas':          N_HLAS,
        'seed':            MODEL_SEED,
        'epochs':          EPOCHS,
        'hlas':            list(sqq),
        'new_model_label': NEW_LABEL,
        'complete':        final,
        'results':         validation_results,
        'skipped':         skipped_pairs,
    }
    payload_everything = dict(payload, results=everything_results)
    for path, obj in ((OUT_PKL, payload), (OUT_EVERYTHING_PKL, payload_everything)):
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)


for ep_all in EPOCHS:           # epoch of the all_aa parquet
    for ep_only in EPOCHS:      # epoch of the only_HLA_aa parquet
        n_iter += 1
        ratios_per_hla = {}
        missing = []

        for HLA in sqq:
            hqq = hqq_of[HLA]

            new_all  = load_new_geomean('all_aa',      hqq, ep_all)
            new_only = load_new_geomean('only_HLA_aa', hqq, ep_only)
            if new_all is None or new_only is None:
                missing.append((HLA,
                                'all_aa' if new_all is None else '',
                                'only_HLA_aa' if new_only is None else ''))
                continue

            a = base_a[HLA].copy()

            a['GeoMean_new_all_aa']      = a['Peptide'].map(new_all)
            a['GeoMean_new_only_HLA_aa'] = a['Peptide'].map(new_only)

            # Partial overlap is fine: rows of `a` with no parquet value stay NaN
            # and are skipped by rank(). Reported once per HLA, for information.
            if HLA not in _coverage_reported:
                _coverage_reported.add(HLA)
                n_un_all  = a['GeoMean_new_all_aa'].isna().sum()
                n_un_only = a['GeoMean_new_only_HLA_aa'].isna().sum()
                print(f'[info] {HLA}: new GeoMean mapped for '
                      f'all_aa {len(a) - n_un_all}/{len(a)}, '
                      f'only_HLA_aa {len(a) - n_un_only}/{len(a)} rows of a '
                      f'(parquet AAs: {len(new_all)} / {len(new_only)})', flush=True)

            a['GeoMean_new_all_aa_ranked']      = rank_normalise(a['GeoMean_new_all_aa'])
            a['GeoMean_new_only_HLA_aa_ranked'] = rank_normalise(a['GeoMean_new_only_HLA_aa'])

            # Geometric mean of the two new ranks.
            a['min_value_new'] = (a['GeoMean_new_all_aa_ranked'] *
                                  a['GeoMean_new_only_HLA_aa_ranked']) ** (1 / 2)

            grouped = a.groupby('Original')
            ratios = list(base_ratios[HLA])
            compute_ratios(grouped, NEW_MODEL[0], NEW_MODEL[1], HLA,
                           known_epitopes_of(HLA), ratios)

            if not ratios:
                print(f'[warn] {HLA} ep({ep_all},{ep_only}): no ratios computed, skipping HLA', flush=True)
                continue
            ratios_per_hla[HLA] = pd.DataFrame(ratios)

        if missing:
            print(f'[skip] ep({ep_all},{ep_only}): parquet missing for {missing}', flush=True)
            skipped_pairs[(ep_all, ep_only)] = missing
            continue

        combined_df = pd.concat(ratios_per_hla.values(), ignore_index=True)

        # Original and the GNN+NetMHCIIpan combination are left out of
        # this table. Report 1 - ratio so it reads like an AUC (higher is better).
        everything_df = combined_df[~combined_df['df'].isin(['Original', 'MHCIIFold-GNN+NetMHCIIPan'])].copy()
        everything_df['Ratio'] = 1 - everything_df['Ratio']

        summary_df = everything_df.groupby('df')['Ratio'].apply(mean_ci).reset_index()
        pivot_df = summary_df.pivot(index='df', columns='level_1', values='Ratio').reset_index()
        pivot_df['df'] = pd.Categorical(pivot_df['df'], categories=custom_order, ordered=True)
        pivot_df = pivot_df.sort_values('df').reset_index(drop=True)
        pivot_df = pivot_df[['df', 'mean', 'ci_lower', 'ci_upper']]
        pivot_df.columns.name = None

        validation_results[(ep_all, ep_only)] = pivot_df
        everything_results[(ep_all, ep_only)] = everything_df.reset_index(drop=True)

        _new = pivot_df.loc[pivot_df['df'] == NEW_LABEL, 'mean']
        _new = float(_new.iloc[0]) if len(_new) else float('nan')
        print(f'[{n_iter:4d}/{len(EPOCHS) ** 2}] ep_all_aa={ep_all:3d} ep_only_HLA_aa={ep_only:3d}  '
              f'{NEW_LABEL} mean AUC={_new:.4f}  '
              f'(n={len(everything_df) // everything_df["df"].nunique()} epitopes, '
              f'{(time.time() - t0) / 60:.1f} min)', flush=True)
        print(pivot_df.to_string(index=False), flush=True)

        if n_iter % CHECKPOINT_EVERY == 0:
            save_results(final=False)


# =====================================================================================
#   5) Save
# =====================================================================================
save_results(final=True)

print('=' * 70)
print(f'DONE  N_HLAS={N_HLAS}: {len(validation_results)} epoch pairs stored, '
      f'{len(skipped_pairs)} skipped, {(time.time() - t0) / 60:.1f} min')
if validation_results:
    best_key = max(validation_results,
                   key=lambda k: float(validation_results[k].loc[
                       validation_results[k]['df'] == NEW_LABEL, 'mean'].iloc[0]))
    print(f'best {NEW_LABEL} pair (epoch_all_aa, epoch_only_HLA_aa) = {best_key}')
    print(validation_results[best_key].to_string(index=False))
print(f'pickle written to  : {OUT_PKL}')
print(f'everything_df to   : {OUT_EVERYTHING_PKL}')
print('=' * 70, flush=True)
