"""
MHCIIFold-GNN -- TESTING / INFERENCE
====================================

Scores peptides with a trained model and writes one parquet per allele.

TWO WAYS TO SUPPLY THE MODEL
----------------------------
1. The released weights (the default). The published peptide-aware and
   peptide-agnostic models are in input_data/:

       all_aa       -> peptide_aware_model_train_weights.pth
       only_HLA_aa  -> peptide_agnostic_model_train_weights.pth

   These are the models the paper reports: the epoch for each was already
   chosen on the validation set, so no epoch sweep is needed or possible here.
   Output: <HLA>_epitope_test_released_<aa_set>.parquet in output_data/.

2. `--all-epochs`, after retraining with train_mhciifold_gnn.py: scores every
   10th-epoch checkpoint and writes one parquet per (allele, epoch), which is
   what select_epochs_from_validation.py consumes to choose the epoch pair.
   `--checkpoint <file>` scores one specific checkpoint instead.

The model classes (GNN / GCNGraphClassifier) are defined here identically to
the training definitions, so that both a state_dict and a whole pickled module
can be loaded.

CRITICAL: the test-time features must be assembled exactly as they were at
training time. The <aa_set> argument therefore has to match the checkpoint
being loaded. A width mismatch raises (check_widths /
check_trained_width guard this); an aa_set mismatch would NOT change any width
and would silently produce meaningless scores, which is why it is asserted
explicitly.

As in training, each peptide is cut into overlapping 9mer windows, every window
is a predicted structure / graph, and the per-window scores are pooled into one
peptide score. Remember the polarity: the model was trained against `1 - y`, so
the score is LOW for binders.

TWO INPUT MODES
---------------
Mode A (default) -- one allele from the validation or test set:

    python test_mhciifold_gnn.py <HLA> <aa_set> [batch_size] \
                                 [--checkpoint FILE | --all-epochs]
                                 [--n-hlas N] [--seed S]

    Peptides come from external_validation_and_test_set.csv for that allele;
    the 9mer graphs are read from full_test_and_validation_set.tar in
    input_data/ (or from an extracted .pkl there, if one exists). <HLA> may be
    given with or without the DRB1_ prefix.

Mode B -- a directory tree of individually built 9mer graphs:

    python test_mhciifold_gnn.py --rerun --aa-set <s> \
                                 [--root DIR] [--hla NAME] [--bs N] \
                                 [--epochs 70,300] [--one-batch]

    Walks every allele subdirectory of --root, where each 9mer has its own
    pickle named <9mer>_<Original>_... . The 9mers belonging to one Original
    peptide are grouped and scored together as that peptide's windows, exactly
    as in Mode A; only the peptide list comes from file names instead of a CSV.
    Window order follows position in the Original, not file name -- alphabetical
    order would scramble the windows. `--one-batch` is a diagnostic that pools
    every peptide into a single forward pass to confirm scores are independent
    of batching.

ARGUMENTS SELECTING THE CHECKPOINTS
-----------------------------------
    <aa_set>         'all_aa' or 'only_HLA_aa', must match the model
    --checkpoint F   score this checkpoint file instead of the released weights
    --all-epochs     score every 10th-epoch checkpoint of a retrained model
    --n-hlas N       retrained allele-panel size (--all-epochs only)
    --seed S         retrained replicate to load (--all-epochs only)

Batch size is an inference-only memory knob: peptides are scored independently,
so it never changes the numbers. run_test_with_backoff halves it on CUDA OOM
and keeps the smaller value for the remaining epochs.

NOTE ON PATHS: inputs are read from input_data/ and outputs written to
output_data/, both next to this script.
"""

import os

# Must be set before CUDA initialises. Reduces fragmentation across the many
# sequential checkpoint loads.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import pickle
import random
import sys
import tarfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats.mstats import gmean
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# Inputs are read from input_data/ and everything generated is written to
# output_data/, both next to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'input_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_data')

seed = 3
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# =====================================================================================
#                      CONFIGURATION  (mirrors train_mhciifold_gnn.py)
# =====================================================================================
# Node features: pLDDT (1) | molecule-type one-hot (3) | amino-acid one-hot (20)
# Edge features: contact probability | edge length
IN_CHANNELS = 24
EDGE_DIM = 2
GRAPH_TH = 5   # edge distance threshold the graphs were built with


# =====================================================================================
#                                  COMMAND LINE
# =====================================================================================
def _pop_flag_value(flag, default):
    """Remove `flag VALUE` from sys.argv and return VALUE (or `default`)."""
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        val = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2]
        return val
    return default


# Both flags are accepted in either mode and are removed from sys.argv before
# any positional parsing, so they never shift the other arguments.
#   --n-hlas : trained panel size (the new_order_new_set{N} directory).
#   --seed   : which trained replicate to load; NOT the inference seed above.
def _pop_flag(flag):
    """Remove a valueless `flag` from sys.argv; return whether it was there."""
    if flag in sys.argv:
        sys.argv.remove(flag)
        return True
    return False


# Which model to score with. By default the released weights shipped in
# input_data/; --all-epochs / --checkpoint are for a retrained model.
CHECKPOINT = _pop_flag_value('--checkpoint', None)
ALL_EPOCHS = _pop_flag('--all-epochs')

N_HLAS = int(_pop_flag_value('--n-hlas', 32))
MODEL_SEED = int(_pop_flag_value('--seed', 3))

RERUN_MODE = len(sys.argv) > 1 and sys.argv[1] == '--rerun'

if RERUN_MODE:
    AA_SET          = 'all_aa'
    RERUN_ROOT      = os.path.join(INPUT_DIR, 'alphafold3_rerun_9mers')
    RERUN_ONLY_HLA  = None
    RERUN_BS        = 64
    RERUN_EPOCHS    = [70, 300]
    # --one-batch: diagnostic. Pool every peptide from every HLA into a single
    # batch, score it in one forward pass, and write one parquet per epoch to the
    # root directory instead of one file per HLA subdirectory.
    RERUN_ONE_BATCH = False

    _args = sys.argv[2:]
    _i = 0
    while _i < len(_args):
        _a = _args[_i]
        if _a == '--one-batch':
            RERUN_ONE_BATCH = True
            _i += 1
            continue
        if _a == '--aa-set':
            AA_SET = _args[_i + 1]
        elif _a == '--root':
            RERUN_ROOT = _args[_i + 1]
        elif _a == '--hla':
            RERUN_ONLY_HLA = _args[_i + 1]
        elif _a == '--bs':
            RERUN_BS = int(_args[_i + 1])
        elif _a == '--epochs':
            RERUN_EPOCHS = [int(e) for e in _args[_i + 1].split(',') if e.strip()]
        else:
            raise ValueError(
                f"unknown argument {_a!r}. Usage: --rerun --aa-set <s> "
                f"[--root DIR] [--hla NAME] [--bs N] [--epochs 70,300] [--one-batch]")
        _i += 2
else:
    AA_SET = sys.argv[2] if len(sys.argv) > 2 else 'all_aa'

if AA_SET not in ('all_aa', 'only_HLA_aa'):
    raise ValueError(f"Unknown aa_set '{AA_SET}'. Choose 'all_aa' or 'only_HLA_aa'.")
ZERO_PEPTIDE_AA = (AA_SET == 'only_HLA_aa')

print('=' * 70)
print(f'aa set           : {AA_SET}  (zero peptide aa one-hot: {ZERO_PEPTIDE_AA})')
print(f'IN_CHANNELS      : {IN_CHANNELS}')
print(f'EDGE_DIM         : {EDGE_DIM}')
print(f'GRAPH_TH         : {GRAPH_TH}')
print(f'N_HLAS (panel)   : {N_HLAS}')
print('=' * 70)


# =====================================================================================
#                          MODEL (identical to training)
# =====================================================================================
otch = 40
otch2 = 60
otch3 = 80
head_num = 10


class GNN(torch.nn.Module):
    """GATv2 message passing over the residue graph -> one score per 9mer window.

    See train_mhciifold_gnn.py. Unused layers are constructed only so the class
    matches the pickled training checkpoints.
    """

    def __init__(self, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=otch, heads=head_num, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(in_channels=otch * head_num, out_channels=otch2, heads=head_num, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(in_channels=otch2 * head_num, out_channels=otch3, heads=head_num, edge_dim=edge_dim)

        self.dropout = torch.nn.Dropout(p=0.10)
        self.fc = torch.nn.Linear(40, 20)
        self.fc1 = torch.nn.Linear(otch3 * head_num, otch * head_num)
        self.fc2 = torch.nn.Linear(otch * head_num, 1)

        self.res_lin = torch.nn.Linear(in_channels, otch2 * head_num)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=30)
        self.res_lin_input = torch.nn.Linear(1, 20)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_feature = data.edge_feature.float()

        # Per-window pooling ids: unique, contiguous id per (peptide, window).
        # See the matching comment in train_mhciifold_gnn.py.
        k = data.windows.max().item()

        unique_ids_batch = data.batch * k + data.windows - 1
        unique_ids_batch, _ = unique_ids_batch.sort()
        diffs = unique_ids_batch[1:] - unique_ids_batch[:-1]
        mask = diffs > 1
        adjustments = torch.cumsum(mask * (diffs - 1), dim=0)
        adjusted_tensor = unique_ids_batch.clone()
        adjusted_tensor[1:] -= adjustments.to(unique_ids_batch.device)
        unique_ids_batch = adjusted_tensor

        x = self.conv1(x, edge_index, edge_feature)
        x = F.sigmoid(x)
        x = global_mean_pool(x, unique_ids_batch)

        x = self.fc2(x)
        x = x.squeeze(-1)
        x = F.sigmoid(x)
        return x


class GCNGraphClassifier(torch.nn.Module):
    """Combines the per-window GNN scores into one peptide-level score.

    See train_mhciifold_gnn.py.
    """

    def __init__(self, gnn):
        super().__init__()
        self.GNN = gnn
        self.fin_lin = torch.nn.Linear(2, 1)

    def forward(self, data):
        windows = data.windows
        batch = data.batch
        geo_means = data.geo_mean.float()

        # Number of windows of every peptide in the batch.
        unique_batches = torch.unique(batch, sorted=True)
        results = torch.empty(len(unique_batches), dtype=torch.int64, device=windows.device)
        for i, batch_id in enumerate(unique_batches):
            results[i] = torch.max(windows[batch == batch_id]).to(torch.int64)

        scalar_tensor = self.GNN(data)

        if torch.all(results == results[0]):
            scalar_tensor = scalar_tensor.view(data.y.shape[0], results[0])
            geo_mean = self.geometric_mean1(scalar_tensor)
        else:
            geo_mean = torch.zeros(data.y.shape[0], device=scalar_tensor.device)
            start = 0
            for m, count in enumerate(results):
                end = start + count
                geo_mean[m] = self.geometric_mean2(scalar_tensor[start:end])
                start = end

        concatenated = torch.cat((geo_mean.view(-1, 1), geo_means.view(-1, 1)), dim=1)
        concatenated = self.fin_lin(concatenated).squeeze(-1)
        return F.sigmoid(concatenated)

    def geometric_mean1(self, x):
        """(n_graphs, n_windows) -> (n_graphs,)"""
        return torch.exp(torch.log(x + 1e-10).mean(dim=1))

    def geometric_mean2(self, x):
        """(n_windows,) -> scalar"""
        return torch.exp(torch.log(x + 1e-10).mean())


# =====================================================================================
#                                 GRAPH ASSEMBLY
# =====================================================================================
def test(loader, test_df, peptide_names_t):
    """Score every graph in `loader` and return (preds, labels, merged frame).

    Predictions are aligned to `peptide_names_t` positionally -- the loader must
    be built with shuffle=False -- and merged onto the peptide metadata by 'AA'
    as the 'GeoMean' column. Low GeoMean = predicted binder. The y labels are a
    placeholder (build_graphs sets y=1 for every peptide); the real labels live
    in the validation CSV and are applied in select_epochs_from_validation.py.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            all_preds.append(out.cpu())
            all_labels.append(data.y.cpu())
            if torch.any(torch.isnan(out)):
                break

    # reshape(-1), not squeeze(): squeeze() would turn a single-peptide result
    # into a 0-d tensor.
    all_preds = torch.cat(all_preds).reshape(-1)
    all_labels = torch.cat(all_labels).reshape(-1)

    print(len(all_preds))
    print('Indices of NaN values:', torch.where(torch.isnan(all_preds))[0])
    print('Indices of NaN values:', torch.where(torch.isnan(all_labels))[0])

    fff = pd.DataFrame({'AA': peptide_names_t, 'GeoMean': all_preds})
    fff = pd.merge(fff, test_df, on='AA')
    print(fff.head())

    return all_preds, all_labels, fff


# The released graph pickles ship inside this archive; an extracted copy in
# input_data/ is used in preference to it when present.
GRAPH_TAR = os.path.join(INPUT_DIR, 'full_test_and_validation_set.tar')


def hla_spellings(hla):
    """The spellings this allele's graph pickle may be named with.

    DRB1 alleles appear both as the bare number ('0405') and with the prefix
    ('DRB1_0818'), so both are tried.
    """
    names = [hla]
    if hla.startswith('DRB1_'):
        names.append(hla[5:])
    elif len(hla) == 4 and hla.isdigit():
        names.append(f'DRB1_{hla}')
    return names


def load_graph_pickle(hla, kind):
    """The merged graph pickle of one allele, from input_data/ or the archive."""
    names = [f'alpha_9mers_{h}_{kind}_AA_level_GNN_{GRAPH_TH}_with_letters.pkl'
             for h in hla_spellings(hla)]

    for name in names:
        path = os.path.join(INPUT_DIR, name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

    if os.path.exists(GRAPH_TAR):
        with tarfile.open(GRAPH_TAR, 'r') as tar:
            members = {os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()}
            for name in names:
                if name in members:
                    print(f'reading {name} from {os.path.basename(GRAPH_TAR)}', flush=True)
                    with tar.extractfile(members[name]) as f:
                        return pickle.load(f)

    raise FileNotFoundError(
        f'no graph pickle for {hla!r}: looked for {names} in {INPUT_DIR} '
        f'and in {os.path.basename(GRAPH_TAR)}')


def windows_for(long_string, short_string_data):
    """The 9mer Data objects of one long peptide, in sliding-window order.

    Returns None if any window is missing, which is how both modes decide to
    drop a peptide rather than score it from a partial window set.
    """
    data_objects_list = []
    for i in range(len(long_string) - 9 + 1):
        short_string = long_string[i:i + 9]
        if short_string in short_string_data:
            data_objects_list.append(short_string_data[short_string])
        else:
            return None
    return data_objects_list


def build_graphs(new_dict):
    """Assemble one combined graph per long peptide from its 9mer windows.

    Feature construction mirrors graph_generator() in train_mhciifold_gnn.py.
    Peptides containing X or U are skipped.
    """
    graphs = []
    peptide_names = []

    for key, windows in new_dict.items():
        if 'X' in key or 'x' in key:
            print(key)
            continue

        if 'U' in key or 'u' in key:
            print(key)
            continue

        combined_x = []
        combined_edge_index = []
        combined_edge_features = []
        entity_id = []   # 1-based window id of every node
        offset = 0       # node offset of the current window within the combined graph

        means = []
        for entity_index, data_obj in enumerate(windows, start=1):
            data_obj = data_obj.clone()
            features = data_obj.x.clone()

            # pLDDT (column 3), inverted and scaled -> (N, 1)
            conf_feature = features[:, 3].float().unsqueeze(1)
            conf_feature = 1 - conf_feature / 100.0

            # Mean inverted pLDDT over the peptide (last 9 nodes).
            means.append(torch.mean(conf_feature[-9:, :]).item())

            # Molecule type (column 4): 0/1/2 -> one-hot (N, 3)
            molecule_type = features[:, 4].long()
            one_hot_encoded_mol = F.one_hot(molecule_type, num_classes=3).float()

            # Amino-acid type (column 5): 1..20 -> one-hot (N, 20)
            features[:, 5] = features[:, 5] - 1
            aa_type = features[:, 5].long()
            one_hot_encoded_aa_type = F.one_hot(aa_type, num_classes=20).float()

            # 'only_HLA_aa' hides the peptide's amino-acid identity, as in training.
            if ZERO_PEPTIDE_AA:
                one_hot_encoded_aa_type[-9:, :] = 0

            # Node features: pLDDT (1) | molecule type (3) | amino acid (20).
            combined_x.append(torch.cat([conf_feature, one_hot_encoded_mol, one_hot_encoded_aa_type], dim=1))

            entity_id.extend([entity_index] * data_obj.x.size(0))

            adjusted_edge_index = data_obj.edge_index + offset
            combined_edge_index.append(adjusted_edge_index)

            # Edge features: contact probability | edge length -> (E, 2).
            combined_edge_features.append(torch.stack([torch.as_tensor(data_obj.edge_features),
                                                       torch.as_tensor(data_obj.edge_lengths)], dim=1))

            offset += data_obj.x.size(0)

        graph_data = Data(x=torch.cat(combined_x, dim=0),
                          edge_index=torch.cat(combined_edge_index, dim=1),
                          edge_feature=torch.cat(combined_edge_features, dim=0),
                          geo_mean=torch.tensor([gmean(means)]),
                          y=torch.tensor([1]),
                          windows=torch.tensor(entity_id))
        graphs.append(graph_data)
        peptide_names.append(key)
    return graphs, peptide_names


if not RERUN_MODE:
    HLA = sys.argv[1]
    if len(HLA) == 3:
        HLA = '0' + HLA

    sss = load_graph_pickle(HLA, 'epitopes')

    # The evaluation scripts strip the DRB1_ prefix when they look for these
    # parquets, so the allele is written out the same way however it was given.
    HLA_KEY = HLA[5:] if HLA.startswith('DRB1_') else HLA

    print('HLA is: ', HLA)
    val_data = pd.read_csv(os.path.join(INPUT_DIR, 'external_validation_and_test_set.csv'), index_col=0)
    if len(HLA) == 4:
        test_df = val_data[val_data.HLA == f'DRB1_{HLA}']
    else:
        test_df = val_data[val_data.HLA == HLA]

    short_string_data = {key.upper(): value for key, value in sss.items()}

    new_dict = {}
    for long_string in test_df['AA']:
        data_objects_list = windows_for(long_string, short_string_data)
        if data_objects_list is not None:
            new_dict[long_string] = data_objects_list

    graphs_test, peptide_names_test = build_graphs(new_dict)
    print(f'Number of graphs created: {len(graphs_test)}')


# =====================================================================================
#                                     SCORING
# =====================================================================================
# Batch size is an inference-only memory knob; it never changes the scores.
# bs=1000 runs out of memory on a 40GB A100 (GATv2Conv materialises a dense
# [E, heads, out_channels] tensor), hence the default of 64.
if RERUN_MODE:
    bs = RERUN_BS
else:
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
MIN_BS = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.OutOfMemoryError only exists on newer torch; fall back to sniffing RuntimeError.
_oom_types = tuple(t for t in (getattr(torch, 'OutOfMemoryError', None),
                               getattr(getattr(torch, 'cuda', None), 'OutOfMemoryError', None))
                   if isinstance(t, type))


def _is_oom(exc):
    if _oom_types and isinstance(exc, _oom_types):
        return True
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


def free_gpu():
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def run_test_with_backoff(start_bs, graphs_list=None, meta_df=None, names=None):
    """Run test(), halving the batch size on CUDA OOM until it fits.

    Returns (preds, labels, df, batch_size_used). The three optional arguments
    default to Mode A's module-level values.
    """
    graphs_list = graphs_test        if graphs_list is None else graphs_list
    meta_df     = test_df            if meta_df     is None else meta_df
    names       = peptide_names_test if names       is None else names

    b = start_bs
    while True:
        loader = DataLoader(graphs_list, batch_size=b, shuffle=False)
        try:
            preds, labels, fff = test(loader, meta_df, names)
            return preds, labels, fff, b
        except Exception as exc:
            if not _is_oom(exc) or b <= MIN_BS:
                raise
            del loader
            free_gpu()
            b = max(MIN_BS, b // 2)
            print(f'CUDA OOM -- retrying this epoch with batch_size={b}', flush=True)


def check_widths(graphs_list, tag=''):
    """The test-set features must be exactly as wide as the model expects."""
    built_in = graphs_list[0].x.shape[1]
    built_ed = graphs_list[0].edge_feature.shape[1]
    print(f'{tag}built node feature width : {built_in} (expected {IN_CHANNELS})')
    print(f'{tag}built edge feature width : {built_ed} (expected {EDGE_DIM})')
    assert built_in == IN_CHANNELS, (
        f'test-set node features are {built_in} wide, expected {IN_CHANNELS}')
    assert built_ed == EDGE_DIM, (
        f'test-set edge features are {built_ed} wide, expected {EDGE_DIM}')
    return built_in


if not RERUN_MODE:
    built_in_channels = check_widths(graphs_test)


# Directory the matching training run wrote to (CHECKPOINT_DIR in
# train_mhciifold_gnn.py).
_TRAIN_STEM = os.path.join(
    OUTPUT_DIR,
    f'netmhciipan_train_included_only_peptide_conf_meansigmoid_new_order_new_set{N_HLAS}'
    f'_with_nethmciipan_weights_{AA_SET}_and_yes_res_con_FCM_14_pepseq_hlas_fixed_seed{MODEL_SEED}',
)
TRAIN_DIR = f'{_TRAIN_STEM}/'
if ALL_EPOCHS:
    if not os.path.isdir(TRAIN_DIR):
        print(f'[warn] checkpoint directory does not exist: {TRAIN_DIR}')
    print('loading checkpoints from:', TRAIN_DIR)


def ckpt_path(epoch):
    """Checkpoint file for this epoch (may not exist; callers check)."""
    return os.path.join(TRAIN_DIR, f'train_GNN_model_epoch{epoch}_regular22_baseline_14_pepseq_hlas_fixed.pth')


# The published models, one per aa_set, shipped in input_data/.
RELEASED_WEIGHTS = {
    'all_aa':      'peptide_aware_model_train_weights.pth',
    'only_HLA_aa': 'peptide_agnostic_model_train_weights.pth',
}


def load_full_model(path):
    """Load a model from `path`, whether it holds a state_dict or a module.

    train_mhciifold_gnn.py saves whole pickled modules; the released files hold
    state_dicts. torch >= 2.6 defaults to weights_only=True, which cannot
    unpickle a whole module, hence weights_only=False.
    """
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        return obj

    # A state_dict, possibly wrapped by the saving code.
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
        model = GCNGraphClassifier(GNN())
        model.load_state_dict(obj)
        return model

    raise TypeError(f'{path}: expected a state_dict or a torch module, got {type(obj).__name__}')


# (epoch or None, path) of every model to score with, in order.
if CHECKPOINT is not None:
    CHECKPOINTS = [(None, CHECKPOINT)]
elif ALL_EPOCHS:
    CHECKPOINTS = [(e, ckpt_path(e)) for e in range(10, 301, 10)]
else:
    CHECKPOINTS = [(None, os.path.join(INPUT_DIR, RELEASED_WEIGHTS[AA_SET]))]
    print(f'scoring with the released {AA_SET} model: {CHECKPOINTS[0][1]}')


def check_trained_width(model_obj, built_in, epoch):
    """The trained conv1 knows how wide its input must be; confirm we agree."""
    trained_in = getattr(model_obj.GNN.conv1, 'in_channels', None)
    if isinstance(trained_in, int) and trained_in != built_in:
        raise RuntimeError(
            f'epoch {epoch}: trained conv1 expects in_channels={trained_in} but the test '
            f'features are {built_in} wide -- aa_set mismatch '
            f'(AA_SET={AA_SET})')


if not RERUN_MODE:
    # ======================= Mode A: one HLA ==================================
    for i, ckpt in CHECKPOINTS:
        if not os.path.exists(ckpt):
            print(f'checkpoint missing, skipping: {ckpt}')
            continue

        model = load_full_model(ckpt)
        model = model.to(device)
        model.eval()

        check_trained_width(model, built_in_channels, i)

        test_preds, test_labels, fffq, used_bs = run_test_with_backoff(bs)
        if used_bs != bs:
            # Keep the smaller size for the remaining epochs.
            print(f'epoch {i}: batch size settled at {used_bs}', flush=True)
            bs = used_bs

        if i is None:
            out_parquet = os.path.join(OUTPUT_DIR, f'{HLA_KEY}_epitope_test_released_{AA_SET}.parquet')
        else:
            out_parquet = os.path.join(
                TRAIN_DIR, f'{HLA_KEY}_epitope_test_epoch{i}_regular22_baseline_{AA_SET}.parquet')
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        fffq.to_parquet(out_parquet)
        print(f'wrote {out_parquet}', flush=True)

        # Release this checkpoint before loading the next one.
        del model
        free_gpu()

else:
    # ======================= Mode B: the rerun 9mer tree ======================
    # Only pick up pickles built at GRAPH_TH.
    PKL_SUFFIX = f'_AA_level_GNN_{GRAPH_TH}_with_letters_max_value_no_voting.pkl'

    def parse_pkl_name(fname):
        """'<9mer>_<Original><PKL_SUFFIX>' -> (9mer, Original).

        The 9mer never contains '_', so splitting on the first underscore is
        unambiguous even when the Original does contain one.
        """
        stem = fname[:-len(PKL_SUFFIX)]
        nine, sep, original = stem.partition('_')
        if not sep or not nine or not original:
            return None
        return nine, original

    if not os.path.isdir(RERUN_ROOT):
        raise NotADirectoryError(f'rerun root does not exist: {RERUN_ROOT}')

    hla_dirs = sorted(d for d in os.listdir(RERUN_ROOT)
                      if os.path.isdir(os.path.join(RERUN_ROOT, d)))
    if RERUN_ONLY_HLA is not None:
        hla_dirs = [d for d in hla_dirs if d == RERUN_ONLY_HLA]
        if not hla_dirs:
            raise ValueError(f'--hla {RERUN_ONLY_HLA!r} is not a subdirectory of {RERUN_ROOT}')

    print(f'[rerun] root={RERUN_ROOT}')
    print(f'[rerun] {len(hla_dirs)} HLA directories, epochs={RERUN_EPOCHS}, '
          f'pickle suffix={PKL_SUFFIX}', flush=True)

    # Load each checkpoint once and reuse it for every HLA.
    models = {}
    for epoch in RERUN_EPOCHS:
        ckpt = ckpt_path(epoch)
        if not os.path.exists(ckpt):
            print(f'[rerun] checkpoint missing, skipping epoch {epoch}: {ckpt}', flush=True)
            continue
        m = load_full_model(ckpt).to(device)
        m.eval()
        models[epoch] = m
    if not models:
        raise FileNotFoundError(f'no checkpoints in {TRAIN_DIR}')

    totals = dict(hlas=0, originals=0, skipped=0, written=0)
    problems = []

    # --one-batch accumulates every HLA's graphs here instead of scoring per HLA.
    pooled_graphs = []
    pooled_names  = []
    pooled_hla    = []
    pooled_nwin   = []

    for hla_name in hla_dirs:
        hla_dir = os.path.join(RERUN_ROOT, hla_name)
        pkls = sorted(f for f in os.listdir(hla_dir) if f.endswith(PKL_SUFFIX))
        if not pkls:
            print(f'[rerun] {hla_name}: no {PKL_SUFFIX} files, skipping', flush=True)
            continue

        # ---- load every 9mer pickle in this HLA directory ----
        short_string_data = {}
        by_original = defaultdict(set)
        for fname in pkls:
            parsed = parse_pkl_name(fname)
            if parsed is None:
                problems.append((hla_name, fname, 'unparsable name'))
                continue
            nine, original = parsed
            by_original[original].add(nine)
            try:
                with open(os.path.join(hla_dir, fname), 'rb') as f:
                    one = pickle.load(f)
            except Exception as exc:
                problems.append((hla_name, fname, f'unreadable: {exc!r}'))
                continue
            # Each file is {9mer: Data} with a single entry.
            for k, v in one.items():
                short_string_data[str(k).upper()] = v

        # ---- group the 9mers of one Original and score them together ----
        new_dict_rerun = {}
        n_windows = {}
        for original in sorted(by_original):
            key = original.upper()
            if 'X' in key or 'U' in key:
                problems.append((hla_name, original, 'contains X/U'))
                totals['skipped'] += 1
                continue
            # Sliding-window order, not file order.
            objs = windows_for(key, short_string_data)
            if objs is None:
                have, need = len(by_original[original]), max(len(key) - 9 + 1, 0)
                problems.append((hla_name, original,
                                 f'incomplete windows: {have} pickles, needs {need}'))
                totals['skipped'] += 1
                continue
            new_dict_rerun[key] = objs
            n_windows[key] = len(objs)

        if not new_dict_rerun:
            print(f'[rerun] {hla_name}: no complete Originals, skipping', flush=True)
            continue

        graphs_rerun, names_rerun = build_graphs(new_dict_rerun)
        if not graphs_rerun:
            print(f'[rerun] {hla_name}: build_graphs produced nothing, skipping', flush=True)
            continue

        built_in_rerun = check_widths(graphs_rerun, tag=f'[{hla_name}] ')

        # test() merges its scores onto this frame by 'AA'.
        meta_rerun = pd.DataFrame({
            'AA': names_rerun,
            'hla': hla_name,
            'n_windows': [n_windows[n] for n in names_rerun],
        })

        totals['hlas'] += 1
        totals['originals'] += len(graphs_rerun)
        print(f'[rerun] {hla_name}: {len(pkls)} pickles -> {len(by_original)} Originals -> '
              f'{len(graphs_rerun)} scored', flush=True)

        if RERUN_ONE_BATCH:
            # Collect only; everything is scored together after the loop.
            pooled_graphs.extend(graphs_rerun)
            pooled_names.extend(names_rerun)
            pooled_hla.extend([hla_name] * len(names_rerun))
            pooled_nwin.extend(n_windows[n] for n in names_rerun)
            continue

        for epoch, m in models.items():
            check_trained_width(m, built_in_rerun, epoch)
            model = m   # test() reads the module-level name
            _, _, fffq, used_bs = run_test_with_backoff(bs, graphs_rerun, meta_rerun, names_rerun)
            if used_bs != bs:
                print(f'[rerun] {hla_name} epoch {epoch}: batch size settled at {used_bs}',
                      flush=True)
                bs = used_bs

            out_parquet = os.path.join(
                hla_dir,
                f'{hla_name}_rerun9mer_test_epoch{epoch}_regular22_baseline_{AA_SET}.parquet')
            fffq.to_parquet(out_parquet)
            totals['written'] += 1
            print(f'[rerun] wrote {out_parquet}', flush=True)

        del graphs_rerun
        free_gpu()

    # ---- --one-batch: score every HLA's peptides together, one batch, one file
    if RERUN_ONE_BATCH:
        if not pooled_graphs:
            print('[rerun] one-batch: nothing to score', flush=True)
        else:
            built_in_pooled = check_widths(pooled_graphs, tag='[one-batch] ')
            print(f'[rerun] one-batch: {len(pooled_graphs)} peptides from '
                  f'{len(set(pooled_hla))} HLAs in a single batch', flush=True)

            for epoch, m in models.items():
                check_trained_width(m, built_in_pooled, epoch)
                model = m   # test() reads the module-level name

                # Batch size = the whole pooled set, i.e. one forward pass.
                # test() needs a metadata frame to merge on; its merged output is
                # discarded below, so a frame of unique AAs is enough.
                _pool_meta = pd.DataFrame({'AA': list(dict.fromkeys(pooled_names))})
                _preds, _, _, used_bs = run_test_with_backoff(
                    len(pooled_graphs), pooled_graphs, _pool_meta, pooled_names)
                if used_bs != len(pooled_graphs):
                    print(f'[rerun] one-batch epoch {epoch}: WARNING did NOT fit in one '
                          f'batch, backed off to batch_size={used_bs}', flush=True)

                # Build the output from the raw predictions, aligned positionally
                # (loader is shuffle=False). A merge on AA would cross-join an
                # Original that appears under two HLAs.
                _p = np.asarray(_preds).reshape(-1)
                _n = min(len(_p), len(pooled_names))
                if _n != len(pooled_names):
                    print(f'[rerun] one-batch epoch {epoch}: WARNING got {len(_p)} preds for '
                          f'{len(pooled_names)} peptides (early NaN break?)', flush=True)
                out_df = pd.DataFrame({
                    'AA':        pooled_names[:_n],
                    'hla':       pooled_hla[:_n],
                    'n_windows': pooled_nwin[:_n],
                    'GeoMean':   _p[:_n],
                })

                out_parquet = os.path.join(
                    RERUN_ROOT,
                    f'ALL_HLAS_rerun9mer_onebatch_test_epoch{epoch}'
                    f'_regular22_baseline_{AA_SET}.parquet')
                out_df.to_parquet(out_parquet)
                totals['written'] += 1
                print(f'[rerun] wrote {out_parquet}', flush=True)

            free_gpu()

    print('=' * 70, flush=True)
    print(f'[rerun] {totals["hlas"]} HLAs scored, {totals["originals"]} Originals, '
          f'{totals["written"]} parquets written, {totals["skipped"]} Originals skipped',
          flush=True)
    for hla_name, item, why in problems[:40]:
        print(f'   {hla_name}  {item}  {why}', flush=True)
    if len(problems) > 40:
        print(f'   ... and {len(problems) - 40} more', flush=True)
