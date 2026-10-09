"""
MHCIIFold-GNN -- TRAINING
=========================

Stage 1 of 3 in the MHCIIFold-GNN pipeline:

    train_mhciifold_gnn.py          <- THIS FILE: fits the model, saves a
                                       checkpoint every 10 epochs
    test_mhciifold_gnn.py           <- scores held-out peptides with EVERY
                                       saved checkpoint
    select_epochs_from_validation.py<- reads those per-epoch scores and decides
                                       WHICH epochs to use

WHAT THE MODEL DOES
-------------------
Predicts whether a peptide binds a given MHC class II allele, using the 3D
structure of the peptide-MHC complex predicted by AlphaFold3 rather than the
sequence alone.

Each AlphaFold3 complex is turned into a graph (built upstream by
graph_creation_from_AF3_outputs.py, loaded here from .pkl):

  * nodes = residues of the HLA alpha chain, HLA beta chain and the peptide
  * node features  = [ pLDDT (1) | molecule-type one-hot (3) | amino-acid one-hot (20) ]
  * edges  = residue pairs within GRAPH_TH (5) Angstrom of each other
  * edge features  = [ AF3 contact probability | edge length ]

A peptide can be longer than 9 residues, so it is cut into overlapping 9mer
windows (step 1). Every window is a separate predicted structure / graph. The
windows of one peptide are concatenated into a single PyG Data object and
tagged with `windows`, so the network can pool per window and then combine the
per-window scores into one peptide-level score.

ARCHITECTURE (two nested modules)
---------------------------------
  GNN                 GATv2 attention convolution over the residue graph ->
                      sigmoid -> mean-pool within each 9mer window -> Linear ->
                      sigmoid = one score per window.
  GCNGraphClassifier  Combines the per-window scores into one peptide score
                      (geometric mean), concatenates the graph-level pLDDT
                      prior (`geo_mean`), and passes both through a final
                      Linear + sigmoid.

NOTE ON LABEL POLARITY: the loss is computed against `1 - y`, so the network's
output is LOW for binders and HIGH for non-binders. Downstream scripts rank
accordingly.

TRAINING DATA
-------------
Two sources, concatenated:
  * PepSeq in-house binding data (pepseq_drb1_binding_data.csv), DRB1 alleles -- always the
    first 14 alleles of `hla_list` (N_PEPSEQ_FIXED).
  * NetMHCIIpan 4.3 binding-affinity data (netmhciipan_ba_peptides_full.csv) for
    non-DRB1 alleles (DRB3/4/5, DP, DQ) -- how many of these are included is the
    sweep axis, chosen by the `hla_groups` index given on the command line.

Class imbalance is handled twice: non-binders are down-sampled per peptide
length to match the binder length distribution (in graph_generator), and the
BCE loss is weighted per sample by `pos_weight` (binder vs non-binder) times
`net_weight` (NetMHCIIpan-derived vs PepSeq-derived).

COMMAND LINE
------------
    python train_mhciifold_gnn.py <aa_set> <hla_group_idx> [seed]

    <aa_set>         'all_aa'      -> peptide residues keep their aa one-hot
                                      (peptide-aware model)
                     'only_HLA_aa' -> peptide aa one-hot zeroed
                                      (peptide-agnostic model)
                     The final score is the geometric mean of the two, so BOTH
                     are trained; see select_epochs_from_validation.py.
    <hla_group_idx>  0..15, index into `hla_groups`; picks how many
                     NetMHCIIpan alleles join the fixed 14 PepSeq ones.
    [seed]           random seed / replicate id, default 3. Goes into the
                     output directory name so replicates do not collide.

OUTPUT
------
One whole-model pickle (torch.save of the module, NOT a state_dict) per
checkpoint epoch, under CHECKPOINT_DIR. test_mhciifold_gnn.py reconstructs that
same path from the same (panel size, aa_set, seed) arguments.

NOTE ON PATHS: inputs are read from input_data/ and outputs written to
output_data/, both next to this script (see the README in each folder).
"""

import math
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats.mstats import gmean
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# Inputs are read from input_data/ and everything generated is written to
# output_data/, both next to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'input_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_data')

# argv: 1=AA_SET  2=NetMHCIIpan group index  3=seed (default 3)
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
print('seed:', seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# =====================================================================================
#                                  CONFIGURATION
# =====================================================================================
# Node feature layout coming out of the pkl (data_obj.x):
#     [ x, y, z, atom_confs(pLDDT), mol_type, AminoAcid_type ]
# where mol_type: 0 = HLA alpha chain, 1 = HLA beta chain, 2 = peptide.
# After slicing we build:  [ pLDDT (1) | mol_type one-hot (3) | aa one-hot (20) ] = 24
# Edge features:           [ contact_prob | edge_length ] = 2
# Graph-level extra:       geo_mean = gmean over windows of mean peptide pLDDT
# =====================================================================================
IN_CHANNELS = 24
EDGE_DIM = 2

# Edge distance threshold (Angstrom) the graphs were built with; part of the
# graph pickle file names.
GRAPH_TH = 5

# Peptide-aware / peptide-agnostic axis, mirrored in test_mhciifold_gnn.py.
#   'all_aa'      -> peptide residues keep their aa one-hot   (peptide aware)
#   'only_HLA_aa' -> peptide aa one-hot set to zero          (peptide agnostic)
AA_SET = sys.argv[1] if len(sys.argv) > 1 else 'only_HLA_aa'
if AA_SET not in ('all_aa', 'only_HLA_aa'):
    raise ValueError(f"Unknown aa_set '{AA_SET}'. Choose 'all_aa' or 'only_HLA_aa'.")
ZERO_PEPTIDE_AA = (AA_SET == 'only_HLA_aa')

# A checkpoint is written at every 10th epoch; training runs to max(SAVE_EPOCHS).
SAVE_EPOCHS = tuple(range(10, 501, 10))

print('=' * 70)
print(f'aa set           : {AA_SET}  (zero peptide aa one-hot: {ZERO_PEPTIDE_AA})')
print(f'IN_CHANNELS      : {IN_CHANNELS}')
print(f'EDGE_DIM         : {EDGE_DIM}')
print(f'GRAPH_TH         : {GRAPH_TH}')
print(f'SAVE_EPOCHS      : {SAVE_EPOCHS}')
print('=' * 70)


WINDOW_SIZE = 9


def graph_generator(sss, df, netmhciipan_or_not):
    """Build one PyG graph per peptide from its 9mer-window structures.

    Parameters
    ----------
    sss : dict
        {9mer sequence -> PyG Data} for one allele, unpickled from the
        AlphaFold3 graph file. One entry per predicted 9mer-HLA structure.
    df : DataFrame
        Binding data for that allele; needs 'peptide' and 'status'
        ('binder'/'nonbinder'), which becomes the 0/1 'score' label.
    netmhciipan_or_not : str
        'netmhciipan' if this allele's labels came from the NetMHCIIpan 4.3
        training data, anything else for in-house PepSeq data. Stored on every
        graph so the loss can weight the two sources differently.

    Returns
    -------
    (graphs, labels)
        graphs : list of Data, one per peptide. Each concatenates the peptide's
                 overlapping 9mer windows into a single graph and carries
                 `windows` (1-based window id per node) so the model can pool
                 per window, plus `geo_mean`, the graph-level pLDDT prior
                 (geometric mean over windows of the mean peptide pLDDT).
        labels : LongTensor of the 0/1 labels, same order.

    Notes
    -----
    1. CLASS / LENGTH BALANCING: non-binders are down-sampled so their peptide
       LENGTH distribution matches the binders', and over-long non-binders are
       trimmed to the target length from a randomly chosen end. Without this the
       model can separate the classes on length alone.
    2. ALL-OR-NOTHING WINDOWS: a peptide is kept only if EVERY one of its
       sliding 9mer windows has a predicted structure in `sss`. Scoring a
       peptide from a partial window set would not be comparable to a full one.
    3. pLDDT is inverted and rescaled (1 - pLDDT/100) so that, like the model
       output, LOW means confident/binder.
    """
    source_flag = torch.tensor([1]) if netmhciipan_or_not == 'netmhciipan' else torch.tensor([0])

    df['score'] = df['status'].map({'binder': 1, 'nonbinder': 0})
    df['length'] = df['peptide'].str.len()

    # ---- length-matched down-sampling of the non-binders --------------------
    df_1s = df[df['score'] == 1]
    df_0s = df[df['score'] == 0]

    length_counts_1s = df_1s['length'].value_counts().sort_index()
    neg_to_pos_ratio = len(df[df.score == 0]) / length_counts_1s.values.sum()
    length_allocation_0s = (length_counts_1s * neg_to_pos_ratio).astype(int)

    if len(df_1s) != 0:
        trimmed_0s = []

        for length, count in length_allocation_0s.items():
            # Non-binders at least this long, trimmed down to `length` if longer.
            available_0s = df_0s[df_0s['length'] >= length]
            sample_size = min(count, len(available_0s))
            selected_0s = available_0s.sample(sample_size, replace=False)

            for _, row in selected_0s.iterrows():
                peptide = row['peptide']
                if len(peptide) > length:
                    trim_start = random.choice([True, False])
                    peptide = peptide[-length:] if trim_start else peptide[:length]
                trimmed_0s.append({'peptide': peptide, 'score': 0, 'length': length})

            # Each non-binder is used at most once.
            df_0s = df_0s.drop(selected_0s.index)

        df = pd.concat([df_1s, pd.DataFrame(trimmed_0s)]).reset_index(drop=True)
        df = df[df.length >= WINDOW_SIZE]

    # ---- collect the 9mer windows of every peptide --------------------------
    short_string_data = {key.upper(): value for key, value in sss.items()}

    new_dict = {}
    for long_string in df['peptide']:
        if len(long_string) < WINDOW_SIZE:
            continue

        data_objects_list = []
        for i in range(len(long_string) - WINDOW_SIZE + 1):
            short_string = long_string[i:i + WINDOW_SIZE]
            if short_string in short_string_data:
                data_objects_list.append(short_string_data[short_string])

        # Only keep this peptide if every expected 9mer is available.
        if len(data_objects_list) == len(long_string) - WINDOW_SIZE + 1:
            new_dict[long_string] = data_objects_list

    # ---- one combined graph per peptide -------------------------------------
    graphs = []
    labels = []

    for key, windows in new_dict.items():
        label = torch.tensor([df[df.peptide == key]['score'].iloc[0]])
        labels.append(df[df.peptide == key]['score'].iloc[0])

        combined_x = []
        combined_edge_index = []
        combined_edge_features = []
        entity_id = []   # 1-based window id of every node
        offset = 0       # node offset of the current window within the combined graph

        means = []
        for entity_index, data_obj in enumerate(windows, start=1):
            data_obj = data_obj.clone()
            features = data_obj.x.clone()
            if features.shape[1] == 7:
                features = features[:, :-1]
            features[:, 3] = 1 - features[:, 3] / 100

            # Mean inverted pLDDT over the peptide (last 9 nodes).
            means.append(torch.mean(features[-9:, 3]).item())

            features = features[:, 3:]

            main_features = features[:, :-2]
            molecule_type = features[:, -2].long()
            aa_type = (features[:, -1] - 1).long()

            one_hot_encoded_mol = F.one_hot(molecule_type, num_classes=3).float()[:, :3]
            one_hot_encoded_aa = F.one_hot(aa_type, num_classes=20).float()

            # 'only_HLA_aa' hides the peptide's amino-acid identity (the last 9
            # rows are the peptide residues); test_mhciifold_gnn.py does the same.
            if ZERO_PEPTIDE_AA:
                one_hot_encoded_aa[-9:, :] = 0

            # Node features: pLDDT (1) | molecule type (3) | amino acid (20).
            combined_x.append(torch.cat([main_features, one_hot_encoded_mol, one_hot_encoded_aa], dim=1))

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
                          y=label,
                          netmhciipan_or_not=source_flag,
                          geo_mean=torch.tensor([gmean(means)]),
                          windows=torch.tensor(entity_id))
        graphs.append(graph_data)

    print(f'Number of graphs created: {len(graphs)}')
    return graphs, torch.tensor(labels)


# =====================================================================================
#                                    ALLELE PANEL
# =====================================================================================
# hla_list: the in-house PepSeq alleles (bare DRB1 numbers). The first
# N_PEPSEQ_FIXED are always used.
hla_list = ['0405', '1201', '0301', '0901', '0701', '1001', '1301', '1101',
            '0406', '1503', '1602', '0804', '1401', '0102', '1102', '1302']
N_PEPSEQ_FIXED = 14

# Non-DRB1 alleles (DRB3/4/5, DP, DQ) whose labels come from the NetMHCIIpan 4.3
# training data, in the order they are added to the panel. hla_groups[i] is the
# cumulative panel for group i: groups 0..11 add one allele each and groups
# 12..15 add two, so group 15 holds all 20. This is the allele-count sweep axis.
NETMHCIIPAN_ALLELE_ORDER = [
    'DRB5_0101', 'DRB3_0101', 'DRB4_0101', 'HLA-DQA10501-DQB10301',
    'DRB3_0202', 'HLA-DQA10301-DQB10302', 'HLA-DQA10101-DQB10501',
    'HLA-DQA10501-DQB10201', 'HLA-DPA10103-DPB10401', 'HLA-DQA10102-DQB10602',
    'HLA-DPA10301-DPB10402', 'HLA-DPA10201-DPB10501',
    'DRB3_0301', 'HLA-DQA10301-DQB10201',
    'DRB4_0103', 'HLA-DQA10301-DQB10301',
    'HLA-DQA10501-DQB10302', 'HLA-DQA10103-DQB10603',
    'HLA-DPA10103-DPB10201', 'HLA-DQA10401-DQB10402',
]
_GROUP_SIZES = list(range(1, 13)) + [14, 16, 18, 20]
hla_groups = {i: NETMHCIIPAN_ALLELE_ORDER[:n] for i, n in enumerate(_GROUP_SIZES)}

netmhciipan_group_idx = int(sys.argv[2])
if netmhciipan_group_idx not in hla_groups:
    raise ValueError(f'netmhciipan group index {netmhciipan_group_idx} not in '
                     f'hla_groups; valid: {min(hla_groups)}..{max(hla_groups)}')

hla_list = hla_list[:N_PEPSEQ_FIXED]
netmhciipan_hla_list = hla_groups[netmhciipan_group_idx]
hla_list.extend(netmhciipan_hla_list)
print(f'netmhciipan group index: {netmhciipan_group_idx} '
      f'(groups 0..{netmhciipan_group_idx} = {len(netmhciipan_hla_list)} HLAs)')
print('I am using these HLAs: ', hla_list)
print('Number of HLAs: ', len(hla_list))


# =====================================================================================
#                                     LOAD DATA
# =====================================================================================
pepseq_data = pd.read_csv(os.path.join(INPUT_DIR, 'pepseq_drb1_binding_data.csv'))
netmhciipan_data = pd.read_csv(os.path.join(INPUT_DIR, 'netmhciipan_ba_peptides_full.csv'))

all_graphs = []
all_labels = []
total_netmhciipan_graphs = 0

for hla in hla_list:
    graph_stem = os.path.join(INPUT_DIR, f'alpha_9mers_{hla}')
    try:
        with open(f'{graph_stem}_new_data_AA_level_GNN_{GRAPH_TH}_with_letters.pkl', 'rb') as f:
            sss = pickle.load(f)
        netmhciipan_or_not = 'NO'
    except Exception:
        # NetMHCIIpan alleles: graphs may be split over two pickles (the _2 split
        # only exists for the TH=5 set).
        netmhciipan_or_not = 'netmhciipan'
        sss = {}
        for part in ('', '_2'):
            path = f'{graph_stem}_NetMHCIIPan_AA_level_GNN_{GRAPH_TH}_with_letters{part}.pkl'
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    sss.update(pickle.load(f))

    if len(hla) == 4:
        df = pepseq_data[pepseq_data.HLA == f'DRB1_{hla}']
    else:
        df = netmhciipan_data[netmhciipan_data.HLA == hla]

    print('length of the df: ', len(df))

    graphs_i, labels_i = graph_generator(sss, df, netmhciipan_or_not)

    print('length of labels', len(labels_i))

    if netmhciipan_or_not == 'netmhciipan':
        total_netmhciipan_graphs += len(graphs_i)

    all_graphs.extend(graphs_i)
    all_labels.append(labels_i)

graphs = all_graphs
labels = torch.cat(all_labels)


# =====================================================================================
#                                       MODEL
# =====================================================================================
otch = 40
otch2 = 60
otch3 = 80
head_num = 10


class GNN(torch.nn.Module):
    """GATv2 message passing over the residue graph -> one score per 9mer window.

    Only conv1 and fc2 are used in forward(). The other
    layers (conv2, conv3, dropout, fc, fc1, res_lin, layer_norm, res_lin_input)
    are unused; they are still constructed so that parameter initialisation
    under a given seed, and the saved checkpoints, stay identical.
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

        # ---- Per-window pooling ids ----------------------------------------
        # A peptide's overlapping 9mer windows are concatenated into ONE graph,
        # with data.windows holding a 1-based window id per node. data.batch
        # only says which PEPTIDE a node belongs to, so (batch * k + windows - 1)
        # gives a unique id per (peptide, window). Peptides differ in length, so
        # those ids are sparse; the sort + cumsum renumbers them to be
        # contiguous, as global_mean_pool requires.
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

    Per-window scores are ensembled by geometric mean, concatenated with
    `geo_mean`, the graph-level pLDDT prior, and passed through a final
    Linear + sigmoid.
    The two code paths in forward() are the same computation: the fast one
    reshapes when every peptide in the batch has the same number of windows,
    the slow one loops when peptide lengths differ.
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
#                                     TRAINING
# =====================================================================================
# Two independent imbalances, multiplied per sample in train():
#   pos_weight : binders are the minority class after length-matched
#                down-sampling, so each binder is up-weighted.
#   net_weight : the NetMHCIIpan-derived alleles contribute a different number
#                of graphs than the in-house PepSeq ones, so each NetMHCIIpan
#                graph is re-weighted to keep one source from dominating.
pos_weight = len(labels) / sum(labels)
neg_weight = 1

net_weight = len(labels) / total_netmhciipan_graphs
net_no_weight = 1

print('pos_weight', pos_weight)
print('net_weight', net_weight)


def train(loader):
    """One epoch. Returns (mean loss, train AUC, predictions, labels)."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.float()
        weights = (torch.where(y == 1, pos_weight, torch.tensor(neg_weight)).to(device)
                   * torch.where(data.netmhciipan_or_not == 1, net_weight,
                                 torch.tensor(net_no_weight)).to(device))

        criterion = torch.nn.BCELoss(weight=weights)
        # Target is 1 - y, NOT y: the model is trained to output LOW for binders.
        loss = criterion(out, 1 - y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_preds.append(out.detach())
        all_labels.append(data.y)

    all_preds = torch.cat(all_preds).cpu().squeeze()
    all_labels = torch.cat(all_labels).cpu().squeeze()
    auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())

    return total_loss / len(loader), auc, all_preds, all_labels


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 128
train_loader = DataLoader(graphs, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

gnn_model = GNN().to(device)
model = GCNGraphClassifier(gnn_model).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'Total model parameters')

# ---- Optimiser + LR schedule ----------------------------------------------------
#  AdamW (decoupled weight decay) with a per-epoch schedule:
#    epochs 0 .. WARMUP_EPOCHS-1 : constant PEAK_LR
#    epochs WARMUP_EPOCHS .. END : cosine decay from PEAK_LR down to MIN_LR
#  scheduler.step() is called once per epoch, after train(), so epoch e trains
#  with lr_schedule(e).
PEAK_LR       = 0.003
MIN_LR        = PEAK_LR * 0.01         # floor reached at the last epoch
WARMUP_EPOCHS = 70
TOTAL_EPOCHS  = max(SAVE_EPOCHS) + 1   # epochs 0 .. max(SAVE_EPOCHS)
WEIGHT_DECAY  = 1e-3


def lr_schedule(epoch):
    """Multiplier on PEAK_LR for a given epoch (used by LambdaLR)."""
    if epoch < WARMUP_EPOCHS:
        return 1.0
    progress = (epoch - WARMUP_EPOCHS) / max(1, TOTAL_EPOCHS - 1 - WARMUP_EPOCHS)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (MIN_LR / PEAK_LR) + (1.0 - MIN_LR / PEAK_LR) * cosine


optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
print(f'optimizer: AdamW  peak_lr={PEAK_LR}  min_lr={MIN_LR}  constant for {WARMUP_EPOCHS} epochs, then '
      f'cosine decay to epoch {TOTAL_EPOCHS - 1}  weight_decay={WEIGHT_DECAY}')

# AA_SET is part of the path so the peptide-aware and peptide-agnostic runs never
# overwrite each other; test_mhciifold_gnn.py builds the same path.
CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    f'netmhciipan_train_included_only_peptide_conf_meansigmoid_new_order_new_set{len(hla_list)}'
    f'_with_nethmciipan_weights_{AA_SET}_and_yes_res_con_FCM_14_pepseq_hlas_fixed_seed{seed}/',
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

for epoch in range(max(SAVE_EPOCHS) + 1):
    learning_rate = optimizer.param_groups[0]['lr']   # lr used for THIS epoch
    loss, train_auc, train_preds, train_labels = train(train_loader)
    scheduler.step()                                  # advance to next epoch's lr
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, '
          f'learning_rate: {learning_rate:.2e}, batch_size: {bs}, otch: {otch}, number of heads: {head_num}')

    if epoch in SAVE_EPOCHS:
        torch.save(model, os.path.join(
            CHECKPOINT_DIR,
            f'train_GNN_model_epoch{epoch}_regular22_baseline_14_pepseq_hlas_fixed.pth'))
