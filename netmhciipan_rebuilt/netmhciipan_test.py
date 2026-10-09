import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# --- PATHS ---------------------------------------------------------------
# Everything is resolved relative to THIS FILE, so the script runs from any
# working directory: input_data/ holds the data files it reads, output_data/
# receives the checkpoints and predictions it writes. Both sit next to this
# script in the repository.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'input_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_data')


full_data = pd.read_csv(
    os.path.join(INPUT_DIR, 'external_validation_and_test_set.csv'), index_col=0)


full_data['Peptide'] = full_data['AA']
full_data = full_data[['Peptide','HLA','Original','score']]


hlas = pd.read_csv(os.path.join(INPUT_DIR, 'hla_pseudosequences_2023.dat'),
                   sep=r"\s+", header=None)
hlas = hlas.rename(columns={0: "HLA", 1: "Pseudosequence"})



full_data = full_data.merge(hlas, on="HLA", how="left")
full_data['real_HLA'] = full_data['HLA']
full_data = full_data.drop(columns=["HLA"]).rename(columns={"Pseudosequence": "HLA"})



AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

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
HLA_SEQUENCE_LENGTH = 34


class PeptideWindowDataset(Dataset):
    def __init__(self, x_list, y):
        self.x_list = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
            for x in x_list
        ]
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.y[idx]


def peptide_collate_fn(batch):
    x_list, y_list = zip(*batch)

    flat_x = []
    segment_ids = []

    for peptide_idx_in_batch, x in enumerate(x_list):
        flat_x.append(x)
        segment_ids.append(torch.full((x.shape[0],), peptide_idx_in_batch, dtype=torch.long))

    return torch.cat(flat_x, dim=0), torch.cat(segment_ids, dim=0), torch.stack(y_list, dim=0)


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
        return torch.exp(sums / counts.clamp_min(1.0))

    def forward(self, flat_x, segment_ids, n_peptides=None):
        if n_peptides is None:
            n_peptides = int(segment_ids.max().item()) + 1
        raw_window_scores = self.row_net(flat_x)
        window_scores = self._activate_window_scores(raw_window_scores)
        peptide_scores = self._geometric_mean_by_segment(window_scores, segment_ids, n_peptides)
        return peptide_scores, window_scores


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
        raise ValueError(f"Peptide '{peptide}' is shorter than {PEPTIDE_CORE_LENGTH} amino acids")
    if len(hla_sequence) != HLA_SEQUENCE_LENGTH:
        raise ValueError(f"HLA sequence must be length {HLA_SEQUENCE_LENGTH}, got {len(hla_sequence)}")

    hla_encoded = encode_sequence_with_blosum(hla_sequence, expected_length=HLA_SEQUENCE_LENGTH)

    window_features = []
    peptide_length = float(len(peptide))
    for start in range(len(peptide) - PEPTIDE_CORE_LENGTH + 1):
        end = start + PEPTIDE_CORE_LENGTH
        peptide_encoded = encode_sequence_with_blosum(peptide[start:end], expected_length=PEPTIDE_CORE_LENGTH)
        flank_features = np.array([float(start), float(len(peptide) - end), peptide_length], dtype=np.float32)
        window_features.append(np.concatenate([peptide_encoded, flank_features, hla_encoded]).astype(np.float32))

    return np.stack(window_features, axis=0)


def filter_standard_peptides(df, peptide_col="Peptide", hla_col="HLA", score_col="score"):
    rows = []
    removed_count = 0
    removal_reasons = {
        "invalid_peptide_amino_acids": 0,
        "invalid_hla_amino_acids": 0,
        "peptide_too_short": 0,
        "hla_wrong_length": 0,
        "score_not_binary": 0,
    }

    for _, row in df.reset_index(drop=True).iterrows():
        peptide = str(row[peptide_col]).strip().upper()
        hla = str(row[hla_col]).strip().upper()
        score = float(row[score_col])

        keep = True

        if not is_standard_sequence(peptide):
            removal_reasons["invalid_peptide_amino_acids"] += 1
            keep = False
        if not is_standard_sequence(hla):
            removal_reasons["invalid_hla_amino_acids"] += 1
            keep = False
        if len(peptide) < PEPTIDE_CORE_LENGTH:
            removal_reasons["peptide_too_short"] += 1
            keep = False
        if len(hla) != HLA_SEQUENCE_LENGTH:
            removal_reasons["hla_wrong_length"] += 1
            keep = False
        if score not in (0.0, 1.0):
            removal_reasons["score_not_binary"] += 1
            keep = False

        if keep:
            cleaned_row = dict(row)
            cleaned_row[peptide_col] = peptide
            cleaned_row[hla_col] = hla
            cleaned_row[score_col] = score
            rows.append(cleaned_row)
        else:
            removed_count += 1

    print(f"Filtered out {removed_count} test rows")
    for reason, count in removal_reasons.items():
        print(f"  {reason}: {count}")
    return pd.DataFrame(rows, columns=df.columns)


def dataframe_to_window_dataset(df, peptide_col="Peptide", hla_col="HLA", score_col="score"):
    x_list = []
    scores = []

    for row_idx, row in df.reset_index(drop=True).iterrows():
        try:
            x_list.append(peptide_to_9mer_features(row[peptide_col], row[hla_col]))
        except ValueError as exc:
            raise ValueError(f"Row {row_idx}: {exc}") from exc
        scores.append(float(row[score_col]))

    return x_list, np.asarray(scores, dtype=np.float32)


@torch.no_grad()
def predict_dataset_single_model(model, x_list, batch_size=32, device="cpu"):
    dataset = PeptideWindowDataset(x_list, np.zeros(len(x_list), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=peptide_collate_fn)

    model.eval()
    all_peptide_scores = []
    for flat_x, segment_ids, y_batch in loader:
        flat_x = flat_x.to(device)
        segment_ids = segment_ids.to(device)
        peptide_scores, _ = model(flat_x=flat_x, segment_ids=segment_ids, n_peptides=y_batch.shape[0])
        all_peptide_scores.append(peptide_scores.cpu())

    return torch.cat(all_peptide_scores, dim=0).numpy()


# --- RUN LAYOUT ----------------------------------------------------------
# Mirrors netmhciipan_code.py so this script can find a training run's
# checkpoint directory from the SAME two arguments daily_use_auto.sh passes to
# the trainer: <model_index> <number_of_netmhciipan_hlas>. Keep these three
# things in sync with netmhciipan_code.py: FIXED_PEPSEQ_HLAS,
# get_model_config_from_index(), and build_run_name().
# The run directories netmhciipan_code.py wrote, under output_data/.
SAVE_DIR = OUTPUT_DIR
FIXED_PEPSEQ_HLAS = 14


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


FULL_DP_DQ_N = 999   # the "all peptides come from the big file" tag


def build_run_name(model_index, number_of_netmhciipan_hlas, model_config):
    """The training run's directory name -- no epoch in it.

    Must match netmhciipan_code.py: runs whose netmhciipan peptides come from
    netmhciipan_ba_peptides_full.csv carry a '999' tag. N=999 is the full DP/DQ
    panel and already IS the tag; every other N gets it appended after the count.
    """
    nmp_tag = (f"{number_of_netmhciipan_hlas}netmhciipan_hlas"
               if number_of_netmhciipan_hlas == FULL_DP_DQ_N
               else f"{number_of_netmhciipan_hlas}netmhciipan_hlas_{FULL_DP_DQ_N}")
    return (
        f"netmhciipan_model_{model_index:02d}"
        f"_{FIXED_PEPSEQ_HLAS}_pepseq_hlas_fixed"
        f"_{nmp_tag}"
        f"_hd{model_config['hidden_dim']}"
        f"_seed{model_config['seed']}_new_split"
    )


def epoch_of(checkpoint_path):
    """Sort key: the NNN in <run_name>_epoch<NNN>.pt (-1 if absent)."""
    m = re.search(r"_epoch(\d+)\.pt$", os.path.basename(checkpoint_path))
    return int(m.group(1)) if m else -1


def collect_checkpoints(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "*_epoch*.pt")), key=epoch_of)


def load_full_model(checkpoint_path, device):
    """Load a whole pickled model saved by netmhciipan_code.py's torch.save(model).

    torch >= 2.6 flipped torch.load's `weights_only` default to True, which
    refuses a full pickled module. Pass weights_only=False where the argument
    exists and fall back to the plain call on older torch.
    """
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def main():
    # Two call styles:
    #   python netmhciipan_test.py <model_index> <number_of_netmhciipan_hlas>
    #       -> tests EVERY epoch checkpoint of that training run. This is what
    #          daily_use_auto.sh drives, passing the same args as the trainer.
    #   python netmhciipan_test.py <saved_model.pt>
    #       -> the old single-checkpoint behaviour, still supported.
    if len(sys.argv) < 2:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <model_index> <number_of_netmhciipan_hlas>\n"
            f"   or: {sys.argv[0]} <saved_model.pt>"
        )

    single_model = sys.argv[1].endswith(".pt")

    if single_model:
        checkpoints = [sys.argv[1]]
        out_dir = SAVE_DIR
        run_dir = None
    else:
        if len(sys.argv) < 3:
            raise SystemExit(
                f"Usage: {sys.argv[0]} <model_index> <number_of_netmhciipan_hlas>"
            )
        model_index = int(sys.argv[1])
        number_of_netmhciipan_hlas = int(sys.argv[2])
        model_config = get_model_config_from_index(model_index)
        run_name = build_run_name(model_index, number_of_netmhciipan_hlas, model_config)
        run_dir = os.path.join(SAVE_DIR, run_name)

        print(
            f"model_index={model_index} -> hidden_dim={model_config['hidden_dim']}, "
            f"seed={model_config['seed']}"
        )
        print(f"pepseq HLAs (fixed): {FIXED_PEPSEQ_HLAS}")
        print(f"netmhciipan HLAs: {number_of_netmhciipan_hlas}")
        print(f"Run directory: {run_dir}")

        if not os.path.isdir(run_dir):
            raise SystemExit(
                f"Run directory does not exist: {run_dir}\n"
                "Has netmhciipan_code.py been run for this (model_index, N) pair?"
            )

        checkpoints = collect_checkpoints(run_dir)
        if not checkpoints:
            raise SystemExit(f"No *_epoch*.pt checkpoints found in {run_dir}")
        # Predictions land next to the checkpoints they came from.
        out_dir = run_dir
        print(f"Found {len(checkpoints)} checkpoints: "
              f"epoch{epoch_of(checkpoints[0]):03d} .. epoch{epoch_of(checkpoints[-1]):03d}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Featurise the validation set ONCE and reuse it for every checkpoint --
    # this is the expensive part and it does not depend on the model.
    test_df = filter_standard_peptides(full_data)
    if not len(test_df):
        raise SystemExit("No test rows survived filtering -- nothing to score.")
    x_test_list, _ = dataframe_to_window_dataset(test_df)
    print(f"Test rows: {len(test_df)}")

    for checkpoint_path in checkpoints:
        print(f"Loading model from: {checkpoint_path}", flush=True)
        model = load_full_model(checkpoint_path, device)
        model = model.to(device)

        predictions = predict_dataset_single_model(
            model, x_test_list, batch_size=32, device=device
        )

        result_df = test_df.copy()
        result_df["predicted_score"] = predictions

        output_path = os.path.join(
            out_dir,
            f"{os.path.splitext(os.path.basename(checkpoint_path))[0]}_test_predictions.csv",
        )
        result_df.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}", flush=True)

        del model

    print(f"Done. Tested {len(checkpoints)} checkpoint(s).")


if __name__ == "__main__":
    main()
