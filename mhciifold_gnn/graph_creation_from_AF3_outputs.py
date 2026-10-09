"""
MHCIIFold-GNN -- GRAPH CONSTRUCTION FROM ALPHAFOLD3 OUTPUTS
===========================================================

Stage 0 of the MHCIIFold-GNN pipeline -- the step that produces the .pkl graph
files every other script loads:

    graph_creation_from_AF3_outputs.py  <- THIS FILE: AlphaFold3 output -> graph
    train_mhciifold_gnn.py              <- fits the model on those graphs
    test_mhciifold_gnn.py               <- scores peptides with each checkpoint
    select_epochs_from_validation.py    <- picks which epochs to use

WHAT IT DOES
------------
Takes one AlphaFold3 prediction of a 9mer-peptide / MHC class II complex and
converts it into a PyTorch Geometric graph. Inputs are the two files AF3 writes
per prediction:

    model.cif          the predicted structure
    confidences.json   per-atom pLDDT (`atom_plddts`) and the pairwise
                       contact-probability matrix (`contact_probs`)

The CIF is expected to have three chains, named HLA (alpha chain), HLB (beta
chain) and PEPTIDE.

HOW THE GRAPH IS BUILT
----------------------
1. TRIM TO THE BINDING GROOVE. Only HLA/HLB residues with at least one atom
   within 5 A of any peptide atom are kept. The peptide is kept in full. This
   is what keeps the graphs small enough to train on.

2. NODES ARE RESIDUES, NOT ATOMS. One node per kept residue, positioned at its
   alpha carbon (`Atom_types == 'CA'`). Node feature row:

       [ x, y, z, AA_conf, mol_type, AminoAcid_type, AA_indices_edited ]

   AA_conf          mean pLDDT over all atoms of that residue
   mol_type         0 = HLA alpha, 1 = HLA beta, 2 = peptide
   AminoAcid_type   1..22 via letter_to_number_amino_acid
   AA_indices_edited  residue number within its own chain

   The training and test scripts slice this row: they take AA_conf, one-hot the
   mol_type into 3 columns and the AminoAcid_type into 20, and discard x/y/z
   (the geometry enters only through which edges exist and how long they are).
   They also invert pLDDT as 1 - pLDDT/100.

3. EDGES ARE RESIDUE-TO-RESIDUE, THRESHOLDED ON ATOM-TO-ATOM DISTANCE. Two
   residues are connected if ANY pair of their atoms is within TH Angstrom
   (TH comes from graph_config.json, default 5). Edges are stored in both
   directions, so the graph is undirected. Because the test is atom-level while
   the stored length is between alpha carbons, some edge lengths exceed TH --
   that is expected and harmless, since the length is itself a feature.

   Each edge carries:
       edge_features   the AF3 contact probability for that residue pair
       edge_lengths    the CA-CA distance

   TH is baked into the graphs at this point and is part of the output file
   name.

4. OUTPUT. One pickle per prediction, containing {peptide_key: Data}. The
   downstream scripts merge these per allele, so a peptide longer than 9
   residues is reassembled from the pickles of its overlapping 9mer windows.

TWO INPUT MODES
---------------
Mode A -- one peptide subdirectory (what the Nextflow pipeline drives):

    python graph_creation_from_AF3_outputs.py <HLA> <sub_dir> <base_directory> <origin>

    Reads  <base_directory>/<HLA>/confidences/<sub_dir>/
    Writes <out_base>/<out_root>/<HLA>/<sub_dir><filename_suffix>
    <origin> is 'epitope' | 'pepseq' | 'netmhciipan' and selects the output
    tree, so training graphs and validation/test graphs stay separate.

Mode B -- a CSV listing prediction directories (run by hand):

    python graph_creation_from_AF3_outputs.py --csv <file.csv>
           [--path-col Path] [--peptide-col 9mer] [--hla-col hla]
           [--name-col Original] [--out-base DIR] [--row N] [--force]

    One output per row, written to
    <csv_out_base>/<hla>/<9mer>_<Original>_AA_level_GNN_<TH>_with_letters.pkl.
    Rows resolving to the same output file are deduped up front; a row that
    fails is reported and the run continues, exiting non-zero at the end.

CONFIGURATION AND RESUME
------------------------
Output locations, the distance threshold and the file-name templates all live in
`graph_config.json` beside this script (included in this folder). They are NOT
hardcoded here because the Nextflow pipeline has to derive the same names to
work out what is already finished.

Both modes skip a pickle that already exists and is non-empty -- in Mode A
before importing torch or Biopython at all. Pickles are written to a temp file
and renamed into place, so a file on disk is always complete and a job killed
mid-write can never leave a truncated result that the skip logic would mistake
for finished work. `--force` (or FORCE_REBUILD=1) rebuilds regardless.

NOTE ON PATHS: the output directories in graph_config.json are placeholders;
set them to real directories before running this script.
The AlphaFold3 predictions themselves, and the Nextflow pipeline that submits
these jobs, are not part of this folder.
"""

import json
import os
import sys

# ============================================================================
#  Configuration, argument parsing and the resume guard.
#
#  This block deliberately runs before the heavy imports below, so that in
#  Mode A a job whose output already exists exits without loading torch or
#  Biopython.
# ============================================================================

# Output location and file naming live in graph_config.json because the
# Nextflow pipeline has to derive the exact same names to count what is
# already done. If the two ever disagree, finished work stops being recognised.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph_config.json')
with open(CONFIG_PATH) as _f:
    CFG = json.load(_f)

TH = CFG['th']

# Rebuild every output, ignoring the resume guard. Set either with
# FORCE_REBUILD=1 in the environment or with --force on the command line
# (the latter survives containers that wipe the environment).
FORCE_REBUILD = os.environ.get('FORCE_REBUILD', '') not in ('', '0')


def output_is_done(path):
    """True when `path` is a finished pickle that should not be rebuilt."""
    return (not FORCE_REBUILD) and os.path.exists(path) and os.path.getsize(path) > 0


CSV_MODE = len(sys.argv) > 1 and sys.argv[1] == '--csv'

if CSV_MODE:
    # ---- Mode B: a CSV of prediction directories ----------------------------
    csv_path     = None
    path_col     = 'Path'       # directory holding model.cif / confidences.json
    peptide_col  = '9mer'       # key inside the pickle, and first half of the file name
    hla_col      = 'hla'        # output subdirectory
    name_col     = 'Original'   # second half of the file name
    only_row     = None
    csv_out_base = CFG['csv_out_base']

    _args = sys.argv[1:]
    _i = 0
    while _i < len(_args):
        _a = _args[_i]
        if _a == '--force':
            FORCE_REBUILD = True
            _i += 1
            continue
        if _a == '--csv':
            csv_path = _args[_i + 1]
        elif _a == '--path-col':
            path_col = _args[_i + 1]
        elif _a == '--peptide-col':
            peptide_col = _args[_i + 1]
        elif _a == '--hla-col':
            hla_col = _args[_i + 1]
        elif _a == '--name-col':
            name_col = _args[_i + 1]
        elif _a == '--out-base':
            csv_out_base = _args[_i + 1]
        elif _a == '--row':
            only_row = int(_args[_i + 1])
        else:
            raise ValueError(
                f"unknown argument {_a!r}. Usage: --csv <file.csv> "
                f"[--path-col Path] [--peptide-col 9mer] [--hla-col hla] "
                f"[--name-col Original] [--out-base DIR] [--row N] [--force]")
        _i += 2

    if csv_path is None:
        raise ValueError('--csv given without a file path')

    if FORCE_REBUILD:
        print('[force] rebuilding all rows, ignoring existing pickles', flush=True)

    # TH is part of the suffix so outputs built at different thresholds can
    # coexist, and "<9mer>_<Original>" stays the prefix so it can be recovered.
    CSV_OUT_SUFFIX = CFG['csv_filename_suffix'].format(th=TH)

    print(f'[mode] csv  file={csv_path}  path_col={path_col!r}  '
          f'peptide_col={peptide_col!r}  hla_col={hla_col!r}  '
          f'name_col={name_col!r}  row={only_row}  TH={TH}', flush=True)
    print(f'[mode] output per row -> {csv_out_base}/<{hla_col}>/'
          f'<{peptide_col}>_<{name_col}>{CSV_OUT_SUFFIX}', flush=True)

else:
    # ---- Mode A: one peptide subdirectory -----------------------------------
    HLA            = sys.argv[1]
    sub_dir        = sys.argv[2]   # peptide subdirectory under <base>/<HLA>/confidences/
    base_directory = sys.argv[3]   # analysis root holding the AF3 predictions
    origin         = sys.argv[4]   # 'epitope' | 'pepseq' | 'netmhciipan'

    if origin not in CFG['origins']:
        raise ValueError(f"unknown origin {origin!r}; expected one of {sorted(CFG['origins'])}")

    out_root   = CFG['origins'][origin]['out_root']
    origin_tag = CFG['origins'][origin]['tag']

    out_dir  = os.path.join(CFG['out_base'], out_root, HLA)
    out_name = sub_dir + CFG['filename_suffix'].format(hla=HLA, tag=origin_tag, th=TH)
    out_path = os.path.join(out_dir, out_name)

    if FORCE_REBUILD:
        print(f'[force] FORCE_REBUILD set, rebuilding: {out_path}', flush=True)
    elif output_is_done(out_path):
        print(f'[skip] output already exists, nothing to do: {out_path}', flush=True)
        sys.exit(0)

    print(f'[run] {origin} {HLA} {sub_dir} -> {out_path}', flush=True)


# ============================================================================
#  Heavy imports (only reached when there is work to do)
# ============================================================================
import pickle

import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data


# Amino-acid letter -> integer code (1..22). The 20 standard residues are 1..20.
letter_to_number_amino_acid = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWYXB', start=1)}

# Chain id in the CIF -> molecule type stored as a node feature.
CHAIN_IDS = ('HLA', 'HLB', 'PEPTIDE')
MOL_TYPE = {'HLA': 0, 'HLB': 1, 'PEPTIDE': 2}

# HLA residues are kept only if one of their atoms is this close to the peptide.
GROOVE_CUTOFF = 5


def compute_residue_edges(df, threshold):
    """Residue-level edges from atom-level distances.

    `df` is the per-atom table: columns 0-2 are x/y/z and column 3 is the
    (global) residue index. Two residues are connected when any pair of their
    atoms is within `threshold`. Every edge is stored in both directions, and
    residue indices are remapped to 0..n_residues-1 in sorted order.

    Returns (edges, edge_distances).
    """
    coordinates = df.iloc[:, :3].values
    amino_acid_indices = df.iloc[:, 3].values

    pairwise_distances = squareform(pdist(coordinates))

    # Upper triangle only: the distance matrix is symmetric.
    atom_i, atom_j = np.nonzero(np.triu(pairwise_distances <= threshold, k=1))
    aa_i = amino_acid_indices[atom_i]
    aa_j = amino_acid_indices[atom_j]
    different_residue = aa_i != aa_j   # no self-loops
    aa_i, aa_j = aa_i[different_residue], aa_j[different_residue]

    amino_acid_edges = set(zip(aa_i, aa_j)) | set(zip(aa_j, aa_i))

    sorted_edges = np.array(sorted(amino_acid_edges))
    unique_values = np.unique(sorted_edges)
    mapping_dict = {orig: new for new, orig in enumerate(unique_values)}
    remapped_edges = np.vectorize(mapping_dict.get)(sorted_edges)

    # Edge length = distance between the two residues' alpha carbons.
    ca_coords = df[df['Atom_types'] == 'CA'].set_index('AA_indices')[['x', 'y', 'z']]
    ca_coords = ca_coords.loc[unique_values].values   # row r = CA of remapped residue r
    edge_dist_vals = np.linalg.norm(ca_coords[remapped_edges[:, 0]] - ca_coords[remapped_edges[:, 1]], axis=1)

    return remapped_edges, edge_dist_vals


def process_cif_file(cif_file):
    """Parse one AF3 structure into the per-atom and per-residue tables.

    Returns
    -------
    edges : (E, 2) array of residue-level edges (both directions)
    edges_distance_values : (E,) edge lengths
    residue_df : one row per kept residue (its CA atom) -- the node table
    atom_df : one row per atom of every kept residue -- used for the
              per-residue mean pLDDT and the contact-probability lookup
    """
    structure = MMCIFParser().get_structure('structure_id', cif_file)

    # Per chain: residue sequence, and per atom its coordinates, atom name,
    # position in the CIF (= index into confidences.json) and residue number.
    sequence    = {c: [] for c in CHAIN_IDS}
    coords      = {c: [] for c in CHAIN_IDS}
    atom_types  = {c: [] for c in CHAIN_IDS}
    atom_ids    = {c: [] for c in CHAIN_IDS}
    residue_ids = {c: [] for c in CHAIN_IDS}

    atom_index = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if chain.id in sequence:
                    sequence[chain.id].append(seq1(residue.get_resname()))
                else:
                    print('PROBLEM')

                for atom in residue:
                    if chain.id in coords:
                        coords[chain.id].append(atom.get_coord())
                        atom_types[chain.id].append(atom.get_id())
                        atom_ids[chain.id].append(atom_index)
                        residue_ids[chain.id].append(residue.id[1])
                    atom_index += 1

    coords = {c: np.array(v) for c, v in coords.items()}
    pep_coords = coords['PEPTIDE']

    # HLA residues with at least one atom within GROOVE_CUTOFF of the peptide.
    groove_residues = {}
    for c in ('HLA', 'HLB'):
        groove_residues[c] = []
        if len(pep_coords) > 0 and len(coords[c]) > 0:
            near = (distance_matrix(coords[c], pep_coords) < GROOVE_CUTOFF).any(axis=1)
            groove_residues[c] = np.array(residue_ids[c])[near].tolist()

    def chain_table(c):
        seq = sequence[c]
        df = pd.DataFrame({'x': coords[c][:, 0], 'y': coords[c][:, 1], 'z': coords[c][:, 2],
                           'AA_indices': residue_ids[c],
                           'atom_index': atom_ids[c],
                           'Atom_types': atom_types[c]})
        df['mol_type'] = [MOL_TYPE[c]] * len(df)
        df['AminoAcid_type'] = df['AA_indices'].apply(lambda i: seq[i - 1] if i <= len(seq) else None)
        return df

    alpha_df   = chain_table('HLA')
    beta_df    = chain_table('HLB')
    peptide_df = chain_table('PEPTIDE')

    alpha_df = alpha_df[alpha_df['AA_indices'].isin(groove_residues['HLA'])].copy()
    beta_df  = beta_df[beta_df['AA_indices'].isin(groove_residues['HLB'])].copy()

    # Keep the within-chain residue number, then offset the chains so that
    # AA_indices is one global, 0-based residue index across HLA, HLB, PEPTIDE.
    for df in (alpha_df, beta_df, peptide_df):
        df['AA_indices_edited'] = df['AA_indices']

    max_alpha = max(residue_ids['HLA'])
    max_beta  = max(residue_ids['HLB'])
    beta_df['AA_indices']    += max_alpha
    peptide_df['AA_indices'] += max_beta + max_alpha

    atom_df = pd.concat([alpha_df, beta_df, peptide_df])
    atom_df['AA_indices'] -= 1

    # Nodes are residues: keep one row per residue, its alpha carbon.
    residue_df = atom_df[atom_df['Atom_types'] == 'CA'].copy()
    residue_df['AminoAcid_type'] = residue_df['AminoAcid_type'].map(letter_to_number_amino_acid)

    # The distance test is atom-to-atom but the edge is drawn residue-to-residue,
    # so some edge lengths can exceed the threshold; they are still stored.
    edges, edges_distance_values = compute_residue_edges(atom_df, threshold=TH)

    return edges, edges_distance_values, residue_df, atom_df


def build_and_save(cif_path, json_path, key, out_path):
    """Build one residue-level graph and write it as {key: Data} to out_path."""
    edges, edges_distance_values, residue_df, atom_df = process_cif_file(cif_path)
    edges = torch.tensor(edges).t()

    with open(json_path, 'r') as json_file:
        confidences = json.load(json_file)

    atom_df['atom_confs'] = [confidences['atom_plddts'][i] for i in atom_df['atom_index'].values]

    # Per-residue mean pLDDT, attached to the node table by residue index.
    aa_mean_conf = atom_df.groupby('AA_indices')['atom_confs'].mean()
    residue_df['AA_conf'] = residue_df['AA_indices'].map(aa_mean_conf)

    # Map the 0..n-1 edge node ids back to global residue indices to look up
    # the residue-level contact probabilities.
    value_map = dict(zip(np.array(edges.unique()), atom_df['AA_indices'].unique()))
    residue_edges = np.vectorize(value_map.get)(np.array(edges))

    cont_matrix = np.array(confidences['contact_probs'])
    cont_matrix_vals_for_edges = cont_matrix[residue_edges[0, :], residue_edges[1, :]]

    features = torch.tensor(residue_df[['x', 'y', 'z', 'AA_conf', 'mol_type',
                                        'AminoAcid_type', 'AA_indices_edited']].values)

    graph_data = Data(x=features, edge_index=edges,
                      edge_features=cont_matrix_vals_for_edges,
                      edge_lengths=edges_distance_values)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write to a per-job temp file, then rename into place. os.replace is atomic,
    # so a job killed mid-write never leaves a truncated .pkl behind.
    tmp_path = f'{out_path}.tmp.{os.getpid()}'
    with open(tmp_path, 'wb') as file:
        pickle.dump({key: graph_data}, file)
    os.replace(tmp_path, out_path)

    print(f'[done] wrote {out_path}', flush=True)


def find_inputs_generic(row_dir):
    """Locate the cif + confidences json in a Mode B directory.

    Matches exact basenames on purpose: 'summary_confidences.json' also ends
    with 'confidences.json' but carries neither atom_plddts nor contact_probs.
    """
    if not os.path.isdir(row_dir):
        raise NotADirectoryError(f'not a directory: {row_dir}')

    json_path = os.path.join(row_dir, 'confidences.json')
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f'no confidences.json in {row_dir}')

    cif_path = os.path.join(row_dir, 'model.cif')
    if not os.path.isfile(cif_path):
        # Fall back to a single unambiguous .cif if it is named something else.
        cifs = sorted(f for f in os.listdir(row_dir) if f.endswith('.cif'))
        if len(cifs) == 1:
            cif_path = os.path.join(row_dir, cifs[0])
        elif not cifs:
            raise FileNotFoundError(f'no .cif in {row_dir}')
        else:
            raise FileNotFoundError(
                f'no model.cif in {row_dir} and {len(cifs)} other .cif files to choose from: {cifs}')

    return cif_path, json_path


def resolve_column(df, wanted):
    """Exact match, then case-insensitive; otherwise fail with the real column names."""
    if wanted in df.columns:
        return wanted
    lowered = {str(c).lower(): c for c in df.columns}
    if wanted.lower() in lowered:
        found = lowered[wanted.lower()]
        print(f'[warn] column {wanted!r} not found, using {found!r}', flush=True)
        return found
    raise KeyError(f'column {wanted!r} is not in {list(df.columns)}')


def clean_component(value, what, idx):
    """A value used as a directory or file name: non-empty, no path separator."""
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        raise ValueError(f'empty {what} in row {idx}')
    if '/' in s or os.sep in s:
        raise ValueError(f'{what} {s!r} in row {idx} contains a path separator')
    return s


if not CSV_MODE:
    # ======================= Mode A: single peptide subdir ===================
    sub_dir_path = os.path.join(base_directory, HLA, 'confidences', sub_dir)
    print(sub_dir_path)

    cif_path  = None
    json_path = None
    for file in os.listdir(sub_dir_path):
        file_path = os.path.join(sub_dir_path, file)
        if file.endswith('.cif'):
            cif_path = file_path
        elif file.endswith(f'{sub_dir}_confidences.json'):
            json_path = file_path

    if cif_path is None or json_path is None:
        raise FileNotFoundError(f'missing .cif and/or {sub_dir}_confidences.json in {sub_dir_path}')

    build_and_save(cif_path, json_path, sub_dir, out_path)

else:
    # ======================= Mode B: every row of the CSV ====================
    table = pd.read_csv(csv_path)
    path_col    = resolve_column(table, path_col)
    peptide_col = resolve_column(table, peptide_col)
    hla_col     = resolve_column(table, hla_col)
    name_col    = resolve_column(table, name_col)

    # Rows sharing (hla, 9mer, Original) resolve to the same output file, so
    # only the first of each set is kept.
    first_seen = {}
    dropped    = []   # (idx, index it duplicates, whether the Path matched too)
    unkeyable  = []   # bad hla/9mer/Original; kept so the loop reports them

    for idx, row in table.iterrows():
        try:
            k = (clean_component(row[hla_col], hla_col, idx),
                 clean_component(row[peptide_col], peptide_col, idx),
                 clean_component(row[name_col], name_col, idx))
        except ValueError:
            unkeyable.append(idx)
            continue
        if k in first_seen:
            kept = first_seen[k]
            same_path = (str(table.loc[idx, path_col]).strip()
                         == str(table.loc[kept, path_col]).strip())
            dropped.append((idx, kept, same_path))
        else:
            first_seen[k] = idx

    keep_idx = set(first_seen.values()) | set(unkeyable)

    if dropped:
        print(f'[dedupe] {len(dropped)} duplicate rows dropped '
              f'(same {hla_col}/{peptide_col}/{name_col}, so same output file); '
              f'kept the first of each', flush=True)
        for idx, kept, same_path in dropped[:20]:
            print(f'   row {idx} duplicates row {kept} '
                  f'({"same" if same_path else "DIFFERENT"} {path_col})', flush=True)
        if len(dropped) > 20:
            print(f'   ... and {len(dropped) - 20} more', flush=True)
        n_diff = sum(1 for _, _, same_path in dropped if not same_path)
        if n_diff:
            print(f'[warn] {n_diff} of the dropped rows pointed at a different {path_col} '
                  f'than the row kept -- those structures are NOT being built', flush=True)

    rows = [(idx, row) for idx, row in table.iterrows() if idx in keep_idx]
    if only_row is not None:
        rows = [rows[only_row]]   # index into the deduped list, not the raw CSV

    print(f'[csv] {len(table)} rows in {csv_path}; {len(keep_idx)} after dedupe; '
          f'processing {len(rows)}', flush=True)

    n_done = n_built = 0
    failures = []

    for idx, row in rows:
        row_dir = str(row[path_col]).strip()
        try:
            hla_value  = clean_component(row[hla_col], hla_col, idx)
            pep_value  = clean_component(row[peptide_col], peptide_col, idx)
            name_value = clean_component(row[name_col], name_col, idx)
        except ValueError as exc:
            failures.append((idx, row_dir, repr(exc)))
            print(f'[fail] row {idx}: {exc}', flush=True)
            continue

        key = pep_value   # the 9mer is the key inside the pickled dict
        out_path_row = os.path.join(csv_out_base, hla_value,
                                    f'{pep_value}_{name_value}' + CSV_OUT_SUFFIX)

        if output_is_done(out_path_row):
            n_done += 1
            print(f'[skip] row {idx}: already built -> {out_path_row}', flush=True)
            continue

        try:
            cif_path, json_path = find_inputs_generic(row_dir)
            print(f'[run] row {idx}: {key} <- {row_dir}', flush=True)
            build_and_save(cif_path, json_path, key, out_path_row)
            n_built += 1
        except Exception as exc:
            # One bad row must not cost the rest of the run.
            failures.append((idx, row_dir, repr(exc)))
            print(f'[fail] row {idx} ({row_dir}): {exc!r}', flush=True)

    print('=' * 70, flush=True)
    print(f'[csv] built {n_built}, already done {n_done}, failed {len(failures)}', flush=True)
    for idx, row_dir, err in failures:
        print(f'  row {idx}  {row_dir}  {err}', flush=True)

    # Exit non-zero so the scheduler marks the job as failed.
    if failures:
        sys.exit(1)
