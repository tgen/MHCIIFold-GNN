import argparse
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import pandas as pd
import glob

def cosine_between_segments(A, B, C, D):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    v1 = B - A
    v2 = D - C

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def analyzeDistance(peptide, position1, position2):
    mmcif_dict1 = MMCIF2Dict(peptide)
    pos80180 =((np.array(mmcif_dict1['_atom_site.label_seq_id']).astype(int) == 80) | (np.array(mmcif_dict1['_atom_site.label_seq_id']).astype(int) == 180)) &(np.array(mmcif_dict1['_atom_site.label_entity_id']).astype(int) ==1) & (np.array(mmcif_dict1['_atom_site.label_atom_id']) == 'CA')
    pos19 =((np.array(mmcif_dict1['_atom_site.label_seq_id']).astype(int) == position1) | (np.array(mmcif_dict1['_atom_site.label_seq_id']).astype(int) == position2)) &(np.array(mmcif_dict1['_atom_site.label_entity_id']).astype(int) ==3) & (np.array(mmcif_dict1['_atom_site.label_atom_id']) == 'CA')

    x = np.array(mmcif_dict1['_atom_site.Cartn_x']).astype(float)[pos80180]
    y = np.array(mmcif_dict1['_atom_site.Cartn_y']).astype(float)[pos80180]
    z = np.array(mmcif_dict1['_atom_site.Cartn_z']).astype(float)[pos80180]
    p80 =np.column_stack([x,y,z])

    x = np.array(mmcif_dict1['_atom_site.Cartn_x']).astype(float)[pos19]
    y = np.array(mmcif_dict1['_atom_site.Cartn_y']).astype(float)[pos19]
    z = np.array(mmcif_dict1['_atom_site.Cartn_z']).astype(float)[pos19]
    p19 =np.column_stack([x,y,z])

    cos = cosine_between_segments(p80[0], p80[1], p19[0], p19[1])
    return cos, np.degrees(np.arccos(cos))

def get_arguments():
    parser=argparse.ArgumentParser(description="Commands to pass to scripts")
    parser.add_argument('-hla', "--hla_info", 
            help = "HLA of MSA to add", required=True, type=str)
    parser.add_argument('-p', '--path',
                        help = 'Path of cif files', required=True, type=str)
    return parser.parse_args()



args = get_arguments()
hla = args.hla_info
path = args.path

peptides = []
degs = []

# all cifs expected to be nested from nextflow output
# path will have hla/confidence/peptide
# will calculate angle for all peptides in directory
cifs = glob.glob(f'{path}/{hla}/confidences/*/*.cif')

for i,cifdata in enumerate(cifs):
    seq = cifdata.split('/')[-2]

    position1 = np.floor(len(seq)*.25)
    position2 = np.floor(len(seq)*.75)
    peptides.append(seq)

    j,deg = analyzeDistance(cifdata, position1, position2)  

    degs.append(deg)

comboDict = {'peptides':peptides, 'angles':degs}
angleDF = pd.DataFrame(comboDict)
angleDF['hla'] = hla
angleDF[['hla', 'peptides', 'angles']].to_csv(f'{hla}BindingAngle.csv',index = False)

