import argparse
import json
import pickle

def get_arguments():
    parser=argparse.ArgumentParser(description="Commands to pass to scripts")
    parser.add_argument('-j', "--json", 
            help = "json file of MSA peptide", required=True, type=str)
    parser.add_argument('-h', "--hla", 
            help = "HLA of MSA to add", required=True, type=str)
    parser.add_argument('-hD', "--hlaDictionary", 
            help = "HLA of MSA to add", required=True, type=str)
    parser.add_argument('-p', "--peptide", 
            help = "peptide sequece", required=True, type=str)
    return parser.parse_args()

args = get_arguments()
json_file = args.json
hla = args.hla_info
peptide = args.peptide
hlaDict = args.hlaDictionary

with open(json_file) as f:
    peptide_json = json.load(f)

# HLA dict expects an HLA ID as key and the value as the extracted 'sequences' portion of the HLA
# alpha and beta chain after running through MSA only need to do once per hla

with open(hlaDict) as f:
    hla_dict = pickle.load(f)
    
peptide_json['sequences'].extend(hla_dict[hla])


with open(f"{peptide}_full_data.json", 'w') as f:
    json.dump(peptide_json,f,indent=2)