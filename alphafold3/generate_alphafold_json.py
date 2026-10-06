import argparse
import json

def get_arguments():
    parser=argparse.ArgumentParser(description="Commands to pass to scripts")
    parser.add_argument('-p', "--peptide", 
            help = "File of individual you wish to calculate base frequencies", required=True, type=str)
    return parser.parse_args()

args = get_arguments()
peptide = args.peptide


json_dict = {
    "name": peptide,
    "modelSeeds": [42],
    "sequences": [
        {
            "protein": {
                "id": "PEPTIDE",
                "sequence": peptide
            }
        }
    ],
    "dialect": "alphafold3",
    "version": 1
}



with open(f"{peptide}.json", 'w') as f:
    json.dump(json_dict,f,indent=2)