# MHCIIFold-GNN
## Repository for Learned Geometry, Predicted Binding: Prediction of Peptide:MHC Binding Using AlphaFold 3 Enables CD4 T Cell Epitope Prediction

- alphafold3 contains the nextflow & python scripts to run alphafold 3
- af3_vs_crystal contains the script to compare predicted alphafold 3 structures against known X-crystallography structures
- angles contains the code and output data to calculate the binding of a peptide with the MHC complex
- mhciifold_gnn contains the code to preprocess alphafold 3 output, train and test the model. A few example input HLAs have been provided
- netmhciipan_rebuilt contains the reconstructed model from NetMHCIIpan
