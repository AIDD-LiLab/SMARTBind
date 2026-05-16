# Datasets and Splittings

This folder contains data artifacts and analysis notebooks used to build and
evaluate the SMARTBind benchmark splits, decoy sets, and temporal test set.

## Folder Overview
### `decoyfinder/`

Scripts for generating ligand-specific decoys with a modified DecoyFinder
workflow. `df_script.py` contains the decoy search logic and `df_submit.sh`
shows the corresponding environment setup and submission entry point.

### `decoys/`

Serialized decoy sets used for virtual-screening evaluation.

- `decoyfinder_decoys/`: DecoyFinder-generated decoys stored as a pickle file.
- `deepcoy_decoys/`: DeepCoy-generated decoy SMILES pickles for the DUD-E and
  DEKOIS decoy sources.

### `splittings/`

Jupyter notebooks used to construct and validate the 5-fold cross-validation HARIBOSS dataset.

- `sequences_split.ipynb`: RNA sequence-based split, in which similar sequence (sequence identity ≥ 0.3) were assigned to the same training or test set.
- `chain_split.ipynb`: RNA structure-based split, in which structurally similar RNA chains (RMscore ≥ 0.75) were assigned to the same training or test set.
- `pocket_split.ipynb`: RNA pocket-based split, in which structurally similar binding pockets (RMscore ≥ 0.75) were assigned to the same training or test set.
- `ligand_split.ipynb`: Ligand-existence split, ensuring no overlap between small molecules in the training and test sets. 
- `pair_split.ipynb`: Pair-based split, ensuring no overlap between both RNA targets and small molecules in the training and test sets.

The training data for all the splits above can be downloaded from https://drive.google.com/drive/folders/1j7QMGHKyhpJLasyXLjYZ2Db9SbvKNKbR?usp=sharing.
The GerNA-Bind subset of the training data can be downloaded from https://drive.google.com/drive/folders/1V5jDPI4scKlNuW9RbB7uS90xDNT15Oke?usp=sharing.

### `time_dependent_set/`

Curated temporal test set for evaluating generalization to RNA-ligand complexes
released after the training data. See `time_dependent_set/README.md` for the
full construction details.

- `time_dependent_test_set.csv`: main temporal benchmark table with PDB IDs,
  ligands, RNA chains, sequences, and SMILES.
- `case_study_test_set.csv`: small case-study subset used for focused analysis.
- `mmseq_cal.py`: script for MMseqs2 sequence-similarity checks against the
  training set.
- `analysis.ipynb`: notebook for summary statistics and plotting.
- `structure_analysis/`: RMalign chain- and pocket-level similarity matrices and
  nearest-neighbor summaries.

