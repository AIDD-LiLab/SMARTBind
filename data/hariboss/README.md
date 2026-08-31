# HARIBOSS training data

The processed HARIBOSS pickles are distributed through the Zenodo record linked in the
repository README, not stored in git. Download them here:

```
data/hariboss/
├── hariboss_10fd_with_decoys.pkl                            10-fold random split (the master file)
├── hariboss_5fd_cov08_clustered_seqid_0.3_clean.pkl         RNA sequence-based 5-fold split
├── hariboss_5fd_by_chain_structure_no_identical_clean.pkl   RNA structure-based
├── hariboss_5fd_by_pocket_structure_no_identical_clean.pkl  RNA pocket-based
├── hariboss_5fd_by_pair_split_clean.pkl                     pair-based
├── hariboss_5fd_by_ligand_existence_clean.pkl               ligand-existence
└── subset/                                                  the same five, GerNA-Bind subset
```
