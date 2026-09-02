# Construction of temporal test set

### Introduction
To construct a temporal test set of RNA–ligand complexes, we queried the RCSB PDB Bank using advanced search filters to select structures containing at least one distinct RNA entity and one non-polymer instance. From this initial pool, we applied additional filtering criteria: ligands were required to be small molecules (100–1000 Da), solvent molecules and metal ions were excluded, and only complexes containing RNA and the ligand of interest, with no DNA or protein chains, were retained. RNA sequence overlap with the Hariboss training set was then quantified explicitly rather than assumed; see `sequence_similarity_mmseqs/`. For each retained complex, motif annotation was performed using RNA 3D Hub (https://rna.bgsu.edu/rna3dhub/) by identifying residues within 10 Å of the ligand to characterize local RNA 3D structural environments. For each complex, the canonical SMILES representation of the bound ligand was retrieved from the RCSB Chemical Component Dictionary via the RCSB REST API (https://data.rcsb.org/redoc/index.html#tag/Chemical-Component). Binding site ground truth labels were defined as RNA residues located within 10 Å of the ligand. The resulting temporal test set comprises 34 RNA–ligand complexes released between January 22, 2025, and November 26, 2025, providing a temporally independent benchmark for evaluating model generalization beyond the training data.

### Similarity analysis
To investigate potential data leakage and evaluation bias, we conducted both sequence- and structure-level homology analyses between the RNAs in the curated test set and those in the Hariboss training set. Pairwise RNA sequence similarity was quantified using MMseqs2 (v15.6f452), with sequence identity defined as the fraction of identical nucleotides over aligned columns (fident). The k-mer length was set explicitly to 4 rather than left at the nucleotide default of 15, which cannot seed on RNAs of 14-84 nt; see `sequence_similarity_mmseqs/README.md`. Structural similarity was assessed at the level of the full RNA chain using RMalign v1.1 (http://www.rnabinding.com/RMalign/RMalign.html).

RMalign structural comparison is described in `structure_analysis/`.

MMseqs2 sequence-similarity analysis: `sequence_similarity_mmseqs/` (script, results and
README). It finds five test complexes -- 9IO0, 9IO1, 9IOR, 9IOS, 9IOU, all with ligand
53D -- matching a training chain at 78.9-89.4% identity over 100% of both sequences.
These are the complexes the RNA-filtered evaluation subset removes.

### Further information and analysis for the results in plots and statistics can be found in `analysis.ipynb`