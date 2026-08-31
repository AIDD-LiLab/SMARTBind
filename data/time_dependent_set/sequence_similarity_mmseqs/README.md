# Time-dependent test set vs the HARIBOSS training set — RNA sequence similarity

MMseqs2 comparison of the 34 time-dependent test RNAs against every RNA chain SMARTBind
was trained on. Same protocol, same criterion and same layout as the ROBIN leakage
analysis in `reproduce/supplementary/robin/leakage/rna_mmseqs/`.

```bash
python mmseq_cal.py          # needs `mmseqs` on PATH (or $MMSEQS); run with MMseqs2 15.6f452
```

Also needs the HARIBOSS training set (`data/hariboss/hariboss_10fd_with_decoys.pkl`,
from the Zenodo record linked in the repository README) — set `SMARTBIND_HARIBOSS_PKL`
if you keep it elsewhere.

## Criterion

A test complex is flagged when some training chain reaches

> identity ≥ 0.3 **and** query coverage ≥ 0.8 **and** target coverage ≥ 0.8

which is the criterion of our RNA sequence split. The search is left deliberately wide
open (`--search-type 3 -s 7.5 --min-seq-id 0 --cov-mode 0 -c 0 -e 10000 -k 4`) and the
three thresholds are applied afterwards, so nothing is discarded before it has been
scored.

`-k 4` is used in place of the nucleotide default of 15. The test RNAs are 14–84 nt, so
a 15-mer prefilter has almost nothing to seed on: with the default, MMseqs2 returns a hit
for only **2 of the 34** queries and flags none. At `-k 4` all 34 queries align.

## Result

**Five of the 34 complexes are flagged: 9IO0, 9IO1, 9IOR, 9IOS, 9IOU.**

All five carry ligand **53D**, all are 19-nt RNAs, and each matches a training chain at
78.9–89.4% identity over **100% of both** sequences:

| PDB | chain | ligand | best training chain | identity | query cov | target cov |
|---|---|---|---|---|---|---|
| 9IO0 | A | 53D | `8i46_A_B_53D_20` | 89.4% | 100.0% | 100.0% |
| 9IO1 | A | 53D | `8i44_A_B_53D_20` | 89.4% | 100.0% | 100.0% |
| 9IOR | A | 53D | `8i46_A_B_53D_20` | 89.4% | 100.0% | 100.0% |
| 9IOS | A | 53D | `8i46_A_B_53D_20` | 78.9% | 100.0% | 100.0% |
| 9IOU | A | 53D | `8i46_A_B_53D_20` | 78.9% | 100.0% | 100.0% |

`mmseqs_results.csv` holds all 6,572 scored hits; `temporal_vs_training_BEST_HITS.csv`
holds the 9 pairs that meet the criterion.

These five are what the **RNA-filtered** evaluation subset removes, alongside the
chain-level RMscore filter — see `../structure_analysis/`. The resulting per-case
flags ship with the figure that uses them, as the
`chain_structure_similar_to_training`, `rna_sequence_similar_to_training`,
`ligand_present_in_training` and `passes_rna_filter` columns of
`raw_data/Fig3/Fig3c_auroc_auprc_by_filter_scheme_per_case.csv`,
which `reproduce/fig3/figure3.ipynb` reads to build the four evaluation subsets.

`query_temporal.fasta` and `target_training.fasta` are intermediates the script
regenerates from `../time_dependent_test_set.csv` and the training pickle, so they are
not tracked.