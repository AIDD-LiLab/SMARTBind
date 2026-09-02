# Baseline scores on the leakage-filtered ROBIN SMM benchmark

Per-binder rank percentiles for the three comparison methods, in the same shape as
SMARTBind's own output so `../evaluate_ef.py` treats all four identically:

```
target,SMILES,score,rank_percentile
```

| file | method | model used |
|---|---|---|
| `gernabind_smm_filtered_per_binder.csv` | GerNA-Bind | retrained on the PDB-derived HARIBOSS subset |
| `drugban_fold10_smm_filtered_per_binder.csv` | DrugBAN_RNA | fold 10 of the HARIBOSS 10-fold random split |
| `graphdta_fold7_smm_filtered_per_binder.csv` | GraphDTA_RNA | fold 7 of the HARIBOSS 10-fold random split |

Every file covers all 21 targets and all 2,681 target–binder pairs.

## Why these are shipped as results rather than scripts

All three were scored against **this exact decoy pool**: the 22,368 SMM non-binders in
`../data/decoys.smi`, after the leakage filter. Rank percentile is defined against the
pool, so removing 42 decoys changes every value; the published 26-target scores could
not simply be subsetted, and each method had to be re-run by whoever held its weights.

Re-running them from this repository is not possible:

* **GerNA-Bind** requires its own trained checkpoint, which is not ours to redistribute.
  See the upstream repository cited in the manuscript's Code Availability statement.
* **DrugBAN_RNA** and **GraphDTA_RNA** are our adaptations of the upstream protein–ligand
  models: their protein-sequence encoders were replaced to accept RNA sequences, and the
  models retrained on HARIBOSS. The upstream repositories cited in Code Availability do
  **not** contain the RNA encoder, so the adapted code and checkpoints are needed in
  addition to them.

`rank_percentile` is `(# decoys scoring strictly higher) / 22,368 × 100`, so lower is
better and a random scorer averages 50. Each method reported that it parsed the whole
decoy pool; no molecules were dropped, so all four methods share the same denominator.
