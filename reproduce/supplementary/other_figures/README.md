# Fig. 2a, Supplementary Fig. 17a, and the per-case binding-site chart

Two scripts, one for each half of the same result.

## Drawing the panels — no weights, no inference

```bash
python rnamigos1_benchmark.py
```

Draws Fig. 2a (SMARTBind against RNAmigos1) and Supplementary Fig. 17a (the same plus
SMARTBind-noAug) from `raw_data/Fig2/` and `raw_data/SupplementaryFig17/`, and checks every
drawn bar — height, error bar and n — against the shipped `*_summary.csv`. It exits non-zero
if any of them disagree. Seconds, CPU only.

RNAmigos1's own bars are the values its paper reports, read from the summary tables rather
than recomputed: no per-binder record exists for RNAmigos1.

## Re-deriving the numbers from the checkpoints

```bash
cd ../swap_test
python benchmark_10fold.py --dataset rnamigos1 --decoy-set both
```

## Supplementary Fig. 7b–d

`binding_site_by_case_visualization.ipynb` renders the per-case binding-site structures.
