# SMARTBind on the leakage-filtered ROBIN SMM benchmark

The benchmark that survives the analysis in `../leakage/`, and a one-click rescreen of
SMARTBind on it. Supporting the SMM enrichment panel of **Supplementary Fig. 10**.

## Run it

```bash
bash run.sh --eval-only     # no GPU, no weights: evaluate the shipped reference scores
bash run.sh                 # full rescreen with the 10-model ensemble, then evaluate
```

`run.sh` ends with a PASS/FAIL line comparing what it just computed against the published
tables in `reference/`. Measured on one CPU core, the full screen takes **~90 s**
(21 targets × 25,049 molecules × 10 models); on a GPU it is faster still.

The screen needs the 10-model ensemble in `SMARTBind_weight/` at the repository root
(from the Zenodo record linked in the top-level README). If you keep it elsewhere:

```bash
export SMARTBIND_WEIGHTS=/path/to/SMARTBind_weight
```

## Files

```
inference/
├── run.sh                     one-click driver
├── run_smartbind.py           the screen: FP2 → 10-model ensemble → rank percentiles
├── evaluate_ef.py             per-target EF, AUROC, paired Wilcoxon, the panel
├── data/                      the leakage-filtered benchmark
│   ├── sequences.fasta        21 RNA targets
│   ├── binders.csv            2,681 target–binder pairs (1,582 distinct molecules)
│   └── decoys.smi             22,368 experimentally measured SMM non-binders
├── baselines/                 per-binder scores for the three comparison methods
├── reference/                 the published SMARTBind scores and result tables
└── results/                   written by evaluate_ef.py (git-ignored)
```

## The benchmark

| | official release | here | removed |
|---|---|---|---|
| RNA targets | 26 | **21** | PreQ1, ZTP, TPP, SAM_ll, TERRA |
| target–binder pairs | 3,508 | **2,681** | 813 on the dropped targets + 14 by the ligand filter |
| distinct binder molecules | 1,908 | **1,582** | 8 by the ligand filter, the rest with the dropped targets |
| SMM decoys | 22,410 | **22,368** | 42 by the ligand filter |
| binders per target | 54–292 | **54–291** | |

See `../leakage/README.md` for how each exclusion was decided. Ligand SMILES are used
exactly as released. They are **not** re-canonicalised here, because OpenBabel FP2 is
computed from the SMILES string.

## Method

```
score  = mean over the 10 models of cos(RNA embedding, ligand embedding)
rank%  = (# decoys scoring strictly higher) / 22,368 × 100      lower is better
EF@t%  = (fraction of a target's binders with rank% ≤ t) / (t / 100)
AUROC  = 1 − mean(rank%) / 100
```

EF is averaged over the 21 targets. Significance is a **paired two-sided Wilcoxon
signed-rank test** of SMARTBind against each baseline over those 21 targets, with the
normal approximation and no continuity correction (EF values are heavily tied, so the
exact permutation null does not apply). **p-values are not corrected for multiple
comparisons.**

The decoy pool changed relative to the published 26-target screen, so old scores cannot
be reused: dropping 42 decoys shifts every binder's rank percentile. This is a full
rescoring, not a subset of stored values.

## Results

Mean EF across the 21 targets (mean ± s.d.), reproduced by `evaluate_ef.py`:

| model | EF@0.5% | EF@1% | EF@3% | EF@5% |
|---|---|---|---|---|
| **SMARTBind** | 1.99 ± 2.38 | **1.77 ± 1.35** | **1.67 ± 0.75** | **1.44 ± 0.53** |
| GerNA-Bind | **2.38 ± 2.06** | 1.69 ± 1.33 | 1.20 ± 0.76 | 1.00 ± 0.56 |
| DrugBAN | 1.24 ± 1.87 | 1.20 ± 1.33 | 1.08 ± 0.64 | 1.10 ± 0.65 |
| GraphDTA | 0.91 ± 1.16 | 0.90 ± 0.93 | 1.02 ± 0.60 | 1.04 ± 0.50 |

## Reproducibility notes

* Re-running `evaluate_ef.py` on the **shipped** per-binder file reproduces the published
  tables to 1e-15.
* A **fresh** screen reproduces the EF values exactly and AUROC to ~1e-6. Torch
  reductions are not bit-reproducible across devices and thread counts, which moves a
  handful of near-tied binders past a decoy and shifts their individual rank percentile
  in the third decimal. EF is a threshold count and is unaffected; AUROC is a mean of
  rank percentiles and therefore moves very slightly. `evaluate_ef.py` applies the
  appropriate tolerance for each case and says which one it used.
* `run_smartbind.py` caches FP2 fingerprints in `cache/`; delete it to re-featurise.
