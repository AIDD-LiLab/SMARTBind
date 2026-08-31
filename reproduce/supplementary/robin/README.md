# SMARTBind on the ROBIN benchmark

Zero-shot virtual screening of SMARTBind on the ROBIN small-molecule-microarray
benchmark, and the leakage analysis that defines the target and ligand sets it is run
on. Supporting **Supplementary Fig. 10**.

ROBIN is an experimentally derived nucleic-acid binder library: 36 nucleic-acid targets
screened against 24,572 compounds by small-molecule microarray. We use the refined RNA
subset curated by GerNA-Bind (26 RNA targets). SMARTBind sees no ROBIN data during
training.

## Data provenance

Everything in `leakage/robin_official/` is upstream data, the filtering described below is applied on top of it, and every excluded target
and compound is listed explicitly.

| file | source | relation to upstream |
|---|---|---|
| `sequences.fasta` (26 targets) | GerNA-Bind release, Zenodo [17770641](https://zenodo.org/records/17770641), `Robin/sequences.fasta` | **byte-identical** (md5 `1226bf9c5f03f4daea01b2caa99628b5`) |
| `Robin_random.csv` (46,557 pairs; 3,508 binders over 1,908 ligands) | same release, `Robin/Robin_random.csv` | **byte-identical** (md5 `34d3c3d978da68c5f2449e91a0387f53`) |
| `robin_smm_nonbinders.smi` (22,410 unique) | the official ROBIN release, [`SMM_RNA_Non_Binder_3D.sdf`](https://doi.org/10.1002/ange.202211358) | read from the SDF with RDKit and written as standardised canonical SMILES |
| `robin_rna_targets.csv` (27 rows) | ROBIN's target table (name, 3D motif, sequence) | transcribed; the motif column agrees with GerNA-Bind's own per-target key suffixes on all 26 targets |

```
robin/
├── leakage/       leakage analysis
└── inference/     the leakage-filtered benchmark 
```

## Experiment

**1. `leakage/`** compares the *official, unfiltered* 26-target release against everything
SMARTBind was trained on, on both axes:

| axis | method | criterion | removed |
|---|---|---|---|
| RNA sequence | MMseqs2 | identity ≥ 0.3 **and** query and target coverage both ≥ 0.8 | **5 targets**: PreQ1, ZTP, TPP, SAM_ll, TERRA |
| ligand identity | RDKit canonical (isomeric) SMILES existence-based filter | exact string match against HARIBOSS training binders *or* their augmented decoys | **8 ROBIN binders**, **42 SMM decoys** |

**2. `inference/`** obtain the filtered ROBIN subset, and rescreens SMARTBind on it:

| | official release | after filtering |
|---|---|---|
| RNA targets | 26 | **21** |
| target–binder pairs | 3,508 | **2,681** |
| distinct binder molecules | 1,908 | **1,582** |
| SMM non-binder decoys | 22,410 | **22,368** |
| binders per target | 54–292 | **54–291** |

## Quick start

```bash
cd inference
bash run.sh --eval-only     # evaluate the shipped reference scores, no GPU needed
bash run.sh                 # rescreen with the 10-model ensemble, then evaluate
```

`run.sh` prints a PASS/FAIL line comparing what it just computed against the published
tables in `inference/reference/`.

## What you need to supply

Two artifacts are too large for git and come from the Zenodo record linked in the
repository README:

| needed for | artifact | where it goes |
|---|---|---|
| `inference/run_smartbind.py` | the 10-model SMARTBind ensemble | `SMARTBind_weight/` at the repository root, or `$SMARTBIND_WEIGHTS` |
| every script in `leakage/` | `hariboss_10fd_with_decoys.pkl` (the training set the ROBIN data is compared against) | `data/hariboss/`, or `$SMARTBIND_HARIBOSS_PKL` |

`leakage/` additionally needs the `mmseqs` binary on `PATH` (or `$MMSEQS`) for the
MMseqs2 script.
