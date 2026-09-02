# ROBIN vs the SMARTBind training set — leakage analysis

What the *official, unfiltered* 26-target ROBIN release shares with everything SMARTBind
was trained on, measured on two axes. The result of this analysis is the 21-target
benchmark in `../inference/`.

```
leakage/
├── robin_official/   the official release, before any filtering
├── rna_mmseqs/       RNA sequence similarity, MMseqs2
└── ligand/           ligand identity, exact canonical SMILES
```

Both scripts need the HARIBOSS training set
(`data/hariboss/hariboss_10fd_with_decoys.pkl`, from the Zenodo record linked in the
repository README). Set `SMARTBIND_HARIBOSS_PKL` if you keep it elsewhere. Each script
ships its outputs, so the conclusions can be checked without re-running anything.

---

## `robin_official/` — the input

Upstream data, unmodified by us. Full provenance and checksums are in `../README.md`.

| file | contents | source |
|---|---|---|
| `sequences.fasta` | the **26** official RNA targets | GerNA-Bind, Zenodo [17770641](https://zenodo.org/records/17770641) — byte-identical |
| `Robin_random.csv` | the official pairs; the 3,508 label = 1 rows are the true binders (1,908 distinct molecules) | same release — byte-identical |
| `robin_smm_nonbinders.smi` | 22,410 unique experimentally measured SMM non-binders | official ROBIN release, [`SMM_RNA_Non_Binder_3D.sdf`](https://doi.org/10.1002/ange.202211358), read with RDKit and written as standardised canonical SMILES (**not** part of the GerNA-Bind release) |
| `robin_rna_targets.csv` | ROBIN's own target table, incl. the 3D-motif annotation | transcribed from ROBIN; motif column cross-checked against GerNA-Bind's per-target keys (26/26 agree) |

> `robin_rna_targets.csv` writes `ENE_A9` as a two-strand construct
> (`AAAAAAAAA and GGCUGGG...`). `sequences.fasta`, which is what is actually screened
> and aligned, carries only the second strand. The table column is descriptive; nothing
> downstream reads it for sequence.

ROBIN's target table lists 27 RNAs. `RRE2B_MeA` is `RRE2B` carrying two
N6-methyladenosines, so it cannot be written as a distinct A/C/G/U sequence and
GerNA-Bind's released benchmark leaves it out. Everything here is over the remaining 26.

---

## `rna_mmseqs/` — RNA sequence similarity

```bash
python mmseq_cal.py          # needs `mmseqs` on PATH (or $MMSEQS); run with MMseqs2 15.6f452
```

Same protocol as the time-dependent test set, and the same criterion as our RNA sequence
split:

> identity ≥ 0.3 **and** query coverage ≥ 0.8 **and** target coverage ≥ 0.8

The search is left deliberately wide open (`--search-type 3 -s 7.5 --min-seq-id 0
--cov-mode 0 -c 0 -e 10000 -k 4`) and the three thresholds are applied afterwards, so
nothing is discarded before it has been scored. `-k 4` is used in place of the nucleotide
default of 15, which cannot seed on RNA targets of 10–84 nt.

**Five targets are flagged: PreQ1, SAM_ll, TERRA, TPP, ZTP.**

`mmseqs_results.csv` holds all 5,181 scored hits; `robin_vs_training_BEST_HITS.csv` holds
the 39 pairs that meet the criterion. Four of the five are riboswitches essentially
identical to a training chain over their whole length:

| ROBIN target | best training chain | identity | query cov | target cov |
|---|---|---|---|---|
| PreQ1 | `3fu2_A_D_PRF_101` | 100.0% | 100.0% | 100.0% |
| ZTP | `6od9_A_D_AMZ_102` | 100.0% | 94.9% | 100.0% |
| TPP | `2gdi_X_K_TPP_100` | 98.7% | 91.7% | 96.2% |
| SAM_ll | `2qwy_A_H_SAM_100` | 90.1% | 98.1% | 98.1% |
| TERRA | `3mij_A_D_R14_30` | 100.0% | 100.0% | 83.3% |

TERRA is a 10-nt telomeric repeat sitting inside a 12-nt training chain.

`query.fasta` and `target_training.fasta` are intermediates the script regenerates from
the ROBIN FASTA and the training pickle, so they are not tracked.

---

## `ligand/` — exact canonical SMILES

```bash
python ligand_overlap.py
```

Two molecules count as the same **only** if RDKit canonicalises them to the identical
isomeric SMILES string. No fingerprint, no Tanimoto, no similarity cutoff.
Canonicalisation is what makes the comparison meaningful: the two sides write SMILES in
different conventions, so a raw string comparison reports far fewer matches than there
are molecules in common.

Four sets are crossed (`ligand_overlap_summary.csv`):

| ROBIN set | vs HARIBOSS binders (311) | vs HARIBOSS augmented decoys (22,686) |
|---|---|---|
| binders (1,908) | **1** | **7** |
| SMM decoys (22,410) | **6** | **37** |

`ligand_overlap_matches.csv` names every matched molecule and, for training binders, the
ligand ID and how many complexes it appears in.

The compounds removed downstream are listed in `excluded_ligands.csv` (**8** ROBIN
binders: 1 matching a training binder, 7 matching an augmented decoy) and
`excluded_decoys.csv` (**42** SMM decoys: 6 training binders, 37 augmented decoys, one
molecule matching both). The two lists do not overlap, so **50 distinct ROBIN compounds**
are removed in total.

The 8 excluded binders account for 15 target–binder pairs, 14 of them on the 21 targets
that survive the RNA filter, which is why per-target binder counts drop by 1–2 relative
to the official release (e.g. EWSR1 292 → 291, AKTIP 223 → 221).
