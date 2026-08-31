# Reproducibility Pipelines

The detailed reproducibility pipelines and the corresponding links are listed
below.

## Fig. 2a, Supplementary Fig. 17a, and Supplementary Fig. 5c-d

Scripts for reproducing benchmark performance on the RNAmigos1 10-fold
random-split dataset and figure panels, together with the corresponding
swapping-test experiments:

- [Fig. 2a and Supplementary Fig. 17a figures](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/supplementary/other_figures/rnamigos1_benchmark.py)
  — `python rnamigos1_benchmark.py`; draws both panels from `raw_data/` and checks every bar
  against the shipped summary tables. No weights, no inference
- [RNAmigos1 10-fold benchmark](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/supplementary/other_figures/rnamigos1_benchmark.py)
  — `python benchmark_10fold.py --dataset rnamigos1`; re-derives the per-binder rank
  percentiles from the released checkpoints and checks them against the published means
- [Swapping tests](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/supplementary/other_figures/swapping_test.py)
  — `python swapping_test.py --dataset rnamigos1 --decoy-set both`; runs both the RNA and the
  ligand arm and writes the per-fold and summary tables. 

## Supplementary Fig. 5a-b

RNA- and ligand-swapping tests on the HARIBOSS 5-fold sequence-based split:

- [Swapping tests](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/supplementary/other_figures/swapping_test.py)
  — `python swapping_test.py --dataset hariboss --decoy-set pdb`

## Fig. 2b

Scripts for reproducing benchmark performance on the HARIBOSS 10-fold
cross-validation dataset using DecoyFinder- and DeepCoy-generated decoys:

- [DecoyFinder and DeepCoy inference](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig2/decoyfinder_deepcoy_inference.py)
  — `python decoyfinder_deepcoy_inference.py`; needs only the 10-fold pickle, the decoy
  libraries in `data/decoys/` and `SMARTBind_weight/`, and checks itself against the
  published values
- [Fig. 2b figure](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig2/figure2b.ipynb) — draws the panel from
  `raw_data/Fig2/`

## Fig. 2c-e and Supplementary Fig. 4a-c
Scripts for reproducing benchmark performance on the HARIBOSS 5-fold cross-validation datasets.

- [Fig. 2c-e figures](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig2/figure2cde.ipynb) — draws the three panels
  from `raw_data/Fig2/`
- [HARIBOSS 5-fold cross-validation inference](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig2/performance_reproduce_5fd_hariboss.py)
  — `python performance_reproduce_5fd_hariboss.py`; runs all ten split/weight pairs,
  prints both the per-sample (Fig. 2c) and fold-wise (Fig. 2d) summaries and checks
  itself against the published values

## Supplementary Fig. 1

Analysis of the ligand-specific augmented decoys (all seven panels, one notebook):

- [Decoy analysis](https://github.com/AIDD-LiLab/SMARTBind/tree/main/reproduce/supplementary/other_figures/supplementary_figure1.ipynb)
- [Decoy augmentation pipeline](https://github.com/AIDD-LiLab/SMARTBind/blob/main/notebook/augmentation_data.ipynb)

## Supplementary Fig. 3

The five data-splitting strategies applied to the HARIBOSS dataset:

- [HARIBOSS splitting notebooks](https://github.com/AIDD-LiLab/SMARTBind/tree/main/data/splittings)
  — `chain_split`, `pocket_split`, `ligand_split`, `pair_split`, `sequences_split`

## Fig. 3a-d and Supplementary Fig. 7a

Scripts for reproducing SMARTBind-predicted binding sites and the corresponding
figure panels:

- [Temporal-set binding prediction](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig3/temporal_set_binding_prediction.ipynb)
  — applies the released 10-model ensemble zero-shot to the 34 complexes and
  reports the per-case AUROC
- [Fig. 3a-d figures](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig3/figure3.ipynb) — draws the four panels
  from `raw_data/Fig3/`
- [Binding-site case visualization](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/supplementary/other_figures/binding_site_by_case_visualization.ipynb)

## Fig. 4

Scripts and outputs for reproducing virtual-screening performance on the
time-dependent test set and the corresponding figure panels:

- [Time-dependent virtual screening](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig4/time_dependent_vs.ipynb)
- [Fig. 4b, g-j figures](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig4/figure4.ipynb) — draws the panels
  from `raw_data/Fig4/`
- [Time-dependent ranking scores](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig4/ranking_percentile_scores.txt)
- [Case-study ranking scores](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig4/case_study_ranking_percentile_scores.txt)
- [Fig. 4c / 4e case-study screening](https://github.com/AIDD-LiLab/SMARTBind/blob/main/reproduce/fig4/case_study_vs.py)
  — `python case_study_vs.py`; screens the five case-study RNAs against the same 1M library
  and checks each SMARTBind cell of Fig. 4c and 4e against the published value.

## Fig. 4b

Implementation pipeline for SMARTBind-FAISS:

- [SMARTBind-FAISS implementation pipeline](https://github.com/AIDD-LiLab/SMARTBind/tree/main/reproduce/supplementary/faiss_index)

## Supplementary Fig. 8

Analysis scripts for structural comparison between the time-dependent test set
and the HARIBOSS dataset:

- [Time-dependent test-set analysis](https://github.com/AIDD-LiLab/SMARTBind/tree/main/data/time_dependent_set)

## Supplementary Fig. 10

Zero-shot benchmarking on the ROBIN dataset: the leakage analysis that defines the
21-target subset, and the SMARTBind rescreen on it.

- [ROBIN benchmark overview](https://github.com/AIDD-LiLab/SMARTBind/tree/main/reproduce/supplementary/robin)
- [ROBIN leakage analysis](https://github.com/AIDD-LiLab/SMARTBind/tree/main/reproduce/supplementary/robin/leakage)
- [SMARTBind inference on ROBIN](https://github.com/AIDD-LiLab/SMARTBind/tree/main/reproduce/supplementary/robin/inference)
