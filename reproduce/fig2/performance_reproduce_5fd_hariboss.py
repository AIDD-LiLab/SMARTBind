#!/usr/bin/env python3
"""SMARTBind on the HARIBOSS 5-fold cross-validation splits — the numbers behind
Fig. 2c-e and Supplementary Fig. 4a-b.

Ten runs: five data-splitting strategies, each on the full HARIBOSS set and on the
GerNA-Bind subset. For every fold the fold's model scores each validation RNA-ligand
pair against that fold's decoy pool (all unique validation ligands that are not known
binders of the RNA), and records the pair's rank percentile and AUROC.

    rank percentile = (n_decoys + 1 - rank) / (n_decoys + 1)     1.0 = binder ranks first
    AUROC           = binder vs its decoys, one value per pair

Two summary conventions appear in the paper and both are printed:

    per-sample   mean +/- s.d. over every validation pair      -> Fig. 2c  (0.797 +/- 0.225)
    fold-wise    mean +/- s.d. over the five fold means        -> Fig. 2d  (0.760 +/- 0.024, ...)

Inputs, all released with the paper:

    data/hariboss/*_clean.pkl            the five 5-fold splits           (Zenodo)
    data/hariboss/subset/*_clean.pkl     the same five, GerNA-Bind subset (Zenodo)
    SMARTBind_weight/cross_validation/   the 50 cross-validation checkpoints

The cross-validation checkpoints are separate from the 10-model ensemble in
SMARTBind_weight/ — they are the models trained on each split, 5 folds each.

Usage
    python performance_reproduce_5fd_hariboss.py                        # all ten runs
    python performance_reproduce_5fd_hariboss.py --datasets full_sequenceSplit0803
    python performance_reproduce_5fd_hariboss.py --folds 1              # smoke test

Runtime: roughly 10 min per fold on one CPU core, far less on a GPU.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import roc_auc_score


def find_repo_root(start):
    """Walk up until the SMARTBind package is found."""
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "smartbind", "__init__.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("could not locate the SMARTBind repository root from " + start)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = find_repo_root(HERE)
sys.path.insert(0, REPO)

from smartbind import load_smartbind_models                       # noqa: E402
from smartbind.dataloader import RLDataLoader, RLDataset          # noqa: E402
from smartbind.model.margin import sigmoid_cosine_distance_test   # noqa: E402
from smartbind.device import resolve_device                 # noqa: E402
from smartbind.preprocess import build_val_test_set               # noqa: E402

FULL = os.path.join(REPO, "data", "hariboss")
SUBSET = os.path.join(FULL, "subset")

# name -> (checkpoint directory, dataset pickle)
DATASETS = {
    "full_sequenceSplit0803": ("newFP2_sequenceSplit0803_v2",
                               f"{FULL}/hariboss_5fd_cov08_clustered_seqid_0.3_clean.pkl"),
    "full_chainStructureNoIdentical": ("newFP2_chainStructureNoIdentical_v2",
                                       f"{FULL}/hariboss_5fd_by_chain_structure_no_identical_clean.pkl"),
    "full_pocketStructureNoIdentical": ("newFP2_pocketStructureNoIdentical_v2",
                                        f"{FULL}/hariboss_5fd_by_pocket_structure_no_identical_clean.pkl"),
    "full_pairSplit": ("newFP2_pairSplit_v2",
                       f"{FULL}/hariboss_5fd_by_pair_split_clean.pkl"),
    "full_ligandExist": ("newFP2_ligandExist_v2",
                         f"{FULL}/hariboss_5fd_by_ligand_existence_clean.pkl"),
    "GB_sequenceSplit0803": ("newFP2_GB_sequenceSplit0803_v2",
                             f"{SUBSET}/hariboss_5fd_cov08_clustered_seqid_0.3_subset_clean.pkl"),
    "GB_chainStructureNoIdentical": ("newFP2_GB_chainStructureNoIdentical_v2",
                                     f"{SUBSET}/hariboss_5fd_by_chain_structure_no_identical_subset_clean.pkl"),
    "GB_pocketStructureNoIdentical": ("newFP2_GB_pocketStructureNoIdentical_v2",
                                      f"{SUBSET}/hariboss_5fd_by_pocket_structure_no_identity_subset_clean.pkl"),
    "GB_pairSplit": ("newFP2_GB_pairSplit_v2",
                     f"{SUBSET}/hariboss_5fd_by_pair_split_subset_clean.pkl"),
    "GB_ligandExist": ("newFP2_GB_ligandExist_v2",
                       f"{SUBSET}/hariboss_5fd_by_ligand_existence_subset_clean.pkl"),
}

# published values: n_pairs, per-sample mean, fold-wise mean, fold-wise s.d.
PUBLISHED = {
    "full_sequenceSplit0803": (1246, 0.7971, 0.7970, 0.0446),
    "full_chainStructureNoIdentical": (1246, 0.7599, 0.7601, 0.0240),
    "full_pocketStructureNoIdentical": (1232, 0.7104, 0.7104, 0.0679),
    "full_pairSplit": (751, 0.7199, 0.7212, 0.0461),
    "full_ligandExist": (1246, 0.8166, 0.8197, 0.0254),
    "GB_sequenceSplit0803": (681, 0.7842, 0.7842, 0.0414),
    "GB_chainStructureNoIdentical": (681, 0.7535, 0.7534, 0.0690),
    "GB_pocketStructureNoIdentical": (681, 0.7896, 0.7896, 0.0650),
    "GB_pairSplit": (574, 0.7427, 0.7427, 0.0274),
    "GB_ligandExist": (681, 0.8327, 0.8326, 0.0516),
}

TOPK_DECOY = 100
BATCH_SIZE = 24
NUM_WORKERS = 2


def checkpoint_root():
    path = os.environ.get("SMARTBIND_CV_WEIGHTS") or os.path.join(
        REPO, "SMARTBind_weight", "cross_validation")
    if not os.path.isdir(path):
        raise SystemExit(
            "\nThe cross-validation checkpoints were not found at:\n  " + path + "\n\n"
            "They are separate from the 10-model ensemble, and unpack to\n"
            "SMARTBind_weight/cross_validation/ as ten newFP2_*_v2/ folders of\n"
            "fold1.pth .. fold5.pth. Or point SMARTBIND_CV_WEIGHTS at them.\n")
    return path


def require(path, what):
    if not os.path.exists(path):
        raise SystemExit(
            f"\n{what} was not found at:\n  {path}\n\n"
            "The processed HARIBOSS splits are distributed through the Zenodo record linked\n"
            "in the repository README; download them into data/hariboss/.\n")
    return path


def run_fold(data_path, weight_path, fold_num, device):
    """Score every validation pair of one fold. fold_num is 1-based."""
    (val_rna_seq_list, _, val_rna_name_list, val_match_smol_name_list,
     val_binding_index_list, val_match_smol_fp_list, *rest) = build_val_test_set(
        fold_num=fold_num - 1, dict_path=str(data_path), topk_decoy=TOPK_DECOY,
        return_rna_smol_map=True)
    rna_smol_map = rest[-1]

    dataset = RLDataset(rna_sequences=val_rna_seq_list,
                        rna_sequences_names=val_rna_name_list,
                        match_smols=val_match_smol_fp_list,
                        match_smols_names=val_match_smol_name_list,
                        non_binding_index_list=val_binding_index_list,
                        rna_smol_map=rna_smol_map,
                        is_val=True)
    loader = RLDataLoader(dataset=dataset, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS, if_shuffle=False)

    model = load_smartbind_models(model_path=str(weight_path), device=device, vs_mode=True)
    model.eval()

    records = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader.dataloader, desc=f"  fold {fold_num}", leave=False):
            rna_sequences, match_smols, decoy_smols_list, rna_names, smol_names, _ = batch
            rna_embeddings = [model.inference_single_rna(seq[1]) for seq in rna_sequences]

            for anchor, match_smol, decoy_smols, rna_name, smol_name in zip(
                    rna_embeddings, match_smols, decoy_smols_list, rna_names, smol_names):
                if len(decoy_smols) == 0:
                    print(f"  no decoys for {rna_name} / {smol_name}, skipped")
                    continue
                anchor = anchor.squeeze(1)
                match_tok = model.inference_list_smols([match_smol[1]])[0].unsqueeze(dim=0)
                match_score = sigmoid_cosine_distance_test(anchor, match_tok).item()
                decoy_scores = [
                    sigmoid_cosine_distance_test(anchor, tok.unsqueeze(dim=0)).item()
                    for tok in model.inference_list_smols(decoy_smols)
                ]
                auroc = roc_auc_score([1] + [0] * len(decoy_scores),
                                      [match_score] + decoy_scores)
                # a decoy scoring at least as high as the true binder outranks it
                rank = sum(1 for d in decoy_scores if match_score <= d)
                n = len(decoy_scores)
                records.append({"fold": fold_num, "rna_name": rna_name,
                                "ligand_name": smol_name, "n_decoys": n, "rank": rank,
                                "rank_percentile": (n + 1 - rank) / (n + 1),
                                "auroc": float(auroc)})
    del model
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="auto",
                        help="Device: 'cpu', 'gpu' (alias for 'cuda'), 'cuda:N', or 'auto' "
                             "(default: a GPU when one is visible, otherwise CPU).")
    parser.add_argument("--out", default=os.path.join(HERE, "fig2cde_per_pair.csv"))
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    start = time.time()
    weights = checkpoint_root()
    print(f"device={args.device} | checkpoints {weights}")

    rows = []
    for name in args.datasets:
        weight_dir, data_path = DATASETS[name]
        require(data_path, f"The dataset for {name}")
        print(f"\n=== {name} ===\n  data    {os.path.basename(data_path)}"
              f"\n  weights {weight_dir}", flush=True)
        for fold in args.folds:
            checkpoint = os.path.join(weights, weight_dir, f"fold{fold}.pth")
            if not os.path.exists(checkpoint):
                raise SystemExit(f"\nmissing checkpoint: {checkpoint}\n")
            for record in run_fold(data_path, checkpoint, fold, args.device):
                record["dataset"] = name
                rows.append(record)
            done = [r for r in rows if r["dataset"] == name and r["fold"] == fold]
            print(f"  fold {fold}: {len(done):>4} pairs, mean rank percentile "
                  f"{np.mean([r['rank_percentile'] for r in done]):.4f}"
                  f"  ({time.time() - start:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(f"\n[OUT] {args.out}  ({len(table):,} pairs)")

    complete = set(args.folds) == {1, 2, 3, 4, 5}
    print(f"\n{'dataset':<34}{'n':>6}{'per-sample mean±sd':>22}{'fold-wise mean±sd':>22}"
          f"{'  published (fold-wise)':>24}")
    worst = 0.0
    for name, block in table.groupby("dataset"):
        v = block.rank_percentile
        folds = block.groupby("fold").rank_percentile.mean()
        n_pub, mean_pub, fmean_pub, fsd_pub = PUBLISHED[name]
        print(f"{name:<34}{len(v):>6}   {v.mean():.4f} ± {v.std(ddof=0):.4f}"
              f"      {folds.mean():.4f} ± {folds.std(ddof=1):.4f}"
              f"      {fmean_pub:.4f} ± {fsd_pub:.4f}")
        if complete:
            worst = max(worst, abs(v.mean() - mean_pub), abs(folds.mean() - fmean_pub))
            if len(v) != n_pub:
                print(f"    ! n differs from the published {n_pub}")
                worst = float("inf")
    ok = True
    if complete:
        ok = worst < 2e-2
        verdict = "PASS" if ok else "MISMATCH"
        print(f"\n  {verdict}: worst deviation from the published means = {worst:.2e}")
        print("  (a few 1e-2 deviations are expected: ~9% of pairs are exactly tied with a\n"
              "   decoy - molecules whose FP2 is bit-identical - so their rank flips under\n"
              "   any 1e-7 floating-point difference between machines.)")
    else:
        print("\n  (partial run: --folds was not 1-5, so no regression check)")
    print(f"total {time.time() - start:.0f}s")
    # non-zero on MISMATCH, so a piped or CI run can tell success from failure
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
