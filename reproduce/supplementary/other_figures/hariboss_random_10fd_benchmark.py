#!/usr/bin/env python3
"""SMARTBind on the HARIBOSS 10-fold random split.

NOT A PUBLISHED EXPERIMENT.  No figure, table or number in the manuscript comes from
this script, and there is therefore nothing to check the output against -- unlike
`reproduce/fig2/performance_reproduce_5fd_hariboss.py`, which compares itself to the
published Fig. 2c-e means and exits non-zero on a mismatch.  Treat what this prints as
a fresh measurement, not as a reproduction.

What it does: for each fold, the released ensemble checkpoint trained on that fold
(`SMARTBind_weight/fold{N}.pth`) scores every validation pair of that fold against the
fold's decoy pool -- all unique validation ligands that are not known binders of the
RNA, the same pool definition the 5-fold benchmark uses.

    rank percentile = (n_decoys + 1 - rank) / (n_decoys + 1)     1.0 = binder ranks first
                      rank counts decoys scoring at least as high as the binder

That is the definition used by the training code (`smartbind/model/pl_train/contact.py`
::_rank_eval) and by every other benchmark script in this repository.

A caveat that matters for reading the result: in
`data/hariboss/hariboss_10fd_with_decoys.pkl`, 490 of the 1,246 entries have `train_split`
all-True, i.e. they are in the training half of every fold and are never validated.  The
10 folds therefore cover 756 pairs, not 1,246.  The script reports both numbers.

Usage
    python hariboss_random_10fd_benchmark.py                    # all ten folds
    python hariboss_random_10fd_benchmark.py --folds 1 2        # smoke test
    python hariboss_random_10fd_benchmark.py --device cpu

Runtime: a few minutes per fold on a CPU core, far less on a GPU.
"""
import argparse
import os
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
    raise SystemExit("could not locate the SMARTBind repository root from " + start)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = find_repo_root(HERE)
sys.path.insert(0, REPO)

from smartbind import load_smartbind_models                       
from smartbind.dataloader import RLDataLoader, RLDataset          
from smartbind.model.margin import sigmoid_cosine_distance_test   
from smartbind.device import resolve_device                      
from smartbind.preprocess import build_val_test_set               

DEFAULT_DATA = os.path.join(REPO, "data", "hariboss", "hariboss_10fd_with_decoys.pkl")
TOPK_DECOY = 100
BATCH_SIZE = 24
NUM_WORKERS = 2


def require(path, what):
    if not os.path.exists(path):
        raise SystemExit(
            f"\n{what} was not found at:\n  {path}\n\n"
            "The processed HARIBOSS splits are distributed with the paper; download them\n"
            "into data/hariboss/, or pass --data.\n")
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
                                "ligand_name": smol_name, "rank": rank,
                                "rank_percentile": (n + 1 - rank) / (n + 1),
                                "auroc": float(auroc)})
    del model
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="the HARIBOSS 10-fold random-split pickle")
    parser.add_argument("--weights", default=os.environ.get("SMARTBIND_WEIGHTS")
                        or os.path.join(REPO, "SMARTBind_weight"),
                        help="directory holding fold1.pth .. fold10.pth")
    parser.add_argument("--device", default="auto",
                        help="'cpu', 'gpu', 'cuda:N', or 'auto' (default: GPU if visible)")
    parser.add_argument("--out", default=os.path.join(HERE, "hariboss_random_10fd_per_pair.csv"))
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    require(args.data, "The HARIBOSS 10-fold split")
    if not os.path.isdir(args.weights):
        raise SystemExit(
            f"\nThe ensemble checkpoints were not found at:\n  {args.weights}\n\n"
            "This benchmark needs fold1.pth .. fold10.pth in that directory.\n"
            "Pass --weights, or point SMARTBIND_WEIGHTS at them.\n")

    start = time.time()
    print(f"device={args.device}\ndata    {args.data}\nweights {args.weights}\n")

    rows = []
    for fold in args.folds:
        checkpoint = os.path.join(args.weights, f"fold{fold}.pth")
        if not os.path.exists(checkpoint):
            raise SystemExit(f"\nmissing checkpoint: {checkpoint}\n")
        rows.extend(run_fold(args.data, checkpoint, fold, args.device))
        done = [r for r in rows if r["fold"] == fold]
        print(f"  fold {fold:>2}: {len(done):>4} pairs, mean rank percentile "
              f"{np.mean([r['rank_percentile'] for r in done]):.4f}, "
              f"mean AUROC {np.mean([r['auroc'] for r in done]):.4f}"
              f"  ({time.time() - start:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(f"\n[OUT] {args.out}  ({len(table):,} pairs)")

    print(f"\n{'fold':>6}{'n':>7}{'rank percentile':>20}{'AUROC':>18}")
    for fold, block in table.groupby("fold"):
        print(f"{fold:>6}{len(block):>7}"
              f"{block.rank_percentile.mean():>13.4f} +/- {block.rank_percentile.std(ddof=1):.4f}"
              f"{block.auroc.mean():>11.4f} +/- {block.auroc.std(ddof=1):.4f}")

    folds = table.groupby("fold").rank_percentile.mean()
    print(f"\n  per-sample  mean +/- s.d. over all {len(table)} pairs   "
          f"{table.rank_percentile.mean():.4f} +/- {table.rank_percentile.std(ddof=0):.4f}")
    if len(folds) > 1:
        print(f"  fold-wise   mean +/- s.d. over the {len(folds)} fold means   "
              f"{folds.mean():.4f} +/- {folds.std(ddof=1):.4f}")
    print(f"  AUROC       mean +/- s.d. over all pairs            "
          f"{table.auroc.mean():.4f} +/- {table.auroc.std(ddof=0):.4f}")
    print(f"  median rank percentile                              "
          f"{table.rank_percentile.median():.4f}")
    print(f"  binder ranked first                                 "
          f"{int((table['rank'] == 0).sum())}/{len(table)} "
          f"({100 * (table['rank'] == 0).mean():.1f}%)")

    if set(args.folds) == set(range(1, 11)):
        print(f"\n  Coverage: {len(table)} of the 1,246 entries in the pickle are validated by "
              f"some fold;\n  the other 490 are in the training half of all ten folds and are "
              f"never scored here.")
    print(f"\ntotal {time.time() - start:.0f}s")
    # No published counterpart exists for these numbers, so there is nothing to assert.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
