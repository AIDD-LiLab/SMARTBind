#!/usr/bin/env python3
"""10-fold cross-validation benchmark on RNAmigos1 and on HARIBOSS.

Replaces `performance_reproduce_rnamigos1.ipynb`, generalised to either
10-fold dataset. For every fold it loads that fold's checkpoint, scores each
validation RNA-ligand pair against the fold's decoy pool, and reports the rank
percentile and AUROC.

    --dataset rnamigos1   RNAmigos1 10-fold random split (Fig. 2a, Supp. Fig. 17a)
                          weights: the RNAmigos1-trained 10-fold checkpoints
    --dataset hariboss    HARIBOSS 10-fold random split
                          weights: the released 10-model ensemble, SMARTBind_weight/

The HARIBOSS arm is not reported in the manuscript -- Fig. 2b evaluates the same
10-fold split against the DecoyFinder and DeepCoy libraries instead, and Fig. 2c-e
use the 5-fold splits. It is here because it is the natural companion number to
the RNAmigos1 benchmark: the same protocol on the paper's own dataset.

Decoy sets
    --decoy-set pdb           the fold's other validation ligands, minus known
                              binders of that RNA chain (both datasets)
    --decoy-set decoyfinder   per-ligand DecoyFinder decoys from
                              data/rnamigos1_decoyfinder/clean/ (RNAmigos1 only)

Rank percentile follows the convention of the rest of the repository,
`(N + 1 - rank) / (N + 1)` with `rank` counting decoys scoring >= the active, so
1.0 means the active outranked every decoy and 0.5 is chance.

Usage
    python benchmark_10fold.py --dataset rnamigos1
    python benchmark_10fold.py --dataset hariboss --device gpu
    python benchmark_10fold.py --dataset rnamigos1 --decoy-set both --folds 1 2

Runtime: a few minutes per dataset on one L4.

Outputs (results/<dataset>_<decoyset>_benchmark/):
    benchmark_per_pair.csv   one row per validation pair
    benchmark_per_fold.csv   fold means
    benchmark_summary.txt    the pooled and fold-wise summary
"""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score

SHOW_PROGRESS = sys.stdout.isatty()


def _find_repo_root(start):
    """Walk up to the checkout that contains the smartbind package."""
    d = Path(start).resolve()
    for folder in (d, *d.parents):
        if (folder / "smartbind" / "__init__.py").is_file():
            return folder
    raise SystemExit("Run this script from inside a checkout of the SMARTBind repository.")


SMARTBIND_REPO = _find_repo_root(__file__)
sys.path.insert(0, str(SMARTBIND_REPO))

from smartbind import load_smartbind_models                   
from smartbind.dataloader import RLDataLoader, RLDataset       
from smartbind.device import resolve_device                     
from smartbind.model.margin import sigmoid_cosine_distance_test  
from smartbind.preprocess import build_val_test_set, convert_smiles_to_pf2  

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DECOYFINDER_DIR = HERE / "data" / "rnamigos1_decoyfinder" / "clean"

DATASETS = {
    "rnamigos1": {
        "data": SMARTBIND_REPO / "data" / "rnamigos" / "rnamigos_10fd_with_decoys.pkl",
        "weights": HERE / "weights_rnamigos1",
        "decoy_sets": ["pdb", "decoyfinder"],
        "title": "RNAmigos1 10-fold random split",
    },
    "hariboss": {
        "data": SMARTBIND_REPO / "data" / "hariboss" / "hariboss_10fd_with_decoys.pkl",
        "weights": SMARTBIND_REPO / "SMARTBind_weight",
        "decoy_sets": ["pdb"],
        "title": "HARIBOSS 10-fold random split",
    },
}

TOPK_DECOY = 100
BATCH_SIZE = 24
NUM_WORKERS = 2
FOLDS = list(range(1, 11))


def decoyfinder_fps(ligand_name, cache):
    """Per-ligand DecoyFinder decoys as FP2 vectors; empty when the ligand has no file."""
    if ligand_name not in cache:
        path = DECOYFINDER_DIR / f"{ligand_name}.txt"
        fps = []
        if path.exists():
            for line in path.read_text().splitlines():
                smiles = line.strip().split()[0] if line.strip() else ""
                if smiles:
                    fps.append(convert_smiles_to_pf2(smiles))
        cache[ligand_name] = fps
    return cache[ligand_name]


def run_fold(model, data_path, fold_num, decoy_set, decoy_cache):
    """Score every validation pair of one fold."""
    (val_rna_seq_list, _, val_rna_name_list, val_match_smol_name_list,
     val_binding_index_list, val_match_smol_fp_list, *rest) = build_val_test_set(
        fold_num=fold_num - 1, dict_path=str(data_path), topk_decoy=TOPK_DECOY,
        return_rna_smol_map=True)

    dataset = RLDataset(rna_sequences=val_rna_seq_list,
                        rna_sequences_names=val_rna_name_list,
                        match_smols=val_match_smol_fp_list,
                        match_smols_names=val_match_smol_name_list,
                        non_binding_index_list=val_binding_index_list,
                        rna_smol_map=rest[-1], is_val=True)
    dataloader = RLDataLoader(dataset=dataset, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, if_shuffle=False)

    records = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader.dataloader, desc=f"  fold {fold_num}",
                               leave=False, disable=not SHOW_PROGRESS):
            rna_sequences, match_smols, decoy_smols_list, rna_names, smol_names, _ = batch
            for rna_seq, match_smol, pool, rna_name, ligand_name in zip(
                    rna_sequences, match_smols, decoy_smols_list, rna_names, smol_names):
                decoys = (decoyfinder_fps(ligand_name, decoy_cache)
                          if decoy_set == "decoyfinder" else list(pool))
                if len(decoys) == 0:
                    continue
                anchor = model.inference_single_rna(rna_seq[1]).squeeze(1)
                match_tok = model.inference_list_smols([match_smol[1]])[0].unsqueeze(0)
                match_score = sigmoid_cosine_distance_test(anchor, match_tok).item()
                decoy_scores = [
                    sigmoid_cosine_distance_test(anchor, t.unsqueeze(0)).item()
                    for t in model.inference_list_smols(decoys)]

                rank = sum(1 for d in decoy_scores if match_score <= d)
                n = len(decoy_scores)
                records.append({
                    "fold": fold_num,
                    "rna_name": rna_name,
                    "ligand_name": ligand_name,
                    "n_decoys": n,
                    "rank": rank,
                    "rank_percentile": (n + 1 - rank) / (n + 1),
                    "auroc": float(roc_auc_score([1] + [0] * n,
                                                 [match_score] + decoy_scores)),
                })
    return records


def summarise(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), np.median(v), v.std(ddof=1) if v.size > 1 else 0.0


# Published per-sample rank percentile, from raw_data/Fig2/Fig2a_summary.csv (the same
# numbers appear in Supplementary Fig. 17a). The HARIBOSS 10-fold arm is deliberately
# absent: the paper does not report it, so there is nothing to check that run against.
PUBLISHED = {
    ("rnamigos1", "pdb"): (765, 0.8256, 0.2256),
    ("rnamigos1", "decoyfinder"): (717, 0.8795, 0.2040),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="rnamigos1")
    parser.add_argument("--decoy-set", choices=["pdb", "decoyfinder", "both"], default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--weight-dir", default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--device", default="auto",
                        help="'cpu', 'gpu', 'cuda:N' or 'auto' (default)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--tolerance", type=float, default=5e-3,
                        help="allowed absolute deviation from the published "
                             "mean and s.d. (rank-percentile units)")
    args = parser.parse_args()

    verdicts = []
    cfg = DATASETS[args.dataset]
    sets = cfg["decoy_sets"] if args.decoy_set in (None, "both") else [args.decoy_set]
    bad = [d for d in sets if d not in cfg["decoy_sets"]]
    if bad:
        raise SystemExit(f"--decoy-set {bad[0]} is not available for {args.dataset}; "
                         f"this dataset supports {cfg['decoy_sets']}")

    device = resolve_device(args.device)
    data = Path(args.data) if args.data else cfg["data"]
    weights = Path(args.weight_dir) if args.weight_dir else cfg["weights"]
    if not data.exists():
        raise SystemExit(f"\nDataset not found: {data}\nDownload it from the Zenodo record "
                         "linked in the repository README.\n")

    for decoy_set in sets:
        out_dir = Path(args.out_dir) / f"{args.dataset}_{decoy_set}_benchmark"
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print("=" * 78)
        print(f"{cfg['title']} — benchmark")
        print("=" * 78)
        print(f"data       {data}")
        print(f"weights    {weights}")
        print(f"folds      {args.folds}   device {device}")
        print("decoy set  " + ("per-ligand DecoyFinder decoys" if decoy_set == "decoyfinder"
                               else "PDB ligands (the fold's other validation ligands)"))

        records, decoy_cache = [], {}
        for fold_num in args.folds:
            weight_path = weights / f"fold{fold_num}.pth"
            if not weight_path.exists():
                raise SystemExit(f"\nMissing checkpoint: {weight_path}\n")
            model = load_smartbind_models(model_path=str(weight_path), device=device,
                                          vs_mode=True)
            model.eval()
            fold_records = run_fold(model, data, fold_num, decoy_set, decoy_cache)
            records.extend(fold_records)
            rp = [r["rank_percentile"] for r in fold_records]
            au = [r["auroc"] for r in fold_records]
            print(f"  fold {fold_num:>2}: {len(fold_records):>4} pairs   "
                  f"rank pct {np.mean(rp):.4f}   AUROC {np.mean(au):.4f}", flush=True)
            del model

        with open(out_dir / "benchmark_per_pair.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        per_fold = []
        for fold_num in args.folds:
            block = [r for r in records if r["fold"] == fold_num]
            per_fold.append({"fold": fold_num, "n_pairs": len(block),
                             "mean_rank_percentile": np.mean([r["rank_percentile"] for r in block]),
                             "mean_auroc": np.mean([r["auroc"] for r in block])})
        with open(out_dir / "benchmark_per_fold.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_fold[0]))
            writer.writeheader()
            writer.writerows(per_fold)

        rp_mean, rp_med, rp_sd = summarise([r["rank_percentile"] for r in records])
        au_mean, au_med, au_sd = summarise([r["auroc"] for r in records])
        fold_rp = [f["mean_rank_percentile"] for f in per_fold]
        fold_au = [f["mean_auroc"] for f in per_fold]
        lines = [
            f"{cfg['title']} — decoy set: {decoy_set}",
            f"  pairs {len(records):,} over {len(args.folds)} folds, "
            f"mean {np.mean([r['n_decoys'] for r in records]):.1f} decoys/pair",
            "",
            "  per-pair (pooled over every validation pair)",
            f"    rank percentile  mean {rp_mean:.4f}  median {rp_med:.4f}  sd {rp_sd:.4f}",
            f"    AUROC            mean {au_mean:.4f}  median {au_med:.4f}  sd {au_sd:.4f}",
            "",
            "  fold-wise (each fold's mean, then across folds)",
            f"    rank percentile  {np.mean(fold_rp):.4f} +/- {np.std(fold_rp, ddof=1):.4f}",
            f"    AUROC            {np.mean(fold_au):.4f} +/- {np.std(fold_au, ddof=1):.4f}",
        ]
        (out_dir / "benchmark_summary.txt").write_text("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        print(f"\n[OUT] {out_dir}   {time.time() - started:.0f}s\n")

        # ---- self-check against the published values, where the paper reports them ----
        want = PUBLISHED.get((args.dataset, decoy_set))
        if want is None:
            print("  (no published value for this arm: the HARIBOSS 10-fold benchmark is\n"
                  "   not reported in the paper, so there is nothing to check against)")
            continue
        n_pub, mean_pub, sd_pub = want
        deltas = {"n": abs(len(records) - n_pub),
                  "mean": abs(rp_mean - mean_pub),
                  "sd": abs(rp_sd - sd_pub)}
        ok = deltas["n"] == 0 and deltas["mean"] <= args.tolerance and deltas["sd"] <= args.tolerance
        print(f"  published  n={n_pub}  rank percentile {mean_pub:.4f} +/- {sd_pub:.4f}")
        print(f"  recomputed n={len(records)}  rank percentile {rp_mean:.4f} +/- {rp_sd:.4f}")
        print(f"  {'PASS' if ok else 'MISMATCH'}: worst deviation "
              f"{max(deltas['mean'], deltas['sd']):.2e} (tolerance {args.tolerance:g})")
        if not ok:
            verdicts.append(f"{args.dataset}/{decoy_set}: n delta {deltas['n']}, "
                            f"mean delta {deltas['mean']:.2e}, sd delta {deltas['sd']:.2e}")

    if verdicts:
        print(f"\nMISMATCH -- {len(verdicts)} arm(s) differ from the published values:")
        for v in verdicts:
            print(f"  {v}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
