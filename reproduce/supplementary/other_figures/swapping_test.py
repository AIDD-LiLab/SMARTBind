#!/usr/bin/env python3
"""RNA- and ligand-swapping tests for Supplementary Fig. 5.

One script for all four published tests; it replaces the four notebooks that used
to live here (`{rna,ligand}_swapping_test_{hariboss,rnamigos1}.ipynb`), which
duplicated the same protocol with different inputs.

    --dataset hariboss    HARIBOSS 5-fold sequence-based split, 5 folds
                          (Supplementary Fig. 5a-b)
    --dataset rnamigos1   RNAmigos1 10-fold random split, 10 folds
                          (Supplementary Fig. 5c-d)

Both swaps are measured in the same run: the "without swapping" baseline is scored
once and both are compared against it, which is also what the published table
shows: the panels quote the same baseline.

What a swap is
    RNA swap     each positive test pair keeps its ligand and its decoys, but
                 its RNA is replaced by another test RNA
    ligand swap  each pair keeps its RNA and its decoys, but its true ligand is
                 replaced by another test ligand

Both are a global derangement over the fold's distinct RNAs / distinct ligands,
i.e. a permutation with no fixed point, so nothing is left paired with itself,
redrawn K times. The decoy pool is never touched, so the label structure of every
ranking problem is unchanged; only the positive is made wrong. A model that had
learned real RNA-ligand complementarity should collapse; one that had learned
ligand priors alone would not move.

Decoy sets
    --decoy-set pdb           the fold's other validation ligands, minus known
                              binders of that RNA chain (both datasets)
    --decoy-set decoyfinder   per-ligand DecoyFinder decoys read from
                              data/rnamigos1_decoyfinder/clean/<LIGAND>.txt
                              (RNAmigos1 only)
    --decoy-set both          run both and report them side by side

Metrics, each computed per fold and then averaged unweighted over the folds:

    Global AUROC     one AUROC over every (label, score) in the fold, pooled
    Target AUROC     AUROC within each pair (1 positive vs its decoys), averaged
    Rank percentile  (N + 1 - rank) / (N + 1), rank counting decoys scoring
                     >= the active, averaged over the fold's pairs

The original notebooks inserted an extra step into the last two: pairs sharing an
RNA chain were averaged together first, giving each chain equal weight rather
than each pair. Those are still computed and land in the per-fold CSV as the
"RNA-grouped" rows, because the published table used them, but they are worth
~0.003 and do not reconcile with the official inference results, so the summary
table reports the per-pair convention only.

Usage
    python swapping_test.py --dataset hariboss
    python swapping_test.py --dataset rnamigos1 --decoy-set both
    python swapping_test.py --dataset rnamigos1 --folds 1 2 --repetitions 2

Runtime: roughly 30 min per dataset on one L4, a few hours on a CPU core.

Outputs (results/<dataset>_<decoyset>/):
    swapping_test_summary.csv     the table
    swapping_test_per_fold.csv    every fold x repetition cell behind it
    swapping_test_raw.pkl         per-pair scores for every condition and repetition,
                                  so a different aggregation needs no rerun
"""

import argparse
import csv
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import tqdm

# tqdm writes \r-separated frames, which turn a redirected log into one
# unreadable mega-line. Show the bars only when a terminal is attached.
SHOW_PROGRESS = sys.stdout.isatty()
from sklearn.metrics import roc_auc_score

def _find_repo_root(start):
    """Walk up to the checkout that contains the smartbind package."""
    d = Path(start).resolve()
    for folder in (d, *d.parents):
        if (folder / "smartbind" / "__init__.py").is_file():
            return folder
    raise SystemExit("Run this script from inside a checkout of the SMARTBind repository.")


SMARTBIND_REPO = _find_repo_root(__file__)
sys.path.insert(0, str(SMARTBIND_REPO))

from smartbind import load_smartbind_models                      # noqa: E402
from smartbind.dataloader import RLDataLoader, RLDataset         # noqa: E402
from smartbind.model.margin import sigmoid_cosine_distance_test  # noqa: E402
from smartbind.device import resolve_device                      # noqa: E402
from smartbind.preprocess import build_val_test_set, convert_smiles_to_pf2  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# One entry per published test: where its data, weights and folds come from.
DATASETS = {
    "hariboss": {
        "data": SMARTBIND_REPO / "data" / "hariboss"
                / "hariboss_5fd_cov08_clustered_seqid_0.3_clean.pkl",
        "weights": HERE / "weights_hariboss_sequenceSplit",
        "folds": [1, 2, 3, 4, 5],
        "decoy_sets": ["pdb"],
        "title": "HARIBOSS (5-fold sequence-based split)",
    },
    "rnamigos1": {
        "data": SMARTBIND_REPO / "data" / "rnamigos" / "rnamigos_10fd_with_decoys.pkl",
        "weights": HERE / "weights_rnamigos1",
        "folds": list(range(1, 11)),
        "decoy_sets": ["pdb", "decoyfinder"],
        "title": "RNAmigos1 (10-fold random split)",
    },
}
DECOYFINDER_DIR = HERE / "data" / "rnamigos1_decoyfinder" / "clean"

TOPK_DECOY = 100
BATCH_SIZE = 24
NUM_WORKERS = 2
K_SWAP_REPETITIONS = 5
RANDOM_SEED = 42


# ------------------------------------------------------------------ helpers
# Copied unchanged from the notebooks so the numbers stay comparable.

def normalize_scores(scores):
    """Min-max normalize scores to [0, 1]."""
    scores = np.array(scores)
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return scores


def compute_scores_for_pair(model, rna_emb, match_smol_fp, decoy_fps):
    """Similarity of one RNA embedding to the true ligand and to every decoy."""
    anchor = rna_emb.squeeze(1)
    match_tok = model.inference_list_smols([match_smol_fp])[0].unsqueeze(0)
    match_score = sigmoid_cosine_distance_test(anchor, match_tok).item()
    decoy_scores = []
    for dtok in model.inference_list_smols(decoy_fps):
        decoy_scores.append(sigmoid_cosine_distance_test(anchor, dtok.unsqueeze(0)).item())
    return match_score, decoy_scores


def compute_rank_percentile(match_score, decoy_scores):
    rank = sum(1 for ds in decoy_scores if match_score <= ds)
    return (len(decoy_scores) + 1 - rank) / (len(decoy_scores) + 1)


def compute_pair_auroc(match_score, decoy_scores):
    scores = [match_score] + decoy_scores
    labels = [1] + [0] * len(decoy_scores)
    return roc_auc_score(labels, normalize_scores(scores))


def pack(result):
    """Per-pair scores in a compact, re-analysable form."""
    return {
        "match_scores": np.asarray([m for m, _ in result["scores"]], dtype=np.float32),
        "decoy_scores": [np.asarray(d, dtype=np.float32) for _, d in result["scores"]],
        "rank_percentiles": np.asarray(result["rank_percentiles"], dtype=np.float32),
        "pair_aurocs": np.asarray(result["pair_aurocs"], dtype=np.float32),
    }


def derangement(names, rng):
    """A permutation of `names` with no fixed point."""
    permuted = list(names)
    rng.shuffle(permuted)
    if len(names) > 1:
        while any(permuted[i] == names[i] for i in range(len(names))):
            rng.shuffle(permuted)
    return dict(zip(names, permuted))


# ------------------------------------------------------------------ per fold

def decoyfinder_fps(ligand_name):
    """Per-ligand DecoyFinder decoys, as FP2 vectors.

    One file per ligand id, one SMILES per line. A ligand with no file has no
    DecoyFinder decoys and its pair is dropped from that test -- the notebooks did
    the same, which is why the DecoyFinder n is smaller than the PDB n.
    """
    path = DECOYFINDER_DIR / f"{ligand_name}.txt"
    if not path.exists():
        return []
    fps = []
    for line in path.read_text().splitlines():
        smiles = line.strip().split()[0] if line.strip() else ""
        if smiles:
            fps.append(convert_smiles_to_pf2(smiles))
    return fps


def load_fold(model, data_path, fold_num, decoy_set):
    """Every validation pair of one fold, with its RNA embedding cached."""
    (val_rna_seq_list, _, val_rna_name_list, val_match_smol_name_list,
     val_binding_index_list, val_match_smol_fp_list, *rest) = build_val_test_set(
        fold_num=fold_num - 1, dict_path=str(data_path), topk_decoy=TOPK_DECOY,
        return_rna_smol_map=True)

    dataset = RLDataset(rna_sequences=val_rna_seq_list,
                        rna_sequences_names=val_rna_name_list,
                        match_smols=val_match_smol_fp_list,
                        match_smols_names=val_match_smol_name_list,
                        non_binding_index_list=val_binding_index_list,
                        rna_smol_map=rest[-1],
                        is_val=True)
    dataloader = RLDataLoader(dataset=dataset, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, if_shuffle=False)

    entries, rna_emb_cache, decoy_cache = [], {}, {}
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader.dataloader, desc=f"  fold {fold_num} load",
                               leave=False, disable=not SHOW_PROGRESS):
            rna_sequences, match_smols, decoy_smols_list, _, _, _ = batch
            for rna_seq, match_smol, decoy_smols in zip(
                    rna_sequences, match_smols, decoy_smols_list):
                ligand_name = match_smol[0]
                if decoy_set == "decoyfinder":
                    if ligand_name not in decoy_cache:
                        decoy_cache[ligand_name] = decoyfinder_fps(ligand_name)
                    decoys = decoy_cache[ligand_name]
                else:
                    decoys = list(decoy_smols)
                if len(decoys) == 0:
                    continue
                rna_name = rna_seq[0]
                if rna_name not in rna_emb_cache:
                    rna_emb_cache[rna_name] = model.inference_single_rna(rna_seq[1])
                entries.append({
                    "rna_name": rna_name,
                    "match_smol_name": ligand_name,
                    "match_smol_fp": match_smol[1],
                    "decoy_fps": decoys,
                })
    return entries, rna_emb_cache


def score_all(model, entries, rna_emb_cache, pick, desc):
    """Score every entry. `pick(entry)` returns the (rna_embedding, ligand_fp) to use."""
    scores, rank_percentiles, pair_aurocs = [], [], []
    with torch.no_grad():
        for entry in tqdm.tqdm(entries, desc=desc, leave=False,
                                   disable=not SHOW_PROGRESS):
            rna_emb, ligand_fp = pick(entry)
            match_score, decoy_scores = compute_scores_for_pair(
                model, rna_emb, ligand_fp, entry["decoy_fps"])
            scores.append((match_score, decoy_scores))
            rank_percentiles.append(compute_rank_percentile(match_score, decoy_scores))
            pair_aurocs.append(compute_pair_auroc(match_score, decoy_scores))
    return {"scores": scores, "rank_percentiles": rank_percentiles,
            "pair_aurocs": pair_aurocs}


# ------------------------------------------------------------------ metrics

def flat_level(values):
    """Plain mean over the fold's pairs."""
    return float(np.mean(values))


def rna_grouped_level(rna_names, values):
    """Average within each distinct RNA first, then across RNAs.

    This is what the original notebooks did. It gives every RNA chain equal
    weight instead of every pair, so an RNA appearing in several complexes
    counts once. Kept because the published table was produced this way, but
    it is not the primary number here: it cannot be reconciled with
    `inference_results/full_*.pkl`, which averages over pairs.
    """
    by_target = defaultdict(list)
    for name, value in zip(rna_names, values):
        by_target[name].append(value)
    return float(np.mean([np.mean(v) for v in by_target.values()]))


def global_auroc(scores):
    labels, flat = [], []
    for match_score, decoy_scores in scores:
        flat.append(match_score)
        labels.append(1)
        flat.extend(decoy_scores)
        labels.extend([0] * len(decoy_scores))
    return float(roc_auc_score(labels, normalize_scores(flat)))


def fold_metrics(rna_names, result):
    return {
        "global_auroc": global_auroc(result["scores"]),
        "target_auroc": flat_level(result["pair_aurocs"]),
        "rank_percentile": flat_level(result["rank_percentiles"]),
        "target_auroc_rna_grouped": rna_grouped_level(rna_names, result["pair_aurocs"]),
        "rank_percentile_rna_grouped": rna_grouped_level(rna_names, result["rank_percentiles"]),
    }


# Every metric is a per-fold number, averaged unweighted over the 5 folds.
# The summary table reports one convention only -- a per-fold mean over that
# fold's pairs, then an unweighted mean over the 5 folds -- so every row is
# comparable and the "without swapping" column reconciles with
# inference_results/summary.csv. The RNA-grouped variants are still computed
# and land in swapping_test_per_fold.csv, but they stay out of the table.
METRICS = ["global_auroc", "target_auroc", "rank_percentile"]
DETAIL_METRICS = METRICS + ["target_auroc_rna_grouped", "rank_percentile_rna_grouped"]
METRIC_LABEL = {"global_auroc": "AUROC (global)",
                "target_auroc": "AUROC (target)",
                "rank_percentile": "Rank percentile",
                "target_auroc_rna_grouped": "AUROC (target, RNA-grouped)",
                "rank_percentile_rna_grouped": "Rank percentile (RNA-grouped)"}
EVAL_LEVEL = {"global_auroc": "Global",
              "target_auroc": "Target",
              "rank_percentile": "N/A",
              "target_auroc_rna_grouped": "Target",
              "rank_percentile_rna_grouped": "N/A"}


def run(cfg, args, decoy_set, device):
    """One dataset x one decoy set: both swaps, all folds, all repetitions."""
    out_dir = Path(args.out_dir) / f"{args.dataset}_{decoy_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    K = args.repetitions
    folds = args.folds or cfg["folds"]
    weight_dir = Path(args.weight_dir) if args.weight_dir else cfg["weights"]
    data = Path(args.data) if args.data else cfg["data"]

    print("\n" + "=" * 78)
    print(f"RNA- and ligand-swapping tests on {cfg['title']}")
    print("=" * 78)
    print(f"data       {data}")
    print(f"weights    {weight_dir}")
    print(f"folds      {folds}   repetitions {K}   seed {RANDOM_SEED}   device {device}")
    print("decoy set  " + ("per-ligand DecoyFinder decoys" if decoy_set == "decoyfinder"
                           else "PDB ligands (the fold's other validation ligands)"))
    args = argparse.Namespace(**{**vars(args), "folds": folds,
                                 "weight_dir": str(weight_dir), "data": str(data),
                                 "out_dir": out_dir})

    started = time.time()
    # metric -> list over folds of {original, rna_swap[k], ligand_swap[k]}
    per_fold = []
    raw = {}

    for fold_num in args.folds:
        weight_path = Path(args.weight_dir) / f"fold{fold_num}.pth"
        if not weight_path.exists():
            raise SystemExit(
                f"\nMissing checkpoint: {weight_path}\n\n"
                f"This test needs the {args.dataset} swap-test checkpoints in that "
                f"directory, one per requested fold: "
                f"{', '.join(f'fold{n}.pth' for n in args.folds)}.\n"
                "Point --weight-dir at them if they live elsewhere.\n")

        print(f"\n[fold {fold_num}] {weight_path.name}", flush=True)
        model = load_smartbind_models(model_path=str(weight_path), device=device,
                                      vs_mode=True)
        model.eval()

        entries, rna_emb_cache = load_fold(model, args.data, fold_num, decoy_set)
        rna_names = [e["rna_name"] for e in entries]
        # sorted(), not list(set(...)): a set of strings iterates in an order that depends on
        # PYTHONHASHSEED, which is randomised per process, so the seeded RandomState below
        # would deal a different derangement on every run. Sorting pins the input order and
        # makes RANDOM_SEED actually determine the permutation.
        unique_rnas = sorted(set(rna_names))
        unique_ligands = sorted(set(e["match_smol_name"] for e in entries))
        ligand_fp = {}
        for e in entries:
            ligand_fp.setdefault(e["match_smol_name"], e["match_smol_fp"])
        print(f"  {len(entries)} pairs, {len(unique_rnas)} distinct RNAs, "
              f"{len(unique_ligands)} distinct ligands, "
              f"{np.mean([len(e['decoy_fps']) for e in entries]):.1f} decoys/pair",
              flush=True)

        original = score_all(model, entries, rna_emb_cache,
                             lambda e: (rna_emb_cache[e["rna_name"]], e["match_smol_fp"]),
                             "  original")
        row = {"fold": fold_num, "n_pairs": len(entries),
               "original": fold_metrics(rna_names, original)}
        print(f"  original      global {row['original']['global_auroc']:.4f}  "
              f"target {row['original']['target_auroc']:.4f}  "
              f"rank pct {row['original']['rank_percentile']:.4f}", flush=True)

        # A fresh RNG per test, so each reproduces its own notebook's permutations.
        swapped_raw = {}
        for test, unique_names in (("rna_swap", unique_rnas),
                                   ("ligand_swap", unique_ligands)):
            rng = np.random.RandomState(RANDOM_SEED + fold_num)
            row[test] = []
            swapped_raw[test] = []
            for k in range(K):
                sigma = derangement(unique_names, rng)
                if test == "rna_swap":
                    def pick(e, sigma=sigma):
                        return rna_emb_cache[sigma[e["rna_name"]]], e["match_smol_fp"]
                else:
                    def pick(e, sigma=sigma):
                        return (rna_emb_cache[e["rna_name"]],
                                ligand_fp[sigma[e["match_smol_name"]]])
                swapped = score_all(model, entries, rna_emb_cache, pick,
                                    f"  {test} k{k + 1}")
                row[test].append(fold_metrics(rna_names, swapped))
                swapped_raw[test].append(swapped)
            print(f"  {test:12s}  global "
                  f"{np.mean([m['global_auroc'] for m in row[test]]):.4f}  "
                  f"target {np.mean([m['target_auroc'] for m in row[test]]):.4f}  "
                  f"rank pct {np.mean([m['rank_percentile'] for m in row[test]]):.4f}",
                  flush=True)

        per_fold.append(row)
        raw[fold_num] = {
            "rna_names": rna_names,
            "ligand_names": [e["match_smol_name"] for e in entries],
            "n_pairs": len(entries),
            "n_decoys": [len(e["decoy_fps"]) for e in entries],
            # Per-pair scores, so a different aggregation never needs a rerun.
            # float32 keeps the pickle small; the metrics are computed above in
            # full precision, these are for re-analysis only.
            "original": pack(original),
            "rna_swap": [pack(x) for x in swapped_raw["rna_swap"]],
            "ligand_swap": [pack(x) for x in swapped_raw["ligand_swap"]],
        }
        del model

    # -------------------------------------------------------------- aggregate
    summary = []
    for test in ("rna_swap", "ligand_swap"):
        for metric in METRICS:
            original = float(np.mean([r["original"][metric] for r in per_fold]))
            # mean across folds for each k -> K numbers -> mean and sd over K
            per_k = [float(np.mean([r[test][k][metric] for r in per_fold]))
                     for k in range(K)]
            gaps = [100.0 * (v - original) / original for v in per_k]
            summary.append({
                "test": "RNA swapping" if test == "rna_swap" else "Ligand swapping",
                "decoy_set": "DecoyFinder" if decoy_set == "decoyfinder" else "PDB",
                "metric": METRIC_LABEL[metric],
                "evaluation_level": EVAL_LEVEL[metric],
                "without_swapping": round(original, 4),
                "target_swapping_mean": round(float(np.mean(per_k)), 4),
                "target_swapping_std": round(float(np.std(per_k, ddof=1)), 4),
                "performance_gap_pct_mean": round(float(np.mean(gaps)), 2),
                "performance_gap_pct_std": round(float(np.std(gaps, ddof=1)), 2),
                "n_folds": len(per_fold),
                "n_repetitions": K,
            })

    fields = list(summary[0])
    with open(args.out_dir / "swapping_test_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    detail = []
    for row in per_fold:
        for metric in DETAIL_METRICS:
            detail.append({"fold": row["fold"], "n_pairs": row["n_pairs"],
                           "metric": METRIC_LABEL[metric], "condition": "without swapping",
                           "repetition": "", "value": round(row["original"][metric], 6)})
            for test in ("rna_swap", "ligand_swap"):
                for k, m in enumerate(row[test], start=1):
                    detail.append({"fold": row["fold"], "n_pairs": row["n_pairs"],
                                   "metric": METRIC_LABEL[metric], "condition": test,
                                   "repetition": k, "value": round(m[metric], 6)})
    with open(args.out_dir / "swapping_test_per_fold.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)

    with open(args.out_dir / "swapping_test_raw.pkl", "wb") as handle:
        pickle.dump({"config": {"data": str(args.data), "weight_dir": str(args.weight_dir),
                                "folds": args.folds, "repetitions": K,
                                "seed": RANDOM_SEED, "topk_decoy": TOPK_DECOY},
                     "per_fold": per_fold, "summary": summary, "raw": raw}, handle)

    print("\n" + "=" * 96)
    print("SUMMARY  (mean +/- sd over K permutations; each value first averaged across folds)")
    print("=" * 96)
    print(f"{'Test':<17}{'Decoy':<7}{'Metric':<18}{'Level':<9}"
          f"{'Without swap':>13}{'Target swapping':>22}{'Gap (%)':>19}")
    print("-" * 96)
    for row in summary:
        print(f"{row['test']:<17}{row['decoy_set']:<7}{row['metric']:<18}"
              f"{row['evaluation_level']:<9}{row['without_swapping']:>13.4f}"
              f"{row['target_swapping_mean']:>14.4f} +/- {row['target_swapping_std']:.4f}"
              f"{row['performance_gap_pct_mean']:>12.2f} +/- {row['performance_gap_pct_std']:.2f}")
    print(f"\nwrote {args.out_dir}/swapping_test_summary.csv "
          f"(+ _per_fold.csv, _raw.pkl)   {time.time() - started:.0f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="hariboss")
    parser.add_argument("--decoy-set", choices=["pdb", "decoyfinder", "both"],
                        default=None, help="default: every set the dataset supports")
    parser.add_argument("--data", default=None, help="override the dataset pickle")
    parser.add_argument("--weight-dir", default=None, help="override the checkpoint folder")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=K_SWAP_REPETITIONS)
    parser.add_argument("--device", default="auto",
                        help="'cpu', 'gpu', 'cuda:N' or 'auto' (default)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    if args.decoy_set is None:
        decoy_sets = cfg["decoy_sets"]
    elif args.decoy_set == "both":
        decoy_sets = cfg["decoy_sets"]
    else:
        decoy_sets = [args.decoy_set]
    unsupported = [d for d in decoy_sets if d not in cfg["decoy_sets"]]
    if unsupported:
        raise SystemExit(
            f"--decoy-set {unsupported[0]} is not available for {args.dataset}; "
            f"this dataset supports {cfg['decoy_sets']}")

    device = resolve_device(args.device)
    for decoy_set in decoy_sets:
        run(cfg, args, decoy_set, device)


if __name__ == "__main__":
    main()
