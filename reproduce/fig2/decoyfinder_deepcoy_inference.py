#!/usr/bin/env python3
"""SMARTBind on the HARIBOSS 10-fold random split against the DecoyFinder and DeepCoy
decoy libraries — the numbers behind Fig. 2b and Supplementary Table 1.

Everything is derived from two inputs that ship with the paper; there are no
intermediate files to obtain:

  * data/hariboss/hariboss_10fd_with_decoys.pkl   the 10-fold split (Zenodo)
  * data/decoys/                                  the three decoy libraries (in this repo)

plus the published 10-model ensemble in SMARTBind_weight/ (Zenodo).

The evaluation set is built from the pickle alone:

    fold f validation = records whose train_split[f] is False          (756 records)
                        de-duplicated by (rna_chain_sequence, ligand_id)  ->  577 pairs

De-duplication matters: several PDB entries can carry the same RNA sequence with the
same ligand, and SMARTBind scores those identically, so counting them more than once
would silently weight the frequent complexes. Intersecting the 577 pairs with each
decoy library — a ligand is only scorable if the library generated decoys for it —
gives the n quoted in the figure caption:

    DecoyFinder      181 ligands covered  ->  336 pairs
    DeepCoy-DUD-E    254 ligands covered  ->  429 pairs
    DeepCoy-DEKOIS   254 ligands covered  ->  429 pairs

Usage
    python decoyfinder_deepcoy_inference.py                        # all three libraries
    python decoyfinder_deepcoy_inference.py --libraries DecoyFinder
    python decoyfinder_deepcoy_inference.py --folds 1              

Runtime: about 40 min on one CPU core for all three libraries (the DeepCoy libraries
carry ~1,000 decoys per ligand); a few minutes on a GPU.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity


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

from smartbind import load_smartbind_models                 
from smartbind.device import resolve_device               
from smartbind.preprocess import convert_smiles_to_pf2       

LIBRARIES = {
    "DecoyFinder": "data/decoys/decoyfinder_decoys/decoyfinder_decoys.pkl",
    "DeepCoy-DUD-E": "data/decoys/deepcoy_decoys/molecules_ligands_DUDE_smiles.pickle",
    "DeepCoy-DEKOIS": "data/decoys/deepcoy_decoys/molecules_ligands_DEKOIS_smiles.pickle",
}

# Fig. 2b / Supplementary Table 1, for the regression check at the end
PUBLISHED = {
    "DecoyFinder": (336, 0.8951, 0.2426),
    "DeepCoy-DUD-E": (429, 0.9101, 0.2310),
    "DeepCoy-DEKOIS": (429, 0.9093, 0.2305),
}

BATCH = 4096


def hariboss_pickle():
    path = os.environ.get("SMARTBIND_HARIBOSS_PKL") or os.path.join(
        REPO, "data", "hariboss", "hariboss_10fd_with_decoys.pkl")
    if not os.path.exists(path):
        raise SystemExit(
            "\nThe HARIBOSS 10-fold split is required but was not found at:\n  " + path +
            "\n\nDownload it from the Zenodo record linked in the repository README to\n"
            "data/hariboss/, or point SMARTBIND_HARIBOSS_PKL at it.\n")
    return path


def weights_dir():
    path = os.environ.get("SMARTBIND_WEIGHTS") or os.path.join(REPO, "SMARTBind_weight")
    missing = [f"fold{i}.pth" for i in range(1, 11)
               if not os.path.exists(os.path.join(path, f"fold{i}.pth"))]
    if missing:
        raise SystemExit(
            "\nThe 10-model ensemble is required but " + path + " is missing "
            f"{len(missing)} checkpoint(s), e.g. {missing[0]}.\n\n"
            "Download it from the Zenodo record linked in the repository README, or point\n"
            "SMARTBIND_WEIGHTS at it.\n")
    return path


def validation_pairs(raw_dict, fold):
    """Fold's validation records, de-duplicated by (RNA sequence, ligand). fold is 1-based."""
    seen = {}
    for pdb in raw_dict.values():
        for record in pdb.values():
            if record["train_split"][fold - 1]:          # True -> training
                continue
            seen.setdefault((record["rna_chain_sequence"], record["ligand_id"]), record)
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--libraries", nargs="+", default=list(LIBRARIES),
                        choices=list(LIBRARIES))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--device", default="auto",
                        help="Device: 'cpu', 'gpu' (alias for 'cuda'), 'cuda:N', or 'auto' "
                             "(default: a GPU when one is visible, otherwise CPU).")
    parser.add_argument("--out", default=os.path.join(HERE, "fig2b_per_pair.csv"))
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    start = time.time()
    weights = weights_dir()
    raw = pickle.load(open(hariboss_pickle(), "rb"))
    print(f"device={args.device} | {sum(len(v) for v in raw.values())} records in the split")

    fp_cache = {}

    def fingerprint(smiles):
        if smiles not in fp_cache:
            fp_cache[smiles] = convert_smiles_to_pf2(smiles)
        return fp_cache[smiles]

    rows = []
    for library in args.libraries:
        decoys = pickle.load(open(os.path.join(REPO, LIBRARIES[library]), "rb"))
        print(f"\n=== {library}: library covers {len(decoys)} ligands ===", flush=True)
        for fold in args.folds:
            pairs = [r for r in validation_pairs(raw, fold) if r["ligand_id"] in decoys]
            if not pairs:
                continue
            model = load_smartbind_models(
                model_path=os.path.join(weights, f"fold{fold}.pth"),
                device=args.device, vs_mode=True)
            with torch.no_grad():
                for record in pairs:
                    decoy_smiles = decoys[record["ligand_id"]]
                    fps = ([fingerprint(record["downloaded_ligand_smiles"])]
                           + [fingerprint(s) for s in decoy_smiles])
                    chunks = [model.inference_list_smols(fps[i:i + BATCH])
                              for i in range(0, len(fps), BATCH)]
                    embeddings = torch.cat(chunks, dim=0)
                    rna = model.inference_single_rna(record["rna_chain_sequence"])
                    scores = cosine_similarity(rna, embeddings).tolist()
                    binder, decoy_scores = scores[0], scores[1:]
                    # Same definition as training (smartbind/model/pl_train/contact.py
                    # ::_rank_eval) and as Fig. 2c-e: rank counts the decoys scoring at
                    # least as high as the binder, ties against the binder, and 1.0 means
                    # the binder ranks first.
                    rank = sum(1 for s in decoy_scores if binder <= s)
                    rows.append({
                        "model": "SMARTBind",
                        "decoy_library": library,
                        "fold": fold,
                        "pdb_id": record["pdb_id"],
                        "rna_chain_id": record["rna_chain_id"],
                        "ligand_id": record["ligand_id"],
                        "n_decoys": len(decoy_scores),
                        "rank_percentile": (len(decoy_scores) + 1 - rank) / (len(decoy_scores) + 1),
                    })
            print(f"  fold {fold:>2}: {len(pairs):>3} pairs  ({time.time() - start:.0f}s)",
                  flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(f"\n[OUT] {args.out}  ({len(table):,} pairs)")
    print("      same schema as raw_data/Fig2/Fig2b_rank_percentile_per_pair.csv, so these "
          "rows\n      can replace its SMARTBind rows and figure2b.ipynb will redraw from them.")

    print("\n--- Mean rank percentile ---")
    print(f"{'library':<16}{'n':>6}{'mean':>10}{'s.d.':>10}{'median':>10}"
          f"{'   published (n, mean, s.d.)':>30}")
    complete = set(args.folds) == set(range(1, 11))
    worst = 0.0
    for library, block in table.groupby("decoy_library"):
        v = block.rank_percentile
        n, mean, sd = PUBLISHED[library]
        print(f"{library:<16}{len(v):>6}{v.mean():>10.4f}{v.std(ddof=1):>10.4f}"
              f"{v.median():>10.4f}      {n}, {mean:.4f}, {sd:.4f}")
        if complete:
            worst = max(worst, abs(v.mean() - mean), abs(v.std(ddof=1) - sd))
            if len(v) != n:
                worst = float("inf")
    ok = True
    if complete:
        ok = worst < 5e-4
        print(f"\n  {'PASS' if ok else 'MISMATCH'}: worst deviation from the "
              f"published values = {worst:.2e}")
    else:
        print("\n  (partial run: --folds was not the full 1-10, so no regression check)")
    print(f"total {time.time() - start:.0f}s")
    # non-zero on MISMATCH, so a piped or CI run can tell success from failure
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
