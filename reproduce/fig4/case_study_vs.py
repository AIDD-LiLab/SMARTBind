#!/usr/bin/env python3
"""Reproduce the SMARTBind cells of Fig. 4c and Fig. 4e.

Both panels are tables of the inverse rank percentile of a known binder among a
1,000,000-compound Chemspace background, for six methods.  This script reproduces the
SMARTBind row only; the baseline rows were produced by those methods' own pipelines and
are not part of this repository.

  Fig. 4c   three r(CUG)-repeat complexes and one pre-miR-372 fragment
  Fig. 4e   the same riboswitch aptamer in its apo (5E54) and holo (4TZX) forms

SMARTBind reads sequence, not structure, so 5E54 and 4TZX are one and the same input and
one run fills both cells of Fig. 4e - which is the point that panel makes.

Inputs
    ../../data/time_dependent_set/case_study_test_set.csv   the five RNA-ligand pairs
    Chemspace_Screening_Compounds_SMILES_sampled1M.smi      background library (Zenodo)
    ../../SMARTBind_weight/                                 the ten checkpoints

Usage
    python case_study_vs.py                     # GPU if one is visible, else CPU
    python case_study_vs.py --device cpu
    python case_study_vs.py --library /path/to/library.smi --weights /path/to/weights
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from torch.nn.functional import cosine_similarity


def _repo_root():
    for folder in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (folder / "smartbind" / "__init__.py").is_file():
            return folder
    raise SystemExit("Run this script from inside a checkout of the SMARTBind repository.")


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root()))

from smartbind import load_smartbind_models, logger          # noqa: E402
from smartbind.preprocess import convert_smiles_to_pf2       # noqa: E402

# The published SMARTBind cells, as printed in Fig. 4c and Fig. 4e.
PUBLISHED = {
    "9cpd":       ("Fig. 4c", "9CPD / r(CUG) repeats",      0.1896),
    "9cpi":       ("Fig. 4c", "9CPI / r(CUG) repeats",      1.4906),
    "9cpg":       ("Fig. 4c", "9CPG / r(CUG) repeats",     38.4217),
    "premir372":  ("Fig. 4c", "pre-miR-372 / Fragment-3",   3.1381),
    "4tzx":       ("Fig. 4e", "4TZX holo, and 5E54 apo",    0.0008),
}
KEY = {"9CPD": "9cpd", "9CPI": "9cpi", "9CPG": "9cpg", "4TZX": "4tzx",
       "pre-mir-372": "premir372"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(_repo_root() / "data" / "time_dependent_set"
                                          / "case_study_test_set.csv"))
    ap.add_argument("--library", default=str(HERE / "Chemspace_Screening_Compounds_SMILES_sampled1M.smi"),
                    help="one SMILES per line; the 1M Chemspace sample from Zenodo")
    ap.add_argument("--weights", default=os.environ.get("SMARTBIND_WEIGHTS")
                    or str(_repo_root() / "SMARTBind_weight"))
    ap.add_argument("--device", default="auto", help="auto | cpu | gpu | cuda:N")
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--out", default=str(HERE / "case_study_ranking_percentile_scores_rerun.txt"))
    ap.add_argument("--tolerance", type=float, default=5e-4,
                    help="allowed absolute difference in percentage points")
    a = ap.parse_args()

    for path, what in ((a.cases, "case table"), (a.library, "background library")):
        if not os.path.exists(path):
            raise SystemExit(
                f"\nMissing {what}: {path}\n\n"
                "The 1M Chemspace sample and the model weights are distributed through the\n"
                "Zenodo record linked in the repository README; unpack them as described there,\n"
                "or pass --library / --weights.\n")
    if not os.path.isdir(a.weights):
        raise SystemExit(f"\nMissing weights directory: {a.weights}\n")

    cases = pd.read_csv(a.cases)
    logger.info(f"{len(cases)} case-study RNA-ligand pairs from {a.cases}")

    with open(a.library) as f:
        background = [line.strip() for line in f if line.strip()]
    logger.info(f"{len(background):,} background compounds from {a.library}")
    background_fp2 = [convert_smiles_to_pf2(s) for s in background]

    models = load_smartbind_models(model_path=a.weights, device=a.device, vs_mode=True)
    logger.info(f"{len(models)} checkpoints: {[m.checkpoint_name for m in models]}")

    rows = []
    with open(a.out, "w") as log:
        for _, case in cases.iterrows():
            pdb_id, ligand_id = str(case["PDB ID"]), str(case["Ligand ID"])
            key = KEY.get(pdb_id, pdb_id.lower())
            t0 = time.time()
            logger.info(f"Virtual screening for {pdb_id} ({ligand_id})")

            # the true binder occupies index 0, the background follows
            candidates = [convert_smiles_to_pf2(case["SMILES"])] + background_fp2

            per_model = {}
            for model in models:
                rna_embed = model.inference_single_rna(case["RNA sequence"])
                scores = []
                for start in range(0, len(candidates), a.batch_size):
                    chunk = candidates[start:start + a.batch_size]
                    scores.extend(cosine_similarity(rna_embed,
                                                    model.inference_list_smols(chunk)).tolist())
                per_model[model.checkpoint_name] = scores

            averaged = pd.DataFrame(per_model).mean(axis=1)
            binder_score = averaged.iloc[0]
            # rank counts candidates scoring at least as high as the binder, binder included
            rank = int((averaged >= binder_score).sum())
            pct = rank / len(averaged) * 100

            log.write(f"For {key}, the rank is {rank} out of {len(averaged)}\n")
            log.write(f"Rank percentage: {pct}%\n\n")
            log.flush()

            panel, label, published = PUBLISHED.get(key, ("", "", float("nan")))
            rows.append({"key": key, "panel": panel, "case": label,
                         "rank": rank, "n_candidates": len(averaged),
                         "inverse_rank_percentile": round(pct, 4),
                         "published": published,
                         "abs_diff": abs(round(pct, 4) - published) if published == published else float("nan")})
            logger.info(f"  rank {rank:,} / {len(averaged):,}  =  {pct:.4f}%   "
                        f"(published {published}%)   [{time.time() - t0:.0f}s]")

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    print(f"\nPer-case log written to {a.out}")

    checked = table.dropna(subset=["published"])
    worst = checked.abs_diff.max()
    bad = checked[checked.abs_diff > a.tolerance]
    if len(bad):
        print(f"\nMISMATCH: {len(bad)} of {len(checked)} cells differ from the published "
              f"value by more than {a.tolerance} percentage points (worst {worst:.2e}):")
        print(bad.to_string(index=False))
        return 1
    print(f"\nPASS: all {len(checked)} SMARTBind cells of Fig. 4c and Fig. 4e reproduce "
          f"within {a.tolerance} percentage points (worst {worst:.2e}).")
    print("Fig. 4e's two columns, 5E54 (apo) and 4TZX (holo), are the same sequence and "
          "therefore the same value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
