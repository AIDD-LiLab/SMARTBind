"""Ligand overlap: ROBIN molecules vs everything SMARTBind was trained on.

Four sets are compared, all by **exact canonical SMILES**. No fingerprint, no
Tanimoto - two molecules count as the same only if RDKit canonicalises them to the
identical string.

    ROBIN binders      1,908 distinct ligands from Robin_random.csv, every one of
                       which binds at least one of the 26 official targets
    ROBIN SMM decoys   22,410 deduplicated non-binders from the official release
    HARIBOSS binders   the ligand of each training complex
    HARIBOSS decoys    the 200 augmented decoys attached to each training complex

Canonicalisation is what makes this comparison meaningful: the two sides write SMILES
in different conventions (RCSB records carry no stereo annotation, the training file
does), so a raw string comparison reports far fewer matches than there are molecules
in common.

    python ligand_overlap.py
"""
import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# --- repo-root resolution -------------------------------------------------------
# Walk up from this file until the SMARTBind package is found, so the script keeps
# working no matter how deep it sits in the tree or where it is invoked from.
def find_repo_root(start):
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "smartbind", "__init__.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("could not locate the SMARTBind repository root from " + start)


def hariboss_pickle(repo_root):
    """Path to the HARIBOSS training pickle, with a helpful error if it is absent."""
    path = os.environ.get("SMARTBIND_HARIBOSS_PKL") or os.path.join(
        repo_root, "data", "hariboss", "hariboss_10fd_with_decoys.pkl")
    if not os.path.exists(path):
        raise SystemExit(
            "\nThe HARIBOSS training set is required for this analysis but was not found at:\n"
            "  " + path + "\n\n"
            "It is distributed with the processed datasets (see the Zenodo record linked from\n"
            "the repository README), not stored in git. Download it to data/hariboss/ or point\n"
            "SMARTBIND_HARIBOSS_PKL at it:\n"
            "  export SMARTBIND_HARIBOSS_PKL=/path/to/hariboss_10fd_with_decoys.pkl\n")
    return path
# --------------------------------------------------------------------------------


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = find_repo_root(HERE)
ROBIN_DATA = os.path.join(HERE, "..", "robin_official")
TRAINING_PKL = hariboss_pickle(REPO)


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=True)


def canonicalise(smiles_iterable, label):
    """{key -> one original SMILES}.

    The key is the canonical SMILES. A molecule RDKit cannot parse keeps its original
    string as the key instead of being dropped, so nothing leaves the comparison - but
    such an entry can only match a byte-identical string on the other side, not a
    differently written form of the same molecule.
    """
    out, fallback = {}, 0
    for smiles in smiles_iterable:
        c = canonical(smiles)
        if c is None:
            c, _ = smiles, fallback
            fallback += 1
        out.setdefault(c, smiles)
    note = f"   ({fallback} unparseable, kept as the raw string)" if fallback else ""
    print(f"  {label:22s} {len(out):7,d} unique{note}")
    return out


print("--- Loading ---")
robin = pd.read_csv(os.path.join(ROBIN_DATA, "Robin_random.csv"))
robin_binders = sorted(robin["ligand"].unique())
smm_decoys = sorted({l.strip() for l in
                     open(os.path.join(ROBIN_DATA, "robin_smm_nonbinders.smi"))
                     if l.strip()})

with open(TRAINING_PKL, "rb") as f:
    training = pickle.load(f)
train_binders, train_decoys = set(), set()
for pdb in training.values():
    for entry in pdb.values():
        train_binders.add(entry["downloaded_ligand_smiles"])
        train_decoys.update(np.asarray(entry["decoy_smiles"]).tolist())

print(f"ROBIN binders {len(robin_binders):,} | SMM decoys {len(smm_decoys):,} | "
      f"training binders {len(train_binders):,} | training decoys {len(train_decoys):,}")

print("\n--- Canonicalising ---")
SETS = {
    "ROBIN binder": canonicalise(robin_binders, "ROBIN binder"),
    "ROBIN SMM decoy": canonicalise(smm_decoys, "ROBIN SMM decoy"),
    "HARIBOSS binder": canonicalise(sorted(train_binders), "HARIBOSS binder"),
    "HARIBOSS decoy": canonicalise(sorted(train_decoys), "HARIBOSS decoy"),
}

# ligand_id and the complexes each training binder belongs to, for the report
binder_meta = {}
for pdb in training.values():
    for key, entry in pdb.items():
        raw = entry["downloaded_ligand_smiles"]
        c = canonical(raw) or raw
        binder_meta.setdefault(c, {"ligand_id": set(), "complexes": 0})
        binder_meta[c]["ligand_id"].add(entry["ligand_id"])
        binder_meta[c]["complexes"] += 1

print("\n--- Overlap (exact canonical SMILES) ---")
rows, detail = [], []
for left in ("ROBIN binder", "ROBIN SMM decoy"):
    for right in ("HARIBOSS binder", "HARIBOSS decoy"):
        shared = sorted(set(SETS[left]) & set(SETS[right]))
        rows.append({"robin_set": left, "training_set": right,
                     "robin_n": len(SETS[left]), "training_n": len(SETS[right]),
                     "n_shared": len(shared),
                     "pct_of_robin_set": round(100 * len(shared) / len(SETS[left]), 4),
                     "pct_of_training_set": round(100 * len(shared) / len(SETS[right]), 4)})
        print(f"  {left:16s} vs {right:16s} : {len(shared):5,d} / {len(SETS[left]):,}")
        for c in shared:
            record = {"robin_set": left, "training_set": right, "match_key": c,
                      "robin_smiles": SETS[left][c], "training_smiles": SETS[right][c]}
            if right == "HARIBOSS binder" and c in binder_meta:
                record["training_ligand_id"] = ",".join(sorted(binder_meta[c]["ligand_id"]))
                record["training_complexes"] = binder_meta[c]["complexes"]
            detail.append(record)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(HERE, "ligand_overlap_summary.csv"), index=False)
matches = pd.DataFrame(detail)
matches.to_csv(os.path.join(HERE, "ligand_overlap_matches.csv"), index=False)

print("\n[OUT] ligand_overlap_summary.csv, ligand_overlap_matches.csv")
print("=" * 70)
print(summary.to_string(index=False))
if len(matches):
    print("\nmatched molecules:")
    show = [c for c in ("robin_set", "training_set", "training_ligand_id",
                        "training_complexes", "match_key") if c in matches.columns]
    print(matches[show].to_string(index=False))
