"""Re-screen SMARTBind on the leakage-filtered ROBIN SMM benchmark.

Both the 21 targets and the 22,368 decoys in this folder have already had every
sequence- and ligand-level overlap with the training set removed, so this is a full
rescoring rather than a filter applied to old scores: removing decoys changes every
binder's rank percentile, which cannot be recomputed from stored per-binder values.

Protocol is the published one (`1_smartbind_robin_screening.ipynb`): the 10-model
ensemble, cosine similarity between the RNA embedding and the ligand embedding,
averaged over the ten models.

    rank% = (# decoys scoring strictly higher) / n_decoys * 100      lower is better

Writes `smartbind_smm_filtered_per_binder.csv` in the same shape as the per-binder
files this benchmark already ships, so downstream EF code needs no change.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(HERE, "cache")
BATCH = 512


def find_repo_root(start):
    """Walk up until the SMARTBind package is found."""
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "smartbind", "__init__.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("could not locate the SMARTBind repository root from " + start)


REPO = find_repo_root(HERE)
WEIGHTS = os.environ.get("SMARTBIND_WEIGHTS") or os.path.join(REPO, "SMARTBind_weight")
if not os.path.isdir(WEIGHTS) or not any(f.endswith(".pth") for f in os.listdir(WEIGHTS)):
    raise SystemExit(
        "\nNo SMARTBind checkpoints found at:\n  " + WEIGHTS + "\n\n"
        "Download the 10-model ensemble from the Zenodo record linked in the repository\n"
        "README and unpack it to SMARTBind_weight/, or point SMARTBIND_WEIGHTS at it:\n"
        "  export SMARTBIND_WEIGHTS=/path/to/SMARTBind_weight\n")

os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, REPO)
from smartbind.device import resolve_device
from smartbind import load_smartbind_models
from smartbind.preprocess import convert_smiles_to_pf2


def read_fasta(path):
    out, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            out[name] = ""
        elif line:
            out[name] += line
    return out


def fp2_matrix(smiles_list, cache):
    if os.path.exists(cache):
        matrix = np.load(cache)
        if len(matrix) == len(smiles_list):
            print(f"  loaded {os.path.basename(cache)} {matrix.shape}", flush=True)
            return matrix
    print(f"  featurising {len(smiles_list):,} SMILES", flush=True)
    matrix = np.zeros((len(smiles_list), 1024), dtype=np.uint8)
    for i, smiles in enumerate(smiles_list):
        matrix[i] = convert_smiles_to_pf2(smiles)
        if (i + 1) % 5000 == 0:
            print(f"    {i + 1:,}/{len(smiles_list):,}", flush=True)
    np.save(cache, matrix)
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto",
                        help="Device: 'cpu', 'gpu' (alias for 'cuda'), 'cuda:N', or 'auto' "
                             "(default: a GPU when one is visible, otherwise CPU).")
    args = parser.parse_args()

    start = time.time()
    sequences = read_fasta(os.path.join(DATA, "sequences.fasta"))
    binders = pd.read_csv(os.path.join(DATA, "binders.csv"))
    decoys = [l.strip() for l in open(os.path.join(DATA, "decoys.smi")) if l.strip()]
    ligands = sorted(binders.SMILES.unique())
    print(f"{len(sequences)} targets | {len(binders):,} binder pairs "
          f"({len(ligands):,} unique) | {len(decoys):,} decoys", flush=True)

    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    ligand_fp = fp2_matrix(ligands, os.path.join(CACHE, "fp2_binders.npy"))
    decoy_fp = fp2_matrix(decoys, os.path.join(CACHE, "fp2_decoys.npy"))

    models = load_smartbind_models(model_path=WEIGHTS, device=device, vs_mode=True)
    print(f"{len(models)} models loaded", flush=True)

    @torch.no_grad()
    def embed(model, fp):
        chunks = [model.inference_list_smols(fp[i:i + BATCH].astype(np.float32).tolist())
                  for i in range(0, len(fp), BATCH)]
        return torch.cat(chunks, dim=0)

    names = list(sequences)
    sim_ligand = np.zeros((len(names), len(ligands)), dtype=np.float64)
    sim_decoy = np.zeros((len(names), len(decoys)), dtype=np.float64)
    with torch.no_grad():
        for i, model in enumerate(models):
            el, ed = embed(model, ligand_fp), embed(model, decoy_fp)
            for ti, target in enumerate(names):
                rna = model.inference_single_rna(sequences[target])
                sim_ligand[ti] += cosine_similarity(rna, el).cpu().numpy()
                sim_decoy[ti] += cosine_similarity(rna, ed).cpu().numpy()
            print(f"  model {i + 1}/{len(models)}  {time.time() - start:.0f}s", flush=True)
    sim_ligand /= len(models)
    sim_decoy /= len(models)

    index = {s: i for i, s in enumerate(ligands)}
    frames = []
    for ti, target in enumerate(names):
        block = binders[binders.target == target]
        scores = np.array([sim_ligand[ti, index[s]] for s in block.SMILES])
        ordered = np.sort(sim_decoy[ti])
        above = len(ordered) - np.searchsorted(ordered, scores, side="right")
        frames.append(pd.DataFrame({"target": target, "SMILES": block.SMILES.to_numpy(),
                                    "score": scores,
                                    "rank_percentile": above / len(ordered) * 100.0}))
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(HERE, "smartbind_smm_filtered_per_binder.csv")
    out.to_csv(path, index=False)
    print(f"\n[OUT] {path}  ({len(out):,} binders)")
    print(f"mean rank percentile {out.rank_percentile.mean():.3f}")
    for t in (0.5, 1, 3, 5):
        ef = out.groupby("target").rank_percentile.apply(
            lambda r: (r <= t).mean() / (t / 100)).mean()
        print(f"  EF@{t}% = {ef:.3f}")
    print(f"total {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
