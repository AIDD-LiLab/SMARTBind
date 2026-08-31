"""MMseqs2 sequence similarity: the 34 time-dependent test RNAs vs the HARIBOSS training set.

Same protocol and the same criterion as our RNA sequence split, and as the ROBIN leakage
analysis in reproduce/supplementary/robin/leakage/rna_mmseqs/:

    identity >= 0.3  AND  query coverage >= 0.8  AND  target coverage >= 0.8

The search itself is left wide open (`--min-seq-id 0 --cov-mode 0 -c 0 -e 10000`) and
the three thresholds are applied afterwards, so nothing is discarded before it has been
scored. `-k 4` is used in place of the nucleotide default of 15, which cannot seed on
RNAs of 14-84 nt: with the default the search returns a hit for only 2 of the 34
queries and flags none of them.

Needs the `mmseqs` binary on PATH (or $MMSEQS) and the HARIBOSS training set at
data/hariboss/hariboss_10fd_with_decoys.pkl (or $SMARTBIND_HARIBOSS_PKL).

    python mmseq_cal.py

Writes
    query_temporal.fasta                 the 34 test-set RNA chains
    target_training.fasta                the HARIBOSS training chains
    mmseqs_results.csv                   every scored hit
    temporal_vs_training_BEST_HITS.csv   the hits meeting the criterion above
"""
import os
import pickle
import re
import shutil
import subprocess

import pandas as pd


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
TEMPORAL_CSV = os.path.join(HERE, "..", "time_dependent_test_set.csv")
TRAINING_PKL = hariboss_pickle(REPO)
MMSEQS = os.environ.get("MMSEQS", "mmseqs")

K = 4
MIN_SEQ_ID, MIN_COV = 0.3, 0.8

COLUMNS = ["query", "target", "fident", "alnlen", "qlen", "tlen",
           "qcov", "tcov", "evalue", "bits"]


def sanitize_id(s):
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", str(s))


def clean_seq(s):
    return s.strip().upper().replace(" ", "").replace("\n", "").replace("U", "T")


def write_fasta(d, path):
    with open(path, "w") as f:
        for k, v in d.items():
            if v:
                f.write(f">{sanitize_id(k)}\n{v}\n")
    print(f"[FASTA] {os.path.basename(path)}: {sum(1 for v in d.values() if v)} sequences")


print("--- Loading Data ---")
temporal = pd.read_csv(TEMPORAL_CSV)
queries, meta = {}, {}
for _, row in temporal.iterrows():
    qid = sanitize_id(f"{row['PDB ID']}_{row['Chain ID']}")
    queries[qid] = clean_seq(str(row["RNA sequence"]))
    meta[qid] = {"PDB_ID": str(row["PDB ID"]), "Chain_ID": str(row["Chain ID"]),
                 "Ligand_ID": str(row["Ligand ID"])}
print(f"[INFO] time-dependent test chains: {len(queries)}")

with open(TRAINING_PKL, "rb") as f:
    training = pickle.load(f)
targets = {sanitize_id(kk): clean_seq(vv["rna_chain_sequence"])
           for v in training.values() for kk, vv in v.items() if vv.get("rna_chain_sequence")}
print(f"[INFO] training chains: {len(targets)} ({len(set(targets.values()))} unique)")

q_fa = os.path.join(HERE, "query_temporal.fasta")
t_fa = os.path.join(HERE, "target_training.fasta")
write_fasta(queries, q_fa)
write_fasta(targets, t_fa)

print(f"\n--- MMseqs2 search (k={K}) ---")
# Write into results/, not over the copies shipped in this directory: those define the
# published leakage subsets, and the README quotes their exact hit counts.
OUT_DIR = os.environ.get("MMSEQS_OUT_DIR", os.path.join(HERE, "results"))
os.makedirs(OUT_DIR, exist_ok=True)
out_tsv = os.path.join(OUT_DIR, "mmseqs_results.tsv")
out_csv = os.path.join(OUT_DIR, "mmseqs_results.csv")
tmp = os.path.join(HERE, "tmp")
shutil.rmtree(tmp, ignore_errors=True)
for path in (out_tsv, out_csv):
    if os.path.exists(path):
        os.remove(path)

cmd = [MMSEQS, "easy-search", q_fa, t_fa, out_tsv, tmp,
       "--search-type", "3", "-s", "7.5", "--min-seq-id", "0",
       "--cov-mode", "0", "-c", "0", "-e", "10000", "-k", str(K),
       # --threads 1: MMseqs2's prefilter splits work across threads, so the hit list and
       # its order depend on the thread count without it
       "--threads", "1",
       "--format-output", ",".join(COLUMNS)]
print("  " + " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
shutil.rmtree(tmp, ignore_errors=True)
if not os.path.exists(out_tsv) or os.path.getsize(out_tsv) == 0:
    raise SystemExit("MMseqs2 produced no output:\n" + result.stdout + result.stderr)

hits = pd.read_csv(out_tsv, sep="\t", names=COLUMNS)
hits.to_csv(out_csv, index=False)          # MMseqs2 writes no header; add one
os.remove(out_tsv)
print(f"  {len(hits):,} hits over {hits['query'].nunique()}/{len(queries)} queries")
print(f"[OUT] {out_csv}")

passing = hits[(hits.fident >= MIN_SEQ_ID) & (hits.qcov >= MIN_COV)
               & (hits.tcov >= MIN_COV)].copy()
passing["PDB_ID"] = passing["query"].map(lambda q: meta[q]["PDB_ID"])
passing["Chain_ID"] = passing["query"].map(lambda q: meta[q]["Chain_ID"])
passing["Ligand_ID"] = passing["query"].map(lambda q: meta[q]["Ligand_ID"])
passing = passing.rename(columns={"target": "target_id"})
passing["Identity_Pct"] = (passing["fident"] * 100).round(2)
passing["Query_Coverage_Pct"] = (passing["qcov"] * 100).round(2)
passing["Target_Coverage_Pct"] = (passing["tcov"] * 100).round(2)
passing = passing[["PDB_ID", "Chain_ID", "Ligand_ID", "target_id", "Identity_Pct",
                   "Query_Coverage_Pct", "Target_Coverage_Pct", "qlen", "tlen"]]
passing = passing.sort_values(["PDB_ID", "Identity_Pct"], ascending=[True, False], kind="stable")
passing.to_csv(os.path.join(OUT_DIR, "temporal_vs_training_BEST_HITS.csv"), index=False)

flagged = sorted(passing.PDB_ID.unique())
print(f"\n[OUT] {os.path.join(OUT_DIR, 'temporal_vs_training_BEST_HITS.csv')}  (identity >= {MIN_SEQ_ID}, "
      f"both coverages >= {MIN_COV})")
print("=" * 60)
print(f"{len(passing)} pairs over {len(flagged)} of {len(queries)} test complexes")
print(f"flagged: {', '.join(flagged)}")
print("=" * 60)
print(passing.to_string(index=False))
