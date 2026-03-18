import os
import re
import shutil
import subprocess
import pandas as pd
import pickle
import numpy as np

# 1. Configuration paths
TEMPORAL_CSV = "time_dependent_test_set.csv"
TRAINING_PKL = "hariboss_5fd_by_chain_structure.pkl"

# Output directory
WORK_DIR = "verification_temporal_vs_train"
os.makedirs(WORK_DIR, exist_ok=True)


def sanitize_id(s: str) -> str:
    """Clean ID to avoid special characters causing MMseqs2 errors"""
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", str(s))

def clean_seq(seq: str) -> str:
    """Clean sequence: convert to uppercase, remove spaces, convert U to T"""
    if not isinstance(seq, str):
        return ""
    # Convert RNA U to T for easier alignment tool processing
    return seq.strip().upper().replace(" ", "").replace("\n", "").replace("U", "T")

def write_fasta(seq_dict, out_fa: str):
    """Write dictionary to FASTA file"""
    count = 0
    with open(out_fa, "w") as f:
        for k, seq in seq_dict.items():
            seq_clean = clean_seq(seq)
            if not seq_clean:
                continue
            f.write(f">{sanitize_id(k)}\n{seq_clean}\n")
            count += 1
    print(f"[FASTA] Generated {out_fa} with {count} sequences.")

print("--- Loading Data ---")

# Load Temporal Test Set
temporal_df = pd.read_csv(TEMPORAL_CSV)
temporal_seqs = {}
# Build mapping for retrieving original information later
query_meta = {} 

for i, row in temporal_df.iterrows():
    pdb = str(row["PDB ID"])
    chain = str(row["Chain ID"])
    seq = row["RNA sequence"]
    
    # Construct unique ID
    qid = f"{pdb}_{chain}"
    clean_qid = sanitize_id(qid)
    
    temporal_seqs[clean_qid] = seq
    query_meta[clean_qid] = {"PDB ID": pdb, "Chain ID": chain}

print(f"[INFO] Temporal sequences loaded: {len(temporal_seqs)}")

# Load Training Set (Pickle)
training_seqs = {}
with open(TRAINING_PKL, "rb") as f:
    train_data = pickle.load(f)

for k, v in train_data.items():
    for kk, vv in v.items():
        seq = vv.get("rna_chain_sequence", "")
        if seq:
            tid = sanitize_id(kk)
            training_seqs[tid] = seq

print(f"[INFO] Training sequences loaded: {len(training_seqs)}")

# Run MMseqs2 (Permissive Mode)
q_fa = os.path.join(WORK_DIR, "query_temporal.fasta")
t_fa = os.path.join(WORK_DIR, "target_training.fasta")

write_fasta(temporal_seqs, q_fa)
write_fasta(training_seqs, t_fa)

out_tsv = os.path.join(WORK_DIR, "mmseqs_results.tsv")
tmp_dir = os.path.join(WORK_DIR, "tmp")
if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)

print("\n--- Running MMseqs2 (Finding Best Hits) ---")
# Parameter explanation:
# -s 7.5: Maximum sensitivity
# --min-seq-id 0: Do not filter similarity (to find best hits even with low similarity)
# -c 0: Do not filter coverage
# --e-value 10000: Extremely relaxed E-value
# --kmer-per-seq 100: Ensure short sequences can be indexed
cmd = [
    "mmseqs", "easy-search",
    q_fa, t_fa,
    out_tsv, tmp_dir,
    "--search-type", "3", 
    "-s", "7.5",
    "--min-seq-id", "0",
    "--cov-mode", "0",
    "-c", "0",
    "-e", "10000",
    "--format-output", "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"
]
subprocess.run(cmd, check=True)

# Process results & select Best Hit
print("\n--- Processing Results ---")

cols = ["query_id", "target_id", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]

# Read results, create empty DataFrame if file is empty (theoretically rare, unless no matches at all)
if os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 0:
    df_res = pd.read_csv(out_tsv, sep="\t", names=cols)
else:
    df_res = pd.DataFrame(columns=cols)

# Core logic: keep only one best hit for each query_id
# Sorting priority:
# 1. bits (Bitscore) descending: Bitscore is the most reliable indicator of alignment quality
# 2. evalue ascending: if Bitscore is the same, choose lower E-value
# 3. fident descending: if all above are the same, choose higher identity
if not df_res.empty:
    df_res = df_res.sort_values(by=["query_id", "bits", "evalue", "fident"], 
                                ascending=[True, False, True, False])
    # Remove duplicates, keep only the first row (i.e., the best one)
    best_hits = df_res.drop_duplicates(subset=["query_id"], keep="first")
else:
    best_hits = pd.DataFrame(columns=cols)

# Merge original list
all_queries_df = pd.DataFrame([
    {"query_id": k, "Temporal_PDB": v["PDB ID"], "Temporal_Chain": v["Chain ID"]}
    for k, v in query_meta.items()
])

final_df = pd.merge(all_queries_df, best_hits, on="query_id", how="left")

final_df["target_id"] = final_df["target_id"].fillna("NO_HIT")
final_df["fident"] = final_df["fident"].fillna(0.0)
final_df["qcov"] = final_df["qcov"].fillna(0.0)
final_df["tcov"] = final_df["tcov"].fillna(0.0)
final_df["bits"] = final_df["bits"].fillna(0.0)

# Convert to percentage for readability
final_df["Identity_Pct"] = final_df["fident"] * 100
final_df["Query_Coverage_Pct"] = final_df["qcov"] * 100

# Organize column order
output_cols = [
    "Temporal_PDB", "Temporal_Chain", "query_id", 
    "target_id", "Identity_Pct", "Query_Coverage_Pct", 
    "alnlen", "qlen", "tlen", "evalue", "bits"
]
final_df = final_df[output_cols]

# Save and summary
out_csv = os.path.join(WORK_DIR, "temporal_vs_training_BEST_HITS.csv")
final_df.to_csv(out_csv, index=False)

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print(f"Results saved to: {out_csv}")
print(f"Total Temporal Sequences: {len(final_df)}")

# Simple statistics
no_hits = len(final_df[final_df["target_id"] == "NO_HIT"])
high_sim = len(final_df[final_df["Identity_Pct"] > 80])
print(f"- Sequences with NO detectable similarity: {no_hits}")
print(f"- Sequences with >80% Identity to training: {high_sim}")
print("="*50)

print("\nPreview of the CSV:")
print(final_df[["Temporal_PDB", "target_id", "Identity_Pct", "Query_Coverage_Pct"]].head(10))