# Enabling efficient large-scale virtual screening with Faiss index using CPU
[Faiss](https://faiss.ai/) is a library for efficient similarity search and clustering of dense vectors. 
It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.

To enable efficient search of top ligand candidates against a RNA target, we build FAISS index for a virtual screening
library of small molecules. The implementation details are within this tutorial.

## Installation
```bash
conda install -y -c pytorch -c nvidia faiss-gpu=1.9.0 mkl mkl-service
```

## Workflow

### Step 1: Build FAISS Index (One-time Setup)

Use the bash command `build_faiss_index.sh` to pre-compute embeddings for your ligand library and build FAISS indexes.

**Output files:**
- `{prefix}_model1.faiss`, `{prefix}_model2.faiss`, ..., `{prefix}_model10.faiss` - FAISS index files for each model
- `{prefix}_metadata.pkl` - Metadata containing SMILES list and other information

### Step 2: Fast Virtual Screening

Once the FAISS index is built, use `virtual_screening_with_faiss_cpu.sh` to perform fast virtual screening against any RNA target.

This script only needs to:
- Compute the RNA embedding
- Search the pre-computed ligand embeddings using FAISS 

**Example usage:**
```bash
python virtual_screening_with_faiss_cpu.py \
    --rna "GACAGCUGCUGUC" \
    --index_prefix ligand_library_10M \
    --output_dir faiss_vs_results_cpu_10M/ \
    --top_k 100000 \
    --device cpu \
    --num_threads 4
```

**Arguments:**
- `--rna`: RNA sequence string
- `--index_prefix`: Prefix used when building the FAISS index
- `--output_dir`: Output directory path for ranked results of 10 models.
- `--device`: Device to use (`cuda` or `cpu`)
- `--top_k`: (Optional) Return only top K results
- `--num_threads`: (Optional) Number of CPU threads to use for FAISS search (default: 4)

**Output format:**
```
Rank    Ligand_ID    SMILES              Binding_Score
1       42           CC(C)Oc1ccc...      0.856234
2       157          CN1C=NC2=C1...      0.842156
...
```