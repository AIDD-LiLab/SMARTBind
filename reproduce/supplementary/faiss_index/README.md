# Enabling efficient large-scale virtual screening with Faiss index using CPU
[Faiss](https://faiss.ai/) is a library for efficient similarity search and clustering of dense vectors. 
It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.

To enable efficient search of top ligand candidates against a RNA target, we build FAISS index for a virtual screening
library of small molecules. The implementation details are within this tutorial.

## Installation

FAISS is not part of `environment.yml` — it is only needed for this optional pipeline.
Add it to the SMARTBind environment with:

```bash
conda activate SMARTBind_env
conda install -y -c pytorch -c nvidia faiss-gpu=1.9.0 mkl mkl-service
```

## Workflow

### Step 1: Build FAISS Index (One-time Setup)

Use `bash build_faiss_index.sh` to pre-compute embeddings for your ligand library and build
the FAISS indexes. The library is a text file with one SMILES per line — the paper used the
1M ChemSpace sample from the Zenodo record. Point `SMILES_PATH` at yours, or call
`build_faiss_index.py --smiles <file>` directly. Needs a GPU and `SMARTBind_weight/`.

Fig. 4b is a throughput benchmark over a 10-million-vector index, built by replicating that
1M library ten times (`REPLICATE=10`); search cost depends on how many vectors the index
holds, not on how many are distinct. For real screening leave `REPLICATE` at 1 — a
replicated index returns each hit ten times.

**Output files:**
- `{prefix}_model1.faiss`, `{prefix}_model2.faiss`, ..., `{prefix}_model10.faiss` - FAISS index files for each model
- `{prefix}_metadata.pkl` - Metadata: checkpoint names, embedding dimension, model count, ligand count

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

**Output format:** one file per ensemble model, `<output_dir>/model<N>_top<K>.txt`.
`Ligand_Index` is the 0-based position of the SMILES in the library file passed to step 1.

```
# Search time: 0.153 seconds
Rank    Ligand_Index    Binding_Score
1       3               -0.006330
2       43              -0.006330
...
```

**Output files:** the step-1 metadata pickle records `checkpoint_names`, `embedding_dim`,
`num_models` and `num_ligands` — not the SMILES themselves; map indices back through your
own library file.