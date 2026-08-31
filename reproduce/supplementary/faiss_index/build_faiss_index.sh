#!/bin/bash
# Step 1 of the SMARTBind-FAISS pipeline: embed a ligand library once and build the
# per-model FAISS indexes.  Needs a GPU and the 10-model ensemble in SMARTBind_weight/.
set -euo pipefail

conda activate SMARTBind_env      # `module load conda` first on a module-based cluster

# REPO is the repository root, so this runs from any directory.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# One SMILES per line.  The paper used the 1M ChemSpace sample from the Zenodo record,
# which unpacks to reproduce/fig4/ (too large to ship in git); any .smi file works.
SMILES_PATH="${SMILES_PATH:-$REPO/reproduce/fig4/Chemspace_Screening_Compounds_SMILES_sampled1M.smi}"
MODEL_PATH="$REPO/SMARTBind_weight"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ligand_library_1M}"
# Fig. 4b's 10M timing benchmark used REPLICATE=10 over this 1M library; leave at 1 to screen.
REPLICATE="${REPLICATE:-1}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

echo "Configuration:"
echo "  SMILES library: $SMILES_PATH"
echo "  Model path:     $MODEL_PATH"
echo "  Output prefix:  $OUTPUT_PREFIX"
echo "  Batch size:     $BATCH_SIZE"
echo "  Replicate:      $REPLICATE"
echo ""

python "$REPO/reproduce/supplementary/faiss_index/build_faiss_index.py" \
    --smiles "$SMILES_PATH" \
    --model_path "$MODEL_PATH" \
    --output_prefix "$OUTPUT_PREFIX" \
    --device cuda \
    --batch_size "$BATCH_SIZE" \
    --replicate "$REPLICATE"
