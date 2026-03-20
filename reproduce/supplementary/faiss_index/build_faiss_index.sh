module load conda
conda activate smartbind_prev

# Configuration
FINGERPRINTS_PATH="data/smol_fp2_list.pkl"
MODEL_PATH="../SMARTBind_weight"
OUTPUT_PREFIX="ligand_library_10M"
BATCH_SIZE=10000

echo "Configuration:"
echo "  Fingerprints path: $FINGERPRINTS_PATH"
echo "  Model path: $MODEL_PATH"
echo "  Output prefix: $OUTPUT_PREFIX"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Run the script
python build_faiss_index.py \
    --fingerprints_path "$FINGERPRINTS_PATH" \
    --model_path "$MODEL_PATH" \
    --output_prefix "$OUTPUT_PREFIX" \
    --device cuda \
    --batch_size "$BATCH_SIZE"

