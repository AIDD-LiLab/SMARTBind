#!/bin/bash
# Step 2: screen one RNA against the pre-built FAISS indexes.  CPU only.
#
# On SLURM, submit with something like (adjust partition/qos for your site):
#   sbatch --job-name=faiss_cpu --ntasks=1 --cpus-per-task=4 --mem=64gb \
#          --time=96:00:00 --output=faiss_inference_cpu.out virtual_screening_with_faiss_cpu.sh
set -euo pipefail

conda activate SMARTBind_env      # `module load conda` first on a module-based cluster

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

THREADS="${THREADS:-4}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

python "$REPO/reproduce/supplementary/faiss_index/virtual_screening_with_faiss_cpu.py" \
    --rna "GACAGCUGCUGUC" \
    --index_prefix "${OUTPUT_PREFIX:-ligand_library_1M}" \
    --output_dir faiss_vs_results_cpu/ \
    --top_k 100000 \
    --device cpu \
    --num_threads "$THREADS"
