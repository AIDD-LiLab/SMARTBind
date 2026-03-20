#!/bin/bash
#SBATCH --job-name=faiss_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --partition=bigmem
#SBATCH --qos=yanjun.li-b
#SBATCH --time=96:00:00
#SBATCH --output=faiss_inference_cpu.out
pwd; hostname; date

module load conda
conda activate smartbind_prev

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python virtual_screening_with_faiss_cpu.py \
    --rna "GACAGCUGCUGUC" \
    --index_prefix ligand_library_10M \
    --output_dir faiss_vs_results_cpu_10M/ \
    --top_k 100000 \
    --device cpu \
    --num_threads 4
