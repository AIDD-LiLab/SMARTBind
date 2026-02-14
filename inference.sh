#!/bin/bash
#SBATCH --job-name=api_check
#SBATCH --output=api_check_multi.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12gb
#SBATCH --gpus=1
#SBATCH --account=yanjun.li
#SBATCH --qos=yanjun.li
#SBATCH --time=72:00:00

module load cuda/12.4.1
module load conda

conda activate smartbind_prev

python virtual_screening.py \
    --rna "GGCGUAUAUCCUUAAUGAUAUGGUUUAAGGGCAAUACAUAGAAACCACAAAUUUCUUACUGCGUC" \
    --smiles "notebook/ligand_library_1M.txt" \
    --model_path "SMARTBind_weight" \
    --output "notebook/binding_score_1M_multi.txt" \
    --device "cuda" \
    --batch_size 10000 \
    --num_workers 4
