conda activate SMARTBind_env

python inference.py \
    --rna "GGCGUAUAUCCUUAAUGAUAUGGUUUAAGGGCAAUACAUAGAAACCACAAAUUUCUUACUGCGUC" \
    --smiles "notebook/ligand_library.txt" \
    --model_path "SMARTBind_weight" \
    --output "notebook/binding_score.txt" \
    --device "cuda" \
    --batch_size 10000 \
    --num_workers 4
