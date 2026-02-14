"""
SmartBind Virtual Screening Function

Given an RNA sequence and a txt/smi file of SMILES, predict binding scores
and save a scored result file.

Usage:
    from virtual_screening import virtual_screening
    virtual_screening(
        rna_sequence='GACAGCUGCUGUC',
        smiles_file='ligands.smi',
        model_path='../SMARTBind_weight',
        output_file='binding_scores.txt',
        device='cpu',
        batch_size=10000,
    )
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tqdm
from multiprocessing import Pool, cpu_count
from torch.nn.functional import cosine_similarity
from smartbind.preprocess import convert_smiles_to_pf2
from smartbind import load_smartbind_models, logger


def virtual_screening(
    rna_sequence: str,
    smiles_file: str,
    model_path: str,
    output_file: str,
    device: str = 'cpu',
    batch_size: int = 10000,
    num_workers: int = 1,
):
    """
    End-to-end virtual screening: read SMILES file, predict binding scores, save results.

    Args:
        rna_sequence: RNA nucleotide sequence, e.g. 'GACAGCUGCUGUC'.
        smiles_file:  Path to a txt/smi file with one SMILES per line.
        model_path:   Path to directory containing SmartBind .pth weight files.
        output_file:  Path to save the scored output (tab-separated: SMILES, score).
        device:       'cpu' or 'cuda'.
        batch_size:   Number of ligands per inference batch.
        num_workers:  Number of parallel workers for SMILES→FP2 conversion (default: 1).
    """
    # ── 1. Read SMILES ────────────────────────────────────────────────────────
    with open(smiles_file, 'r') as f:
        smiles_list_raw = f.read().splitlines()
    logger.info(f'Read {len(smiles_list_raw)} SMILES from {smiles_file}')

    # ── 2. Convert SMILES → FP2 fingerprints (multiprocessing) ────────────────
    n_workers = min(num_workers, cpu_count())
    logger.info(f'Converting SMILES to FP2 with {n_workers} worker(s)')

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            fp2_results = list(tqdm.tqdm(
                pool.imap(convert_smiles_to_pf2, smiles_list_raw, chunksize=1024),
                total=len(smiles_list_raw),
                desc='Converting SMILES to fingerprints',
            ))
    else:
        fp2_results = [
            convert_smiles_to_pf2(smi)
            for smi in tqdm.tqdm(smiles_list_raw, desc='Converting SMILES to fingerprints')
        ]

    smol_fp2_list = []
    smiles_list = []
    for smi, fp2 in zip(smiles_list_raw, fp2_results):
        if fp2 is not None:
            smol_fp2_list.append(fp2)
            smiles_list.append(smi)
        else:
            logger.warning(f'Failed to convert SMILES: {smi}')
    logger.info(f'{len(smiles_list)}/{len(smiles_list_raw)} SMILES successfully converted')

    # ── 3. Load models ────────────────────────────────────────────────────────
    logger.info(f'Loading models from {model_path}')
    smartbind_models = load_smartbind_models(
        model_path=model_path,
        device=device,
        vs_mode=True,
    )
    if not isinstance(smartbind_models, list):
        smartbind_models = [smartbind_models]
    logger.info(f'Loaded {len(smartbind_models)} model(s)')

    # ── 4. Predict binding scores ─────────────────────────────────────────────
    rank_result_by_models = {}
    for model in tqdm.tqdm(smartbind_models, desc='Predicting binding scores by models'):
        rna_embed = model.inference_single_rna(rna_sequence)

        rank_result_by_models[smartbind_models.index(model)] = []
        num_batches = (len(smol_fp2_list) + batch_size - 1) // batch_size
        for i in tqdm.tqdm(range(num_batches), desc='Batching ligands'):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(smol_fp2_list))
            ligand_embeds = model.inference_list_smols(smol_fp2_list[start:end])
            similarities = cosine_similarity(rna_embed, ligand_embeds).tolist()
            rank_result_by_models[smartbind_models.index(model)].extend(similarities)

    # ── 5. Average across models & save ───────────────────────────────────────
    n_models = len(smartbind_models)
    n_ligands = len(smiles_list)
    avg_scores = [
        sum(rank_result_by_models[m][i] for m in range(n_models)) / n_models
        for i in range(n_ligands)
    ]

    with open(output_file, 'w') as f:
        f.write('SMILES\tBinding_Score\n')
        for smi, score in zip(smiles_list, avg_scores):
            f.write(f'{smi}\t{score}\n')

    logger.info(f'Results saved to {output_file} ({n_ligands} ligands, {n_models} models)')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SmartBind Virtual Screening')
    parser.add_argument('--rna', type=str, required=True, help='RNA sequence')
    parser.add_argument('--smiles', type=str, required=True, help='Path to SMILES file (one per line)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to SmartBind model weights directory')
    parser.add_argument('--output', type=str, default='binding_scores.txt', help='Output file path')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for ligand inference')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers for SMILES to FP2 conversion')
    args = parser.parse_args()

    virtual_screening(
        rna_sequence=args.rna,
        smiles_file=args.smiles,
        model_path=args.model_path,
        output_file=args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
