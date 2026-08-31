"""
Build FAISS Index for Small Molecule Library

Simple workflow:
1. Read the ligand library (one SMILES per line) and compute its FP2 fingerprints
2. Load SmartBind models
3. Compute embeddings in batches
4. Build FAISS indexes

Usage:
    python build_faiss_index.py --smiles library.smi --output_prefix ligand_library

`--fingerprints_path` is accepted instead of `--smiles` if you already hold a pickle of
FP2 vectors; exactly one of the two is required.
"""

import argparse
import sys
from pathlib import Path


def _find_repo_root(start):
    """Walk up until the SMARTBind package is found.

    `sys.path.append("..")` only worked when the working directory happened to be a
    direct child of the repository root; this makes the script importable wherever it
    is run from.
    """
    d = Path(start).resolve()
    for folder in (d, *d.parents):
        if (folder / "smartbind" / "__init__.py").is_file():
            return folder
    raise SystemExit(
        "\nCould not locate the SMARTBind repository root from " + str(start) +
        ".\nRun this script from inside a checkout of the repository.\n")


sys.path.insert(0, str(_find_repo_root(__file__)))

import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
from smartbind import load_smartbind_models
from smartbind import logger

# Try to import faiss
try:
    import faiss
    logger.info("FAISS imported successfully")
except ImportError as e:
    logger.error(f"Failed to import FAISS: {e}")
    raise


def process_single_model(model, model_idx, num_models, smol_fp2_list, batch_size, output_prefix, device):
    """Process one model: compute embeddings, build index, save, and release memory."""
    logger.info(f'Processing model {model_idx + 1}/{num_models}')
    logger.info(f'Computing embeddings in batches of {batch_size}')
    
    # Create FAISS index (CPU only to avoid GPU memory issues)
    embedding_dim = None
    index = None
    
    num_batches = len(smol_fp2_list) // batch_size
    if len(smol_fp2_list) % batch_size != 0:
        num_batches += 1
    
    # Process in batches and add to FAISS incrementally
    for i in tqdm(range(num_batches), desc=f'Model {model_idx + 1}'):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(smol_fp2_list))
        
        # Compute embeddings for this batch
        ligand_embeds = model.inference_list_smols(smol_fp2_list[start:end])
        embeddings_batch = ligand_embeds.detach().cpu().numpy().astype('float32')
        
        # Initialize index on first batch
        if index is None:
            embedding_dim = embeddings_batch.shape[1]
            index = faiss.IndexFlatIP(embedding_dim)
            logger.info(f'Created FAISS index with dimension {embedding_dim}')
        
        # Add to index
        index.add(embeddings_batch)
        
        # Clear GPU cache if using CUDA
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Clear batch from memory
        del ligand_embeds, embeddings_batch
    
    logger.info(f'Model {model_idx + 1} index built: {index.ntotal} vectors')
    
    # Save index immediately
    # Name the index after the checkpoint, not after a position in the model list: the
    # search script calls load_smartbind_models() separately, and pairing by position
    # silently mismatches RNA embeddings to ligand indexes if the two listings differ.
    index_path = f'{output_prefix}_{model.checkpoint_name}.faiss'
    faiss.write_index(index, index_path)
    logger.info(f'Saved {index_path}')
    
    # Clear index from memory
    del index
    
    return embedding_dim


def save_metadata(num_ligands, embedding_dim, num_models, output_prefix, checkpoint_names):
    """Save metadata."""
    metadata = {
        'checkpoint_names': list(checkpoint_names),
        'embedding_dim': embedding_dim,
        'num_models': num_models,
        'num_ligands': num_ligands
    }
    
    metadata_path = f'{output_prefix}_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f'Saved {metadata_path}')
    logger.info('='*80)
    logger.info(f'Summary: {num_ligands} ligands, {num_models} models, dim={embedding_dim}')
    logger.info('='*80)


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index for ligand library')
    parser.add_argument('--smiles', type=str,
                        help='Ligand library: a text file with one SMILES string per line')
    parser.add_argument('--fingerprints_path', type=str,
                        help='Alternative to --smiles: a pickle holding a list of FP2 vectors')
    parser.add_argument('--model_path', type=str,
                        default=str(_find_repo_root(__file__) / 'SMARTBind_weight'))
    parser.add_argument('--output_prefix', type=str, default='ligand_library')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--replicate', type=int, default=1,
                        help='Repeat the library N times to build a larger index. Only for '
                             'throughput benchmarking; Fig. 4b used 10 over the 1M library.')
    
    args = parser.parse_args()
    
    logger.info('='*80)
    logger.info('Build FAISS Index')
    logger.info('='*80)
    logger.info(f'Library: {args.smiles or args.fingerprints_path}')
    logger.info(f'Models: {args.model_path}')
    logger.info(f'Output: {args.output_prefix}')
    logger.info(f'Device: {args.device}')
    
    # Step 1: fingerprints, either computed from the library or loaded from a pickle
    if bool(args.smiles) == bool(args.fingerprints_path):
        raise SystemExit('\nPass exactly one of --smiles or --fingerprints_path.\n')
    if args.smiles:
        from smartbind.preprocess import convert_smiles_to_pf2
        logger.info('Computing FP2 fingerprints from the ligand library...')
        with open(args.smiles) as handle:
            library = [line.strip() for line in handle if line.strip()]
        smol_fp2_list = [convert_smiles_to_pf2(s) for s in tqdm(library, desc='FP2')]
    else:
        logger.info('Loading pre-computed fingerprints...')
        with open(args.fingerprints_path, 'rb') as f:
            smol_fp2_list = pickle.load(f)

    # Fig. 4b is a throughput benchmark on a 10-million-vector index, and that index was
    # built by replicating the 1M ChemSpace sample ten times -- search cost depends on the
    # number of vectors, not on how many of them are distinct.  Keep --replicate at its
    # default of 1 for real screening; a replicated index returns each hit `replicate`
    # times.
    if args.replicate > 1:
        smol_fp2_list = smol_fp2_list * args.replicate
        logger.warning(f'Replicated the library {args.replicate}x -> {len(smol_fp2_list)} '
                       f'vectors. This is a throughput benchmark setting, not a screening one.')
    logger.info(f'Loaded {len(smol_fp2_list)} fingerprints')
    
    # Step 2: Load models
    logger.info('Loading SmartBind models')
    smartbind_models = load_smartbind_models(
        model_path=args.model_path,
        device=args.device,
        vs_mode=True
    )
    num_models = len(smartbind_models)
    logger.info(f'Loaded {num_models} models')
    
    # Step 3-5: Process each model sequentially to save memory
    embedding_dim = None
    for model_idx, model in enumerate(smartbind_models):
        dim = process_single_model(
            model, model_idx, num_models, smol_fp2_list,
            args.batch_size, args.output_prefix, args.device
        )
        
        if embedding_dim is None:
            embedding_dim = dim
        
        # Release model from memory (keep only next one in GPU)
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info(f'Completed {model_idx + 1}/{num_models} models\n')
    
    # Step 6: Save metadata
    save_metadata(len(smol_fp2_list), embedding_dim, num_models, args.output_prefix,
                  [m.checkpoint_name for m in smartbind_models])


if __name__ == '__main__':
    main()
