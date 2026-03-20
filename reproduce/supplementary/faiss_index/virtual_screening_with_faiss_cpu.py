import argparse
import sys
sys.path.append("..")

import numpy as np
import torch
import faiss
import pickle
import time
import os
from smartbind import load_smartbind_models
from smartbind import logger


def load_metadata(index_prefix):
    """
    Load metadata from disk (without loading FAISS indexes).
    
    Args:
        index_prefix: Prefix used when building the indexes
        
    Returns:
        metadata: Dict containing metadata
    """
    logger.info(f'Loading metadata from {index_prefix}_metadata.pkl')
    
    # Load metadata
    with open(f'{index_prefix}_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    num_models = metadata['num_models']
    logger.info(f'Found metadata for {num_models} models')
    logger.info(f'Library contains {metadata["num_ligands"]} ligands')
    
    return metadata


def load_single_faiss_index(index_prefix, model_idx):
    """
    Load a single FAISS index for one model.
    
    Args:
        index_prefix: Prefix used when building the indexes
        model_idx: Model index (0-based)
        
    Returns:
        FAISS index for the specified model
    """
    index_path = f'{index_prefix}_model{model_idx + 1}.faiss'
    logger.info(f'Loading FAISS index from {index_path}')
    return faiss.read_index(index_path)


def search_with_rna_embedding(rna_sequence, index_prefix, smartbind_models, device, top_k):
    """
    Search for top ligands using pre-computed FAISS indexes (CPU version).
    Loads one index at a time to avoid OOM.
    
    Args:
        rna_sequence: RNA sequence string
        index_prefix: Prefix for FAISS index files
        smartbind_models: List of loaded SmartBind models
        device: Device to use for computation
        top_k: Number of top results to return per model
        
    Returns:
        results_by_model: Dict mapping model index to (indices, scores, time) tuples
    """
    results_by_model = {}
    
    logger.info('Using CPU FAISS indexes (no top_k limit)')
    logger.info('Processing one index at a time to save memory')
    
    # Compute RNA embeddings and search for each model
    for model_idx, model in enumerate(smartbind_models):
        logger.info(f'Processing model {model_idx + 1}/{len(smartbind_models)}')
        
        start_time = time.time()
        
        # Compute RNA embedding
        rna_embed = model.inference_single_rna(rna_sequence)
        rna_embed_np = rna_embed.detach().cpu().numpy().astype('float32')
        
        rna_time = time.time() - start_time
        
        # Load FAISS index for this model only
        search_start = time.time()
        index = load_single_faiss_index(index_prefix, model_idx)
        
        # FAISS search returns (distances, indices)
        distances, indices = index.search(rna_embed_np, top_k)
        
        search_time = time.time() - search_start
        total_time = time.time() - start_time
        
        logger.info(f'  RNA embedding: {rna_time:.3f}s, FAISS search (incl. loading): {search_time:.3f}s, Total: {total_time:.3f}s')
        
        # Store results for this model
        results_by_model[model_idx] = (indices[0], distances[0], total_time)
        
        # Delete index to free memory
        del index
        logger.info(f'  Freed memory for model {model_idx + 1}')
    
    return results_by_model


def main():
    parser = argparse.ArgumentParser(
        description='Virtual screening with pre-computed FAISS index (CPU version)'
    )
    parser.add_argument(
        '--rna',
        type=str,
        required=True,
        help='RNA sequence string'
    )
    parser.add_argument(
        '--index_prefix',
        type=str,
        required=True,
        help='Prefix for FAISS index files (without _model1.faiss suffix)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='../SMARTBind_weight',
        help='Path to SmartBind model weights'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results (will create model-specific files)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for RNA embedding computation'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=1000,
        help='Return top K results per model (no limit on CPU)'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=8,
        help='Number of CPU threads for FAISS'
    )
    
    args = parser.parse_args()
    
    # Set number of threads for CPU computation
    faiss.omp_set_num_threads(args.num_threads)
    logger.info(f'Set FAISS to use {args.num_threads} CPU threads')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('='*80)
    logger.info('Virtual Screening with FAISS Index (CPU Version)')
    logger.info('='*80)
    logger.info(f'RNA sequence: {args.rna}')
    logger.info(f'Index prefix: {args.index_prefix}')
    logger.info(f'Output directory: {args.output_dir}')
    logger.info(f'Device (for RNA embedding): {args.device}')
    logger.info(f'Top K per model: {args.top_k}')
    
    # Load metadata only (not FAISS indexes yet)
    metadata = load_metadata(args.index_prefix)
    
    # Load SmartBind models
    logger.info(f'Loading SmartBind models from {args.model_path}')
    smartbind_models = load_smartbind_models(
        model_path=args.model_path,
        device=args.device,
        vs_mode=True
    )
    logger.info(f'Loaded {len(smartbind_models)} models')
    
    # Verify number of models matches
    if len(smartbind_models) != metadata['num_models']:
        logger.warning(
            f'Number of loaded models ({len(smartbind_models)}) does not match '
            f'number of models in index ({metadata["num_models"]})'
        )
    
    # Perform virtual screening (will load indexes one at a time)
    logger.info('Starting virtual screening...')
    results_by_model = search_with_rna_embedding(
        args.rna,
        args.index_prefix,
        smartbind_models,
        args.device,
        args.top_k
    )
    
    # Save results for each model
    logger.info('Saving results...')
    for model_idx, (ligand_indices, scores, search_time) in results_by_model.items():
        output_file = os.path.join(args.output_dir, f'model{model_idx + 1}_top{args.top_k}.txt')
        
        with open(output_file, 'w') as f:
            f.write(f'# Search time: {search_time:.3f} seconds\n')
            f.write('Rank\tLigand_Index\tBinding_Score\n')
            for rank, (idx, score) in enumerate(zip(ligand_indices, scores), 1):
                f.write(f'{rank}\t{idx}\t{score:.6f}\n')
        
        logger.info(f'Saved model {model_idx + 1} results to {output_file} (search time: {search_time:.3f}s)')
        logger.info(f'  Top 3: {ligand_indices[:3].tolist()} with scores {scores[:3].tolist()}')
    
    logger.info('='*80)
    logger.info('Virtual screening complete!')
    logger.info('='*80)


if __name__ == '__main__':
    main()
