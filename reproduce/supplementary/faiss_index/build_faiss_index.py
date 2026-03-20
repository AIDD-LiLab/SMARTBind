"""
Build FAISS Index for Small Molecule Library

Simple workflow:
1. Load pre-computed fingerprints from pkl file
2. Load SmartBind models
3. Compute embeddings in batches
4. Build FAISS indexes

Usage:
    python build_faiss_index.py --fingerprints_path data/smol_fp2_list.pkl --output_prefix ligand_library
"""

import argparse
import sys
sys.path.append("..") 

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
    index_path = f'{output_prefix}_model{model_idx + 1}.faiss'
    faiss.write_index(index, index_path)
    logger.info(f'Saved {index_path}')
    
    # Clear index from memory
    del index
    
    return embedding_dim


def save_metadata(num_ligands, embedding_dim, num_models, output_prefix):
    """Save metadata."""
    metadata = {
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
    parser.add_argument('--fingerprints_path', type=str, required=True, 
                        help='Path to pre-computed fingerprints pkl file')
    parser.add_argument('--model_path', type=str, default='../SMARTBind_weight')
    parser.add_argument('--output_prefix', type=str, default='ligand_library')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=10000)
    
    args = parser.parse_args()
    
    logger.info('='*80)
    logger.info('Build FAISS Index')
    logger.info('='*80)
    logger.info(f'Fingerprints: {args.fingerprints_path}')
    logger.info(f'Models: {args.model_path}')
    logger.info(f'Output: {args.output_prefix}')
    logger.info(f'Device: {args.device}')
    
    # Step 1: Load pre-computed fingerprints
    logger.info('Loading pre-computed fingerprints...')
    with open(args.fingerprints_path, 'rb') as f:
        smol_fp2_list = pickle.load(f)

    # duplicae the list 10 times to create a larger dataset for testing
    smol_fp2_list_10fold = smol_fp2_list * 10
    smol_fp2_list = smol_fp2_list_10fold
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
    save_metadata(len(smol_fp2_list), embedding_dim, num_models, args.output_prefix)


if __name__ == '__main__':
    main()
