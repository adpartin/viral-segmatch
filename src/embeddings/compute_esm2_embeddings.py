"""
Precompute ESM-2 embeddings for protein sequences.

"""

import sys
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.timer_utils import Timer
from src.utils.torch_utils import determine_device
from src.utils.esm2_utils import compute_esm2_embeddings
from src.utils.config import (
    USE_CORE_PROTEINS_ONLY, VIRUS_NAME, DATA_VERSION, 
    ESM2_MAX_RESIDUES, ESM2_PROTEIN_SEQ_COL
)
from src.utils.protein_utils import get_core_protein_filter_mask

# Local config (GPU-specific, keep here for now)
CUDA_NAME = 'cuda:7'  # Specify GPU device

# MODEL_CKPT = 'facebook/esm2_t6_8M_UR50D'    # embedding dim: 320D
# MODEL_CKPT = 'facebook/esm2_t12_35M_UR50D'  # embedding dim: 480D
# MODEL_CKPT = 'facebook/esm2_t30_150M_UR50D' # embedding dim: 640D
MODEL_CKPT = 'facebook/esm2_t33_650M_UR50D'   # embedding dim: 1280D
# MODEL_CKPT = 'facebook/esm2_t36_3B_UR50D'   # embedding dim: 2560D
# MODEL_CKPT = 'facebook/esm2_t48_15B_UR50D'  # embedding dim: 5120D
batch_size = 16  # Adjust based on GPU memory (e.g., 8 for A100 40GB)

# Define paths
main_data_dir = project_root / 'data'
processed_data_dir = main_data_dir / 'processed' / VIRUS_NAME / DATA_VERSION
output_dir = main_data_dir / 'embeddings' / VIRUS_NAME / DATA_VERSION # embeddings_dir
output_dir.mkdir(parents=True, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = determine_device(CUDA_NAME)
model_name = MODEL_CKPT.split('/')[-1]

print(f'\nmain_data_dir:      {main_data_dir}')
print(f'processed_data_dir: {processed_data_dir}')
print(f'output_dir:         {output_dir}')
print(f'device:             {device}')


# Load protein data
total_timer = Timer()
print('\nLoad preprocessed protein data.')
fname = 'protein_final.csv'
datapath = processed_data_dir / fname
try:
    df = pd.read_csv(datapath)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {datapath}")

# Filter to core proteins if configured
if USE_CORE_PROTEINS_ONLY:
    print(f"Filter to core proteins only (USE_CORE_PROTEINS_ONLY={USE_CORE_PROTEINS_ONLY})")
    original_count = len(df)
    mask = get_core_protein_filter_mask(df)
    df = df[mask].reset_index(drop=True)
    print(f"Filtered from {original_count} to {len(df)} core proteins")
else:
    print(f"Use all proteins (USE_CORE_PROTEINS_ONLY={USE_CORE_PROTEINS_ONLY})")


# Validate and deduplicate
# breakpoint()
print('\nValidate protein data.')
# brc_fea_ids must be unique because they are used as keys to query embeddings
# from esm2_embeddings.h5 (HDF5 datasets can’t have duplicate keys)
if df['brc_fea_id'].duplicated().any():
    raise ValueError("Duplicate brc_fea_id found in protein_filtered.csv")
# standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')  # 20 canonical amino acids
invalid_seqs = df[df[ESM2_PROTEIN_SEQ_COL].str.contains(r'[^ACDEFGHIKLMNPQRSTVWY]', na=False)]
if not invalid_seqs.empty:
    print(f'Warning: {len(invalid_seqs)} invalid sequences found:')
    print(invalid_seqs[['brc_fea_id', ESM2_PROTEIN_SEQ_COL]].head())
# Identify sequences longer than ESM-2 max length
truncated = df[df['length'] > ESM2_MAX_RESIDUES][['brc_fea_id', 'canonical_segment', 'length']]
if not truncated.empty:
    print(f'Warning: {len(truncated)} sequences truncated (>{ESM2_MAX_RESIDUES} residues):')
    print(truncated.head())
# We expect that each brc_fea_id corresponds to a unique protein sequence
n_org_proteins = len(df)
df = df[['brc_fea_id', ESM2_PROTEIN_SEQ_COL]].drop_duplicates().reset_index(drop=True)
if len(df) < n_org_proteins:
    print(f'Dropped {n_org_proteins - len(df)} duplicate protein sequences')
print(f'Processing {len(df)} unique proteins')


# Compute embeddings
# breakpoint()
print(f'\nCompute ESM-2 embeddings ({model_name}).')
embeddings, brc_fea_ids, failed_ids = compute_esm2_embeddings(
    sequences=df[ESM2_PROTEIN_SEQ_COL].tolist(),
    brc_fea_ids=df['brc_fea_id'].tolist(),
    model_name=MODEL_CKPT,
    batch_size=batch_size,
    device=device,
    max_length=ESM2_MAX_RESIDUES + 2
)


# Save embeddings
# breakpoint()
output_file = output_dir / 'esm2_embeddings.h5'
print(f'\nSave embeddings to: {output_file}.')
with h5py.File(output_file, 'w') as file:
    for i, brc_id in enumerate(brc_fea_ids):
        file.create_dataset(brc_id, data=embeddings[i])


# Validate
# breakpoint()
print('\nValidate saved embeddings.')
with h5py.File(output_file, 'r') as file:
    saved_ids = list(file.keys())
    assert len(saved_ids) == len(df), f'Mismatch: {len(saved_ids)} embeddings, {len(df)} proteins'
    assert len(saved_ids) == len(df) - len(failed_ids), f'Mismatch: {len(saved_ids)} embeddings, {len(df) - len(failed_ids)} expected'
    emb_dim = file[saved_ids[0]].shape[0]
    print(f'Saved {len(saved_ids)} embeddings (dim: {emb_dim})')
if failed_ids:
    print(f'Failed to process {len(failed_ids)} sequences: {failed_ids[:5]}...')

total_timer.display_timer()
print('\nDone.')