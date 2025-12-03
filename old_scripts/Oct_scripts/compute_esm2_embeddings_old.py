"""
Precompute ESM-2 embeddings for protein sequences.
"""
import sys
from pathlib import Path
import random

import h5py
import pandas as pd
import numpy as np
import torch


# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Set random seeds for reproducible ESM-2 embeddings
# NOTE: This is separate from the preprocessing random_seed which controls file sampling
EMBEDDING_SEED = 42  # Fixed seed for ESM-2 determinism
random.seed(EMBEDDING_SEED)  # Python's random module (used by transformers library)
torch.manual_seed(EMBEDDING_SEED)
np.random.seed(EMBEDDING_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(EMBEDDING_SEED)
    torch.cuda.manual_seed_all(EMBEDDING_SEED)

# Set deterministic behavior for ESM-2 computation
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f'Set ESM-2 embedding seed to {EMBEDDING_SEED} for reproducible results')
print('Note: This is separate from preprocessing random_seed which controls file sampling')

from src.utils.timer_utils import Timer
from src.utils.torch_utils import determine_device
from src.utils.esm2_utils import compute_esm2_embeddings
from src.utils.config import (
    USE_CORE_PROTEINS_ONLY, VIRUS_NAME, DATA_VERSION, 
    ESM2_MAX_RESIDUES, ESM2_PROTEIN_SEQ_COL
)
from src.utils.protein_utils import get_core_protein_filter_mask, STANDARD_AMINO_ACIDS

# Local config (GPU-specific, keep here for now)
CUDA_NAME = 'cuda:7'  # Specify GPU device
MODEL_CKPT = 'facebook/esm2_t33_650M_UR50D'   # embedding dim: 1280D
batch_size = 16  # Adjust based on GPU memory (e.g., 8 for A100 40GB)

# Define paths
main_data_dir = project_root / 'data'
processed_data_dir = main_data_dir / 'processed' / VIRUS_NAME / DATA_VERSION
output_dir = main_data_dir / 'embeddings' / VIRUS_NAME / f'{DATA_VERSION}_v1' # embeddings_dir
output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"✅ Loaded {len(df):,} protein records from {datapath}")
except Exception as e:
    raise RuntimeError(f"Error loading data from {datapath}: {e}")

# Validate required columns
required_cols = ['brc_fea_id', ESM2_PROTEIN_SEQ_COL, 'length']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

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
# from esm2_embeddings.h5 (HDF5 datasets can't have duplicate keys)
if df['brc_fea_id'].duplicated().any():
    raise ValueError("Duplicate brc_fea_id found in protein_filtered.csv")
# Check for invalid sequences (non-standard amino acids)
standard_aa_pattern = ''.join(STANDARD_AMINO_ACIDS)
invalid_seqs = df[df[ESM2_PROTEIN_SEQ_COL].str.contains(f'[^{standard_aa_pattern}]', na=False)]
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
output_file = output_dir / 'esm2_embeddings.h5'
print(f'\nSave embeddings to: {output_file}.')
with h5py.File(output_file, 'w') as file:
    for i, brc_id in enumerate(brc_fea_ids):
        file.create_dataset(brc_id, data=embeddings[i])


# Save embeddings to csv
emb_df = pd.DataFrame(
    embeddings,
    columns=[f'emb_{i}' for i in range(embeddings.shape[1])]
)
emb_df.insert(0, 'brc_fea_id', brc_fea_ids)
emb_df.to_csv(output_dir / 'esm2_embeddings.csv', index=False)


# Save failed ids to csv
failed_df = pd.DataFrame({'brc_fea_id': failed_ids})
failed_df.to_csv(output_dir / 'failed_ids.csv', index=False)


# Validate
print('\nValidate saved embeddings.')
with h5py.File(output_file, 'r') as file:
    saved_ids = list(file.keys())
    expected_count = len(df) - len(failed_ids)
    if len(saved_ids) != expected_count:
        raise ValueError(f'Embedding count mismatch: {len(saved_ids)} saved, {expected_count} expected')
    emb_dim = file[saved_ids[0]].shape[0]
    print(f'✅ Saved {len(saved_ids)} embeddings (dim: {emb_dim})')
if failed_ids:
    print(f'Failed to process {len(failed_ids)} sequences: {failed_ids[:5]}...')

total_timer.display_timer()
print('\nDone.')