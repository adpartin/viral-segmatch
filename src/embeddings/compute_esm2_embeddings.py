"""
Precompute ESM-2 embeddings for protein sequences.


"""

import sys
from pathlib import Path
from tqdm import tqdm

import h5py
import pandas as pd
import numpy as np

import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.timer_utils import Timer
from src.utils.torch_utils import determine_device
from src.utils.esm2_utils import compute_esm2_embeddings

# Config
TASK_NAME = 'segment_pair_classifier'
CUDA_NAME = 'cuda:7'  # Specify GPU device

data_version = 'April_2025'
virus_name = 'bunya'
main_data_dir = project_root / 'data'
processed_data_dir = main_data_dir / 'processed' / virus_name / data_version
output_dir = processed_data_dir / 'embeddings' / virus_name / data_version # embeddings_dir
output_dir.mkdir(parents=True, exist_ok=True)

# ESM-2 model
# MODEL_CKPT = 'facebook/esm2_t6_8M_UR50D'
# MODEL_CKPT = 'facebook/esm2_t12_35M_UR50D'
MODEL_CKPT = 'facebook/esm2_t33_650M_UR50D'
batch_size = 16  # Adjust based on GPU memory (e.g., 8 for A100 40GB)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = determine_device(CUDA_NAME)
model_name = MODEL_CKPT.split('/')[-1]

print(f'main_data_dir:      {main_data_dir}')
print(f'processed_data_dir: {processed_data_dir}')
print(f'output_dir:         {output_dir}')
print(f'device:             {device}')


# Load protein data
total_timer = Timer()
print('\nLoad filtered protein data.')
fname = 'protein_filtered.csv'
datapath = processed_data_dir / fname
try:
    df = pd.read_csv(datapath)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {datapath}")

# Validate and deduplicate
# breakpoint()
print('\nValidate protein data.')
if df['brc_fea_id'].duplicated().any():
    raise ValueError("Duplicate brc_fea_id found in protein_filtered.csv")
max_len = df['prot_seq'].str.len().max()
esm2_max_len = 1024  # ESM-2 max sequence length
if max_len > esm2_max_len:
    print(f"Warning: Max sequence length {max_len} exceeds ESM-2 limit ({esm2_max_len}). Truncating.")
df = df[['brc_fea_id', 'prot_seq']].drop_duplicates().reset_index(drop=True)
print(f"Processing {len(df)} unique proteins")

# Compute embeddings
# breakpoint()
print(f'\nComputing ESM-2 embeddings ({model_name}).')
embeddings, brc_fea_ids = compute_esm2_embeddings(
    sequences=df['prot_seq'].tolist(),
    brc_fea_ids=df['brc_fea_id'].tolist(),
    model_name=MODEL_CKPT,
    batch_size=batch_size,
    device=device
)

# Save embeddings
# breakpoint()
output_file = output_dir / 'esm2_embeddings.h5'
print(f'\nSave embeddings to: {output_file}.')
with h5py.File(output_file, 'w') as file:
    for i, brc_id in enumerate(brc_fea_ids):
        file.create_dataset(brc_id, data=embeddings[i])
del file

# Validate
breakpoint()
print('\nValidate saved embeddings.')
with h5py.File(output_file, 'r') as file:
    saved_ids = list(file.keys())
    assert len(saved_ids) == len(df), f'Mismatch: {len(saved_ids)} embeddings, {len(df)} proteins'
    emb_dim = file[saved_ids[0]].shape[0]
    print(f"Saved {len(saved_ids)} embeddings (dim: {emb_dim})")
del file

total_timer.display_timer()
print('\nDone.')