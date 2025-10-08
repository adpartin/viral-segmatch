"""
Precompute ESM-2 embeddings for protein sequences.
Virus-agnostic version with command-line interface.
"""
import argparse
import sys
from pathlib import Path

import h5py
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# print(f'project_root: {project_root}')

from src.utils.timer_utils import Timer
from src.utils.torch_utils import determine_device
from src.utils.esm2_utils import compute_esm2_embeddings
from src.utils.protein_utils import STANDARD_AMINO_ACIDS
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds
from src.utils.config_hydra import (
    get_virus_config_hydra, 
    print_config_summary
)

# Manual configs
# TODO: consider setting these somewhere else (e.g., config.yaml)
ESM2_PROTEIN_SEQ_COL = 'esm2_ready_seq' # Column with cleaned, ESM-2-ready protein seqs

total_timer = Timer()

# Local config (GPU-specific, keep here for now)
## CUDA_NAME = 'cuda:7'  # Specify GPU device

# Define paths - make configurable via command line or environment
parser = argparse.ArgumentParser(description='Compute ESM-2 embeddings for protein sequences')
parser.add_argument(
    '--input_file',
    type=str, default=None, 
    help='Path to input CSV file (e.g., protein_final.csv). If not provided, will be derived from config.'
)
parser.add_argument(
    '--output_dir',
    type=str, default=None,
    help='Path to output directory for embeddings. If not provided, will be derived from config.'
)
parser.add_argument(
    '--virus_name', '-v',
    type=str, default=None,
    help='Virus name to load selected_functions from config (e.g., flu_a, bunya)'
)
parser.add_argument(
    '--use_selected_only',
    action='store_true',
    help='Filter to selected_functions from config (requires --virus_name)'
)
parser.add_argument(
    '--cuda_name', '-c',
    type=str, default='cuda:7',
    help='CUDA device to use (default: cuda:7)'
)
args = parser.parse_args()

# Get config
config_path = str(project_root / 'conf') # Pass the config path explicitly
config = get_virus_config_hydra(args.virus_name, config_path=config_path)
print_config_summary(config)

# Extract configuration values (same pattern as preprocessing script)
# Handle embeddings seed using utility function
DATA_VERSION = config.virus.data_version
RUN_SUFFIX = config.run_suffix if config.run_suffix else ''
ESM2_MAX_RESIDUES = config.embeddings.esm2_max_residues
RANDOM_SEED = resolve_process_seed(config, 'embeddings')
CUDA_NAME = args.cuda_name
MODEL_CKPT = config.embeddings.model_ckpt
BATCH_SIZE = config.embeddings.batch_size

# Set deterministic seeds for ESM-2 computation
if RANDOM_SEED is not None:
    set_deterministic_seeds(RANDOM_SEED, cuda_deterministic=True)
    print(f'Set deterministic seeds for ESM-2 embeddings computation (seed: {RANDOM_SEED})')
else:
    print('No seed set - ESM-2 computation will be non-deterministic')

# Define paths
# Build run directory name from DATA_VERSION + optional RUN_SUFFIX
run_dir = f'{DATA_VERSION}{RUN_SUFFIX}'
print(f'\nRun directory: {run_dir}')
main_data_dir = project_root / 'data'

# Input: read from processed data
input_base_dir = main_data_dir / 'processed' / args.virus_name / run_dir
default_input_file = input_base_dir / 'protein_final.csv'

# Output: write to embeddings directory (matching the run_dir)
output_base_dir = main_data_dir / 'embeddings' / args.virus_name / run_dir
default_output_dir = output_base_dir

# Apply CLI overrides if provided
input_file = Path(args.input_file) if args.input_file else default_input_file
output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
output_dir.mkdir(parents=True, exist_ok=True)

device = determine_device(CUDA_NAME)

print(f'\ninput_file:      {input_file}')
print(f'output_dir:      {output_dir}')
print(f'device:          {device}')
print(f'model:           {MODEL_CKPT}')
print(f'batch_size:      {BATCH_SIZE}')


# Load protein data
print('\nLoad preprocessed protein data.')
try:
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df):,} protein records from {input_file}")
except Exception as e:
    raise RuntimeError(f"Error loading data from {input_file}: {e}")

# Validate required columns
required_cols = ['brc_fea_id', ESM2_PROTEIN_SEQ_COL, 'length']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Filter to selected proteins if configured
if args.use_selected_only:
    if args.virus_name is None:
        raise ValueError("--virus_name must be provided when --use_selected_only is specified")
    if 'function' not in df.columns:
        raise ValueError("DataFrame must contain 'function' column to filter selected proteins")
    
    # Load selected functions from config
    try:
        if 'selected_functions' in config.virus:
            selected_functions = config.virus.selected_functions
            print(f"Loaded {len(selected_functions)} selected functions from config for virus '{args.virus_name}':")
            for i, func in enumerate(selected_functions, 1):
                print(f"  {i}. {func}")
        else:
            raise ValueError(f"No 'selected_functions' found in config for virus '{args.virus_name}'")
    except Exception as e:
        raise ValueError(f"Failed to load config for virus '{args.virus_name}': {e}")
    
    print(f"Filter to selected proteins only: {len(selected_functions)} functions")
    original_count = len(df)
    df = df[df['function'].isin(selected_functions)].reset_index(drop=True)
    print(f"Filtered from {original_count} to {len(df)} selected proteins")
else:
    print("Use all proteins")


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
print(f'\nCompute ESM-2 embeddings ({MODEL_CKPT}).')
embeddings, brc_fea_ids, failed_ids = compute_esm2_embeddings(
    sequences=df[ESM2_PROTEIN_SEQ_COL].tolist(),
    brc_fea_ids=df['brc_fea_id'].tolist(),
    model_name=MODEL_CKPT,
    batch_size=BATCH_SIZE,
    device=device,
    max_length=ESM2_MAX_RESIDUES + 2
)


# Save embeddings to h5
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
        raise ValueError(f'Embeddings count mismatch: {len(saved_ids)} saved, {expected_count} expected')
    emb_dim = file[saved_ids[0]].shape[0]
    print(f'✅ Saved {len(saved_ids)} embeddings (dim: {emb_dim})')
if failed_ids:
    print(f'Failed to process {len(failed_ids)} sequences: {failed_ids[:5]}...')

total_timer.display_timer()
print(f'\nFinished {Path(__file__).name}!')
