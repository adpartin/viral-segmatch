"""
Precompute ESM-2 embeddings for protein sequences.
Virus-agnostic version with command-line interface.
"""
import argparse
import sys
from pathlib import Path

import torch # Preload heavy module
import h5py
import numpy as np
import pandas as pd
import psutil

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# print(f'project_root: {project_root}')

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds
from src.utils.path_utils import resolve_run_suffix, build_embeddings_paths, load_dataframe
from src.utils.protein_utils import STANDARD_AMINO_ACIDS
from src.utils.torch_utils import determine_device
from src.utils.esm2_utils import compute_esm2_embeddings

# Manual configs
# TODO: consider setting these somewhere else (e.g., config.yaml)
ESM2_PROTEIN_SEQ_COL = 'esm2_ready_seq' # Column with cleaned, ESM-2-ready protein seqs

total_timer = Timer()

def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"📊 Memory usage {stage}: {memory_mb:.1f} MB")

# Parser
parser = argparse.ArgumentParser(description='Compute ESM-2 embeddings for protein sequences')
parser.add_argument(
    '--config_bundle',
    type=str, default=None,
    help='Config bundle to use (e.g., flu_a, bunya).'
)
parser.add_argument(
    '--cuda_name', '-c',
    type=str, default='cuda:7',
    help='CUDA device to use (default: cuda:7)'
)
parser.add_argument(
    '--input_file',
    type=str, default=None, 
    help='Path to input CSV file (e.g., protein_final.csv). If not provided, derived from config.'
)
parser.add_argument(
    '--output_dir',
    type=str, default=None,
    help='Path to output dir for embeddings. If not provided, derived from config.'
)
parser.add_argument(
    '--force-recompute',
    action='store_true',
    help='Force recompute of embeddings, bypassing cache.'
)
args = parser.parse_args()

# Load config
config_path = str(project_root / 'conf') # Pass the config path explicitly
config_bundle = args.config_bundle
if config_bundle is None:
    raise ValueError("❌ Must provide --config_bundle")
config = get_virus_config_hydra(config_bundle, config_path=config_path)
print_config_summary(config)

# Extract config values
VIRUS_NAME = config.virus.virus_name
DATA_VERSION = config.virus.data_version
RANDOM_SEED = resolve_process_seed(config, 'embeddings')
# USE_SELECTED_ONLY = config.embeddings.use_selected_only
MAX_ISOLATES_TO_PROCESS = config.max_isolates_to_process
MODEL_CKPT = config.embeddings.model_ckpt
ESM2_MAX_RESIDUES = config.embeddings.esm2_max_residues
BATCH_SIZE = config.embeddings.batch_size
POOLING = config.embeddings.pooling
LAYER = config.embeddings.layer
EMB_STORAGE_PRECISION = config.embeddings.emb_storage_precision

print(f"\n{'='*40}")
print(f"Virus: {VIRUS_NAME}")
print(f"Config bundle: {config_bundle}")
print(f"{'='*40}")

# Resolve run suffix (manual override in config or auto-generate from sampling params)
RUN_SUFFIX = resolve_run_suffix(
    config=config,
    max_isolates=MAX_ISOLATES_TO_PROCESS,
    seed=RANDOM_SEED,
    auto_timestamp=True
)

# Set deterministic seeds for ESM-2 computation
if RANDOM_SEED is not None:
    set_deterministic_seeds(RANDOM_SEED, cuda_deterministic=True)
    print(f'Set deterministic seeds for ESM-2 embeddings computation (seed: {RANDOM_SEED})')
else:
    print('No seed set - ESM-2 embeddings computation will be non-deterministic')

# Build canonical paths (no suffix) for master cache (shared across runs)
canonical_paths = build_embeddings_paths(
    project_root=project_root,
    virus_name=VIRUS_NAME,
    data_version=DATA_VERSION,
    run_suffix="",  # Empty suffix = canonical location
    config=config
)

# Build embeddings paths (run-specific for metadata files)
paths = build_embeddings_paths(
    project_root=project_root,
    virus_name=VIRUS_NAME,
    data_version=DATA_VERSION,
    run_suffix=RUN_SUFFIX,
    config=config
)
default_input_file = paths['input_file']
default_output_dir = paths['output_dir']

# Apply CLI overrides if provided
input_file = Path(args.input_file) if args.input_file else default_input_file
output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
output_dir.mkdir(parents=True, exist_ok=True)

print(f'\nRun directory:         {DATA_VERSION}{RUN_SUFFIX}')
print(f'input_file:            {input_file}')
print(f'output_dir:            {output_dir}')
print(f'model:                 {MODEL_CKPT}')
print(f'batch_size:            {BATCH_SIZE}')
print(f'pooling:               {POOLING}')
print(f'layer:                 {LAYER}')
print(f'emb_storage_precision: {EMB_STORAGE_PRECISION}')

# Load protein data
total_timer = Timer()
print('\nLoad preprocessed protein sequence data.')
try:
    df = load_dataframe(input_file)
    print(f"Loaded {len(df):,} protein records")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Data file not found at: {input_file}")
except Exception as e:
    raise RuntimeError(f"❌ Error loading data from {input_file}: {e}")

# Validate required columns
required_cols = ['brc_fea_id', ESM2_PROTEIN_SEQ_COL, 'length']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required columns: {missing_cols}.")

# Check for duplicate IDs
# Duplicate brc_fea_id values would cause issues in the parquet index mapping
# (only the last occurrence would be mapped). This should be caught upstream
# in preprocessing, but we validate here as a safeguard.
if df['brc_fea_id'].duplicated().any():
    raise ValueError("❌ Duplicate brc_fea_id found in input data.")

# # Apply isolate-based sampling if specified
# if MAX_ISOLATES_TO_PROCESS:
#     print(f'\nSample {MAX_ISOLATES_TO_PROCESS} isolates from protein sequence data ({input_file.name}).')
#     # Get unique isolates (assembly_id)
#     unique_isolates = df['assembly_id'].unique()
#     print(f"Found {len(unique_isolates)} unique isolates.")

#     if len(unique_isolates) > MAX_ISOLATES_TO_PROCESS:
#         print(f"Sample {MAX_ISOLATES_TO_PROCESS} isolates from {len(unique_isolates)} total isolates.")
#         # Sample isolates, not individual records
#         sampled_isolates = np.random.choice(
#             unique_isolates, 
#             size=MAX_ISOLATES_TO_PROCESS, 
#             replace=False
#         )
#         # Extract protein records based on the sampled isolates
#         df = df[df['assembly_id'].isin(sampled_isolates)].reset_index(drop=True)
#         print(f"Extracted {len(df)} protein records based on the {len(sampled_isolates)} sampled isolates.")

#         # Save sampled isolates for dataset script
#         with open(output_dir / 'sampled_isolates.txt', 'w') as f:
#             for isolate in sorted(sampled_isolates):
#                 f.write(f"{isolate}\n")
#         print(f"Saved list of {len(sampled_isolates)} sampled isolates to sampled_isolates.txt")
#     else:
#         print(f"Use all {len(unique_isolates)} isolates (isolates required to sample > isolates available, no sampling needed).")
#         print(f"Total protein records: {len(df)}")
# else:
#     print(f"Use all {len(df)} records from all isolates (no sampling needed)")
print(f"Use all {len(df)} records from all isolates (sampling disabled in embeddings stage)")

# # Filter to selected proteins if configured
# # breakpoint()
# if USE_SELECTED_ONLY:
#     if 'function' not in df.columns:
#         raise ValueError("❌ DataFrame must contain 'function' column to filter selected protein records.")

#     # Load selected functions from config
#     try:
#         if 'selected_functions' in config.virus:
#             selected_functions = config.virus.selected_functions
#             print(f"\nRetrieved {len(selected_functions)} selected functions from config for virus '{VIRUS_NAME}':")
#             for i, func in enumerate(selected_functions, 1):
#                 print(f"  {i}. {func}")
#         else:
#             raise ValueError(f"❌ No 'selected_functions' found in config for virus '{VIRUS_NAME}'.")
#     except Exception as e:
#         raise ValueError(f"❌ Failed to load config for virus '{VIRUS_NAME}': {e}")

#     print(f"Filter protein records based on {len(selected_functions)} selected functions.")
#     original_count = len(df)
#     df = df[df['function'].isin(selected_functions)].reset_index(drop=True)
#     print(f"Filtered {len(df)} protein records from {original_count} based on selected functions.")
# else:
#     print("Use all proteins.")

# Note: Embeddings are computed for ALL proteins in the dataset.
# USE_SELECTED_ONLY and MAX_ISOLATES_TO_PROCESS toggles are intentionally disabled here;
# dataset creation handles any filtering or sampling so the master cache stays universal.
print("Computing embeddings for all proteins (embedding stage ignores selected-only & isolate limits).")


# Validate and deduplicate
# Note: Validation warnings are still useful even though we don't filter here.
# The HDF5 structure handles sequence-level deduplication, but we still want to:
# 1. Warn about invalid sequences (non-standard amino acids)
# 2. Warn about sequences exceeding ESM-2 max length
# 3. Remove exact duplicate records (same brc_fea_id + same sequence) before processing
print(f"\n{'='*50}")
print('Validate protein data.')
print('='*50)

# Filter out rows with missing/None sequences FIRST (defensive check)
# Note: Bad sequences should already be filtered during preprocessing, but this handles:
# 1. Legacy data files that may contain None sequences
# 2. Edge cases where sequences failed ESM-2 preparation
# Must be done before other validation checks to avoid errors
n_before_missing_filter = len(df)
df = df[df[ESM2_PROTEIN_SEQ_COL].notna()].reset_index(drop=True)
if len(df) < n_before_missing_filter:
    n_missing = n_before_missing_filter - len(df)
    print(f'⚠️ Dropped {n_missing} records with missing/None sequences (should be rare - sequences are filtered during preprocessing)')

# Check for invalid sequences (non-standard amino acids)
standard_aa_pattern = ''.join(STANDARD_AMINO_ACIDS)
invalid_seqs = df[df[ESM2_PROTEIN_SEQ_COL].str.contains(f'[^{standard_aa_pattern}]', na=False)]
# TODO: should we filter out invalid sequences??
if not invalid_seqs.empty:
    print(f'⚠️ {len(invalid_seqs)} invalid sequences found:')
    print(invalid_seqs[['brc_fea_id', ESM2_PROTEIN_SEQ_COL]].head())

# Identify proteins longer than ESM-2 max length
truncated_seqs = df[df['length'] > ESM2_MAX_RESIDUES][['brc_fea_id', 'canonical_segment', 'length']]
if not truncated_seqs.empty:
    print(f'⚠️ {len(truncated_seqs)} sequences truncated (>{ESM2_MAX_RESIDUES} residues):')
    print(truncated_seqs.head())

# Remove exact duplicate records (same brc_fea_id + same sequence)
# Note: This removes data quality issues (duplicate rows), NOT biological duplicates.
# Biological duplicates (same sequence, different brc_fea_id) are preserved and handled
# by the master cache's sequence-based deduplication.
n_org_proteins = len(df)
df = df[['brc_fea_id', ESM2_PROTEIN_SEQ_COL]].drop_duplicates().reset_index(drop=True)
if len(df) < n_org_proteins:
    print(f'Dropped {n_org_proteins - len(df)} duplicate records (same brc_fea_id + sequence)')
print(f'Retained {len(df)} unique protein records.')
log_memory_usage("after protein data loaded")


# Master cache file (per virus and data version, shared across runs)
# Derived from canonical paths (no suffix) to ensure sharing across all runs
master_embeddings_file = canonical_paths['output_dir'] / 'master_esm2_embeddings.h5'
master_embeddings_file.parent.mkdir(parents=True, exist_ok=True)
print(f"\nMaster cache file: {master_embeddings_file}")
print(f"  (Shared across all runs for {VIRUS_NAME}/{DATA_VERSION})")

# Compute embeddings (with master cache support)
# breakpoint()
print(f"\n{'='*50}")
print(f'Compute ESM-2 embeddings ({MODEL_CKPT}).')
print('='*50)
CUDA_NAME = args.cuda_name  # CUDA device
device = determine_device(CUDA_NAME)
log_memory_usage("before embeddings computation")
comp_timer = Timer()
embeddings, brc_fea_ids, failed_ids = compute_esm2_embeddings(
    sequences=df[ESM2_PROTEIN_SEQ_COL].tolist(),
    brc_fea_ids=df['brc_fea_id'].tolist(),
    embeddings_file=str(master_embeddings_file),
    model_name=MODEL_CKPT,
    batch_size=BATCH_SIZE,
    device=device,
    max_length=ESM2_MAX_RESIDUES + 2,
    force_recompute=args.force_recompute,
    use_parquet=True,
    pooling=POOLING,
    layer=LAYER,
    emb_storage_precision=EMB_STORAGE_PRECISION
)
comp_timer.stop_timer()
print(f"  ⏱️  Computation time: {comp_timer.get_elapsed_string()}")
log_memory_usage("after embeddings computation")

# Note: Embeddings are now saved directly to master cache during computation
# The old per-run HDF5 files are no longer needed


# Failed ids to csv (save to master cache directory for shared tracking)
# Failures are typically sequence-specific and consistent across runs,
# so we track them in the shared master cache directory
failed_df = pd.DataFrame({'brc_fea_id': failed_ids})
failed_file = master_embeddings_file.parent / 'failed_brc_fea_ids.csv'

if failed_ids:
    if failed_file.exists():
        # Merge with existing failures (avoid duplicates)
        existing_failed = pd.read_csv(failed_file)
        combined_failed = pd.concat([existing_failed, failed_df]).drop_duplicates().reset_index(drop=True)
        combined_failed.to_csv(failed_file, index=False)
        print(f"\nUpdated failed IDs file: {len(combined_failed)} total failures ({len(failed_df)} new)")
    else:
        failed_df.to_csv(failed_file, index=False)
        print(f"\nCreated failed IDs file: {len(failed_df)} failures")
else:
    print("\nNo failed sequences to track.")


# Validate master cached
print('\nValidate master cache.')
if master_embeddings_file.exists():
    with h5py.File(master_embeddings_file, 'r') as f:
        if 'emb' in f and 'emb_keys' in f:
            n_embeddings = f['emb'].shape[0]
            n_keys = f['emb_keys'].shape[0]
            emb_dim = f['emb'].shape[1]
            print(f'Master cache contains {n_embeddings} embeddings (dim: {emb_dim})')
            print(f'Keys: {n_keys}')

            # Print metadata attributes
            if f.attrs:
                print('📋 Metadata attributes:')
                for key, value in f.attrs.items():
                    print(f'   {key}: {value}')
        else:
            print('⚠️ Master cache exists but missing required datasets')
else:
    print('⚠️ Master cache file not found')
if failed_ids:
    print(f'⚠️ Failed to process {len(failed_ids)} sequences: {failed_ids[:5]}...')

print(f'\n✅ Finished {Path(__file__).name}!')
total_timer.display_timer()
