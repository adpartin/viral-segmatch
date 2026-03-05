"""
Compute k-mer frequency features for genome segments (Stage 2b).

Input:  genome_final.csv (all segments — master cache pattern)
Output: kmer_features_k{k}.npz        — scipy sparse CSR matrix, shape (N_segments, 4^k)
        kmer_features_k{k}_index.parquet — maps (assembly_id, genbank_ctg_id, canonical_segment) → row index
        kmer_features_k{k}_metadata.json — k, normalize, n_sequences, vocab_size, timestamp

CPU-only; no GPU needed. Simple existence-check caching (recompute with --force-recompute).

Example:
```bash
python -m ipdb src/embeddings/compute_kmer_features.py \
    --config_bundle flu_debug \
    --input_file data/processed/flu/July_2025_debug/genome_final.csv \
    --force_recompute
or
run using ./scripts/stage2b_kmer.sh
```
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
from src.utils.path_utils import resolve_run_suffix, build_embeddings_paths, load_dataframe

DNA_SEQ_COL_NAME = 'dna_seq'

total_timer = Timer()


# =============================================================================
# Core k-mer functions
# =============================================================================

def build_kmer_vocabulary(k: int) -> list[str]:
    """Return sorted list of all 4^k k-mers (ACGT alphabet)."""
    return [''.join(bases) for bases in product('ACGT', repeat=k)]


def compute_kmer_counts(seq: str, k: int) -> Counter:
    """Count k-mers in *seq*, skipping windows that contain non-ACGT characters."""
    counts = Counter()
    seq_upper = seq.upper()
    valid = set('ACGT')
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i + k]
        if all(c in valid for c in kmer):
            counts[kmer] += 1
    return counts


def sequences_to_sparse_kmer_matrix(
    sequences: list[str],
    k: int,
    normalize: str = 'none',
    ) -> sparse.csr_matrix:
    """Convert a list of DNA sequences to a sparse CSR k-mer count matrix.

    Each sequence becomes one row. Each column corresponds to one of the 4^k
    possible k-mers (sorted lexicographically: AAA...A, AAA...C, ..., TTT...T).
    Cell values are raw occurrence counts (before normalization).

    Example (k=2, vocab = [AA, AC, AG, AT, CA, CC, ..., TT] — 16 columns):

        sequences = ["AACG", "TTAC"]

        "AACG" has 2-mers: AA(1), AC(1), CG(1)  ->  [1,1,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]
        "TTAC" has 2-mers: TT(1), TA(1), AC(1)  ->  [0,1,0,0, 0,0,0,0, 0,0,0,0, 1,0,1,1]

        Output matrix shape: (2, 16) — 2 sequences x 4^2 k-mers

    For the actual pipeline (k=6): each row has 4^6 = 4096 columns. Most cells
    are zero (a ~2 kb flu segment has ~2000 6-mers out of 4096 possible), so a
    sparse representation is efficient.

    Args:
        sequences: list of DNA strings
        k: k-mer size (default 6 -> 4096-dim; k=10 -> 1M-dim, must be sparse)
        normalize: 'none' (raw counts), 'l1' (row sums to 1), 'l2' (unit L2 norm)

    Returns:
        scipy CSR matrix of shape (len(sequences), 4^k)
    """
    # Build vocabulary: sorted list of all 4^k k-mers and their column indices
    vocab = build_kmer_vocabulary(k)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(vocab)}
    vocab_size = len(vocab)

    # Accumulate COO-format triplets (row, col, value) for sparse matrix construction
    rows, cols, data = [], [], []
    for row_i, seq in enumerate(tqdm(sequences, desc=f'Computing {k}-mer counts')):
        counts = compute_kmer_counts(seq, k)
        for kmer, cnt in counts.items():
            if kmer in kmer_to_idx:
                rows.append(row_i)              # sequence index
                cols.append(kmer_to_idx[kmer])  # k-mer column index
                data.append(cnt)                # occurrence count

    # Build sparse matrix from COO triplets, then convert to CSR for efficient row slicing
    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(sequences), vocab_size),
        dtype=np.float32,
    )

    # Optional row-wise normalization
    if normalize == 'l1':
        # Divide each row by its sum -> frequencies (row sums to 1.0)
        row_sums = np.array(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # avoid division by zero for empty sequences
        mat = sparse.diags(1.0 / row_sums) @ mat
    elif normalize == 'l2':
        # Divide each row by its L2 norm -> unit vectors
        norms = sparse.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        mat = sparse.diags(1.0 / norms) @ mat
    elif normalize != 'none':
        raise ValueError(f"Unknown normalize mode: {normalize!r}")

    return mat


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description='Compute k-mer features for genome segments')
parser.add_argument(
    '--config_bundle', type=str, required=True,
    help='Config bundle to use (e.g., flu).'
)
parser.add_argument(
    '--input_file', type=str, default=None,
    help='Path to genome_final.csv. If not provided, derived from config.'
)
parser.add_argument(
    '--output_dir', type=str, default=None,
    help='Path to output dir. If not provided, derived from config.'
)
parser.add_argument(
    '--force_recompute', action='store_true',
    help='Force recompute, bypassing cache.'
)
args = parser.parse_args()

# Load config
config_path = str(project_root / 'conf')
config = get_virus_config_hydra(args.config_bundle, config_path=config_path)
print_config_summary(config)

VIRUS_NAME = config.virus.virus_name
DATA_VERSION = config.virus.data_version

# K-mer config (use defaults if kmer group not in bundle)
from omegaconf import OmegaConf
if OmegaConf.is_missing(config, 'kmer') or not hasattr(config, 'kmer') or config.get('kmer') is None:
    K = 6
    NORMALIZE = 'none'
else:
    K = int(config.kmer.get('k', 6))
    NORMALIZE = str(config.kmer.get('normalize', 'none'))

print(f"\n{'='*40}")
print(f"Stage 2b: K-mer Features")
print(f"Virus: {VIRUS_NAME}")
print(f"Config bundle: {args.config_bundle}")
print(f"k: {K}  (vocab size: {4**K:,})")
print(f"normalize: {NORMALIZE}")
print(f"{'='*40}")

# Build paths (reuse embeddings path structure — output goes alongside ESM-2 cache)
canonical_paths = build_embeddings_paths(
    project_root=project_root,
    virus_name=VIRUS_NAME,
    data_version=DATA_VERSION,
    run_suffix="",
    config=config,
)

# Input: genome_final.csv (same directory as protein_final.csv)
if args.input_file:
    input_file = Path(args.input_file)
else:
    # protein_final.csv path → replace with genome_final.csv
    input_file = canonical_paths['input_file'].parent / 'genome_final.csv'

# Output directory: same as embeddings output
output_dir = Path(args.output_dir) if args.output_dir else canonical_paths['output_dir']
output_dir.mkdir(parents=True, exist_ok=True)

# Output file names
npz_file = output_dir / f'kmer_features_k{K}.npz'
index_file = output_dir / f'kmer_features_k{K}_index.parquet'
meta_file = output_dir / f'kmer_features_k{K}_metadata.json'

print(f'\ninput_file:  {input_file}')
print(f'output_dir: {output_dir}')
print(f'npz_file:   {npz_file}')

# Cache check
if npz_file.exists() and index_file.exists() and not args.force_recompute:
    print(f'\nK-mer features already exist at: {npz_file}')
    print('Use --force-recompute to regenerate.')
    mat = sparse.load_npz(npz_file)
    print(f'Cached matrix shape: {mat.shape}')
    sys.exit(0)

# Load genome data
print(f'\nLoad genome data from: {input_file}')
df = load_dataframe(input_file)
print(f'Loaded {len(df):,} genome records')

# Validate required columns
required_cols = ['assembly_id', 'genbank_ctg_id', 'canonical_segment', DNA_SEQ_COL_NAME]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Drop rows with missing sequences
n_before = len(df)
df = df[df[DNA_SEQ_COL_NAME].notna()].reset_index(drop=True)
if len(df) < n_before:
    print(f'Dropped {n_before - len(df)} rows with missing DNA sequences')

print(f'Unique assemblies: {df["assembly_id"].nunique():,}')
print(f'Unique segments:   {df["canonical_segment"].nunique()}')
print(f'Total records:     {len(df):,}')

# Compute k-mer features
print(f'\nCompute {K}-mer features for {len(df):,} sequences ...')
comp_timer = Timer()
mat = sequences_to_sparse_kmer_matrix(
    sequences=df[DNA_SEQ_COL_NAME].tolist(),
    k=K,
    normalize=NORMALIZE,
)
comp_timer.stop_timer()
print(f'Matrix shape: {mat.shape}  (nnz: {mat.nnz:,})')
print(f'Computation time: {comp_timer.get_elapsed_string()}')

# Save sparse matrix
print(f'\nSave k-mer features to: {npz_file}')
sparse.save_npz(npz_file, mat)

# Save index (row → identifiers)
index_df = pd.DataFrame({
    'assembly_id': df['assembly_id'].values,
    'genbank_ctg_id': df['genbank_ctg_id'].values,
    'canonical_segment': df['canonical_segment'].values,
    'row': np.arange(len(df)),
})
index_df.to_parquet(index_file, index=False)
print(f'Saved index to: {index_file}')

# Save metadata
metadata = {
    'k': K,
    'normalize': NORMALIZE,
    'n_sequences': len(df),
    'vocab_size': 4 ** K,
    'nnz': int(mat.nnz),
    'sparsity': 1.0 - mat.nnz / (mat.shape[0] * mat.shape[1]),
    'timestamp': datetime.now().isoformat(),
    'input_file': str(input_file),
}
with open(meta_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'Saved metadata to: {meta_file}')

# Validate
print(f'\nValidation:')
print(f'  Matrix shape:     {mat.shape}')
print(f'  Index rows:       {len(index_df):,}')
print(f'  Shape match:      {mat.shape[0] == len(index_df)}')
print(f'  Vocab size (4^{K}): {4**K:,}')

print(f'\nDone. Finished {Path(__file__).name}.')
total_timer.display_timer()
