"""
Compute k-mer frequency features for biological sequences (Stage 2b).

Dual-alphabet pipeline. The alphabet is driven by `kmer.alphabet` in the
Hydra config (default `nt`):

  nt: reads `genome_final.csv → dna_seq`, vocab `ACGT`,
      output `kmer_features_nt_k{k}.*`,
      parquet keys `(assembly_id, genbank_ctg_id)` per row (one row per contig).
  aa: reads `protein_final.csv → prot_seq`, vocab `ACDEFGHIKLMNPQRSTVWY`,
      output `kmer_features_aa_k{k}.*`,
      parquet keys `(assembly_id, brc_fea_id)` per row, with N-to-1
      mapping into a sequence-deduplicated matrix (one row per unique
      `prot_seq`).

Outputs (per alphabet/k):
  kmer_features_{alphabet}_k{k}.npz        scipy sparse CSR
  kmer_features_{alphabet}_k{k}_index.parquet
        nt: assembly_id, genbank_ctg_id, canonical_segment, row
        aa: assembly_id, brc_fea_id, canonical_segment, function, cache_key, row
  kmer_features_{alphabet}_k{k}_metadata.json   k, normalize, alphabet,
        vocab_size, n_index_rows, n_unique_sequences, nnz, sparsity, ...

CPU-only; no GPU. Simple existence-check caching (recompute with
`--force_recompute`). See
`docs/plans/2026-05-13_aa_kmer_and_cache_symmetry_plan.md` for the
cache-symmetry design.

Example:
```bash
./scripts/stage2b_kmer.sh --config_bundle flu_ha_na_kmer_aa_k3
# or directly:
python src/embeddings/compute_kmer_features.py \
    --config_bundle flu_ha_na_kmer_aa_k3 --force_recompute
```
"""
import argparse
import hashlib
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

NT_ALPHABET = 'ACGT'
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

total_timer = Timer()


# =============================================================================
# Core k-mer functions
# =============================================================================

def build_kmer_vocabulary(k: int, alphabet: str = 'ACGT') -> list[str]:
    """Return sorted list of all len(alphabet)^k k-mers over `alphabet`."""
    return [''.join(bases) for bases in product(alphabet, repeat=k)]


def compute_kmer_counts(seq: str, k: int, alphabet: str = 'ACGT') -> Counter:
    """Count k-mers in *seq*, skipping any window with a character outside `alphabet`.

    The sequence is uppercased before scanning, so `alphabet` should be
    uppercase (canonical DNA 'ACGT' or canonical 20-AA protein
    'ACDEFGHIKLMNPQRSTVWY'). Gaps, IUPAC ambiguity codes, and any other
    non-alphabet character disqualify the containing k-mer.
    """
    counts = Counter()
    seq_upper = seq.upper()
    valid = set(alphabet)
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i + k]
        if all(c in valid for c in kmer):
            counts[kmer] += 1
    return counts


def sequences_to_sparse_kmer_matrix(
    sequences: list[str],
    k: int,
    normalize: str = 'none',
    alphabet: str = 'ACGT',
    ) -> sparse.csr_matrix:
    """Convert a list of biological sequences to a sparse CSR k-mer count matrix.

    Each sequence becomes one row. Columns enumerate **all** ``len(alphabet)^k``
    possible k-mers, sorted lexicographically (e.g., for DNA k=3:
    AAA, AAC, AAG, ..., TTT). Cell values are raw occurrence counts before
    optional normalization.

    Why a fixed ``len(alphabet)^k`` vocabulary (vs data-driven from observed k-mers):
      Columns are aligned across runs, datasets, and splits with zero
      handshake. Two matrices computed at different times can be merged
      row-wise or compared column-by-column without remapping. The cost is
      that unobserved-k-mer columns hold zeros — but CSR makes that
      essentially free (only non-zero cells occupy memory). The alternative
      (only columns for observed k-mers) is smaller per matrix but breaks
      cross-dataset alignment.

    Why float32 cells:
      - int8 overflows at 127; common k-mers in long sequences can exceed
        that. int16 would be safe for raw counts.
      - Normalization (``'l1'``, ``'l2'``) produces fractional values, so
        float is required regardless. Picking float32 up front avoids a
        dtype switch later.
      - Downstream consumers (sklearn, LightGBM, scipy) expect float
        matrices; conversion would happen anyway.
      - Storage: CSR only counts non-zeros, so the 4-byte/cell vs
        2-byte/cell delta vs int16 is negligible at our scale.

    Alphabet (DNA vs protein):
      - DNA (default ``'ACGT'``): 4^k columns. Standard for genome k-mers.
      - Protein (canonical 20-AA: ``'ACDEFGHIKLMNPQRSTVWY'``): 20^k columns.
        Practical only up to k≈4 (160K cols). At k=5 (3.2M cols) the
        exhaustive vocabulary becomes expensive to build (Python iteration
        over ``product``) and the matrix gains many always-zero columns;
        observed-vocabulary or feature-hashing approaches are more
        appropriate at that scale.
      Custom alphabets should be uppercase (sequences are uppercased
      internally). Duplicate characters or empty alphabet → ValueError.

    Example (DNA, k=2, vocab = [AA, AC, AG, AT, ..., TT] — 16 columns):
        sequences = ["AACG", "TTAC"]
        "AACG" 2-mers: AA(1), AC(1), CG(1)  ->  [1,1,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]
        "TTAC" 2-mers: TT(1), TA(1), AC(1)  ->  [0,1,0,0, 0,0,0,0, 0,0,0,0, 1,0,1,1]
        Output matrix shape: (2, 16) — 2 sequences x 4^2 k-mers

    Args:
        sequences: list of biological sequence strings (uppercased internally).
        k: k-mer size. For DNA: k=6 -> 4096-dim; k=10 -> 1M-dim (sparse only).
            For protein: k=3 -> 8000-dim; k=4 -> 160K-dim (sparse strongly
            recommended beyond k=4).
        normalize: 'none' (raw counts), 'l1' (row sums to 1), 'l2' (unit L2).
        alphabet: characters defining the vocabulary; default ``'ACGT'`` (DNA).
            For canonical protein k-mers pass ``'ACDEFGHIKLMNPQRSTVWY'``.

    Returns:
        scipy CSR matrix of shape (len(sequences), len(alphabet) ** k).
    """
    if not alphabet:
        raise ValueError("alphabet must be non-empty")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError(f"alphabet has duplicate characters: {alphabet!r}")

    # Build vocabulary: sorted list of all |alphabet|^k k-mers and their column indices.
    vocab = build_kmer_vocabulary(k, alphabet=alphabet)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(vocab)}
    vocab_size = len(vocab)

    # Accumulate COO-format triplets (row, col, value) for sparse matrix construction
    rows, cols, data = [], [], []
    for row_i, seq in enumerate(tqdm(sequences, desc=f'Computing {k}-mer counts')):
        counts = compute_kmer_counts(seq, k, alphabet=alphabet)
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
    ALPHABET_CHOICE = 'nt'
else:
    K = int(config.kmer.get('k', 6))
    NORMALIZE = str(config.kmer.get('normalize', 'none'))
    ALPHABET_CHOICE = str(config.kmer.get('alphabet', 'nt')).lower()

if ALPHABET_CHOICE == 'nt':
    VOCAB_ALPHABET = NT_ALPHABET
    SEQ_COL = 'dna_seq'
    OCCURRENCE_COL = 'genbank_ctg_id'
    INPUT_BASENAME = 'genome_final.csv'
elif ALPHABET_CHOICE == 'aa':
    VOCAB_ALPHABET = AA_ALPHABET
    SEQ_COL = 'prot_seq'
    OCCURRENCE_COL = 'brc_fea_id'
    INPUT_BASENAME = 'protein_final.csv'
else:
    raise ValueError(f"kmer.alphabet must be 'nt' or 'aa'; got {ALPHABET_CHOICE!r}")

VOCAB_SIZE = len(VOCAB_ALPHABET) ** K

print(f"\n{'='*40}")
print(f"Stage 2b: K-mer Features")
print(f"Virus: {VIRUS_NAME}")
print(f"Config bundle: {args.config_bundle}")
print(f"alphabet: {ALPHABET_CHOICE}  ({VOCAB_ALPHABET!r})")
print(f"k: {K}  (vocab size: {VOCAB_SIZE:,})")
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

# Input: protein_final.csv (for aa) or genome_final.csv (for nt) in the
# processed/ directory.
if args.input_file:
    input_file = Path(args.input_file)
else:
    input_file = canonical_paths['input_file'].parent / INPUT_BASENAME

# Output directory: same as embeddings output
output_dir = Path(args.output_dir) if args.output_dir else canonical_paths['output_dir']
output_dir.mkdir(parents=True, exist_ok=True)

# Output file names — alphabet tag in the filename keeps nt and aa caches separable.
npz_file = output_dir / f'kmer_features_{ALPHABET_CHOICE}_k{K}.npz'
index_file = output_dir / f'kmer_features_{ALPHABET_CHOICE}_k{K}_index.parquet'
meta_file = output_dir / f'kmer_features_{ALPHABET_CHOICE}_k{K}_metadata.json'

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

# Load source data
print(f'\nLoad source data from: {input_file}')
df = load_dataframe(input_file)
print(f'Loaded {len(df):,} records')

# Validate required columns
required_cols = ['assembly_id', OCCURRENCE_COL, 'canonical_segment', SEQ_COL]
if ALPHABET_CHOICE == 'aa' and 'function' in df.columns:
    required_cols.append('function')  # carried into the parquet for downstream use
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Drop rows with missing sequences
n_before = len(df)
df = df[df[SEQ_COL].notna()].reset_index(drop=True)
if len(df) < n_before:
    print(f'Dropped {n_before - len(df)} rows with missing {SEQ_COL}')

print(f'Unique assemblies: {df["assembly_id"].nunique():,}')
print(f'Unique segments:   {df["canonical_segment"].nunique()}')
print(f'Total records:     {len(df):,}')
print(f'Unique sequences:  {df[SEQ_COL].nunique():,}')

# Compute k-mer features.
#
# nt path: one row per occurrence (no sequence-level dedup yet — see
#   Phase 6 in the cache-symmetry plan).
# aa path: dedup by md5(prot_seq). Matrix has one row per unique
#   sequence; parquet maps each occurrence (assembly_id, brc_fea_id) to
#   the corresponding row, allowing N-to-1.
comp_timer = Timer()
if ALPHABET_CHOICE == 'aa':
    # Canonicalize once for hashing (uppercase, no whitespace). The k-mer
    # function uppercases internally; this matches that.
    seqs_canon = df[SEQ_COL].astype(str).str.upper().str.strip()
    cache_keys = seqs_canon.map(lambda s: hashlib.md5(s.encode('ascii', 'replace')).hexdigest())
    df = df.assign(_cache_key=cache_keys.values, _seq_canon=seqs_canon.values)

    unique_df = df.drop_duplicates(subset='_cache_key', keep='first').reset_index(drop=True)
    print(f'\nDedup by md5({SEQ_COL}): {len(df):,} occurrences -> {len(unique_df):,} unique sequences '
          f'({len(unique_df) / max(len(df), 1):.1%} retention)')

    print(f'Compute {K}-mer features over {VOCAB_ALPHABET!r} for {len(unique_df):,} unique sequences ...')
    mat = sequences_to_sparse_kmer_matrix(
        sequences=unique_df['_seq_canon'].tolist(),
        k=K,
        normalize=NORMALIZE,
        alphabet=VOCAB_ALPHABET,
    )
    cache_key_to_row = dict(zip(unique_df['_cache_key'], np.arange(len(unique_df))))
    df['row'] = df['_cache_key'].map(cache_key_to_row).astype(int)
    n_index_rows = len(df)
    n_unique = len(unique_df)
else:
    print(f'\nCompute {K}-mer features over {VOCAB_ALPHABET!r} for {len(df):,} sequences ...')
    mat = sequences_to_sparse_kmer_matrix(
        sequences=df[SEQ_COL].tolist(),
        k=K,
        normalize=NORMALIZE,
        alphabet=VOCAB_ALPHABET,
    )
    df['row'] = np.arange(len(df))
    n_index_rows = len(df)
    n_unique = len(df)  # not deduped in this path

comp_timer.stop_timer()
print(f'Matrix shape: {mat.shape}  (nnz: {mat.nnz:,})')
print(f'Computation time: {comp_timer.get_elapsed_string()}')

# Save sparse matrix
print(f'\nSave k-mer features to: {npz_file}')
sparse.save_npz(npz_file, mat)

# Save index (one row per occurrence). The (assembly_id, OCCURRENCE_COL) pair
# is the public lookup key; multiple occurrences can map to the same matrix
# `row` in the aa path.
index_cols = ['assembly_id', OCCURRENCE_COL, 'canonical_segment']
if ALPHABET_CHOICE == 'aa':
    if 'function' in df.columns:
        index_cols.append('function')
    index_cols.append('_cache_key')
index_df = df[index_cols + ['row']].copy()
if ALPHABET_CHOICE == 'aa':
    index_df = index_df.rename(columns={'_cache_key': 'cache_key'})
index_df.to_parquet(index_file, index=False)
print(f'Saved index to: {index_file}')

# Save metadata
metadata = {
    'k': K,
    'normalize': NORMALIZE,
    'alphabet': ALPHABET_CHOICE,
    'vocab_alphabet': VOCAB_ALPHABET,
    'vocab_size': VOCAB_SIZE,
    'n_index_rows': n_index_rows,
    'n_unique_sequences': n_unique,
    'matrix_rows': int(mat.shape[0]),
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
print(f'  Matrix shape:        {mat.shape}')
print(f'  Index rows:          {len(index_df):,}')
print(f'  Unique sequences:    {n_unique:,}')
print(f'  Matrix rows match:   {mat.shape[0] == n_unique}')
print(f'  Vocab size ({len(VOCAB_ALPHABET)}^{K}): {VOCAB_SIZE:,}')

print(f'\nDone. Finished {Path(__file__).name}.')
total_timer.display_timer()
