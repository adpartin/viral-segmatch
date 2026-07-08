"""Verify the across-cluster OOD guarantee for a single-segment cluster set.

Given an OOD cluster parquet (`<hash>`, `cluster_id`) built by
`build_mmseqs_clusters.py` with `--cluster-mode 1`, run an exhaustive mmseqs
all-vs-all over the same unique-sequence FASTA and confirm:

    no pair of sequences in DIFFERENT clusters has identity >= t AND
    coverage >= c   (the "across: different" guarantee)

The all-vs-all reuses the clustering's own identity/coverage rule
(`--min-seq-id`, `-c`, `--cov-mode`, `--seq-id-mode`), so the check is
self-consistent with how the clusters were defined; `--exhaustive-search`
removes the k-mer prefilter so no qualifying pair is missed. Every returned
hit therefore already meets the rule, and any hit whose two endpoints fall in
different clusters is a genuine violation. Exit code 0 iff zero violations.

`--sensitivity <s>` swaps `--exhaustive-search` for `-s <s>` (pair with a high
`--max_seqs`) as an independent cross-check: the violation count should match.

Example:
    python -m src.analysis.verify_ood_clusters \\
        --cluster_parquet data/processed/flu/July_2025/clusters_aa_ood/t099/M1_cluster.parquet \\
        --fasta           data/processed/flu/July_2025/clusters_aa_ood/fasta/M1.fasta \\
        --threshold 0.99 \\
        --mmseqs_bin /homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils import schema  # noqa: E402


def load_hash_to_cluster(cluster_parquet: Path) -> tuple:
    """Load (`<hash>` -> `cluster_id`) from an OOD cluster parquet.

    Returns (mapping dict, alphabet, cluster_sizes). The hash column name is
    alphabet-specific (from the schema registry); the alphabet is read from the
    parquet's `alphabet` column.
    """
    df = pd.read_parquet(cluster_parquet)
    alphabet = str(df['alphabet'].iloc[0])
    if alphabet not in schema.ALPHABETS:
        raise ValueError(f"unknown alphabet {alphabet!r} in {cluster_parquet}")
    hash_col = schema.SCHEMA[alphabet].hash_col
    if hash_col not in df.columns:
        raise KeyError(f"expected hash column {hash_col!r} in {cluster_parquet}; "
                       f"got {df.columns.tolist()}")
    mapping = dict(zip(df[hash_col], df['cluster_id']))
    cluster_sizes = df['cluster_id'].value_counts()
    return mapping, alphabet, cluster_sizes


def run_all_vs_all(fasta: Path, out_tsv: Path, tmp_dir: Path, *, alphabet: str,
                   threshold: float, coverage: float, cov_mode: int,
                   seq_id_mode: int, sensitivity, max_seqs, gpu: int,
                   mmseqs_bin: str, threads) -> list:
    """Run mmseqs all-vs-all (FASTA vs itself) under the clustering's rule.

    Writes a headerless TSV: query, target, fident, qcov, tcov. Uses
    `--exhaustive-search` when `sensitivity` is None, else `-s <sensitivity>`.
    Every returned hit already meets identity >= `threshold` and coverage >=
    `coverage`. Returns the command (for logging).
    """
    cmd = [
        mmseqs_bin, 'easy-search',
        str(fasta), str(fasta), str(out_tsv), str(tmp_dir),
        '--min-seq-id', f'{threshold:g}',
        '-c', f'{coverage:g}',
        '--cov-mode', str(cov_mode),
        '--seq-id-mode', str(seq_id_mode),
        '-e', '0.001',
        '--format-output', 'query,target,fident,qcov,tcov',
    ]
    if sensitivity is None:
        cmd += ['--exhaustive-search', '1']
    else:
        cmd += ['-s', f'{sensitivity:g}']
    if max_seqs is not None:
        cmd += ['--max-seqs', str(max_seqs)]
    if gpu:
        cmd += ['--gpu', str(gpu), '--createdb-mode', '2']
    if threads is not None:
        cmd += ['--threads', str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mmseqs easy-search failed (returncode={proc.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"{(proc.stdout or '')[-2000:]}{(proc.stderr or '')[-2000:]}"
        )
    return cmd


def find_violations(hits: pd.DataFrame, hash_to_cluster: dict) -> pd.DataFrame:
    """Cross-cluster hits are the across-different violations.

    Drops self-hits, maps both endpoints to `cluster_id`, keeps pairs whose
    endpoints are in different clusters, and dedups unordered pairs. Every input
    hit already meets the identity/coverage rule (applied in the search), so any
    cross-cluster hit is a genuine violation.
    """
    h = hits[hits['query'] != hits['target']].copy()
    h['cluster_q'] = h['query'].map(hash_to_cluster)
    h['cluster_t'] = h['target'].map(hash_to_cluster)
    unmapped = h['cluster_q'].isna() | h['cluster_t'].isna()
    if unmapped.any():
        raise ValueError(
            f"{int(unmapped.sum())} hit(s) have a sequence absent from the "
            "cluster parquet -- FASTA and parquet are not from the same build.")
    viol = h[h['cluster_q'] != h['cluster_t']].copy()
    viol['_pair'] = [frozenset((q, t)) for q, t in zip(viol['query'], viol['target'])]
    return viol.drop_duplicates('_pair').drop(columns='_pair')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cluster_parquet', required=True,
                   help='OOD cluster parquet (<hash>, cluster_id).')
    p.add_argument('--fasta', required=True,
                   help='Unique-sequence FASTA used to build the clusters '
                        '(headers = hash).')
    p.add_argument('--threshold', type=float, required=True,
                   help='Identity threshold t the clusters were built at (0..1).')
    p.add_argument('--coverage', type=float, default=0.8,
                   help='-c coverage (default 0.8; match the clustering).')
    p.add_argument('--cov_mode', type=int, default=0,
                   help='--cov-mode (default 0 = bidirectional).')
    p.add_argument('--seq_id_mode', type=int, default=0,
                   help='--seq-id-mode (default 0 = matches / alignment length).')
    p.add_argument('--sensitivity', type=float, default=None,
                   help='If set, use -s <val> instead of --exhaustive-search '
                        '(independent cross-check). Pair with a high --max_seqs.')
    p.add_argument('--max_seqs', type=int, default=None,
                   help='--max-seqs (raise well above the sequence count for the '
                        '-s cross-check so no neighbors are truncated).')
    p.add_argument('--gpu', type=int, default=0,
                   help='Use GPU (CUDA): 1=on (adds --createdb-mode 2 for a '
                        'GPU-compatible db). Pick the device with '
                        'CUDA_VISIBLE_DEVICES and make sure it is free.')
    p.add_argument('--mmseqs_bin', default=None,
                   help='Path/name of the mmseqs binary. Default: $MMSEQS_BIN, then '
                        '"mmseqs" on PATH.')
    p.add_argument('--tmp_dir', default=None,
                   help='mmseqs scratch dir (default: a temp dir, auto-removed).')
    p.add_argument('--out_tsv', default=None,
                   help='Keep the all-vs-all TSV here (default: temp, auto-removed).')
    args = p.parse_args()

    mmseqs_bin = args.mmseqs_bin or os.environ.get('MMSEQS_BIN', 'mmseqs')
    if shutil.which(mmseqs_bin) is None:
        raise SystemExit(f"mmseqs binary not found: {mmseqs_bin!r}")

    mapping, alphabet, sizes = load_hash_to_cluster(Path(args.cluster_parquet))
    method = ('--exhaustive-search' if args.sensitivity is None
              else f'-s {args.sensitivity:g}')
    if args.gpu:
        method += ' +GPU'
    print(f"Verifying OOD clusters: {args.cluster_parquet}")
    print(f"  alphabet={alphabet}  n_seqs={len(mapping):,}  n_clusters={len(sizes):,}  "
          f"largest={int(sizes.max()):,}  singletons={int((sizes == 1).sum()):,}")
    print(f"  t={args.threshold}  cov>={args.coverage} (cov-mode {args.cov_mode})  "
          f"method={method}")

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(td) / 'tmp'
        out_tsv = Path(args.out_tsv) if args.out_tsv else Path(td) / 'allvsall.tsv'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        run_all_vs_all(
            Path(args.fasta), out_tsv, tmp_dir, alphabet=alphabet,
            threshold=args.threshold, coverage=args.coverage,
            cov_mode=args.cov_mode, seq_id_mode=args.seq_id_mode,
            sensitivity=args.sensitivity, max_seqs=args.max_seqs,
            gpu=args.gpu, mmseqs_bin=mmseqs_bin, threads=16)
        hits = pd.read_csv(out_tsv, sep='\t',
                           names=['query', 'target', 'fident', 'qcov', 'tcov'])
        n_hits = int((hits['query'] != hits['target']).sum())
        viol = find_violations(hits, mapping)

    print(f"  alignment hits (excl. self): {n_hits:,}")
    print(f"  cross-cluster violations (id>=t and cov>={args.coverage}): {len(viol):,}")
    if len(viol):
        cols = ['query', 'target', 'fident', 'qcov', 'tcov', 'cluster_q', 'cluster_t']
        print("WARNING: across-different guarantee VIOLATED. Sample:")
        print(viol[cols].head(10).to_string(index=False))
        raise SystemExit(f"ERROR: {len(viol):,} cross-cluster pair(s) meet the link rule.")
    print("Done. Guarantee holds: 0 cross-cluster pairs meet the link rule.")


if __name__ == '__main__':
    main()
