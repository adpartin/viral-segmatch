"""mmseqs2 clustering utilities for protein-sequence-similarity-aware splits.

Used by the cluster-disjoint routing experiment (Experiment B in
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`). Provides
pure-Python helpers around the external `mmseqs` binary:

- export per-function unique-sequence FASTAs (with light cleaning)
- invoke `mmseqs easy-cluster`
- parse `_cluster.tsv` into a (seq_hash, cluster_id) lookup
- summarize cluster-size distributions for the redundancy assessment
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


def compute_seq_hash(prot_seq: str) -> str:
    """md5 hex of the protein string (matches preprocess_flu.py)."""
    return hashlib.md5(prot_seq.encode()).hexdigest()


def clean_for_mmseqs(prot_seq: str) -> str:
    """Strip trailing '*' (terminal stop). Leaves X residues intact (mmseqs handles them).

    Raises if the sequence contains an internal '*' (Stage 1 already filters these,
    but the assertion keeps callers honest if they pass a different df).
    """
    if not isinstance(prot_seq, str) or len(prot_seq) == 0:
        raise ValueError(f"prot_seq must be a non-empty string, got: {prot_seq!r}")
    stripped = prot_seq.rstrip('*')
    if '*' in stripped:
        raise ValueError("prot_seq contains an internal '*' (stop codon) — drop the row before clustering")
    return stripped


def export_function_fasta(
    prot_df: pd.DataFrame,
    function_name: str,
    out_path: Path,
) -> dict:
    """Export unique cleaned protein sequences for a single function to FASTA.

    The FASTA header is the **raw-sequence** md5 hash (matching Stage 1
    `preprocess_flu.py`, which hashes prot_seq INCLUDING the trailing '*').
    The FASTA body is the **cleaned** sequence (trailing '*' stripped) so
    mmseqs can parse it. The cluster lookup's `seq_hash` therefore joins
    back to Stage 3's `seq_hash_a` / `seq_hash_b` columns without re-hashing.

    Inputs:
        prot_df: must contain columns 'function' and 'prot_seq'.
        function_name: exact value to match in the 'function' column (full
            descriptive name, not the short alias).
        out_path: FASTA destination.

    Returns a small stats dict.
    """
    sub = prot_df[prot_df['function'] == function_name]
    if len(sub) == 0:
        raise ValueError(f"No rows match function={function_name!r}")
    n_rows = len(sub)

    raw = sub['prot_seq'].astype(str)
    seq_hash = raw.map(compute_seq_hash)               # raw-sequence hash (matches Stage 1)
    cleaned = raw.map(clean_for_mmseqs)                # stripped for mmseqs

    seq_df = pd.DataFrame({'seq_hash': seq_hash.values, 'cleaned_seq': cleaned.values})
    seq_df = seq_df.drop_duplicates(subset='seq_hash').reset_index(drop=True)

    # Sanity: same seq_hash should map to the same cleaned_seq (since hash is on raw)
    dup_check = (
        seq_df.groupby('seq_hash')['cleaned_seq'].nunique()
        if seq_df['seq_hash'].duplicated().any() else None
    )
    if dup_check is not None and (dup_check > 1).any():
        bad = dup_check[dup_check > 1].head(3).index.tolist()
        raise AssertionError(
            f"Same seq_hash maps to multiple cleaned sequences (e.g. {bad}) — "
            f"would break the FASTA<->lookup join."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for _, row in seq_df.iterrows():
            f.write(f">{row['seq_hash']}\n{row['cleaned_seq']}\n")

    return {
        'function': function_name,
        'n_rows_input': int(n_rows),
        'n_unique_sequences': int(len(seq_df)),
        'n_with_x': int(seq_df['cleaned_seq'].str.contains('X', regex=False).sum()),
        'fasta_path': str(out_path),
    }


@dataclass
class MMseqsResult:
    """Output paths from a single mmseqs easy-cluster run."""
    out_prefix: Path              # the --output-prefix that mmseqs was given
    cluster_tsv: Path             # <prefix>_cluster.tsv (rep_id <tab> member_id)
    rep_fasta: Path               # <prefix>_rep_seq.fasta
    all_seqs_fasta: Path          # <prefix>_all_seqs.fasta
    tmp_dir: Path                 # the tmp dir used (left in place; caller can rm)
    log_path: Optional[Path] = None


def run_mmseqs_easy_cluster(
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float = 0.8,
    cov_mode: int = 0,
    threads: Optional[int] = None,
    mmseqs_bin: str = 'mmseqs',
    log_path: Optional[Path] = None,
    extra_args: Optional[list] = None,
) -> MMseqsResult:
    """Run `mmseqs easy-cluster` as a subprocess.

    Defaults match the plan: --min-seq-id={min_seq_id} -c 0.8 --cov-mode 0.
    Produces <out_prefix>_cluster.tsv among other outputs.

    Args:
        fasta_path: input FASTA file.
        out_prefix: output path prefix (mmseqs writes several files starting with this).
        tmp_dir: mmseqs scratch directory (created if missing).
        min_seq_id: minimum sequence identity for clustering (0..1).
        coverage: -c argument (default 0.8).
        cov_mode: --cov-mode argument (default 0 = bidirectional).
        threads: --threads argument; None lets mmseqs decide (uses all cores).
        mmseqs_bin: binary name on PATH (default 'mmseqs').
        log_path: if given, mmseqs stdout/stderr is written here.
        extra_args: any additional flags to append.
    """
    fasta_path = Path(fasta_path)
    out_prefix = Path(out_prefix)
    tmp_dir = Path(tmp_dir)

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    if shutil.which(mmseqs_bin) is None:
        raise RuntimeError(f"mmseqs binary not on PATH: {mmseqs_bin!r}")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        mmseqs_bin, 'easy-cluster',
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        '--min-seq-id', f'{min_seq_id:g}',
        '-c', f'{coverage:g}',
        '--cov-mode', str(cov_mode),
    ]
    if threads is not None:
        cmd += ['--threads', str(threads)]
    if extra_args:
        cmd += list(extra_args)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as f:
            f.write(f"# CMD: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        tail = ''
        if log_path is not None and log_path.exists():
            tail = log_path.read_text()[-2000:]
        elif hasattr(proc, 'stdout'):
            tail = (proc.stdout or '')[-2000:] + (proc.stderr or '')[-2000:]
        raise RuntimeError(
            f"mmseqs easy-cluster failed (returncode={proc.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"Tail of output:\n{tail}"
        )

    cluster_tsv = out_prefix.parent / f"{out_prefix.name}_cluster.tsv"
    if not cluster_tsv.exists():
        raise FileNotFoundError(
            f"mmseqs reported success but cluster TSV is missing: {cluster_tsv}"
        )

    return MMseqsResult(
        out_prefix=out_prefix,
        cluster_tsv=cluster_tsv,
        rep_fasta=out_prefix.parent / f"{out_prefix.name}_rep_seq.fasta",
        all_seqs_fasta=out_prefix.parent / f"{out_prefix.name}_all_seqs.fasta",
        tmp_dir=tmp_dir,
        log_path=log_path,
    )


def parse_cluster_tsv(
    cluster_tsv: Path,
    cluster_id_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Parse `<prefix>_cluster.tsv` into a (seq_hash, cluster_id) DataFrame.

    mmseqs writes one row per (representative, member) pair, tab-separated.
    The representative is the cluster identifier; every member (including the
    representative itself) gets one row.

    Args:
        cluster_tsv: path to the *_cluster.tsv produced by easy-cluster.
        cluster_id_prefix: if given, cluster_ids are formatted as
            f"{prefix}_{integer_index}" — useful when concatenating per-function
            results into one global lookup. Otherwise an integer cluster_id is
            assigned starting at 0 in order of first appearance.

    Returns a DataFrame with columns:
        - seq_hash: member identifier (matches the FASTA we wrote)
        - cluster_id: stable identifier (integer or "{prefix}_{int}")
        - cluster_rep: representative seq_hash for that cluster

    Asserts each seq_hash appears exactly once.
    """
    cluster_tsv = Path(cluster_tsv)
    df = pd.read_csv(
        cluster_tsv,
        sep='\t',
        header=None,
        names=['cluster_rep', 'seq_hash'],
        dtype=str,
    )

    if df['seq_hash'].duplicated().any():
        n_dup = int(df['seq_hash'].duplicated().sum())
        raise ValueError(
            f"parse_cluster_tsv: {n_dup} duplicate seq_hash rows in {cluster_tsv}; "
            f"every member should appear exactly once."
        )

    # Assign integer cluster_ids in order of first appearance of each rep.
    rep_to_int = {}
    next_id = 0
    int_ids = []
    for rep in df['cluster_rep']:
        if rep not in rep_to_int:
            rep_to_int[rep] = next_id
            next_id += 1
        int_ids.append(rep_to_int[rep])
    df['cluster_int'] = int_ids

    if cluster_id_prefix is None:
        df['cluster_id'] = df['cluster_int'].astype('int64')
    else:
        df['cluster_id'] = [f"{cluster_id_prefix}_{i}" for i in df['cluster_int']]

    return df[['seq_hash', 'cluster_id', 'cluster_rep']].reset_index(drop=True)


def cluster_size_distribution(cluster_lookup: pd.DataFrame) -> dict:
    """Summary stats over cluster sizes.

    Inputs:
        cluster_lookup: DataFrame with at least 'cluster_id' column (one row per
            sequence). Typically the output of `parse_cluster_tsv`.

    Returns a dict with: n_sequences, n_clusters, largest_cluster, median_cluster_size,
    mean_cluster_size, fraction_singletons, p90_cluster_size, p99_cluster_size.
    """
    sizes = cluster_lookup.groupby('cluster_id').size().values
    if len(sizes) == 0:
        return {
            'n_sequences': 0, 'n_clusters': 0, 'largest_cluster': 0,
            'median_cluster_size': 0.0, 'mean_cluster_size': 0.0,
            'p90_cluster_size': 0, 'p99_cluster_size': 0, 'fraction_singletons': 0.0,
        }
    sizes_sorted = sorted(sizes)
    n_singletons = int((cluster_lookup.groupby('cluster_id').size() == 1).sum())
    import numpy as np
    return {
        'n_sequences': int(len(cluster_lookup)),
        'n_clusters': int(len(sizes)),
        'largest_cluster': int(max(sizes)),
        'median_cluster_size': float(np.median(sizes)),
        'mean_cluster_size': float(np.mean(sizes)),
        'p90_cluster_size': int(np.percentile(sizes, 90)),
        'p99_cluster_size': int(np.percentile(sizes, 99)),
        'fraction_singletons': float(n_singletons / len(sizes)),
    }
