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
import os
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


def export_function_cds_fasta(
    cds_df: pd.DataFrame,
    function_name: str,
    out_path: Path,
) -> dict:
    """Export unique CDS DNA sequences for one function to FASTA.

    Nt counterpart of `export_function_fasta`. Takes a cds_final-style
    DataFrame (must contain 'function', 'cds_dna', 'cds_dna_hash')
    instead of the aa protein table, and writes one FASTA entry per
    unique `cds_dna_hash` with the hash as the header. The body is the
    raw CDS DNA — no `*`-stripping (DNA has no terminal-stop
    representation) and no IUPAC scrubbing (mmseqs accepts ambiguity
    codes natively in nt search-type 3).

    Returns a stats dict mirroring `export_function_fasta`.
    """
    if 'cds_dna' not in cds_df.columns or 'cds_dna_hash' not in cds_df.columns:
        raise ValueError(
            "cds_df must contain 'cds_dna' and 'cds_dna_hash' columns "
            "(build via src/preprocess/extract_cds_dna.py)"
        )
    sub = cds_df[cds_df['function'] == function_name]
    if len(sub) == 0:
        raise ValueError(f"No rows match function={function_name!r}")
    n_rows = len(sub)

    seq_df = (
        sub[['cds_dna_hash', 'cds_dna']]
        .drop_duplicates(subset='cds_dna_hash')
        .reset_index(drop=True)
    )

    # Sanity: every cds_dna_hash should map to a single distinct cds_dna.
    if seq_df['cds_dna_hash'].duplicated().any():
        n_dup = int(seq_df['cds_dna_hash'].duplicated().sum())
        raise AssertionError(
            f"{n_dup} duplicate cds_dna_hash rows after drop_duplicates — "
            f"hash<->dna mapping is inconsistent."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for _, row in seq_df.iterrows():
            f.write(f">{row['cds_dna_hash']}\n{row['cds_dna']}\n")

    return {
        'function': function_name,
        'n_rows_input': int(n_rows),
        'n_unique_sequences': int(len(seq_df)),
        'n_with_ambiguity': int(
            seq_df['cds_dna'].str.contains('[^ACGTacgt]', regex=True).sum()
        ),
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
    mmseqs_bin: Optional[str] = None,
    log_path: Optional[Path] = None,
    extra_args: Optional[list] = None,
    alphabet: str = 'aa',
    algorithm: str = 'linclust',
) -> MMseqsResult:
    """Run `mmseqs easy-linclust` (or `easy-cluster`) as a subprocess.

    Produces <out_prefix>_cluster.tsv among other outputs.

    **Pinned mmseqs parameters** (set explicitly on the command line even
    when the value equals the mmseqs2 default, so the per-run log
    unambiguously records what we chose vs what we inherited):

    - `--min-seq-id <t>` — identity threshold (caller-supplied via `min_seq_id`).
    - `-c 0.8` / `--cov-mode 0` — bidirectional ≥80% coverage rule.
    - `--dbtype 1` (aa) or `--dbtype 2` (nt) — explicit alphabet, no auto-detect.
    - `--cluster-mode 0` — Set-Cover greedy assignment (mmseqs2 default; pinned
      because mode 2 = CDHIT-style and mode 1 = BLASTclust are valid alternatives).
    - `--seq-id-mode 0` — identity = matches / alignment_length (mmseqs2 default;
      pinned because mode 1 = shorter, mode 2 = longer are valid alternatives,
      and they give materially different cluster radii on length-variant corpora).
    - `--similarity-type 2` — sequence-identity scoring (mmseqs2 default; pinned
      because alignment-score scoring is the alternative).
    - `-e 0.001` — alignment e-value cutoff (mmseqs2 default; pinned for
      reproducibility against future default drifts).

    All other mmseqs2 flags (sensitivity, k-mer size, masking, alignment-mode,
    etc.) are left at mmseqs2's tuned defaults. The per-run log dumps the full
    parameter set; consult that log for forensic reproduction.

    Args:
        fasta_path: input FASTA file.
        out_prefix: output path prefix (mmseqs writes several files starting with this).
        tmp_dir: mmseqs scratch directory (created if missing).
        min_seq_id: minimum sequence identity for clustering (0..1).
        coverage: -c argument (default 0.8).
        cov_mode: --cov-mode argument (default 0 = bidirectional).
        threads: --threads argument; None lets mmseqs decide (uses all cores).
        mmseqs_bin: path or PATH-name of the mmseqs binary. If None (default),
            falls back to the `MMSEQS_BIN` env var, and then to `'mmseqs'`
            (lookup on PATH). Use this to point at an isolated mmseqs2 env
            without putting it on PATH (see .claude/memory.md "Env Management").
        log_path: if given, mmseqs stdout/stderr is written here.
        extra_args: any additional flags to append. Caller-supplied flags
            appear AFTER the pinned set on the command line, so they can
            override (mmseqs2 takes the last value when a flag is repeated).
        alphabet: 'aa' (default) or 'nt'. Selects `--dbtype 1` (aa) or
            `--dbtype 2` (nt). Required for the Experiment B-nt CDS clustering
            path (nt).
        algorithm: 'linclust' (default, `easy-linclust` — linear-time, single-pass)
            or 'cluster' (`easy-cluster` — sensitive 3-round cascade). Both
            alphabets use linclust since 2026-05-22; see
            `docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md`
            for the rationale (the asymmetric easy-cluster/easy-linclust choice
            conflated algorithm sensitivity with alphabet diversity in §8/§9
            of the methods doc; symmetric linclust gives a cleaner comparison).
    """
    fasta_path = Path(fasta_path)
    out_prefix = Path(out_prefix)
    tmp_dir = Path(tmp_dir)

    if mmseqs_bin is None:
        mmseqs_bin = os.environ.get('MMSEQS_BIN', 'mmseqs')

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    if shutil.which(mmseqs_bin) is None:
        raise RuntimeError(
            f"mmseqs binary not found: {mmseqs_bin!r}. "
            f"Set MMSEQS_BIN env var to the dedicated-env binary "
            f"(e.g. /homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs) "
            f"or put 'mmseqs' on PATH."
        )
    if alphabet not in {'aa', 'nt'}:
        raise ValueError(f"alphabet must be 'aa' or 'nt', got {alphabet!r}")
    if algorithm not in {'cluster', 'linclust'}:
        raise ValueError(
            f"algorithm must be 'cluster' or 'linclust', got {algorithm!r}"
        )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    subcmd = 'easy-cluster' if algorithm == 'cluster' else 'easy-linclust'
    # `--dbtype` is the only flag that differs between alphabets:
    #   --dbtype 1 = amino acid, --dbtype 2 = nucleotide.
    # We pin it explicitly (no --dbtype 0 auto-detect) so the alphabet is
    # part of the audit trail. `--search-type 3` is a `search`-only flag
    # in mmseqs 18 and gets rejected on easy-cluster / easy-linclust.
    dbtype = '1' if alphabet == 'aa' else '2'
    cmd = [
        mmseqs_bin, subcmd,
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        # Pinned set — see docstring for rationale on each.
        '--min-seq-id', f'{min_seq_id:g}',
        '-c', f'{coverage:g}',
        '--cov-mode', str(cov_mode),
        '--dbtype', dbtype,
        '--cluster-mode', '0',     # Set-Cover greedy (mmseqs2 default; explicit for audit)
        '--seq-id-mode', '0',      # matches / alignment_length (mmseqs2 default)
        '--similarity-type', '2',  # sequence-identity scoring (mmseqs2 default)
        '-e', '0.001',             # alignment e-value cutoff (mmseqs2 default)
    ]
    if threads is not None:
        cmd += ['--threads', str(threads)]
    if extra_args:
        # Caller-supplied flags go LAST. mmseqs2 takes the last value on
        # repetition, so extra_args can override anything in the pinned set.
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
            f"mmseqs {subcmd} failed (returncode={proc.returncode}).\n"
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
