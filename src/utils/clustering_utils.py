"""mmseqs2 clustering utilities for sequence-similarity-aware splits.

Used by the cluster-disjoint routing pipeline. Provides pure-Python helpers
around the external `mmseqs` binary:

- export per-function unique-sequence FASTAs (with alphabet-aware cleaning)
- invoke `mmseqs easy-linclust` / `easy-cluster`
- parse `_cluster.tsv` into a (hash, cluster_id) lookup
- summarize cluster-size distributions for the redundancy assessment

Function-metadata loading lives in `src/utils/config_hydra.py`
(see `load_function_metadata` there).

Alphabets:
    'aa'     -> protein sequences from protein_final-style input
                    (columns: function, prot_seq, prot_hash)
    'nt_cds' -> CDS DNA from cds_dna_final-style input
                    (columns: function, cds_dna_seq, cds_dna_hash)
    'nt_ctg' -> full-contig DNA from ctg_dna_final-style input
                    (columns: function, ctg_dna_seq, ctg_dna_hash). ctg_dna_final
                    carries no `function`; attach it via a verified 1-1 join from
                    cds_dna_final (see attach_function_to_contigs). Subject to the
                    contig UTR caveat noted on _COLS_BY_ALPHABET below.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import schema

# Dispatch table: alphabet -> {seq_col, hash_col}, derived from the schema
# registry (single source of truth) for the unified exporter + parse_cluster_tsv
# output column dispatch. All three alphabets are active; nt_ctg requires the
# `function` label to be attached first (ctg_dna_final carries none) via
# attach_function_to_contigs.
#
# nt_ctg data caveat (unresolved): contig DNA carries inconsistent
# flanking UTR. Verified on the contig source: per-row (contig_len - cds_len)
# spans 0..310 nt (median 36; ~25% of contigs have zero UTR, i.e. contig == CDS).
# So two biologically identical isolates can land in different contig clusters
# purely from how much flank the submitter included (assembly/submission
# artifact, not biology). Decision: keep nt_ctg as-is for now, do NOT pre-trim.
# TODO(nt_ctg): revisit contig UTR normalization before trusting nt_ctg
# cluster-disjoint results for publication.
_COLS_BY_ALPHABET = {a: {'seq_col': s.seq_col, 'hash_col': s.hash_col}
                     for a, s in schema.SCHEMA.items()}
_ACTIVE_ALPHABETS = {'aa', 'nt_cds', 'nt_ctg'}


def compute_prot_hash(prot_seq: str) -> str:
    """md5 hex of the protein string (matches preprocess_flu.py).

    Caller-side helper for pre-hashing aa input before calling
    `export_function_fasta(alphabet='aa', ...)`: the unified exporter
    requires the hash column to be already present (matches the nt path
    where Stage 1.5 writes `cds_dna_hash` into cds_dna_final.parquet).
    """
    return hashlib.md5(prot_seq.encode()).hexdigest()


def clean_aa_for_mmseqs(prot_seq: str) -> str:
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


def clean_nt_for_mmseqs(dna_seq: str) -> str:
    """Pass-through for nt: no `*` to strip; mmseqs accepts IUPAC codes natively.

    Raises only on empty input. This is the right home for any future
    nt-specific cleaning (e.g., flanking-UTR normalization for nt_ctg --
    deferred by decision; exp-2 runs contig DNA artifact-included, see the
    UTR caveat on _COLS_BY_ALPHABET).
    """
    if not isinstance(dna_seq, str) or len(dna_seq) == 0:
        raise ValueError(f"dna_seq must be a non-empty string, got: {dna_seq!r}")
    return dna_seq


def _clean_for_mmseqs(seq: str, alphabet: str) -> str:
    """Alphabet dispatcher for cleaning. Internal."""
    if alphabet == 'aa':
        return clean_aa_for_mmseqs(seq)
    if alphabet in ('nt_cds', 'nt_ctg'):
        # Both nt molecules are pass-through (no '*' to strip; mmseqs scores
        # IUPAC codes natively). nt_ctg is NOT pre-trimmed for flanking UTR --
        # see the UTR caveat on _COLS_BY_ALPHABET (exp-2 runs artifact-included
        # by decision).
        return clean_nt_for_mmseqs(seq)
    raise ValueError(f"alphabet must be in {sorted(_ACTIVE_ALPHABETS)}, got {alphabet!r}")


def attach_function_to_contigs(
    ctg_df: pd.DataFrame,
    function_source: Path,
    *,
    key: tuple = ('assembly_id', 'genbank_ctg_id'),
    ) -> pd.DataFrame:
    """Attach a `function` label to per-contig rows via a verified 1-1 join.

    `ctg_dna_final` is per-contig and carries no `function` column (the
    clustering/membership paths need one to group contigs by segment). We take it
    from a CDS source (`function_source`, typically `cds_dna_final`) that is
    one-row-per-major-CDS and therefore 1-1 with the major-segment contigs on
    `key`. Verified on the July_2025 corpus: both tables are 868,240 rows with
    identical, unique `(assembly_id, genbank_ctg_id)` keys, so every major-segment
    contig maps to exactly one of the 8 selected majors.

    Inner-joins `[*key, function]` onto `ctg_df` and asserts no contig maps to >1
    function (the 1-1 guarantee -- a contig clustered under two functions would
    corrupt the partition). Contigs with no matching function row (non-major
    contigs, if any) are dropped and the count reported.

    Args:
        ctg_df: per-contig rows; must contain the `key` columns plus whatever
            sequence/hash/length columns the caller needs.
        function_source: path to `cds_dna_final` (or another `[*key, function]`
            source), `.parquet` or `.csv`.
        key: join key; `(assembly_id, genbank_ctg_id)` by default.

    Returns `ctg_df` with a `function` column added (inner-joined).
    """
    key = list(key)
    missing = set(key) - set(ctg_df.columns)
    if missing:
        raise ValueError(
            f"attach_function_to_contigs: ctg_df missing key columns {sorted(missing)}")

    fsrc = Path(function_source)
    cols = key + ['function']
    if fsrc.suffix == '.csv':
        # keep_default_na guards the 'NA'-string trap (Neuraminidase); function
        # uses full names here, but it is the safe default for any CSV read.
        fn = pd.read_csv(fsrc, usecols=cols, keep_default_na=False, na_values=[''])
    else:
        fn = pd.read_parquet(fsrc, columns=cols)
    if fn.duplicated(key).any():
        n = int(fn.duplicated(key).sum())
        raise ValueError(
            f"attach_function_to_contigs: function_source has {n:,} duplicate "
            f"{key} rows; expected one function per contig.")

    n_in = len(ctg_df)
    merged = ctg_df.merge(fn, on=key, how='inner')
    if merged.duplicated(key).any():
        n = int(merged.duplicated(key).sum())
        raise AssertionError(
            f"attach_function_to_contigs: {n:,} contig(s) map to >1 function "
            f"after join -- not 1-1.")
    n_dropped = n_in - len(merged)
    if n_dropped:
        print(f"  NOTE: {n_dropped:,} contig(s) had no matching function row "
              f"(non-major; dropped).")
    return merged


def export_function_fasta(
    df: pd.DataFrame,
    function_name: str,
    alphabet: str,
    out_path: Path,
    ) -> dict:
    """Export unique cleaned sequences for one function to FASTA (alphabet-aware).

    The FASTA header is the precomputed per-alphabet hash from the input
    DataFrame (`prot_hash` for aa, `cds_dna_hash` for nt_cds, `ctg_dna_hash`
    for nt_ctg; see `_COLS_BY_ALPHABET`). The FASTA body is the cleaned
    sequence: trailing `*` stripped for aa, pass-through for both nt molecules.
    The cluster lookup's hash column therefore joins back to Stage 3's
    pair-table hash columns without re-hashing.

    Caller responsibility: the input DataFrame must carry the
    precomputed hash column corresponding to `alphabet` (per
    `_COLS_BY_ALPHABET`). For `'nt_cds'`, that's `cds_dna_hash`
    (written by Stage 1.5 into `cds_dna_final.parquet`). For `'aa'`,
    that's `prot_hash` (which Stage 1 writes for cds_dna_final but NOT
    for protein_final, so aa callers reading protein_final must
    pre-hash via `compute_prot_hash` before calling).

    Safety net: at entry, recomputes the hash from the seq column and
    asserts it matches the precomputed hash. Catches stale precomputed
    hashes at one md5/row cost.

    Args:
        df: input DataFrame. Must contain 'function' plus the seq+hash
            columns for `alphabet` (see `_COLS_BY_ALPHABET`). For nt_ctg the
            `function` label is attached via `attach_function_to_contigs`.
        function_name: exact value to match in the 'function' column.
        alphabet: 'aa', 'nt_cds', or 'nt_ctg'.
        out_path: FASTA destination.

    Returns a stats dict: function, n_rows_input, n_uniq_seqs,
    n_with_ambiguity, fasta_path.
    """
    if alphabet not in _ACTIVE_ALPHABETS:
        raise ValueError(f"alphabet must be in {sorted(_ACTIVE_ALPHABETS)}, got {alphabet!r}")
    seq_col = _COLS_BY_ALPHABET[alphabet]['seq_col']
    hash_col = _COLS_BY_ALPHABET[alphabet]['hash_col']

    required = {'function', seq_col, hash_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"export_function_fasta: df is missing required columns for "
            f"alphabet={alphabet!r}: {sorted(missing)}. "
            f"Pre-hash with compute_prot_hash() if needed."
        )

    sub = df[df['function'] == function_name]
    if len(sub) == 0:
        raise ValueError(f"No rows match function={function_name!r}")
    n_rows = len(sub)

    raw = sub[seq_col].astype(str)
    precomputed = sub[hash_col].astype(str)

    # Safety net: recompute hash from seq and assert against precomputed.
    # One md5/row; catches stale hashes that would silently produce wrong
    # FASTA headers (the footgun the unified-exporter refactor was meant
    # to close).
    recomputed = raw.map(lambda s: hashlib.md5(s.encode()).hexdigest())
    mismatch = (precomputed.values != recomputed.values)
    if mismatch.any():
        n_bad = int(mismatch.sum())
        bad_idx = sub.index[mismatch][:3].tolist()
        raise ValueError(
            f"export_function_fasta: {n_bad} row(s) have stale {hash_col!r} "
            f"(md5({seq_col}) disagrees with precomputed). "
            f"Example indices: {bad_idx}"
        )

    cleaned = raw.map(lambda s: _clean_for_mmseqs(s, alphabet))
    seq_df = pd.DataFrame({
        hash_col: precomputed.values,
        '_cleaned_seq': cleaned.values,
    }).drop_duplicates(subset=hash_col).reset_index(drop=True)

    # Cross-check: same hash -> same cleaned sequence (since the hash is on
    # raw input and cleaning is deterministic). drop_duplicates kept the
    # first occurrence; this asserts the dropped duplicates agreed.
    if seq_df[hash_col].duplicated().any():
        n_dup = int(seq_df[hash_col].duplicated().sum())
        raise AssertionError(
            f"{n_dup} duplicate {hash_col} rows after drop_duplicates -- "
            f"hash<->seq mapping is inconsistent."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for _, row in seq_df.iterrows():
            f.write(f">{row[hash_col]}\n{row['_cleaned_seq']}\n")

    # Ambiguity count per alphabet: aa counts X residues (mmseqs handles
    # them natively but they're the salient ambiguity character); nt counts
    # any non-ACGT character (IUPAC codes — mmseqs scores them natively).
    ambig_pat = '[X]' if alphabet == 'aa' else '[^ACGTacgt]'
    return {
        'function': function_name,
        'n_rows_input': int(n_rows),
        'n_uniq_seqs': int(len(seq_df)),
        'n_with_ambiguity': int(
            seq_df['_cleaned_seq'].str.contains(ambig_pat, regex=True).sum()
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


def _build_mmseqs_clust_cmd(
    *,
    mmseqs_bin: str,
    subcmd: str,
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    dbtype: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    sensitivity: Optional[float] = None,
    single_step_clustering: bool = False,
    max_seqs: Optional[int] = None,
    threads: Optional[int] = 16,
    extra_args: Optional[list] = None,
    ) -> list:
    """Assemble the `mmseqs easy-cluster` / `easy-linclust` command line.

    Pure (no I/O): the single source of truth for the emitted flags, so the
    back-compat contract — default args reproduce the historical Set-Cover /
    linclust command byte-for-byte — is unit-testable without invoking mmseqs.
    `-s` and `--single-step-clustering` are emitted only when set, so the default
    command is unchanged. See `run_mmseqs_easy_clust` for each flag's meaning.
    """
    cmd = [
        mmseqs_bin, subcmd,
        str(fasta_path),
        str(out_prefix),
        str(tmp_dir),
        # Pinned set — see run_mmseqs_easy_clust docstring for rationale on each.
        '--min-seq-id', f'{min_seq_id:g}',
        '-c', f'{coverage:g}',
        '--cov-mode', str(cov_mode),
        '--dbtype', dbtype,
        '--cluster-mode', str(cluster_mode),  # 0=Set-Cover; 1=connected-comp (OOD)
        '--seq-id-mode', '0',      # matches / alignment_length (mmseqs2 default)
        '--similarity-type', '2',  # sequence-identity scoring (mmseqs2 default)
        '-e', '0.001',             # alignment e-value cutoff (mmseqs2 default)
    ]
    # OOD recipe (connected-component clustering): emitted only when set, so the
    # default command stays byte-identical.
    if sensitivity is not None:
        cmd += ['-s', f'{sensitivity:g}']
    if single_step_clustering:
        cmd += ['--single-step-clustering', '1']
    if max_seqs is not None:
        cmd += ['--max-seqs', str(max_seqs)]
    if threads is not None:
        cmd += ['--threads', str(threads)]
    if extra_args:
        # Caller-supplied flags go LAST. mmseqs2 takes the last value on
        # repetition, so extra_args can override anything in the pinned set.
        cmd += list(extra_args)
    return cmd


def run_mmseqs_easy_clust(
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float = 0.8,
    cov_mode: int = 0,
    threads: Optional[int] = 16,
    mmseqs_bin: Optional[str] = None,
    log_path: Optional[Path] = None,
    extra_args: Optional[list] = None,
    alphabet: str = 'aa',
    algorithm: str = 'linclust',
    cluster_mode: int = 0,
    sensitivity: Optional[float] = None,
    single_step_clustering: bool = False,
    max_seqs: Optional[int] = None,
    ) -> MMseqsResult:
    """Run `mmseqs easy-linclust` (or `easy-cluster`) as a subprocess.

    Produces <out_prefix>_cluster.tsv among other outputs.

    **Pinned mmseqs parameters** (set explicitly on the command line even
    when the value equals the mmseqs2 default, so the per-run log
    unambiguously records what we chose vs what we inherited):

    - `--min-seq-id <t>` — identity threshold (caller-supplied via `min_seq_id`).
    - `-c 0.8` / `--cov-mode 0` — bidirectional ≥80% coverage rule.
    - `--dbtype 1` (aa) or `--dbtype 2` (nt) — explicit alphabet, no auto-detect.
    - `--cluster-mode <cluster_mode>` — cluster assignment (default 0 = Set-Cover
      greedy / star; 1 = connected-component, the OOD across-different topology;
      2 = greedy-incremental / CD-HIT-style).
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
        threads: --threads argument; default 16; Pass None to omit --threads
            entirely and let mmseqs decide (uses all cores; rude on shared
            boxes). Pass an explicit int for dedicated runs.
        mmseqs_bin: path or PATH-name of the mmseqs binary. If None (default),
            falls back to the `MMSEQS_BIN` env var, and then to `'mmseqs'`
            (lookup on PATH). Use this to point at an isolated mmseqs2 env
            without putting it on PATH (see .claude/memory.md "Env Management").
        log_path: if given, mmseqs stdout/stderr is written here.
        extra_args: any additional flags to append. Caller-supplied flags
            appear AFTER the pinned set on the command line, so they can
            override (mmseqs2 takes the last value when a flag is repeated).
        alphabet: 'aa' (default), 'nt_cds', or 'nt_ctg'. Selects `--dbtype 1`
            (aa) or `--dbtype 2` (both nt molecules). Required for the
            nt CDS / contig clustering paths.
        algorithm: 'linclust' (default, `easy-linclust` — linear-time, single-pass)
            or 'cluster' (`easy-cluster` — sensitive 3-round cascade). Both
            alphabets use linclust.
        cluster_mode: `--cluster-mode` (default 0 = Set-Cover / star). Set to 1 for
            connected-component clustering — the topology behind the OOD
            across-cluster guarantee.
        sensitivity: `-s` prefilter sensitivity (None = mmseqs default; 7.5 = most
            sensitive). `algorithm='cluster'` only.
        single_step_clustering: `--single-step-clustering` (default False = cascaded).
            True = one non-cascaded pass (clusters = connected components of a single
            alignment graph). `algorithm='cluster'` only.
        max_seqs: `--max-seqs` (prefilter neighbors kept per sequence). For the
            connected-component OOD guarantee it must be >= the sequence count so no
            true >= t neighbor is truncated (mmseqs cluster's default of 20 is far too
            low on dense inputs and silently fragments the graph).
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
    if alphabet not in _ACTIVE_ALPHABETS:
        raise ValueError(f"alphabet must be in {sorted(_ACTIVE_ALPHABETS)}, got {alphabet!r}")
    if algorithm not in {'cluster', 'linclust'}:
        raise ValueError(
            f"algorithm must be 'cluster' or 'linclust', got {algorithm!r}"
        )
    if cluster_mode not in {0, 1, 2}:
        raise ValueError(f"cluster_mode must be 0, 1, or 2, got {cluster_mode!r}")
    if algorithm != 'cluster' and (sensitivity is not None or single_step_clustering):
        raise ValueError(
            "sensitivity (-s) and single_step_clustering (--single-step-clustering) "
            f"require algorithm='cluster'; got algorithm={algorithm!r}"
        )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if algorithm == 'cluster':
        subcmd = 'easy-cluster'
    elif algorithm == 'linclust':
        subcmd = 'easy-linclust'
    else:
        raise ValueError(f"Invalid mmseqs algorithm name in our codebase: {algorithm!r}")
    # `--dbtype` is the only flag that differs between alphabets:
    #   --dbtype 1 = amino acid, --dbtype 2 = nucleotide.
    # We pin it explicitly (no --dbtype 0 auto-detect) so the alphabet is
    # part of the audit trail. `--search-type 3` is a `search`-only flag
    # in mmseqs 18 and gets rejected on easy-cluster / easy-linclust.
    dbtype = '1' if alphabet == 'aa' else '2'
    cmd = _build_mmseqs_clust_cmd(
        mmseqs_bin=mmseqs_bin,
        subcmd=subcmd,
        fasta_path=fasta_path,
        out_prefix=out_prefix,
        tmp_dir=tmp_dir,
        dbtype=dbtype,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        cluster_mode=cluster_mode,
        sensitivity=sensitivity,
        single_step_clustering=single_step_clustering,
        max_seqs=max_seqs,
        threads=threads,
        extra_args=extra_args,
    )

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
    alphabet: str,
    cluster_id_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
    """Parse `<prefix>_cluster.tsv` into a (<hash>, cluster_id) DataFrame.

    mmseqs writes one row per (representative, member) pair, tab-separated.
    The representative is the cluster identifier; every member (including the
    representative itself) gets one row.

    The output hash-column name is alphabet-specific (per
    `_COLS_BY_ALPHABET`): `prot_hash` for aa, `cds_dna_hash` for nt_cds,
    `ctg_dna_hash` for nt_ctg. The downstream join key in
    `_split_helpers.attach_cluster_ids` matches this directly (no rename needed).

    Args:
        cluster_tsv: path to the *_cluster.tsv produced by easy-cluster.
        alphabet: 'aa', 'nt_cds', or 'nt_ctg'. Selects the output hash-column name.
        cluster_id_prefix: if given, cluster_ids are formatted as
            f"{prefix}_{integer_index}" — useful when concatenating per-function
            results into one global lookup. Otherwise an integer cluster_id is
            assigned starting at 0 in order of first appearance.

    Returns a DataFrame with columns:
        - <hash_col>: member identifier (matches the FASTA we wrote);
              `prot_hash` for aa, `cds_dna_hash` for nt_cds, `ctg_dna_hash`
              for nt_ctg.
        - cluster_id: stable identifier (integer or "{prefix}_{int}")
        - cluster_rep: representative <hash> for that cluster

    Asserts each member appears exactly once.
    """
    if alphabet not in _ACTIVE_ALPHABETS:
        raise ValueError(f"alphabet must be in {sorted(_ACTIVE_ALPHABETS)}, got {alphabet!r}")
    hash_col = _COLS_BY_ALPHABET[alphabet]['hash_col']

    cluster_tsv = Path(cluster_tsv)
    df = pd.read_csv(
        cluster_tsv,
        sep='\t',
        header=None,
        names=['cluster_rep', hash_col],
        dtype=str,
    )

    if df[hash_col].duplicated().any():
        n_dup = int(df[hash_col].duplicated().sum())
        raise ValueError(
            f"parse_cluster_tsv: {n_dup} duplicate {hash_col} rows in {cluster_tsv}; "
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

    return df[[hash_col, 'cluster_id', 'cluster_rep']].reset_index(drop=True)


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


def cluster_sizes_unique(cluster_pq: Path) -> pd.Series:
    """Unique-weighted cluster sizes for one cluster parquet (any alphabet / builder).

    Reads only `cluster_id` (each parquet row is one unique sequence), so
    `value_counts()` counts unique sequences per cluster regardless of the hash
    column's name. Returned descending (largest cluster first). Works on both the
    set-cover and the OOD cluster parquets.
    """
    return pd.read_parquet(Path(cluster_pq), columns=['cluster_id'])['cluster_id'].value_counts()


def threshold_label(threshold: float) -> str:
    """Format an identity threshold as a stable dir label, e.g. 0.95 -> 't095'.

    The `t<XXX>` prefix is the canonical user-facing form (CLAUDE.md
    "Threshold notation"). Shared by both cluster builders.
    """
    return f"t{int(round(threshold * 100)):03d}"


def threshold_decimal(threshold_id: str) -> float:
    """Inverse of `threshold_label`: 't095' -> 0.95 (the tXXX dir label -> identity)."""
    return int(str(threshold_id)[1:]) / 100.0


def load_sequence_frame(
    *,
    protein_final: Optional[str] = None,
    cds_dna_final: Optional[str] = None,
    ctg_dna_final: Optional[str] = None,
    alphabet: Optional[str] = None,
    function_source: Optional[str] = None,
    ) -> tuple:
    """Load the per-function sequence frame for clustering; resolve alphabet from the source.

    Exactly one of `protein_final` / `cds_dna_final` / `ctg_dna_final` must be
    given; the alphabet follows from it (aa / nt_cds / nt_ctg). Returns
    `(df, alphabet)` where `df` carries `function` plus the alphabet's seq+hash
    columns. aa input (`protein_final`) does not carry `prot_hash`, so it is
    computed here; nt inputs already carry their hash. For nt_ctg the `function`
    label is attached from `function_source` (default: the sibling
    `cds_dna_final.parquet`) via a verified 1-1 join.

    Shared by `build_mmseqs_clusters.py` (set-cover) and `build_ood_clusters.py`
    (connected-component) so both cluster the identical input universe.
    """
    sources = {'aa': protein_final, 'nt_cds': cds_dna_final, 'nt_ctg': ctg_dna_final}
    given = {a: p for a, p in sources.items() if p}
    if len(given) != 1:
        raise ValueError(
            "load_sequence_frame: pass exactly one of protein_final / cds_dna_final "
            f"/ ctg_dna_final; got {sorted(given)}")
    src_alphabet, in_path = next(iter(given.items()))
    alphabet = alphabet or src_alphabet
    if alphabet != src_alphabet:
        raise ValueError(
            f"{src_alphabet} source implies alphabet={src_alphabet!r}, not {alphabet!r}")

    in_path = Path(in_path)
    # Seq/hash column names come from the schema registry (single source of truth).
    # aa loads only the seq (prot_hash is computed below); nt inputs carry their hash.
    cols = _COLS_BY_ALPHABET[alphabet]
    if alphabet == 'aa':
        usecols = ['function', cols['seq_col']]
    elif alphabet == 'nt_cds':
        usecols = ['function', cols['seq_col'], cols['hash_col']]
    else:  # nt_ctg: ctg_dna_final has no `function`; attach it from function_source below.
        usecols = ['assembly_id', 'genbank_ctg_id', cols['seq_col'], cols['hash_col']]
    print(f"Loading {in_path} (alphabet={alphabet}) ...")
    if in_path.suffix == '.csv':
        df = pd.read_csv(in_path, usecols=usecols)
    else:
        df = pd.read_parquet(in_path, columns=usecols)
    print(f"  Loaded {len(df):,} rows")

    # nt_ctg: attach `function` (ctg_dna_final carries none) via a 1-1 join, then
    # drop the join keys so the df matches the {function, seq, hash} exporter shape.
    if alphabet == 'nt_ctg':
        fsrc = (Path(function_source) if function_source
                else in_path.with_name('cds_dna_final.parquet'))
        if not fsrc.exists():
            raise FileNotFoundError(
                f"nt_ctg needs a function_source for the contig->function join; "
                f"default sibling not found: {fsrc}")
        print(f"  Attaching `function` from {fsrc.name} (1-1 join) ...")
        df = attach_function_to_contigs(df, fsrc)
        df = df.drop(columns=['assembly_id', 'genbank_ctg_id'])

    # aa: protein_final has no prot_hash; compute it once (reused across thresholds).
    if alphabet == 'aa':
        # protein_final has no prot_hash; compute it once (md5), reused across thresholds.
        hc, sc = cols['hash_col'], cols['seq_col']
        df[hc] = df[sc].map(compute_prot_hash)
        print(f"  Hashed {sc} -> {hc} ({df[hc].nunique():,} unique)")
    return df, alphabet


def filter_present_functions(df: pd.DataFrame, short_names: list, short_to_function: dict) -> tuple:
    """Split requested function short-names into (present, skipped) for this input.

    Raises on any unknown short-name. `skipped` = requested functions whose full
    name has no rows in `df['function']` (e.g. M2/NEP are absent from
    cds_dna_final). Lets a sweep skip them instead of aborting mid-run.
    """
    unknown = [f for f in short_names if f not in short_to_function]
    if unknown:
        raise ValueError(
            f"Unknown function short names: {unknown}. Known: {sorted(short_to_function)}")
    present_full = set(df['function'].unique())
    present = [f for f in short_names if short_to_function[f] in present_full]
    skipped = [f for f in short_names if f not in present]
    return present, skipped


def read_fasta_hashes(fasta_path: Path) -> list:
    """Return the record headers (= sequence hashes) of a FASTA, in file order.

    `export_function_fasta` writes one record per unique sequence with the hash
    as the header, so this is the exact node set for connected-component
    clustering — including sequences with no `>= t` neighbour, which must stay
    singletons.
    """
    hashes = []
    with Path(fasta_path).open() as f:
        for line in f:
            if line.startswith('>'):
                hashes.append(line[1:].strip())
    return hashes


def _build_mmseqs_search_cmd(
    *,
    mmseqs_bin: str,
    fasta_path: Path,
    out_tsv: Path,
    tmp_dir: Path,
    dbtype: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    seq_id_mode: int = 0,
    evalue: float = 0.001,
    format_output: str = 'query,target,fident,qcov,tcov',
    sensitivity: Optional[float] = None,
    prefilter_mode: Optional[int] = None,
    max_seqs: Optional[int] = None,
    gpu: int = 0,
    threads: Optional[int] = 16,
    extra_args: Optional[list] = None,
    ) -> list:
    """Assemble the `mmseqs easy-search` all-vs-all command (FASTA vs itself).

    Pure (no I/O): the single source of truth for the emitted flags, so the OOD
    search rule is unit-testable. The identity/coverage rule (`--min-seq-id`,
    `-c`, `--cov-mode`, `--seq-id-mode`, `-e`) mirrors
    `src/analysis/verify_ood_clusters.py`, so a cluster set built from these hits
    verifies against the same graph. Completeness knob: `prefilter_mode=2`
    (nofilter) gives a provably-complete all-vs-all; otherwise `sensitivity`
    (`-s`) uses the fast heuristic prefilter (empirically complete at high `t`).
    Every returned hit already meets `identity >= min_seq_id` and
    `coverage >= coverage`, i.e. it is an edge of the `>= t`/cov graph.
    """
    cmd = [
        mmseqs_bin, 'easy-search',
        str(fasta_path), str(fasta_path), str(out_tsv), str(tmp_dir),
        '--min-seq-id', f'{min_seq_id:g}',
        '-c', f'{coverage:g}',
        '--cov-mode', str(cov_mode),
        '--dbtype', dbtype,
        '--seq-id-mode', str(seq_id_mode),
        '-e', f'{evalue:g}',
        '--format-output', format_output,
    ]
    # prefilter-mode 2 (nofilter/exhaustive) takes precedence over -s, so exactly
    # one prefilter setting reaches the command line.
    if prefilter_mode is not None:
        cmd += ['--prefilter-mode', str(prefilter_mode)]
    elif sensitivity is not None:
        cmd += ['-s', f'{sensitivity:g}']
    if max_seqs is not None:
        cmd += ['--max-seqs', str(max_seqs)]
    if gpu:
        cmd += ['--gpu', str(gpu), '--createdb-mode', '2']  # GPU-compatible db
    if threads is not None:
        cmd += ['--threads', str(threads)]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def run_mmseqs_search(
    fasta_path: Path,
    out_tsv: Path,
    tmp_dir: Path,
    min_seq_id: float,
    *,
    coverage: float = 0.8,
    cov_mode: int = 0,
    seq_id_mode: int = 0,
    alphabet: str = 'aa',
    sensitivity: Optional[float] = 7.5,
    prefilter_mode: Optional[int] = None,
    max_seqs: Optional[int] = None,
    gpu: int = 0,
    threads: Optional[int] = 16,
    mmseqs_bin: Optional[str] = None,
    log_path: Optional[Path] = None,
    extra_args: Optional[list] = None,
    ) -> Path:
    """Run `mmseqs easy-search` (FASTA vs itself); return the hits-TSV path.

    The hits are the edges of the `>= t`/cov similarity graph (the search filters
    guarantee every hit meets the rule). See `_build_mmseqs_search_cmd` for the
    rule and the sensitivity / prefilter-mode completeness knobs.
    """
    fasta_path, out_tsv, tmp_dir = Path(fasta_path), Path(out_tsv), Path(tmp_dir)
    if mmseqs_bin is None:
        mmseqs_bin = os.environ.get('MMSEQS_BIN', 'mmseqs')
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    if shutil.which(mmseqs_bin) is None:
        raise RuntimeError(
            f"mmseqs binary not found: {mmseqs_bin!r}. Set MMSEQS_BIN or put "
            f"'mmseqs' on PATH.")
    if alphabet not in _ACTIVE_ALPHABETS:
        raise ValueError(f"alphabet must be in {sorted(_ACTIVE_ALPHABETS)}, got {alphabet!r}")

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    # Start from a clean tmp: easy-search can skip-and-reuse intermediate DBs left in
    # tmp, so a stale one would silently run against old sequences on a re-run.
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Perf: easy-search = createdb -> prefilter (k-mer) -> gapped align -> convertalis.
    # `gpu` offloads the PREFILTER to an Ampere+/Hopper GPU -- so it speeds the default
    # `-s` (non-exhaustive) path; ~no effect under prefilter_mode=2 (nofilter = no
    # prefilter). `threads` parallelizes the CPU work: the gapped alignment (usually the
    # bottleneck) and the prefilter when it runs on CPU.
    dbtype = '1' if alphabet == 'aa' else '2'
    cmd = _build_mmseqs_search_cmd(
        mmseqs_bin=mmseqs_bin, fasta_path=fasta_path, out_tsv=out_tsv, tmp_dir=tmp_dir,
        dbtype=dbtype, min_seq_id=min_seq_id, coverage=coverage, cov_mode=cov_mode,
        seq_id_mode=seq_id_mode, sensitivity=sensitivity, prefilter_mode=prefilter_mode,
        max_seqs=max_seqs, gpu=gpu, threads=threads, extra_args=extra_args,
    )

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
            f"mmseqs easy-search failed (returncode={proc.returncode}).\n"
            f"CMD: {' '.join(cmd)}\nTail:\n{tail}")
    if not out_tsv.exists():
        raise FileNotFoundError(f"mmseqs reported success but hits TSV is missing: {out_tsv}")
    return out_tsv


def connected_components_from_hits(
    hits_tsv: Path,
    nodes: list,
    *,
    alphabet: str,
    cluster_id_prefix: str,
    ) -> pd.DataFrame:
    """Cluster = connected component of the similarity graph, via union-find.

    Nodes are all unique sequence hashes (`nodes`, from the FASTA); edges are the
    hits in `hits_tsv` (each already meets the `>= t`/cov rule). Sequences with no
    edge stay singletons. Labels are canonical and deterministic: components are
    ranked by size (desc), ties by smallest member hash, so re-runs on the same
    graph give identical `cluster_id`s regardless of union order.

    NOTE: this is a connected component of the single-segment SIMILARITY graph
    (nodes = sequences) -- a *cluster* / *mega-cluster*, NOT the bipartite CC /
    mega-CC of 2D-CD routing (see docs/methods/glossary.md).

    Returns a DataFrame `[<hash_col>, cluster_id, cluster_rep]` -- the same shape
    as `parse_cluster_tsv`, so downstream code is unchanged.
    """
    if alphabet not in _ACTIVE_ALPHABETS:
        raise ValueError(f"alphabet must be in {sorted(_ACTIVE_ALPHABETS)}, got {alphabet!r}")
    hash_col = _COLS_BY_ALPHABET[alphabet]['hash_col']

    nodes = list(nodes)
    parent = {n: n for n in nodes}
    if len(parent) != len(nodes):
        raise ValueError("connected_components_from_hits: duplicate node hashes in the FASTA")

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression: point every node on the path at root
            parent[x], x = root, parent[x]
        return root

    hits = read_hits_tsv(hits_tsv, usecols=['query', 'target'])
    for q, t in zip(hits['query'], hits['target']):
        if q == t:
            continue
        if q not in parent or t not in parent:
            raise ValueError(
                "connected_components_from_hits: a hit references a hash absent from "
                "the FASTA node set -- hits TSV and FASTA are not from the same build.")
        rq, rt = find(q), find(t)
        if rq != rt:
            parent[rq] = rt

    members = {}
    for n in nodes:
        members.setdefault(find(n), []).append(n)
    ordered = sorted(members.values(), key=lambda m: (-len(m), min(m)))

    recs = []
    for idx, member_hashes in enumerate(ordered):
        cluster_id = f"{cluster_id_prefix}_{idx}"
        rep = min(member_hashes)
        recs.extend((h, cluster_id, rep) for h in member_hashes)
    return pd.DataFrame(recs, columns=[hash_col, 'cluster_id', 'cluster_rep'])


def read_hits_tsv(hits_tsv: Path, usecols: Optional[list] = None) -> pd.DataFrame:
    """Read an `easy-search` hits TSV -- the single owner of its column schema.

    The column order is produced only by `_build_mmseqs_search_cmd`'s
    `--format-output`; parsing it in one place keeps every reader in step if that
    ever changes. `usecols` selects a subset (default: all five columns).
    """
    names = ['query', 'target', 'fident', 'qcov', 'tcov']
    return pd.read_csv(hits_tsv, sep='\t', header=None, names=names,
                       usecols=usecols or names, dtype=str)


def aggregate_combined_lookup(out_root: Path, threshold: float, function_shorts: list) -> Path:
    """Concatenate the per-function cluster parquets at one threshold into one file.

    Shared by both cluster builders. Raises FileNotFoundError naming the first
    missing per-function parquet (clearer than an opaque read error).
    """
    tdir = Path(out_root) / threshold_label(threshold)
    parts = []
    for short in function_shorts:
        p = tdir / f"{short}_cluster.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing per-function parquet: {p}")
        parts.append(pd.read_parquet(p))
    combined_path = tdir / 'combined_cluster.parquet'
    pd.concat(parts, ignore_index=True).to_parquet(combined_path, index=False)
    return combined_path


def write_or_merge_stats_csv(out_root: Path, all_stats: list, filename: str) -> Path:
    """Write per-(function, threshold) stats, merging with any prior CSV.

    A re-run at a subset of thresholds keeps the earlier rows; re-running the same
    (function_short, threshold) overwrites just those. `keep_default_na=False` guards
    the function_short='NA' (Neuraminidase) read trap (CLAUDE.md convention).
    """
    new = pd.DataFrame(all_stats)
    stats_csv = Path(out_root) / filename
    if stats_csv.exists():
        prior = pd.read_csv(stats_csv, keep_default_na=False, na_values=[''])
        if set(prior.columns) == set(new.columns):
            key = ['function_short', 'threshold']
            now = set(map(tuple, new[key].itertuples(index=False, name=None)))
            keep = [tuple(r) not in now for r in prior[key].itertuples(index=False, name=None)]
            new = pd.concat([prior[keep], new], ignore_index=True)
    new = new.sort_values(['threshold', 'function_short'], ascending=[False, True])
    new.to_csv(stats_csv, index=False)
    return stats_csv


def write_runtime_json(out_dir: Path, config: dict, all_stats: list) -> Path:
    """Write `runtime.json` into `out_dir`: the run config + a per-run timing rollup.

    Both cluster builders call this once per threshold, so `out_dir` is that
    threshold's `t<NN>` dir and `runtime.json` lands next to that threshold's cluster
    parquets -- a self-contained record of how that threshold was built (search /
    cluster knobs, functions, threshold). The timing summary is derived from
    `all_stats` (fresh runs only; cached rows have elapsed_seconds=None).

    Preserve-on-cache: if every run was a cache hit (nothing recomputed) and a
    runtime.json already exists, the existing file is kept rather than overwritten.
    That prior file holds the real build timing plus the args the threshold was
    actually built with; a purely-cached re-run would otherwise clobber it with zero
    timing and the re-run's (possibly different) args.
    """
    fresh = [float(r['elapsed_seconds']) for r in all_stats
             if not r['cached'] and r['elapsed_seconds'] is not None]
    runtime_json = Path(out_dir) / 'runtime.json'
    if not fresh and runtime_json.exists():
        return runtime_json  # nothing recomputed: keep the existing build record
    rt = dict(config)
    rt.update({
        'n_runs_total': len(all_stats),
        'n_cached': sum(1 for r in all_stats if r['cached']),
        'n_fresh': len(fresh),
        'fresh_elapsed_seconds_total': sum(fresh),
        'fresh_elapsed_seconds_median': float(pd.Series(fresh).median()) if fresh else None,
        'fresh_elapsed_seconds_max': max(fresh) if fresh else None,
        'per_run': [{k: r.get(k) for k in (
            'function_short', 'threshold', 'alphabet', 'n_sequences',
            'n_clusters', 'elapsed_seconds', 'cached')} for r in all_stats],
    })
    runtime_json.write_text(json.dumps(rt, indent=2))
    return runtime_json
