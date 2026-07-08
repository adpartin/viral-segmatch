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
