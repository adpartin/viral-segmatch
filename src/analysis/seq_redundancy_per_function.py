"""Pre-clustering redundancy assessment per protein function or segment.

TODO: rename this script to build_mmseqs_clusters.py

Step 1 of the cluster-disjoint splits plan
(`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`):
characterize the within-function sequence redundancy at several mmseqs2 identity
thresholds. The cluster-size distribution per (function, threshold) decides which
thresholds are feasible for cluster-disjoint routing (a threshold that collapses
too much of a function into one giant cluster makes the partition trivial / forces
unacceptable pair-drops).

Supports two alphabets, both clustered with `mmseqs easy-linclust` since
2026-05-22 (symmetric algorithm choice — see
`docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md`
for why we no longer use easy-cluster on aa):
  - aa: clusters `prot_seq` from `protein_final.parquet` (Stage 1 output).
        `--dbtype 1` (amino acid alphabet).
  - nt: clusters `cds_dna` from `cds_final.parquet` (Stage 1.5 output from
        `src/preprocess/extract_cds_dna.py`). `--dbtype 2` (nucleotide
        alphabet).

The wrapper at `src/utils/clustering_utils.py::run_mmseqs_easy_clust`
pins the decision-relevant mmseqs flags explicitly (--cluster-mode 0,
--seq-id-mode 0, --similarity-type 2, -e 0.001, in addition to the
identity / coverage / dbtype flags); see that wrapper's docstring for
the full pinned set + rationale.

This produces the per-function cluster lookups that the routing
helper (`src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`)
consumes downstream.

Typical out_root values:
  data/processed/flu/{version}/clusters_aa  (aa; renamed from `clusters/` on 2026-05-15)
  data/processed/flu/{version}/clusters_nt  (nt)

Artifact layout:
  <out_root>/
    fasta/<short_name>.fasta            (one per function; reused across thresholds)
    id<th>/<short_name>_cluster.parquet (one per (function, threshold))
    id<th>/combined_cluster.parquet     (concatenation of per-function parquets)

CLI:
    # aa redundancy sweep (easy-linclust is the default since 2026-05-22).
    python -m src.analysis.seq_redundancy_per_function \\
        --protein_final data/processed/flu/July_2025/protein_final.parquet \\
        --out_root      data/processed/flu/July_2025/clusters_aa \\
        --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85 0.80 \\
        --threads 8

    # nt redundancy sweep (same default).
    python -m src.analysis.seq_redundancy_per_function \\
        --cds_final  data/processed/flu/July_2025/cds_final.parquet \\
        --out_root   data/processed/flu/July_2025/clusters_nt \\
        --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85 0.80 \\
        --threads 8

    # The default is symmetric. Pass --algorithm cluster explicitly if you
    # want easy-cluster's 3-round cascade for a one-off comparison.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.clustering_utils import (  # noqa: E402
    cluster_size_distribution,
    compute_seq_hash,
    export_function_fasta,
    parse_cluster_tsv,
    run_mmseqs_easy_clust,
)
from src.utils.config_hydra import load_function_metadata  # noqa: E402


# Function-name <-> short alias and canonical orderings, loaded from
# conf/virus/flu.yaml so the script stays in sync with the rest of the
# pipeline. Covers every protein in `function_short_names` (currently 20:
# 9 core + 11 aux). `--functions` defaults to the 8 ML-relevant majors
# (`selected_functions`).
_FLU_YAML = PROJECT_ROOT / 'conf' / 'virus' / 'flu.yaml'
_FLU_META = load_function_metadata(_FLU_YAML)
FUNCTION_TO_SHORT = _FLU_META.function_to_short
SHORT_TO_FUNCTION = _FLU_META.short_to_function
# Canonical row order within redundancy_summary.md tables: segment-ordered
# (PB2 first), with each segment's primary product listed before its
# alternative-reading-frame products. Sourced from `protein_order`.
SHORT_CANONICAL_ORDER = _FLU_META.short_canonical_order

# Flu A canonical segment id per short name. Segments 7 and 8 host two
# proteins each (M1+M2 on 7, NS1+NEP on 8). Used by write_results_markdown
# to add a Segment column to the per-threshold tables and to sort rows by
# segment order (PB2 first, NEP last) instead of alphabetical. Not in the
# YAML in this form (the YAML's segment_mapping is full-name keyed); kept
# inline here. Covers the 10 historically-relevant shorts; extra functions
# (e.g., PB1-F2, HA1, HA2) get no segment column entry and fall through
# to '' in the markdown.
SHORT_TO_SEGMENT = {
    'PB2': 1, 'PB1': 2, 'PA': 3, 'HA': 4, 'NP': 5, 'NA': 6,
    'M1':  7, 'M2':  7, 'NS1': 8, 'NEP': 8,
}


def _threshold_label(threshold: float) -> str:
    """Format float threshold as a stable directory label, e.g. 0.95 -> 'id095'."""
    # TODO: prefix with `f` rather than `id`
    pct = int(round(threshold * 100))
    return f"id{pct:03d}"


def cluster_one_function_one_threshold(
    df: pd.DataFrame,
    short_name: str,
    threshold: float,
    out_root: Path,
    threads: Optional[int] = 16,
    force: bool = False,
    alphabet: str = 'aa',
    algorithm: str = 'linclust',
    mmseqs_bin: Optional[str] = None,
    ) -> dict:
    """Run mmseqs at one threshold on the FASTA for one function.

    Caches the FASTA per function (re-used across thresholds).
    Caches the cluster parquet per (function, threshold); skip if present unless force=True.

    Args:
        df: input rows. For `alphabet='aa'` must contain `function`,
            `prot_seq`, `seq_hash` (pre-hashed by caller via
            `compute_seq_hash`; protein_final.parquet does not carry
            seq_hash so callers must add it after load). For
            `alphabet='nt'` must contain `function`, `cds_dna`,
            `cds_dna_hash` (from `cds_final.parquet`).
        alphabet: 'aa' (default) or 'nt'. Selects the unified exporter's
            seq/hash column pair and triggers `--dbtype 2` (nt) for the
            mmseqs invocation. `--search-type 3` is NOT a valid flag on
            easy-cluster/easy-linclust in mmseqs 18; the original plan
            referenced it, but the current code uses `--dbtype 2` instead.
        threads: int for the number of threads; None defers to mmseqs
            default (all cores).

    Returns the redundancy stats dict for this (function, threshold).
    """
    if alphabet not in {'aa', 'nt'}:
        raise ValueError(f"alphabet must be 'aa' or 'nt', got {alphabet!r}")
    if short_name not in SHORT_TO_FUNCTION:
        raise KeyError(f"Unknown short_name={short_name!r}. Known: {sorted(SHORT_TO_FUNCTION)}")
    full_name = SHORT_TO_FUNCTION[short_name]

    out_root = Path(out_root)
    fasta_dir = out_root / 'fasta'
    fasta_path = fasta_dir / f"{short_name}.fasta"

    threshold_dir = out_root / _threshold_label(threshold)
    threshold_dir.mkdir(parents=True, exist_ok=True)
    cluster_parquet = threshold_dir / f"{short_name}_cluster.parquet"
    log_path = threshold_dir / f"{short_name}_mmseqs.log"
    tmp_dir = threshold_dir / f"{short_name}_tmp"
    out_prefix = threshold_dir / f"{short_name}"

    # FASTA export (cached across thresholds). The unified exporter
    # handles both alphabets; only the print label changes.
    if not fasta_path.exists() or force:
        export_stats = export_function_fasta(df, full_name, alphabet, fasta_path)
        unit = 'unique seqs' if alphabet == 'aa' else 'unique CDS'
        ambig_label = 'contain X' if alphabet == 'aa' else 'contain non-ACGT'
        print(f"  [{short_name}] FASTA ({alphabet}): {export_stats['n_uniq_seqs']:,} "
              f"{unit} ({export_stats['n_with_ambiguity']:,} {ambig_label})")
    else:
        print(f"  [{short_name}] FASTA cached at {fasta_path.name}")

    # mmseqs run
    if cluster_parquet.exists() and not force:
        lookup = pd.read_parquet(cluster_parquet)
        elapsed_seconds = None  # cached — no fresh runtime
        cached = True
        print(f"  [{short_name} @ {threshold:.2f}] cluster parquet cached "
              f"({len(lookup):,} rows, {lookup['cluster_id'].nunique():,} clusters)")
    else:
        t0 = time.time()
        result = run_mmseqs_easy_clust(
            fasta_path=fasta_path,
            out_prefix=out_prefix,
            tmp_dir=tmp_dir,
            min_seq_id=float(threshold),
            coverage=0.8,
            cov_mode=0,
            threads=threads,
            log_path=log_path,
            alphabet=alphabet,
            algorithm=algorithm,
            mmseqs_bin=mmseqs_bin,
        )
        elapsed_seconds = time.time() - t0
        cached = False
        lookup = parse_cluster_tsv(result.cluster_tsv, cluster_id_prefix=short_name)
        lookup['function'] = full_name
        lookup['function_short'] = short_name
        lookup['threshold'] = float(threshold)
        lookup['alphabet'] = alphabet
        lookup.to_parquet(cluster_parquet, index=False)
        print(f"  [{short_name} @ {threshold:.2f}] clustered "
              f"{len(lookup):,} seqs -> {lookup['cluster_id'].nunique():,} clusters "
              f"in {elapsed_seconds:.1f}s")

    # Stats
    dist = cluster_size_distribution(lookup[['cluster_id']])
    dist.update({
        'function': full_name,
        'function_short': short_name,
        'threshold': float(threshold),
        'alphabet': alphabet,
        'algorithm': algorithm,
        'elapsed_seconds': elapsed_seconds,
        'cached': cached,
        'cluster_parquet': str(cluster_parquet),
    })
    return dist


def aggregate_combined_lookup(out_root: Path, threshold: float, function_shorts: list) -> Path:
    """Concatenate per-function cluster parquets at one threshold into a single parquet.

    Output: out_root/<threshold_label>/combined_cluster.parquet
    """
    threshold_dir = Path(out_root) / _threshold_label(threshold)
    parts = []
    for short in function_shorts:
        p = threshold_dir / f"{short}_cluster.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing per-function parquet: {p}")
        parts.append(pd.read_parquet(p))
    combined = pd.concat(parts, ignore_index=True)
    combined_path = threshold_dir / 'combined_cluster.parquet'
    combined.to_parquet(combined_path, index=False)
    return combined_path


def write_results_markdown(
    out_md: Path,
    stats_df: pd.DataFrame,
    protein_final_path: str,
    alphabet: str = 'aa',
    algorithm: str = 'linclust',
    ) -> None:
    """Write a human-readable markdown table per threshold."""
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    today = time.strftime('%Y-%m-%d')
    alphabet_label = 'aa' if alphabet == 'aa' else 'nt (CDS DNA)'
    subcmd = 'easy-cluster' if algorithm == 'cluster' else 'easy-linclust'
    dbtype_flag = ' --dbtype 2' if alphabet == 'nt' else ''

    lines = []
    lines.append(f"# Per-function redundancy ({alphabet_label}) — mmseqs2 sweep")
    lines.append("")
    lines.append(f"**Date.** {today}.")
    lines.append(f"**Input.** `{protein_final_path}`.")
    lines.append(f"**Alphabet.** {alphabet_label}.")
    lines.append(f"**Tool.** mmseqs2 `{subcmd} --min-seq-id <th> -c 0.8 --cov-mode 0{dbtype_flag}`.")
    lines.append(f"**Script.** `src/analysis/seq_redundancy_per_function.py`.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    if alphabet == 'aa':
        lines.append(f"For each major protein function, dedup `prot_seq` "
                     f"on md5(`prot_seq.rstrip('*')`), export to FASTA, and cluster "
                     f"at multiple aa-identity thresholds with mmseqs2 `{subcmd}`. "
                     f"`X` residues are left in place (mmseqs handles them natively); "
                     f"internal `*` rows would be dropped but none exist in this corpus.")
    else:
        sensitivity_note = (
            'linclust (linear-time, less sensitive than easy-cluster) was '
            'chosen for speed on the longer nt sequences. Cross-algorithm '
            'equivalence on this corpus has not been independently verified.'
            if algorithm == 'linclust' else
            'easy-cluster (sensitive cascaded clustering) was used.'
        )
        lines.append(f"For each major protein function, dedup `cds_dna` on "
                     f"`cds_dna_hash` (md5 of the CDS DNA), export to FASTA, and "
                     f"cluster at multiple nt-identity thresholds with mmseqs2 "
                     f"`{subcmd} --dbtype 2`. IUPAC ambiguity codes (N, R, Y, "
                     f"...) are left in place — mmseqs scores them natively. "
                     f"CDS is reconstructed by `src/preprocess/extract_cds_dna.py` "
                     f"from Stage 1 outputs (validated via translate-back). "
                     f"{sensitivity_note}")
    lines.append("")
    lines.append("## Results — cluster-size distribution per (function, threshold)")
    lines.append("")
    # Order column data: Segment goes leftmost; threshold is the per-table
    # header value, not a column; remaining columns are the actual stats.
    data_cols = ['n_sequences', 'n_clusters', 'largest_cluster',
                 'p99_cluster_size', 'p90_cluster_size',
                 'median_cluster_size', 'fraction_singletons']
    header_cols = ['Segment', 'function_short'] + data_cols
    # Iterate strict-to-loose (id100 first, id080 last) to mirror the
    # cluster-collapse-as-threshold-relaxes narrative used in
    # docs/methods/clusters.md §8.
    sort_index = {s: i for i, s in enumerate(SHORT_CANONICAL_ORDER)}
    for th, sub in stats_df.sort_values('threshold', ascending=False).groupby('threshold', sort=False):
        lines.append(f"### threshold = {th:.2f}")
        lines.append("")
        tbl = sub.copy()
        # Sort by segment order (PB2 first, NEP last). Functions absent
        # from SHORT_CANONICAL_ORDER get pushed to the end alphabetically.
        tbl['_sort'] = tbl['function_short'].map(
            lambda s: sort_index.get(s, len(SHORT_CANONICAL_ORDER))
        )
        tbl = tbl.sort_values(['_sort', 'function_short']).drop(columns='_sort')
        tbl['fraction_singletons'] = tbl['fraction_singletons'].map(lambda x: f"{x:.3f}")
        tbl['median_cluster_size'] = tbl['median_cluster_size'].astype(int)
        lines.append("| " + " | ".join(header_cols) + " |")
        # Right-align all numeric columns; left-align function_short.
        align = ['---:', '---'] + ['---:'] * len(data_cols)
        lines.append("|" + "|".join(align) + "|")
        for _, row in tbl.iterrows():
            seg = SHORT_TO_SEGMENT.get(row['function_short'], '')
            row_cells = [str(seg), str(row['function_short'])]
            for c in data_cols:
                v = row[c]
                if isinstance(v, float):
                    row_cells.append(f"{v:.3f}")
                else:
                    row_cells.append(f"{v:,}" if isinstance(v, int) and v > 0 else str(v))
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    lines.append("## Reading the table")
    lines.append("")
    lines.append("- `n_sequences`: unique protein sequences input to clustering "
                 "(constant across thresholds for a given function).")
    lines.append("- `n_clusters`: clusters produced at this threshold. Smaller = "
                 "more aggressive collapse.")
    lines.append("- `largest_cluster`: dominant cluster size. If this exceeds the "
                 "max per-split capacity (10% of n_pairs at 80/10/10), the routing "
                 "is forced.")
    lines.append("- `fraction_singletons`: clusters of size 1 / total clusters. "
                 "Higher = more sequences with no near-neighbor.")
    lines.append("")
    lines.append("## Related")
    lines.append("")
    lines.append("- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — "
                 "parent plan (Experiment B).")
    lines.append("- `docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md` — "
                 "the diagnostic that motivated this sweep.")
    lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"Wrote results doc: {out_md}")


def main() -> None:
    p = argparse.ArgumentParser(description="Per-function mmseqs2 redundancy assessment.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--protein_final',
                     help='aa-mode input: path to protein_final.parquet (or .csv).')
    src.add_argument('--cds_final',
                     help='nt-mode input: path to cds_final.parquet (built by '
                          'src/preprocess/extract_cds_dna.py). Implies --alphabet nt.')
    p.add_argument('--alphabet',
                   choices=['aa', 'nt'], default=None,
                   help='Sequence alphabet (default: aa for --protein_final, nt for --cds_final).')
    p.add_argument('--out_root',
                   required=True,
                   help='Output directory root (e.g. data/processed/flu/July_2025/clusters_aa or '
                        'clusters_nt).')
    p.add_argument('--thresholds',
                   nargs='+', type=float, required=True,
                   help='Identity thresholds (e.g. 1.00 0.99 0.95 0.90 0.80).')
    p.add_argument('--functions',
                   nargs='+', default=_FLU_META.selected_short_names,
                   help=(f'Function short names to cluster. Default: the '
                         f'{len(_FLU_META.selected_short_names)} ML-relevant majors from '
                         f'conf/virus/flu.yaml::selected_functions '
                         f'({" ".join(_FLU_META.selected_short_names)}).'))
    p.add_argument('--threads', type=int, default=16,
                   help='mmseqs --threads (default: 16, shared-machine-friendly).')
    p.add_argument('--mmseqs_bin',
                   default=None,
                   help='Path/name of the mmseqs binary. Default: $MMSEQS_BIN, then "mmseqs" on PATH. '
                        'On Lambda set MMSEQS_BIN=/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs '
                        'so the dedicated env is used without putting mmseqs on PATH.')
    p.add_argument('--algorithm',
                   choices=['cluster', 'linclust'], default='linclust',
                   help='mmseqs subcommand. linclust=easy-linclust (default since '
                        '2026-05-22; linear time, single-pass — used on both '
                        'alphabets for cross-alphabet consistency). cluster=easy-cluster '
                        '(sensitive 3-round cascade; retained as an option but no '
                        'longer the default — the asymmetric easy-cluster/easy-linclust '
                        'choice conflated algorithm with alphabet in clusters.md '
                        '§8/§9; see docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md).')
    p.add_argument('--force', action='store_true', help='Recompute even if cached.')
    p.add_argument('--results_md',
                   default=None,
                   help='Path to results markdown. Default: <out_root>/redundancy_summary.md '
                        '(alongside the stats CSV; not under docs/).')
    p.add_argument('--no_combined',
                   action='store_true',
                   help='Skip writing combined_cluster.parquet per threshold.')
    args = p.parse_args()

    if args.protein_final and not args.alphabet:
        args.alphabet = 'aa'
    if args.cds_final and not args.alphabet:
        args.alphabet = 'nt'
    if args.protein_final and args.alphabet == 'nt':
        raise SystemExit("--protein_final is aa-only; use --cds_final for nt mode.")
    if args.cds_final and args.alphabet == 'aa':
        raise SystemExit("--cds_final is nt-only; use --protein_final for aa mode.")

    in_path = Path(args.protein_final or args.cds_final)
    print(f"Loading {in_path}  (alphabet={args.alphabet}) ...")
    t0 = time.time()
    if args.alphabet == 'aa':
        usecols = ['function', 'prot_seq']
    else:
        usecols = ['function', 'cds_dna', 'cds_dna_hash']
    if in_path.suffix == '.csv':
        df = pd.read_csv(in_path, usecols=usecols)
    else:
        df = pd.read_parquet(in_path, columns=usecols)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Pre-hash aa input. The unified exporter requires the hash column to
    # be present (matches the nt path where cds_final.parquet carries
    # cds_dna_hash). protein_final.parquet does NOT carry seq_hash, so we
    # compute it here once and reuse across thresholds.
    if args.alphabet == 'aa':
        t_hash = time.time()
        df['seq_hash'] = df['prot_seq'].map(compute_seq_hash)
        print(f"  Hashed prot_seq -> seq_hash in {time.time()-t_hash:.1f}s "
              f"({df['seq_hash'].nunique():,} unique)")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    unknown = [f for f in args.functions if f not in SHORT_TO_FUNCTION]
    if unknown:
        raise SystemExit(f"Unknown function short names: {unknown}. "
                         f"Known: {sorted(SHORT_TO_FUNCTION)}")

    # Skip functions absent from the input (e.g., M2/NEP are not in
    # cds_final.parquet because the CDS extractor only covers the 8
    # majors). Avoids aborting mid-sweep on the first missing function.
    present_functions = set(df['function'].unique())
    skipped = [f for f in args.functions
               if SHORT_TO_FUNCTION[f] not in present_functions]
    if skipped:
        print(f"  NOTE: skipping {len(skipped)} function(s) with no rows in "
              f"this input: {skipped}")
        args.functions = [f for f in args.functions if f not in skipped]
        if not args.functions:
            raise SystemExit("No functions to process after skipping. "
                             "Check the input file's `function` column.")

    all_stats = []
    for threshold in args.thresholds:
        print(f"\n=== threshold = {threshold:.2f} ===")
        for short in args.functions:
            stats = cluster_one_function_one_threshold(
                df=df,
                short_name=short,
                threshold=threshold,
                out_root=out_root,
                threads=args.threads,
                force=args.force,
                alphabet=args.alphabet,
                algorithm=args.algorithm,
                mmseqs_bin=args.mmseqs_bin,
            )
            all_stats.append(stats)

        if not args.no_combined:
            cpath = aggregate_combined_lookup(out_root, threshold, args.functions)
            print(f"  combined parquet -> {cpath}")

    # Save the per-(function, threshold) stats CSV.
    # Append/merge with any prior CSV so re-running with a subset of
    # thresholds doesn't wipe out earlier sweep results. Re-running
    # the same (function_short, threshold) overwrites just those rows.
    # `keep_default_na=False, na_values=['']` is required when reading
    # because function_short='NA' (Neuraminidase) would otherwise be
    # parsed as NaN (see CLAUDE.md Conventions).
    stats_df_new = pd.DataFrame(all_stats)
    stats_csv = out_root / 'redundancy_stats.csv'
    if stats_csv.exists():
        prior = pd.read_csv(stats_csv, keep_default_na=False, na_values=[''])
        same_schema = set(prior.columns) == set(stats_df_new.columns)
        if same_schema:
            key_cols = ['function_short', 'threshold']
            keys_now = set(map(tuple, stats_df_new[key_cols].itertuples(index=False, name=None)))
            prior_keys = list(map(tuple, prior[key_cols].itertuples(index=False, name=None)))
            keep_mask = [k not in keys_now for k in prior_keys]
            prior = prior[keep_mask].reset_index(drop=True)
            stats_df = pd.concat([prior, stats_df_new], ignore_index=True)
            n_carried = len(prior)
            print(f"\nMerging stats CSV: carried {n_carried:,} prior rows + "
                  f"{len(stats_df_new):,} new rows = {len(stats_df):,} total.")
        else:
            print(f"\nWARNING: prior {stats_csv.name} has a different column "
                  f"schema; overwriting instead of merging.")
            stats_df = stats_df_new
    else:
        stats_df = stats_df_new
    stats_df = stats_df.sort_values(
        ['threshold', 'function_short'], ascending=[False, True]
    ).reset_index(drop=True)
    stats_df.to_csv(stats_csv, index=False)
    print(f"Wrote stats CSV: {stats_csv}")

    # Save a runtime.json — per-(function, threshold) wall time + a small
    # rollup. Cached rows have elapsed_seconds = None and contribute to
    # n_cached. Lets future readers verify aa-vs-nt timing trade-offs from
    # data instead of recollection (and informs easy-cluster vs easy-linclust
    # choices on new corpora).
    fresh = [r for r in all_stats if not r['cached'] and r['elapsed_seconds'] is not None]
    fresh_secs = [float(r['elapsed_seconds']) for r in fresh]
    rt = {
        'alphabet': args.alphabet,
        'algorithm': args.algorithm,
        'threads': args.threads,
        'functions': list(args.functions),
        'thresholds': [float(t) for t in args.thresholds],
        'n_runs_total': len(all_stats),
        'n_cached': sum(1 for r in all_stats if r['cached']),
        'n_fresh': len(fresh),
        'fresh_elapsed_seconds_total': sum(fresh_secs),
        'fresh_elapsed_seconds_min': min(fresh_secs) if fresh_secs else None,
        'fresh_elapsed_seconds_median': float(pd.Series(fresh_secs).median()) if fresh_secs else None,
        'fresh_elapsed_seconds_max': max(fresh_secs) if fresh_secs else None,
        'per_run': [
            {
                'function_short': r['function_short'],
                'threshold': r['threshold'],
                'alphabet': r['alphabet'],
                'algorithm': r['algorithm'],
                'n_sequences': r['n_sequences'],
                'n_clusters': r['n_clusters'],
                'elapsed_seconds': r['elapsed_seconds'],
                'cached': r['cached'],
            }
            for r in all_stats
        ],
    }
    runtime_json = out_root / 'runtime.json'
    with runtime_json.open('w') as f:
        json.dump(rt, f, indent=2)
    print(f"Wrote runtime JSON: {runtime_json} "
          f"(n_fresh={rt['n_fresh']}, n_cached={rt['n_cached']}, "
          f"total fresh = {rt['fresh_elapsed_seconds_total']:.1f}s)")

    # Markdown report. Defaults to <out_root>/redundancy_summary.md so
    # the auto-generated markdown lives next to the stats CSV and the
    # cluster parquets, NOT in docs/results/. CLAUDE.md "Aggregator
    # Output Convention" reserves docs/ for hand-authored writeups —
    # machine outputs belong with the data they describe.
    if args.results_md is None:
        results_md = out_root / 'redundancy_summary.md'
    else:
        results_md = Path(args.results_md)
    write_results_markdown(results_md, stats_df, str(in_path),
                           alphabet=args.alphabet, algorithm=args.algorithm)


if __name__ == '__main__':
    main()
