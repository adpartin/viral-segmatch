"""Pre-clustering redundancy assessment per protein function.

Step 1 of the cluster-disjoint splits plan
(`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`):
characterize the within-function protein-sequence redundancy of
`protein_final` at several mmseqs2 identity thresholds. The cluster-size
distribution per (function, threshold) decides which thresholds are
feasible for cluster-disjoint routing (a threshold that collapses too
much of a function into one giant cluster makes the partition trivial /
forces unacceptable pair-drops).

Side effect: produces the per-function cluster lookups that the routing
helper consumes downstream. Two artifact layouts are written:

  data/processed/flu/{version}/clusters/
    fasta/<short_name>.fasta           (one per function; reused across thresholds)
    id<th>/<short_name>_cluster.parquet (one per (function, threshold))
    id<th>/combined_cluster.parquet     (concatenation of per-function parquets)

CLI:
    python -m src.analysis.protein_redundancy_per_function \
        --protein_final data/processed/flu/July_2025/protein_final.parquet \
        --out_root data/processed/flu/July_2025/clusters \
        --thresholds 1.00 0.99 0.95 0.90 0.80 \
        --functions HA NA PB2 PB1 PA NP M1 M2 NEP NS1 \
        --threads 8
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
    export_function_fasta,
    run_mmseqs_easy_cluster,
    parse_cluster_tsv,
    cluster_size_distribution,
)


# Function-name → short alias (mirrors conf/virus/flu.yaml::function_short_names).
# Only the 9 major core functions + NS1 are listed by default; auxiliary functions
# (PB1-F2 etc.) can be added by extending --functions on the CLI.
FUNCTION_TO_SHORT = {
    'RNA-dependent RNA polymerase PB2 subunit': 'PB2',
    'RNA-dependent RNA polymerase catalytic core PB1 subunit': 'PB1',
    'RNA-dependent RNA polymerase PA subunit': 'PA',
    'Hemagglutinin precursor': 'HA',
    'Nucleocapsid protein': 'NP',
    'Neuraminidase protein': 'NA',
    'Matrix protein 1': 'M1',
    'M2 ion channel': 'M2',
    'Nuclear export protein': 'NEP',
    'Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor': 'NS1',
}
SHORT_TO_FUNCTION = {v: k for k, v in FUNCTION_TO_SHORT.items()}


def _threshold_label(threshold: float) -> str:
    """Format float threshold as a stable directory label, e.g. 0.95 -> 'id095'."""
    pct = int(round(threshold * 100))
    return f"id{pct:03d}"


def cluster_one_function_one_threshold(
    prot_df: pd.DataFrame,
    short_name: str,
    threshold: float,
    out_root: Path,
    threads: Optional[int] = None,
    force: bool = False,
) -> dict:
    """Run mmseqs at one threshold on the FASTA for one function.

    Caches the FASTA per function (re-used across thresholds).
    Caches the cluster parquet per (function, threshold); skip if present unless force=True.

    Returns the redundancy stats dict for this (function, threshold).
    """
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

    # FASTA export (cached across thresholds)
    if not fasta_path.exists() or force:
        export_stats = export_function_fasta(prot_df, full_name, fasta_path)
        print(f"  [{short_name}] FASTA: {export_stats['n_unique_sequences']:,} unique seqs "
              f"({export_stats['n_with_x']:,} contain X)")
    else:
        print(f"  [{short_name}] FASTA cached at {fasta_path.name}")

    # mmseqs run
    if cluster_parquet.exists() and not force:
        lookup = pd.read_parquet(cluster_parquet)
        print(f"  [{short_name} @ {threshold:.2f}] cluster parquet cached "
              f"({len(lookup):,} rows, {lookup['cluster_id'].nunique():,} clusters)")
    else:
        t0 = time.time()
        result = run_mmseqs_easy_cluster(
            fasta_path=fasta_path,
            out_prefix=out_prefix,
            tmp_dir=tmp_dir,
            min_seq_id=float(threshold),
            coverage=0.8,
            cov_mode=0,
            threads=threads,
            log_path=log_path,
        )
        lookup = parse_cluster_tsv(result.cluster_tsv, cluster_id_prefix=short_name)
        lookup['function'] = full_name
        lookup['function_short'] = short_name
        lookup['threshold'] = float(threshold)
        lookup.to_parquet(cluster_parquet, index=False)
        print(f"  [{short_name} @ {threshold:.2f}] clustered "
              f"{len(lookup):,} seqs -> {lookup['cluster_id'].nunique():,} clusters "
              f"in {time.time()-t0:.1f}s")

    # Stats
    dist = cluster_size_distribution(lookup[['cluster_id']])
    dist.update({
        'function': full_name,
        'function_short': short_name,
        'threshold': float(threshold),
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
) -> None:
    """Write a human-readable markdown table per threshold."""
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    today = time.strftime('%Y-%m-%d')

    lines = []
    lines.append("# Protein redundancy per function — mmseqs2 sweep")
    lines.append("")
    lines.append(f"**Date.** {today}.")
    lines.append(f"**Input.** `{protein_final_path}`.")
    lines.append(f"**Tool.** mmseqs2 `easy-cluster --min-seq-id <th> -c 0.8 --cov-mode 0`.")
    lines.append(f"**Script.** `src/analysis/protein_redundancy_per_function.py`.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("For each major protein function, dedup `prot_seq` "
                 "on md5(`prot_seq.rstrip('*')`), export to FASTA, and cluster "
                 "at multiple aa-identity thresholds with mmseqs2 `easy-cluster`. "
                 "`X` residues are left in place (mmseqs handles them natively); "
                 "internal `*` rows would be dropped but none exist in this corpus.")
    lines.append("")
    lines.append("## Results — cluster-size distribution per (function, threshold)")
    lines.append("")
    cols = ['function_short', 'threshold', 'n_sequences', 'n_clusters',
            'largest_cluster', 'p99_cluster_size', 'p90_cluster_size',
            'median_cluster_size', 'fraction_singletons']
    for th, sub in stats_df.groupby('threshold'):
        lines.append(f"### threshold = {th:.2f}")
        lines.append("")
        tbl = sub[cols].copy().sort_values('function_short')
        tbl['fraction_singletons'] = tbl['fraction_singletons'].map(lambda x: f"{x:.3f}")
        tbl['median_cluster_size'] = tbl['median_cluster_size'].astype(int)
        lines.append("| " + " | ".join(c for c in cols if c != 'threshold') + " |")
        lines.append("|" + "|".join(["---:"] * (len(cols) - 1)) + "|")
        for _, row in tbl.iterrows():
            row_cells = []
            for c in cols:
                if c == 'threshold':
                    continue
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
    p.add_argument('--protein_final', required=True,
                   help='Path to protein_final.parquet (or .csv).')
    p.add_argument('--out_root', required=True,
                   help='Output directory root (e.g. data/processed/flu/July_2025/clusters).')
    p.add_argument('--thresholds', nargs='+', type=float, required=True,
                   help='Identity thresholds (e.g. 1.00 0.99 0.95 0.90 0.80).')
    p.add_argument('--functions', nargs='+', default=['HA', 'NA', 'PB2', 'PB1', 'PA', 'NP', 'M1', 'M2', 'NEP', 'NS1'],
                   help='Function short names to cluster.')
    p.add_argument('--threads', type=int, default=None, help='mmseqs --threads.')
    p.add_argument('--force', action='store_true', help='Recompute even if cached.')
    p.add_argument('--results_md', default=None,
                   help='Path to results markdown (default: docs/results/<date>_protein_redundancy_per_function.md).')
    p.add_argument('--no_combined', action='store_true',
                   help='Skip writing combined_cluster.parquet per threshold.')
    args = p.parse_args()

    # Load protein_final
    pf_path = Path(args.protein_final)
    print(f"Loading {pf_path}...")
    t0 = time.time()
    if pf_path.suffix == '.csv':
        df = pd.read_csv(pf_path, usecols=['function', 'prot_seq'])
    else:
        df = pd.read_parquet(pf_path, columns=['function', 'prot_seq'])
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Validate function names
    unknown = [f for f in args.functions if f not in SHORT_TO_FUNCTION]
    if unknown:
        raise SystemExit(f"Unknown function short names: {unknown}. "
                         f"Known: {sorted(SHORT_TO_FUNCTION)}")

    # Sweep
    all_stats = []
    for threshold in args.thresholds:
        print(f"\n=== threshold = {threshold:.2f} ===")
        for short in args.functions:
            stats = cluster_one_function_one_threshold(
                prot_df=df,
                short_name=short,
                threshold=threshold,
                out_root=out_root,
                threads=args.threads,
                force=args.force,
            )
            all_stats.append(stats)

        if not args.no_combined:
            cpath = aggregate_combined_lookup(out_root, threshold, args.functions)
            print(f"  combined parquet -> {cpath}")

    # Save the per-(function, threshold) stats CSV
    stats_df = pd.DataFrame(all_stats)
    stats_csv = out_root / 'redundancy_stats.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nWrote stats CSV: {stats_csv}")

    # Markdown report
    if args.results_md is None:
        today = time.strftime('%Y-%m-%d')
        results_md = PROJECT_ROOT / 'docs' / 'results' / f"{today}_protein_redundancy_per_function.md"
    else:
        results_md = Path(args.results_md)
    write_results_markdown(results_md, stats_df, str(pf_path))


if __name__ == '__main__':
    main()
