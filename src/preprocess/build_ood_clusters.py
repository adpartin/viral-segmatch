"""Per-function OOD sequence clustering: connected components of the >=t/cov graph.

Builds per-(function, threshold) clusters that GUARANTEE "across clusters:
different" -- any two sequences in different clusters are < t identical (or
< 0.8 covered). A cluster is a connected component of the per-function
SIMILARITY graph (nodes = unique sequences; an edge = an mmseqs all-vs-all hit
with identity >= t and coverage >= 0.8), computed by union-find.

This is the input a cluster-disjoint (OOD) split needs: routing whole clusters
to one fold leaves no test sequence >= t identical to any train sequence.

Component here = single-segment SIMILARITY-graph connected component (a *cluster*
/ *mega-cluster*), NOT the bipartite CC / mega-CC of 2D-CD routing (see
docs/methods/glossary.md).

Contrast with build_mmseqs_clusters.py (set-cover / linclust, the default): that
path does NOT give the guarantee -- mmseqs easy-cluster fragments the graph. See
docs/plans/2026-07-08_single_segment_ood_clusters_plan.md.

Output layout mirrors build_mmseqs_clusters.py, under a separate `_ood` root:
  <out_root>/                (e.g. data/processed/flu/July_2025/clusters_aa_ood)
    fasta/<short>.fasta            (one per function; reused across thresholds)
    t<NN>/<short>_cluster.parquet  (<hash>, cluster_id, cluster_rep, function, ...)
    t<NN>/combined_cluster.parquet

CLI:
    python -m src.preprocess.build_ood_clusters \\
        --protein_final data/processed/flu/July_2025/protein_final.parquet \\
        --out_root      data/processed/flu/July_2025/clusters_aa_ood \\
        --thresholds 0.99 --functions M1 \\
        --mmseqs_bin /homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs

The `-s 7.5` prefilter is empirically complete at high `t`; pass `--exhaustive`
(mmseqs `--prefilter-mode 2`, nofilter) for a provably-complete all-vs-all.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.clustering_utils import (  # noqa: E402
    aggregate_combined_lookup,
    cluster_size_distribution,
    connected_components_from_hits,
    export_function_fasta,
    filter_present_functions,
    load_sequence_frame,
    read_fasta_hashes,
    run_mmseqs_search,
    threshold_label,
    write_or_merge_stats_csv,
)
from src.utils.config_hydra import load_function_metadata  # noqa: E402

_FLU_YAML = PROJECT_ROOT / 'conf' / 'virus' / 'flu.yaml'
_FLU_META = load_function_metadata(_FLU_YAML)
SHORT_TO_FUNCTION = _FLU_META.short_to_function


def cluster_one_function_one_threshold(
    df: pd.DataFrame,
    short_name: str,
    threshold: float,
    out_root: Path,
    *,
    alphabet: str,
    sensitivity: float,
    prefilter_mode: int | None,
    max_seqs: int | None,
    coverage: float,
    gpu: int,
    threads: int,
    mmseqs_bin: str | None,
    force: bool,
    ) -> dict:
    """Build OOD clusters for one (function, threshold): FASTA -> all-vs-all -> union-find CCs.

    Caches the FASTA per function and the cluster parquet per (function,
    threshold); re-use unless `force`. Returns the cluster-size stats dict.
    """
    full_name = SHORT_TO_FUNCTION[short_name]
    out_root = Path(out_root)
    fasta_path = out_root / 'fasta' / f"{short_name}.fasta"
    tdir = out_root / threshold_label(threshold)
    tdir.mkdir(parents=True, exist_ok=True)
    cluster_parquet = tdir / f"{short_name}_cluster.parquet"
    hits_tsv = tdir / f"{short_name}_hits.tsv"
    tmp_dir = tdir / f"{short_name}_tmp"
    log_path = tdir / f"{short_name}_mmseqs.log"

    # FASTA of unique sequences (cached across thresholds).
    if not fasta_path.exists() or force:
        stats = export_function_fasta(df, full_name, alphabet, fasta_path)
        print(f"  [{short_name}] FASTA ({alphabet}): {stats['n_uniq_seqs']:,} unique seqs")
    else:
        print(f"  [{short_name}] FASTA cached at {fasta_path.name}")

    if cluster_parquet.exists() and not force:
        lookup = pd.read_parquet(cluster_parquet)
        elapsed, cached = None, True
        print(f"  [{short_name} @ {threshold:.2f}] cluster parquet cached "
              f"({len(lookup):,} rows, {lookup['cluster_id'].nunique():,} clusters)")
    else:
        t0 = time.time()
        # Read the node set first so --max-seqs can't truncate the neighbour graph:
        # hold it >= the unique-seq count, else a dense function silently drops >=t
        # edges and fragments the components.
        nodes = read_fasta_hashes(fasta_path)
        eff_max_seqs = max(max_seqs, len(nodes)) if max_seqs else len(nodes)
        run_mmseqs_search(
            fasta_path, hits_tsv, tmp_dir, float(threshold),
            coverage=coverage, alphabet=alphabet, sensitivity=sensitivity,
            prefilter_mode=prefilter_mode, max_seqs=eff_max_seqs, gpu=gpu,
            threads=threads, mmseqs_bin=mmseqs_bin, log_path=log_path)
        # Clusters ARE the connected components of this hit graph, so no hit crosses a
        # cluster boundary by construction; verify_ood_clusters.py certifies the
        # guarantee end-to-end against an independent search.
        lookup = connected_components_from_hits(
            hits_tsv, nodes, alphabet=alphabet, cluster_id_prefix=short_name)
        lookup['function'] = full_name
        lookup['function_short'] = short_name
        lookup['threshold'] = float(threshold)
        lookup['alphabet'] = alphabet
        lookup.to_parquet(cluster_parquet, index=False)
        elapsed, cached = time.time() - t0, False
        print(f"  [{short_name} @ {threshold:.2f}] {len(lookup):,} seqs -> "
              f"{lookup['cluster_id'].nunique():,} clusters in {elapsed:.1f}s")

    dist = cluster_size_distribution(lookup[['cluster_id']])
    dist.update({
        'function': full_name, 'function_short': short_name,
        'threshold': float(threshold), 'alphabet': alphabet,
        'elapsed_seconds': elapsed, 'cached': cached,
        'cluster_parquet': str(cluster_parquet),
    })
    return dist


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-function OOD clustering (connected components of the >=t/cov graph).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--protein_final', help='aa input: protein_final.parquet (or .csv).')
    src.add_argument('--cds_dna_final', help='nt_cds input: cds_dna_final.parquet.')
    src.add_argument('--ctg_dna_final', help='nt_ctg input: ctg_dna_final.parquet.')
    p.add_argument('--function_source', default=None,
                   help='nt_ctg only: [assembly_id, genbank_ctg_id, function] source for '
                        'the contig->function join (default: sibling cds_dna_final.parquet).')
    p.add_argument('--out_root', required=True,
                   help='Output root, e.g. data/processed/flu/July_2025/clusters_aa_ood.')
    p.add_argument('--thresholds', nargs='+', type=float, required=True,
                   help='Identity thresholds, e.g. 0.99 0.98 0.95.')
    p.add_argument('--functions', nargs='+', default=_FLU_META.selected_short_names,
                   help=f'Function short names (default: the {len(_FLU_META.selected_short_names)} '
                        f'ML-relevant majors: {" ".join(_FLU_META.selected_short_names)}).')
    p.add_argument('--sensitivity', type=float, default=7.5,
                   help='mmseqs -s prefilter sensitivity (7.5=most sensitive; default). '
                        'Ignored when --exhaustive is set.')
    p.add_argument('--exhaustive', action='store_true',
                   help='Use --prefilter-mode 2 (nofilter) for a provably-complete all-vs-all '
                        'instead of the -s prefilter. Slower; use for a hard completeness check.')
    p.add_argument('--max_seqs', type=int, default=100000,
                   help='mmseqs --max-seqs (neighbors kept per query; keep >= the per-function '
                        'unique-seq count so no >=t edge is truncated). Default 100000.')
    p.add_argument('--coverage', type=float, default=0.8, help='mmseqs -c coverage (default 0.8).')
    p.add_argument('--gpu', type=int, default=0,
                   help='Use GPU (CUDA) for the search prefilter: 1=on (needs Ampere+/Hopper). '
                        'Pick the device with CUDA_VISIBLE_DEVICES.')
    p.add_argument('--threads', type=int, default=16, help='mmseqs --threads (default 16).')
    p.add_argument('--mmseqs_bin', default=None,
                   help='mmseqs binary path/name. Default: $MMSEQS_BIN, then "mmseqs" on PATH.')
    p.add_argument('--force', action='store_true', help='Recompute even if cached.')
    p.add_argument('--no_combined', action='store_true',
                   help='Skip writing combined_cluster.parquet per threshold.')
    args = p.parse_args()

    # Reject thresholds that collapse to the same tXXX directory label (percent
    # rounding) -- they would otherwise silently share/overwrite one cluster dir.
    labels = [threshold_label(t) for t in args.thresholds]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"--thresholds collide on directory labels {sorted(set(labels))}; "
                         "use distinct thresholds (percent granularity).")

    df, alphabet = load_sequence_frame(
        protein_final=args.protein_final, cds_dna_final=args.cds_dna_final,
        ctg_dna_final=args.ctg_dna_final, alphabet=None,
        function_source=args.function_source,
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    functions, skipped = filter_present_functions(df, args.functions, SHORT_TO_FUNCTION)
    if skipped:
        print(f"  NOTE: skipping {len(skipped)} function(s) with no rows in this input: {skipped}")
    if not functions:
        raise SystemExit("No functions to process. Check the input file's `function` column.")

    prefilter_mode = 2 if args.exhaustive else None
    all_stats = []
    for threshold in args.thresholds:
        print(f"\n=== threshold = {threshold:.2f} ===")
        for short in functions:
            all_stats.append(cluster_one_function_one_threshold(
                df, short, threshold, out_root, alphabet=alphabet,
                sensitivity=args.sensitivity, prefilter_mode=prefilter_mode,
                max_seqs=args.max_seqs, coverage=args.coverage, gpu=args.gpu,
                threads=args.threads, mmseqs_bin=args.mmseqs_bin, force=args.force))
        if not args.no_combined:
            print(f"  combined parquet -> {aggregate_combined_lookup(out_root, threshold, functions)}")

    # Per-(function, threshold) cluster-size stats, merged with any prior CSV so a
    # subset-threshold re-run doesn't drop earlier rows.
    stats_csv = write_or_merge_stats_csv(out_root, all_stats, 'cluster_stats.csv')
    print(f"\nWrote stats CSV: {stats_csv}")


if __name__ == '__main__':
    main()
