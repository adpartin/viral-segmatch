"""Unified per-(function, threshold) cluster builder: easy-linclust / easy-cluster / easy-search.

One driver, one swappable core. Two ORTHOGONAL axes (never a combined enum):
  - `--method {linclust, cluster, search}` — the mmseqs subcommand / clustering paradigm.
      linclust / cluster  = set-cover assignment via `run_mmseqs_easy_clust` + `parse_cluster_tsv`.
      search              = connected components of the all-vs-all >=t/cov graph, via
                            `run_mmseqs_search` + union-find (`connected_components_from_hits`).
                            This is the OOD path: whole-cluster splits then guarantee no test
                            sequence links to any train sequence (across-cluster separation).
  - `--cluster-mode {0,1,2}` — a parameter of linclust/cluster ONLY (rejected for search).

The shared skeleton (input load + function filter, the per-(function, threshold) cache wrapper,
the threshold sweep, combined-parquet aggregation, runtime.json, and the merged cluster_stats.csv)
lives here once; only the actual clustering call differs by method. `build_mmseqs_clusters.py`
(default `--method linclust`) and `build_ood_clusters.py` (default `--method search`) are thin
back-compat shims over this file.

CLI help convention: each arg is tagged `[mmseqs2: --flag]` (passed through to mmseqs; source of
truth is `mmseqs <sub> --help`, not the web docs) or `[builder]` (our orchestration).

Artifact layout (under --out_root):
    fasta/<short>.fasta                 one per function; reused across thresholds
    t<NN>/<short>_cluster.parquet       (<hash>, cluster_id, cluster_rep, function, ...)
    t<NN>/<short>_hits.parquet          search only, unless --delete_hits
    t<NN>/combined_cluster.parquet      concatenation of per-function parquets (unless --no_combined)
    t<NN>/runtime.json                  per-threshold build config + timing rollup
    cluster_stats.csv                   per-(function, threshold) size stats (merged across runs)

CLI:
    python -m src.preprocess.build_clusters --method linclust \\
        --cds_dna_final data/processed/flu/July_2025/cds_dna_final.parquet \\
        --out_root      data/processed/flu/July_2025/clusters_nt_cds_cm0 \\
        --thresholds 0.99 0.98 0.97 0.96 0.95 --threads 64
    python -m src.preprocess.build_clusters --method search \\
        --protein_final data/processed/flu/July_2025/protein_final.parquet \\
        --out_root      data/processed/flu/July_2025/clusters_aa_ood \\
        --thresholds 0.99 --functions HA NA --threads 64 --scratch_dir /tmp/ood_scratch
"""
from __future__ import annotations

import argparse
import shutil
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
    parse_cluster_tsv,
    read_fasta_hashes,
    read_hits_tsv,
    run_mmseqs_easy_clust,
    run_mmseqs_search,
    threshold_label,
    write_or_merge_stats_csv,
    write_runtime_json,
)
from src.utils.config_hydra import load_function_metadata  # noqa: E402

_FLU_YAML = PROJECT_ROOT / 'conf' / 'virus' / 'flu.yaml'
_FLU_META = load_function_metadata(_FLU_YAML)
SHORT_TO_FUNCTION = _FLU_META.short_to_function

_METHODS = ('linclust', 'cluster', 'search')


def cluster_one_function_one_threshold(
    df: pd.DataFrame,
    short_name: str,
    threshold: float,
    out_root: Path,
    *,
    method: str,
    alphabet: str,
    coverage: float,
    sensitivity,
    max_seqs,
    threads,
    mmseqs_bin,
    force: bool,
    cluster_mode: int = 0,
    single_step_clustering: bool = False,
    gpu: int = 0,
    prefilter_mode=None,
    delete_hits: bool = False,
    scratch_dir=None,
    ) -> dict:
    """Build (or load cached) clusters for one (function, threshold); return the size-stats dict.

    Shared cache wrapper for all methods: caches the FASTA per function and the cluster parquet
    per (function, threshold). The only method-specific step is the clustering itself:
      - linclust/cluster: `run_mmseqs_easy_clust` (set-cover) -> `parse_cluster_tsv`.
      - search:           `run_mmseqs_search` (all-vs-all) -> `connected_components_from_hits`.
    """
    full_name = SHORT_TO_FUNCTION[short_name]
    out_root = Path(out_root)
    fasta_path = out_root / 'fasta' / f"{short_name}.fasta"
    tdir = out_root / threshold_label(threshold)
    tdir.mkdir(parents=True, exist_ok=True)
    cluster_parquet = tdir / f"{short_name}_cluster.parquet"
    log_path = tdir / f"{short_name}_mmseqs.log"

    # Transient scratch. search routes the (huge) hits TSV + mmseqs tmp through --scratch_dir
    # (local disk) when given; set-cover keeps its small tmp next to the parquet on NFS.
    if method == 'search':
        scr = (Path(scratch_dir) / f"{threshold_label(threshold)}_{short_name}") if scratch_dir else tdir
        scr.mkdir(parents=True, exist_ok=True)
        hits_tsv = scr / f"{short_name}_hits.tsv"
        tmp_dir = scr / f"{short_name}_tmp"
        hits_parquet = tdir / f"{short_name}_hits.parquet"
    else:
        tmp_dir = tdir / f"{short_name}_tmp"
        out_prefix = tdir / f"{short_name}"

    # FASTA of unique sequences (cached across thresholds).
    if not fasta_path.exists() or force:
        export_stats = export_function_fasta(df, full_name, alphabet, fasta_path)
        print(f"  [{short_name}] FASTA ({alphabet}): {export_stats['n_uniq_seqs']:,} unique seqs")
    else:
        print(f"  [{short_name}] FASTA cached at {fasta_path.name}")

    if cluster_parquet.exists() and not force:
        lookup = pd.read_parquet(cluster_parquet)
        elapsed, cached = None, True
        print(f"  [{short_name} @ {threshold:.2f}] cluster parquet cached "
              f"({len(lookup):,} rows, {lookup['cluster_id'].nunique():,} clusters)")
    else:
        t0 = time.time()
        if method == 'search':
            # Read the node set first so --max-seqs can't truncate the neighbour graph:
            # hold it >= the unique-seq count, else a dense function silently drops >=t
            # edges and fragments the components.
            nodes = read_fasta_hashes(fasta_path)
            eff_max_seqs = max(max_seqs, len(nodes)) if max_seqs else len(nodes)
            run_mmseqs_search(
                fasta_path, hits_tsv, tmp_dir, float(threshold),
                coverage=coverage, alphabet=alphabet, sensitivity=sensitivity,
                prefilter_mode=prefilter_mode, max_seqs=eff_max_seqs, gpu=gpu,
                threads=threads, mmseqs_bin=mmseqs_bin, log_path=log_path,
            )
            # Clusters ARE the connected components of this hit graph, so no hit crosses a
            # cluster boundary by construction; verify_ood_clusters.py certifies the guarantee.
            lookup = connected_components_from_hits(
                hits_tsv, nodes, alphabet=alphabet, cluster_id_prefix=short_name)
        else:
            result = run_mmseqs_easy_clust(
                fasta_path=fasta_path, out_prefix=out_prefix, tmp_dir=tmp_dir,
                min_seq_id=float(threshold), coverage=coverage, cov_mode=0, threads=threads,
                log_path=log_path, alphabet=alphabet, algorithm=method, cluster_mode=cluster_mode,
                sensitivity=sensitivity, single_step_clustering=single_step_clustering,
                max_seqs=max_seqs, mmseqs_bin=mmseqs_bin,
            )
            lookup = parse_cluster_tsv(result.cluster_tsv, alphabet=alphabet, cluster_id_prefix=short_name)
        lookup['function'] = full_name
        lookup['function_short'] = short_name
        lookup['threshold'] = float(threshold)
        lookup['alphabet'] = alphabet
        lookup.to_parquet(cluster_parquet, index=False)
        elapsed, cached = time.time() - t0, False
        print(f"  [{short_name} @ {threshold:.2f}] {len(lookup):,} seqs -> "
              f"{lookup['cluster_id'].nunique():,} clusters in {elapsed:.1f}s")

    # Hits persistence (search only). Union-find (above) and the separation figure are the hits'
    # only consumers, and the raw mmseqs TSV is transiently huge, so we never keep the TSV.
    # Default: re-store it as a compact parquet (query/target/fident). --delete_hits keeps neither.
    if method == 'search':
        if delete_hits:
            for path in (hits_tsv, hits_parquet):
                if path.exists():
                    path.unlink()
        elif hits_tsv.exists():
            read_hits_tsv(hits_tsv, usecols=['query', 'target', 'fident']).to_parquet(hits_parquet, index=False)
            hits_tsv.unlink()
        if scratch_dir:
            shutil.rmtree(scr, ignore_errors=True)  # drop the local scratch (mmseqs tmp + leftovers)

    dist = cluster_size_distribution(lookup[['cluster_id']])
    dist.update({
        'function': full_name, 'function_short': short_name,
        'threshold': float(threshold), 'alphabet': alphabet,
        'method': method, 'cluster_mode': cluster_mode if method != 'search' else None,
        'elapsed_seconds': elapsed, 'cached': cached,
        'cluster_parquet': str(cluster_parquet),
    })
    return dist


def add_input_args(p: argparse.ArgumentParser) -> None:
    """The mutually-exclusive sequence-source group (+ function_source, alphabet). Shared by all methods."""
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--protein_final', help='[builder] aa input:     protein_final.parquet (or .csv)')
    src.add_argument('--cds_dna_final', help='[builder] nt_cds input: cds_dna_final.parquet')
    src.add_argument('--ctg_dna_final', help='[builder] nt_ctg input: ctg_dna_final.parquet')
    p.add_argument('--function_source',
                   help='[builder] nt_ctg only: [assembly_id, genbank_ctg_id, function] source for the '
                        'contig->function join (default: sibling cds_dna_final.parquet)')
    p.add_argument('--alphabet', choices=['aa', 'nt_cds', 'nt_ctg'], default=None,
                   help='[builder] sequence alphabet (default: inferred from the input file)')


def load_and_filter(args) -> tuple:
    """load_sequence_frame + out_root mkdir + filter_present_functions; returns (df, alphabet, functions)."""
    df, alphabet = load_sequence_frame(
        protein_final=args.protein_final,
        cds_dna_final=args.cds_dna_final,
        ctg_dna_final=args.ctg_dna_final,
        alphabet=args.alphabet,
        function_source=args.function_source,
    )
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    functions, skipped = filter_present_functions(df, args.functions, SHORT_TO_FUNCTION)
    if skipped:
        print(f"  NOTE: skipping {len(skipped)} function(s) with no rows in this input: {skipped}")
    if not functions:
        raise SystemExit("No functions to process. Check the input file's `function` column.")
    return df, alphabet, functions


def _validate_method_args(args) -> None:
    """Reject arg/method combinations that don't apply (see the arg matrix in the plan/docstring)."""
    m = args.method
    if m != 'search':
        bad = [f for f, v in (('--exhaustive', args.exhaustive), ('--gpu', args.gpu),
                              ('--delete_hits', args.delete_hits), ('--scratch_dir', args.scratch_dir)) if v]
        if bad:
            raise SystemExit(f"{bad} are only valid with --method search.")
    if m == 'search':
        bad = []
        if args.cluster_mode != 0:
            bad.append('--cluster_mode')
        if args.single_step_clustering:
            bad.append('--single_step_clustering')
        if bad:
            raise SystemExit(f"{bad} apply to linclust/cluster only, not --method search "
                             "(search builds connected components via union-find).")
    if m == 'linclust':
        bad = [f for f, v in (('--sensitivity', args.sensitivity is not None),
                              ('--single_step_clustering', args.single_step_clustering)) if v]
        if bad:
            raise SystemExit(f"{bad} require --method cluster (not linclust).")


def main(default_method=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # --help shows each default

    p.add_argument('--method', choices=_METHODS, default=default_method, required=default_method is None,
                   help='[builder] clustering paradigm: linclust/cluster (set-cover) or search (OOD CCs)')
    add_input_args(p)

    # Shared build.
    p.add_argument('--out_root', required=True, help='[builder] output root, e.g. .../clusters_nt_cds')
    p.add_argument('--thresholds', nargs='+', type=float, required=True,
                   help='[mmseqs2: --min-seq-id] identity thresholds, e.g. 0.99 0.98 0.95')
    p.add_argument('--functions', nargs='+', default=_FLU_META.selected_short_names,
                   help='[builder] function short names to cluster')
    p.add_argument('--threads', type=int, default=16, help='[mmseqs2: --threads]')
    p.add_argument('--mmseqs_bin', help='[builder] mmseqs binary (default: $MMSEQS_BIN, then "mmseqs" on PATH)')
    p.add_argument('--force', action='store_true', help='[builder] recompute even if cached')
    p.add_argument('--no_combined', action='store_true',
                   help='[builder] skip combined_cluster.parquet per threshold')

    # Shared mmseqs knobs (method-specific defaults resolved below).
    p.add_argument('--coverage', type=float, default=0.8, help='[mmseqs2: -c] coverage (all methods)')
    p.add_argument('--sensitivity', type=float, default=None,
                   help='[mmseqs2: -s] prefilter sensitivity (7.5=most sensitive); cluster + search '
                        '(not linclust). search default 7.5 when unset')
    p.add_argument('--max_seqs', type=int, default=None,
                   help='[mmseqs2: --max-seqs] neighbours kept per query; search default 100000 when unset '
                        '(auto-raised to >= the unique-seq count)')

    # Set-cover only (linclust / cluster).
    p.add_argument('--cluster_mode', type=int, choices=[0, 1, 2], default=0,
                   help='[mmseqs2: --cluster-mode] linclust/cluster only. 0=Set-Cover (default). '
                        '1=connected-component -- NOTE: carries NO exhaustive across-cluster-separation '
                        'guarantee (the linclust prefilter can miss edges); unlike --method search, whose '
                        'union-find CCs DO carry it (search is far slower than linclust cluster-mode 1). '
                        '2=greedy-incremental')
    p.add_argument('--single_step_clustering', action='store_true',
                   help='[mmseqs2: --single-step-clustering] easy-cluster only (one non-cascaded pass)')

    # Search only.
    p.add_argument('--exhaustive', action='store_true',
                   help='[mmseqs2: --prefilter-mode 2] search only: nofilter all-vs-all (provably complete, slower)')
    p.add_argument('--gpu', type=int, default=0,
                   help='[mmseqs2: --gpu] search only: 1=on (Ampere+/Hopper; pick with CUDA_VISIBLE_DEVICES)')
    p.add_argument('--delete_hits', action='store_true',
                   help='[builder] search only: delete each hits TSV/parquet after clustering (saves disk; '
                        'separation figures then need a re-search)')
    p.add_argument('--scratch_dir',
                   help='[builder] search only: local dir for the transient hits TSV + mmseqs tmp (spares a '
                        'full NFS); the cluster parquet still goes to --out_root')

    args = p.parse_args()

    # Method-specific default resolution (search inherits the historical OOD defaults).
    if args.method == 'search':
        if args.sensitivity is None:
            args.sensitivity = 7.5
        if args.max_seqs is None:
            args.max_seqs = 100000

    _validate_method_args(args)

    prefilter_mode = 2 if args.exhaustive else None

    # Reject thresholds that collapse to the same tXXX directory label (percent rounding) --
    # they would otherwise silently share/overwrite one cluster dir.
    labels = [threshold_label(t) for t in args.thresholds]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"--thresholds collide on directory labels {sorted(set(labels))}; "
                         "use distinct thresholds (percent granularity).")

    df, alphabet, functions = load_and_filter(args)
    out_root = Path(args.out_root)

    all_stats = []
    for threshold in args.thresholds:
        print(f"\n=== {args.method} threshold = {threshold:.2f} ===")
        threshold_stats = []
        for short in functions:
            stats = cluster_one_function_one_threshold(
                df, short, threshold, out_root, method=args.method, alphabet=alphabet,
                coverage=args.coverage, sensitivity=args.sensitivity, max_seqs=args.max_seqs,
                threads=args.threads, mmseqs_bin=args.mmseqs_bin, force=args.force,
                cluster_mode=args.cluster_mode, single_step_clustering=args.single_step_clustering,
                gpu=args.gpu, prefilter_mode=prefilter_mode,
                delete_hits=args.delete_hits, scratch_dir=args.scratch_dir,
            )
            threshold_stats.append(stats)
        all_stats.extend(threshold_stats)
        if not args.no_combined:
            print(f"  combined parquet -> {aggregate_combined_lookup(out_root, threshold, functions)}")

        # Per-threshold provenance: the build config + a per-function timing rollup, written into
        # this t<NN> dir next to its cluster parquets.
        config = {
            'alphabet': alphabet, 'method': args.method, 'threshold': float(threshold),
            'functions': list(functions), 'coverage': args.coverage,
            'sensitivity': args.sensitivity, 'max_seqs': args.max_seqs, 'threads': args.threads,
        }
        if args.method == 'search':
            config.update({'prefilter_mode': prefilter_mode, 'exhaustive': args.exhaustive, 'gpu': args.gpu})
        else:
            config.update({'cluster_mode': args.cluster_mode,
                           'single_step_clustering': args.single_step_clustering})
        runtime_json = write_runtime_json(out_root / threshold_label(threshold), config, threshold_stats)
        print(f"  Runtime JSON: {runtime_json}")  # written fresh; preserved on a purely-cached re-run

    # Per-(function, threshold) cluster-size stats, merged with any prior CSV so a subset re-run
    # doesn't drop earlier rows.
    stats_csv = write_or_merge_stats_csv(out_root, all_stats, 'cluster_stats.csv')
    print(f"\nWrote stats CSV: {stats_csv}")


if __name__ == '__main__':
    main()
