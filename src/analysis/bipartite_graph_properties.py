"""Bipartite graph properties on the (cluster-level) HA-NA cooccurrence graph.

For each (schema_pair, alphabet, threshold), builds the bipartite multigraph
where:
  - Side A = slot-a clusters (e.g., HA clusters)
  - Side B = slot-b clusters (e.g., NA clusters)
  - Edges = one per row in the pair_key-deduped pair universe; multiple
    distinct sequence pairs that map to the same (cluster_a, cluster_b)
    tuple contribute parallel edges (multigraph).

Then computes per-CC structural properties used to inform splitter design:
  - Per-CC: node counts on each side, unique-edge count, pair count (with
    multiplicity), n_bridges, n_cut_nodes (always — both O(V+E)).
  - Optional, opt-in: λ(G) (edge connectivity) and the actual minimum edge
    cut for the largest CC (expensive — O(V·E·poly); set
    --compute_lambda_largest with a time budget).
  - Optional, opt-in: GraphML export of the largest CC subgraph for
    visualization in Gephi / Cytoscape.

The largest CC's bridges and cut nodes are always dumped to a per-slice
subdirectory.

Terminology (see docs/methods/glossary.md):
  - Bridge — edge whose removal increases CC count.
  - Cut node — node whose removal increases CC count. (Also known as
    "articulation point" or "cut vertex" in standard graph theory;
    we use "cut node" throughout this project for consistency.)
  - λ(G) — edge connectivity; minimum size of an edge cut.

CLI:
    python -m src.analysis.bipartite_graph_properties \\
        [--schema_pair HA NA] \\
        [--alphabets aa nt_cds] \\
        [--thresholds id100 id099 id098 id097 id096 id095 \\
                       id094 id093 id092 id091 id090] \\
        [--compute_lambda_largest] \\
        [--export_graphml] \\
        [--out_dir results/flu/July_2025/runs/bipartite_graph_properties]

Outputs (under --out_dir):
    graph_props.csv               long-form, columns:
        schema_pair, alphabet, threshold, cc_id, n_nodes_a, n_nodes_b,
        n_unique_edges, n_pairs, n_bridges, n_cut_nodes, lambda,
        is_largest
    largest_cc/{slug}_{alphabet}_{threshold}/
        bridges.csv               always — edge list of bridges
        cut_nodes.csv             always — cut node list (side + cluster_id)
        min_cut.csv               only if --compute_lambda_largest
        subgraph.graphml          only if --export_graphml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe


# Default threshold range. Matches cluster_pair_weight_topk._DEFAULT_THRESHOLDS.
_DEFAULT_THRESHOLDS = [f't{i:03d}' for i in range(100, 89, -1)]


def load_cluster_map(clusters_root: Path, slot_protein: str, threshold_id: str) -> dict[str, str]:
    """Load {hash -> cluster_id} for one (slot_protein, threshold). Same shape
    as cluster_pair_weight_topk.load_cluster_map. Empty dict if file missing.
    """
    cluster_pq = clusters_root / threshold_id / f'{slot_protein}_cluster.parquet'
    if not cluster_pq.exists():
        return {}
    df = pd.read_parquet(cluster_pq, columns=['seq_hash', 'cluster_id'])
    return dict(zip(df['seq_hash'].values, df['cluster_id'].values))


def build_bipartite_multigraph(
    pair_universe: pd.DataFrame,
    ha_cluster_map: dict,
    na_cluster_map: dict,
    alphabet: str,
) -> tuple[nx.MultiGraph, int]:
    """Build the cluster-level bipartite multigraph from the pair universe.

    Side-prefixed node IDs ('a:HA_5', 'b:NA_12') keep the two sides
    disjoint even when cluster IDs collide across slots. Each row in
    pair_universe contributes one edge; multiple rows mapping to the
    same (cluster_a, cluster_b) tuple become parallel edges.

    Args:
        pair_universe: from load_pair_universe; one row per unique
            canonical protein pair.
        ha_cluster_map: {hash -> cluster_id} for slot-a (HA).
        na_cluster_map: {hash -> cluster_id} for slot-b (NA).
        alphabet: 'aa' (uses seq_hash_{a,b}) or 'nt_cds' (uses dna_hash_{a,b}).

    Returns:
        (G, n_unmapped). G is the multigraph; n_unmapped is the number of
        pair-universe rows dropped because either endpoint lacked a
        cluster assignment (should be 0 if clusters cover the corpus).
    """
    if alphabet == 'aa':
        col_a, col_b = 'seq_hash_a', 'seq_hash_b'
    elif alphabet == 'nt_cds':
        col_a, col_b = 'dna_hash_a', 'dna_hash_b'
    else:
        raise ValueError(f"alphabet must be 'aa' or 'nt_cds', got {alphabet!r}")

    df = pair_universe.copy()
    df['_cluster_a'] = df[col_a].map(ha_cluster_map)
    df['_cluster_b'] = df[col_b].map(na_cluster_map)
    n_unmapped = int(df[['_cluster_a', '_cluster_b']].isna().any(axis=1).sum())
    df = df.dropna(subset=['_cluster_a', '_cluster_b'])

    edges = [
        (f'a:{ca}', f'b:{cb}')
        for ca, cb in zip(df['_cluster_a'].values, df['_cluster_b'].values)
    ]
    G = nx.MultiGraph()
    G.add_edges_from(edges)
    return G, n_unmapped


def per_cc_stats(
    G: nx.MultiGraph,
    compute_lambda_largest: bool = False,
    max_sec_lambda: int = 600,
) -> tuple[pd.DataFrame, Optional[dict]]:
    """One row per CC; bridges + cut nodes always; λ optional (largest only).

    Bridges and cut nodes are computed on the simple-graph projection of
    each CC (`nx.Graph(subgraph)`) because parallel edges in the
    multigraph would never qualify as bridges (their parallel partner
    keeps the endpoints connected). For routing purposes parallel edges
    are "free" — they all travel with their endpoints regardless — so
    bridge/cut analysis on the simple projection is the relevant view.

    Returns:
        (cc_df, largest_cc_artifacts) where largest_cc_artifacts is a
        dict with keys 'cc_id', 'subgraph_simple', 'bridges',
        'cut_nodes', and optionally 'min_cut', 'lambda' — used by the
        caller to write the per-largest-CC subdir.
    """
    ccs = list(nx.connected_components(G))
    if not ccs:
        return pd.DataFrame(), None
    # Identify largest CC by edge count in the multigraph (= n_pairs).
    cc_sizes = [(i, G.subgraph(cc).number_of_edges()) for i, cc in enumerate(ccs)]
    cc_sizes.sort(key=lambda x: -x[1])
    largest_cc_id = cc_sizes[0][0]

    rows = []
    largest_artifacts: Optional[dict] = None
    for cc_id, cc_nodes in enumerate(ccs):
        subg_multi = G.subgraph(cc_nodes)
        subg_simple = nx.Graph(subg_multi)

        n_nodes_a = sum(1 for n in cc_nodes if n.startswith('a:'))
        n_nodes_b = sum(1 for n in cc_nodes if n.startswith('b:'))
        n_pairs = subg_multi.number_of_edges()
        n_unique_edges = subg_simple.number_of_edges()

        bridges = list(nx.bridges(subg_simple))
        cut_nodes = list(nx.articulation_points(subg_simple))

        lam: Optional[int] = None
        min_cut: Optional[list] = None
        if cc_id == largest_cc_id and compute_lambda_largest and n_unique_edges > 0:
            t0 = time.time()
            try:
                lam = int(nx.edge_connectivity(subg_simple))
                if time.time() - t0 < max_sec_lambda:
                    min_cut = list(nx.minimum_edge_cut(subg_simple))
            except Exception as e:
                print(f"  WARNING: edge_connectivity / minimum_edge_cut "
                      f"failed on largest CC ({type(e).__name__}: {e})")

        rows.append({
            'cc_id': cc_id,
            'n_nodes_a': n_nodes_a,
            'n_nodes_b': n_nodes_b,
            'n_unique_edges': int(n_unique_edges),
            'n_pairs': int(n_pairs),
            'n_bridges': len(bridges),
            'n_cut_nodes': len(cut_nodes),
            'lambda': lam,
            'is_largest': cc_id == largest_cc_id,
        })

        if cc_id == largest_cc_id:
            largest_artifacts = {
                'cc_id': cc_id,
                'subgraph_simple': subg_simple,
                'bridges': bridges,
                'cut_nodes': cut_nodes,
                'min_cut': min_cut,
                'lambda': lam,
            }

    df = pd.DataFrame(rows).sort_values('n_pairs', ascending=False).reset_index(drop=True)
    return df, largest_artifacts


def write_largest_cc_artifacts(
    artifacts: dict,
    out_dir: Path,
    export_graphml: bool,
) -> None:
    """Write bridges, cut nodes, (optionally) min_cut and graphml subgraph."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _split(node_id: str) -> tuple[str, str]:
        side, cid = node_id.split(':', 1)
        return side, cid

    pd.DataFrame(
        [_split(u) + _split(v) for u, v in artifacts['bridges']],
        columns=['side_a', 'cluster_a', 'side_b', 'cluster_b'],
    ).to_csv(out_dir / 'bridges.csv', index=False)

    pd.DataFrame(
        [_split(n) for n in artifacts['cut_nodes']],
        columns=['side', 'cluster_id'],
    ).to_csv(out_dir / 'cut_nodes.csv', index=False)

    if artifacts['min_cut'] is not None:
        pd.DataFrame(
            [_split(u) + _split(v) for u, v in artifacts['min_cut']],
            columns=['side_a', 'cluster_a', 'side_b', 'cluster_b'],
        ).to_csv(out_dir / 'min_cut.csv', index=False)

    if export_graphml:
        nx.write_graphml(artifacts['subgraph_simple'], str(out_dir / 'subgraph.graphml'))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'))
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'))
    p.add_argument('--clusters_nt',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_nt'))
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'],
                   metavar=('SLOT_A', 'SLOT_B'),
                   help='Two protein shorts (default HA NA).')
    p.add_argument('--alphabets', nargs='+', default=['aa'],
                   choices=['aa', 'nt_cds'],
                   help='Alphabets to sweep (default aa only).')
    p.add_argument('--thresholds', nargs='+', default=_DEFAULT_THRESHOLDS,
                   help='Cluster thresholds (default t100..t90).')
    p.add_argument('--compute_lambda_largest', action='store_true',
                   help='Compute λ(G) and the actual minimum edge cut for the '
                        'largest CC at each slice. Expensive: O(V·E·poly), '
                        'may take many minutes to hours on the mega-CC. '
                        'Off by default.')
    p.add_argument('--max_sec_lambda', type=int, default=600,
                   help='Time budget for the λ/min_cut computation per slice '
                        '(default 600s = 10 min). Only used with '
                        '--compute_lambda_largest.')
    p.add_argument('--export_graphml', action='store_true',
                   help='Export the largest CC subgraph as GraphML for '
                        'visualization in Gephi / Cytoscape. Off by default.')
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/bipartite_graph_properties')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slot_a, slot_b = args.schema_pair
    schema_pair_label = f'{slot_a}-{slot_b}'
    schema_pair_slug = f'{slot_a.lower()}_{slot_b.lower()}'

    print(f"Loading pair universe for {schema_pair_label} ...")
    universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
    print(f"  {len(universe):,} unique canonical protein pairs\n")

    clusters_aa = Path(args.clusters_aa)
    clusters_nt = Path(args.clusters_nt)

    long_frames = []
    for alphabet in args.alphabets:
        clusters_root = clusters_aa if alphabet == 'aa' else clusters_nt
        for threshold in args.thresholds:
            ha_cmap = load_cluster_map(clusters_root, slot_a, threshold)
            na_cmap = load_cluster_map(clusters_root, slot_b, threshold)
            if not ha_cmap or not na_cmap:
                print(f"  [{alphabet} {threshold}] missing cluster parquet; skipping.")
                continue

            print(f"=== {alphabet} {threshold} ===")
            t0 = time.time()
            G, n_unmapped = build_bipartite_multigraph(universe, ha_cmap, na_cmap, alphabet)
            if n_unmapped > 0:
                print(f"  WARNING: {n_unmapped} pair-universe rows dropped (unmapped endpoint).")
            print(f"  graph: {G.number_of_nodes():,} nodes, "
                  f"{G.number_of_edges():,} edges (multigraph)")

            cc_df, largest_artifacts = per_cc_stats(
                G,
                compute_lambda_largest=args.compute_lambda_largest,
                max_sec_lambda=args.max_sec_lambda,
            )
            cc_df.insert(0, 'threshold', threshold)
            cc_df.insert(0, 'alphabet', alphabet)
            cc_df.insert(0, 'schema_pair', schema_pair_label)
            long_frames.append(cc_df)

            largest_row = cc_df.iloc[0]
            print(f"  largest CC: {largest_row['n_nodes_a']} + {largest_row['n_nodes_b']} nodes, "
                  f"{largest_row['n_unique_edges']:,} unique edges, "
                  f"{largest_row['n_pairs']:,} pairs (with parallel), "
                  f"{largest_row['n_bridges']} bridges, "
                  f"{largest_row['n_cut_nodes']} cut nodes"
                  + (f", λ={largest_row['lambda']}" if pd.notna(largest_row['lambda']) else "")
                  + f"  ({time.time() - t0:.1f}s)")

            if largest_artifacts is not None:
                largest_dir = (
                    out_dir / 'largest_cc'
                    / f'{schema_pair_slug}_{alphabet}_{threshold}'
                )
                write_largest_cc_artifacts(
                    largest_artifacts, largest_dir,
                    export_graphml=args.export_graphml,
                )

    if not long_frames:
        print("\nNo slices processed.")
        return

    long_df = pd.concat(long_frames, ignore_index=True)
    long_csv = out_dir / 'graph_props.csv'
    long_df.to_csv(long_csv, index=False)
    print(f"\nwrote {long_csv} ({len(long_df):,} rows)")
    print("\nDone.")


if __name__ == '__main__':
    main()
