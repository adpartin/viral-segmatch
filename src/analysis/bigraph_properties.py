"""Bigraph properties on the (cluster-level) HA-NA cooccurrence graph.

For each (cluster source, schema pair, threshold), loads the cluster-level bigraph
from the persisted CC artifact (`_cc_artifacts.load_cc_bigraph` -> the shared
`_bigraph.build_pair_bigraph`) where:
  - Side A = slot-a clusters (e.g., HA clusters)
  - Side B = slot-b clusters (e.g., NA clusters)
  - Edges = one per (cluster_a, cluster_b) cluster pair, with `weight` = the number
    of pair_key-deduped pair-universe rows on it. So a component's PAIR count is
    `size(weight='weight')` and its CLUSTER-PAIR count is `number_of_edges()`.

Then computes per-CC structural properties used to inform splitter design:
  - Per-CC: node counts on each side, unique-edge count, pair count (with
    multiplicity), n_bridges, n_cut_nodes (always — both O(V+E)), and
    per-side hub-concentration scalars: top1/top5 pair-mass share, max
    simple-degree, and pair-mass Gini (high Gini = a few bigraph hubs
    carry the component).
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

Input is the persisted CC artifact (`_cc_artifacts`), not a rebuilt pair universe, so
the graph is the one the splitter routed. Build missing slices with
`src/datasets/build_cc_structure.py`.

CLI:
    python -m src.analysis.bigraph_properties \\
        [--cc_source nt_cds_cm0] [--pair HA-NA] [--alphabet nt_cds] \\
        [--thresholds t099 t098 t097 t096 t095] [--fragmented] \\
        [--compute_lambda_largest] \\
        [--export_graphml] \\
        [--out_dir results/flu/July_2025/runs/bigraph_properties]

Outputs (under --out_dir):
    graph_props.csv               long-form, columns:
        schema_pair, alphabet, threshold, cc_id, n_nodes_a, n_nodes_b,
        n_unique_edges, n_pairs, n_bridges, n_cut_nodes, lambda,
        is_largest, top1_pairmass_frac_{a,b}, top5_pairmass_frac_{a,b},
        max_simple_degree_{a,b}, pairmass_gini_{a,b}
    largest_cc/{slug}_{alphabet}_{threshold}/
        node_degrees.csv          always — per-node simple_degree + pair_mass
                                  (weighted degree = incident pairs) +
                                  is_cut_node, sorted by pair_mass
        bridges.csv               always — edge list of bridges
        cut_nodes.csv             always — cut nodes with simple_degree +
                                  pair_mass (the data dropped if removed)
        min_cut.csv               only if --compute_lambda_largest
        subgraph.graphml          only if --export_graphml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis._cc_artifacts import add_cc_source_args, cc_dir, load_cc_bigraph
from src.datasets._bigraph import ranked_ccs

# Default threshold sweep: the range the CC artifacts are built for (t099..t095).
# Wider sweeps need build_cc_structure.py run for the extra thresholds first.
_DEFAULT_THRESHOLDS = [f't{i:03d}' for i in range(99, 94, -1)]


def _gini(values) -> float:
    """Gini coefficient of a non-negative sequence (0 = even, 1 = all mass on one).

    Summarizes how concentrated a side's pair-mass is across its clusters:
    a high Gini means a few bigraph hubs carry the component.
    """
    x = np.sort(np.asarray(list(values), dtype=float))
    n = x.size
    s = x.sum()
    if n == 0 or s == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * s) - (n + 1.0) / n)


def _side_concentration(masses: list, simple_degrees: list, n_pairs: int, side: str) -> dict:
    """Per-side hub-concentration scalars for one CC.

    Each edge has exactly one endpoint per side, so a side's per-node
    pair_mass sums to n_pairs; top-k pair_mass / n_pairs is the share of the
    CC's pairs carried by that side's k heaviest clusters. `pair_mass` is the
    weighted degree (incident pairs = data dropped if the node is removed);
    `simple_degree` is the count of distinct opposite-side partners.
    """
    m = np.sort(np.asarray(masses, dtype=float))[::-1]
    denom = float(n_pairs) if n_pairs > 0 else 1.0
    return {
        f'top1_pairmass_frac_{side}': round(float(m[:1].sum()) / denom, 4) if m.size else 0.0,
        f'top5_pairmass_frac_{side}': round(float(m[:5].sum()) / denom, 4) if m.size else 0.0,
        f'max_simple_degree_{side}': int(max(simple_degrees)) if simple_degrees else 0,
        f'pairmass_gini_{side}': round(_gini(masses), 4) if masses else 0.0,
    }


def per_cc_stats(
    H: nx.Graph,
    compute_lambda_largest: bool = False,
    max_sec_lambda: int = 600,
) -> tuple[pd.DataFrame, Optional[dict]]:
    """One row per CC; bridges + cut nodes always; λ optional (largest only).

    Takes the weighted simple bigraph (`_bigraph.build_pair_bigraph`), so bridges and
    cut nodes run on it directly — no `nx.Graph(multigraph)` projection per component.
    That projection was never about losing information: parallel edges can never be
    bridges (their partner keeps the endpoints connected), and for routing they are
    "free" — they travel with their endpoints regardless. The weighted simple graph is
    that view natively, with the multiplicity kept on `weight` where it is still needed
    (pair mass, per-CC pair count).

    `cc_id` is the canonical `ranked_ccs` order — largest CC first by pair count, ties on
    the lowest node id — so `cc_id == 0` is the mega-CC and ids do not depend on row or
    insertion order (they previously did).

    Returns:
        (cc_df, largest_cc_artifacts) where largest_cc_artifacts is a
        dict with keys 'cc_id', 'subgraph_simple', 'bridges',
        'cut_nodes', and optionally 'min_cut', 'lambda' — used by the
        caller to write the per-largest-CC subdir.
    """
    ccs = ranked_ccs(H)
    if not ccs:
        return pd.DataFrame(), None
    largest_cc_id = 0   # ranked_ccs is largest-pair-count first

    rows = []
    largest_artifacts: Optional[dict] = None
    for cc_id, cc_nodes in enumerate(ccs):
        subg_simple = H.subgraph(cc_nodes)

        # Per-node pair_mass (summed incident edge weight = incident pairs) and
        # simple_degree (distinct opposite-side partners), split by side.
        masses_a, masses_b = [], []
        sdeg_a, sdeg_b = [], []
        for n in cc_nodes:
            pm = subg_simple.degree(n, weight='weight')
            sd = subg_simple.degree(n)
            if n.startswith('a:'):
                masses_a.append(pm)
                sdeg_a.append(sd)
            else:
                masses_b.append(pm)
                sdeg_b.append(sd)

        n_nodes_a = len(masses_a)
        n_nodes_b = len(masses_b)
        n_pairs = int(subg_simple.size(weight='weight'))   # pairs, not cluster pairs
        n_unique_edges = subg_simple.number_of_edges()     # cluster pairs

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

        row = {
            'cc_id': cc_id,
            'n_nodes_a': n_nodes_a,
            'n_nodes_b': n_nodes_b,
            'n_unique_edges': int(n_unique_edges),
            'n_pairs': int(n_pairs),
            'n_bridges': len(bridges),
            'n_cut_nodes': len(cut_nodes),
            'lambda': lam,
            'is_largest': cc_id == largest_cc_id,
        }
        row.update(_side_concentration(masses_a, sdeg_a, n_pairs, 'a'))
        row.update(_side_concentration(masses_b, sdeg_b, n_pairs, 'b'))
        rows.append(row)

        if cc_id == largest_cc_id:
            cut_set = set(cut_nodes)
            node_rows = []
            for n in cc_nodes:
                side, cid = n.split(':', 1)
                node_rows.append({
                    'side': side,
                    'cluster_id': cid,
                    'simple_degree': int(subg_simple.degree(n)),
                    'pair_mass': int(subg_simple.degree(n, weight='weight')),
                    'is_cut_node': n in cut_set,
                })
            node_df = (pd.DataFrame(node_rows)
                       .sort_values('pair_mass', ascending=False)
                       .reset_index(drop=True))
            largest_artifacts = {
                'cc_id': cc_id,
                'subgraph_simple': subg_simple,
                'bridges': bridges,
                'cut_nodes': cut_nodes,
                'node_degrees': node_df,
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
    """Write node degrees, bridges, cut nodes, (optionally) min_cut and graphml."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _split(node_id: str) -> tuple[str, str]:
        side, cid = node_id.split(':', 1)
        return side, cid

    # Per-node degree table (sorted by pair_mass) and the cut-node subset
    # carrying its degree + pair_mass — the pair_mass column is the data
    # that would be dropped if that hub were removed.
    node_df = artifacts['node_degrees']
    node_df.to_csv(out_dir / 'node_degrees.csv', index=False)
    (node_df[node_df['is_cut_node']]
        .drop(columns='is_cut_node')
        .reset_index(drop=True)
        .to_csv(out_dir / 'cut_nodes.csv', index=False))

    # `nx.bridges` yields each edge in DFS-traversal order, so the endpoint that lands
    # first -- and the row order -- depend on node insertion order. Canonicalize both:
    # slot-A endpoint into the `_a` columns, then sort. Same bridge SET either way, but
    # the file is now reproducible rather than an artifact of traversal.
    bridge_rows = sorted(
        (_split(u) + _split(v)) if u.startswith('a:') else (_split(v) + _split(u))
        for u, v in artifacts['bridges']
    )
    pd.DataFrame(
        bridge_rows, columns=['side_a', 'cluster_a', 'side_b', 'cluster_b'],
    ).to_csv(out_dir / 'bridges.csv', index=False)

    if artifacts['min_cut'] is not None:
        pd.DataFrame(
            [_split(u) + _split(v) for u, v in artifacts['min_cut']],
            columns=['side_a', 'cluster_a', 'side_b', 'cluster_b'],
        ).to_csv(out_dir / 'min_cut.csv', index=False)

    if export_graphml:
        nx.write_graphml(artifacts['subgraph_simple'], str(out_dir / 'subgraph.graphml'))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cc_source_args(p)
    p.add_argument('--alphabet', default='nt_cds',
                   help='alphabet label for the output column/titles (default nt_cds; must match '
                        '--cc_source, which is what actually selects the data).')
    p.add_argument('--thresholds', nargs='+', default=_DEFAULT_THRESHOLDS,
                   help=f'Cluster thresholds (default {" ".join(_DEFAULT_THRESHOLDS)}); a threshold '
                        f'with no persisted artifact is skipped with a note.')
    p.add_argument('--fragmented', action='store_true',
                   help='read the post-edge-cut slice (tXXX/fragmented/) instead of the natural CCs.')
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
                   default=PROJ / 'results/flu/July_2025/runs/bigraph_properties')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_pair_label = args.pair
    schema_pair_slug = args.pair.lower().replace('-', '_')
    alphabet = args.alphabet
    # Slot labels for the console summary only; the data comes from the artifact's
    # cluster_id_a / cluster_id_b columns, so a malformed --pair cannot mis-assign sides.
    slot_a, slot_b = (args.pair.split('-', 1) + ['b'])[:2] if '-' in args.pair else ('a', 'b')

    long_frames = []
    for threshold in args.thresholds:
        d = (args.cc_dir if args.cc_dir else
             cc_dir(args.cc_source, args.pair, threshold, fragmented=args.fragmented))
        if not (d / 'pairs_with_cc.parquet').exists():
            print(f"  [{alphabet} {threshold}] no CC artifact at {d}; skipping.")
            continue

        print(f"=== {alphabet} {threshold} ===")
        t0 = time.time()
        H, cc_pairs = load_cc_bigraph(d)
        print(f"  artifact: {d}")
        print(f"  graph: {H.number_of_nodes():,} nodes, "
              f"{H.number_of_edges():,} cluster pairs, "
              f"{int(H.size(weight='weight')):,} pairs")

        cc_df, largest_artifacts = per_cc_stats(
            H,
            compute_lambda_largest=args.compute_lambda_largest,
            max_sec_lambda=args.max_sec_lambda,
        )
        cc_df.insert(0, 'threshold', threshold)
        cc_df.insert(0, 'alphabet', alphabet)
        cc_df.insert(0, 'schema_pair', schema_pair_label)
        long_frames.append(cc_df)

        largest_row = cc_df.iloc[0]
        bridge_frac = (largest_row['n_bridges'] / largest_row['n_unique_edges']
                       if largest_row['n_unique_edges'] else 0.0)
        print(f"  largest CC: {largest_row['n_nodes_a']} + {largest_row['n_nodes_b']} nodes, "
              f"{largest_row['n_unique_edges']:,} unique edges, "
              f"{largest_row['n_pairs']:,} pairs, "
              f"{largest_row['n_bridges']} bridges ({bridge_frac:.0%}), "
              f"{largest_row['n_cut_nodes']} cut nodes"
              + (f", λ={largest_row['lambda']}" if pd.notna(largest_row['lambda']) else "")
              + f"  ({time.time() - t0:.1f}s)")
        print(f"    hub concentration  "
              f"{slot_a}: top1={largest_row['top1_pairmass_frac_a']:.1%} "
              f"top5={largest_row['top5_pairmass_frac_a']:.1%} "
              f"gini={largest_row['pairmass_gini_a']:.2f} "
              f"maxdeg={largest_row['max_simple_degree_a']}  |  "
              f"{slot_b}: top1={largest_row['top1_pairmass_frac_b']:.1%} "
              f"top5={largest_row['top5_pairmass_frac_b']:.1%} "
              f"gini={largest_row['pairmass_gini_b']:.2f} "
              f"maxdeg={largest_row['max_simple_degree_b']}")

        if largest_artifacts is not None:
            nd = largest_artifacts['node_degrees']
            for side, lbl in [('a', slot_a), ('b', slot_b)]:
                top = nd[nd['side'] == side].head(3)
                hub_str = ", ".join(
                    f"{r.cluster_id}(deg {r.simple_degree}, {r.pair_mass:,}p)"
                    for r in top.itertuples()
                )
                print(f"    top {lbl} hubs: {hub_str}")
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
