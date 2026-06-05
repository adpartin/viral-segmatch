"""Balanced edge min-cut on the cluster-level bigraph: the efficient drop-budget.

Companion to `bipartite_hub_peel.py`. The hub-peel removes whole hub *nodes*
(whole clusters) and is dominated by their pair mass — a loose upper bound on
the cost of recovering feasibility. This script runs the *edge-cut* operation
instead: recursively bisect the largest connected component with a balanced
min-cut (Kernighan-Lin on the pair-weighted simple graph, networkx, in-env),
dropping only the *straddling pairs* (crossing edges) — the DataSAIL S2 /
drop-budget move (splits.md § 4.1). This is the efficient fragmentation path.

Two stopping targets are reported:
  - largest_le_80: the first cut where the largest kept CC <= `target_frac` of
    the retained pairs (removes the bilateral-infeasibility, splits.md § 1.7);
  - lpt_feasible: the cut where the kept atoms LPT-bin-pack into 80/10/10
    within `drift_pp` (the real feasibility gate, splits.md § 1.3 / § 3.3) —
    this matches the bicc audit's "recover 80/10/10" target
    (docs/results/2026-05-21_bicc_pair_drop_audit.md).

Greedy recursive bisection => an UPPER BOUND on the true balanced min-drop, but
far tighter than node-peel. Determinism: KL is seeded; same (graph, seed) gives
the same cut.

CLI:
    python -m src.analysis.bipartite_min_cut \\
        [--schema_pair HA NA] [--alphabet aa] [--threshold t095] \\
        [--method kl] [--target_frac 0.80] [--drift_pp 0.05] \\
        [--out_dir results/flu/July_2025/runs/bipartite_min_cut]

Outputs (under --out_dir):
    min_cut_{slug}_{alphabet}_{threshold}_{method}.csv   per-cut log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import networkx as nx

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe
from src.analysis.bipartite_graph_properties import (
    load_cluster_map,
    build_bipartite_multigraph,
)

_TARGETS = {'train': 0.80, 'val': 0.10, 'test': 0.10}
_BIN_ORDER = ['train', 'val', 'test']


def weighted_simple(G: nx.MultiGraph) -> nx.Graph:
    """Collapse the multigraph to a simple graph with edge weight = pair count.

    Sum of all edge weights = number of multigraph edges = pair-universe size.
    """
    H = nx.Graph()
    for x, y in G.edges():
        if H.has_edge(x, y):
            H[x][y]['weight'] += 1
        else:
            H.add_edge(x, y, weight=1)
    return H


def piece_pairs(H: nx.Graph, nodes) -> int:
    """Kept pairs (summed edge weight) inside one component."""
    return int(H.subgraph(nodes).size(weight='weight'))


def lpt_max_drift(sizes: list, targets: dict = _TARGETS, bin_order: list = _BIN_ORDER) -> float:
    """Max |achieved - target| over bins for LPT-greedy on these atom sizes.

    Mirrors `_pair_helpers._lpt_bin_pack`: largest atom first, to the bin with
    the biggest deficit (raw counts). Returns the max absolute fraction drift.
    """
    total = sum(sizes)
    if total <= 0:
        return 1.0
    caps = {b: targets[b] * total for b in bin_order}
    filled = {b: 0.0 for b in bin_order}
    for s in sorted(sizes, reverse=True):
        winner = max(bin_order, key=lambda b: caps[b] - filled[b])
        filled[winner] += s
    return max(abs(filled[b] / total - targets[b]) for b in bin_order)


def _bisect(H: nx.Graph, method: str, seed: int, max_iter: int) -> tuple[set, set, int]:
    """Balanced bisection of connected graph H; returns (A, B, crossing_weight)."""
    if method == 'kl':
        A, B = nx.algorithms.community.kernighan_lin_bisection(
            H, weight='weight', max_iter=max_iter, seed=seed)
        A = set(A)
    elif method == 'spectral':
        fv = nx.fiedler_vector(H, weight='weight', seed=seed)
        nodes = list(H.nodes())
        A = {n for n, v in zip(nodes, fv) if v < 0}
        if not A or len(A) == len(nodes):  # degenerate: fall back to median split
            order = sorted(range(len(nodes)), key=lambda i: fv[i])
            A = {nodes[i] for i in order[:len(nodes) // 2]}
    else:
        raise ValueError(f"method must be 'kl' or 'spectral', got {method!r}")
    cross = sum(d['weight'] for x, y, d in H.edges(data=True)
                if (x in A) != (y in A))
    return A, set(H.nodes()) - A, int(cross)


def min_cut_recursive(
    G: nx.MultiGraph,
    method: str = 'kl',
    target_frac: float = 0.80,
    drift_pp: float = 0.05,
    seed: int = 1,
    kl_max_iter: int = 10,
    max_cuts: int = 200,
    return_partition: bool = False,
):
    """Recursively bisect the largest CC until the kept atoms are LPT-feasible.

    Each row is the state BEFORE a cut (or the final feasible state). Drops only
    crossing edges (straddling pairs). `dropped_frac` is vs the full pair universe.

    Returns the per-cut DataFrame. If `return_partition`, returns
    `(df, H_kept, dropped_edges)` — the kept weighted simple graph whose
    connected components are the final atoms, and the list of cut (u, v) edges.
    """
    H = weighted_simple(G)
    total_pairs = int(H.size(weight='weight'))
    dropped = 0
    dropped_edges: list[tuple] = []
    rows: list[dict] = []
    cut = 0
    reached_le80 = False

    while True:
        comps = list(nx.connected_components(H))
        sizes = [piece_pairs(H, c) for c in comps]
        retained = total_pairs - dropped
        largest = max(sizes)
        largest_frac = largest / retained if retained else 0.0
        drift = lpt_max_drift(sizes)
        feasible = drift <= drift_pp

        rows.append({
            'cut': cut,
            'pairs_dropped': dropped,
            'dropped_frac': round(dropped / total_pairs, 6),
            'retained_pairs': retained,
            'n_pieces': len(comps),
            'largest_cc_pairs': largest,
            'largest_frac_of_retained': round(largest_frac, 6),
            'lpt_max_drift': round(drift, 6),
            'lpt_feasible': feasible,
            'largest_le_target': largest_frac <= target_frac,
        })

        if feasible or cut >= max_cuts:
            break

        # Bisect the largest piece, drop its crossing edges.
        big = max(comps, key=lambda c: piece_pairs(H, c))
        sub = H.subgraph(big)
        A, _, cross = _bisect(sub, method, seed, kl_max_iter)
        cross_edges = [(x, y) for x, y in sub.edges() if (x in A) != (y in A)]
        H.remove_edges_from(cross_edges)
        dropped_edges.extend(cross_edges)
        dropped += cross
        cut += 1

    df = pd.DataFrame(rows)
    if return_partition:
        return df, H, dropped_edges
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'))
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'))
    p.add_argument('--clusters_nt',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_nt'))
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'],
                   metavar=('SLOT_A', 'SLOT_B'))
    p.add_argument('--alphabet', default='aa', choices=['aa', 'nt_cds'])
    p.add_argument('--threshold', default='t095')
    p.add_argument('--method', default='kl', choices=['kl', 'spectral'])
    p.add_argument('--target_frac', type=float, default=0.80)
    p.add_argument('--drift_pp', type=float, default=0.05,
                   help='LPT feasibility gate: max |achieved-target| over bins (splits.md § 3.3).')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--kl_max_iter', type=int, default=10)
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/bipartite_min_cut')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slot_a, slot_b = args.schema_pair
    slug = f'{slot_a.lower()}_{slot_b.lower()}'
    clusters_root = Path(args.clusters_aa if args.alphabet == 'aa' else args.clusters_nt)

    print(f"Loading pair universe for {slot_a}-{slot_b} ...")
    universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
    print(f"  {len(universe):,} unique canonical protein pairs")
    cmap_a = load_cluster_map(clusters_root, slot_a, args.threshold)
    cmap_b = load_cluster_map(clusters_root, slot_b, args.threshold)
    if not cmap_a or not cmap_b:
        raise SystemExit(f"missing cluster parquet for {args.alphabet} {args.threshold}")
    G, n_unmapped = build_bipartite_multigraph(universe, cmap_a, cmap_b, args.alphabet)
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} pair-universe rows dropped (unmapped endpoint).")
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges (multigraph)")

    print(f"\nRecursive {args.method.upper()} min-cut until LPT-feasible "
          f"(drift <= {args.drift_pp:.0%}) ...")
    t0 = time.time()
    df = min_cut_recursive(G, method=args.method, target_frac=args.target_frac,
                           drift_pp=args.drift_pp, seed=args.seed,
                           kl_max_iter=args.kl_max_iter)
    elapsed = time.time() - t0

    stem = f'min_cut_{slug}_{args.alphabet}_{args.threshold}_{args.method}'
    csv_path = out_dir / f'{stem}.csv'
    df.to_csv(csv_path, index=False)

    total = G.number_of_edges()
    le80 = df[df['largest_le_target']]
    feas = df[df['lpt_feasible']]
    print(f"\n  ({elapsed:.1f}s, {len(df) - 1} cut(s))")
    if len(le80):
        r = le80.iloc[0]
        print(f"  largest CC <= {args.target_frac:.0%}:  after {int(r['cut'])} cut(s), "
              f"dropped {r['pairs_dropped']:,} ({r['dropped_frac']:.1%}); "
              f"largest now {r['largest_frac_of_retained']:.1%} of retained")
    if len(feas):
        r = feas.iloc[0]
        print(f"  LPT 80/10/10 feasible:    after {int(r['cut'])} cut(s), "
              f"dropped {r['pairs_dropped']:,} ({r['dropped_frac']:.1%}); "
              f"{int(r['n_pieces'])} atoms, max drift {r['lpt_max_drift']:.1%}")
    else:
        print(f"  LPT 80/10/10 NOT reached within max_cuts.")

    print(f"\n  per-cut log:")
    for r in df.itertuples():
        tag = 'FEASIBLE' if r.lpt_feasible else ('<=target' if r.largest_le_target else '')
        print(f"    cut {r.cut:>2}: dropped {r.pairs_dropped:>7,} ({r.dropped_frac:>5.1%})  "
              f"{int(r.n_pieces):>5} atoms  largest {r.largest_frac_of_retained:>5.1%}  "
              f"drift {r.lpt_max_drift:>5.1%}  {tag}")

    print(f"\nwrote {csv_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
