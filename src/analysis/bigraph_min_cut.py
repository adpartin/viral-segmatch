"""Balanced edge min-cut on the cluster-level bigraph: the efficient drop-budget.

Companion to `bigraph_hub_peel.py`. The hub-peel removes whole hub *nodes*
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

This module is the CLI/report layer only: the cut loop itself is
`src/datasets/_megacc_cut.fragment_weighted` (one implementation shared with the
production splitter). What lives here is the multigraph -> weighted-simple
projection and the per-cut report.

CLI:
    python -m src.analysis.bigraph_min_cut \\
        [--schema_pair HA NA] [--alphabet aa] [--threshold t095] \\
        [--method kl] [--target_frac 0.80] [--drift_pp 0.05] \\
        [--out_dir results/flu/July_2025/runs/bigraph_min_cut]

Outputs (under --out_dir):
    min_cut_{slug}_{alphabet}_{threshold}_{method}.csv   per-cut log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import networkx as nx

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.bigraph_properties import build_bipartite_multigraph
from src.analysis.cluster_pair_weight_topk import load_pair_universe
from src.datasets._megacc_cut import fragment_weighted
from src.utils.cluster_source import CLUSTERS_ROOT, cluster_map_for_root


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


def min_cut_recursive(
    G: nx.MultiGraph,
    method: str = 'kl',
    target_frac: float = 0.80,
    drift_pp: float = 0.05,
    seed: int = 1,
    kl_max_iter: int = 10,
    max_cuts: int = 200,
    return_partition: bool = False,
    targets: dict | None = None,
    ):
    """Recursively bisect the largest CC until the kept atoms are LPT-feasible.

    Thin wrapper over `_megacc_cut.fragment_weighted`: collapses the multigraph `G`
    to its weighted simple projection, then fragments to `targets` (default `None`
    -> the cut module's 80/10/10; pass `uniform_targets(k)` for K-fold CV). Each row
    is the state BEFORE a cut (or the final feasible state); drops only crossing
    edges (straddling pairs); `dropped_frac` is vs the full pair universe.

    Returns the per-cut DataFrame. If `return_partition`, returns
    `(df, H_kept, dropped_edges)` — the kept weighted simple graph whose
    connected components are the final atoms, and the list of cut (u, v) edges.
    """
    H = weighted_simple(G)
    # Omit `targets` when None so `fragment_weighted`'s own 80/10/10 default applies --
    # the constant lives there, not duplicated here.
    targets_kw = {} if targets is None else {'targets': targets}
    df, H, dropped_edges = fragment_weighted(
        H, cut_method=method, target_frac=target_frac, drift_pp=drift_pp, seed=seed,
        kl_max_iter=kl_max_iter, max_cuts=max_cuts, **targets_kw)
    if return_partition:
        return df, H, dropped_edges
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet'))
    p.add_argument('--clusters_aa', default=str(CLUSTERS_ROOT['aa']))
    p.add_argument('--clusters_nt', default=str(CLUSTERS_ROOT['nt_cds']))
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
                   default=PROJ / 'results/flu/July_2025/runs/bigraph_min_cut')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slot_a, slot_b = args.schema_pair
    slug = f'{slot_a.lower()}_{slot_b.lower()}'
    clusters_root = Path(args.clusters_aa if args.alphabet == 'aa' else args.clusters_nt)

    print(f"Loading pair universe for {slot_a}-{slot_b} ...")
    universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
    print(f"  {len(universe):,} unique canonical protein pairs")
    cmap_a = cluster_map_for_root(clusters_root, slot_a, args.threshold)
    cmap_b = cluster_map_for_root(clusters_root, slot_b, args.threshold)
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
        print("  LPT 80/10/10 NOT reached within max_cuts.")

    print("\n  per-cut log:")
    for r in df.itertuples():
        tag = 'FEASIBLE' if r.lpt_feasible else ('<=target' if r.largest_le_target else '')
        print(f"    cut {r.cut:>2}: dropped {r.pairs_dropped:>7,} ({r.dropped_frac:>5.1%})  "
              f"{int(r.n_pieces):>5} atoms  largest {r.largest_frac_of_retained:>5.1%}  "
              f"drift {r.lpt_max_drift:>5.1%}  {tag}")

    print(f"\nwrote {csv_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
