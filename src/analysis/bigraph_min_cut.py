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
`src/datasets/_megacc_cut.fragment_to_targets` (one implementation shared with the
production splitter). What lives here is the CLI and the per-cut report.

Input is the NATURAL (pre-cut) CC artifact (`_cc_artifacts`) -- this script performs the
cut, so handing it `tXXX/fragmented/` would re-cut an already-cut graph.

CLI:
    python -m src.analysis.bigraph_min_cut \\
        [--cc_source nt_cds_cm0] [--pair HA-NA] [--alphabet nt_cds] \\
        [--threshold t095] [--method kl] [--target_frac 0.80] [--drift_pp 0.05] \\
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

from src.analysis._cc_artifacts import add_cc_source_args, cc_dir, load_cc_bigraph
from src.datasets._megacc_cut import fragment_to_targets


def min_cut_recursive(
    G: nx.Graph,
    method: str = 'spectral',
    target_frac: float = 0.80,
    drift_pp: float = 0.05,
    seed: int = 1,
    kl_max_iter: int = 10,
    max_cuts: int = 200,
    return_partition: bool = False,
    targets: dict | None = None,
    ):
    """Recursively bisect the largest CC until the kept atoms are LPT-feasible.

    Thin wrapper over `_megacc_cut.fragment_to_targets`: fragments the pair-weighted
    simple bigraph `G` (from `_cc_artifacts.load_cc_bigraph`) to `targets`
    (default `None` -> the cut module's 80/10/10; pass `uniform_targets(k)` for K-fold
    CV). Each row is the state BEFORE a cut (or the final feasible state); drops only
    crossing edges (straddling pairs); `dropped_frac` is vs the full pair universe.

    Does NOT mutate `G` — `fragment_to_targets` removes the cut edges in place, so it is
    handed a copy. (This used to fall out of the `weighted_simple` multigraph
    projection, which always returned a fresh graph.)

    Returns the per-cut DataFrame. If `return_partition`, returns
    `(df, H_kept, dropped_edges)` — the kept weighted simple graph whose
    connected components are the final atoms, and the list of cut (u, v) edges.
    """
    H = G.copy()
    # Omit `targets` when None so `fragment_to_targets`'s own 80/10/10 default applies --
    # the constant lives there, not duplicated here.
    targets_kw = {} if targets is None else {'targets': targets}
    df, H, dropped_edges = fragment_to_targets(
        H, cut_method=method, target_frac=target_frac, drift_pp=drift_pp, seed=seed,
        kl_max_iter=kl_max_iter, max_cuts=max_cuts, **targets_kw)
    if return_partition:
        return df, H, dropped_edges
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cc_source_args(p)
    p.add_argument('--alphabet', default='nt_cds',
                   help='alphabet label for output names (default nt_cds; must match --cc_source, '
                        'which is what actually selects the data).')
    p.add_argument('--threshold', default='t095')
    p.add_argument('--method', default='spectral', choices=['kl', 'spectral'],
                   help='Bisection heuristic. Default matches production, which is spectral '
                        'everywhere; kl is order-sensitive and gives different cuts.')
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
    slug = args.pair.lower().replace('-', '_')
    # The NATURAL slice: this script performs the cut, so it must start from the pre-cut
    # graph. Pointing it at tXXX/fragmented/ would re-cut an already-cut graph.
    d = args.cc_dir or cc_dir(args.cc_source, args.pair, args.threshold)

    print(f"Loading CC artifact (natural, pre-cut): {d}")
    G, pairs = load_cc_bigraph(d)
    print(f"  {len(pairs):,} positive pairs, {pairs['cc_id'].nunique():,} natural CCs")
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} cluster pairs, "
          f"{int(G.size(weight='weight')):,} pairs")

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
