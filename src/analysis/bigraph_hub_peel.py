"""Greedy hub-peel on the cluster-level bigraph: pairs-dropped vs largest-CC fraction.

Tests the foundation claim that the bipartite mega-CC is held together by a few
high-pair-mass bipartite hubs that are also cut nodes (articulation points), so
fragmenting it to recover an 80/10/10-feasible largest component costs data
proportional to those hubs' pair mass — not the cheap peripheral bridges. The
companion property script (`bigraph_properties.py`) measures the static
structure (degrees, bridges, cut nodes); this script runs the dynamic surgery.

Algorithm (greedy, deterministic). Build the bipartite multigraph for one
(schema_pair, alphabet, threshold) from the pair universe (same construction as
`bigraph_properties.build_bipartite_multigraph`). Then repeat:
  1. find the largest CC (by pair count = multigraph edges);
  2. if its share of the *retained* pairs is <= target, stop;
  3. else remove the largest CC's heaviest node by pair_mass, restricted to cut
     nodes (strategy=cut_node, default — only a cut node can split a component)
     or any node (strategy=any_node, for contrast);
  4. record the step.

Removing a node deletes its incident pairs — they are "dropped" (the DataSAIL
S2 / drop-budget move; see splits.md § 4.1). The resulting curve is the
drop-budget cost of recovering feasibility. Greedy gives an UPPER BOUND on the
optimal min-drop: it confirms the mechanism and yields a representative cost,
not the provable minimum (true min-drop would need METIS/KaHIP-style balanced
min-cut — bicc audit direction #3, docs/results/2026-05-21_bicc_pair_drop_audit.md).

Denominator note. `dropped_frac` and the largest-CC fractions are measured
against the canonical pair universe (HA-NA aa = 58,826 pairs). The 2026-05-21
audit's "~18.5%" is relative to a larger dataset-build positive count, so
compare shape/magnitude, not exact percentages.

CLI:
    python -m src.analysis.bigraph_hub_peel \\
        [--schema_pair HA NA] [--alphabet aa] [--threshold t095] \\
        [--target_frac 0.80] [--strategy cut_node] \\
        [--out_dir results/flu/July_2025/runs/bigraph_hub_peel]

Outputs (under --out_dir):
    hub_peel_{slug}_{alphabet}_{threshold}_{strategy}.csv   per-step curve
    hub_peel_{slug}_{alphabet}_{threshold}_{strategy}.png   drop% vs largest-CC%
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe
from src.analysis.bigraph_properties import (
    load_cluster_map,
    build_bipartite_multigraph,
)


def _largest_cc_by_pairs(H: nx.MultiGraph) -> set:
    """Node set of the CC with the most multigraph edges (pairs)."""
    return max(nx.connected_components(H),
               key=lambda cc: H.subgraph(cc).number_of_edges())


def hub_peel(
    G: nx.MultiGraph,
    target_frac: float = 0.80,
    strategy: str = 'cut_node',
    max_steps: int = 100_000,
) -> pd.DataFrame:
    """Greedily peel hubs from the largest CC until it fits `target_frac`.

    Each row is the state BEFORE a removal plus the node removed at that step
    (the final row, where the target is met, has null `removed_*`). pair_mass
    is the multigraph degree (= incident pairs dropped when the node is
    removed); fractions are vs the full pair universe (`dropped_frac`,
    `largest_frac_of_original`) or vs the retained set (`largest_frac_of_retained`,
    the 80/10/10 feasibility gate).
    """
    if strategy not in ('cut_node', 'any_node'):
        raise ValueError(f"strategy must be 'cut_node' or 'any_node', got {strategy!r}")

    H = G.copy()
    total_pairs = H.number_of_edges()
    rows: list[dict] = []
    dropped = 0
    step = 0

    while True:
        largest = _largest_cc_by_pairs(H)
        sub = H.subgraph(largest)
        lp_pairs = sub.number_of_edges()
        retained = total_pairs - dropped
        frac_ret = lp_pairs / retained if retained > 0 else 0.0
        n_pieces = sum(1 for cc in nx.connected_components(H)
                       if H.subgraph(cc).number_of_edges() > 0)

        row = {
            'step': step,
            'pairs_dropped': dropped,
            'dropped_frac': round(dropped / total_pairs, 6),
            'retained_pairs': retained,
            'largest_cc_pairs': lp_pairs,
            'largest_frac_of_retained': round(frac_ret, 6),
            'largest_frac_of_original': round(lp_pairs / total_pairs, 6),
            'n_components': n_pieces,
        }

        if frac_ret <= target_frac or step >= max_steps:
            rows.append(row)
            break

        # Choose the node to remove from the largest CC.
        simple_sub = nx.Graph(sub)
        arts = set(nx.articulation_points(simple_sub))
        if strategy == 'cut_node':
            cand_pool = arts if arts else set(largest)
        else:
            cand_pool = set(largest)
        cand = max(cand_pool, key=lambda n: H.degree(n))  # heaviest by pair_mass
        side, cid = cand.split(':', 1)

        row['removed_node'] = cid
        row['removed_side'] = side
        row['removed_pair_mass'] = int(H.degree(cand))
        row['removed_simple_degree'] = int(simple_sub.degree(cand))
        row['removed_is_cut_node'] = cand in arts
        rows.append(row)

        dropped += H.degree(cand)
        H.remove_node(cand)
        step += 1

    return pd.DataFrame(rows)


def plot_peel_curve(df: pd.DataFrame, title: str, target_frac: float, out_png: Path) -> None:
    """Drop% (x) vs largest-CC-share-of-retained% (y), with the target line."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = df['dropped_frac'] * 100
    y = df['largest_frac_of_retained'] * 100
    ax.plot(x, y, color='#1f77b4', linewidth=1.8, marker='o', markersize=2.5)

    ax.axhline(target_frac * 100, color='#d62728', linestyle='--', linewidth=1.2,
               label=f'feasibility target ({target_frac:.0%})')

    crossed = df[df['largest_frac_of_retained'] <= target_frac]
    if len(crossed):
        c = crossed.iloc[0]
        ax.axvline(c['dropped_frac'] * 100, color='#2ca02c', linestyle=':', linewidth=1.2)
        ax.annotate(
            f"{int(c['step'])} hubs removed\n"
            f"{c['pairs_dropped']:,} pairs ({c['dropped_frac']:.1%}) dropped",
            xy=(c['dropped_frac'] * 100, target_frac * 100),
            xytext=(c['dropped_frac'] * 100 + 1.5, target_frac * 100 + 8),
            fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.0),
        )

    ax.set_xlabel('cumulative pairs dropped (% of pair universe)')
    ax.set_ylabel('largest CC (% of retained pairs)')
    ax.set_ylim(0, 102)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=8.5, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet'))
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'))
    p.add_argument('--clusters_nt',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_nt'))
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'],
                   metavar=('SLOT_A', 'SLOT_B'))
    p.add_argument('--alphabet', default='aa', choices=['aa', 'nt_cds'])
    p.add_argument('--threshold', default='t095',
                   help='Single cluster threshold to peel at (default t095).')
    p.add_argument('--target_frac', type=float, default=0.80,
                   help='Stop when largest CC <= this share of retained pairs (default 0.80).')
    p.add_argument('--strategy', default='cut_node', choices=['cut_node', 'any_node'],
                   help="cut_node (default): remove the heaviest articulation point. "
                        "any_node: remove the heaviest node regardless (contrast).")
    p.add_argument('--max_steps', type=int, default=100_000)
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/bigraph_hub_peel')
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
        raise SystemExit(f"missing cluster parquet for {args.alphabet} {args.threshold} "
                         f"under {clusters_root}")

    G, n_unmapped = build_bipartite_multigraph(universe, cmap_a, cmap_b, args.alphabet)
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} pair-universe rows dropped (unmapped endpoint).")
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges (multigraph)")

    print(f"\nPeeling ({args.strategy}) until largest CC <= {args.target_frac:.0%} of retained ...")
    t0 = time.time()
    df = hub_peel(G, target_frac=args.target_frac, strategy=args.strategy,
                  max_steps=args.max_steps)
    elapsed = time.time() - t0

    stem = f'hub_peel_{slug}_{args.alphabet}_{args.threshold}_{args.strategy}'
    csv_path = out_dir / f'{stem}.csv'
    df.to_csv(csv_path, index=False)

    final = df.iloc[-1]
    reached = final['largest_frac_of_retained'] <= args.target_frac
    n_removed = int(final['step'])
    print(f"\n  {'REACHED' if reached else 'STOPPED (max_steps)'} after removing "
          f"{n_removed} node(s) in {elapsed:.1f}s")
    print(f"  pairs dropped: {final['pairs_dropped']:,} "
          f"({final['dropped_frac']:.1%} of the {G.number_of_edges():,}-pair universe)")
    print(f"  largest CC now: {final['largest_cc_pairs']:,} pairs "
          f"= {final['largest_frac_of_retained']:.1%} of retained, "
          f"{int(final['n_components'])} components")

    removed = df[df['removed_node'].notna()] if 'removed_node' in df.columns else df.iloc[0:0]
    if len(removed):
        # state AFTER a removal = the next step's row (each row records the
        # state BEFORE its own removal), so show largest_frac before -> after.
        frac_by_step = dict(zip(df['step'], df['largest_frac_of_retained']))
        print(f"\n  removal order (heaviest hubs first; largest CC before -> after each cut):")
        for r in removed.head(8).itertuples():
            after = frac_by_step.get(r.step + 1, float('nan'))
            cut_tag = 'cut' if r.removed_is_cut_node else 'non-cut'
            print(f"    {r.step:>3}. {r.removed_side}:{r.removed_node:<12} "
                  f"deg={int(r.removed_simple_degree):<5} {int(r.removed_pair_mass):>6,} pairs "
                  f"[{cut_tag}]  largest {r.largest_frac_of_retained:.1%} -> {after:.1%}")

    png_path = out_dir / f'{stem}.png'
    plot_peel_curve(
        df,
        title=(f'{slot_a}-{slot_b} {args.alphabet} {args.threshold} — greedy hub-peel ({args.strategy})'),
        target_frac=args.target_frac,
        out_png=png_path,
    )
    print(f"\nwrote {csv_path}")
    print(f"wrote {png_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
