"""Isolate-level test: are the min-cut's dropped bridges biological reassortants?

The spectral min-cut (`bipartite_min_cut.py`) fragments the HA-NA mega-CC into
subtype-organized atoms by dropping <1% of pairs; `bipartite_cut_subtype.py`
showed 92% of dropped pairs bridge atoms of *different* dominant subtype. This
script confirms that at the ISOLATE level, without relying on atom labels:

  - Decompose each pair's isolate hn_subtype into its H-lineage and N-lineage
    (H3N2 -> H3, N2). The pair's HA carries the H-lineage, its NA the N-lineage.
  - A pair is a REASSORTANT when its HA and NA come from different lineages: the
    NA is not the H-lineage's canonical (global-modal) N-partner, AND the HA is
    not the N-lineage's canonical H-partner (off both diagonals). E.g. H5N2 (H5
    usually pairs N1; N2 usually pairs H3) is a reassortant; H3N2 is not.
  - Test: are the cut's DROPPED (bridge) pairs enriched for reassortants vs the
    KEPT pairs? If yes, the structural bottleneck = reassortment events.

CLI:
    python -m src.analysis.bipartite_reassortment_check \\
        [--schema_pair HA NA] [--alphabet aa] [--threshold t095] [--method spectral]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import networkx as nx

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe
from src.analysis.bipartite_graph_properties import load_cluster_map, build_bipartite_multigraph
from src.analysis.bipartite_min_cut import min_cut_recursive
from src.analysis.bipartite_cut_subtype import pair_key_to_subtype

_HN = re.compile(r'^H(\d+)N(\d+)$')


def split_hn(s: str) -> tuple[str, str]:
    m = _HN.match(str(s))
    return (f'H{m.group(1)}', f'N{m.group(2)}') if m else ('unk', 'unk')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'))
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'))
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'])
    p.add_argument('--alphabet', default='aa')
    p.add_argument('--threshold', default='t095')
    p.add_argument('--method', default='spectral', choices=['kl', 'spectral'])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/bipartite_min_cut')
    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slot_a, slot_b = args.schema_pair
    slug = f'{slot_a.lower()}_{slot_b.lower()}'
    clusters_root = Path(args.clusters_aa)

    print(f"Loading universe + clusters for {slot_a}-{slot_b} {args.threshold} ...")
    universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
    cmap_a = load_cluster_map(clusters_root, slot_a, args.threshold)
    cmap_b = load_cluster_map(clusters_root, slot_b, args.threshold)
    G, _ = build_bipartite_multigraph(universe, cmap_a, cmap_b, args.alphabet)

    print(f"Running {args.method} min-cut ...")
    df, H, _ = min_cut_recursive(G, method=args.method, seed=args.seed, return_partition=True)
    fin = df.iloc[-1]
    print(f"  dropped {int(fin['pairs_dropped']):,} ({fin['dropped_frac']:.1%}); "
          f"{int(fin['n_pieces'])} atoms")
    node_atom = {n: i for i, c in enumerate(nx.connected_components(H)) for n in c}

    # Label each pair: kept (same atom) vs dropped (straddles), + isolate (H, N).
    u = universe.copy()
    u['atom_a'] = ('a:' + u['seq_hash_a'].map(cmap_a).astype(str)).map(node_atom)
    u['atom_b'] = ('b:' + u['seq_hash_b'].map(cmap_b).astype(str)).map(node_atom)
    u = u.dropna(subset=['atom_a', 'atom_b'])
    u['kept'] = u['atom_a'] == u['atom_b']

    sub = pair_key_to_subtype(Path(args.cds_final), slot_a, slot_b)
    u = u.merge(sub, on='pair_key', how='left')
    u['H'], u['N'] = zip(*u['subtype'].map(split_hn))

    clf = u[(u['H'] != 'unk') & (u['N'] != 'unk')].copy()
    print(f"\n  classifiable pairs: {len(clf):,} / {len(u):,} "
          f"({len(u) - len(clf):,} unk subtype)")

    # Canonical (global-modal) partner per lineage, from the full classifiable set.
    canon_N = clf.groupby('H')['N'].agg(lambda s: s.value_counts().idxmax())
    canon_H = clf.groupby('N')['H'].agg(lambda s: s.value_counts().idxmax())
    clf['reassortant'] = (clf['N'] != clf['H'].map(canon_N)) & (clf['H'] != clf['N'].map(canon_H))

    print("\n  canonical H->N partners:",
          ", ".join(f"{h}:{n}" for h, n in canon_N.sort_index().items()))

    kept = clf[clf['kept']]
    drop = clf[~clf['kept']]
    rk = kept['reassortant'].mean() if len(kept) else 0.0
    rd = drop['reassortant'].mean() if len(drop) else 0.0
    print(f"\n  KEPT pairs:    {len(kept):,}  reassortant = {rk:.1%}")
    print(f"  DROPPED pairs: {len(drop):,}  reassortant = {rd:.1%}")
    if rk > 0:
        print(f"  --> dropped pairs are {rd / rk:.1f}x enriched for reassortants")

    print(f"\n  top DROPPED (H,N) combos [* = reassortant]:")
    vc = drop.groupby(['H', 'N']).size().sort_values(ascending=False)
    for (h, n), c in vc.head(12).items():
        star = '*' if bool(((clf['H'] == h) & (clf['N'] == n) & clf['reassortant']).any()) else ' '
        print(f"    {star} {h}{n}: {c:,}")

    print(f"\n  top KEPT (H,N) combos:")
    for (h, n), c in kept.groupby(['H', 'N']).size().sort_values(ascending=False).head(6).items():
        print(f"      {h}{n}: {c:,}")

    csv = out_dir / f'reassortment_{slug}_{args.alphabet}_{args.threshold}_{args.method}.csv'
    (drop.groupby(['H', 'N']).size().rename('n_dropped').reset_index()
        .sort_values('n_dropped', ascending=False)).to_csv(csv, index=False)
    print(f"\nwrote {csv}\n\nDone.")


if __name__ == '__main__':
    main()
