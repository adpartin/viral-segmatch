"""Are the bigraph's communities the antigenic subtypes? Annotate the min-cut.

The spectral min-cut (`bigraph_min_cut.py`) fragments the HA-NA mega-CC into
feasible atoms by dropping <1% of pairs — the hypothesis (docs/results/2026-06-04_bigraph_megacc_structure_and_cutting.md)
is that those atoms are the antigenic subtypes (H3N2, H1N1, ...) and the dropped
pairs are the rare inter-subtype reassortants. This script tests that directly:

  1. run the recursive min-cut, keep the final partition (atoms = components of
     the kept graph);
  2. label every canonical pair with its isolate's `hn_subtype`
     (flu_genomes_metadata_parsed.csv, keyed by assembly_id);
  3. report, per atom, the dominant subtype and its purity; and for the dropped
     (straddling) pairs, whether the two sides are different subtypes.

A kept pair sits inside one atom (both cluster endpoints in the same component);
a dropped pair straddles two atoms (its cluster endpoints were separated). If the
big atoms are subtype-pure and the dropped pairs join different subtypes, the
cheap cut is a cross-subtype split.

CLI:
    python -m src.analysis.bigraph_cut_subtype \\
        [--schema_pair HA NA] [--alphabet aa] [--threshold t095] \\
        [--method spectral] [--top_atoms 8] \\
        [--out_dir results/flu/July_2025/runs/bigraph_min_cut]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import networkx as nx

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._pair_helpers import canonical_pair_key
from src.utils.metadata_enrichment import load_flu_metadata
from src.analysis.cluster_pair_weight_topk import load_pair_universe, _FUNCTION_TO_SHORT
from src.analysis.bigraph_properties import load_cluster_map, build_bipartite_multigraph
from src.analysis.bigraph_min_cut import min_cut_recursive


def pair_key_to_subtype(cds_final: Path, slot_a: str, slot_b: str) -> pd.DataFrame:
    """Modal hn_subtype per canonical pair_key, from isolate co-occurrence + metadata."""
    cds = pd.read_parquet(cds_final, columns=['assembly_id', 'function', 'prot_hash'])
    cds['fs'] = cds['function'].map(_FUNCTION_TO_SHORT)
    a = (cds[cds['fs'] == slot_a][['assembly_id', 'prot_hash']]
         .rename(columns={'prot_hash': 'hash_a'}))
    b = (cds[cds['fs'] == slot_b][['assembly_id', 'prot_hash']]
         .rename(columns={'prot_hash': 'hash_b'}))
    iso = a.merge(b, on='assembly_id')
    iso['pair_key'] = [canonical_pair_key(x, y)
                       for x, y in zip(iso['hash_a'], iso['hash_b'])]

    meta = load_flu_metadata()[['assembly_id', 'hn_subtype']].copy()
    meta['assembly_id'] = meta['assembly_id'].astype(str)
    iso['assembly_id'] = iso['assembly_id'].astype(str)
    iso = iso.merge(meta, on='assembly_id', how='left')
    iso['hn_subtype'] = iso['hn_subtype'].fillna('unknown')

    def _modal(s: pd.Series) -> str:
        m = s.mode()
        return m.iat[0] if len(m) else 'unknown'

    g = iso.groupby('pair_key')['hn_subtype']
    return pd.DataFrame({
        'subtype': g.agg(_modal),
        'n_subtypes': g.nunique(),
    }).reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'))
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'))
    p.add_argument('--clusters_nt',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_nt'))
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'])
    p.add_argument('--alphabet', default='aa', choices=['aa', 'nt_cds'])
    p.add_argument('--threshold', default='t095')
    p.add_argument('--method', default='spectral', choices=['kl', 'spectral'])
    p.add_argument('--drift_pp', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--top_atoms', type=int, default=8)
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/bigraph_min_cut')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slot_a, slot_b = args.schema_pair
    slug = f'{slot_a.lower()}_{slot_b.lower()}'
    clusters_root = Path(args.clusters_aa if args.alphabet == 'aa' else args.clusters_nt)

    print(f"Loading pair universe + cluster maps for {slot_a}-{slot_b} {args.threshold} ...")
    universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
    cmap_a = load_cluster_map(clusters_root, slot_a, args.threshold)
    cmap_b = load_cluster_map(clusters_root, slot_b, args.threshold)
    G, _ = build_bipartite_multigraph(universe, cmap_a, cmap_b, args.alphabet)
    print(f"  {len(universe):,} pairs; graph {G.number_of_nodes():,} nodes")

    print(f"Running {args.method} min-cut ...")
    df, H, dropped_edges = min_cut_recursive(
        G, method=args.method, drift_pp=args.drift_pp, seed=args.seed,
        return_partition=True)
    final = df.iloc[-1]
    print(f"  dropped {int(final['pairs_dropped']):,} pairs ({final['dropped_frac']:.1%}); "
          f"{int(final['n_pieces'])} atoms; feasible={bool(final['lpt_feasible'])}")

    # node -> atom (community) id, by descending atom pair-mass
    comps = sorted(nx.connected_components(H),
                   key=lambda c: H.subgraph(c).size(weight='weight'), reverse=True)
    node_atom = {n: i for i, c in enumerate(comps) for n in c}

    # Label every canonical pair: cluster endpoints -> nodes -> atoms; + subtype.
    hash_col_a = 'prot_hash_a' if args.alphabet == 'aa' else 'cds_dna_hash_a'
    hash_col_b = 'prot_hash_b' if args.alphabet == 'aa' else 'cds_dna_hash_b'
    u = universe.copy()
    u['node_a'] = 'a:' + u[hash_col_a].map(cmap_a).astype(str)
    u['node_b'] = 'b:' + u[hash_col_b].map(cmap_b).astype(str)
    u['atom_a'] = u['node_a'].map(node_atom)
    u['atom_b'] = u['node_b'].map(node_atom)
    u = u.dropna(subset=['atom_a', 'atom_b'])
    u['atom_a'] = u['atom_a'].astype(int)
    u['atom_b'] = u['atom_b'].astype(int)
    u['kept'] = u['atom_a'] == u['atom_b']

    sub = pair_key_to_subtype(Path(args.cds_final), slot_a, slot_b)
    u = u.merge(sub, on='pair_key', how='left')
    u['subtype'] = u['subtype'].fillna('unknown')

    # Per-atom subtype composition (kept pairs only).
    kept = u[u['kept']].copy()
    rows = []
    for atom_id in sorted(kept['atom_a'].unique()):
        a = kept[kept['atom_a'] == atom_id]
        vc = a['subtype'].value_counts()
        dom = vc.index[0]
        rows.append({
            'atom': atom_id,
            'n_pairs': len(a),
            'frac_of_kept': round(len(a) / len(kept), 4),
            'dominant_subtype': dom,
            'dominant_frac': round(vc.iloc[0] / len(a), 4),
            'n_subtypes': int(a['subtype'].nunique()),
            'subtype_mix': '; '.join(f"{k} {v}" for k, v in vc.head(4).items()),
        })
    atom_df = pd.DataFrame(rows).sort_values('n_pairs', ascending=False).reset_index(drop=True)

    print(f"\n  top {args.top_atoms} atoms by pairs (kept pairs = {len(kept):,}):")
    for r in atom_df.head(args.top_atoms).itertuples():
        print(f"    atom {r.atom:>3}: {r.n_pairs:>6,} pairs ({r.frac_of_kept:>5.1%})  "
              f"{r.dominant_subtype:>8} {r.dominant_frac:>5.1%} pure  "
              f"[{r.subtype_mix}]")

    # Dropped (straddling) pairs: do the two atoms differ in dominant subtype?
    dropped = u[~u['kept']].copy()
    atom_dom = dict(zip(atom_df['atom'], atom_df['dominant_subtype']))
    dropped['dom_a'] = dropped['atom_a'].map(atom_dom)
    dropped['dom_b'] = dropped['atom_b'].map(atom_dom)
    cross_subtype = (dropped['dom_a'] != dropped['dom_b']).mean() if len(dropped) else 0.0
    print(f"\n  dropped (straddling) pairs: {len(dropped):,}")
    if len(dropped):
        print(f"    {cross_subtype:.1%} join atoms whose dominant subtypes DIFFER "
              f"(cross-subtype reassortant bridges)")
        side_pairs = (dropped.assign(
            pair=dropped.apply(lambda r: ' <-> '.join(sorted([str(r['dom_a']), str(r['dom_b'])])), axis=1))
            ['pair'].value_counts().head(8))
        for k, v in side_pairs.items():
            print(f"      {k}: {v}")

    csv_path = out_dir / f'cut_subtype_{slug}_{args.alphabet}_{args.threshold}_{args.method}.csv'
    atom_df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
