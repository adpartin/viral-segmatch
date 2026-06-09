"""2D matrix view of the HA-NA cluster bigraph: islands of positive pairs + straddling pairs.

The `bigraph_*` family treats the cluster-level co-occurrence structure as a
*graph* (CCs, hubs, cuts). This script renders the *same object* as its
**biadjacency matrix**: y-axis = slot-a (HA) clusters, x-axis = slot-b (NA)
clusters, one marker per occupied cell, marker size ∝ the number of positive
pairs on that (HA-cluster, NA-cluster) — i.e. the pair-weighted edge weight
(clusters.md §6.0 pair-weighted view). This is the geometric analogue of the
DataSAIL Fig-1 interaction matrix.

The structure is only legible once the axes are *ordered* so co-occurring
clusters sit adjacent. Two orderings, side by side (the same pairs, two lenses):

  - LEFT — spectral-atom order. Axes ordered by the recursive spectral min-cut's
    atoms (`bigraph_min_cut.min_cut_recursive`). Kept pairs land ON the
    block-diagonal of atom bands; the cut's dropped (straddling) pairs fall OFF
    it. Cells colored by atom (top-K islands; rest grey). This is "what the cut
    produces".
  - RIGHT — subtype-grouped order. Axes grouped by each cluster's modal H- and
    N-subtype (`bigraph_cut_subtype.pair_key_to_subtype`, from
    flu_genomes_metadata_parsed.csv). Islands = (Hx,Ny) subtype blocks; reassortant
    pairs sit off the dominant blocks. This is "the latent biological islands".

Both panels overlay the cut's straddling pairs as red ×. A companion
subtype-level heatmap (H-subtype × N-subtype pair counts) gives the coarse,
immediately-readable version of the same islands.

Both alphabets work: `load_cluster_map` is membership-backed (`cluster_source`),
keying aa on `seq_hash` and nt_cds on `cds_dna_hash`, so `--alphabet nt_cds`
runs against `clusters_nt_cds`.

CLI:
    python -m src.analysis.bigraph_pair_2d \\
        [--schema_pair HA NA] [--alphabet aa] [--thresholds t095 t090] \\
        [--method spectral] [--drift_pp 0.05] [--top_atoms 8] \\
        [--out_dir results/flu/July_2025/runs/2D_cluster_maps]

Outputs (under --out_dir):
    pair_2d_{slug}_{alphabet}_{tXXX}.png        two-panel cluster scatter
    subtype_heatmap_{slug}_{alphabet}_{tXXX}.png  H-subtype × N-subtype counts
    cells_{slug}_{alphabet}_{tXXX}.csv          per-occupied-cell table
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe
from src.analysis.bigraph_properties import load_cluster_map, build_bipartite_multigraph
from src.analysis.bigraph_min_cut import min_cut_recursive
from src.analysis.bigraph_cut_subtype import pair_key_to_subtype

_H_RE = re.compile(r'H\d+')
_N_RE = re.compile(r'N\d+')


def _h_part(subtype: str) -> str:
    m = _H_RE.search(str(subtype))
    return m.group(0) if m else 'H?'


def _n_part(subtype: str) -> str:
    m = _N_RE.search(str(subtype))
    return m.group(0) if m else 'N?'


def _modal(s: pd.Series) -> str:
    m = s.mode()
    return m.iat[0] if len(m) else 'unknown'


def build_pair_table(universe, cmap_a, cmap_b, alphabet, node_atom, subtype_df):
    """One row per canonical pair: cluster nodes, atoms, kept flag, H/N subtype.

    `kept` is True when both cluster endpoints landed in the same min-cut atom
    (an on-block pair); False = a straddling/dropped pair. `h`/`n` are the modal
    H- and N-subtype carried by the pair's isolate(s).
    """
    col_a = 'seq_hash_a' if alphabet == 'aa' else 'dna_hash_a'
    col_b = 'seq_hash_b' if alphabet == 'aa' else 'dna_hash_b'
    u = universe.copy()
    u = u[u[col_a].isin(cmap_a) & u[col_b].isin(cmap_b)].copy()
    u['node_a'] = 'a:' + u[col_a].map(cmap_a).astype(str)
    u['node_b'] = 'b:' + u[col_b].map(cmap_b).astype(str)
    u['atom_a'] = u['node_a'].map(node_atom)
    u['atom_b'] = u['node_b'].map(node_atom)
    u = u.dropna(subset=['atom_a', 'atom_b'])
    u['atom_a'] = u['atom_a'].astype(int)
    u['atom_b'] = u['atom_b'].astype(int)
    u['kept'] = u['atom_a'] == u['atom_b']
    u = u.merge(subtype_df[['pair_key', 'subtype']], on='pair_key', how='left')
    u['subtype'] = u['subtype'].fillna('unknown')
    u['h'] = u['subtype'].map(_h_part)
    u['n'] = u['subtype'].map(_n_part)
    return u


def _rank_by(keys_sorted) -> dict:
    return {k: i for i, k in enumerate(keys_sorted)}


def _band_edges(ordered_labels):
    """Boundaries + (center, label) for contiguous runs of equal label."""
    bounds, centers = [], []
    start = 0
    for i in range(1, len(ordered_labels) + 1):
        if i == len(ordered_labels) or ordered_labels[i] != ordered_labels[start]:
            bounds.append(i)
            centers.append(((start + i - 1) / 2.0, ordered_labels[start]))
            start = i
    return bounds[:-1], centers  # drop the final outer boundary


def _node_orderings(u):
    """Per-node atom, modal subtype, mass; spectral-atom and subtype rank dicts."""
    a = u.groupby('node_a').agg(atom=('atom_a', 'first'),
                                sub=('h', _modal), mass=('node_a', 'size'))
    b = u.groupby('node_b').agg(atom=('atom_b', 'first'),
                                sub=('n', _modal), mass=('node_b', 'size'))
    a_atom = _rank_by(a.sort_values(['atom', 'mass'], ascending=[True, False]).index)
    b_atom = _rank_by(b.sort_values(['atom', 'mass'], ascending=[True, False]).index)
    a_sub_idx = a.sort_values(['sub', 'mass'], ascending=[True, False]).index
    b_sub_idx = b.sort_values(['sub', 'mass'], ascending=[True, False]).index
    a_sub = _rank_by(a_sub_idx)
    b_sub = _rank_by(b_sub_idx)
    a_bands = _band_edges([a.loc[k, 'sub'] for k in a_sub_idx])
    b_bands = _band_edges([b.loc[k, 'sub'] for k in b_sub_idx])
    return {'a_atom': a_atom, 'b_atom': b_atom, 'a_sub': a_sub, 'b_sub': b_sub,
            'a_bands': a_bands, 'b_bands': b_bands}


def aggregate_cells(u):
    """One row per occupied (HA-cluster, NA-cluster) cell."""
    return u.groupby(['node_a', 'node_b']).agg(
        n_pairs=('node_a', 'size'),
        kept=('kept', 'first'),
        atom=('atom_a', 'first'),   # == atom_b for kept cells; ignored for straddling
    ).reset_index()


def _atom_dominant_subtype(u, atom_id):
    a = u[(u['atom_a'] == atom_id) & u['kept']]
    if len(a) == 0:
        return 'unknown', 0.0
    vc = a['subtype'].value_counts()
    return vc.index[0], vc.iloc[0] / len(a)


def _sizes(n_pairs):
    return 4.0 + 7.0 * np.log10(n_pairs.to_numpy(dtype=float) + 1.0)


def plot_two_panel(u, cells, ranks, *, slot_a, slot_b, alphabet, threshold,
                   top_atoms, dropped_frac, n_atoms, out_png, no_cut=False, legend='cc'):
    cmap = plt.get_cmap('tab10')
    atom_color = {a: cmap(i % 10) for i, a in enumerate(range(top_atoms))}
    kept = cells[cells['kept']].copy()
    strad = cells[~cells['kept']].copy()

    def kept_colors(atoms):
        return [atom_color.get(int(a), (0.80, 0.80, 0.80, 1.0)) for a in atoms]

    fig, axes = plt.subplots(1, 2, figsize=(21, 9.2))

    # ---- LEFT: spectral-atom order -----------------------------------------
    ax = axes[0]
    kx = kept['node_b'].map(ranks['b_atom']).to_numpy()
    ky = kept['node_a'].map(ranks['a_atom']).to_numpy()
    ax.scatter(kx, ky, s=_sizes(kept['n_pairs']), c=kept_colors(kept['atom']),
               edgecolors='none', alpha=0.85)
    if len(strad):
        sx = strad['node_b'].map(ranks['b_atom']).to_numpy()
        sy = strad['node_a'].map(ranks['a_atom']).to_numpy()
        ax.scatter(sx, sy, s=_sizes(strad['n_pairs']) + 6, marker='x',
                   c='#d62728', linewidths=0.9, alpha=0.9, label='straddling (cut)')
    _order = 'connected component' if no_cut else 'min-cut atom'
    ax.set_xlabel(f'{slot_b} clusters — rank-ordered by {_order}', fontsize=10)
    ax.set_ylabel(f'{slot_a} clusters — rank-ordered by {_order}', fontsize=10)
    ax.set_title(('Natural-CC order — pairs on the CC block-diagonal (no cut)'
                  if no_cut else
                  'Spectral-atom order — kept pairs on the atom block-diagonal;\n'
                  'red × = straddling pairs the cut drops'), fontsize=11)
    atom_pairs = u.groupby('atom_a').size()
    total = len(u)
    if legend == 'cc':
        def _atom_label(a):
            n = int(atom_pairs.get(a, 0))
            return f'CC {a + 1}: {n:,} ({100 * n / total:.1f}%)'
        legend_title = 'top CCs (by pair count)'
    else:
        def _atom_label(a):
            dom, pur = _atom_dominant_subtype(u, a)
            return f'atom {a}: {dom} ({pur:.0%})'
        legend_title = 'islands (by pair-mass)'
    legend_atoms = [
        Line2D([0], [0], marker='o', linestyle='', markersize=7,
               markerfacecolor=atom_color[a], markeredgecolor='none', label=_atom_label(a))
        for a in range(min(top_atoms, n_atoms))
    ]
    if len(strad):
        legend_atoms.append(Line2D([0], [0], marker='x', linestyle='', markersize=8,
                                   markeredgecolor='#d62728', label='straddling (cut)'))
    ax.legend(handles=legend_atoms, loc='lower right', fontsize=7.5,
              frameon=True, framealpha=0.9, title=legend_title)

    # ---- RIGHT: subtype-grouped order --------------------------------------
    ax = axes[1]
    kx = kept['node_b'].map(ranks['b_sub']).to_numpy()
    ky = kept['node_a'].map(ranks['a_sub']).to_numpy()
    ax.scatter(kx, ky, s=_sizes(kept['n_pairs']), c='#4c72b0',
               edgecolors='none', alpha=0.6, label='kept pair')
    if len(strad):
        sx = strad['node_b'].map(ranks['b_sub']).to_numpy()
        sy = strad['node_a'].map(ranks['a_sub']).to_numpy()
        ax.scatter(sx, sy, s=_sizes(strad['n_pairs']) + 6, marker='x',
                   c='#d62728', linewidths=0.9, alpha=0.9, label='straddling (cut)')
    for xb in ranks['b_bands'][0]:
        ax.axvline(xb - 0.5, color='#bbbbbb', linewidth=0.5, alpha=0.7)
    for yb in ranks['a_bands'][0]:
        ax.axhline(yb - 0.5, color='#bbbbbb', linewidth=0.5, alpha=0.7)
    ax.set_xticks([c for c, _ in ranks['b_bands'][1]])
    ax.set_xticklabels([lab for _, lab in ranks['b_bands'][1]], rotation=90, fontsize=7)
    ax.set_yticks([c for c, _ in ranks['a_bands'][1]])
    ax.set_yticklabels([lab for _, lab in ranks['a_bands'][1]], fontsize=7)
    ax.set_xlabel(f'{slot_b} clusters — grouped by modal N-subtype', fontsize=10)
    ax.set_ylabel(f'{slot_a} clusters — grouped by modal H-subtype', fontsize=10)
    ax.set_title(('Subtype-grouped order — islands = (Hx,Ny) blocks (no cut)'
                  if no_cut else
                  'Subtype-grouped order — islands = (Hx,Ny) blocks;\n'
                  'red × straddling pairs sit off the dominant blocks'), fontsize=11)
    ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

    cut_note = 'no cut (natural CCs)' if no_cut else f'cut drops {dropped_frac:.2%} (straddling)'
    fig.suptitle(
        f'{slot_a}-{slot_b} {alphabet} {threshold}: cluster-pair map  ·  '
        f'{len(u):,} pairs, {len(cells):,} occupied cells, {n_atoms:,} atoms  ·  {cut_note}\n'
        f'each point = one cluster-pair (occupied cell);  marker size ∝ # sequence-pairs',
        fontsize=11, y=1.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_subtype_heatmap(u, *, slot_a, slot_b, alphabet, threshold,
                         max_h, max_n, out_png):
    ct = u.groupby(['h', 'n']).size().unstack(fill_value=0)
    h_order = ct.sum(axis=1).sort_values(ascending=False).index[:max_h]
    n_order = ct.sum(axis=0).sort_values(ascending=False).index[:max_n]
    ct = ct.loc[h_order, n_order]

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(n_order) + 3),
                                    max(5, 0.5 * len(h_order) + 2)))
    im = ax.imshow(np.log10(ct.to_numpy() + 1.0), cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(n_order)))
    ax.set_xticklabels(n_order, rotation=0, fontsize=9)
    ax.set_yticks(range(len(h_order)))
    ax.set_yticklabels(h_order, fontsize=9)
    ax.set_xlabel(f'{slot_b} N-subtype', fontsize=10)
    ax.set_ylabel(f'{slot_a} H-subtype', fontsize=10)
    total = int(ct.to_numpy().sum())
    for i in range(len(h_order)):
        for j in range(len(n_order)):
            v = int(ct.iat[i, j])
            if v > 0:
                ax.annotate(f'{v:,}', (j, i), ha='center', va='center', fontsize=7,
                            color='white' if np.log10(v + 1) < np.log10(ct.to_numpy().max() + 1) * 0.6 else 'black')
    fig.colorbar(im, ax=ax, label='log10(pairs + 1)')
    ax.set_title(f'{slot_a}-{slot_b} {alphabet} {threshold}: positive pairs by subtype combo\n'
                 f'(top {len(h_order)} H × top {len(n_order)} N; {total:,} pairs shown)',
                 fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    plt.close(fig)


def run_threshold(universe, subtype_df, clusters_root, *, slot_a, slot_b, alphabet,
                  threshold, method, drift_pp, seed, top_atoms, max_h, max_n, out_dir,
                  no_cut=False, skip_heatmap=False, legend='cc'):
    cmap_a = load_cluster_map(clusters_root, slot_a, threshold)
    cmap_b = load_cluster_map(clusters_root, slot_b, threshold)
    if not cmap_a or not cmap_b:
        print(f'  [{alphabet} {threshold}] missing cluster parquet; skipping.')
        return
    G, n_unmapped = build_bipartite_multigraph(universe, cmap_a, cmap_b, alphabet)
    if no_cut:
        # Pre-fragmentation view: atoms = natural CCs. Every pair's two endpoints
        # share its CC by definition, so nothing straddles -> no red x markers.
        comps = sorted(nx.connected_components(G),
                       key=lambda c: G.subgraph(c).number_of_edges(), reverse=True)
        dropped_frac = 0.0
    else:
        df, H, dropped_edges = min_cut_recursive(
            G, method=method, drift_pp=drift_pp, seed=seed, return_partition=True)
        dropped_frac = float(df.iloc[-1]['dropped_frac'])
        comps = sorted(nx.connected_components(H),
                       key=lambda c: H.subgraph(c).size(weight='weight'), reverse=True)
    node_atom = {n: i for i, c in enumerate(comps) for n in c}
    n_atoms = len(comps)

    u = build_pair_table(universe, cmap_a, cmap_b, alphabet, node_atom, subtype_df)
    cells = aggregate_cells(u)
    ranks = _node_orderings(u)

    slug = f'{slot_a.lower()}_{slot_b.lower()}'
    plot_two_panel(u, cells, ranks, slot_a=slot_a, slot_b=slot_b, alphabet=alphabet,
                   threshold=threshold, top_atoms=top_atoms, dropped_frac=dropped_frac,
                   n_atoms=n_atoms, no_cut=no_cut, legend=legend,
                   out_png=out_dir / f'pair_2d_{slug}_{alphabet}_{threshold}.png')
    if not skip_heatmap:
        plot_subtype_heatmap(u, slot_a=slot_a, slot_b=slot_b, alphabet=alphabet,
                             threshold=threshold, max_h=max_h, max_n=max_n,
                             out_png=out_dir / f'subtype_heatmap_{slug}_{alphabet}_{threshold}.png')
    cells_out = cells.copy()
    cells_out.to_csv(out_dir / f'cells_{slug}_{alphabet}_{threshold}.csv', index=False)
    n_strad = int((~cells['kept']).sum())
    wrote = 'pair_2d + cells' if skip_heatmap else 'pair_2d + subtype_heatmap + cells'
    print(f'  [{alphabet} {threshold}] {len(u):,} pairs, {len(cells):,} cells, '
          f'{n_atoms:,} atoms, cut drops {dropped_frac:.2%} '
          f'({n_strad:,} straddling cells); wrote {wrote}.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'))
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'))
    p.add_argument('--clusters_nt_cds',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_nt_cds'))
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'],
                   metavar=('SLOT_A', 'SLOT_B'))
    p.add_argument('--alphabet', default='aa', choices=['aa', 'nt_cds'])
    p.add_argument('--thresholds', nargs='+', default=['t095', 't090'])
    p.add_argument('--method', default='spectral', choices=['spectral', 'kl'])
    p.add_argument('--drift_pp', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--top_atoms', type=int, default=8)
    p.add_argument('--max_h', type=int, default=14, help='Max H-subtype rows in the heatmap.')
    p.add_argument('--max_n', type=int, default=11, help='Max N-subtype cols in the heatmap.')
    p.add_argument('--no_cut', action='store_true',
                   help='Pre-fragmentation view: order by natural CC, draw no straddling (red x) markers.')
    p.add_argument('--skip_heatmap', action='store_true',
                   help='Skip the subtype heatmap (t-invariant; one copy per pair suffices).')
    p.add_argument('--legend', default='cc', choices=['cc', 'subtype'],
                   help="Left-panel legend: 'cc' (top CCs by pair count + %) or 'subtype' (dominant hn_subtype + purity).")
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/2D_cluster_maps')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slot_a, slot_b = args.schema_pair
    clusters_root = Path(args.clusters_aa if args.alphabet == 'aa' else args.clusters_nt_cds)

    print(f'Loading pair universe + subtype map for {slot_a}-{slot_b} ...')
    universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
    subtype_df = pair_key_to_subtype(Path(args.cds_final), slot_a, slot_b)
    print(f'  {len(universe):,} pairs; {subtype_df["subtype"].nunique()} distinct modal subtypes')

    for threshold in args.thresholds:
        run_threshold(universe, subtype_df, clusters_root, slot_a=slot_a, slot_b=slot_b,
                      alphabet=args.alphabet, threshold=threshold, method=args.method,
                      drift_pp=args.drift_pp, seed=args.seed, top_atoms=args.top_atoms,
                      max_h=args.max_h, max_n=args.max_n, out_dir=out_dir,
                      no_cut=args.no_cut, skip_heatmap=args.skip_heatmap, legend=args.legend)

    print('\nDone.')


if __name__ == '__main__':
    main()
