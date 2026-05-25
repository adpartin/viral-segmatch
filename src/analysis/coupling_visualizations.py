"""HA-cluster ↔ metadata coupling visualizations across the idXX sweep.

Sibling to the single-slot HA-only sweep work. Quantifies how strongly
the HA cluster boundary predicts each of three biological metadata axes
(NA subtype, host, year-bin), and how that coupling evolves as the
cluster-identity threshold loosens (id100 → id095).

Three plots:

  (a) sweep_coupling_vs_idxx.png — Cramér's V per axis (subtype, host,
      year-bin) at each of the 6 idXX. Answers "does single-slot
      relaxation eventually decouple the slots?"

  (b) sweep_coupling_heatmap_id<NN>.png — for one chosen idXX, top-N
      HA clusters × metadata categories, color = fraction of cluster's
      pairs that fall in each category. Visually shows the cluster-
      purity pattern.

  (c) sweep_coupling_purity_bars.png — fraction of pairs that sit in
      HA clusters that are ≥95% pure for each axis, at each idXX.
      Compact summary of (a).

Reads the 6 single-slot HA-only datasets and pulls `cluster_id_a`
(joined onto the dataset CSVs by Stage 3) and the metadata cols
`hn_subtype_b`, `host_a`, `year_a`. The y_a/host_a/etc. cols are
positive-pair-only; we restrict to label==1.

Run with:
    python -m src.analysis.coupling_visualizations \\
        --out_dir results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate/coupling \\
        --heatmap_idxx 98
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SWEEP_THRESHOLDS = [100, 99, 98, 97, 96, 95]
DATASET_GLOB = ('data/datasets/flu/July_2025/runs/'
                'dataset_flu_ha_na_cluster_aa_id{idxx:03d}_HAonly_*')

# Axis definitions: (column, label-extractor, plot title).
# NA subtype comes from `hn_subtype_b` (the NA slot's hn_subtype) — we
# extract the 'Nx' part. host is `host_a`. year is binned into 5-year
# windows to keep the contingency table tractable.
AXES = {
    'NA_subtype': lambda df: df['hn_subtype_b'].astype(str).str.extract(r'(N\d+)')[0],
    'host':       lambda df: df['host_a'].astype(str),
    'year_bin':   lambda df: (pd.to_numeric(df['year_a'], errors='coerce') // 5 * 5).astype('Int64').astype(str),
}


def _load_positives(idxx: int) -> pd.DataFrame:
    import glob
    matches = sorted(glob.glob(DATASET_GLOB.format(idxx=idxx)))
    if not matches:
        raise FileNotFoundError(f"No dataset dir for idxx={idxx}")
    d = Path(matches[0])
    frames = []
    for sp in ('train', 'val', 'test'):
        df = pd.read_csv(d / f'{sp}_pairs.csv', low_memory=False,
                         keep_default_na=False, na_values=[''])
        df['orig_split'] = sp
        frames.append(df)
    pos = pd.concat(frames, ignore_index=True)
    return pos[pos['label'] == 1].reset_index(drop=True)


def _cramers_v(xt: pd.DataFrame) -> float:
    """V = sqrt(χ² / (n · min(r-1, k-1))). 0 = independent; 1 = perfect."""
    if xt.size == 0 or xt.values.sum() == 0:
        return float('nan')
    chi2 = chi2_contingency(xt.values)[0]
    n = xt.values.sum()
    r, k = xt.shape
    denom = n * max(min(r, k) - 1, 1)
    return float(np.sqrt(chi2 / denom))


def _purity_fraction(xt: pd.DataFrame, threshold: float = 0.95) -> float:
    """Fraction of pairs that sit in a row (HA cluster) where one column
    (axis category) holds ≥ `threshold` of the row's mass.
    Weighted by row size so big clusters count more than singletons."""
    row_sums = xt.sum(axis=1).values
    if row_sums.sum() == 0:
        return float('nan')
    p = xt.values / row_sums.reshape(-1, 1)
    pure = p.max(axis=1) >= threshold
    return float(np.average(pure, weights=row_sums))


def aggregate_coupling(thresholds: list = SWEEP_THRESHOLDS) -> pd.DataFrame:
    """Returns one row per (idxx, axis) with Cramér's V and purity %."""
    rows = []
    for idxx in thresholds:
        pos = _load_positives(idxx)
        for axis_name, extractor in AXES.items():
            pos['_axis_val'] = extractor(pos)
            sub = pos.dropna(subset=['cluster_id_a', '_axis_val'])
            xt = pd.crosstab(sub['cluster_id_a'], sub['_axis_val'])
            v = _cramers_v(xt)
            pur95 = _purity_fraction(xt, threshold=0.95)
            pur90 = _purity_fraction(xt, threshold=0.90)
            rows.append({'idxx': idxx, 'axis': axis_name,
                         'n_pairs': len(sub),
                         'n_clusters': xt.shape[0],
                         'n_axis_categories': xt.shape[1],
                         'cramers_v': v,
                         'pct_pairs_in_95pct_pure_clusters': 100.0 * pur95,
                         'pct_pairs_in_90pct_pure_clusters': 100.0 * pur90})
    return pd.DataFrame(rows)


def plot_v_vs_idxx(coupling_df: pd.DataFrame, out_path: Path) -> None:
    """Cramér's V per axis vs idXX. One line per axis, x = idXX (id↓ right)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    idxx_sorted = sorted(coupling_df['idxx'].unique(), reverse=True)
    x_positions = {v: i for i, v in enumerate(idxx_sorted)}

    style = {
        'NA_subtype': {'color': '#1f77b4', 'marker': 'o', 'label': 'NA subtype'},
        'host':       {'color': '#ff7f0e', 'marker': 's', 'label': 'host'},
        'year_bin':   {'color': '#2ca02c', 'marker': '^', 'label': 'year (5-yr bin)'},
    }
    for axis_name, s in style.items():
        sub = coupling_df[coupling_df['axis'] == axis_name].sort_values('idxx', ascending=False)
        if sub.empty:
            continue
        xs = [x_positions[v] for v in sub['idxx']]
        ax.plot(xs, sub['cramers_v'].values, color=s['color'], linewidth=1.5,
                marker=s['marker'], markersize=10, alpha=0.85,
                markeredgecolor='black', markeredgewidth=0.5, label=s['label'])

    ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5, label='independence')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3, label='perfect coupling')
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([f'id{v:03d}' for v in x_positions.keys()])
    ax.set_xlabel('HA cluster threshold (lighter constraint → heavier)')
    ax.set_ylabel("Cramér's V (HA cluster ↔ metadata axis)")
    ax.set_title("HA cluster boundary ↔ metadata-axis coupling across the sweep\n"
                 "(0 = independent; 1 = HA cluster perfectly predicts the axis value)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote V-vs-idXX plot: {out_path}')


def plot_purity_bars(coupling_df: pd.DataFrame, out_path: Path) -> None:
    """Fraction of pairs in ≥95%-pure HA clusters, per (axis, idxx)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    axes_order = ['NA_subtype', 'host', 'year_bin']
    idxx_sorted = sorted(coupling_df['idxx'].unique(), reverse=True)
    x = np.arange(len(idxx_sorted))
    width = 0.25
    style = {'NA_subtype': '#1f77b4', 'host': '#ff7f0e', 'year_bin': '#2ca02c'}
    for i, axis_name in enumerate(axes_order):
        sub = coupling_df[coupling_df['axis'] == axis_name].set_index('idxx').loc[idxx_sorted]
        ax.bar(x + (i - 1) * width, sub['pct_pairs_in_95pct_pure_clusters'].values,
               width, color=style[axis_name], label=axis_name.replace('_', ' '),
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'id{v:03d}' for v in idxx_sorted])
    ax.set_ylabel('% of pairs in HA clusters ≥95% pure for this axis')
    ax.set_xlabel('HA cluster threshold (lighter → heavier)')
    ax.set_title('Cluster-purity fraction per axis across the sweep\n'
                 '(high % = HA cluster boundary essentially is the axis boundary)')
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote purity-bars plot: {out_path}')


def plot_heatmap(idxx: int, out_path: Path, top_n: int = 30) -> None:
    """Top-N HA clusters × NA subtype, color = fraction of cluster in subtype."""
    pos = _load_positives(idxx)
    pos['na_subtype'] = AXES['NA_subtype'](pos)
    sub = pos.dropna(subset=['cluster_id_a', 'na_subtype'])
    xt = pd.crosstab(sub['cluster_id_a'], sub['na_subtype'])
    sizes = xt.sum(axis=1).sort_values(ascending=False)
    top = xt.loc[sizes.head(top_n).index]
    # Sort columns by overall frequency (most common subtypes on the left).
    col_order = xt.sum(axis=0).sort_values(ascending=False).index.tolist()
    top = top[col_order]
    p = top.div(top.sum(axis=1), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(col_order) + 3),
                                     max(6, 0.25 * top_n + 2)))
    im = ax.imshow(p.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha='right')
    ax.set_yticks(range(top.shape[0]))
    ax.set_yticklabels([f'{c} (n={int(sizes.loc[c])})' for c in top.index],
                        fontsize=8)
    ax.set_xlabel('NA subtype')
    ax.set_ylabel(f'HA cluster id (top {top_n} by size)')
    ax.set_title(f'HA cluster × NA subtype at id{idxx:03d}\n'
                 f'(color = fraction of cluster\'s pairs in that subtype; '
                 f'1 row ≈ 1 color → cluster is subtype-pure)')
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('fraction')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote heatmap plot: {out_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_dir', type=Path,
                    default=Path('results/flu/July_2025/runs/split_separation_mmd'
                                 '/sweep_aggregate/coupling'))
    ap.add_argument('--heatmap_idxx', type=int, default=98,
                    help='Which idXX to use for the heatmap. Default 98.')
    ap.add_argument('--top_n', type=int, default=30,
                    help='Top-N HA clusters by size in the heatmap.')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coupling_df = aggregate_coupling()
    coupling_csv = args.out_dir / 'coupling_summary.csv'
    coupling_df.to_csv(coupling_csv, index=False)
    print(f'Wrote coupling summary CSV: {coupling_csv}')
    print()
    pivot = coupling_df.pivot(index='idxx', columns='axis', values='cramers_v').sort_index(ascending=False).round(4)
    print("Cramér's V per (idxx, axis):")
    print(pivot.to_string())

    plot_v_vs_idxx(coupling_df, args.out_dir / 'sweep_coupling_vs_idxx.png')
    plot_purity_bars(coupling_df, args.out_dir / 'sweep_coupling_purity_bars.png')
    plot_heatmap(args.heatmap_idxx,
                 args.out_dir / f'sweep_coupling_heatmap_id{args.heatmap_idxx:03d}.png',
                 top_n=args.top_n)


if __name__ == '__main__':
    main()
