"""Per-cluster pair-weight concentration (top-K) for cluster_disjoint splitter design.

For each (schema_pair, alphabet, slot-protein, threshold), rank the slot-
protein's clusters by the number of positive pairs whose endpoint on that
protein lands in the cluster, and emit the top K. The top-1 weight is the
floor on whichever split absorbs that cluster — i.e., if it exceeds the
val/test target, no whole-cluster routing can hit that target via this
cluster alone. The cumulative curve tells you where the heavy clusters end
and the long tail begins; that informs whether an alternative to LPT-greedy
bin-packing (weighted balanced-cut, ILP-with-cluster-atoms, recursive
bisection) is feasible.

Pair universe (alphabet-independent): raw isolate co-occurrence positives
per schema pair, deduped by `canonical_pair_key(seq_hash_a, seq_hash_b)` on
protein hashes — same definition the v2 builder feeds into
`cluster_disjoint_route_pos_df`. Edges = positives only (negatives are
sampled within split after routing, so they do not constrain the splitter).

Cluster mapping varies per (alphabet, slot-protein, threshold):
  - aa     uses each pair's `seq_hash_{side}`  → aa cluster.
  - nt_cds uses each pair's `dna_hash_{side}`  → nt cluster.

CLI:
    python -m src.analysis.cluster_pair_weight_topk \\
        [--cds_final    data/processed/flu/July_2025/cds_final.parquet] \\
        [--clusters_aa  data/processed/flu/July_2025/clusters_aa] \\
        [--clusters_nt  data/processed/flu/July_2025/clusters_nt] \\
        [--out_dir      results/flu/July_2025/runs/cluster_pair_weight] \\
        [--top_k 100] \\
        [--thresholds id100 id099 ...]

Outputs (under --out_dir):
    top{K}_{pair_slug}.csv     one long-form CSV per schema pair (slug =
                               lowercase joined-by-underscore, e.g. `ha_na`,
                               `pb2_pb1`). Per-pair files keep each pair's
                               table browsable independently and scale to
                               28+ pairs without one mega-CSV. Columns:
                                   alphabet, slot_protein, threshold,
                                   rank, cluster_id, weight, pct, cum_pct
    rollup.csv                 ONE combined file with all pairs (top-1/5/10/K
                               cumulative %, n_pairs_total, n_clusters_used).
                               Combined because cross-pair comparison is the
                               main rollup use case and the row count is
                               small. Columns:
                                   schema_pair, alphabet, slot_protein, threshold,
                                   n_pairs_total, n_clusters_used,
                                   top1_pct, top5_pct, top10_pct, top100_pct
    plots/concentration_{pair_slug}_{slot}_{alphabet}.png
                               Two-panel concentration plot per
                               (schema_pair, slot, alphabet) slice (rank vs
                               pct on log-y; rank vs cum_pct on linear-y),
                               one line per threshold. Only emitted for
                               HA-NA in the current driver.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._pair_helpers import canonical_pair_key


# Function full → short (matches conf/virus/flu.yaml::function_short_names).
_FUNCTION_TO_SHORT = {
    'RNA-dependent RNA polymerase PB2 subunit': 'PB2',
    'RNA-dependent RNA polymerase catalytic core PB1 subunit': 'PB1',
    'RNA-dependent RNA polymerase PA subunit': 'PA',
    'Hemagglutinin precursor': 'HA',
    'Nucleocapsid protein': 'NP',
    'Neuraminidase protein': 'NA',
    'Matrix protein 1': 'M1',
    'Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor': 'NS1',
}

# Schema pairs to analyze. HA-NA only while the bipartite-splitter
# exploration is HA-NA-focused; add other pairs back here when ready.
# Tuple: (display_label, slug, slot_a, slot_b). Slug is the lowercase
# underscore form used in output filenames; matches the convention in
# cluster_analysis_summary.py::_SCHEMA_PAIRS.
_SCHEMA_PAIRS = [
    ('HA-NA', 'ha_na', 'HA', 'NA'),
]

# Default threshold range when --thresholds is not given. Restricted to
# t100..t90 because lower thresholds collapse to ≤ a handful of clusters
# per protein for most pairs (see clusters.md § 6.1) and are uninteresting
# for splitter-design diagnostics.
_DEFAULT_THRESHOLDS = [f't{i:03d}' for i in range(100, 89, -1)]


def load_pair_universe(cds_final: Path, slot_a: str, slot_b: str) -> pd.DataFrame:
    """Raw cooccurrence pair universe for one schema pair, deduped by protein pair_key.

    Each row = one unique canonical protein pair appearing in at least one
    isolate (multi-occurrence within and across isolates collapsed).

    Columns: pair_key, seq_hash_a, seq_hash_b, dna_hash_a, dna_hash_b.
    The seq_hash_* values are protein hashes (md5 of prot_seq); the
    dna_hash_* values are DNA hashes (md5 of cds_dna). Naming follows the
    project convention: seq_hash = protein, dna_hash = DNA.
    """
    df = pd.read_parquet(
        cds_final,
        columns=['assembly_id', 'function', 'seq_hash', 'cds_dna_hash'],
    )
    df['function_short'] = df['function'].map(_FUNCTION_TO_SHORT)
    df = df[df['function_short'].isin([slot_a, slot_b])].copy()

    a_df = (
        df[df['function_short'] == slot_a][['assembly_id', 'seq_hash', 'cds_dna_hash']]
        .rename(columns={'seq_hash': 'seq_hash_a', 'cds_dna_hash': 'dna_hash_a'})
    )
    b_df = (
        df[df['function_short'] == slot_b][['assembly_id', 'seq_hash', 'cds_dna_hash']]
        .rename(columns={'seq_hash': 'seq_hash_b', 'cds_dna_hash': 'dna_hash_b'})
    )
    pairs = a_df.merge(b_df, on='assembly_id', how='inner')

    pairs['pair_key'] = [
        canonical_pair_key(a, b)
        for a, b in zip(pairs['seq_hash_a'], pairs['seq_hash_b'])
    ]
    # Mode-1 leakage dedup: collapse rows so each unique canonical protein
    # pair appears once (within- and across-isolate duplicates removed).
    pairs = pairs.drop_duplicates(subset='pair_key', keep='first').reset_index(drop=True)
    return pairs[['pair_key', 'seq_hash_a', 'seq_hash_b', 'dna_hash_a', 'dna_hash_b']]


def load_cluster_map(clusters_root: Path, slot_protein: str, threshold_id: str) -> dict[str, str]:
    """Load {hash → cluster_id} for one (slot_protein, threshold).

    The cluster parquet's `seq_hash` column holds protein hashes for aa
    clusters and DNA hashes for nt clusters (column reused). Caller is
    responsible for pairing this map with the right hash column on the
    pair-universe side.

    Returns an empty dict if the parquet does not exist.
    """
    cluster_pq = clusters_root / threshold_id / f'{slot_protein}_cluster.parquet'
    if not cluster_pq.exists():
        return {}
    df = pd.read_parquet(cluster_pq, columns=['seq_hash', 'cluster_id'])
    return dict(zip(df['seq_hash'].values, df['cluster_id'].values))


def compute_top_k_cluster_weights(
    pair_universe: pd.DataFrame,
    cluster_map: dict[str, str],
    side: str,
    alphabet: str,
    top_k: int = 100,
) -> tuple[pd.DataFrame, int, int]:
    """Rank slot-protein clusters by pair weight; return top-K + slice totals.

    Args:
        pair_universe: From `load_pair_universe`.
        cluster_map:   From `load_cluster_map`.
        side:          'a' or 'b' — which slot of the pair the slot-protein occupies.
        alphabet:      'aa' (uses seq_hash_{side}) or 'nt_cds' (uses dna_hash_{side}).
        top_k:         Number of top rows to keep.

    Returns:
        (top_k_df, n_pairs_total, n_clusters_used)
        top_k_df columns: rank, cluster_id, weight, pct, cum_pct.
    """
    if side not in ('a', 'b'):
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")
    if alphabet == 'aa':
        hash_col = f'seq_hash_{side}'
    elif alphabet == 'nt_cds':
        hash_col = f'dna_hash_{side}'
    else:
        raise ValueError(f"alphabet must be 'aa' or 'nt_cds', got {alphabet!r}")

    clusters = pair_universe[hash_col].map(cluster_map)
    n_unmapped = int(clusters.isna().sum())
    if n_unmapped > 0:
        # Should not happen if clusters cover the whole corpus; be loud so the
        # discrepancy is visible to the caller rather than silently dropped.
        print(f'WARNING: {n_unmapped} pairs unmapped at this (slot={side}, '
              f'alphabet={alphabet}); dropping from weight tally.')
        clusters = clusters.dropna()

    counts = clusters.value_counts()  # descending
    total = int(counts.sum())
    n_clusters_used = int(len(counts))
    if total == 0:
        return pd.DataFrame(columns=['rank', 'cluster_id', 'weight', 'pct', 'cum_pct']), 0, 0

    top = counts.head(top_k)
    df = pd.DataFrame({
        'rank': np.arange(1, len(top) + 1, dtype=int),
        'cluster_id': top.index.values,
        'weight': top.values.astype(int),
        'pct': (top.values / total * 100).round(4),
    })
    df['cum_pct'] = df['pct'].cumsum().round(4)
    return df, total, n_clusters_used


def plot_concentration(
    slice_df: pd.DataFrame,
    title: str,
    out_png: Path,
) -> None:
    """Two-panel cluster pair-weight concentration plot for one slice.

    Slice = one (schema_pair, slot_protein, alphabet) combination. The
    slice contains one (rank, pct, cum_pct) curve per threshold.

    Left panel:  rank vs pct (per-rank weight share, log-y).
    Right panel: rank vs cum_pct (cumulative weight share, linear-y).
    One line per threshold; colored via viridis (dark = strict / id100,
    light = loose / id090).
    """
    # sort id100 first so iteration index 0 -> darkest viridis end.
    thresholds = sorted(slice_df['threshold'].unique(), reverse=True)
    cmap = plt.get_cmap('viridis')
    n = max(len(thresholds), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for i, t in enumerate(thresholds):
        sub = slice_df[slice_df['threshold'] == t].sort_values('rank')
        if len(sub) == 0:
            continue
        color = cmap(i / max(n - 1, 1))
        axes[0].plot(sub['rank'], sub['pct'], color=color, linewidth=1.3,
                     alpha=0.85, label=t)
        axes[1].plot(sub['rank'], sub['cum_pct'], color=color, linewidth=1.3,
                     alpha=0.85, label=t)

    axes[0].set_xlabel('rank')
    axes[0].set_ylabel('weight share per rank (% of total pairs, log)')
    axes[0].set_yscale('log')
    axes[0].grid(True, which='both', linestyle=':', alpha=0.4)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc='upper right', fontsize=7, frameon=False, ncol=2,
                   title='threshold')

    axes[1].set_xlabel('rank')
    axes[1].set_ylabel('cumulative weight share (% of total pairs)')
    axes[1].set_ylim(0, 102)
    axes[1].grid(True, linestyle=':', alpha=0.5)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc='lower right', fontsize=7, frameon=False, ncol=2,
                   title='threshold')

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


# Schema-pair slugs the plotter currently emits PNGs for. Keep this small
# while the bipartite-splitter exploration is HA-NA-focused; switch to
# `[slug for _, slug, _, _ in _SCHEMA_PAIRS]` once we want all pairs.
_PLOT_SLUGS = ['ha_na']


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'),
                   help='Stage 1.5 cds_final.parquet — provides seq_hash and cds_dna_hash per (isolate, function).')
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'),
                   help='Root containing id{XXX}/{PROTEIN}_cluster.parquet for aa.')
    p.add_argument('--clusters_nt',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_nt'),
                   help='Root containing id{XXX}/{PROTEIN}_cluster.parquet for nt.')
    p.add_argument('--out_dir',
                   default=str(PROJ / 'results/flu/July_2025/runs/cluster_pair_weight'),
                   help='Output directory for per-pair long CSVs, combined rollup, and plots.')
    p.add_argument('--top_k', type=int, default=100,
                   help='Number of top-ranked clusters per slice (default 100).')
    p.add_argument('--thresholds', nargs='*', default=None,
                   help='Optional threshold filter, e.g. id100 id099 id095. '
                        'Default: t100..t90 (see _DEFAULT_THRESHOLDS).')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters_aa = Path(args.clusters_aa)
    clusters_nt = Path(args.clusters_nt)
    cds_final = Path(args.cds_final)

    thresholds = list(args.thresholds) if args.thresholds else list(_DEFAULT_THRESHOLDS)

    rollup_rows: list[dict] = []

    for schema_pair, slug, slot_a, slot_b in _SCHEMA_PAIRS:
        print(f'\nLoading pair universe for {schema_pair} ...')
        universe = load_pair_universe(cds_final, slot_a, slot_b)
        print(f'  {len(universe):,} unique canonical protein pairs')

        long_frames_for_pair: list[pd.DataFrame] = []
        for slot_protein, side in [(slot_a, 'a'), (slot_b, 'b')]:
            for alphabet, clusters_root in [('aa', clusters_aa), ('nt_cds', clusters_nt)]:
                for t in thresholds:
                    cluster_map = load_cluster_map(clusters_root, slot_protein, t)
                    if not cluster_map:
                        continue
                    top_df, total, n_clusters = compute_top_k_cluster_weights(
                        universe, cluster_map, side, alphabet, top_k=args.top_k,
                    )
                    if len(top_df) == 0:
                        continue
                    top_df.insert(0, 'threshold', t)
                    top_df.insert(0, 'slot_protein', slot_protein)
                    top_df.insert(0, 'alphabet', alphabet)
                    long_frames_for_pair.append(top_df)

                    pct = top_df['pct'].values
                    rollup_rows.append({
                        'schema_pair': schema_pair,
                        'alphabet': alphabet,
                        'slot_protein': slot_protein,
                        'threshold': t,
                        'n_pairs_total': total,
                        'n_clusters_used': n_clusters,
                        'top1_pct': round(float(pct[0]), 4),
                        'top5_pct': round(float(pct[:5].sum()), 4),
                        'top10_pct': round(float(pct[:10].sum()), 4),
                        'top100_pct': round(float(pct[:100].sum()), 4),
                    })

        if not long_frames_for_pair:
            print(f'  no slices produced for {schema_pair}; skipping CSV.')
            continue
        long_df_for_pair = pd.concat(long_frames_for_pair, ignore_index=True)
        long_csv = out_dir / f'top{args.top_k}_{slug}.csv'
        long_df_for_pair.to_csv(long_csv, index=False)
        print(f'wrote {long_csv} ({len(long_df_for_pair):,} rows)')

        if slug in _PLOT_SLUGS:
            plot_dir = out_dir / 'plots'
            for slot_protein in (slot_a, slot_b):
                for alphabet in ('aa', 'nt_cds'):
                    sub = long_df_for_pair[
                        (long_df_for_pair['slot_protein'] == slot_protein)
                        & (long_df_for_pair['alphabet'] == alphabet)
                    ]
                    if len(sub) == 0:
                        continue
                    out_png = plot_dir / f'concentration_{slug}_{slot_protein}_{alphabet}.png'
                    plot_concentration(
                        sub,
                        title=(f'{schema_pair} cluster pair-weight concentration '
                               f'— slot={slot_protein}, alphabet={alphabet}'),
                        out_png=out_png,
                    )
                    print(f'wrote {out_png}')

    rollup_df = pd.DataFrame(rollup_rows)
    rollup_csv = out_dir / 'rollup.csv'
    rollup_df.to_csv(rollup_csv, index=False)
    print(f'\nwrote {rollup_csv} ({len(rollup_df):,} rows)')

    print('\nDone.')


if __name__ == '__main__':
    main()
