"""Compare restricted and unrestricted distances from negatives to observed positives.

The existing ambiguity analysis holds one slot fixed and changes the other. This script also
searches all observed positive pairs using Hamming distance over the concatenated HA and NA
nucleotide sequences. It does not retrain the model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.plot_negative_pair_ambiguity import (  # noqa: E402
    DEFAULT_BIN_EDGES,
    bin_labels,
    check_bin_edges,
    positive_universe,
)
from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402
from src.utils.site_utils import load_site_cache  # noqa: E402


def hamming_count_matrix(cache, query_hashes: list[str],
                         reference_hashes: list[str]) -> np.ndarray:
    """Return integer Hamming counts between two sets of cached nucleotide sequences."""
    query = cache.codes[[cache.hash_to_row[h] for h in query_hashes]]
    reference = cache.codes[[cache.hash_to_row[h] for h in reference_hashes]]
    fractions = cdist(query, reference, metric='hamming')
    return np.rint(fractions * cache.codes.shape[1]).astype(np.uint16)


def load_negatives(run_dirs: list[Path]) -> pd.DataFrame:
    """Read negative test rows and their saved predictions."""
    columns = [
        'pair_key', 'cds_dna_hash_a', 'cds_dna_hash_b',
        'label', 'pred_prob', 'pred_label',
    ]
    frames = []
    for run_dir in run_dirs:
        path = run_dir / 'test_predicted.csv'
        if not path.exists():
            raise FileNotFoundError(f'missing {path}')
        frame = pd.read_csv(path, usecols=columns, low_memory=False)
        for column in ('label', 'pred_label'):
            if not frame[column].isin([0, 1]).all():
                raise ValueError(f'{path}: {column} must contain only 0 and 1')
        frame = frame[frame['label'] == 0].copy()
        frame['run'] = run_dir.name
        frame['is_false_positive'] = frame['pred_label'].astype(int) == 1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def measure_distances(negatives: pd.DataFrame, cache_a, cache_b,
                      partners_of_a: dict, partners_of_b: dict) -> pd.DataFrame:
    """Add conditional single-slot and unrestricted whole-pair Hamming distances."""
    positive_pairs = sorted({
        (hash_a, hash_b)
        for hash_a, partners in partners_of_a.items()
        for hash_b in partners
    })
    reference_a = sorted(partners_of_a)
    reference_b = sorted(partners_of_b)
    query_a = sorted(negatives['cds_dna_hash_a'].unique())
    query_b = sorted(negatives['cds_dna_hash_b'].unique())

    missing_a = set(query_a) - set(reference_a)
    missing_b = set(query_b) - set(reference_b)
    if missing_a or missing_b:
        raise ValueError(
            f'negative sequences missing from the positive universe: '
            f'{len(missing_a)} slot A, {len(missing_b)} slot B')

    print(f'Computing slot-A Hamming matrix: {len(query_a):,} x {len(reference_a):,}')
    hamming_a = hamming_count_matrix(cache_a, query_a, reference_a)
    print(f'Computing slot-B Hamming matrix: {len(query_b):,} x {len(reference_b):,}')
    hamming_b = hamming_count_matrix(cache_b, query_b, reference_b)

    query_a_index = {h: i for i, h in enumerate(query_a)}
    query_b_index = {h: i for i, h in enumerate(query_b)}
    reference_a_index = {h: i for i, h in enumerate(reference_a)}
    reference_b_index = {h: i for i, h in enumerate(reference_b)}
    positive_a_index = np.array([reference_a_index[a] for a, _ in positive_pairs])
    positive_b_index = np.array([reference_b_index[b] for _, b in positive_pairs])

    distance_a = []
    distance_b = []
    distance_global = []
    for row in negatives.itertuples():
        qa = query_a_index[row.cds_dna_hash_a]
        qb = query_b_index[row.cds_dna_hash_b]

        observed_a = [reference_a_index[h] for h in partners_of_b[row.cds_dna_hash_b]]
        observed_b = [reference_b_index[h] for h in partners_of_a[row.cds_dna_hash_a]]
        distance_a.append(int(hamming_a[qa, observed_a].min()))
        distance_b.append(int(hamming_b[qb, observed_b].min()))

        pair_distances = (
            hamming_a[qa, positive_a_index].astype(np.uint32)
            + hamming_b[qb, positive_b_index].astype(np.uint32)
        )
        distance_global.append(int(pair_distances.min()))

    result = negatives.copy()
    result['distance_slot_a'] = distance_a
    result['distance_slot_b'] = distance_b
    result['distance_min'] = np.minimum(distance_a, distance_b)
    result['distance_global'] = distance_global
    result['distance_gap'] = result['distance_min'] - result['distance_global']

    if (result['distance_global'] > result['distance_min']).any():
        raise AssertionError('unrestricted distance exceeds restricted distance')
    if (result['distance_global'] == 0).any():
        raise AssertionError('a retained negative matches an observed positive pair')
    return result


def summarize_by_distance(negatives: pd.DataFrame, edges: tuple[int, ...]) -> pd.DataFrame:
    """Return disjoint-bin FPRs and cumulative enrichment for both distance definitions."""
    rows = []
    n_false_total = int(negatives['is_false_positive'].sum())
    labels = bin_labels(edges)
    for measure in ('distance_min', 'distance_global'):
        distances = negatives[measure].to_numpy()
        indices = np.searchsorted(np.asarray(edges), distances, side='left')
        for position, label in enumerate(labels):
            selected = indices == position
            n_negative = int(selected.sum())
            n_false = int(negatives.loc[selected, 'is_false_positive'].sum())
            rows.append({
                'measure': measure,
                'summary': 'bin',
                'distance': label,
                'n_negatives': n_negative,
                'n_false_positives': n_false,
                'false_positive_rate': n_false / n_negative if n_negative else np.nan,
                'share_of_negatives': np.nan,
                'share_of_false_positives': np.nan,
                'enrichment': np.nan,
            })
        for edge in edges:
            selected = distances <= edge
            n_negative = int(selected.sum())
            n_false = int(negatives.loc[selected, 'is_false_positive'].sum())
            share_negative = n_negative / len(negatives)
            share_false = n_false / n_false_total
            rows.append({
                'measure': measure,
                'summary': 'within',
                'distance': str(edge),
                'n_negatives': n_negative,
                'n_false_positives': n_false,
                'false_positive_rate': n_false / n_negative if n_negative else np.nan,
                'share_of_negatives': share_negative,
                'share_of_false_positives': share_false,
                'enrichment': share_false / share_negative if share_negative else np.nan,
            })
    return pd.DataFrame(rows)


def slot_heatmap_table(negatives: pd.DataFrame, edges: tuple[int, ...]) -> pd.DataFrame:
    """Return counts and FPR for every pair of slot-distance bins."""
    labels = bin_labels(edges)
    index_a = np.searchsorted(
        np.asarray(edges), negatives['distance_slot_a'].to_numpy(), side='left')
    index_b = np.searchsorted(
        np.asarray(edges), negatives['distance_slot_b'].to_numpy(), side='left')
    rows = []
    for ia, label_a in enumerate(labels):
        for ib, label_b in enumerate(labels):
            selected = (index_a == ia) & (index_b == ib)
            n_negative = int(selected.sum())
            n_false = int(negatives.loc[selected, 'is_false_positive'].sum())
            rows.append({
                'slot_a_bin': label_a,
                'slot_b_bin': label_b,
                'n_negatives': n_negative,
                'n_false_positives': n_false,
                'false_positive_rate': n_false / n_negative if n_negative else np.nan,
            })
    return pd.DataFrame(rows)


def plot_slot_heatmap(table: pd.DataFrame, labels: list[str], protein_a: str,
                      protein_b: str, out_path: Path, dpi: int) -> Path:
    """Plot FPR jointly by the two conditional slot distances."""
    setup_plot_style()
    n_bins = len(labels)
    rates = table['false_positive_rate'].to_numpy().reshape(n_bins, n_bins)
    counts = table['n_negatives'].to_numpy().reshape(n_bins, n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(rates, vmin=0, vmax=1, cmap='viridis')
    for row in range(n_bins):
        for column in range(n_bins):
            if counts[row, column]:
                ax.text(
                    column, row, f'{rates[row, column]:.2f}\nn={counts[row, column]:,}',
                    ha='center', va='center',
                    color='white' if rates[row, column] < 0.65 else 'black',
                    fontsize=8,
                )
    ax.set_xticks(range(n_bins), labels)
    ax.set_yticks(range(n_bins), labels)
    ax.set_xlabel(f'{protein_b} Hamming distance')
    ax.set_ylabel(f'{protein_a} Hamming distance')
    ax.set_title('False-positive rate by both slot distances')
    fig.colorbar(image, ax=ax, label='false-positive rate')
    fig.tight_layout()
    return savefig(out_path, dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run_dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--site_dir', type=Path, default=PROJ / 'data/embeddings/flu/July_2025')
    parser.add_argument('--protein_a', default='HA')
    parser.add_argument('--protein_b', default='NA')
    parser.add_argument('--bin_edges', type=int, nargs='+', default=list(DEFAULT_BIN_EDGES))
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    edges = tuple(args.bin_edges)
    check_bin_edges(edges)
    cache_a = load_site_cache(args.site_dir, 'nt', args.protein_a)
    cache_b = load_site_cache(args.site_dir, 'nt', args.protein_b)
    partners_of_a, partners_of_b = positive_universe(args.dataset_dir, cache_a, cache_b)

    negatives = load_negatives(args.run_dirs)
    measured = measure_distances(
        negatives, cache_a, cache_b, partners_of_a, partners_of_b)
    summary = summarize_by_distance(measured, edges)
    heatmap = slot_heatmap_table(measured, edges)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    measured_path = args.out_dir / 'negative_pair_distance_comparison.csv'
    summary_path = args.out_dir / 'negative_pair_distance_comparison_bins.csv'
    heatmap_path = args.out_dir / 'negative_pair_slot_distance_heatmap.csv'
    figure_path = args.out_dir / 'negative_pair_slot_distance_heatmap.png'
    measured.to_csv(measured_path, index=False)
    summary.to_csv(summary_path, index=False)
    heatmap.to_csv(heatmap_path, index=False)
    plot_slot_heatmap(
        heatmap, bin_labels(edges), args.protein_a, args.protein_b, figure_path, args.dpi)

    changed = measured['distance_gap'] > 0
    print(f'Negatives: {len(measured):,}; false positives: '
          f'{int(measured["is_false_positive"].sum()):,}')
    print(f'Unrestricted distance is smaller for {int(changed.sum()):,} '
          f'({changed.mean():.1%})')
    print(f'Median gap: all={measured["distance_gap"].median():.1f} nt; '
          f'changed={measured.loc[changed, "distance_gap"].median():.1f} nt')
    print(summary.to_string(index=False))
    print(f'Wrote {measured_path}')
    print(f'Wrote {summary_path}')
    print(f'Wrote {heatmap_path}')
    print(f'Wrote {figure_path}')


if __name__ == '__main__':
    main()
