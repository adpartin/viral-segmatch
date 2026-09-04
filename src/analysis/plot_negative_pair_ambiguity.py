"""Relate false-positive rate to sequence distance from observed positive pairs.

For each negative test pair, measure how far each sequence is from the observed partners of the
sequence in the other slot. Observed partners come from `cooccurring_sequence_pairs.csv`, not only
from the test split. The output contains both slot distances and their minimum, along with the saved
model prediction. This is a post-hoc analysis; it does not retrain the model.

Reading the result: the distances and bin sizes describe the negative sampler and the population,
while the false-positive rates and the enrichment also depend on the fitted model and its decision
threshold. See "Near-duplicate negative" in `docs/methods/glossary.md` for what this does and does
not establish.

Outputs (to `--out_dir`, by default derived from the first run dir):
    negative_pair_ambiguity_{unit}_min.png       false-positive rate by distance bin, plus the
                                                 cumulative share for false positives vs all
                                                 negatives
    negative_pair_ambiguity_{unit}_min.csv       one row per negative test pair: run, pair_key,
                                                 distance_slot_a, distance_slot_b, distance_min,
                                                 nearer_slot, pred_prob, pred_label,
                                                 is_false_positive
    negative_pair_ambiguity_{unit}_min_bins.csv  run, bin_label, n_negatives, n_false_positives,
                                                 false_positive_rate

CLI:
    python -m src.analysis.plot_negative_pair_ambiguity \\
        --run_dirs models/flu/July_2025/runs/lgbm_ha_na_h3n2_2024_random_cv4_site_nt_fold{0,1,2,3} \\
        --config_bundle flu_ha_na_h3n2_2024_random_cv4_site_nt --unit nt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.config_hydra import get_function_short_name_map, get_virus_config_hydra  # noqa: E402
from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402
from src.utils.site_utils import load_site_cache  # noqa: E402

# Sampled to match the other per-site figures, so the whole set reads as one series.
TRACE_COLOR = '#4C7CAB'
ACCENT_COLOR = '#CF8793'
MARKER_EDGE = '#222222'

# Upper edges of the distance bins, in sites. A final open-ended bin is appended past the last one.
DEFAULT_BIN_EDGES = (2, 5, 10, 20)
DISTANCE_COLUMN = 'distance_min'


def bin_labels(edges: tuple) -> list:
    """Return labels for the distance bins, including the open-ended final bin."""
    labels, low = [], 0
    for edge in edges:
        labels.append(f'{low}-{edge}' if low != edge else f'{edge}')
        low = edge + 1
    labels.append(f'>{edges[-1]}')
    return labels


def positive_universe(dataset_dir: Path, cache_a, cache_b) -> tuple:
    """Index every observed co-occurrence by its slot-A and slot-B sequence hashes."""
    cooccur_path = dataset_dir / 'cooccurring_sequence_pairs.csv'
    if not cooccur_path.exists():
        raise FileNotFoundError(
            f"missing {cooccur_path}. It is written by src/datasets/dataset_segment_pairs.py; "
            f"pass --dataset_dir if the run directory does not record its dataset.")
    cooccur = pd.read_csv(cooccur_path, keep_default_na=False, na_values=[''])

    partners_of_a: dict = {}
    partners_of_b: dict = {}
    n_unassigned = 0
    for pair_key in cooccur['pair_key']:
        left, _, right = str(pair_key).partition('__')
        if left in cache_a.hash_to_row and right in cache_b.hash_to_row:
            hash_a, hash_b = left, right
        elif right in cache_a.hash_to_row and left in cache_b.hash_to_row:
            hash_a, hash_b = right, left
        else:
            n_unassigned += 1
            continue
        partners_of_a.setdefault(hash_a, []).append(hash_b)
        partners_of_b.setdefault(hash_b, []).append(hash_a)

    if not partners_of_a:
        raise ValueError(
            f"{cooccur_path}: no co-occurrence row maps onto the {cache_a.protein} and "
            f"{cache_b.protein} caches. The caches and the dataset describe different proteins.")
    if n_unassigned:
        print(f"WARNING: {n_unassigned:,} of {len(cooccur):,} co-occurrence rows have a sequence "
              f"outside the per-site caches (off-length or incomplete CDS); skipped.")
    return partners_of_a, partners_of_b


def site_distance(cache, hash_left: str, hash_right: str) -> int:
    """Count differing sites between two aligned sequences in the same cache."""
    codes_left = cache.codes[cache.hash_to_row[hash_left]]
    codes_right = cache.codes[cache.hash_to_row[hash_right]]
    return int((codes_left != codes_right).sum())


def slot_distances(hash_a: str, hash_b: str, partners_of_a: dict, partners_of_b: dict,
                   cache_a, cache_b) -> tuple:
    """Return each slot's distance to its nearest observed partner sequence."""
    distance_b = None
    if hash_a in partners_of_a:
        distance_b = min(site_distance(cache_b, hash_b, true) for true in partners_of_a[hash_a])
    distance_a = None
    if hash_b in partners_of_b:
        distance_a = min(site_distance(cache_a, hash_a, true) for true in partners_of_b[hash_b])
    return distance_a, distance_b


def check_bin_edges(edges: tuple) -> None:
    """Require non-negative, strictly increasing bin edges."""
    if not edges:
        raise ValueError('--bin_edges needs at least one upper edge.')
    if any(edge < 0 for edge in edges):
        raise ValueError(f'--bin_edges must be non-negative; got {list(edges)}.')
    if any(later <= earlier for earlier, later in zip(edges, edges[1:])):
        raise ValueError(f'--bin_edges must be strictly increasing; got {list(edges)}.')


def analyze_run(run_dir: Path, cache_a, cache_b, run_label: str, partners_of_a: dict,
                partners_of_b: dict) -> pd.DataFrame:
    """Measure both slot distances for every negative test pair in one run."""
    predictions_path = run_dir / 'test_predicted.csv'
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"missing {predictions_path}. Train the run first with "
            f"`python src/models/train_pair_baselines.py --baseline lgbm ...`.")
    # keep_default_na: pair tables carry protein names, and 'NA' (Neuraminidase) is a real value.
    predictions = pd.read_csv(predictions_path, keep_default_na=False, na_values=[''],
                              low_memory=False)
    labels = predictions['label'].astype(float)
    predicted = predictions['pred_label'].astype(float)
    for column, values in (('label', labels), ('pred_label', predicted)):
        valid = values.isin([0.0, 1.0])
        if not valid.all():
            unexpected = values[~valid].unique()[:5].tolist()
            raise ValueError(
                f"{predictions_path}: {column} must be 0 or 1; found {unexpected}.")
    labels = labels.astype(int)
    predicted = predicted.astype(int)

    negatives = predictions[labels == 0].copy()
    negatives['pred_label'] = predicted[labels == 0]
    distances = []
    for hash_a, hash_b in zip(negatives['cds_dna_hash_a'], negatives['cds_dna_hash_b']):
        distances.append(slot_distances(hash_a, hash_b, partners_of_a, partners_of_b,
                                        cache_a, cache_b))
    negatives[['distance_slot_a', 'distance_slot_b']] = distances
    negatives[DISTANCE_COLUMN] = negatives[['distance_slot_a', 'distance_slot_b']].min(axis=1)
    negatives['nearer_slot'] = np.where(
        pd.isna(negatives['distance_slot_a']), 'b',
        np.where(pd.isna(negatives['distance_slot_b']), 'a',
                 np.where(negatives['distance_slot_a'] <= negatives['distance_slot_b'], 'a', 'b')))
    negatives['is_false_positive'] = negatives['pred_label'] == 1
    negatives['run'] = run_label

    n_unscored = int(negatives[DISTANCE_COLUMN].isna().sum())
    if n_unscored == len(negatives):
        raise ValueError(
            f"{run_dir}: no negative pair could be matched to an observed co-occurrence. "
            f"The per-site caches and this run's pair tables do not describe the same population.")
    if n_unscored > 0:
        print(f"WARNING: {run_label}: {n_unscored:,} of {len(negatives):,} negatives have neither "
              f"sequence in an observed co-occurrence; dropped.")
    scored = negatives.dropna(subset=[DISTANCE_COLUMN]).copy()

    columns = ['run', 'pair_key', 'distance_slot_a', 'distance_slot_b',
               DISTANCE_COLUMN, 'nearer_slot',
               'pred_prob', 'pred_label', 'is_false_positive']
    return scored[columns]


def bin_false_positive_rate(negatives: pd.DataFrame, edges: tuple,
                            run_label: str) -> pd.DataFrame:
    """Count negatives and false positives in each distance bin."""
    labels = bin_labels(edges)
    indices = np.searchsorted(
        np.asarray(edges), negatives[DISTANCE_COLUMN].to_numpy(), side='left')
    rows = []
    for position, label in enumerate(labels):
        in_bin = negatives[indices == position]
        n_negatives = len(in_bin)
        n_false_positives = int(in_bin['is_false_positive'].sum())
        rate = n_false_positives / n_negatives if n_negatives else float('nan')
        rows.append({'run': run_label, 'bin_label': label, 'n_negatives': n_negatives,
                     'n_false_positives': n_false_positives, 'false_positive_rate': rate})
    return pd.DataFrame(rows)


def plot_ambiguity(negatives: pd.DataFrame, bins: pd.DataFrame, unit: str, out_path: Path,
                   dpi: int) -> Path:
    """Plot false-positive rate and cumulative distance distributions."""
    setup_plot_style()
    fig, (ax_rate, ax_dist) = plt.subplots(1, 2, figsize=(12, 4.5))

    positions = np.arange(len(bins))
    ax_rate.bar(positions, bins['false_positive_rate'], color=TRACE_COLOR,
                edgecolor=MARKER_EDGE, linewidth=0.6)
    for position, row in zip(positions, bins.itertuples()):
        if row.n_negatives:
            ax_rate.text(position, row.false_positive_rate + 0.02, f'n={row.n_negatives:,}',
                         ha='center', va='bottom', fontsize=8)
    ax_rate.set_xticks(positions)
    ax_rate.set_xticklabels(bins['bin_label'])
    ax_rate.set_xlabel(f'minimum partner distance ({unit} sites)')
    ax_rate.set_ylabel('false-positive rate')
    ax_rate.set_ylim(0, 1.12)
    ax_rate.set_title('False-positive rate increases at short distances')

    # Cumulative rather than a density: the distance distribution has a long sparse tail (distant
    # lineages) that squashes the bulk, and the question here is what share of the errors sits
    # inside a given distance, which a cumulative curve answers directly.
    all_distances = negatives[DISTANCE_COLUMN].to_numpy()
    false_distances = negatives.loc[negatives['is_false_positive'], DISTANCE_COLUMN].to_numpy()
    upper = max(int(np.percentile(all_distances, 90)), 10)
    thresholds = np.arange(0, upper + 1)
    false_share = [(false_distances <= t).mean() for t in thresholds]
    negative_share = [(all_distances <= t).mean() for t in thresholds]

    ax_dist.plot(thresholds, false_share, color=ACCENT_COLOR, linewidth=2,
                 label=f'false positives (n={len(false_distances):,})')
    ax_dist.plot(thresholds, negative_share, color=TRACE_COLOR, linewidth=2, linestyle='--',
                 label=f'all negatives (n={len(all_distances):,})')
    ax_dist.set_xlabel(f'minimum partner distance threshold ({unit} sites)')
    ax_dist.set_ylabel('cumulative share at or below threshold')
    ax_dist.set_ylim(0, 1.02)
    ax_dist.legend(frameon=False, loc='lower right')
    ax_dist.set_title('False positives concentrate at short distances')
    n_beyond = int((all_distances > upper).sum())
    n_false_beyond = int((false_distances > upper).sum())
    ax_dist.text(0.02, 0.96, f'beyond {upper} sites: {n_beyond:,} negatives, '
                             f'{n_false_beyond:,} false positives',
                 transform=ax_dist.transAxes, ha='left', va='top', fontsize=8, color='#444444')

    fig.text(0.995, 0.002, f'src/analysis/{Path(__file__).name}', ha='right', va='bottom',
             fontsize=7, color='#666666')
    fig.tight_layout()
    return savefig(out_path, dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run_dirs', type=Path, nargs='+', required=True,
                        help='model run directories, each holding test_predicted.csv')
    parser.add_argument('--config_bundle', required=True,
                        help='bundle the runs were trained with; supplies the protein short names')
    parser.add_argument('--unit', default='nt', choices=['nt', 'codon', 'aa'],
                        help='per-site cache the distance is measured on')
    parser.add_argument('--site_dir', type=Path, default=PROJ / 'data/embeddings/flu/July_2025')
    parser.add_argument('--bin_edges', type=int, nargs='+', default=list(DEFAULT_BIN_EDGES),
                        help='ascending upper edges of the distance bins, in sites')
    parser.add_argument('--dataset_dir', type=Path, default=None,
                        help='dataset run dir holding cooccurring_sequence_pairs.csv; read from '
                             'each run\'s training_info.json when absent')
    parser.add_argument('--out_dir', type=Path, default=None)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    check_bin_edges(tuple(args.bin_edges))
    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    function_to_short = get_function_short_name_map(config)
    edges = tuple(args.bin_edges)

    # Every run must describe the same schema pair, otherwise one pair of caches cannot serve all
    # of them and the pooled table would silently mix two different populations.
    schema_pairs = {}
    for run_dir in args.run_dirs:
        head = pd.read_csv(run_dir / 'test_predicted.csv', keep_default_na=False,
                           na_values=[''], low_memory=False, nrows=1)
        schema_pairs[run_dir.name] = (head['func_a'].iloc[0], head['func_b'].iloc[0])
    distinct = set(schema_pairs.values())
    if len(distinct) > 1:
        raise ValueError(
            f"--run_dirs mix schema pairs, so one set of caches cannot serve them: "
            f"{ {name: pair for name, pair in schema_pairs.items()} }.")
    function_a, function_b = distinct.pop()
    protein_a = function_to_short.get(function_a, function_a)
    protein_b = function_to_short.get(function_b, function_b)
    cache_a = load_site_cache(args.site_dir, args.unit, protein_a)
    cache_b = load_site_cache(args.site_dir, args.unit, protein_b)
    print(f"Slots: a={protein_a} ({cache_a.codes.shape[1]:,} sites), "
          f"b={protein_b} ({cache_b.codes.shape[1]:,} sites), unit={args.unit}")

    # The positive universe is every observed co-occurrence, which is what the negative sampler
    # blocked against. Each run records the dataset it was trained on, so the runs must agree.
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        recorded = []
        for run_dir in args.run_dirs:
            info_path = run_dir / 'training_info.json'
            if not info_path.exists():
                raise FileNotFoundError(
                    f"missing {info_path}, so the dataset cannot be located. Pass --dataset_dir.")
            with open(info_path) as f:
                recorded.append(json.load(f)['dataset_dir'])
        # A CV run records its own fold_k directory, and each fold carries a copy of the
        # co-occurrence file because the CV generator builds the set once and reuses it. Compare
        # the parents so folds of one dataset are accepted, then read from a recorded directory.
        runs_of = {str(Path(d).parent) if Path(d).name.startswith('fold_') else str(d)
                   for d in recorded}
        if len(runs_of) > 1:
            raise ValueError(f"--run_dirs come from different datasets: {sorted(runs_of)}. "
                             f"Pass --dataset_dir to state which positive universe to use.")
        dataset_dir = PROJ / recorded[0]
    partners_of_a, partners_of_b = positive_universe(dataset_dir, cache_a, cache_b)
    print(f"Positive universe: {dataset_dir}, {len(partners_of_a):,} slot-a and "
          f"{len(partners_of_b):,} slot-b sequences with a known partner")

    if args.out_dir is None:
        args.out_dir = args.run_dirs[0] / 'negative_pair_ambiguity'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_run, bin_tables = [], []
    for run_dir in args.run_dirs:
        scored = analyze_run(run_dir, cache_a, cache_b, run_dir.name, partners_of_a, partners_of_b)
        per_run.append(scored)
        bin_tables.append(bin_false_positive_rate(scored, edges, run_dir.name))
        n_false = int(scored['is_false_positive'].sum())
        print(f"  {run_dir.name}: {len(scored):,} negatives, {n_false:,} false positives "
              f"({n_false/len(scored):.1%})")

    negatives = pd.concat(per_run, ignore_index=True)
    pooled = bin_false_positive_rate(negatives, edges, '(all runs)')
    bins = pd.concat([pooled] + bin_tables, ignore_index=True)

    stem = f'negative_pair_ambiguity_{args.unit}_min'
    negatives_path = args.out_dir / f'{stem}.csv'
    bins_path = args.out_dir / f'{stem}_bins.csv'
    negatives.to_csv(negatives_path, index=False)
    bins.to_csv(bins_path, index=False)

    figure_path = plot_ambiguity(negatives, pooled, args.unit, args.out_dir / f'{stem}.png',
                                 args.dpi)

    print(f"\nPooled over {len(args.run_dirs)} run(s): {len(negatives):,} negatives, "
          f"{int(negatives['is_false_positive'].sum()):,} false positives")
    print(pooled[['bin_label', 'n_negatives', 'n_false_positives',
                  'false_positive_rate']].to_string(index=False))

    # Enrichment, not the raw share of false positives. A wide bin can hold most of the false
    # positives simply because it holds most of the negatives, so the share alone overstates how
    # concentrated the errors are.
    n_false_total = int(negatives['is_false_positive'].sum())
    print(f"\n{'within':>8s} {'negatives':>19s} {'false positives':>19s} {'enrichment':>11s}")
    for edge in edges:
        near = negatives[negatives[DISTANCE_COLUMN] <= edge]
        share_of_negatives = len(near) / len(negatives)
        n_false_near = int(near['is_false_positive'].sum())
        if not n_false_total or not share_of_negatives:
            continue
        share_of_false = n_false_near / n_false_total
        enrichment = share_of_false / share_of_negatives
        print(f"{edge:6,d} {'sites':<2s} {len(near):9,d} ({share_of_negatives:6.1%}) "
              f"{n_false_near:9,d} ({share_of_false:6.1%}) {enrichment:10.2f}x")

    print(f"\nWrote {negatives_path}")
    print(f"Wrote {bins_path}")
    print(f"Wrote {figure_path}")
    print('Done.')


if __name__ == '__main__':
    main()
