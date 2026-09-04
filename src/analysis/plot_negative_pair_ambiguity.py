"""Where do a model's false positives sit relative to the nearest real positive pair?

Post-hoc error analysis on an already-trained run. Nothing is retrained and no feature is
corrupted: the saved `test_predicted.csv` is read back and each NEGATIVE row is scored by how far
its assigned partner sits from its nearest observed partner. A sequence can co-occur with several
partners, so there is no single correct one to compare against.

The mechanism it tests. A negative pair is built by giving a slot-A sequence the slot-B sequence of
a different isolate. The sampler rejects exact co-occurrences, so the pair is never a relabelled
positive. It does not reject NEAR co-occurrences. When the substituted partner differs from the
real partner by a handful of nucleotides, the resulting pair resembles a true positive, and the
model is measurably more likely to call it positive.

What the negative label actually means. It means "recombined and not observed co-occurring", not
"biologically incompatible". Sequence proximity makes compatibility plausible, but nothing here
supplies biological ground truth, so a near-duplicate negative should not be treated as a
mislabelled row.

These rows are hard, not unlabelable. On H3N2 2024 HA-NA the per-site `nt` model still classifies
31 of the 176 closest negatives (0-2 nt) correctly, and k-mer classifies 29 of them. No negative
sits at distance 0, so no identical feature vector ever carries contradictory labels. Claiming
these rows cannot be separated goes beyond what this measurement supports.

What this does NOT explain. It does not establish why precision reads lower than recall. Only
negative rows are analysed here, so the false-positive and false-negative mechanisms are never
compared. That gap is also threshold-dependent: on the pooled random-CV folds the errors balance
near a 0.5 threshold shifted to about 0.70 (312 false positives against 310 false negatives), which
a concentration measured only over negatives cannot account for.

The measure. For a negative pair `(a, b)` each slot gets its own distance. `distance_slot_b` is the
fewest differing sites between `b` and a true partner of `a`, and `distance_slot_a` is the mirror.
Both are written out. The summary used for binning is chosen with `--distance_summary`, defaults
to the minimum, and is named after itself: `distance_min`, `distance_mean` or `distance_max`. The
summary also lands in every output filename, so a `mean` run cannot overwrite a `min` run or be
mistaken for one. Only `min` is a distance to the NEAREST positive.

Taking the minimum answers one specific question, "how close is this pair to a positive after
changing whichever single slot needs less change", and it is not the only defensible summary. It is
slot-sensitive, since the shorter NA supplies it far more often than HA, so `mean` and `max` are
offered as alternatives that weight the slots evenly. Distance is measured in sites of `--unit`, on the per-site cache, which is aligned by
construction and so makes a position-by-position comparison meaningful. Every site counts equally,
which is not how the model weights them.

The positive universe is every observed co-occurrence, read from the dataset's
`cooccurring_sequence_pairs.csv`. That is the same set the negative sampler blocks against, so it
is what "nearest true pair" should be measured against. Using only the run's own test positives
would miss real pairs that live in the other splits and would understate closeness.

Read concentration as enrichment, not as a raw share. A bin holding most of the false positives may
simply be holding most of the negatives. On H3N2 2024 HA-NA the enrichment is 5.95x within 2 nt,
2.42x within 5 nt and only 1.25x within 10 nt, so the near bins are what carry the signal.

Which half of the output is which. The distance distribution belongs to the sampler and the
population, and is identical for every model scored on the same negatives. The false-positive rates
and the enrichment belong to one fitted model at one decision threshold, and change if either
changes. Do not read the second half as a property of the data.

Scope. Distance is nucleotide (or codon / amino-acid) identity, which stands in for how closely two
sequences are related. It is not a phylogenetic assignment, so "near-duplicate" here means
sequence-similar, not confirmed same-clade.

Outputs (to `--out_dir`, by default derived from the first run dir; `{how}` is the
`--distance_summary` value):
    negative_pair_ambiguity_{unit}_{how}.png       false-positive rate by distance bin, plus the
                                                   cumulative share for false positives vs all
                                                   negatives
    negative_pair_ambiguity_{unit}_{how}.csv       one row per negative test pair: run, pair_key,
                                                   distance_slot_a, distance_slot_b,
                                                   distance_{how}, nearer_slot, pred_prob,
                                                   pred_label, is_false_positive
    negative_pair_ambiguity_{unit}_{how}_bins.csv  run, bin_label, n_negatives,
                                                   n_false_positives, false_positive_rate

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


def distance_column(how: str) -> str:
    """Name of the summarized-distance column for one summary.

    The summary is part of the name so that a `mean` or `max` run cannot be mistaken for, or
    silently overwrite, a `min` run. Only `min` is a distance to the NEAREST positive; the other
    two are not, so none of them may share one generic name.

    Args:
      how: 'min', 'mean' or 'max'.

    Returns:
      The column name, e.g. `distance_min`.
    """
    return f'distance_{how}'


def bin_labels(edges: tuple) -> list:
    """Build the bin label strings for a set of upper edges.

    Args:
      edges: ascending upper edges, in sites.

    Returns:
      One label per bin, including the trailing open-ended bin.
    """
    labels, low = [], 0
    for edge in edges:
        labels.append(f'{low}-{edge}' if low != edge else f'{edge}')
        low = edge + 1
    labels.append(f'>{edges[-1]}')
    return labels


def assign_bin(distances: np.ndarray, edges: tuple) -> np.ndarray:
    """Map each distance to its bin index.

    Args:
      distances: one distance per negative pair, in sites.
      edges: ascending upper edges, in sites.

    Returns:
      Bin index per input, where `len(edges)` is the trailing open-ended bin.
    """
    index = np.searchsorted(np.asarray(edges), distances, side='left')
    return index


def positive_universe(dataset_dir: Path, cache_a, cache_b) -> tuple:
    """Read every observed co-occurrence and index it by each slot's hash.

    Reads `cooccurring_sequence_pairs.csv`, which the dataset builder writes with one row per
    sequence pair seen together in at least one isolate. That file is the same set the negative
    sampler blocks against, so it is the right universe to measure "nearest true pair" against.
    Building the map from one split's positives instead would miss real pairs that live in the
    other splits.

    Its `pair_key` is the two hashes sorted lexicographically, so which side is which is recovered
    by testing membership in each slot's cache. The two proteins differ in length and so never
    share a cache row.

    Args:
      dataset_dir: dataset run directory holding `cooccurring_sequence_pairs.csv`.
      cache_a: per-site cache for the slot-A protein.
      cache_b: per-site cache for the slot-B protein.

    Returns:
      `(partners_of_a, partners_of_b)`, each `{hash: [partner hash, ...]}`.

    Raises:
      FileNotFoundError: the co-occurrence file is absent.
      ValueError: no row could be assigned to the two slots, which means the caches and the
          dataset describe different proteins.
    """
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
    """Count the sites at which two cached sequences differ.

    Both sequences live in the same per-site cache, so they are the same length and already
    aligned position-by-position.

    Args:
      cache: the `SiteCache` holding both sequences.
      hash_left: `cds_dna_hash` of one sequence.
      hash_right: `cds_dna_hash` of the other.

    Returns:
      Number of differing sites.
    """
    codes_left = cache.codes[cache.hash_to_row[hash_left]]
    codes_right = cache.codes[cache.hash_to_row[hash_right]]
    n_differing = int((codes_left != codes_right).sum())
    return n_differing


def slot_distances(hash_a: str, hash_b: str, partners_of_a: dict, partners_of_b: dict,
                   cache_a, cache_b) -> tuple:
    """Per-slot distance from one negative pair to the closest true positive pair.

    Two independent numbers, one per slot, because either substitution can be the one that makes
    the pair resemble a positive. Holding slot A fixed and asking how far slot B sits from a true
    partner of A gives the slot-B distance; the mirror gives the slot-A distance.

    Both are returned rather than only their minimum. The minimum is slot-sensitive, since the
    shorter protein supplies it far more often, so a caller that summarizes should be able to see
    what it is summarizing.

    Args:
      hash_a: slot-A `cds_dna_hash` of the negative pair.
      hash_b: slot-B `cds_dna_hash` of the negative pair.
      partners_of_a: slot-A hash -> its true slot-B partners.
      partners_of_b: slot-B hash -> its true slot-A partners.
      cache_a: per-site cache for the slot-A protein.
      cache_b: per-site cache for the slot-B protein.

    Returns:
      `(distance_slot_a, distance_slot_b)`, either of which is None when that slot's sequence
      appears in no observed co-occurrence.
    """
    distance_b = None
    if hash_a in partners_of_a:
        distance_b = min(site_distance(cache_b, hash_b, true) for true in partners_of_a[hash_a])
    distance_a = None
    if hash_b in partners_of_b:
        distance_a = min(site_distance(cache_a, hash_a, true) for true in partners_of_b[hash_b])
    return distance_a, distance_b


def summarize_slots(distance_a, distance_b, how: str):
    """Reduce the two per-slot distances to the single value used for binning.

    None of these is uniquely correct. `min` asks how close the pair is after changing whichever
    one slot needs less change, and is the most direct reading of "resembles a positive", but it
    is dominated by whichever slot is easier to match. `mean` and `max` weight the slots evenly.

    Args:
      distance_a: slot-A distance, or None.
      distance_b: slot-B distance, or None.
      how: 'min', 'mean' or 'max'.

    Returns:
      The summarized distance, or None when neither slot could be measured.

    Raises:
      ValueError: `how` is not one of the three supported summaries.
    """
    available = [d for d in (distance_a, distance_b) if d is not None]
    if not available:
        return None
    if how == 'min':
        return min(available)
    if how == 'max':
        return max(available)
    if how == 'mean':
        return sum(available) / len(available)
    raise ValueError(f"summarize_slots: how must be 'min', 'mean' or 'max'; got {how!r}.")


def check_bin_edges(edges: tuple) -> None:
    """Reject bin edges that cannot describe a usable set of bins.

    Args:
      edges: the upper edges, in sites.

    Raises:
      ValueError: the edges are empty, negative, or not strictly increasing.
    """
    if not edges:
        raise ValueError('--bin_edges needs at least one upper edge.')
    if any(edge < 0 for edge in edges):
        raise ValueError(f'--bin_edges must be non-negative; got {list(edges)}.')
    if any(later <= earlier for earlier, later in zip(edges, edges[1:])):
        raise ValueError(f'--bin_edges must be strictly increasing; got {list(edges)}.')


def analyze_run(run_dir: Path, cache_a, cache_b, run_label: str, partners_of_a: dict,
                partners_of_b: dict, summary: str) -> pd.DataFrame:
    """Score every negative test pair of one run by its distance to the nearest positive.

    Args:
      run_dir: model run directory holding `test_predicted.csv`.
      cache_a: per-site cache for the slot-A protein.
      cache_b: per-site cache for the slot-B protein.
      run_label: name recorded in the `run` column.
      partners_of_a: slot-A hash -> its true slot-B partners, over the whole positive universe.
      partners_of_b: slot-B hash -> its true slot-A partners.
      summary: how the two per-slot distances are reduced for binning ('min', 'mean', 'max').

    Returns:
      One row per negative pair, with `distance_slot_a`, `distance_slot_b`,
      the summary column named by `distance_column(summary)`, `nearer_slot`, `pred_prob`,
      `pred_label` and
      `is_false_positive`.

    Raises:
      FileNotFoundError: the run has no `test_predicted.csv`.
      ValueError: `pred_label` is not binary, or no negative pair could be scored, which means the
          caches and the run disagree.
    """
    predictions_path = run_dir / 'test_predicted.csv'
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"missing {predictions_path}. Train the run first with "
            f"`python src/models/train_pair_baselines.py --baseline lgbm ...`.")
    # keep_default_na: pair tables carry protein names, and 'NA' (Neuraminidase) is a real value.
    predictions = pd.read_csv(predictions_path, keep_default_na=False, na_values=[''],
                              low_memory=False)
    labels = predictions['label'].astype(int)
    # Validated rather than rounded: a non-binary pred_label would mean the run wrote something
    # other than a hard decision, and silently rounding it would invent one.
    predicted = predictions['pred_label'].astype(float)
    unexpected = set(predicted.unique()) - {0.0, 1.0}
    if unexpected:
        raise ValueError(
            f"{predictions_path}: pred_label must be 0 or 1; found {sorted(unexpected)[:5]}.")
    predicted = predicted.astype(int)

    negatives = predictions[labels == 0].copy()
    negatives['pred_label'] = predicted[labels == 0]
    distances_a, distances_b, summarized = [], [], []
    for hash_a, hash_b in zip(negatives['cds_dna_hash_a'], negatives['cds_dna_hash_b']):
        distance_a, distance_b = slot_distances(hash_a, hash_b, partners_of_a, partners_of_b,
                                                cache_a, cache_b)
        distances_a.append(distance_a)
        distances_b.append(distance_b)
        summarized.append(summarize_slots(distance_a, distance_b, summary))
    negatives['distance_slot_a'] = distances_a
    negatives['distance_slot_b'] = distances_b
    summary_column = distance_column(summary)
    negatives[summary_column] = summarized
    negatives['nearer_slot'] = np.where(
        pd.isna(negatives['distance_slot_a']), 'b',
        np.where(pd.isna(negatives['distance_slot_b']), 'a',
                 np.where(negatives['distance_slot_a'] <= negatives['distance_slot_b'], 'a', 'b')))
    negatives['is_false_positive'] = negatives['pred_label'] == 1
    negatives['run'] = run_label

    n_unscored = int(negatives[summary_column].isna().sum())
    if n_unscored == len(negatives):
        raise ValueError(
            f"{run_dir}: no negative pair could be matched to an observed co-occurrence. "
            f"The per-site caches and this run's pair tables do not describe the same population.")
    if n_unscored > 0:
        print(f"WARNING: {run_label}: {n_unscored:,} of {len(negatives):,} negatives have neither "
              f"sequence in an observed co-occurrence; dropped.")
    scored = negatives.dropna(subset=[summary_column]).copy()

    columns = ['run', 'pair_key', 'distance_slot_a', 'distance_slot_b',
               summary_column, 'nearer_slot',
               'pred_prob', 'pred_label', 'is_false_positive']
    return scored[columns]


def bin_false_positive_rate(negatives: pd.DataFrame, edges: tuple, run_label: str,
                            distance_col: str) -> pd.DataFrame:
    """Count negatives and false positives in each distance bin.

    Args:
      negatives: scored negative pairs, from `analyze_run`.
      edges: ascending upper edges, in sites.
      run_label: value written to the `run` column.

    Returns:
      One row per bin: `run`, `bin_label`, `n_negatives`, `n_false_positives`,
      `false_positive_rate`.
    """
    labels = bin_labels(edges)
    indices = assign_bin(negatives[distance_col].to_numpy(), edges)
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
                   dpi: int, distance_col: str, summary: str) -> Path:
    """Draw the false-positive rate by distance bin and the distance distributions.

    Args:
      negatives: scored negative pairs pooled across runs.
      bins: pooled binned rates, from `bin_false_positive_rate`.
      unit: site unit, used in the axis labels.
      out_path: where the PNG goes.
      dpi: raster resolution.

    Returns:
      The written path.
    """
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
    ax_rate.set_xlabel(f'{summary} distance to an observed positive pair ({unit} sites)')
    ax_rate.set_ylabel('false-positive rate')
    ax_rate.set_ylim(0, 1.12)
    ax_rate.set_title('This model misreads the negatives that resemble positives')

    # Cumulative rather than a density: the distance distribution has a long sparse tail (distant
    # lineages) that squashes the bulk, and the question here is what share of the errors sits
    # inside a given distance, which a cumulative curve answers directly.
    all_distances = negatives[distance_col].to_numpy()
    false_distances = negatives.loc[negatives['is_false_positive'], distance_col].to_numpy()
    upper = max(int(np.percentile(all_distances, 90)), 10)
    thresholds = np.arange(0, upper + 1)
    false_share = [(false_distances <= t).mean() for t in thresholds]
    negative_share = [(all_distances <= t).mean() for t in thresholds]

    ax_dist.plot(thresholds, false_share, color=ACCENT_COLOR, linewidth=2,
                 label=f'false positives (n={len(false_distances):,})')
    ax_dist.plot(thresholds, negative_share, color=TRACE_COLOR, linewidth=2, linestyle='--',
                 label=f'all negatives (n={len(all_distances):,})')
    ax_dist.set_xlabel(f'{summary} distance threshold ({unit} sites)')
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
    parser.add_argument('--distance_summary', default='min', choices=['min', 'mean', 'max'],
                        help='how the two per-slot distances are reduced for binning; both slot '
                             'distances are written to the CSV regardless')
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

    summary_column = distance_column(args.distance_summary)
    per_run, bin_tables = [], []
    for run_dir in args.run_dirs:
        scored = analyze_run(run_dir, cache_a, cache_b, run_dir.name, partners_of_a, partners_of_b,
                             args.distance_summary)
        per_run.append(scored)
        bin_tables.append(bin_false_positive_rate(scored, edges, run_dir.name, summary_column))
        n_false = int(scored['is_false_positive'].sum())
        print(f"  {run_dir.name}: {len(scored):,} negatives, {n_false:,} false positives "
              f"({n_false/len(scored):.1%})")

    negatives = pd.concat(per_run, ignore_index=True)
    pooled = bin_false_positive_rate(negatives, edges, '(all runs)', summary_column)
    bins = pd.concat([pooled] + bin_tables, ignore_index=True)

    stem = f'negative_pair_ambiguity_{args.unit}_{args.distance_summary}'
    negatives_path = args.out_dir / f'{stem}.csv'
    bins_path = args.out_dir / f'{stem}_bins.csv'
    negatives.to_csv(negatives_path, index=False)
    bins.to_csv(bins_path, index=False)

    figure_path = plot_ambiguity(negatives, pooled, args.unit, args.out_dir / f'{stem}.png',
                                 args.dpi, summary_column, args.distance_summary)

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
        near = negatives[negatives[summary_column] <= edge]
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
