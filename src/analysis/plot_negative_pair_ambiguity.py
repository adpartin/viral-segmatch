"""Where do a model's false positives sit relative to the nearest real positive pair?

Post-hoc error analysis on an already-trained run. Nothing is retrained and no feature is
corrupted: the saved `test_predicted.csv` is read back and each NEGATIVE row is scored by how far
its assigned partner sits from the partner it should have had.

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
26 of the 155 closest negatives (0-2 nt) correctly, and k-mer classifies 24 of them. No negative
sits at distance 0, so no identical feature vector ever carries contradictory labels. Claiming
these rows cannot be separated goes beyond what this measurement supports.

What this does NOT explain. It does not establish why precision reads lower than recall. Only
negative rows are analysed here, so the false-positive and false-negative mechanisms are never
compared. That gap is also threshold-dependent: on the pooled random-CV folds the errors balance
near a 0.5 threshold shifted to about 0.70 (312 false positives against 310 false negatives), which
a concentration measured only over negatives cannot account for.

The measure. For a negative pair `(a, b)`, `distance_to_nearest_positive` is the smallest number of
differing sites between that pair and any true positive pair, counted on the one slot that differs:
the distance from `b` to a true partner of `a`, or from `a` to a true partner of `b`, whichever is
smaller. Taking the minimum answers one specific question, "how close is this pair to a positive
after changing whichever single slot needs less change", and it is not the only defensible
summary. It is also slot-sensitive, since the shorter NA supplies the minimum far more often than
HA. Mean or maximum over the two slots are reasonable alternatives that weight the slots evenly.
Distance is measured in sites of `--unit`, on the per-site cache, which is aligned by construction
and so makes a position-by-position comparison meaningful. Every site counts equally, which is not
how the model weights them.

Read concentration as enrichment, not as a raw share. A bin holding most of the false positives may
simply be holding most of the negatives. On H3N2 2024 HA-NA the enrichment is 6.0x within 2 nt,
2.55x within 5 nt and only 1.25x within 10 nt, so the near bins are what carry the signal.

Read the output as a diagnostic of the negative sampler and the population, not of the model.

Scope. Distance is nucleotide (or codon / amino-acid) identity, which stands in for how closely two
sequences are related. It is not a phylogenetic assignment, so "near-duplicate" here means
sequence-similar, not confirmed same-clade.

True partners are read from the same test split as the negatives, which matches how both negative
samplers work: each split's negatives are recombined from that split's own positives.

Outputs (to `--out_dir`, by default derived from the first run dir):
    negative_pair_ambiguity_{unit}.png       false-positive rate by distance bin, plus the
                                             distance distribution for false positives vs
                                             true negatives
    negative_pair_ambiguity_{unit}.csv       one row per negative test pair: run, pair_key,
                                             distance_to_nearest_positive, differing_slot,
                                             pred_prob, pred_label, is_false_positive
    negative_pair_ambiguity_{unit}_bins.csv  run, bin_label, n_negatives, n_false_positives,
                                             false_positive_rate

CLI:
    python -m src.analysis.plot_negative_pair_ambiguity \\
        --run_dirs models/flu/July_2025/runs/lgbm_ha_na_h3n2_2024_random_cv4_site_nt_fold{0,1,2,3} \\
        --config_bundle flu_ha_na_h3n2_2024_random_cv4_site_nt --unit nt
"""
from __future__ import annotations

import argparse
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


def true_partner_map(positives: pd.DataFrame, key_col: str, partner_col: str) -> dict:
    """Map each sequence hash to the hashes it truly pairs with.

    A hash can carry more than one true partner when the same sequence occurs in several
    isolates, so the values are lists rather than single hashes.

    Args:
      positives: the label==1 rows of one test split.
      key_col: hash column to key on.
      partner_col: hash column holding that key's true partner.

    Returns:
      `{hash: [partner hash, ...]}`.
    """
    partners: dict = {}
    for key, partner in zip(positives[key_col], positives[partner_col]):
        partners.setdefault(key, []).append(partner)
    return partners


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


def nearest_positive_distance(hash_a: str, hash_b: str, partners_of_a: dict, partners_of_b: dict,
                              cache_a, cache_b) -> tuple:
    """Distance from one negative pair to the closest true positive pair.

    Measured on the single slot that differs. Substituting slot B gives the distance from `hash_b`
    to a true partner of `hash_a`; substituting slot A gives the mirror. The smaller of the two is
    returned, because the pair resembles a positive if either side is nearly right.

    Args:
      hash_a: slot-A `cds_dna_hash` of the negative pair.
      hash_b: slot-B `cds_dna_hash` of the negative pair.
      partners_of_a: slot-A hash -> its true slot-B partners.
      partners_of_b: slot-B hash -> its true slot-A partners.
      cache_a: per-site cache for the slot-A protein.
      cache_b: per-site cache for the slot-B protein.

    Returns:
      `(distance, differing_slot)`, or `(None, None)` when neither sequence appears in a positive
      of this split. `differing_slot` is 'b' when slot B was the substituted side, else 'a'.
    """
    candidates = []
    if hash_a in partners_of_a:
        distance_b = min(site_distance(cache_b, hash_b, true) for true in partners_of_a[hash_a])
        candidates.append((distance_b, 'b'))
    if hash_b in partners_of_b:
        distance_a = min(site_distance(cache_a, hash_a, true) for true in partners_of_b[hash_b])
        candidates.append((distance_a, 'a'))
    if not candidates:
        return None, None
    return min(candidates)


def analyze_run(run_dir: Path, cache_a, cache_b, run_label: str) -> pd.DataFrame:
    """Score every negative test pair of one run by its distance to the nearest positive.

    Args:
      run_dir: model run directory holding `test_predicted.csv`.
      cache_a: per-site cache for the slot-A protein.
      cache_b: per-site cache for the slot-B protein.
      run_label: name recorded in the `run` column.

    Returns:
      One row per negative pair, with `distance_to_nearest_positive`, `differing_slot`,
      `pred_prob`, `pred_label` and `is_false_positive`.

    Raises:
      FileNotFoundError: the run has no `test_predicted.csv`.
      ValueError: no negative pair could be scored, which means the caches and the run disagree.
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
    predicted = predictions['pred_label'].astype(float).round().astype(int)

    positives = predictions[labels == 1]
    partners_of_a = true_partner_map(positives, 'cds_dna_hash_a', 'cds_dna_hash_b')
    partners_of_b = true_partner_map(positives, 'cds_dna_hash_b', 'cds_dna_hash_a')

    negatives = predictions[labels == 0].copy()
    negatives['pred_label'] = predicted[labels == 0]
    distances, slots = [], []
    for hash_a, hash_b in zip(negatives['cds_dna_hash_a'], negatives['cds_dna_hash_b']):
        distance, slot = nearest_positive_distance(hash_a, hash_b, partners_of_a, partners_of_b,
                                                   cache_a, cache_b)
        distances.append(distance)
        slots.append(slot)
    negatives['distance_to_nearest_positive'] = distances
    negatives['differing_slot'] = slots
    negatives['is_false_positive'] = negatives['pred_label'] == 1
    negatives['run'] = run_label

    n_unscored = int(negatives['distance_to_nearest_positive'].isna().sum())
    if n_unscored == len(negatives):
        raise ValueError(
            f"{run_dir}: no negative pair could be matched to a positive of the same split. "
            f"The per-site caches and this run's pair tables do not describe the same population.")
    if n_unscored > 0:
        print(f"WARNING: {run_label}: {n_unscored:,} of {len(negatives):,} negatives have neither "
              f"sequence in a positive of this split; dropped.")
    scored = negatives.dropna(subset=['distance_to_nearest_positive']).copy()
    scored['distance_to_nearest_positive'] = scored['distance_to_nearest_positive'].astype(int)

    columns = ['run', 'pair_key', 'distance_to_nearest_positive', 'differing_slot',
               'pred_prob', 'pred_label', 'is_false_positive']
    return scored[columns]


def bin_false_positive_rate(negatives: pd.DataFrame, edges: tuple, run_label: str) -> pd.DataFrame:
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
    indices = assign_bin(negatives['distance_to_nearest_positive'].to_numpy(), edges)
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
    ax_rate.set_xlabel(f'distance to nearest positive pair ({unit} sites)')
    ax_rate.set_ylabel('false-positive rate')
    ax_rate.set_ylim(0, 1.12)
    ax_rate.set_title('Negatives that look like positives are the ones misread')

    # Cumulative rather than a density: the distance distribution has a long sparse tail (distant
    # lineages) that squashes the bulk, and the question here is what share of the errors sits
    # inside a given distance, which a cumulative curve answers directly.
    all_distances = negatives['distance_to_nearest_positive'].to_numpy()
    false_distances = negatives.loc[negatives['is_false_positive'],
                                    'distance_to_nearest_positive'].to_numpy()
    upper = max(int(np.percentile(all_distances, 90)), 10)
    thresholds = np.arange(0, upper + 1)
    false_share = [(false_distances <= t).mean() for t in thresholds]
    negative_share = [(all_distances <= t).mean() for t in thresholds]

    ax_dist.plot(thresholds, false_share, color=ACCENT_COLOR, linewidth=2,
                 label=f'false positives (n={len(false_distances):,})')
    ax_dist.plot(thresholds, negative_share, color=TRACE_COLOR, linewidth=2, linestyle='--',
                 label=f'all negatives (n={len(all_distances):,})')
    ax_dist.set_xlabel(f'distance threshold ({unit} sites)')
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
    parser.add_argument('--out_dir', type=Path, default=None)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    function_to_short = get_function_short_name_map(config)
    edges = tuple(args.bin_edges)

    # Both slots' proteins come from the first run's pair table, so the caches match the runs
    # rather than a bundle default that may name a different schema pair.
    first = pd.read_csv(args.run_dirs[0] / 'test_predicted.csv', keep_default_na=False,
                        na_values=[''], low_memory=False, nrows=1)
    function_a, function_b = first['func_a'].iloc[0], first['func_b'].iloc[0]
    protein_a = function_to_short.get(function_a, function_a)
    protein_b = function_to_short.get(function_b, function_b)
    cache_a = load_site_cache(args.site_dir, args.unit, protein_a)
    cache_b = load_site_cache(args.site_dir, args.unit, protein_b)
    print(f"Slots: a={protein_a} ({cache_a.codes.shape[1]:,} sites), "
          f"b={protein_b} ({cache_b.codes.shape[1]:,} sites), unit={args.unit}")

    if args.out_dir is None:
        args.out_dir = args.run_dirs[0] / 'negative_pair_ambiguity'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_run, bin_tables = [], []
    for run_dir in args.run_dirs:
        scored = analyze_run(run_dir, cache_a, cache_b, run_dir.name)
        per_run.append(scored)
        bin_tables.append(bin_false_positive_rate(scored, edges, run_dir.name))
        n_false = int(scored['is_false_positive'].sum())
        print(f"  {run_dir.name}: {len(scored):,} negatives, {n_false:,} false positives "
              f"({n_false/len(scored):.1%})")

    negatives = pd.concat(per_run, ignore_index=True)
    pooled = bin_false_positive_rate(negatives, edges, '(all runs)')
    bins = pd.concat([pooled] + bin_tables, ignore_index=True)

    negatives_path = args.out_dir / f'negative_pair_ambiguity_{args.unit}.csv'
    bins_path = args.out_dir / f'negative_pair_ambiguity_{args.unit}_bins.csv'
    negatives.to_csv(negatives_path, index=False)
    bins.to_csv(bins_path, index=False)

    figure_path = plot_ambiguity(negatives, pooled, args.unit,
                                 args.out_dir / f'negative_pair_ambiguity_{args.unit}.png',
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
        near = negatives[negatives['distance_to_nearest_positive'] <= edge]
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
