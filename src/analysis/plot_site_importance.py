"""Per-position feature importance along each CDS, read against the per-site entropy.

Step 6 of docs/plans/2026-08-28_per_site_nt_features_plan.md. Per-site features exist so that an
importance score can be traced back to a place in the sequence, which a k-mer count cannot do.
This is where that pays off.

Reads the fitted LightGBM booster from each fold's run dir and turns its per-feature gain into a
per-site score. Gain is total split gain, i.e. how much the loss fell across every split that used
the feature, so it answers "how much did this position help" rather than "how often was it used";
the split count is carried alongside it.

Each fold is normalised to sum to 1 before averaging, because early stopping gives the folds
different tree counts and so different total gain. A site's score is therefore its share of the
model's total gain, averaged over folds, and the four folds are independent fits, so the spread
across them says whether a site is consistently used or was picked up once.

Read against entropy. A site the model leans on must vary -- an invariant column cannot separate
anything -- so importance is bounded by conservation, and the interesting sites are the ones with
importance far above or below what their variability would suggest. Entropy is computed here from
the feature cache restricted to the sequences this dataset holds, not from the whole corpus, and
in the SAME unit as the features, so codon importance is read against codon entropy (ceiling 6
bits) rather than against nucleotide entropy (ceiling 2).

Outputs (to `--out_dir`, by default derived from the dataset dir):
    site_importance_{unit}.png   importance along each CDS, plus importance against entropy
    site_importance_{unit}.csv   column, slot, protein, site, gain_frac, gain_frac_std,
                                 split_count, entropy_bits, n_values, rank

CLI:
    python -m src.analysis.plot_site_importance \\
        --model_run_template lgbm_ha_na_h3n2_2024_random_cv4_site_codon \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_ha_na_h3n2_2024_random_cv4_pinned_length \\
        --unit codon
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.plot_site_entropy import collect_slot_hashes  # noqa: E402
from src.utils.config_hydra import get_function_short_name_map, get_virus_config_hydra  # noqa: E402
from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402
from src.utils.site_utils import column_entropy, load_site_cache, site_feature_columns  # noqa: E402

# Sampled from tmp/score/h3n2_f1_macro_within_fold.png, so the per-site figures read as one series
# with the score plots.
TRACE_COLOR = '#4C7CAB'
ACCENT_COLOR = '#CF8793'
MARKER_EDGE = '#222222'


def fold_gain_fractions(model_root: Path, run_stem: str, n_folds: int,
                        n_columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Read each fold's booster and return its per-column gain share and split count.

    Args:
      model_root: directory holding the per-fold run dirs.
      run_stem: run dir name minus the `_fold{k}` suffix.
      n_folds: how many folds to read.
      n_columns: expected feature count, checked against every booster.

    Returns:
      `(gain_fractions, split_counts)`, shaped (n_folds, n_columns). Each row of
      `gain_fractions` sums to 1.

    Raises:
      FileNotFoundError: a fold's model file is missing.
      ValueError: a booster has a different feature count than expected, or one contributed no
          gain at all.
    """
    gains, splits = [], []
    for fold in range(n_folds):
        model_path = model_root / f'{run_stem}_fold{fold}' / 'best_model.joblib'
        if not model_path.exists():
            raise FileNotFoundError(
                f"missing {model_path}. Train the folds first with "
                f"`python src/models/train_pair_baselines.py --baseline lgbm ...`.")
        booster = joblib.load(model_path).booster_
        if booster.num_feature() != n_columns:
            raise ValueError(
                f"{model_path} was fitted on {booster.num_feature():,} features but the cache "
                f"and encoding give {n_columns:,}. The model and the site cache disagree.")
        gain = np.asarray(booster.feature_importance(importance_type='gain'), dtype=np.float64)
        total = gain.sum()
        if total <= 0:
            raise ValueError(f"{model_path}: every feature has zero gain, so there is nothing "
                             f"to rank.")
        gains.append(gain / total)
        splits.append(np.asarray(booster.feature_importance(importance_type='split'),
                                 dtype=np.int64))
        print(f"  fold {fold}: {booster.num_trees():,} trees, "
              f"{int((gain > 0).sum()):,} of {n_columns:,} sites used")
    return np.vstack(gains), np.vstack(splits)


def site_entropy_from_cache(cache, hashes) -> tuple[np.ndarray, np.ndarray]:
    """Per-site entropy of one protein, over the sequences this dataset actually holds.

    The cache spans every subtype and year at the pinned length, so entropy over all of it would
    describe a different population than the model saw.

    Args:
      cache: that protein's `SiteCache`.
      hashes: the `cds_dna_hash` values the dataset uses for that slot.

    Returns:
      `(entropy_bits, n_values)`, one per site, in the cache's own unit.
    """
    rows = sorted(cache.hash_to_row[h] for h in hashes)
    return column_entropy(cache.codes[rows])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model_run_template', required=True,
                   help='run dir name minus the _fold{k} suffix')
    p.add_argument('--dataset_dir', type=Path, required=True,
                   help='dataset run dir holding fold_*/, used to scope entropy and the slots')
    p.add_argument('--unit', default='codon', choices=['nt', 'codon', 'aa'])
    p.add_argument('--config_bundle', default='flu_ha_na_h3n2_2024_random_cv4_site_codon')
    p.add_argument('--site_dir', type=Path, default=PROJ / 'data/embeddings/flu/July_2025')
    p.add_argument('--models_root', type=Path, default=PROJ / 'models/flu/July_2025/runs')
    p.add_argument('--n_folds', type=int, default=4)
    p.add_argument('--top_n', type=int, default=15, help='sites listed per protein')
    p.add_argument('--out_dir', type=Path, default=None,
                   help='default: results/<virus>/<version>/<run name>/site_importance')
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    if args.out_dir is None:
        parts = args.dataset_dir.resolve().parts
        i = parts.index('datasets')
        args.out_dir = PROJ / 'results' / parts[i + 1] / parts[i + 2] / parts[-1] / 'site_importance'

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    function_to_short = get_function_short_name_map(config)
    protein_order = list(config.virus.protein_order)

    hashes_by_function = collect_slot_hashes(args.dataset_dir, args.n_folds)
    functions = sorted(hashes_by_function, key=protein_order.index)
    if len(functions) != 2:
        raise ValueError(f"expected two slot functions, found {functions}.")
    caches = [load_site_cache(args.site_dir, args.unit, function_to_short.get(f, f))
              for f in functions]

    # Ordinal encoding: one column per site, slot A's sites then slot B's.
    columns = site_feature_columns(caches[0], caches[1], 'ordinal')
    print(f"\nReading {args.n_folds} boosters for {args.model_run_template} "
          f"({len(columns):,} columns, unit={args.unit}):")
    gain_fractions, split_counts = fold_gain_fractions(
        args.models_root, args.model_run_template, args.n_folds, len(columns))

    entropies, n_values = [], []
    for cache, function in zip(caches, functions):
        entropy, values = site_entropy_from_cache(cache, hashes_by_function[function])
        entropies.append(entropy)
        n_values.append(values)

    table = columns.drop(columns=['code']).copy()
    table['gain_frac'] = gain_fractions.mean(axis=0)
    table['gain_frac_std'] = gain_fractions.std(axis=0, ddof=1)
    table['folds_used'] = (gain_fractions > 0).sum(axis=0)
    table['split_count'] = split_counts.mean(axis=0)
    table['entropy_bits'] = np.concatenate(entropies)
    table['n_values'] = np.concatenate(n_values)
    table['rank'] = table['gain_frac'].rank(ascending=False, method='min').astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f'site_importance_{args.unit}.csv'
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    for cache in caches:
        of_protein = table[table['protein'] == cache.protein].sort_values(
            'gain_frac', ascending=False)
        share = of_protein['gain_frac'].sum()
        used = int((of_protein['gain_frac'] > 0).sum())
        # An invariant site cannot separate anything, so this is the ceiling on how many sites
        # could possibly matter.
        varying = int((of_protein['entropy_bits'] > 0).sum())
        print(f"\n{cache.protein}: {len(of_protein):,} sites, {share:.1%} of total model gain")
        print(f"  sites with any gain: {used:,} ({used / len(of_protein):.1%}); "
              f"sites that vary at all: {varying:,} ({varying / len(of_protein):.1%})")
        for n in (10, 50):
            top_share = of_protein['gain_frac'].head(n).sum()
            print(f"  top {n:>3d} sites hold {top_share / share:.1%} of this protein's gain")
        both = of_protein[of_protein['entropy_bits'] > 0]
        rho = both['gain_frac'].corr(both['entropy_bits'], method='spearman')
        print(f"  Spearman(gain, entropy) over varying sites: {rho:+.3f}")
        # Whether the ranking is worth reading at all: the four folds are independent fits, so
        # agreement between them is what separates a real ordering from one fold's noise.
        of_slot = gain_fractions[:, of_protein['column'].to_numpy()]
        pairs = [(i, j) for i in range(args.n_folds) for j in range(i + 1, args.n_folds)]
        agreements = [pd.Series(of_slot[i]).corr(pd.Series(of_slot[j]), method='spearman')
                      for i, j in pairs]
        top_sets = [set(np.argsort(-of_slot[i])[:args.top_n]) for i in range(args.n_folds)]
        shared_top = len(set.intersection(*top_sets))
        print(f"  fold-to-fold Spearman on gain: {np.mean(agreements):.3f} "
              f"(range {min(agreements):.3f}-{max(agreements):.3f}); "
              f"{shared_top}/{args.top_n} top sites shared by all {args.n_folds} folds")
        print(f"  top {args.top_n} sites:")
        for _, r in of_protein.head(args.top_n).iterrows():
            print(f"    site {int(r['site']):>4d}  gain {r['gain_frac']:.5f} "
                  f"+/- {r['gain_frac_std']:.5f}  in {int(r['folds_used'])}/{args.n_folds} folds"
                  f"  entropy {r['entropy_bits']:.3f} bits  ({int(r['n_values'])} values)")

    setup_plot_style()
    fig, axes = plt.subplots(len(caches), 2, figsize=(15, 3.4 * len(caches)),
                             gridspec_kw={'width_ratios': [3.2, 1]}, squeeze=False)
    for row, cache in enumerate(caches):
        of_protein = table[table['protein'] == cache.protein]
        ax_trace, ax_scatter = axes[row][0], axes[row][1]

        ax_trace.plot(of_protein['site'], of_protein['gain_frac'], color=TRACE_COLOR,
                      linewidth=0.7)
        top = of_protein.nlargest(5, 'gain_frac')
        ax_trace.scatter(top['site'], top['gain_frac'], s=32, color=ACCENT_COLOR,
                         edgecolors=MARKER_EDGE, linewidths=0.7, zorder=3)
        for _, r in top.iterrows():
            ax_trace.annotate(f"{int(r['site'])}", (r['site'], r['gain_frac']),
                              textcoords='offset points', xytext=(4, 3), fontsize=8,
                              color=MARKER_EDGE)
        ax_trace.set_xlim(1, int(of_protein['site'].max()))
        ax_trace.set_xlabel(f"{cache.protein} {args.unit} site")
        ax_trace.set_ylabel('share of model gain')
        ax_trace.set_title(f"{cache.protein}: {len(of_protein):,} sites, "
                           f"{int((of_protein['gain_frac'] > 0).sum()):,} used")
        ax_trace.grid(axis='y', alpha=0.3)

        ax_scatter.scatter(of_protein['entropy_bits'], of_protein['gain_frac'], s=9,
                           color=TRACE_COLOR, alpha=0.5, edgecolors='none')
        ax_scatter.scatter(top['entropy_bits'], top['gain_frac'], s=32, color=ACCENT_COLOR,
                           edgecolors=MARKER_EDGE, linewidths=0.7, zorder=3)
        ax_scatter.set_xlabel('entropy (bits)')
        ax_scatter.set_ylabel('share of model gain')
        ax_scatter.set_title('a site must vary to be used', fontsize=10)
        ax_scatter.grid(alpha=0.3)

    fig.tight_layout()
    out_png = savefig(args.out_dir / f'site_importance_{args.unit}.png', dpi=args.dpi)
    print(f"\nDone. Wrote {out_png}")


if __name__ == '__main__':
    main()
