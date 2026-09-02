"""Per-position feature importance along each CDS, read against the per-site entropy.

Step 6 of docs/plans/2026-08-28_per_site_nt_features_plan.md. Per-site features exist so that an
importance score can be traced back to a place in the sequence, which a k-mer count cannot do.
This is where that pays off.

Two importance measures, because neither alone is enough:

- **Gain** -- total split gain from the fitted trees: how much the loss fell across every split
  that used the feature. Cheap, but it is a training-time quantity read off the model's insides,
  so it says what the trees were built on, not what they are worth on new data.
- **SHAP** -- exact TreeSHAP through LightGBM's own `pred_contrib`, so no extra dependency, scored
  on each fold's HELD-OUT test split. Contributions are per prediction and sum to the model's
  output, and they measure what the model does on data it did not fit. The script checks that sum
  against the raw margin rather than assuming it.

Agreement between the two is reported. Where they disagree, SHAP is the one to believe, because it
is measured out of sample.

Each fold is normalised to sum to 1 before averaging, because early stopping gives the folds
different tree counts and so different totals. A site's score is therefore its share of the
model's total, averaged over folds, and the four folds are independent fits, so the spread across
them says whether a site is consistently used or was picked up once.

The split count is carried too -- that is what `lightgbm.plot_importance` shows by default. It is
in the CSV rather than the figure, since counting splits says how often a position was consulted,
not how much it was worth.

Read against entropy. A site the model leans on must vary -- an invariant column cannot separate
anything -- so importance is bounded by conservation, and the interesting sites are the ones with
importance far above or below what their variability would suggest. Entropy is computed here from
the feature cache restricted to the sequences this dataset holds, not from the whole corpus, and
in the SAME unit as the features, so codon importance is read against codon entropy (ceiling 6
bits) rather than against nucleotide entropy (ceiling 2).

Reading the CSVs back: the `protein` column holds the literal string `NA` (Neuraminidase), which
a default `pd.read_csv` parses as NaN and so silently drops every Neuraminidase row. Read with
`keep_default_na=False, na_values=['']`, the same rule CLAUDE.md gives for any `function_short`
column.

Outputs (to `--out_dir`, by default derived from the dataset dir):
    site_importance_{unit}.png   importance along each CDS, plus importance against entropy
    site_importance_{unit}.csv   column, slot, protein, site, shap_frac, shap_frac_std,
                                 gain_frac, gain_frac_std, split_count, folds_used,
                                 entropy_bits, n_values, shap_rank, gain_rank

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
from src.utils.site_utils import (  # noqa: E402
    column_entropy,
    get_site_pair_features,
    load_site_cache,
    site_feature_columns,
)

# Sampled from tmp/score/h3n2_f1_macro_within_fold.png, so the per-site figures read as one series
# with the score plots.
TRACE_COLOR = '#4C7CAB'
ACCENT_COLOR = '#CF8793'
MARKER_EDGE = '#222222'


def fold_importances(model_root: Path, run_stem: str, n_folds: int, n_columns: int,
                     fold_features) -> dict:
    """Read each fold's booster and score every column two ways.

    Args:
      model_root: directory holding the per-fold run dirs.
      run_stem: run dir name minus the `_fold{k}` suffix.
      n_folds: how many folds to read.
      n_columns: expected feature count, checked against every booster.
      fold_features: `fold -> X` for the split SHAP is measured on. Held out from that fold's fit.

    Returns:
      `{'gain', 'shap', 'split'}`, each an (n_folds, n_columns) array. `gain` and `shap` rows each
      sum to 1; `shap` is the mean absolute SHAP value per column.

    Raises:
      FileNotFoundError: a fold's model file is missing.
      ValueError: a booster has a different feature count than expected, one contributed no gain
          at all, or its SHAP contributions do not reconstruct its own predictions.
    """
    gains, shaps, splits = [], [], []
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
        if gain.sum() <= 0:
            raise ValueError(f"{model_path}: every feature has zero gain, so there is nothing "
                             f"to rank.")
        gains.append(gain / gain.sum())
        splits.append(np.asarray(booster.feature_importance(importance_type='split'),
                                 dtype=np.int64))

        # Exact TreeSHAP: the last column is the base value, and contributions plus base must
        # reconstruct the raw margin. Checked rather than assumed -- a silent mismatch here would
        # make the whole ranking wrong in a way nothing else would catch.
        X = fold_features[fold]
        contributions = booster.predict(X, pred_contrib=True)
        reconstructed = contributions[:, :-1].sum(axis=1) + contributions[:, -1]
        margin = booster.predict(X, raw_score=True)
        if not np.allclose(reconstructed, margin, atol=1e-6):
            raise ValueError(
                f"{model_path}: SHAP contributions do not reconstruct the model's own output "
                f"(max difference {np.abs(reconstructed - margin).max():.3g}).")
        mean_abs = np.abs(contributions[:, :-1]).mean(axis=0)
        shaps.append(mean_abs / mean_abs.sum())

        print(f"  fold {fold}: {booster.num_trees():,} trees, "
              f"{int((gain > 0).sum()):,} of {n_columns:,} sites used, "
              f"SHAP over {len(X):,} held-out rows")
    return {'gain': np.vstack(gains), 'shap': np.vstack(shaps), 'split': np.vstack(splits)}


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


def _stamp(fig) -> None:
    """Write the producing script into the figure, so a stray PNG can be traced back.

    Args:
      fig: the figure to stamp.
    """
    fig.text(0.995, 0.002, f'src/analysis/{Path(__file__).name}', ha='right', va='bottom',
             fontsize=7, color='0.45')


def lgbm_plot_importance(model_root: Path, run_stem: str, fold: int, columns, table,
                         top_n: int, out_png: Path, unit: str, dpi: int) -> Path:
    """Draw LightGBM's own ranked bar charts beside the held-out SHAP ranking.

    `lightgbm.plot_importance` is the conventional view and calls the same
    `booster.feature_importance` this module does, so the split and gain panels are LightGBM's
    output unmodified except for the tick labels: the boosters were fitted on plain arrays, so
    their features are named `Column_0..`, and those are rewritten to the site they stand for.

    Both LightGBM panels show ONE fold, since `plot_importance` takes one booster. The SHAP panel
    is the average over all folds, which is why its order differs -- a single fold's ranking is
    noisier than the average, and that difference is itself worth seeing.

    Args:
      model_root: directory holding the per-fold run dirs.
      run_stem: run dir name minus the `_fold{k}` suffix.
      fold: which fold's booster the two LightGBM panels use.
      columns: the site layout from `site_feature_columns`.
      table: the per-site table, for the fold-averaged SHAP panel.
      top_n: features per panel.
      out_png: where to write.
      unit: site unit, for the titles.
      dpi: figure resolution.

    Returns:
      The path written.
    """
    import lightgbm as lgb

    booster = joblib.load(model_root / f'{run_stem}_fold{fold}' / 'best_model.joblib').booster_
    site_label = {int(r.column): f"{r.protein} {int(r.site)}" for r in columns.itertuples()}

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.34 * top_n + 2.2))
    for ax, importance_type in zip(axes[:2], ('split', 'gain')):
        lgb.plot_importance(booster, ax=ax, importance_type=importance_type,
                            max_num_features=top_n, color=TRACE_COLOR,
                            title=f'LightGBM {importance_type} (fold {fold})',
                            xlabel=f'{importance_type} importance', ylabel='')
        # 'Column_123' -> 'HA 124'. The number LightGBM prints is the column index, which means
        # nothing on its own; the site is the point of per-site features.
        ax.set_yticklabels([site_label.get(int(t.get_text().removeprefix('Column_')),
                                           t.get_text())
                            for t in ax.get_yticklabels()])
        ax.grid(axis='x', alpha=0.3)

    top = table.nlargest(top_n, 'shap_frac').iloc[::-1]
    labels = [f"{r.protein} {int(r.site)}" for r in top.itertuples()]
    ax = axes[2]
    ax.barh(range(len(top)), top['shap_frac'], xerr=top['shap_frac_std'],
            color=ACCENT_COLOR, edgecolor=MARKER_EDGE, linewidth=0.7,
            error_kw={'ecolor': MARKER_EDGE, 'elinewidth': 0.8})
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('share of SHAP')
    ax.set_title('held-out SHAP (all folds, mean +/- std)')
    ax.grid(axis='x', alpha=0.3)

    fig.suptitle(f'{run_stem}  |  unit={unit}, top {top_n} sites', fontsize=10, y=1.01)
    fig.tight_layout()
    _stamp(fig)
    written = savefig(out_png, dpi=dpi)
    print(f"Wrote {written}")
    return written


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
    p.add_argument('--shap_split', default='test', choices=['train', 'val', 'test'],
                   help="split SHAP is measured on; 'test' is held out from the fit")
    p.add_argument('--top_n', type=int, default=15, help='sites listed per protein')
    p.add_argument('--barplot_fold', type=int, default=0,
                   help="fold whose booster the LightGBM bar charts use; plot_importance "
                        "takes one booster")
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
    # SHAP is measured on the split each fold did NOT fit, so it says what the model does on
    # data it has not seen.
    fold_features = {}
    for fold in range(args.n_folds):
        pairs_path = args.dataset_dir / f'fold_{fold}' / f'{args.shap_split}_pairs.parquet'
        if not pairs_path.exists():
            raise FileNotFoundError(f"missing {pairs_path}, needed to measure SHAP.")
        pairs = pd.read_parquet(pairs_path)
        fold_features[fold], _ = get_site_pair_features(pairs, caches[0], caches[1], 'ordinal')

    print(f"\nReading {args.n_folds} boosters for {args.model_run_template} "
          f"({len(columns):,} columns, unit={args.unit}, SHAP on {args.shap_split}):")
    importances = fold_importances(args.models_root, args.model_run_template, args.n_folds,
                                   len(columns), fold_features)
    gain_fractions = importances['gain']
    shap_fractions = importances['shap']
    split_counts = importances['split']

    entropies, n_values = [], []
    for cache, function in zip(caches, functions):
        entropy, values = site_entropy_from_cache(cache, hashes_by_function[function])
        entropies.append(entropy)
        n_values.append(values)

    table = columns.drop(columns=['code']).copy()
    table['shap_frac'] = shap_fractions.mean(axis=0)
    table['shap_frac_std'] = shap_fractions.std(axis=0, ddof=1)
    table['gain_frac'] = gain_fractions.mean(axis=0)
    table['gain_frac_std'] = gain_fractions.std(axis=0, ddof=1)
    table['folds_used'] = (gain_fractions > 0).sum(axis=0)
    table['split_count'] = split_counts.mean(axis=0)
    table['entropy_bits'] = np.concatenate(entropies)
    table['n_values'] = np.concatenate(n_values)
    table['shap_rank'] = table['shap_frac'].rank(ascending=False, method='min').astype(int)
    table['gain_rank'] = table['gain_frac'].rank(ascending=False, method='min').astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f'site_importance_{args.unit}.csv'
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Per-fold importance, one column per fold, so the agreement between folds can be recomputed
    # or re-plotted later. The printed Spearman numbers are derived from exactly this.
    per_fold = columns.drop(columns=['code']).copy()
    for fold in range(args.n_folds):
        per_fold[f'shap_frac_fold{fold}'] = shap_fractions[fold]
        per_fold[f'gain_frac_fold{fold}'] = gain_fractions[fold]
    out_folds = args.out_dir / f'site_importance_{args.unit}_per_fold.csv'
    per_fold.to_csv(out_folds, index=False)
    print(f"Wrote {out_folds}")

    for cache in caches:
        of_protein = table[table['protein'] == cache.protein].sort_values(
            'shap_frac', ascending=False)
        columns_here = of_protein['column'].to_numpy()
        used = int((of_protein['gain_frac'] > 0).sum())
        # An invariant site cannot separate anything, so this is the ceiling on how many sites
        # could possibly matter.
        varying = int((of_protein['entropy_bits'] > 0).sum())
        print(f"\n{cache.protein}: {len(of_protein):,} sites, "
              f"{of_protein['shap_frac'].sum():.1%} of total SHAP, "
              f"{of_protein['gain_frac'].sum():.1%} of total gain")
        print(f"  sites with any gain: {used:,} ({used / len(of_protein):.1%}); "
              f"sites that vary at all: {varying:,} ({varying / len(of_protein):.1%})")
        for measure in ('shap_frac', 'gain_frac'):
            share = of_protein[measure].sum()
            by_measure = of_protein[measure].sort_values(ascending=False)
            held = [f"top {n} {by_measure.head(n).sum() / share:.1%}" for n in (10, 50)]
            print(f"  {measure.split('_')[0]:5s}: " + ', '.join(held) +
                  " of this protein's total")

        varies = of_protein[of_protein['entropy_bits'] > 0]
        print(f"  Spearman(SHAP, entropy) over varying sites: "
              f"{varies['shap_frac'].corr(varies['entropy_bits'], method='spearman'):+.3f}; "
              f"Spearman(gain, entropy): "
              f"{varies['gain_frac'].corr(varies['entropy_bits'], method='spearman'):+.3f}")

        # Do the two measures agree? Where they do not, believe SHAP: it is measured out of
        # sample, while gain is read off the fitted trees.
        rho_measures = of_protein['shap_frac'].corr(of_protein['gain_frac'], method='spearman')
        shap_top = set(of_protein.head(args.top_n)['site'])
        gain_top = set(of_protein.nlargest(args.top_n, 'gain_frac')['site'])
        print(f"  SHAP vs gain: Spearman {rho_measures:+.3f}, "
              f"{len(shap_top & gain_top)}/{args.top_n} top sites shared")

        # Whether the ranking is worth reading at all: the four folds are independent fits, so
        # agreement between them is what separates a real ordering from one fold's noise.
        fold_pairs = [(i, j) for i in range(args.n_folds) for j in range(i + 1, args.n_folds)]
        for label, per_fold in (('SHAP', shap_fractions), ('gain', gain_fractions)):
            of_slot = per_fold[:, columns_here]
            agreements = [pd.Series(of_slot[i]).corr(pd.Series(of_slot[j]), method='spearman')
                          for i, j in fold_pairs]
            top_sets = [set(np.argsort(-of_slot[i])[:args.top_n]) for i in range(args.n_folds)]
            print(f"  fold-to-fold Spearman on {label:4s}: {np.mean(agreements):.3f} "
                  f"(range {min(agreements):.3f}-{max(agreements):.3f}); "
                  f"{len(set.intersection(*top_sets))}/{args.top_n} top sites shared by all "
                  f"{args.n_folds} folds")

        print(f"  top {args.top_n} sites by SHAP:")
        for _, r in of_protein.head(args.top_n).iterrows():
            print(f"    site {int(r['site']):>4d}  SHAP {r['shap_frac']:.5f} "
                  f"+/- {r['shap_frac_std']:.5f}  gain rank {int(r['gain_rank']):>4d}"
                  f"  in {int(r['folds_used'])}/{args.n_folds} folds"
                  f"  entropy {r['entropy_bits']:.3f} bits  ({int(r['n_values'])} values)")

    setup_plot_style()
    fig, axes = plt.subplots(len(caches), 2, figsize=(15, 3.4 * len(caches)),
                             gridspec_kw={'width_ratios': [3.2, 1]}, squeeze=False)
    for row, cache in enumerate(caches):
        of_protein = table[table['protein'] == cache.protein]
        ax_trace, ax_scatter = axes[row][0], axes[row][1]

        by_site = of_protein.sort_values('site')
        ax_trace.plot(by_site['site'], by_site['gain_frac'], color=MARKER_EDGE, linewidth=0.6,
                      alpha=0.35, label='gain (in-sample)')
        ax_trace.plot(by_site['site'], by_site['shap_frac'], color=TRACE_COLOR, linewidth=0.8,
                      label=f'SHAP ({args.shap_split}, held out)')
        top = of_protein.nlargest(5, 'shap_frac')
        ax_trace.scatter(top['site'], top['shap_frac'], s=32, color=ACCENT_COLOR,
                         edgecolors=MARKER_EDGE, linewidths=0.7, zorder=3)
        for _, r in top.iterrows():
            ax_trace.annotate(f"{int(r['site'])}", (r['site'], r['shap_frac']),
                              textcoords='offset points', xytext=(4, 3), fontsize=8,
                              color=MARKER_EDGE)
        ax_trace.set_xlim(1, int(of_protein['site'].max()))
        ax_trace.set_xlabel(f"{cache.protein} {args.unit} site")
        ax_trace.set_ylabel('share of importance')
        ax_trace.set_title(f"{cache.protein}: {len(of_protein):,} sites, "
                           f"{int((of_protein['gain_frac'] > 0).sum()):,} used")
        ax_trace.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax_trace.grid(axis='y', alpha=0.3)

        ax_scatter.scatter(of_protein['entropy_bits'], of_protein['shap_frac'], s=9,
                           color=TRACE_COLOR, alpha=0.5, edgecolors='none')
        ax_scatter.scatter(top['entropy_bits'], top['shap_frac'], s=32, color=ACCENT_COLOR,
                           edgecolors=MARKER_EDGE, linewidths=0.7, zorder=3)
        ax_scatter.set_xlabel('entropy (bits)')
        ax_scatter.set_ylabel('share of SHAP')
        ax_scatter.set_title('a site must vary to be used', fontsize=10)
        ax_scatter.grid(alpha=0.3)

    fig.suptitle(f"{args.model_run_template}  |  unit={args.unit}, {args.n_folds} folds, "
                 f"SHAP on {args.shap_split}", fontsize=10, y=1.005)
    fig.tight_layout()
    _stamp(fig)
    out_png = savefig(args.out_dir / f'site_importance_{args.unit}.png', dpi=args.dpi)
    print(f"\nDone. Wrote {out_png}")

    lgbm_plot_importance(
        args.models_root, args.model_run_template, args.barplot_fold, columns, table,
        args.top_n, args.out_dir / f'site_importance_{args.unit}_barplot.png', args.unit,
        args.dpi)


if __name__ == '__main__':
    main()
