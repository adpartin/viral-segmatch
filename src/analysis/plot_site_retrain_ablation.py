"""Corrupt the top N sites, refit from scratch, and see whether the model can do without them.

Step 7b(iii) of docs/plans/2026-08-28_per_site_nt_features_plan.md. The permutation passes in
`plot_site_importance.py` and `plot_site_group_permutation.py` leave a fitted model in place, so
they can only say what THAT model depends on -- it has already committed to those positions and
scrambling them gives it no chance to look elsewhere. Refitting is what answers whether the
information is available at all.

Two corruption modes, which ask different questions:

- `row` -- shuffle the column's values among the rows of each split independently. The same
  sequence then carries different values at that site in different rows, so a refitted model sees
  noise and ignores the column. This is effectively DELETING the feature, and it asks: can a new
  model rebuild the signal from the other sites?
- `sequence` -- permute the site's value across the unique sequences and propagate to every row,
  so each sequence keeps one consistent but wrong value, the same one in train, val and test. The
  column still tells sequences apart; what is destroyed is the correspondence between its value
  and the real lineage. This asks: do these positions matter because of what they MEAN, or only
  because they IDENTIFY a sequence? A score that survives says the model was using them as
  fingerprints, which is the memorisation case.

Both corrupt train, val and test alike. Corrupting train alone would leave real values in test and
create a train/test mismatch, so a drop could mean "the sites mattered" or "the splits no longer
look alike" with no way to separate the two.

Two arms at every N, as in the group permutation: the top N by SHAP, and N drawn at random as the
control. The random arm redraws per fold, so four folds give four site choices. At the largest N
every column is corrupted and the arms coincide; both must land at AUC-ROC 0.5.

Hyperparameters come from `src.models.baselines.lgbm`, the same estimator and fit the baselines
use, so the only thing that differs from a normal run is the corrupted columns.

Outputs (to `--out_dir`, by default derived from the dataset dir):
    site_retrain_ablation_{unit}.png   share of signal lost against N, per mode and arm
    site_retrain_ablation_{unit}.csv   mode, arm, n_sites, fold, auc, clean_auc, signal_lost

CLI:
    python -m src.analysis.plot_site_retrain_ablation \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_ha_na_h3n2_2024_random_cv4_pinned_length \\
        --unit codon
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.plot_site_entropy import collect_slot_hashes  # noqa: E402
from src.models.baselines import lgbm  # noqa: E402
from src.utils.config_hydra import get_function_short_name_map, get_virus_config_hydra  # noqa: E402
from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402
from src.utils.seed_utils import resolve_process_seed  # noqa: E402
from src.utils.site_utils import SiteCache, get_site_pair_features, load_site_cache  # noqa: E402

MODE_COLOR = {'row': '#4C7CAB', 'sequence': '#CF8793'}
ARM_STYLE = {'top': '-', 'random': '--'}
MARKER_EDGE = '#222222'
MODES = ('row', 'sequence')


def corrupt_caches(caches: list, used_rows: list, site_columns, rng: np.random.Generator) -> list:
    """Copy the caches with the named sites permuted across sequences.

    The permutation happens once per site, over the sequences this dataset uses, and the corrupted
    caches then feed every split. So a sequence appearing in both train and test carries the same
    wrong value in both, which is what makes this the sequence-level corruption rather than the
    row-level one.

    Only the rows the dataset uses are permuted among themselves, so the column keeps the marginal
    distribution of this population rather than of the whole corpus the cache spans.

    Args:
      caches: the two `SiteCache` objects, slot A then slot B.
      used_rows: per cache, the row indices this dataset draws on.
      site_columns: `(cache index, site index)` pairs to corrupt.
      rng: draws the permutations.

    Returns:
      New `SiteCache` objects with the same index and metadata and a corrupted code matrix.
    """
    codes = [cache.codes.copy() for cache in caches]
    for which, site in site_columns:
        rows = used_rows[which]
        codes[which][rows, site] = rng.permutation(codes[which][rows, site])
    return [SiteCache(protein=cache.protein, unit=cache.unit, codes=new,
                      hash_to_row=cache.hash_to_row, metadata=cache.metadata)
            for cache, new in zip(caches, codes)]


def corrupt_rows(X: np.ndarray, columns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Copy a feature matrix with the named columns shuffled among its rows.

    Args:
      X: feature matrix.
      columns: column indices to shuffle.
      rng: draws the permutations.

    Returns:
      A corrupted copy.
    """
    out = X.copy()
    for column in columns:
        out[:, column] = rng.permutation(out[:, column])
    return out


def split_columns(columns: np.ndarray, n_sites_a: int) -> list:
    """Map global column indices to `(slot index, site index)` pairs.

    Slot A owns the first `n_sites_a` columns and slot B the rest, which is the layout
    `site_feature_columns` documents.

    Args:
      columns: global column indices.
      n_sites_a: how many columns slot A contributes.

    Returns:
      A list of `(0 or 1, site index within that slot)`.
    """
    return [(0, int(c)) if c < n_sites_a else (1, int(c) - n_sites_a) for c in columns]


def fit_and_score(config, seed: int, matrices: dict, n_columns: int) -> float:
    """Fit the baseline LightGBM on the given matrices and return test AUC-ROC.

    Uses `src.models.baselines.lgbm` so the estimator and fit match a normal baseline run, with
    every column declared categorical as the ordinal per-site encoding requires.

    Args:
      config: resolved Hydra config, read for the LightGBM hyperparameters.
      seed: estimator random state.
      matrices: `{'train': (X, y), 'val': (X, y), 'test': (X, y)}`.
      n_columns: feature count, used to declare the categorical columns.

    Returns:
      AUC-ROC on the test matrix.
    """
    estimator = lgbm.get_estimator(config, random_state=seed)
    lgbm.fit(estimator, matrices['train'][0], matrices['train'][1],
             X_val=matrices['val'][0], y_val=matrices['val'][1], config=config,
             categorical_feature=list(range(n_columns)))
    return roc_auc_score(matrices['test'][1], estimator.predict_proba(matrices['test'][0])[:, 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset_dir', type=Path, required=True)
    p.add_argument('--unit', default='codon', choices=['nt', 'codon', 'aa'])
    p.add_argument('--config_bundle', default='flu_ha_na_h3n2_2024_random_cv4_site_codon')
    p.add_argument('--site_dir', type=Path, default=PROJ / 'data/embeddings/flu/July_2025')
    p.add_argument('--importance_csv', type=Path, default=None)
    p.add_argument('--n_folds', type=int, default=4)
    p.add_argument('--n_sites', type=int, nargs='+', default=[1, 5, 10, 25, 50, 100],
                   help='set sizes to corrupt; the all-columns anchor is added automatically')
    p.add_argument('--modes', nargs='+', default=list(MODES), choices=list(MODES))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=Path, default=None)
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    if args.out_dir is None:
        parts = args.dataset_dir.resolve().parts
        i = parts.index('datasets')
        args.out_dir = (PROJ / 'results' / parts[i + 1] / parts[i + 2] / parts[-1] /
                        'site_importance')
    if args.importance_csv is None:
        args.importance_csv = args.out_dir / f'site_importance_{args.unit}.csv'

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    function_to_short = get_function_short_name_map(config)
    protein_order = list(config.virus.protein_order)
    fit_seed = resolve_process_seed(config, 'training') or args.seed

    hashes_by_function = collect_slot_hashes(args.dataset_dir, args.n_folds)
    functions = sorted(hashes_by_function, key=protein_order.index)
    caches = [load_site_cache(args.site_dir, args.unit, function_to_short.get(f, f))
              for f in functions]
    used_rows = [np.array(sorted(cache.hash_to_row[h] for h in hashes_by_function[fn]))
                 for cache, fn in zip(caches, functions)]
    n_sites_a = caches[0].codes.shape[1]
    n_columns = n_sites_a + caches[1].codes.shape[1]

    # keep_default_na: the `protein` column holds the literal string NA (Neuraminidase).
    importance = pd.read_csv(args.importance_csv, keep_default_na=False, na_values=[''])
    ranked = importance.sort_values('shap_rank')['column'].to_numpy()
    sizes = sorted({min(int(n), n_columns) for n in args.n_sites} | {n_columns})

    pair_tables = {}
    for fold in range(args.n_folds):
        pair_tables[fold] = {split: pd.read_parquet(
            args.dataset_dir / f'fold_{fold}' / f'{split}_pairs.parquet')
            for split in ('train', 'val', 'test')}

    n_fits = args.n_folds * (1 + len(args.modes) * len(sizes) * 2)
    print(f"Columns: {n_columns:,} ({caches[0].protein} {n_sites_a}, "
          f"{caches[1].protein} {n_columns - n_sites_a})")
    print(f"Set sizes: {sizes}; modes: {args.modes}; arms: top, random")
    print(f"{n_fits:,} refits ahead\n")

    def build(cache_pair, fold):
        return {split: get_site_pair_features(pair_tables[fold][split], cache_pair[0],
                                              cache_pair[1], 'ordinal')
                for split in ('train', 'val', 'test')}

    rows, started = [], time.time()
    for fold in range(args.n_folds):
        clean_matrices = build(caches, fold)
        clean = fit_and_score(config, fit_seed, clean_matrices, n_columns)
        signal = clean - 0.5
        print(f"fold {fold}: clean test AUC {clean:.4f}")
        rows.append({'mode': 'none', 'arm': 'none', 'n_sites': 0, 'fold': fold,
                     'auc': clean, 'clean_auc': clean, 'signal_lost': 0.0})

        rng = np.random.default_rng(args.seed + 1000 * fold)
        for mode in args.modes:
            for n in sizes:
                for arm in ('top', 'random'):
                    chosen = (ranked[:n] if arm == 'top'
                              else rng.choice(n_columns, size=n, replace=False))
                    if mode == 'sequence':
                        corrupted = corrupt_caches(caches, used_rows,
                                                   split_columns(chosen, n_sites_a), rng)
                        matrices = build(corrupted, fold)
                    else:
                        matrices = {split: (corrupt_rows(X, chosen, rng), y)
                                    for split, (X, y) in clean_matrices.items()}
                    auc = fit_and_score(config, fit_seed, matrices, n_columns)
                    rows.append({'mode': mode, 'arm': arm, 'n_sites': n, 'fold': fold,
                                 'auc': auc, 'clean_auc': clean,
                                 'signal_lost': (clean - auc) / signal})
                    print(f"  {mode:8s} {arm:6s} N={n:>5,d}  AUC {auc:.4f}  "
                          f"lost {(clean - auc) / signal:+.3f}  "
                          f"[{time.time() - started:.0f}s]")
                    if n == n_columns:
                        break  # every column corrupted: the two arms are the same set

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f'site_retrain_ablation_{args.unit}.csv'
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    ablated = table[table['mode'] != 'none']
    summary = (ablated.groupby(['mode', 'arm', 'n_sites'])['signal_lost']
               .agg(['mean', 'std']).reset_index())
    print("\nShare of the signal lost after refitting, (clean AUC - AUC) / (clean AUC - 0.5):")
    for mode in args.modes:
        print(f"\n  {mode} corruption")
        print(f"    {'N':>6s} {'top':>18s} {'random':>18s}")
        for n in sizes:
            cells = []
            for arm in ('top', 'random'):
                row = summary[(summary['mode'] == mode) & (summary.arm == arm) &
                              (summary.n_sites == n)]
                cells.append(f"{row['mean'].iloc[0]:.3f} +/- {row['std'].iloc[0]:.3f}"
                             if len(row) else '-')
            print(f"    {n:>6,d} {cells[0]:>18s} {cells[1]:>18s}")

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for mode in args.modes:
        for arm in ('top', 'random'):
            part = summary[(summary['mode'] == mode) & (summary.arm == arm)].sort_values('n_sites')
            ax.errorbar(part['n_sites'], part['mean'], yerr=part['std'],
                        color=MODE_COLOR[mode], linestyle=ARM_STYLE[arm], marker='o',
                        markersize=5, markeredgecolor=MARKER_EDGE, markeredgewidth=0.6,
                        capsize=3, linewidth=1.3, label=f'{mode} corruption, {arm} sites')
    ax.axhline(1.0, color=MARKER_EDGE, linewidth=0.8, linestyle=':')
    ax.annotate('all signal lost (AUC 0.5)', (sizes[0], 1.0), textcoords='offset points',
                xytext=(2, 4), fontsize=8, color=MARKER_EDGE)
    ax.set_xscale('log')
    ax.set_xlabel('number of sites corrupted before refitting')
    ax.set_ylabel('share of the signal lost')
    ax.set_title(f'{args.config_bundle}\nrefit on corrupted features, {args.n_folds} folds')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.text(0.995, 0.002, f'src/analysis/{Path(__file__).name}', ha='right', va='bottom',
             fontsize=7, color='0.45')
    out_png = savefig(args.out_dir / f'site_retrain_ablation_{args.unit}.png', dpi=args.dpi)
    print(f"\nDone. Wrote {out_png}")


if __name__ == '__main__':
    main()
