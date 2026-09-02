"""Shuffle the top N sites together and see how much of the model's signal goes with them.

Step 7b(ii) of docs/plans/2026-08-28_per_site_nt_features_plan.md, the group version of the
single-site permutation in `plot_site_importance.py`. Nothing is retrained: the fitted model
re-predicts on a matrix where N whole columns have been scrambled.

Why a group rather than one site at a time. Permuting one feature understates it when another
feature carries the same information -- the model falls back on the twin and the drop looks small.
Shuffling a set together removes that escape route. It also splits a gap the single-site pass
leaves open: every single-site cost added up reaches only 55.5% of the signal, and comparing the
group drop at N against the sum of those N individual drops says whether the missing part is
redundancy INSIDE the top set or spread across everything else.

What it cannot say is whether a NEW model could do the job without those positions. This model has
already committed to using them, so scrambling gives it no chance to find alternatives among the
other ~990 sites. Only refitting answers that.

Two arms at every N. `top` takes the N best sites by SHAP; `random` takes N at random and is the
control -- without it, a drop after corrupting the top 10 cannot be told from "corrupting any 10
columns hurts". Across repeats the top arm holds its site set fixed and varies only the shuffle,
while the random arm redraws the sites as well, so the control covers several site choices rather
than one unlucky draw. At the largest N every column is shuffled, so the two arms meet -- and both
must land at AUC-ROC 0.5. That point is the validity anchor: if it misses, nothing else on the
curve is trustworthy.

Also run on the TRAINING split, as a contrast. If shuffling the top N costs much more on train
than on test, the model fitted those positions to training-specific detail, which is what
memorisation would look like.

Drops are reported as a share of the signal, `(clean AUC - shuffled AUC) / (clean AUC - 0.5)`, so
train and test sit on one scale despite different clean scores.

A constant-fill variant runs at a few N as a spot check. Filling a column with its most common
value is the other way to disable it; shuffling is preferred because it keeps every value
plausible for that column, but for a tree the difference may not matter and this shows whether it
does rather than arguing about it.

Outputs (to `--out_dir`, by default derived from the dataset dir):
    site_group_permutation_{unit}.png   share of signal lost against N, per arm and split
    site_group_permutation_{unit}.csv   split, arm, n_sites, fold, repeat, auc, clean_auc, ...

CLI:
    python -m src.analysis.plot_site_group_permutation \\
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
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.plot_site_entropy import collect_slot_hashes  # noqa: E402
from src.utils.config_hydra import get_function_short_name_map, get_virus_config_hydra  # noqa: E402
from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402
from src.utils.site_utils import get_site_pair_features, load_site_cache  # noqa: E402

# Sampled from tmp/score/h3n2_f1_macro_within_fold.png, so the per-site figures read as one series.
ARM_COLOR = {'top': '#CF8793', 'random': '#4C7CAB'}
SPLIT_STYLE = {'test': '-', 'train': '--'}
MARKER_EDGE = '#222222'


def shuffled_auc(booster, X: np.ndarray, y: np.ndarray, columns: np.ndarray,
                 rng: np.random.Generator) -> float:
    """AUC-ROC after independently shuffling each of `columns` among the rows.

    Each column gets its own permutation, so the set is disabled as a set rather than moved around
    together. Row-level is right here because the model is fixed and predicts one row at a time.

    Args:
      booster: the fitted LightGBM booster.
      X: feature matrix to corrupt a copy of.
      y: labels for those rows.
      columns: column indices to shuffle.
      rng: draws the permutations.

    Returns:
      AUC-ROC of the same model on the corrupted matrix.
    """
    scrambled = X.copy()
    for column in columns:
        scrambled[:, column] = rng.permutation(scrambled[:, column])
    return roc_auc_score(y, booster.predict(scrambled))


def constant_filled_auc(booster, X: np.ndarray, y: np.ndarray, columns: np.ndarray) -> float:
    """AUC-ROC after replacing each of `columns` with its most common value.

    The other way to disable a feature. The most common value is used rather than zero so the rows
    stay somewhere the column actually goes.

    Args:
      booster: the fitted LightGBM booster.
      X: feature matrix to corrupt a copy of.
      y: labels for those rows.
      columns: column indices to overwrite.

    Returns:
      AUC-ROC of the same model on the corrupted matrix.
    """
    filled = X.copy()
    for column in columns:
        values, counts = np.unique(filled[:, column], return_counts=True)
        filled[:, column] = values[counts.argmax()]
    return roc_auc_score(y, booster.predict(filled))


def n_grid(n_columns: int, requested: list[int]) -> list[int]:
    """Clip the requested set sizes to the columns available and always include all of them.

    The all-columns point is the validity anchor: shuffling everything must give AUC-ROC 0.5.

    Args:
      n_columns: how many columns the model has.
      requested: the set sizes asked for.

    Returns:
      Sorted unique sizes, ending at `n_columns`.
    """
    return sorted({min(int(n), n_columns) for n in requested} | {n_columns})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model_run_template', required=True,
                   help='run dir name minus the _fold{k} suffix')
    p.add_argument('--dataset_dir', type=Path, required=True)
    p.add_argument('--unit', default='codon', choices=['nt', 'codon', 'aa'])
    p.add_argument('--config_bundle', default='flu_ha_na_h3n2_2024_random_cv4_site_codon')
    p.add_argument('--site_dir', type=Path, default=PROJ / 'data/embeddings/flu/July_2025')
    p.add_argument('--models_root', type=Path, default=PROJ / 'models/flu/July_2025/runs')
    p.add_argument('--importance_csv', type=Path, default=None,
                   help='site_importance_{unit}.csv; default sits beside --out_dir')
    p.add_argument('--n_folds', type=int, default=4)
    p.add_argument('--n_sites', type=int, nargs='+',
                   default=[1, 2, 5, 10, 20, 50, 100, 200, 500],
                   help='set sizes to shuffle; the all-columns anchor is added automatically')
    p.add_argument('--repeats', type=int, default=5,
                   help='shuffles per point; the random arm redraws its sites each time too')
    p.add_argument('--splits', nargs='+', default=['test', 'train'], choices=['train', 'val', 'test'])
    p.add_argument('--constant_fill_at', type=int, nargs='*', default=[10, 50],
                   help='set sizes to also disable by constant fill, as a spot check')
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
    hashes_by_function = collect_slot_hashes(args.dataset_dir, args.n_folds)
    functions = sorted(hashes_by_function, key=protein_order.index)
    caches = [load_site_cache(args.site_dir, args.unit, function_to_short.get(f, f))
              for f in functions]

    if not args.importance_csv.exists():
        raise FileNotFoundError(
            f"missing {args.importance_csv}. Run `python -m src.analysis.plot_site_importance` "
            f"first -- the top-N arm needs its SHAP ranking.")
    # keep_default_na: the `protein` column holds the literal string NA (Neuraminidase), which a
    # default read turns into NaN and drops.
    importance = pd.read_csv(args.importance_csv, keep_default_na=False, na_values=[''])
    ranked_columns = importance.sort_values('shap_rank')['column'].to_numpy()

    sizes = n_grid(len(ranked_columns), args.n_sites)
    print(f"Set sizes: {sizes}")
    print("Arms: top (fixed set, shuffle varies) and random (set and shuffle both vary)")
    print(f"Splits: {args.splits}; {args.repeats} repeats; {args.n_folds} folds")

    rows = []
    for fold in range(args.n_folds):
        booster = joblib.load(
            args.models_root / f'{args.model_run_template}_fold{fold}' / 'best_model.joblib'
        ).booster_
        for split in args.splits:
            pairs = pd.read_parquet(args.dataset_dir / f'fold_{fold}' / f'{split}_pairs.parquet')
            X, y = get_site_pair_features(pairs, caches[0], caches[1], 'ordinal')
            clean = roc_auc_score(y, booster.predict(X))
            signal = clean - 0.5
            print(f"  fold {fold} {split}: {X.shape[0]:,} rows, clean AUC {clean:.4f}")

            rng = np.random.default_rng(args.seed + 1000 * fold)
            for n in sizes:
                for arm in ('top', 'random'):
                    for repeat in range(args.repeats):
                        if arm == 'top':
                            chosen = ranked_columns[:n]
                        else:
                            chosen = rng.choice(len(ranked_columns), size=n, replace=False)
                        auc = shuffled_auc(booster, X, y, chosen, rng)
                        rows.append({'split': split, 'arm': arm, 'method': 'shuffle',
                                     'n_sites': n, 'fold': fold, 'repeat': repeat,
                                     'auc': auc, 'clean_auc': clean,
                                     'signal_lost': (clean - auc) / signal})
                        if arm == 'top' and n == len(ranked_columns):
                            break  # every column shuffled: the two arms are the same set
            for n in args.constant_fill_at:
                if n > len(ranked_columns):
                    continue
                auc = constant_filled_auc(booster, X, y, ranked_columns[:n])
                rows.append({'split': split, 'arm': 'top', 'method': 'constant',
                             'n_sites': n, 'fold': fold, 'repeat': 0,
                             'auc': auc, 'clean_auc': clean,
                             'signal_lost': (clean - auc) / signal})

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f'site_group_permutation_{args.unit}.csv'
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(table):,} measurements)")

    shuffled = table[table['method'] == 'shuffle']
    summary = (shuffled.groupby(['split', 'arm', 'n_sites'])['signal_lost']
               .agg(['mean', 'std']).reset_index())
    print("\nShare of the signal lost, (clean AUC - shuffled AUC) / (clean AUC - 0.5):")
    for split in args.splits:
        print(f"\n  {split}")
        print(f"    {'N':>6s} {'top':>18s} {'random':>18s}")
        for n in sizes:
            cells = []
            for arm in ('top', 'random'):
                row = summary[(summary.split == split) & (summary.arm == arm) &
                              (summary.n_sites == n)]
                cells.append(f"{row['mean'].iloc[0]:.3f} +/- {row['std'].iloc[0]:.3f}"
                             if len(row) else '-')
            print(f"    {n:>6,d} {cells[0]:>18s} {cells[1]:>18s}")

    constants = table[table['method'] == 'constant']
    if not constants.empty:
        print("\n  constant fill against shuffle, top arm:")
        for split in args.splits:
            for n in sorted(constants['n_sites'].unique()):
                c = constants[(constants.split == split) & (constants.n_sites == n)]['signal_lost']
                s = shuffled[(shuffled.split == split) & (shuffled.arm == 'top') &
                             (shuffled.n_sites == n)]['signal_lost']
                print(f"    {split:5s} N={n:>4d}  constant {c.mean():.3f}  shuffle {s.mean():.3f}"
                      f"  difference {c.mean() - s.mean():+.3f}")

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for split in args.splits:
        for arm in ('top', 'random'):
            part = summary[(summary.split == split) & (summary.arm == arm)].sort_values('n_sites')
            ax.errorbar(part['n_sites'], part['mean'], yerr=part['std'],
                        color=ARM_COLOR[arm], linestyle=SPLIT_STYLE[split], marker='o',
                        markersize=5, markeredgecolor=MARKER_EDGE, markeredgewidth=0.6,
                        capsize=3, linewidth=1.3, label=f'{arm} sites, {split}')
    ax.axhline(1.0, color=MARKER_EDGE, linewidth=0.8, linestyle=':')
    ax.annotate('all signal lost (AUC 0.5)', (sizes[0], 1.0), textcoords='offset points',
                xytext=(2, 4), fontsize=8, color=MARKER_EDGE)
    ax.set_xscale('log')
    ax.set_xlabel('number of sites shuffled together')
    ax.set_ylabel('share of the signal lost')
    ax.set_title(f'{args.model_run_template}\ngroup permutation, no retraining, '
                 f'{args.n_folds} folds x {args.repeats} repeats')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.text(0.995, 0.002, f'src/analysis/{Path(__file__).name}', ha='right', va='bottom',
             fontsize=7, color='0.45')
    out_png = savefig(args.out_dir / f'site_group_permutation_{args.unit}.png', dpi=args.dpi)
    print(f"\nDone. Wrote {out_png}")


if __name__ == '__main__':
    main()
