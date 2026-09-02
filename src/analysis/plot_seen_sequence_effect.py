"""Does the model score better on test pairs whose sequences it saw during training?

The memorisation check the permutation and ablation passes cannot make. Those corrupt features and
watch the score move; this one leaves the model alone and splits the test rows instead, by whether
each slot's sequence also appears in that fold's training split.

Under a random split some sequences recur across splits, and a per-site feature vector nearly names
the sequence it came from, so the model could score well by recalling which HA went with which NA
rather than by learning anything general. If that were happening, rows whose sequences the model
has already met would score much better than rows built from sequences it has never seen. If the
two score alike, recall is not what is carrying the result.

The pairs themselves are never repeated -- `pair_key` overlap between train and test is 0 in every
fold -- so this is about the sequences, not the answers.

Note what recall would actually buy. Having seen HA_x paired with NA_y in training helps reject a
test NEGATIVE that pairs HA_x with something else, but it works against a test POSITIVE where HA_x
has a different true partner. So the effect need not be positive, which is a reason to read the
per-stratum AUC rather than assume a direction.

Reads the saved `test_predicted.csv` from each run rather than rebuilding features, so any feature
source can be compared on the same rows -- the k-mer baseline is included because the risk this
checks is specific to per-site features, and k-mers give the contrast.

Outputs (to `--out_dir`, by default derived from the dataset dir):
    seen_sequence_effect.png   test AUC-ROC per exposure stratum, per feature source
    seen_sequence_effect.csv   arm, fold, stratum, n, n_pos, auc

CLI:
    python -m src.analysis.plot_seen_sequence_effect \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_ha_na_h3n2_2024_random_cv4_pinned_length
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402

# Ordered from least to most exposure, which is how the bars should read left to right.
STRATA = ('neither seen', 'slot a seen', 'slot b seen', 'both seen')
ARM_COLOR = ['#4C7CAB', '#CF8793', '#7FA98C']
MARKER_EDGE = '#222222'


def exposure_stratum(test: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Label each test row by which of its two sequences also appear in training.

    Args:
      test: test pair table with `cds_dna_hash_a` and `cds_dna_hash_b`.
      train: the same fold's training pair table.

    Returns:
      One label per test row, from `STRATA`.
    """
    seen_a = test['cds_dna_hash_a'].isin(set(train['cds_dna_hash_a']))
    seen_b = test['cds_dna_hash_b'].isin(set(train['cds_dna_hash_b']))
    labels = np.where(seen_a & seen_b, STRATA[3],
                      np.where(seen_a, STRATA[1],
                               np.where(seen_b, STRATA[2], STRATA[0])))
    return pd.Series(labels, index=test.index)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset_dir', type=Path, required=True)
    p.add_argument('--arms', nargs='+', default=[
        'lgbm_ha_na_h3n2_2024_random_cv4_pinned_length',
        'lgbm_ha_na_h3n2_2024_random_cv4_site_nt',
        'lgbm_ha_na_h3n2_2024_random_cv4_site_codon'],
        help='run dir names minus the _fold{k} suffix')
    p.add_argument('--arm_labels', nargs='+', default=['k-mer k=6', 'per-site nt', 'per-site codon'])
    p.add_argument('--models_root', type=Path, default=PROJ / 'models/flu/July_2025/runs')
    p.add_argument('--n_folds', type=int, default=4)
    p.add_argument('--out_dir', type=Path, default=None)
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    if len(args.arm_labels) != len(args.arms):
        raise ValueError(
            f"--arm_labels has {len(args.arm_labels)} entries for {len(args.arms)} arms.")
    if args.out_dir is None:
        parts = args.dataset_dir.resolve().parts
        i = parts.index('datasets')
        args.out_dir = (PROJ / 'results' / parts[i + 1] / parts[i + 2] / parts[-1] /
                        'seen_sequence_effect')

    strata_by_fold = {}
    for fold in range(args.n_folds):
        fold_dir = args.dataset_dir / f'fold_{fold}'
        train = pd.read_parquet(fold_dir / 'train_pairs.parquet',
                                columns=['cds_dna_hash_a', 'cds_dna_hash_b', 'pair_key'])
        test = pd.read_parquet(fold_dir / 'test_pairs.parquet',
                               columns=['cds_dna_hash_a', 'cds_dna_hash_b', 'pair_key', 'label'])
        shared_pairs = len(set(train['pair_key']) & set(test['pair_key']))
        if shared_pairs:
            raise ValueError(
                f"fold {fold}: {shared_pairs:,} pair_keys appear in both train and test. This "
                f"check assumes the pairs themselves are never repeated.")
        strata_by_fold[fold] = exposure_stratum(test, train)
        counts = strata_by_fold[fold].value_counts()
        print(f"fold {fold}: {len(test):,} test rows; " +
              ', '.join(f"{s} {int(counts.get(s, 0)):,}" for s in STRATA))

    rows = []
    for arm, label in zip(args.arms, args.arm_labels):
        for fold in range(args.n_folds):
            pred_path = args.models_root / f'{arm}_fold{fold}' / 'test_predicted.csv'
            if not pred_path.exists():
                raise FileNotFoundError(f"missing {pred_path}. Train {arm} fold {fold} first.")
            pred = pd.read_csv(pred_path, usecols=['label', 'pred_prob'])
            strata = strata_by_fold[fold]
            if len(pred) != len(strata):
                raise ValueError(
                    f"{pred_path} has {len(pred):,} rows but fold {fold}'s test split has "
                    f"{len(strata):,}. The predictions and the dataset are out of step.")
            for stratum in STRATA + ('all',):
                mask = np.ones(len(pred), dtype=bool) if stratum == 'all' \
                    else (strata == stratum).to_numpy()
                y, prob = pred['label'].to_numpy()[mask], pred['pred_prob'].to_numpy()[mask]
                # AUC needs both classes; a stratum can be one-sided when it is small.
                auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float('nan')
                rows.append({'arm': label, 'fold': fold, 'stratum': stratum, 'n': int(mask.sum()),
                             'n_pos': int(y.sum()), 'auc': auc})

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / 'seen_sequence_effect.csv'
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    summary = table.groupby(['arm', 'stratum']).agg(
        n=('n', 'sum'), n_pos=('n_pos', 'sum'),
        auc_mean=('auc', 'mean'), auc_std=('auc', 'std')).reset_index()

    print("\nTest AUC-ROC by how much of the pair the model met during training:")
    header = f"  {'arm':16s} " + ' '.join(f'{s:>20s}' for s in ('all',) + STRATA)
    print(header)
    for label in args.arm_labels:
        cells = []
        for stratum in ('all',) + STRATA:
            row = summary[(summary.arm == label) & (summary.stratum == stratum)]
            cells.append(f"{row.auc_mean.iloc[0]:.4f} +/- {row.auc_std.iloc[0]:.4f}"
                         if len(row) else '-')
        print(f"  {label:16s} " + ' '.join(f'{c:>20s}' for c in cells))
    counts = summary[summary.arm == args.arm_labels[0]].set_index('stratum')
    print(f"  {'rows (all folds)':16s} " +
          ' '.join(f"{int(counts.loc[s, 'n']):>20,d}" for s in ('all',) + STRATA))
    print(f"  {'positive rate':16s} " +
          ' '.join(f"{counts.loc[s, 'n_pos'] / counts.loc[s, 'n']:>20.3f}"
                   for s in ('all',) + STRATA))

    print("\n  'both seen' minus 'neither seen', the memorisation gap:")
    for label in args.arm_labels:
        both = summary[(summary.arm == label) & (summary.stratum == STRATA[3])].auc_mean.iloc[0]
        neither = summary[(summary.arm == label) & (summary.stratum == STRATA[0])].auc_mean.iloc[0]
        print(f"    {label:16s} {both - neither:+.4f}")

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(STRATA))
    width = 0.8 / len(args.arm_labels)
    for i, label in enumerate(args.arm_labels):
        part = summary[summary.arm == label].set_index('stratum').loc[list(STRATA)]
        ax.bar(x + (i - (len(args.arm_labels) - 1) / 2) * width, part['auc_mean'], width,
               yerr=part['auc_std'], label=label, color=ARM_COLOR[i % len(ARM_COLOR)],
               edgecolor=MARKER_EDGE, linewidth=0.7,
               error_kw={'ecolor': MARKER_EDGE, 'elinewidth': 0.9, 'capsize': 3})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({int(counts.loc[s, 'n']):,} rows)" for s in STRATA])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('test AUC-ROC')
    ax.set_xlabel("how much of the test pair the model met during training")
    ax.set_title('Does having seen a sequence in training help at test time?\n'
                 f'{args.dataset_dir.name}, {args.n_folds} folds')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.text(0.995, 0.002, f'src/analysis/{Path(__file__).name}', ha='right', va='bottom',
             fontsize=7, color='0.45')
    out_png = savefig(args.out_dir / 'seen_sequence_effect.png', dpi=args.dpi)
    print(f"\nDone. Wrote {out_png}")


if __name__ == '__main__':
    main()
