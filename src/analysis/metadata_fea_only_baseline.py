"""Metadata-feature-only baseline for the pair co-occurrence task (the "head-on" shortcut test).

Trains an LGBM on ONLY the pair's isolate metadata -- host / subtype / year of each side -- with NO
sequence features, on the exact same fold datasets and splits as the sequence LGBM. The gap
(sequence model minus this baseline) is the signal the model gets from sequence beyond metadata.

Why this is the right control for **demographic shortcut leakage** (the canonical leakage name; see
docs/plans/2026-05-07_leakage_diagnostics_plan.md): a POSITIVE pair is two proteins from the SAME
isolate, so its host/subtype/year always match; a NEGATIVE (cross-isolate) may or may not. So this
model learns "does the metadata align -> co-occur", and (by construction) fails on the
host_subtype_year regime -- it measures the ceiling reachable by metadata matching alone. It is also
threshold-independent (metadata does not depend on the clustering t), so it is a flat reference the
sequence curves can be compared against per t.

Features are JOINED per isolate from `load_flu_metadata`, keyed on the pair's `assembly_id_a` /
`assembly_id_b`, rather than read from metadata columns in the pair CSV. Those columns exist only
in the v2 / 1D-CD output -- `dataset_pairs_cc.py` writes neither them nor `neg_regime`, so reading
them would restrict this baseline to one builder. Joining keeps it usable on any fold directory,
since `label` and the two assembly ids are all it needs.

  - one-hot: host_a, host_b, hn_subtype_a, hn_subtype_b
  - numeric: year_a, year_b, year_diff = |year_a - year_b|
  - match flags, derived here: same_host, same_hn_subtype, same_year

`build_features` takes a pair frame and returns a row-aligned matrix, so a model over sequence AND
metadata can hstack it onto the k-mer features for the same rows.

Output mirrors the sequence baseline so `score_vs_threshold.py` can plot seq-vs-metadata directly:
  {out_root}/{run_prefix}_{tXXX}{arm_suffix}_fold{f}/test_predicted.csv  (label, pred_prob, pred_label)
                                                    /metrics_summary.json

CLI:
    python -m src.analysis.metadata_fea_only_baseline \\
        --dataset_prefix dataset_1dcd_nt_cds_cm0_slota \\
        --run_prefix     meta_1dcd_cm0_slota \\
        --thresholds t099 t098 t097 t096 t095
    # paired arms: the suffix sits after the threshold in both the dataset dir and the run dir,
    # matching how the sequence runs are named (..._t099_random_fold0).
    python -m src.analysis.metadata_fea_only_baseline \\
        --dataset_prefix dataset_cc_nt_cds_cm0_h3n2 --run_prefix meta_cc_nt_cds_cm0_h3n2 \\
        --thresholds t099 --n_folds 3 --arm_suffix _random
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.metadata_enrichment import load_flu_metadata  # noqa: E402

_FIELDS = ('host', 'hn_subtype', 'year')      # per-isolate axes; each yields _a, _b and same_*
_ONE_HOT = ('host', 'hn_subtype')             # the unordered ones; year stays numeric
_NEEDED = ['label', 'assembly_id_a', 'assembly_id_b']
# assembly ids are read as text: they are opaque tokens, and a numeric-looking one would otherwise
# be parsed as a float and stop matching the metadata table.
_READ_KW = dict(dtype=str, keep_default_na=False, na_values=[''])


def isolate_metadata_maps() -> dict:
    """`{field: {assembly_id: value}}` for the axes the baseline uses.

    Built once per run and passed to `build_features`, which needs a per-side lookup on two
    columns rather than the single-column merge `attach_isolate_metadata` performs.

    Returns:
        Dict keyed by `_FIELDS`, each an assembly_id -> value mapping.
    """
    meta = load_flu_metadata().drop_duplicates('assembly_id')
    meta['assembly_id'] = meta['assembly_id'].astype(str)
    return {f: dict(zip(meta['assembly_id'], meta[f])) for f in _FIELDS}


def build_features(df: pd.DataFrame, maps: dict, ref_cols=None) -> pd.DataFrame:
    """Metadata-only feature matrix, row-aligned to `df`.

    Args:
        df: pair rows carrying `assembly_id_a` and `assembly_id_b`.
        maps: from `isolate_metadata_maps()`.
        ref_cols: the train split's columns; val/test are reindexed onto them so a category
            unseen in train maps to all-zero instead of shifting the column set.

    Returns:
        float32 frame; unknown or blank values become the sentinel -1.0.
    """
    side = {f'{f}_{s}': df[f'assembly_id_{s}'].map(maps[f]).reset_index(drop=True)
            for f in _FIELDS for s in ('a', 'b')}

    ya, yb = (pd.to_numeric(side['year_a'], errors='coerce'),
              pd.to_numeric(side['year_b'], errors='coerce'))
    num = pd.DataFrame({'year_a': ya, 'year_b': yb, 'year_diff': (ya - yb).abs()})
    # Derived from the joined values, so no parsing of however a CSV spelled the flag. An unknown
    # value never equals itself, so a pair missing that axis reads as NOT matching -- true of both
    # sides of a positive, whose two ids are the same isolate (~0.4% of rows on flu July_2025).
    match = pd.DataFrame({f'same_{f}': (side[f'{f}_a'] == side[f'{f}_b']).astype(int)
                          for f in _FIELDS})
    cat = pd.get_dummies(  # get_dummies prefixes each column with its own name
        pd.DataFrame({f'{f}_{s}': side[f'{f}_{s}'].astype(str)
                      for f in _ONE_HOT for s in ('a', 'b')}), prefix_sep='=')

    X = pd.concat([cat, num, match], axis=1).fillna(-1.0)
    if ref_cols is not None:
        X = X.reindex(columns=ref_cols, fill_value=0)
    return X.astype(np.float32)


def _metrics(y, prob, yh) -> dict:
    return {
        'aucpr': float(average_precision_score(y, prob)),
        'aucroc': float(roc_auc_score(y, prob)),
        'f1': float(f1_score(y, yh)),
        'f1_macro': float(f1_score(y, yh, average='macro')),
        'mcc': float(matthews_corrcoef(y, yh)),
        'brier': float(brier_score_loss(y, prob)),
    }


def _best_threshold(y_val, p_val) -> float:
    """Pick the decision threshold that maximizes macro-F1 on val (mirrors the seq baseline's
    tuned-threshold step; keeps threshold-dependent metrics comparable)."""
    grid = np.linspace(0.05, 0.95, 19)
    return float(max(grid, key=lambda thr: f1_score(y_val, (p_val >= thr).astype(int), average='macro')))


def run_fold(dataset_dir: Path, out_dir: Path, seed: int, maps: dict) -> dict | None:
    """Fit on the fold's train split, tune the threshold on val, score test; None if a CSV is missing."""
    splits = {}
    for name in ('train', 'val', 'test'):
        path = dataset_dir / f'{name}_pairs.csv'
        if not path.exists():
            print(f'  SKIP {dataset_dir} (missing {path.name})')
            return None
        df = pd.read_csv(path, usecols=_NEEDED, **_READ_KW)
        df['label'] = df['label'].astype(int)
        splits[name] = df
    tr, va, te = splits['train'], splits['val'], splits['test']

    Xtr = build_features(tr, maps)
    cols = list(Xtr.columns)
    Xva = build_features(va, maps, ref_cols=cols)
    Xte = build_features(te, maps, ref_cols=cols)

    # Pass numpy (columns are already position-aligned by reindex) so LightGBM does not choke on
    # one-hot names that collide after it sanitizes spaces/special chars in category values.
    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        n_jobs=16, random_state=seed, verbose=-1
    )
    clf.fit(Xtr.values, tr['label'])

    p_val = clf.predict_proba(Xva.values)[:, 1]
    thr = _best_threshold(va['label'].to_numpy(), p_val)
    p_te = clf.predict_proba(Xte.values)[:, 1]
    yh = (p_te >= thr).astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'label': te['label'], 'pred_prob': p_te, 'pred_label': yh}).to_csv(
        out_dir / 'test_predicted.csv', index=False)
    m = _metrics(te['label'], p_te, yh)
    m.update({'threshold': thr, 'n_test': int(len(te)), 'n_features': len(cols)})
    (out_dir / 'metrics_summary.json').write_text(json.dumps(m, indent=2))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--datasets_root', default=str(PROJ / 'data/datasets/flu/July_2025/runs'))
    ap.add_argument('--dataset_prefix', required=True,
        help='Dataset dirs are {dataset_prefix}_{tXXX}/fold_{f} (e.g. dataset_1dcd_nt_cds_cm0_slota).'
    )
    ap.add_argument('--out_root', default=str(PROJ / 'models/flu/July_2025/runs'))
    ap.add_argument('--run_prefix', required=True,
        help='Output run dirs are {run_prefix}_{tXXX}_fold{f} (e.g. meta_1dcd_cm0_slota).'
    )
    ap.add_argument('--thresholds', nargs='+', required=True)
    ap.add_argument('--n_folds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--arm_suffix', default='',
                    help="Appended after the threshold in BOTH the dataset dir and the run dir "
                         "(e.g. '_random' for a paired control arm); default none.")
    args = ap.parse_args()

    maps = isolate_metadata_maps()
    for t in args.thresholds:
        for f in range(args.n_folds):
            ddir = Path(args.datasets_root) / f'{args.dataset_prefix}_{t}{args.arm_suffix}' / f'fold_{f}'
            odir = Path(args.out_root) / f'{args.run_prefix}_{t}{args.arm_suffix}_fold{f}'
            m = run_fold(ddir, odir, args.seed, maps)
            if m is not None:
                print(f'  {t} fold{f}: AUC-PR={m["aucpr"]:.3f}  MCC={m["mcc"]:.3f}  '
                      f'F1_macro={m["f1_macro"]:.3f}  (thr={m["threshold"]:.2f}) -> {odir.name}')
    print('Done.')


if __name__ == '__main__':
    main()
