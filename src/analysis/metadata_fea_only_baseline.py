"""Demographic-only baseline for the pair co-occurrence task (the "head-on" shortcut test).

Trains an LGBM on ONLY the pair's isolate metadata -- host / subtype / year of each side -- with NO
sequence features, on the exact same fold datasets and splits as the sequence LGBM. The gap
(sequence model minus this baseline) is the signal the model gets from sequence beyond demographics.

Why this is the right control: a POSITIVE pair is two proteins from the SAME isolate, so its
host/subtype/year always match; a NEGATIVE (cross-isolate) may or may not. So this model learns
"do the demographics align -> co-occur", and (by construction) fails on the host_subtype_year
regime -- it measures the ceiling reachable by demographic matching alone. It is also
threshold-independent (metadata does not depend on the clustering t), so it is a flat reference the
sequence curves can be compared against per t.

Features (all read straight from the pair CSVs -- no join needed):
  - one-hot: host_a, host_b, hn_subtype_a, hn_subtype_b
  - numeric: year_a, year_b, year_diff = |year_a - year_b|
  - match flags (already in the CSV): same_host, same_hn_subtype, same_year

Output mirrors the sequence baseline so `score_vs_threshold.py` can plot seq-vs-demo directly:
  {out_root}/{run_prefix}_{tXXX}_fold{f}/test_predicted.csv   (label, pred_prob, pred_label)
                                        /metrics_summary.json

CLI:
    python -m src.analysis.metadata_fea_only_baseline \\
        --dataset_prefix dataset_1dcd_nt_cds_cm0_slota \\
        --run_prefix     demo_1dcd_cm0_slota \\
        --thresholds t099 t098 t097 t096 t095
"""
from __future__ import annotations

import argparse
import json
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

_CAT = ['host_a', 'host_b', 'hn_subtype_a', 'hn_subtype_b']       # one-hot
_MATCH = ['same_host', 'same_hn_subtype', 'same_year']            # 0/1 match flags (already in CSV)
_NEEDED = ['label', 'year_a', 'year_b'] + _CAT + _MATCH
# 'NA' is a real hn_subtype-free value here (subtypes are H1N1/H3N2/...), but read defensively so no
# metadata token is silently turned into NaN.
_READ_KW = dict(keep_default_na=False, na_values=[''])


def build_features(df: pd.DataFrame, ref_cols=None) -> pd.DataFrame:
    """Metadata-only feature matrix. `ref_cols` (from the train split) aligns val/test one-hot
    columns so unseen categories map to all-zero and the column set matches train exactly."""
    ya = pd.to_numeric(df['year_a'], errors='coerce')
    yb = pd.to_numeric(df['year_b'], errors='coerce')
    num = pd.DataFrame({'year_a': ya, 'year_b': yb, 'year_diff': (ya - yb).abs()})
    match = df[_MATCH].apply(
        lambda s: s.astype(str).str.strip().str.lower().isin(['true', '1', 'yes']).astype(int))
    cat = pd.get_dummies(df[_CAT].astype(str), prefix=_CAT, prefix_sep='=')
    X = pd.concat([cat, num.reset_index(drop=True), match.reset_index(drop=True)], axis=1)
    X = X.fillna(-1.0)  # unknown/blank year -> sentinel
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


def run_fold(dataset_dir: Path, out_dir: Path, seed: int) -> dict | None:
    need = ['train_pairs.csv', 'val_pairs.csv', 'test_pairs.csv']
    if not all((dataset_dir / f).exists() for f in need):
        print(f'  SKIP {dataset_dir} (missing pair CSVs)')
        return None
    tr = pd.read_csv(dataset_dir / 'train_pairs.csv', usecols=_NEEDED, **_READ_KW)
    va = pd.read_csv(dataset_dir / 'val_pairs.csv', usecols=_NEEDED, **_READ_KW)
    te = pd.read_csv(dataset_dir / 'test_pairs.csv', usecols=_NEEDED, **_READ_KW)

    Xtr = build_features(tr)
    cols = list(Xtr.columns)
    Xva = build_features(va, ref_cols=cols)
    Xte = build_features(te, ref_cols=cols)

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
        help='Output run dirs are {run_prefix}_{tXXX}_fold{f} (e.g. demo_1dcd_cm0_slota).'
    )
    ap.add_argument('--thresholds', nargs='+', required=True)
    ap.add_argument('--n_folds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    for t in args.thresholds:
        for f in range(args.n_folds):
            ddir = Path(args.datasets_root) / f'{args.dataset_prefix}_{t}' / f'fold_{f}'
            odir = Path(args.out_root) / f'{args.run_prefix}_{t}_fold{f}'
            m = run_fold(ddir, odir, args.seed)
            if m is not None:
                print(f'  {t} fold{f}: AUC-PR={m["aucpr"]:.3f}  MCC={m["mcc"]:.3f}  '
                      f'F1_macro={m["f1_macro"]:.3f}  (thr={m["threshold"]:.2f}) -> {odir.name}')
    print('Done.')


if __name__ == '__main__':
    main()
