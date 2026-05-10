"""Cross-model heatmap of per-regime TPR / TNR (MLP + sklearn baselines).

Reads each model's ``post_hoc/level1_neg_regimes.csv`` (produced by
``analyze_stage4_train.py``) and renders a single heatmap with one row
per model and one column per regime: positive => TPR, negatives => TNR.

The MLP and the sklearn baselines write the same level1_neg_regimes.csv
schema -- the input is uniform; this script doesn't care which one is the
MLP (no special-casing). Row labels are inferred from each run dir's
training_info.json (`baseline` field for baselines; otherwise "MLP")
unless --row_labels is provided.

Usage:
    python src/analysis/aggregate_baselines_vs_mlp.py \\
        --model_dirs \\
            models/.../baseline_logistic_<bundle>_<TS>/ \\
            models/.../baseline_lgbm_<bundle>_<TS>/ \\
            models/.../baseline_knn1_margin_<bundle>_<TS>/ \\
            models/.../baseline_knn_vote_<bundle>_<TS>/ \\
            models/.../training_<bundle>_<TS>/ \\
        --row_labels "Logistic Regression,LightGBM,1-NN (margin),k-NN (k=5),MLP" \\
        --output_dir results/.../baselines_vs_mlp_<TS>/

Outputs (under --output_dir):
    baselines_vs_mlp_heatmap.png
    baselines_vs_mlp.csv

Note on featurization parity: the MLP applies a learned per-slot
LayerNorm before concat (slot_transform=slot_norm); sklearn baselines
apply each baseline's natural feature_scaling (StandardScaler for LR,
none for LightGBM / k-NN). The standardization on which the per-row
metrics are computed is therefore not identical across rows -- this is
documented in the heatmap caption.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to sys.path (kept consistent with sibling analysis scripts)
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


# Display order must match analyze_stage4_train.py's _LEVEL1_REGIME_ORDER.
# Re-stating it here (rather than importing) keeps this script independent of
# downstream API churn in analyze_stage4_train.py.
_REGIME_ORDER = (
    'positive',
    'none_match',
    'host_only',
    'subtype_only',
    'year_only',
    'host_subtype_only',
    'host_year_only',
    'subtype_year_only',
    'host_subtype_year',
    'unknown_metadata_neg',
)

# Metric per column: TPR for the positive column (single-class positives have
# no TNR), TNR for every negative regime (single-class negatives have no TPR).
_METRIC_FOR_REGIME = {r: ('tpr' if r == 'positive' else 'tnr') for r in _REGIME_ORDER}


def _infer_row_label(model_dir: Path) -> str:
    """Read training_info.json for a human-readable model label.

    Baselines write `baseline: '<name>'`; the MLP path writes no
    `baseline` key, so any run that doesn't carry one is labeled "MLP".
    Falls back to the directory name if training_info.json is missing.
    """
    info_path = model_dir / 'training_info.json'
    if not info_path.exists():
        return model_dir.name
    try:
        info = json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return model_dir.name
    baseline_name = info.get('baseline')
    if baseline_name:
        return str(baseline_name)
    return 'MLP'


def _read_one_level1(model_dir: Path) -> pd.DataFrame:
    """Read post_hoc/level1_neg_regimes.csv from a model run dir."""
    csv = model_dir / 'post_hoc' / 'level1_neg_regimes.csv'
    if not csv.exists():
        raise FileNotFoundError(
            f"Missing level1_neg_regimes.csv under {model_dir}. "
            f"Run analyze_stage4_train.py on this dir first."
        )
    df = pd.read_csv(csv)
    if 'regime' not in df.columns:
        raise ValueError(
            f"level1_neg_regimes.csv at {csv} missing 'regime' column. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def _build_matrix(model_dirs: list[Path], row_labels: list[str]) -> tuple[
    pd.DataFrame, pd.DataFrame
]:
    """Build (n_models x n_regimes) value matrix and parallel n_samples matrix.

    Each cell is `tpr` for the positive column / `tnr` for the rest. NaN
    when the regime has zero samples in that model's test split (or when
    the metric is undefined for that stratum).
    """
    values = pd.DataFrame(index=row_labels, columns=list(_REGIME_ORDER), dtype=float)
    n_samples = pd.DataFrame(index=row_labels, columns=list(_REGIME_ORDER), dtype=int)
    for label, model_dir in zip(row_labels, model_dirs):
        df = _read_one_level1(model_dir)
        by_regime = df.set_index('regime')
        for regime in _REGIME_ORDER:
            if regime not in by_regime.index:
                values.loc[label, regime] = np.nan
                n_samples.loc[label, regime] = 0
                continue
            row = by_regime.loc[regime]
            n = int(row.get('n_samples', 0))
            n_samples.loc[label, regime] = n
            if n == 0:
                values.loc[label, regime] = np.nan
                continue
            metric_col = _METRIC_FOR_REGIME[regime]
            v = row.get(metric_col, np.nan)
            values.loc[label, regime] = float(v) if pd.notna(v) else np.nan
    return values, n_samples


def _column_n_samples_consistent(n_samples: pd.DataFrame) -> pd.Series:
    """All rows should report the same n_samples per column (same test split,
    same regime classification). Return the column-wise n_samples; warn if
    rows disagree.
    """
    expected = n_samples.iloc[0]
    for label in n_samples.index[1:]:
        diffs = n_samples.loc[label] != expected
        if diffs.any():
            cols = list(n_samples.columns[diffs])
            print(
                f"WARNING: row {label!r} disagrees with row "
                f"{n_samples.index[0]!r} on n_samples for columns {cols}. "
                f"All rows should use the same Stage 3 test split."
            )
    return expected


def _render_heatmap(values: pd.DataFrame, col_n: pd.Series, output_path: Path,
                    *, caption: Optional[str] = None) -> None:
    """One heatmap, rows=models, columns=regimes; cell text = metric value.

    Colormap: 'RdYlGn' so high TPR/TNR is green and low is red. Range
    fixed to [0, 1] (both metrics live there); NaN cells stay grey.
    """
    n_rows, n_cols = values.shape
    fig_w = 1.6 + 1.0 * n_cols
    fig_h = 1.5 + 0.6 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    arr = values.values.astype(float)
    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad(color='lightgrey')
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')

    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            if np.isnan(v):
                txt = 'N/A'
                color = 'dimgrey'
            else:
                txt = f'{v:.3f}'
                # Use white text on dark cells (low values) and black on bright
                # (high values). RdYlGn at v < 0.4 is red/orange-dark; above
                # is yellow/green-light.
                color = 'white' if v < 0.4 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color=color)

    # Column headers: regime + metric + n_samples (n is the same across rows).
    col_headers = []
    for regime in values.columns:
        metric = _METRIC_FOR_REGIME[regime].upper()
        n = int(col_n[regime]) if pd.notna(col_n[regime]) else 0
        col_headers.append(f'{regime}\n{metric}\nn={n:,}')
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_headers, rotation=0, ha='center', fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(values.index, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('TPR / TNR', fontsize=9)

    ax.set_title('Per-regime TPR (positive) / TNR (negatives) — models × regimes',
                 fontsize=11)
    if caption:
        fig.text(0.5, -0.03, caption, ha='center', fontsize=8,
                 color='dimgrey', style='italic', wrap=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate per-regime TPR/TNR across MLP + baseline runs into a heatmap.'
    )
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='One or more model run directories. Each must contain '
                             'post_hoc/level1_neg_regimes.csv. Order = top-to-bottom row '
                             'order of the heatmap.')
    parser.add_argument('--row_labels', type=str, nargs='+', default=None,
                        help='Optional row labels (one arg per label; quote labels that '
                             'contain spaces). Overrides auto-inference from '
                             'training_info.json. Length must match --model_dirs.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to write baselines_vs_mlp_heatmap.png and '
                             'baselines_vs_mlp.csv.')
    parser.add_argument('--caption', type=str, default=None,
                        help='Optional caption rendered below the heatmap. If omitted, '
                             'a default caption flagging featurization parity is used.')
    args = parser.parse_args()

    model_dirs = [Path(p) for p in args.model_dirs]
    for d in model_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Model dir not found: {d}")

    if args.row_labels:
        row_labels = list(args.row_labels)
        if len(row_labels) != len(model_dirs):
            raise ValueError(
                f"--row_labels has {len(row_labels)} entries; "
                f"--model_dirs has {len(model_dirs)}."
            )
    else:
        row_labels = [_infer_row_label(d) for d in model_dirs]

    print(f'Aggregating {len(model_dirs)} model run(s):')
    for label, d in zip(row_labels, model_dirs):
        print(f'  {label:<24s}  {d}')

    values, n_samples = _build_matrix(model_dirs, row_labels)
    col_n = _column_n_samples_consistent(n_samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'baselines_vs_mlp.csv'
    out_csv = values.copy()
    out_csv.index.name = 'model'
    out_csv.to_csv(csv_path)
    print(f"Saved values to: {csv_path}")

    default_caption = (
        "Featurization differs across rows: MLP applies learned per-slot "
        "LayerNorm before concat; baselines apply each model's natural "
        "feature_scaling (StandardScaler for LR; none for LightGBM and k-NN)."
    )
    caption = args.caption if args.caption is not None else default_caption

    png_path = output_dir / 'baselines_vs_mlp_heatmap.png'
    _render_heatmap(values, col_n, png_path, caption=caption)

    print('\nDone.')


if __name__ == '__main__':
    main()
