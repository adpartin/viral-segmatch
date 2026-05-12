"""Cross-model heatmap of per-regime TPR / TNR (MLP + sklearn baselines).

Reads each model's ``post_hoc/level1_neg_regimes.csv`` (produced by
``analyze_stage4_train.py``) and renders a single heatmap with one row
per model and one column per regime: positive => TPR, negatives => TNR.

The MLP and the sklearn baselines write the same level1_neg_regimes.csv
schema -- the input is uniform; this script doesn't care which one is the
MLP (no special-casing). Row labels are inferred from each run dir's
training_info.json (`baseline` field for baselines; otherwise "MLP")
unless --row_labels is provided.

Usage (autodiscovery — preferred):
    python src/analysis/aggregate_baselines_vs_mlp.py \\
        --bundle flu_ha_na \\
        --output_dir results/flu/July_2025/runs/baselines_vs_mlp_<bundle>_<TS>/

    Picks the latest training_<bundle>_<TS> and baseline_<name>_<bundle>_<TS>
    dirs under --runs_root (default: models/flu/July_2025/runs/), asserts
    every picked run reports the same `dataset_dir` in training_info.json,
    and uses canonical row order (MLP first, then logistic, lgbm,
    knn1_margin, knn_vote). The bundle match is anchored: --bundle flu_ha_na
    will NOT match training_flu_ha_na_seq_disjoint_*.

Usage (explicit — for ad-hoc cross-bundle comparisons):
    python src/analysis/aggregate_baselines_vs_mlp.py \\
        --model_dirs \\
            models/.../baseline_logistic_<bundle>_<TS>/ \\
            models/.../baseline_lgbm_<bundle>_<TS>/ \\
            models/.../baseline_knn1_margin_<bundle>_<TS>/ \\
            models/.../baseline_knn_vote_<bundle>_<TS>/ \\
            models/.../training_<bundle>_<TS>/ \\
        --row_labels "Logistic Regression" "LightGBM" "1-NN (margin)" "k-NN (k=5)" "MLP" \\
        --output_dir results/flu/July_2025/runs/baselines_vs_mlp_<TS>/

Outputs (under --output_dir):
    baselines_vs_mlp_heatmap.png    (per-regime TPR/TNR matrix + sidebar)
    baselines_vs_mlp_overall.png    (companion bar chart of overall AUC/F1/MCC)
    baselines_vs_mlp.csv            (numeric values)

Convention: aggregator outputs live under `results/`, not `docs/results/`.
The `docs/` tree is reserved for hand-authored writeups and decision logs;
machine-generated aggregator outputs live alongside dataset and model
runs under their respective top-level dirs.

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
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Autodiscovery: canonical row order. Any baseline outside this list is
# appended alphabetically after the known ones (so a new baseline doesn't
# silently fall off the heatmap).
_BASELINE_CANONICAL_ORDER = ('logistic', 'lgbm', 'knn1_margin', 'knn_vote')

# training_<bundle>_<YYYYMMDD>_<HHMMSS> -- the bundle can contain underscores
# (e.g., flu_ha_na_seq_disjoint) and the suffix is the fixed-format timestamp.
# Anchored so 'training_flu_ha_na_*' does NOT accidentally match
# 'training_flu_ha_na_seq_disjoint_*' and vice versa.
_TRAINING_DIR_RE = re.compile(r'^training_(?P<bundle>.+)_(?P<ts>\d{8}_\d{6})$')
# baseline_<baseline_name>_<bundle>_<TS>. baseline_name is greedy from the
# canonical list to handle multi-token names like 'knn1_margin' / 'knn_vote'.
_BASELINE_NAME_PATTERN = '|'.join(re.escape(b) for b in _BASELINE_CANONICAL_ORDER)
_BASELINE_DIR_RE = re.compile(
    rf'^baseline_(?P<name>{_BASELINE_NAME_PATTERN})_(?P<bundle>.+)_(?P<ts>\d{{8}}_\d{{6}})$'
)

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
)

# Metric per column: TPR for the positive column (single-class positives have
# no TNR), TNR for every negative regime (single-class negatives have no TPR).
_METRIC_FOR_REGIME = {r: ('tpr' if r == 'positive' else 'tnr') for r in _REGIME_ORDER}

# Overall aggregate metrics rendered to the right of the per-regime heatmap
# (sidebar -- option a) and used to drive the companion bar plot
# (overall.png -- option b). Keys must match `post_hoc/metrics.csv` columns
# emitted by analyze_stage4_train.py. All four live in [0,1] so they share
# the per-regime colormap.
_OVERALL_METRIC_KEYS = ('auc_roc', 'avg_precision', 'f1_score', 'mcc')
_OVERALL_METRIC_LABELS = {
    'auc_roc': 'AUC-ROC',
    'avg_precision': 'AUC-PR',
    'f1_score': 'F1',
    'mcc': 'MCC',
}

# Adaptive lower bound for the heatmap colormap (and the bar-plot x-axis).
# When the minimum observed value across all rendered cells is >= this
# threshold, vmin is raised to it so the colormap visibly spreads the data --
# everything-above-0.9 stops being a wall of green and you can actually
# distinguish 0.83 from 0.97. Below the threshold, vmin falls back to 0.0 so
# genuinely broken models (e.g., a collapsed logistic at AUC=0.07) still
# saturate red as expected. The threshold is deliberately non-data-driven
# (no quantiles, no per-run scaling) so the colormap is stable across runs.
_VMIN_HIGH = 0.7
_VMAX = 1.0


def _compute_vmin(*matrices: np.ndarray) -> float:
    """Return _VMIN_HIGH if every observed value across the inputs is >= it,
    else 0.0. NaNs are ignored. Used to share the colormap range between the
    heatmap and the bar plot."""
    vals = [np.asarray(m, dtype=float).ravel() for m in matrices if m is not None]
    if not vals:
        return 0.0
    combined = np.concatenate(vals)
    if combined.size == 0 or np.all(np.isnan(combined)):
        return 0.0
    observed_min = float(np.nanmin(combined))
    return _VMIN_HIGH if observed_min >= _VMIN_HIGH else 0.0


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


def _read_one_metrics(model_dir: Path) -> dict:
    """Read overall aggregate metrics from `post_hoc/metrics.csv` (one-row CSV
    emitted by analyze_stage4_train.py). Returns a dict keyed by metric name.

    Missing file -> empty dict (later filled with NaN); missing column -> that
    metric is NaN for this run. Both are reported in the aggregator's stdout.
    """
    csv = model_dir / 'post_hoc' / 'metrics.csv'
    if not csv.exists():
        print(f"WARNING: missing {csv} -- overall metrics for this run "
              f"will be NaN.")
        return {}
    df = pd.read_csv(csv)
    if len(df) == 0:
        print(f"WARNING: empty {csv}.")
        return {}
    # metrics.csv has one row; take it as a dict.
    return df.iloc[0].to_dict()


def _build_overall_matrix(model_dirs: list[Path], row_labels: list[str]) -> pd.DataFrame:
    """Build a (n_models x n_overall_metrics) DataFrame of aggregate metrics.

    Each cell is the value from `post_hoc/metrics.csv`; NaN when missing.
    Column order matches `_OVERALL_METRIC_KEYS`.
    """
    out = pd.DataFrame(index=row_labels, columns=list(_OVERALL_METRIC_KEYS), dtype=float)
    for label, model_dir in zip(row_labels, model_dirs):
        metrics = _read_one_metrics(model_dir)
        for k in _OVERALL_METRIC_KEYS:
            v = metrics.get(k)
            try:
                out.loc[label, k] = float(v) if v is not None and pd.notna(v) else np.nan
            except (TypeError, ValueError):
                out.loc[label, k] = np.nan
    return out


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
                    *, caption: Optional[str] = None,
                    overall_values: Optional[pd.DataFrame] = None) -> None:
    """Heatmap of per-regime TPR/TNR, with an optional sidebar of aggregate metrics.

    When `overall_values` is provided (shape n_rows x len(_OVERALL_METRIC_KEYS)),
    a second imshow is rendered to the right of the regime heatmap with a thin
    visual gap and a section title. Both share the [0,1] colormap and a single
    colorbar (every metric lives in that range).

    Colormap: 'RdYlGn' so high values are green and low are red. NaN cells
    render as 'N/A' grey.
    """
    n_rows, n_cols = values.shape
    n_overall = overall_values.shape[1] if overall_values is not None else 0
    # Widen the figure to accommodate the sidebar; bump height for the rotated
    # x-axis labels which would otherwise be clipped by tight_layout.
    fig_w = 1.6 + 1.0 * n_cols + (1.0 * n_overall if n_overall else 0)
    fig_h = 1.5 + 0.6 * n_rows + 0.8  # +0.8 for slanted xticks
    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad(color='lightgrey')
    # Shared vmin across both panels so cells with the same value get the same
    # color regardless of which panel they live in.
    vmin = _compute_vmin(
        values.values,
        overall_values.values if overall_values is not None else None,
    )
    vmax = _VMAX
    # Cell-text color flips at 40% of the visible color range. For vmin=0,
    # this is the historical v<0.4 rule; for vmin=0.5, it's v<0.7 so the
    # dark-red half of the [0.5, 1.0] range still gets white text.
    text_color_threshold = vmin + 0.4 * (vmax - vmin)

    def _cell_color(v: float) -> str:
        return 'white' if v < text_color_threshold else 'black'

    if n_overall == 0:
        fig, ax_regime = plt.subplots(figsize=(fig_w, fig_h))
        ax_overall = None
    else:
        # 2-column gridspec: regime heatmap | aggregate sidebar. width_ratios
        # proportional to column counts plus a small gap so the section break
        # is visible.
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[n_cols, n_overall],
            wspace=0.18,
        )
        ax_regime = fig.add_subplot(gs[0])
        ax_overall = fig.add_subplot(gs[1], sharey=ax_regime)

    # -- per-regime panel ------------------------------------------------------
    arr = values.values.astype(float)
    masked = np.ma.masked_invalid(arr)
    im_regime = ax_regime.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            if np.isnan(v):
                txt = 'N/A'
                color = 'dimgrey'
            else:
                txt = f'{v:.3f}'
                color = _cell_color(v)
            ax_regime.text(j, i, txt, ha='center', va='center', fontsize=9, color=color)
    col_headers = []
    for regime in values.columns:
        metric = _METRIC_FOR_REGIME[regime].upper()
        n = int(col_n[regime]) if pd.notna(col_n[regime]) else 0
        col_headers.append(f'{regime}\n{metric}\nn={n:,}')
    ax_regime.set_xticks(range(n_cols))
    # rotation=35 with ha='right' anchored at the top of each tick prevents
    # adjacent multi-line labels (e.g. host_subtype_year, subtype_year_only)
    # from colliding at the previous rotation=0 setting.
    ax_regime.set_xticklabels(
        col_headers, rotation=35, ha='right', rotation_mode='anchor', fontsize=8,
    )
    ax_regime.set_yticks(range(n_rows))
    ax_regime.set_yticklabels(values.index, fontsize=10)
    ax_regime.set_title(
        'Per-regime TPR (positive) / TNR (negatives)',
        fontsize=11,
    )

    # -- overall sidebar (option a) -------------------------------------------
    if ax_overall is not None and overall_values is not None:
        arr_o = overall_values.values.astype(float)
        masked_o = np.ma.masked_invalid(arr_o)
        ax_overall.imshow(masked_o, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        for i in range(arr_o.shape[0]):
            for j in range(arr_o.shape[1]):
                v = arr_o[i, j]
                if np.isnan(v):
                    txt = 'N/A'
                    color = 'dimgrey'
                else:
                    txt = f'{v:.3f}'
                    color = _cell_color(v)
                ax_overall.text(j, i, txt, ha='center', va='center',
                                fontsize=9, color=color)
        overall_headers = [
            _OVERALL_METRIC_LABELS[k] for k in overall_values.columns
        ]
        ax_overall.set_xticks(range(arr_o.shape[1]))
        ax_overall.set_xticklabels(
            overall_headers, rotation=35, ha='right',
            rotation_mode='anchor', fontsize=9,
        )
        # Hide redundant y-tick labels on the sidebar (rows already labeled
        # on the regime panel via sharey).
        plt.setp(ax_overall.get_yticklabels(), visible=False)
        ax_overall.set_title('Aggregate (test set)', fontsize=11)

    # Shared colorbar. Anchor on the rightmost axis (overall sidebar if present,
    # else the regime panel) so layout is balanced.
    cbar_ax = ax_overall if ax_overall is not None else ax_regime
    cbar = fig.colorbar(im_regime, ax=cbar_ax, fraction=0.025, pad=0.04)
    cbar.set_label(f'metric value ({vmin:.1f}–{vmax:.1f})', fontsize=9)

    if caption:
        # Push the caption well below the rotated x-tick labels so they don't
        # collide. Rotated 25-degree multi-line labels eat ~0.07-0.10 of the
        # figure height under tight_layout; -0.12 keeps the caption clear.
        fig.text(0.5, -0.12, caption, ha='center', fontsize=8,
                 color='dimgrey', style='italic', wrap=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def _render_overall_bars(overall_values: pd.DataFrame, output_path: Path,
                         *, caption: Optional[str] = None) -> None:
    """Companion plot (option b): horizontal grouped bar chart of aggregate
    metrics. One row of bars per model, one bar per metric. Visually distinct
    from the heatmap so the user can decide which framing they prefer.

    Bars are color-coded per metric (consistent across models). Sample size
    is not shown here -- it's identical across rows (same test set) and is
    already displayed in the heatmap column headers.
    """
    n_rows = overall_values.shape[0]
    metrics = list(overall_values.columns)
    metric_labels = [_OVERALL_METRIC_LABELS[m] for m in metrics]
    # One row per model; bars stacked vertically with a per-metric color.
    fig_w = 8.5
    fig_h = 1.0 + 0.7 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_h = 0.8 / len(metrics)  # bars per model fit within a 0.8 band
    colors = ['#5B7FB9', '#E89B2E', '#7FB069', '#B05279']  # distinct hues
    y_centers = np.arange(n_rows)
    for j, (m, lab, c) in enumerate(zip(metrics, metric_labels, colors)):
        # Shift each metric's bar within its model's band.
        offsets = (j - (len(metrics) - 1) / 2.0) * bar_h
        ax.barh(
            y_centers + offsets,
            overall_values[m].values,
            height=bar_h,
            color=c,
            edgecolor='black',
            linewidth=0.6,
            label=lab,
        )
        # Annotate each bar with its value (3 decimals).
        for i, v in enumerate(overall_values[m].values):
            if pd.isna(v):
                continue
            ax.text(min(v + 0.005, 0.99), y_centers[i] + offsets, f'{v:.3f}',
                    va='center', ha='left', fontsize=8, color='black')

    ax.set_yticks(y_centers)
    ax.set_yticklabels(overall_values.index, fontsize=10)
    ax.invert_yaxis()  # top-of-list = top-of-plot
    # Adaptive x-axis: when every metric is >= _VMIN_HIGH (0.5) the axis
    # starts at that threshold so the bars visually spread out; otherwise
    # full [0, 1.05] range so a collapsed model still saturates left.
    bar_vmin = _compute_vmin(overall_values.values)
    ax.set_xlim(bar_vmin, 1.05)
    ax.set_xlabel(f'metric value ({bar_vmin:.1f}–1.0)', fontsize=10)
    ax.set_title('Aggregate test metrics — models × metrics', fontsize=11)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    # Legend below the axes so it doesn't crowd the rightmost bar value labels
    # (every metric on a strong model sits near x=1.0 -- 'lower right' would
    # overlap the last-row annotations).
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              fontsize=9, ncol=len(metrics), frameon=True)

    if caption:
        fig.text(0.5, -0.12, caption, ha='center', fontsize=8,
                 color='dimgrey', style='italic', wrap=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved overall bars to: {output_path}")


def _autodiscover_dirs(runs_root: Path, bundle: str
                       ) -> tuple[list[Path], list[str], str]:
    """Find the latest training_ + baseline_ runs for a bundle.

    Anchored bundle match (the regex requires `bundle` to occupy the full
    middle slot between the prefix and the timestamp), so this will NOT
    pick up sibling bundles whose names share a prefix:
      e.g. `--bundle flu_ha_na` does not match
      `training_flu_ha_na_seq_disjoint_20260511_001322`.

    For each baseline type, picks the latest-timestamp run. For MLP,
    same. Cross-checks that every picked run's `training_info.json` reports
    the same `dataset_dir` -- otherwise refuses to aggregate.

    Returns (model_dirs, row_labels, shared_dataset_dir).
    """
    if not runs_root.exists():
        raise FileNotFoundError(f"--runs_root does not exist: {runs_root}")

    training_candidates: list[tuple[str, Path]] = []
    baseline_candidates: dict[str, list[tuple[str, Path]]] = {}
    for sub in runs_root.iterdir():
        if not sub.is_dir():
            continue
        m_t = _TRAINING_DIR_RE.match(sub.name)
        if m_t and m_t.group('bundle') == bundle:
            training_candidates.append((m_t.group('ts'), sub))
            continue
        m_b = _BASELINE_DIR_RE.match(sub.name)
        if m_b and m_b.group('bundle') == bundle:
            baseline_candidates.setdefault(m_b.group('name'), []).append(
                (m_b.group('ts'), sub)
            )

    if not training_candidates and not baseline_candidates:
        raise FileNotFoundError(
            f"No training_{bundle}_*  or  baseline_<name>_{bundle}_*  dirs found "
            f"under {runs_root}. Check --bundle and --runs_root."
        )

    # Pick latest per kind.
    mlp_dir: Optional[Path] = None
    if training_candidates:
        training_candidates.sort()
        mlp_dir = training_candidates[-1][1]
    baseline_dirs: dict[str, Path] = {}
    for bname, lst in baseline_candidates.items():
        lst.sort()
        baseline_dirs[bname] = lst[-1][1]

    # Row order: MLP first, then canonical baseline order (skipping absent),
    # then any extras alphabetical (won't happen with the current registry but
    # future-proofs against a new baseline that isn't in
    # _BASELINE_CANONICAL_ORDER yet).
    model_dirs: list[Path] = []
    row_labels: list[str] = []
    if mlp_dir is not None:
        model_dirs.append(mlp_dir)
        row_labels.append('MLP')
    for bname in _BASELINE_CANONICAL_ORDER:
        if bname in baseline_dirs:
            model_dirs.append(baseline_dirs[bname])
            row_labels.append(bname)
    extras = sorted(set(baseline_dirs.keys()) - set(_BASELINE_CANONICAL_ORDER))
    for bname in extras:
        model_dirs.append(baseline_dirs[bname])
        row_labels.append(bname)

    # Cross-check provenance: every run must point at the same Stage-3 dataset.
    dataset_dirs: dict[str, str] = {}
    for d in model_dirs:
        info = d / 'training_info.json'
        if not info.exists():
            raise FileNotFoundError(
                f"Missing training_info.json in {d}; cannot validate provenance "
                f"for autodiscovered run."
            )
        try:
            ds = json.loads(info.read_text()).get('dataset_dir')
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt training_info.json in {d}: {e}")
        if ds is None:
            raise ValueError(
                f"training_info.json in {d} has no 'dataset_dir' field; refusing "
                f"to aggregate without provenance."
            )
        dataset_dirs[d.name] = ds

    unique = set(dataset_dirs.values())
    if len(unique) > 1:
        msg = ["Autodiscovered runs disagree on dataset_dir; refusing to aggregate:"]
        for n, ds in dataset_dirs.items():
            msg.append(f"  {n}\n      dataset_dir={ds}")
        msg.append("Pass --model_dirs explicitly to override.")
        raise ValueError("\n".join(msg))
    shared = next(iter(unique))

    print(f"\nAutodiscovered {len(model_dirs)} run(s) for bundle={bundle!r}:")
    print(f"  shared dataset_dir: {shared}")
    for label, d in zip(row_labels, model_dirs):
        ts = d.name.rsplit('_', 2)
        ts_str = f"{ts[-2]}_{ts[-1]}"
        print(f"  {label:<14s}  TS={ts_str}  {d.name}")
    return model_dirs, row_labels, shared


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate per-regime TPR/TNR across MLP + baseline runs into a heatmap.'
    )
    # Discovery mode: either supply --bundle (autodiscover the latest runs) or
    # --model_dirs (explicit list). Exactly one of the two must be set.
    parser.add_argument('--bundle', type=str, default=None,
                        help='Bundle name (e.g., flu_ha_na_seq_disjoint). When set, '
                             'autodiscovers the latest training_<bundle>_<TS> and '
                             'baseline_<name>_<bundle>_<TS> dirs under --runs_root, '
                             'and cross-checks that all picked runs share the same '
                             'dataset_dir (refuses to aggregate otherwise). Mutually '
                             'exclusive with --model_dirs.')
    parser.add_argument('--runs_root', type=str,
                        default='models/flu/July_2025/runs',
                        help='Root dir under which to search for training_*/baseline_* '
                             'subdirs (default: models/flu/July_2025/runs).')
    parser.add_argument('--model_dirs', type=str, nargs='+', default=None,
                        help='Explicit run directories (one or more). Each must contain '
                             'post_hoc/level1_neg_regimes.csv. Order = top-to-bottom row '
                             'order of the heatmap. Mutually exclusive with --bundle.')
    parser.add_argument('--row_labels', type=str, nargs='+', default=None,
                        help='Optional row labels (one arg per label; quote labels that '
                             'contain spaces). Overrides auto-inference from '
                             'training_info.json.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to write baselines_vs_mlp_heatmap.png and '
                             'baselines_vs_mlp.csv.')
    parser.add_argument('--caption', type=str, default=None,
                        help='Optional caption rendered below the heatmap. If omitted, '
                             'a default caption flagging featurization parity is used.')
    args = parser.parse_args()

    if (args.bundle is None) == (args.model_dirs is None):
        parser.error('Exactly one of --bundle or --model_dirs must be set.')

    shared_dataset_dir: Optional[str] = None
    if args.bundle is not None:
        model_dirs, auto_labels, shared_dataset_dir = _autodiscover_dirs(
            Path(args.runs_root), args.bundle
        )
        if args.row_labels:
            row_labels = list(args.row_labels)
            if len(row_labels) != len(model_dirs):
                raise ValueError(
                    f"--row_labels has {len(row_labels)} entries but "
                    f"autodiscovery found {len(model_dirs)} runs."
                )
        else:
            row_labels = auto_labels
    else:
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
    overall_values = _build_overall_matrix(model_dirs, row_labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'baselines_vs_mlp.csv'
    # Single CSV with everything: per-regime columns first, then the four
    # aggregate metrics. Column order matches the heatmap's visual layout.
    out_csv = pd.concat([values, overall_values], axis=1)
    out_csv.index.name = 'model'
    out_csv.to_csv(csv_path)
    print(f"Saved values to: {csv_path}")

    default_caption = (
        "Featurization differs across rows: MLP applies learned per-slot "
        "LayerNorm before concat; baselines apply each model's natural "
        "feature_scaling (StandardScaler for LR; none for LightGBM and k-NN)."
    )
    if shared_dataset_dir is not None:
        default_caption = (
            f"Bundle: {args.bundle}. Shared dataset: {Path(shared_dataset_dir).name}. "
            + default_caption
        )
    caption = args.caption if args.caption is not None else default_caption

    png_path = output_dir / 'baselines_vs_mlp_heatmap.png'
    _render_heatmap(values, col_n, png_path, caption=caption,
                    overall_values=overall_values)

    # Companion bar plot (option b) so the user can compare against the
    # sidebar layout. Sharing the same caption keeps the provenance text
    # consistent across both renderings.
    overall_png = output_dir / 'baselines_vs_mlp_overall.png'
    _render_overall_bars(overall_values, overall_png, caption=caption)

    print('\nDone.')


if __name__ == '__main__':
    main()
