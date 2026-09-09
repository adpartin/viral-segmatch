"""Shared readers for the per-site importance table.

`plot_site_importance.py` writes one row per feature column with a `{measure}_frac` share and a
`{measure}_rank` for each of the three measures it computes. Everything downstream that wants to
act on "the most important sites" reads that table and orders it, so the read and the ordering live
here once rather than being repeated per script.

Which measure to order by is a real choice, not a detail. Gain is read off the fitted trees and so
describes what the model was built on. SHAP is measured on held-out rows and so describes what the
model is worth out of sample. Permutation is measured on the model's output. They agree closely at
the top of the list on Flu A HA-NA (Spearman about +0.97 between gain and SHAP, 12 of the top 15
shared) but not exactly, so a run that says it ranked by gain must actually have ranked by gain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The measures `plot_site_importance.py` writes a `_frac` and a `_rank` column for.
IMPORTANCE_MEASURES = ('gain', 'shap', 'perm')


def rank_column(measure: str) -> str:
    """Name of the rank column for one importance measure.

    Args:
      measure: one of `IMPORTANCE_MEASURES`.

    Returns:
      The column name, e.g. `gain_rank`.

    Raises:
      ValueError: the measure is not one this table carries.
    """
    if measure not in IMPORTANCE_MEASURES:
        raise ValueError(
            f"importance measure must be one of {list(IMPORTANCE_MEASURES)}; got {measure!r}.")
    return f'{measure}_rank'


def share_column(measure: str) -> str:
    """Name of the importance-share column for one importance measure.

    Args:
      measure: one of `IMPORTANCE_MEASURES`.

    Returns:
      The column name, e.g. `gain_frac`.

    Raises:
      ValueError: the measure is not one this table carries.
    """
    if measure not in IMPORTANCE_MEASURES:
        raise ValueError(
            f"importance measure must be one of {list(IMPORTANCE_MEASURES)}; got {measure!r}.")
    return f'{measure}_frac'


def load_importance(csv_path: Path) -> pd.DataFrame:
    """Read a `site_importance_{unit}.csv` table.

    Args:
      csv_path: path to the table `plot_site_importance.py` wrote.

    Returns:
      The table, with the `protein` column intact.

    Raises:
      FileNotFoundError: the table has not been produced yet.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"missing {csv_path}. Run `python -m src.analysis.plot_site_importance` first.")
    # keep_default_na: the `protein` column holds the literal string NA (Neuraminidase), which a
    # default read turns into NaN and drops.
    return pd.read_csv(csv_path, keep_default_na=False, na_values=[''])


def rank_columns(importance: pd.DataFrame, measure: str) -> np.ndarray:
    """Order the feature columns from most to least important under one measure.

    Args:
      importance: a table from `load_importance`.
      measure: one of `IMPORTANCE_MEASURES`.

    Returns:
      Feature column indices, most important first.

    Raises:
      ValueError: the measure is unknown, or its rank column is absent from the table.
    """
    column = rank_column(measure)
    if column not in importance.columns:
        raise ValueError(
            f"the importance table has no {column!r} column, so it cannot be ranked by "
            f"{measure!r}. Columns present: {sorted(importance.columns)}.")
    ordered = importance.sort_values(column)['column'].to_numpy()
    return ordered


def plot_importance_trace(importance: pd.DataFrame, measure: str, unit: str, out_path,
                          dpi: int = 200, n_label: int = 5):
    """Draw one importance measure along each protein's CDS, one panel per protein.

    Deliberately shows a single measure. Overlaying several invites the reader to compare curves
    that are on the same axis but answer different questions, and a report that has chosen one
    measure should show that one.

    Args:
      importance: a table from `load_importance`.
      measure: one of `IMPORTANCE_MEASURES`.
      unit: site unit, for the axis label.
      out_path: where the PNG goes.
      dpi: raster resolution.
      n_label: how many top sites to annotate per protein.

    Returns:
      The written path.

    Raises:
      ValueError: the measure is unknown or its share column is absent.
    """
    import matplotlib.pyplot as plt

    from src.utils.plot_utils import savefig, setup_plot_style

    share = share_column(measure)
    if share not in importance.columns:
        raise ValueError(
            f"the importance table has no {share!r} column, so {measure!r} cannot be plotted.")

    setup_plot_style()
    proteins = list(dict.fromkeys(importance['protein']))
    fig, axes = plt.subplots(len(proteins), 1, figsize=(11, 3.1 * len(proteins)), squeeze=False)

    for row, protein in enumerate(proteins):
        ax = axes[row][0]
        of_protein = importance[importance['protein'] == protein]
        ax.plot(of_protein['site'], of_protein[share], color='#4C7CAB', linewidth=0.8)
        top = of_protein.nlargest(n_label, share)
        ax.scatter(top['site'], top[share], s=32, color='#CF8793', edgecolor='#222222',
                   zorder=3, linewidth=0.6)
        for r in top.itertuples():
            ax.annotate(f"{int(getattr(r, 'site'))}", (getattr(r, 'site'), getattr(r, share)),
                        textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)
        n_used = int((of_protein[share] > 0).sum())
        ax.set_xlim(1, int(of_protein['site'].max()))
        ax.set_xlabel(f'{protein} {unit} site')
        ax.set_ylabel(f'share of {measure}')
        ax.set_title(f"{protein}: {len(of_protein):,} sites, {n_used:,} with non-zero {measure}")
        ax.grid(axis='y', alpha=0.3)

    fig.text(0.995, 0.002, 'src/analysis/_importance_helpers.py', ha='right', va='bottom',
             fontsize=7, color='#666666')
    fig.tight_layout()
    return savefig(out_path, dpi=dpi)


def permutation_curve(csv_path: Path, split: str = 'test',
                      method: str = 'shuffle') -> pd.DataFrame:
    """Read a group-permutation table and average the folds and repeats for one split.

    Exists because the raw table is easy to aggregate wrongly. It carries a `method` column with
    both `shuffle` and `constant` rows, so a `groupby` that forgets to filter silently averages
    two different experiments and reports a number that is neither. The N values where a
    constant-fill run exists are the only ones that come out wrong, so the mistake also looks
    plausible: most of the curve is unaffected.

    Args:
      csv_path: `site_shuffle_fixed_{unit}_{measure}.csv`.
      split: which split to read, `test` or `train`.
      method: `shuffle` for the permutation curve, `constant` for the fill comparison.

    Returns:
      One row per `(arm, n_sites)` with the mean and std of `signal_lost` over folds and repeats.

    Raises:
      FileNotFoundError: the table has not been produced yet.
      ValueError: the requested split or method is absent from the table.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"missing {csv_path}. Run `python -m src.analysis.plot_site_shuffle_fixed` first.")
    table = pd.read_csv(csv_path)
    for column, wanted in (('split', split), ('method', method)):
        present = set(table[column].unique())
        if wanted not in present:
            raise ValueError(
                f"{csv_path}: {column}={wanted!r} is absent; the table has {sorted(present)}.")
    selected = table[(table['split'] == split) & (table['method'] == method)]
    curve = (selected.groupby(['arm', 'n_sites'])['signal_lost']
             .agg(['mean', 'std'])
             .reset_index())
    return curve
