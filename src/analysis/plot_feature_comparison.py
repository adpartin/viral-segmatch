"""Held-out scores for every schema pair under every feature set, on one shared population.

The four-pair experiment fits the same population four ways -- k-mer counts, per-site
nucleotides, per-site codons, per-site amino acids -- for each of four schema pairs. The
numbers already live in the per-run `metrics_summary.json`, and the progress report carries
them as a table. A table of 16 cells hides two things a figure shows at once: how far apart
the feature sets are within a pair, and whether that ordering is the same in every pair.

Each cell is drawn as its four folds plus their mean and spread, rather than as a bar. Four
folds is few enough that the individual fits are worth seeing, and a bar would state a mean
with more confidence than four numbers support.

Labels are read from each run's own `resolved_config.yaml` rather than parsed out of the run
directory name. `training.feature_source` is what actually selects the feature set, and it is the
only key that does: every bundle carries both a `kmer` and a `site` block, so a k-mer run still
records `site.unit` and reading that would mislabel it.

Schema pairs are placed left to right in the canonical protein order, the same order
`summarize_pair_capacity` uses, so this figure and the pair-capacity table list pairs alike.

Outputs (to `--out_dir`):
    feature_comparison_{metric}.png   one panel per metric, pairs on x, feature sets grouped
    feature_comparison.csv            one row per pair, feature set, metric and fold

CLI:
    python -m src.analysis.plot_feature_comparison \
        --run_template 'lgbm_{pair}_human_h3n2_2024_n1698_seed42_{features}' \
        --pairs ha_na pb2_pa pb2_na pa_ha \
        --features kmer_nt_cds_k6 site_nt site_codon site_aa \
        --metrics f1_macro \
        --out_dir docs/results/figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.config_hydra import (  # noqa: E402
    canonical_protein_pairs,
    get_function_short_name_map,
    get_virus_config_hydra,
)
from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402

# Each metric names its display label and its chance level, so "draw a chance line on AUC-ROC
# only" is a property of the metric rather than a special case in the drawing code. F1,
# precision and recall have no fixed chance level, since theirs moves with the class balance.
METRIC_PANEL = {
    'f1_macro': ('F1 macro', None),
    'f1': ('F1', None),
    'auc_roc': ('AUC-ROC', 0.5),
    'precision': ('Precision', None),
    'recall': ('Recall', None),
}

# Colour and marker are assigned by position in the feature-set list, not by name, so a run
# with three or six feature sets styles itself. Both vary together so the panel still reads
# when printed in greyscale.
FEATURE_COLORS = ['#4C7CAB', '#5B9E6E', '#CF8793', '#B98B4A', '#7E6BA8', '#4E9DA6']
FEATURE_MARKERS = ['o', 's', '^', 'D', 'v', 'P']

FOLD_DOT_SIZE = 14
MEAN_DOT_SIZE = 35
MARKER_EDGE = '#222222'


def _stamp(fig) -> None:
    """Write the producing script into the figure, so a stray PNG can be traced back.

    Args:
      fig: the figure to stamp.
    """
    fig.text(0.995, 0.002, f'src/analysis/{Path(__file__).name}', ha='right', va='bottom',
             fontsize=7, color='0.45')


def describe_run(run_dir: Path, function_to_short: dict) -> tuple[tuple[str, str], str]:
    """Read a run's schema pair and feature set out of its resolved config.

    Args:
      run_dir: a single fold's run directory, holding `resolved_config.yaml`.
      function_to_short: full function name -> short protein name.

    Returns:
      The schema pair as short names in the order the config records them, and the feature-set
      label.

    Raises:
      FileNotFoundError: the run directory has no resolved config.
      ValueError: the config records an unrecognised `model.feature_source`.
    """
    config_path = run_dir / 'resolved_config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}, needed to label the run.")
    cfg = OmegaConf.load(config_path)

    functions = list(cfg.dataset.schema_pair)
    pair = tuple(function_to_short.get(f, f) for f in functions)

    source = str(cfg.training.feature_source)
    if source == 'kmer':
        features = f'k-mer {cfg.kmer.alphabet} k={cfg.kmer.k}'
    elif source == 'site':
        features = f'site {cfg.site.unit}'
    else:
        raise ValueError(
            f"{config_path} has training.feature_source={source!r}; this script labels "
            f"'kmer' and 'site' runs only.")
    return pair, features


def load_fold_metrics(models_root: Path, run_template: str, pairs: list, features: list,
                      n_folds: int, function_to_short: dict,
                      split: str = 'test') -> pd.DataFrame:
    """Collect every fold's metrics for each schema pair and feature set into one long table.

    Args:
      models_root: directory holding the run directories.
      run_template: run dir name with `{pair}` and `{features}` placeholders, minus the
          `_fold{k}` suffix.
      pairs: `{pair}` substitutions, path tokens rather than display labels.
      features: `{features}` substitutions, path tokens rather than display labels.
      n_folds: folds per run.
      function_to_short: full function name -> short protein name.
      split: which split's metrics to read, e.g. `test`.

    Returns:
      One row per schema pair, feature set, metric and fold, with columns `pair`, `features`,
      `fold`, `metric`, `value`.

    Raises:
      FileNotFoundError: a fold's `metrics_summary.json` is missing.
      KeyError: a fold's metrics have no entry for `split`.
    """
    rows = []
    for pair_token in pairs:
        for feature_token in features:
            stem = run_template.format(pair=pair_token, features=feature_token)
            for fold in range(n_folds):
                run_dir = models_root / f'{stem}_fold{fold}'
                metrics_path = run_dir / 'metrics_summary.json'
                if not metrics_path.exists():
                    raise FileNotFoundError(f"missing {metrics_path}.")
                metrics = json.loads(metrics_path.read_text())
                if split not in metrics:
                    raise KeyError(
                        f"{metrics_path} has no {split!r} split; it holds "
                        f"{sorted(metrics)}.")
                pair, feature_label = describe_run(run_dir, function_to_short)
                for metric, value in metrics[split].items():
                    rows.append({
                        'pair': f'{pair[0]}-{pair[1]}',
                        'features': feature_label,
                        'fold': fold,
                        'metric': metric,
                        'value': float(value),
                    })
    return pd.DataFrame(rows)


def canonical_pair_order(table: pd.DataFrame, canonical_order: list) -> list:
    """Order the table's schema pairs the way `summarize_pair_capacity` orders its rows.

    Only the pairs present in the table are kept, because an experiment trains a chosen subset
    of the pairs its proteins could form.

    Args:
      table: the long metrics table, carrying a `pair` column of `A-B` labels.
      canonical_order: short protein names in canonical order.

    Returns:
      The table's `A-B` labels, canonically ordered.
    """
    present = set(table['pair'])
    proteins = sorted({p for label in present for p in label.split('-')},
                      key=canonical_order.index)
    every_pair = canonical_protein_pairs(proteins, canonical_order)
    return [f'{a}-{b}' for a, b in every_pair if f'{a}-{b}' in present]


def plot_metric_panel(ax, table: pd.DataFrame, metric: str, pair_order: list,
                      feature_order: list) -> None:
    """Draw one metric for every schema pair and feature set onto one axis.

    Each schema pair holds one slot on the x axis, and the feature sets are offset within it.
    A cell shows its folds as small points and their mean as a large one, with an error bar of
    one sample standard deviation.

    Args:
      ax: the axis to draw on.
      table: the long metrics table.
      metric: one of `METRIC_PANEL`.
      pair_order: schema pair labels, left to right.
      feature_order: feature-set labels, in legend and offset order.

    Raises:
      ValueError: the metric is unknown, or the table holds no rows for it.
    """
    if metric not in METRIC_PANEL:
        raise ValueError(f"metric must be one of {sorted(METRIC_PANEL)}; got {metric!r}.")
    display, chance = METRIC_PANEL[metric]
    rows = table[table['metric'] == metric]
    if rows.empty:
        raise ValueError(f"the table holds no rows for metric {metric!r}.")

    # Feature sets are spread across most of the slot, leaving a gap between adjacent pairs.
    n_features = len(feature_order)
    slot_width = 0.72
    offsets = [(i - (n_features - 1) / 2) * slot_width / max(n_features, 1)
               for i in range(n_features)]

    for x, pair in enumerate(pair_order):
        if x % 2:
            ax.axvspan(x - 0.5, x + 0.5, color='0.5', alpha=0.07, linewidth=0)
        for i, features in enumerate(feature_order):
            cell = rows[(rows['pair'] == pair) & (rows['features'] == features)]
            if cell.empty:
                continue
            color = FEATURE_COLORS[i % len(FEATURE_COLORS)]
            marker = FEATURE_MARKERS[i % len(FEATURE_MARKERS)]
            center = x + offsets[i]

            # The folds are spread deterministically inside their own offset, so the figure
            # redraws identically and no fold is hidden behind another.
            values = cell.sort_values('fold')['value'].to_numpy()
            spread = slot_width / max(n_features, 1) * 0.18
            fold_x = [center + spread * (j - (len(values) - 1) / 2)
                      for j in range(len(values))]
            ax.scatter(fold_x, values, s=FOLD_DOT_SIZE, color=color, alpha=0.45,
                       linewidths=0, zorder=2)

            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            ax.errorbar(center, mean, yerr=std, color=MARKER_EDGE, elinewidth=0.9,
                        capsize=3, capthick=0.9, zorder=3, linestyle='none')
            ax.scatter(center, mean, s=MEAN_DOT_SIZE, color=color, marker=marker,
                       alpha=0.7, edgecolor=MARKER_EDGE, linewidths=0.7, zorder=4,
                       label=features if x == 0 else None)

    if chance is not None:
        ax.axhline(chance, color='0.45', linestyle='--', linewidth=0.9, zorder=1)
        ax.annotate(f'chance {chance:g}', xy=(1.0, chance), xycoords=('axes fraction', 'data'),
                    xytext=(-2, 3), textcoords='offset points', ha='right', va='bottom',
                    fontsize=7, color='0.45')

    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order)
    ax.set_xlim(-0.5, len(pair_order) - 0.5)
    ax.set_ylabel(display)
    ax.set_title(f'{display} (mean of all folds ± std)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def plot_metric_panels(table: pd.DataFrame, metrics: list, pair_order: list,
                       feature_order: list, title: str, out_png: Path, dpi: int) -> Path:
    """Arrange one `plot_metric_panel` per metric into a single figure.

    Args:
      table: the long metrics table.
      metrics: which metrics to draw, one panel each.
      pair_order: schema pair labels, left to right.
      feature_order: feature-set labels, in legend and offset order.
      title: figure title.
      out_png: where to write.
      dpi: figure resolution.

    Returns:
      The path written.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.35 * len(pair_order) * len(metrics) + 2.4,
                                                       4.4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        plot_metric_panel(ax, table, metric, pair_order, feature_order)

    fig.suptitle(title, fontsize=10)
    # The axes are shrunk from the bottom to make room for the legend, which sits under them
    # rather than inside: the weakest feature set runs along the floor of the panel, where an
    # inset legend would cover it.
    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 4),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.025))
    _stamp(fig)
    written = savefig(out_png, dpi=dpi)
    print(f"Wrote {written}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run_template', required=True,
                   help='run dir name with {pair} and {features}, minus the _fold{k} suffix')
    p.add_argument('--pairs', nargs='+', required=True,
                   help='{pair} substitutions, e.g. ha_na pb2_pa')
    p.add_argument('--features', nargs='+', required=True,
                   help='{features} substitutions, e.g. kmer_nt_cds_k6 site_codon')
    p.add_argument('--metrics', nargs='+', default=['f1_macro'],
                   choices=sorted(METRIC_PANEL),
                   help='one panel per metric')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                   help="which split to read; 'test' is held out")
    p.add_argument('--config_bundle', default='flu_ha_na_h3n2_2024_random_cv4_site_codon',
                   help='bundle supplying the canonical protein order and the short names')
    p.add_argument('--models_root', type=Path, default=PROJ / 'models/flu/July_2025/runs')
    p.add_argument('--n_folds', type=int, default=4)
    p.add_argument('--title', default=None,
                   help='figure title; defaults to the run template and split')
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    function_to_short = get_function_short_name_map(config)
    canonical_order = [function_to_short[f] for f in config.virus.protein_order]

    table = load_fold_metrics(args.models_root, args.run_template, args.pairs, args.features,
                              args.n_folds, function_to_short, split=args.split)
    pair_order = canonical_pair_order(table, canonical_order)
    # Feature sets keep the order they were asked for on the command line, since unlike the
    # proteins they have no canonical order to fall back on. `load_fold_metrics` walks the
    # feature sets in that order within each schema pair, so the table's rows already carry it.
    feature_order = table['features'].drop_duplicates().tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / 'feature_comparison.csv'
    table.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # The template's placeholders name what varies across the panel, so the title drops them
    # and keeps only the part every run shares.
    shared = args.run_template.replace('{pair}', '').replace('{features}', '')
    shared = '_'.join(part for part in shared.split('_') if part)
    title = args.title or f'{shared}  |  {args.split} split, {args.n_folds} folds'
    for metric in args.metrics:
        summary = (table[table['metric'] == metric]
                   .groupby(['pair', 'features'])['value']
                   .agg(['mean', 'std']))
        print(f"\n{METRIC_PANEL[metric][0]}:")
        print(summary.round(4).to_string())

    plot_metric_panels(table, args.metrics, pair_order, feature_order, title,
                       args.out_dir / f"feature_comparison_{'_'.join(args.metrics)}.png",
                       args.dpi)
    print('\nDone.')


if __name__ == '__main__':
    main()
