"""Cross-threshold CC-count trend of the cluster bigraph.

The per-`t` CC views (`plot_cc_sizes.py`, `plot_cc_composition.py`, `plot_cc_metadata.py`) each
show ONE threshold at a time. This is the cross-`t` summary for one schema pair: how many
connected components (CCs) the cluster bigraph has at each identity threshold, strict-first, with
the largest-CC pair share overlaid so the "many CCs but one dominates" structure is visible in one
view.

Reads the `cc_summary.json` files that `src/datasets/build_cc_structure.py` persists under
`cc_{source}/{pair}/tXXX/` -- one scalar record per threshold (`n_ccs`, `largest_cc_pairs`,
`largest_cc_frac`, `n_pairs_universe`, `n_pairs_joined`). Recomputes nothing and rebuilds no
bigraph. Because the artifacts encode the cluster SOURCE in the directory name
(`nt_cds_ood` / `nt_cds_cm0` / ...), this runs against any cluster source, and the pair counts are
the operational universe the splitter routes (HA-NA nt_cds: 78,764 pairs).

`--fragmented` reads `tXXX/fragmented/cc_summary.json` instead: the same trend AFTER the edge-cut
fragmentation, so the natural and post-cut atom counts can be compared threshold by threshold.

Denominator note: `largest_cc_frac` in the artifact is a fraction of `n_pairs_universe`, not of
`n_pairs_joined`. Under `--fragmented` the straddling pairs the cut dropped leave the numerator but
NOT the denominator, which is what makes the natural and fragmented curves directly comparable.

Note: n_CCs is NOT monotonic in `t`. The cluster bigraph's edge set (the pair universe) is fixed
across `t`; only the cluster labels change, and mmseqs2 clusters each threshold independently (no
refinement hierarchy), so a CC count can tick up as `t` loosens.

CLI:
    python -m src.analysis.bigraph_cc_count_vs_threshold \\
        --cc_dir data/processed/flu/July_2025/cc_nt_cds_ood/HA-NA \\
        [--thresholds t099 t098 t097 t096 t095] [--fragmented] \\
        [--out_dir results/flu/July_2025/runs/2D_cluster_sizes]

Outputs (under --out_dir), one plot per --cc_dir:
    plots/cc_count_vs_threshold_{a}_{b}_{source}[_fragmented].png
    cc_count_vs_threshold.csv   tidy: schema_pair, cc_source, fragmented, threshold, n_ccs,
                                largest_cc_pairs, n_pairs_universe, n_pairs_joined, largest_cc_pct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

_BAR_COLOR = '#4c72b0'     # blue   — n_CCs bars (left axis)
_LINE_COLOR = '#ff7f0e'    # orange — largest-CC % (right axis)
_LINE_LABEL = '#9a4a00'    # darker orange for the % annotations (legible on blue)


def _threshold_sort_key(t: str) -> int:
    """Strict-first ordering key: t099 before t095 (higher identity = stricter)."""
    return -int(t.lstrip('t'))


def load_summaries(cc_dir: Path, thresholds: list[str] | None = None,
                   fragmented: bool = False) -> pd.DataFrame:
    """Per-threshold n_CCs + largest-CC% for one `cc_{source}/{pair}` dir, strict-first.

    Reads one `cc_summary.json` per threshold -- the scalars are already computed by
    `build_cc_structure`, so this only collects and orders them.

    Args:
        cc_dir: a `cc_{source}/{pair}` artifact dir holding `tXXX/` subdirs.
        thresholds: restrict to these (e.g. `['t099', 't095']`); None = every `tXXX` found.
        fragmented: read `tXXX/fragmented/cc_summary.json` (post-edge-cut) instead of `tXXX/`.

    Returns:
        One row per threshold with the tidy columns; empty if no summary was found.
    """
    source = cc_dir.parent.name.removeprefix('cc_')   # 'cc_nt_cds_ood' -> 'nt_cds_ood'
    found = sorted((d.name for d in cc_dir.glob('t[0-9][0-9][0-9]') if d.is_dir()),
                   key=_threshold_sort_key)
    wanted = [t for t in found if thresholds is None or t in thresholds]

    rows = []
    for t in wanted:
        summary = cc_dir / t / ('fragmented' if fragmented else '') / 'cc_summary.json'
        if not summary.exists():
            print(f'WARNING: no cc_summary.json at {summary}; skipping {t}.')
            continue
        s = json.loads(summary.read_text())
        rows.append({
            'schema_pair': f"{s['slot_a']}-{s['slot_b']}",
            'cc_source': source,
            'fragmented': fragmented,
            'threshold': t,
            'n_ccs': int(s['n_ccs']),
            'largest_cc_pairs': int(s['largest_cc_pairs']),
            'n_pairs_universe': int(s['n_pairs_universe']),
            'n_pairs_joined': int(s['n_pairs_joined']),
            # artifact frac is vs the universe (see module docstring) — carried through as-is
            'largest_cc_pct': round(100.0 * float(s['largest_cc_frac']), 2),
        })
    return pd.DataFrame(rows)


def plot_one(d: pd.DataFrame, *, pair: str, source: str, out_png: Path) -> None:
    """n_CCs bars (log-y, left) + largest-CC% line (right) vs threshold, one pair.

    Args:
        d: summary DataFrame from `load_summaries`, t-ordered strict-first.
        pair: schema pair label for the title (e.g. 'HA-NA').
        source: cluster source label for the title (e.g. 'nt_cds_ood').
        out_png: destination PNG; parent dirs are created.
    """
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    x = np.arange(len(d))

    ax.bar(x, d['n_ccs'], color=_BAR_COLOR, edgecolor='black', linewidth=0.5,
           width=0.62, zorder=2)
    ax.set_yscale('log')
    ax.set_ylim(1, d['n_ccs'].max() * 2.4)
    for xi, v in zip(x, d['n_ccs']):
        ax.annotate(f'{int(v):,}', xy=(xi, v), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7.5, color='#222', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(d['threshold'], fontsize=8)
    ax.set_xlabel('cluster identity threshold', fontsize=9)
    ax.set_ylabel('number of CCs (log)', fontsize=9, color=_BAR_COLOR)
    ax.tick_params(axis='y', labelcolor=_BAR_COLOR)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(d) - 0.4)

    ax2 = ax.twinx()
    ax2.plot(x, d['largest_cc_pct'], color=_LINE_COLOR, marker='o', markersize=4.5,
             linewidth=1.8, zorder=5)
    for xi, v in zip(x, d['largest_cc_pct']):
        # to the RIGHT of each marker (not above/below) so the % labels never
        # collide with the centered bar-top n_CCs labels, wherever the line
        # crosses the bars.
        ax2.annotate(f'{v:.0f}%', xy=(xi, v), xytext=(7, 0),
                     textcoords='offset points', ha='left', va='center',
                     fontsize=6.5, color=_LINE_LABEL, zorder=6,
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.4))
    ax2.set_ylim(0, 108)
    ax2.set_ylabel('largest-CC % of pairs', fontsize=9, color=_LINE_COLOR)
    ax2.tick_params(axis='y', labelcolor=_LINE_COLOR)

    suffix = '  ·  fragmented' if bool(d['fragmented'].iloc[0]) else ''
    ax.set_title(
        f'CC count vs identity threshold  ·  {pair} ({source}){suffix}  ·  '
        f'{int(d["n_pairs_universe"].iloc[0]):,} unique pairs', fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cc_dir', type=Path, nargs='+', required=True,
                   help='One or more cc_{source}/{pair} artifact dirs from build_cc_structure.py.')
    p.add_argument('--thresholds', nargs='+', default=None,
                   help='Restrict to these thresholds; default every tXXX found (strict-first).')
    p.add_argument('--fragmented', action='store_true',
                   help='Read tXXX/fragmented/cc_summary.json (post-edge-cut) instead of tXXX/.')
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/2D_cluster_sizes')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    tidy_all = []
    for cc_dir in args.cc_dir:
        cc_dir = Path(cc_dir)
        if not cc_dir.is_dir():
            raise FileNotFoundError(
                f'cc_dir not found: {cc_dir}\n'
                f'Generate it first: python src/datasets/build_cc_structure.py '
                f'--config_bundle <bundle> --thresholds t099 t098 t097 t096 t095'
                + (' --fragment' if args.fragmented else ''))

        d = load_summaries(cc_dir, args.thresholds, args.fragmented)
        if d.empty:
            print(f'WARNING: no cc_summary.json under {cc_dir} at the requested thresholds; skipping.')
            continue

        pair = d['schema_pair'].iloc[0]
        source = d['cc_source'].iloc[0]
        slug = pair.lower().replace('-', '_')
        tag = '_fragmented' if args.fragmented else ''
        out_png = out_dir / 'plots' / f'cc_count_vs_threshold_{slug}_{source}{tag}.png'
        plot_one(d, pair=pair, source=source, out_png=out_png)
        print(f'wrote {out_png}')
        tidy_all.append(d)

    if tidy_all:
        tidy = pd.concat(tidy_all, ignore_index=True)
        out_csv = out_dir / 'cc_count_vs_threshold.csv'
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(out_csv, index=False)
        print(f'wrote {out_csv} ({len(tidy)} rows)')
    print('\nDone.')


if __name__ == '__main__':
    main()
