"""Per-CC metadata composition bars (one figure per field) for the fragmented CC structure.

Sibling of plot_cc_composition.py: that shows per-CC single-side CLUSTER composition; this shows
per-CC METADATA composition (hn_subtype / host / year). Metadata is a per-pair (isolate) property --
for a positive pair the two slots share the isolate -- so it is NOT split by slot: one figure per
field, each CC a stacked bar over its positive pairs' modal metadata. Recomputes nothing new; reuses
`_pair_helpers.pair_key_to_metadata` (modal metadata per pair_key) + `plot_utils.barplot_title` /
`stacked_composition_barplot`. The lean replacement for `bigraph_pair_metadata.py`'s plotting.

CLI:
    python -m src.analysis.plot_cc_metadata \\
        --cc_dir data/processed/flu/July_2025/cc_nt_cds_ood/HA-NA/t095/fragmented \\
        --pair HA-NA --alphabet nt_cds --threshold_id t095 [--top_n 12] [--fields hn_subtype host year]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._pair_helpers import pair_key_to_metadata  # noqa: E402
from src.utils import schema  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402
from src.utils.plot_utils import barplot_title, stacked_composition_barplot  # noqa: E402


def _find_final(cc_dir: Path, file_basename: str) -> Path:
    """The `<file_basename>.parquet` at or above `cc_dir` (the preprocessing base sits above cc_*/)."""
    for parent in (cc_dir, *cc_dir.parents):
        cand = parent / f'{file_basename}.parquet'
        if cand.exists():
            return cand
    raise SystemExit(f"could not find {file_basename}.parquet at/above {cc_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cc_dir', type=Path, required=True,
                   help='cc_{source}/{pair}/tXXX[/fragmented] dir with pairs_with_cc.parquet + '
                        'cc_sizes.csv + cc_summary.json')
    p.add_argument('--pair', required=True, help='slot pair, e.g. HA-NA (title only)')
    p.add_argument('--alphabet', required=True, help='aa / nt_cds / nt_ctg (selects the hash + *_final)')
    p.add_argument('--threshold_id', required=True, help='tXXX (title only)')
    p.add_argument('--fields', nargs='+', default=['hn_subtype', 'host', 'year'],
                   help='metadata fields, one figure each (default hn_subtype host year)')
    p.add_argument('--top_n', type=int, default=12, help='number of largest CCs to draw (default 12)')
    p.add_argument('--top_k', type=int, default=6, help='top categories per bar before Others (default 6)')
    p.add_argument('--note', default='', help='optional extra title line (e.g. "fragmented (3.9% dropped)")')
    p.add_argument('--normalize', action='store_true',
                   help='100%%-stacked bars (share of CC) instead of real pair counts (default off)')
    p.add_argument('--final', type=Path, default=None,
                   help='override *_final parquet (default: found at/above --cc_dir)')
    args = p.parse_args()

    sch = schema.require(args.alphabet)
    final_path = args.final or _find_final(args.cc_dir, sch.file_basename)
    summ = json.loads((args.cc_dir / 'cc_summary.json').read_text())
    sa, sb = summ['slot_a'], summ['slot_b']  # robust to hyphenated short names (PB1-F2, PA-X)
    short_to_full = load_function_metadata(PROJ / 'conf' / 'virus' / 'flu.yaml').short_to_function
    fa, fb = short_to_full[sa], short_to_full[sb]

    pw = pd.read_parquet(args.cc_dir / 'pairs_with_cc.parquet', columns=['pair_key', 'cc_id'])
    sizes = pd.read_csv(args.cc_dir / 'cc_sizes.csv')
    top_ccs = sizes.sort_values('n_pairs', ascending=False).head(args.top_n)['cc_id'].tolist()

    meta = pair_key_to_metadata(final_path, fa, fb, hash_col=sch.hash_col, fields=tuple(args.fields))
    m = pw.merge(meta, on='pair_key', how='left')

    ylabel = 'share of CC' if args.normalize else 'positive pairs in CC'
    n_pairs = int(sizes['n_pairs'].sum())
    largest_pct = 100.0 * sizes['n_pairs'].max() / n_pairs if n_pairs else 0.0
    stats = (f'top {len(top_ccs)} of {len(sizes):,} CCs  ·  {n_pairs:,} positive pairs  ·  '
             f'largest CC {largest_pct:.1f}%')
    for field in args.fields:
        comp = (m.assign(**{field: m[field].fillna('unknown')})
                .groupby(['cc_id', field]).size().reset_index(name='n_pairs'))
        out = (args.cc_dir / 'figures' /
               f'cc_metadata_{field}_{args.pair}_{args.alphabet}_{args.threshold_id}.png')
        title = barplot_title(
            args.pair, args.alphabet, args.threshold_id, stats,
            descriptor=f'{field} composition per CC', note=args.note)
        stacked_composition_barplot(
            comp, item_col='cc_id', category_col=field, value_col='n_pairs',
            item_order=top_ccs, item_labels=[f'CC{i + 1}' for i in range(len(top_ccs))],
            out_png=out, normalize=args.normalize, title=title, top_k=args.top_k, rotation=0,
            xlabel='connected component (sorted by size; not true CC ids)',
            ylabel=ylabel)
        print(f'  wrote {out}')


if __name__ == '__main__':
    main()
