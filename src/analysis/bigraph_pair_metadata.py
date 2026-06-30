"""Per-CC metadata composition of the cluster bigraph (hn_subtype / host / year).

The third member of the 2D cluster-structure family, alongside
`bigraph_pair_feasibility.py` (2D_cluster_sizes) and `bigraph_pair_2d.py`
(2D_cluster_maps). Those show *how big* the connected components (CCs) are and
*where* their cells sit; this shows *what they are made of*.

For a given (schema_pair, alphabet, t): build the cluster-level bigraph (side A =
slot-a clusters, side B = slot-b clusters, an edge per unique canonical positive
pair), rank CCs by pair count (the SAME order as the 2D_cluster_sizes barplot),
and draw the top-N CCs. Each CC gets **three side-by-side bars** — hn_subtype,
host, year — each STACKED by the composition of one metadata field. By default
the bars are normalized to 100% (every CC full-height) so composition stays
readable even when one CC dominates, with the absolute pair count printed above;
pass --absolute to make bar height the raw pair count (bar tops then match the
2D_cluster_sizes bars exactly). A subtype-pure CC is a solid bar; a
reassortant / mixed CC is multi-color — dominant value and purity at a glance.

Attribution unit — per UNIQUE canonical pair (the bigraph's edge / sampling
unit), NOT per sequence. Each pair's metadata is the MODAL (hn_subtype, host,
year) over the isolates in which that exact protein pair co-occurs
(`pair_key_to_metadata`, generalizing `bigraph_cut_subtype.pair_key_to_subtype`
from one field to three). This conditions on BOTH proteins, so it differs from a
per-protein `prot_hash -> cluster_membership` lookup (one HA sequence pairs with
many NA subtypes); the per-pair modal is the right unit for a pair.

The pair universe is t-invariant (the same pairs at every threshold; only the CC
structure changes), so the per-field category palette is fixed across the whole
sweep and colors are comparable t-to-t.

CLI:
    python -m src.analysis.bigraph_pair_metadata \\
        [--schema_pairs HA-NA PB2-PB1] \\
        [--alphabets aa] \\
        [--thresholds t100 t099 ... t090] \\
        [--top_n 10] \\
        [--absolute] \\
        [--out_dir results/flu/July_2025/runs/2D_cluster_metadata]

By default each CC is a 100%-normalized stacked bar (composition stays readable
even when one CC dominates); pass --absolute for raw pair-count heights (bar tops
then match the 2D_cluster_sizes bars exactly).

Outputs (under --out_dir):
    plots/barplot_{a}_{b}_{alphabet}_{tXXX}_norm.png  normalized (default)
    plots/barplot_{a}_{b}_{alphabet}_{tXXX}.png       absolute (--absolute)
    metadata_composition_top{N}.csv               long-form: schema_pair,
                                                  alphabet, threshold, cc_rank,
                                                  cc_pairs, field, category,
                                                  count, pct_of_cc
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._pair_helpers import canonical_pair_key  # noqa: E402
from src.utils.metadata_enrichment import load_flu_metadata  # noqa: E402
from src.analysis.cluster_pair_weight_topk import (  # noqa: E402
    load_pair_universe, load_cluster_map, _FUNCTION_TO_SHORT,
)
from src.analysis.bigraph_properties import build_bipartite_multigraph  # noqa: E402

_DEFAULT_PAIRS = ['HA-NA', 'PB2-PB1']
_DEFAULT_ALPHABETS = ['aa']
_DEFAULT_THRESHOLDS = [f't{i:03d}' for i in range(100, 89, -1)]  # t100..t090
_ROOT = {'aa': PROJ / 'data/processed/flu/July_2025/clusters_aa',
         'nt_cds': PROJ / 'data/processed/flu/July_2025/clusters_nt_cds'}

# Metadata fields drawn per CC, in left-to-right bar order. The short tag is the
# label printed under each bar; the legend title is the full field name.
_FIELDS = ['hn_subtype', 'host', 'year']
_FIELD_TAG = {'hn_subtype': 'sub', 'host': 'host', 'year': 'yr'}
# Per-field top-K category cap (rest collapse to 'other'); 'unknown' is always
# kept separate. subtype/year are long-tailed; host is short. Kept small enough
# that the three stacked right-margin legends fit a single figure.
_FIELD_K = {'hn_subtype': 12, 'host': 8, 'year': 10}
# year is colored chronologically (sequential cmap); the others categorically.
_FIELD_SEQUENTIAL = {'hn_subtype': False, 'host': False, 'year': True}

_OTHER, _UNK = 'other', 'unknown'
_OTHER_COLOR, _UNK_COLOR = '#bdbdbd', '#e8e8e8'


def _threshold_decimal(threshold_id: str) -> float:
    return int(threshold_id[1:]) / 100.0


def pair_key_to_metadata(cds_final: Path, slot_a: str, slot_b: str,
                         fields=_FIELDS) -> pd.DataFrame:
    """Modal (hn_subtype, host, year) per canonical pair_key from isolate co-occurrence.

    Generalizes `bigraph_cut_subtype.pair_key_to_subtype` from one field to
    several. Builds the isolate-level co-occurrence of slot_a and slot_b (one row
    per isolate carrying both), keys each by `canonical_pair_key` on protein
    hashes — the SAME pair_key as `load_pair_universe` — joins isolate metadata,
    and reduces each pair to the modal value of every requested field. Missing
    metadata becomes the explicit category 'unknown'; year is stringified
    ('2018', or 'unknown') so it stacks like the categorical fields.

    Returns a DataFrame with columns: pair_key + one column per field.
    """
    cds = pd.read_parquet(cds_final, columns=['assembly_id', 'function', 'prot_hash'])
    cds['fs'] = cds['function'].map(_FUNCTION_TO_SHORT)
    a = (cds[cds['fs'] == slot_a][['assembly_id', 'prot_hash']]
         .rename(columns={'prot_hash': 'hash_a'}))
    b = (cds[cds['fs'] == slot_b][['assembly_id', 'prot_hash']]
         .rename(columns={'prot_hash': 'hash_b'}))
    iso = a.merge(b, on='assembly_id')
    iso['pair_key'] = [canonical_pair_key(x, y)
                       for x, y in zip(iso['hash_a'], iso['hash_b'])]

    meta = load_flu_metadata()[['assembly_id', 'hn_subtype', 'host', 'year']].copy()
    meta['assembly_id'] = meta['assembly_id'].astype(str)
    iso['assembly_id'] = iso['assembly_id'].astype(str)
    iso = iso.merge(meta, on='assembly_id', how='left')

    # Normalize every field to a plain string category with an explicit 'unknown'.
    iso['hn_subtype'] = iso['hn_subtype'].fillna(_UNK).astype(str)
    iso['host'] = iso['host'].fillna(_UNK).astype(str)
    iso['year'] = iso['year'].apply(lambda v: _UNK if pd.isna(v) else str(int(v)))

    out = pd.DataFrame({'pair_key': sorted(iso['pair_key'].unique())}).set_index('pair_key')
    for f in fields:
        # Modal category per pair_key, deterministic tie-break (count desc, then
        # label asc): groupby-size -> sort -> keep first.
        g = (iso.groupby(['pair_key', f]).size().reset_index(name='n')
             .sort_values(['pair_key', 'n', f], ascending=[True, False, True])
             .drop_duplicates('pair_key', keep='first')
             .set_index('pair_key')[f])
        out[f] = g
    return out.reset_index()


def cc_rank_order(G: nx.MultiGraph) -> tuple[dict, list[int]]:
    """Map each node to its CC rank (0 = largest by pair count) and return CC sizes.

    Ranks CCs by induced multigraph edge count (= unique pairs in the CC) in one
    O(E) pass — identical to `bigraph_pair_feasibility.cc_pair_sizes`, so the CC
    order matches the 2D_cluster_sizes barplot.
    """
    comps = list(nx.connected_components(G))
    node_cc0 = {n: i for i, c in enumerate(comps) for n in c}
    cnt: Counter = Counter()
    for u, v in G.edges():
        cnt[node_cc0[u]] += 1  # endpoints share a component
    order = sorted(range(len(comps)), key=lambda i: cnt.get(i, 0), reverse=True)
    rank_of = {old: new for new, old in enumerate(order)}
    node_cc = {n: rank_of[c] for n, c in node_cc0.items()}
    sizes = [cnt.get(order[r], 0) for r in range(len(order))]
    return node_cc, sizes


def category_scheme(values: pd.Series, *, k: int, sequential: bool):
    """Top-k category mapping + colors for one field over the WHOLE slice.

    Returns (mapper, ordered, colors):
      mapper:  raw value -> display category (top-k value | 'other' | 'unknown')
      ordered: display categories in legend / stack order (kept, then 'other',
               then 'unknown')
      colors:  display category -> RGBA / hex
    Categorical fields keep the k most frequent values and color them with tab20;
    sequential fields (year) keep the k most frequent, then order and color them
    chronologically with viridis (old = dark, recent = yellow).
    """
    vc = values.value_counts()
    have_unknown = _UNK in vc.index
    ranked = [c for c in vc.index if c != _UNK]
    keep = ranked[:k]

    if sequential:
        keep = sorted(keep, key=lambda c: int(c))
        cmap = plt.get_cmap('viridis')
        colors = {c: cmap(i / max(len(keep) - 1, 1)) for i, c in enumerate(keep)}
    else:
        cmap = plt.get_cmap('tab20')
        colors = {c: cmap(i % 20) for i, c in enumerate(keep)}

    keep_set = set(keep)
    mapper = {c: (c if c in keep_set else _OTHER) for c in ranked}
    if have_unknown:
        mapper[_UNK] = _UNK

    ordered = list(keep)
    if any(v == _OTHER for v in mapper.values()):
        ordered.append(_OTHER)
        colors[_OTHER] = _OTHER_COLOR
    if have_unknown:
        ordered.append(_UNK)
        colors[_UNK] = _UNK_COLOR
    return mapper, ordered, colors


def plot_metadata_composition(
    cc_field_mats: dict,
    schemes: dict,
    sizes: list[int],
    *,
    pair_label: str,
    alphabet: str,
    threshold_id: str,
    top_n: int,
    n_ccs: int,
    n_pairs: int,
    normalize: bool = False,
    out_png: Path,
) -> None:
    """Grouped-stacked composition barplot: per CC, one stacked bar per field.

    Args:
        cc_field_mats: field -> DataFrame (index = cc_rank 0..N-1, columns =
            display categories, values = unique-pair counts). Each row sums to
            that CC's pair count.
        schemes: field -> (mapper, ordered, colors) from category_scheme.
        sizes: per-CC pair counts (descending), full list; first top_n are drawn.
        normalize: if True, draw each CC full-height as a 100%-stacked bar
            (composition only — readable even when one CC dominates); the
            absolute pair count is kept as a label above each group. If False
            (default), bar height is the absolute pair count and the bar tops
            match the 2D_cluster_sizes bars exactly.
    """
    n = min(top_n, len(sizes))
    xs = np.arange(n)
    group_w = 0.82
    bw = group_w / len(_FIELDS)
    offsets = {f: (j - (len(_FIELDS) - 1) / 2) * bw for j, f in enumerate(_FIELDS)}

    fig_w = max(12.0, n * 1.05 + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 7.2))

    # In normalize mode every CC row is divided by its pair count so the three
    # bars in every group reach 1.0 (100%) — composition stays readable no
    # matter how skewed the CC sizes are.
    denom = (np.array([sizes[r] if sizes[r] else 1 for r in range(n)], dtype=float)
             if normalize else None)

    for f in _FIELDS:
        _, ordered, colors = schemes[f]
        mat = cc_field_mats[f].reindex(index=range(n), columns=ordered, fill_value=0).astype(float)
        if normalize:
            mat = mat.div(denom, axis=0)
        xo = xs + offsets[f]
        bottoms = np.zeros(n)
        for cat in ordered:
            h = mat[cat].to_numpy()
            ax.bar(xo, h, bottom=bottoms, width=bw * 0.9, color=colors[cat],
                   edgecolor='white', linewidth=0.2)
            bottoms += h
        # field tag under each of this field's bars
        for x in xo:
            ax.annotate(_FIELD_TAG[f], xy=(x, 0), xytext=(0, -2),
                        textcoords='offset points', ha='center', va='top',
                        fontsize=6, color='#555', annotation_clip=False)

    # CC pair-count label above each group (all three bars share this height;
    # in normalize mode the bars top out at 1.0 so the count rides above that).
    y_label = 1.0 if normalize else None
    for r in range(n):
        ytop = y_label if normalize else sizes[r]
        ax.annotate(f'{int(sizes[r]):,}', xy=(xs[r], ytop), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, color='#222')

    ax.set_xticks(xs)
    ax.set_xticklabels([f'CC{r + 1}' for r in range(n)], fontsize=8)
    ax.tick_params(axis='x', length=0, pad=16)  # pad leaves room for the field tags
    ax.set_xlabel('connected component (rank-ordered, largest first)', fontsize=9)
    if normalize:
        ax.set_ylabel('share of CC pairs (stacked by modal metadata)', fontsize=9)
        ax.set_ylim(0, 1.085)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '25%', '50%', '75%', '100%'])
    else:
        ax.set_ylabel('unique pairs in CC (stacked by modal metadata)', fontsize=9)
        ax.set_ylim(0, max(sizes[:n]) * 1.12)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, n - 0.4)

    mode_note = (
        'each group = 3 bars (subtype · host · year) normalized to 100%; count above bar'
        if normalize else
        'each group = 3 bars (subtype · host · year), same height = CC pair count, stacked by modal value'
    )
    ax.set_title(
        f'{pair_label} — {alphabet} — {threshold_id} (id={_threshold_decimal(threshold_id):.2f})  ·  '
        f'per-CC metadata composition\n'
        f'top {n} of {n_ccs:,} CCs  ·  {n_pairs:,} pairs  ·  {mode_note}',
        fontsize=10,
    )

    # Reserve the right margin for the three stacked legends, then place them in
    # figure coordinates so the stacking never depends on the (shrinking) axes
    # box. Two columns keep the long subtype/year lists short enough to fit.
    fig.subplots_adjust(left=0.06, right=0.69, top=0.86, bottom=0.13)
    x_leg, y_top = 0.71, 0.86
    title_h, row_h, gap, ncol = 0.034, 0.030, 0.030, 2
    for f in _FIELDS:
        _, ordered, colors = schemes[f]
        handles = [Patch(facecolor=colors[c], edgecolor='white', label=str(c))
                   for c in ordered]
        fig.legend(handles=handles, title=f, loc='upper left',
                   bbox_to_anchor=(x_leg, y_top), ncol=ncol, fontsize=7,
                   title_fontsize=8, frameon=False, columnspacing=0.9,
                   handlelength=1.0, handleheight=1.0, labelspacing=0.25,
                   alignment='left')
        y_top -= title_h + row_h * math.ceil(len(handles) / ncol) + gap

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet'))
    p.add_argument('--schema_pairs', nargs='+', default=_DEFAULT_PAIRS,
                   help='Pairs as A-B (default: HA-NA PB2-PB1).')
    p.add_argument('--alphabets', nargs='+', default=_DEFAULT_ALPHABETS,
                   choices=['aa', 'nt_cds'])
    p.add_argument('--thresholds', nargs='+', default=_DEFAULT_THRESHOLDS)
    p.add_argument('--top_n', type=int, default=10,
                   help='Number of largest CCs to draw per slice (default 10).')
    p.add_argument('--absolute', action='store_true',
                   help='Use absolute pair-count bar height (bar tops match the '
                        '2D_cluster_sizes bars) instead of the default per-CC 100%% '
                        'normalization. Normalized (default) plots get a "_norm" filename '
                        'suffix; absolute plots are bare-named.')
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/2D_cluster_metadata')
    args = p.parse_args()

    # Normalize is the default view (composition stays readable when one CC
    # dominates); --absolute opts into raw pair-count heights.
    normalize = not args.absolute

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    universe_cache: dict[str, pd.DataFrame] = {}
    meta_cache: dict[str, pd.DataFrame] = {}
    scheme_cache: dict[str, dict] = {}
    long_rows: list[dict] = []
    n_plots = 0

    for pair in args.schema_pairs:
        slot_a, slot_b = pair.split('-')
        if pair not in universe_cache:
            universe_cache[pair] = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
            meta_cache[pair] = pair_key_to_metadata(Path(args.cds_final), slot_a, slot_b)
        universe = universe_cache[pair]
        # universe + per-pair modal metadata (t-invariant); colors fixed per pair.
        u_meta = universe.merge(meta_cache[pair], on='pair_key', how='left')
        for f in _FIELDS:
            u_meta[f] = u_meta[f].fillna(_UNK)
        if pair not in scheme_cache:
            sch = {f: category_scheme(u_meta[f], k=_FIELD_K[f],
                                      sequential=_FIELD_SEQUENTIAL[f]) for f in _FIELDS}
            scheme_cache[pair] = sch
            for f in _FIELDS:
                mapper, _, _ = sch[f]
                u_meta[f + '__cat'] = u_meta[f].map(mapper)
        else:
            for f in _FIELDS:
                mapper, _, _ = scheme_cache[pair][f]
                u_meta[f + '__cat'] = u_meta[f].map(mapper)
        schemes = scheme_cache[pair]

        for alphabet in args.alphabets:
            for t in args.thresholds:
                cmap_a = load_cluster_map(_ROOT[alphabet], slot_a, t)
                cmap_b = load_cluster_map(_ROOT[alphabet], slot_b, t)
                if not cmap_a or not cmap_b:
                    print(f"  [{pair} {alphabet} {t}] missing cluster map; skipping.")
                    continue
                G, n_unmapped = build_bipartite_multigraph(universe, cmap_a, cmap_b, alphabet)
                if n_unmapped:
                    print(f"  WARNING: {pair} {alphabet} {t} dropped {n_unmapped} unmapped pairs.")
                n_pairs = G.number_of_edges()
                node_cc, sizes = cc_rank_order(G)
                n_ccs = len(sizes)

                # assign each universe pair its CC rank via its slot-a node
                hash_a = 'prot_hash_a' if alphabet == 'aa' else 'cds_dna_hash_a'
                u = u_meta.copy()
                u['cc'] = ('a:' + u[hash_a].map(cmap_a).astype(str)).map(node_cc)
                u = u.dropna(subset=['cc'])
                u['cc'] = u['cc'].astype(int)

                n = min(args.top_n, n_ccs)
                top = u[u['cc'] < n]
                cc_field_mats = {}
                for f in _FIELDS:
                    mat = (top.groupby(['cc', f + '__cat']).size()
                           .unstack(fill_value=0))
                    cc_field_mats[f] = mat
                    # long-form rows
                    for cc_rank in range(n):
                        row = mat.loc[cc_rank] if cc_rank in mat.index else None
                        for cat in (row.index if row is not None else []):
                            cnt = int(row[cat])
                            if cnt == 0:
                                continue
                            long_rows.append({
                                'schema_pair': pair, 'alphabet': alphabet, 'threshold': t,
                                'cc_rank': cc_rank + 1, 'cc_pairs': int(sizes[cc_rank]),
                                'field': f, 'category': cat, 'count': cnt,
                                'pct_of_cc': round(100.0 * cnt / sizes[cc_rank], 4),
                            })

                slug = f'{slot_a.lower()}_{slot_b.lower()}'
                suffix = '_norm' if normalize else ''
                out_png = plots_dir / f'barplot_{slug}_{alphabet}_{t}{suffix}.png'
                plot_metadata_composition(
                    cc_field_mats, schemes, sizes,
                    pair_label=pair, alphabet=alphabet, threshold_id=t,
                    top_n=args.top_n, n_ccs=n_ccs, n_pairs=n_pairs,
                    normalize=normalize, out_png=out_png)
                n_plots += 1
                print(f"  [{pair} {alphabet} {t}] {n_pairs:,} pairs, {n_ccs:,} CCs, "
                      f"top{n} drawn; wrote {out_png.name}")

    if long_rows:
        csv = out_dir / f'metadata_composition_top{args.top_n}.csv'
        pd.DataFrame(long_rows).to_csv(csv, index=False)
        print(f"\nwrote {csv} ({len(long_rows):,} rows)")
    print(f"\nDone. {n_plots} barplot(s).")


if __name__ == '__main__':
    main()
