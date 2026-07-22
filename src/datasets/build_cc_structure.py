"""Stage 2.5: persist the positive pair universe + per-t CC structure.

Builds the positive pair universe ONCE (t-invariant) for a (alphabet, schema-pair),
then layers each threshold's cluster/CC assignment on top, persisting reusable
artifacts so CC analysis -- and the Stage-3 builder -- no longer re-run the front-end.

Reuses the Stage-3 positive path (`build_frontend` + `create_positive_pairs_v2`), so the
universe matches `dataset_pairs_cc` exactly (nt_cds-correct; NOT the analysis
`load_pair_universe`, which does the protein-only dedup for nt_cds).

Artifacts (under data/processed/{virus}/{data_version}/):
    pair_universe_{alphabet}/{pair}/pairs.parquet          # t-invariant, cluster-independent
    cc_{source}/{pair}/tXXX/pairs_with_cc.parquet          # pair_key + cluster_id_a/b + cc_id (slim; join universe for the rest)
    cc_{source}/{pair}/tXXX/cc_sizes.csv                   # cc_id, n_pairs
    cc_{source}/{pair}/tXXX/cc_cluster_composition.csv     # cc_id, slot, cluster_id, n_pairs, pct_of_cc
    cc_{source}/{pair}/tXXX/cc_summary.json                # n_ccs, largest CC, floor, max_balanced_k
where {source} = the cluster dir name minus 'clusters_' (e.g. nt_cds_ood), so the
cluster source (ood vs set-cover) is encoded in the path.

CLI:
    python src/datasets/build_cc_structure.py \\
        --config_bundle flu_ha_na_cc_nt_cds_ood_edge_cut \\
        --thresholds t099 t098 t097 t095 [--rebuild]
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

from src.datasets._pair_helpers import bipartite_components  # noqa: E402
from src.datasets._split_helpers import attach_cluster_ids, load_cluster_lookup  # noqa: E402
from src.datasets.dataset_pairs_cc import _POS_HASH, _resolve_schema_pair, build_frontend  # noqa: E402
from src.datasets.dataset_segment_pairs_v2 import create_positive_pairs_v2  # noqa: E402
from src.utils.config_hydra import get_virus_config_hydra  # noqa: E402

# pairs_with_cc holds only the per-t assignment; join to the universe on pair_key for the rest
# (the full frame is ~144 MB/threshold, so we don't duplicate it 4x).
_PAIRS_WITH_CC_COLS = ['pair_key', 'cluster_id_a', 'cluster_id_b', 'cc_id']


def build_pair_universe(config, fa: str, fb: str, alphabet: str, processed_base: Path) -> pd.DataFrame:
    """The t-invariant positive universe via the Stage-3 front-end (so it matches the builder)."""
    input_file = processed_base / 'protein_final.parquet'
    cds_final_path = processed_base / 'cds_dna_final.parquet' if alphabet == 'nt_cds' else None
    df = build_frontend(config, input_file, (fa, fb), cds_final_path=cds_final_path)
    pos, _ = create_positive_pairs_v2(df, schema_pair=(fa, fb), pair_key_alphabet=alphabet)
    return pos


def cc_cluster_composition(pos_ids: pd.DataFrame) -> pd.DataFrame:
    """Long-form per-CC single-side-cluster composition: cc_id, slot, cluster_id, n_pairs, pct_of_cc.

    slot 'a' = the left (slot-A) clusters, slot 'b' = the right (slot-B) clusters. `pct_of_cc`
    is the cluster's share of that CC's pairs, so it exposes hub-dominance (one cluster at ~97%
    of a CC) that the CC-size barplot hides. `pct_of_cc` sums to 100 per (cc_id, slot).
    """
    rows = []
    for cc, g in pos_ids.groupby('cc_id'):
        ncc = len(g)
        for slot, col in (('a', 'cluster_id_a'), ('b', 'cluster_id_b')):
            for cid, n in g[col].value_counts().items():
                rows.append({'cc_id': int(cc), 'slot': slot, 'cluster_id': str(cid),
                             'n_pairs': int(n), 'pct_of_cc': round(100.0 * n / ncc, 3)})
    return pd.DataFrame(rows, columns=['cc_id', 'slot', 'cluster_id', 'n_pairs', 'pct_of_cc'])


def cc_summary(pos_ids: pd.DataFrame, threshold: str) -> dict:
    """CC-structure summary: sizes, the per-slot largest-cluster pair mass (edge-cut floor),
    and `max_balanced_k = floor(total / largest_cluster_pairs)` -- the most balanced folds
    achievable (a single cluster's pair mass cannot be split by edge-cut)."""
    n = int(len(pos_ids))
    sizes = pos_ids.groupby('cc_id').size().sort_values(ascending=False)
    a = pos_ids['cluster_id_a'].value_counts()
    b = pos_ids['cluster_id_b'].value_counts()
    a_pairs, b_pairs = int(a.iloc[0]), int(b.iloc[0])
    floor = max(a_pairs, b_pairs)
    return {
        'threshold': threshold, 'n_pairs': n, 'n_ccs': int(len(sizes)),
        'largest_cc_pairs': int(sizes.iloc[0]), 'largest_cc_frac': round(sizes.iloc[0] / n, 4),
        'largest_cluster_a': {'cluster_id': str(a.index[0]), 'pairs': a_pairs, 'frac': round(a_pairs / n, 4)},
        'largest_cluster_b': {'cluster_id': str(b.index[0]), 'pairs': b_pairs, 'frac': round(b_pairs / n, 4)},
        'floor_pairs': floor, 'floor_frac': round(floor / n, 4),
        'max_balanced_k': n // floor,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config_bundle', required=True,
                   help='Hydra bundle (must set dataset.split_strategy.cluster_alphabet + cluster_id_path).')
    p.add_argument('--thresholds', nargs='+', required=True, help='e.g. t099 t098 t097 t095')
    p.add_argument('--rebuild', action='store_true', help='rebuild the pair universe even if cached.')
    args = p.parse_args()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    ds, ss = config.dataset, config.dataset.split_strategy
    alphabet = str(ss.cluster_alphabet)
    hash_col = _POS_HASH[alphabet]
    fa, fb, sa, sb = _resolve_schema_pair(config, ds)
    pair = f'{sa}-{sb}'

    cluster_id_path = Path(str(ss.cluster_id_path))
    if not cluster_id_path.is_absolute():
        cluster_id_path = PROJ / cluster_id_path
    processed_base = cluster_id_path.parents[2]            # data/processed/{virus}/{data_version}
    clusters_root = cluster_id_path.parents[1]             # e.g. clusters_nt_cds_ood
    source = clusters_root.name.replace('clusters_', '')   # e.g. nt_cds_ood
    cluster_file = cluster_id_path.name                    # e.g. combined_cluster.parquet

    print(f'=== build_cc_structure {pair} {alphabet} (source={source}) ===')

    # 1. pair universe (t-invariant) -- built once, cached.
    uni_dir = processed_base / f'pair_universe_{alphabet}' / pair
    uni_dir.mkdir(parents=True, exist_ok=True)
    uni_file = uni_dir / 'pairs.parquet'
    if uni_file.exists() and not args.rebuild:
        pos = pd.read_parquet(uni_file)
        print(f'pair universe (cached): {len(pos):,} pairs -> {uni_file}')
    else:
        pos = build_pair_universe(config, fa, fb, alphabet, processed_base)
        pos.to_parquet(uni_file, index=False)
        print(f'pair universe: {len(pos):,} pairs -> {uni_file}')

    # 2-4. per-t CC structure.
    for t in args.thresholds:
        cp = clusters_root / t / cluster_file
        if not cp.exists():
            print(f'  {t}: MISSING {cp}; skipping.')
            continue
        lookup = load_cluster_lookup(cp)
        pos_ids, attach_audit = attach_cluster_ids(pos, lookup, pos_hash_col=hash_col)
        dropped = attach_audit['n_input'] - attach_audit['n_kept']
        component_id, _summ = bipartite_components(pos_ids, col_a='cluster_id_a', col_b='cluster_id_b')
        pos_ids = pos_ids.copy()
        pos_ids['cc_id'] = component_id.to_numpy()

        out = processed_base / f'cc_{source}' / pair / t
        out.mkdir(parents=True, exist_ok=True)
        pos_ids[_PAIRS_WITH_CC_COLS].to_parquet(out / 'pairs_with_cc.parquet', index=False)
        sizes = (pos_ids.groupby('cc_id').size().sort_values(ascending=False)
                 .rename('n_pairs'))
        sizes.index.name = 'cc_id'
        sizes.reset_index().to_csv(out / 'cc_sizes.csv', index=False)
        cc_cluster_composition(pos_ids).to_csv(out / 'cc_cluster_composition.csv', index=False)
        summ = cc_summary(pos_ids, t)
        (out / 'cc_summary.json').write_text(json.dumps(summ, indent=2))
        drop_note = f' (dropped {dropped:,} on cluster join)' if dropped else ''
        print(f'  {t}: {summ["n_ccs"]:,} CCs; largest {100*summ["largest_cc_frac"]:.1f}%; '
              f'floor {summ["largest_cluster_b"]["cluster_id"]}={100*summ["largest_cluster_b"]["frac"]:.1f}% / '
              f'{summ["largest_cluster_a"]["cluster_id"]}={100*summ["largest_cluster_a"]["frac"]:.1f}%; '
              f'maxK={summ["max_balanced_k"]}{drop_note} -> {out}')


if __name__ == '__main__':
    main()
