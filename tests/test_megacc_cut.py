"""Tests for the modular edge-min-cut primitives in `src/datasets/_megacc_cut.py`
(`build_pair_bigraph` / `fragment_largest_cc` / `edges_to_row_index` /
`fragment_once`) and the behavior-preserving `apply_drop_budget_cut` refactor
(Phase R of docs/plans/2026-07-17_2d_cc_edge_cut_fragmentation_plan.md).

Fast synthetic tests run everywhere. The OOD tests reproduce the P1 / pre-refactor
numbers on the production OOD nt_cds clusters and SKIP when that data is absent.

Run: pytest tests/test_megacc_cut.py   (or: python tests/test_megacc_cut.py)
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._megacc_cut import (
    apply_drop_budget_cut,
    build_pair_bigraph,
    edges_to_row_index,
    fragment_largest_cc,
    fragment_once,
)

GOLDEN = PROJ / 'tests' / 'golden' / 'megacc_cut' / 'ood_nt_cds_t099.json'
OOD_CLUSTERS = PROJ / 'data' / 'processed' / 'flu' / 'July_2025' / 'clusters_nt_cds_ood'

# Synthetic fixture: two K(2,2) blobs joined by ONE bridge edge. The bridge is the
# unique min-cut, so a single bisection must drop exactly it.
_BLOB1 = [('H1', 'N1'), ('H1', 'N1'), ('H1', 'N2'), ('H2', 'N1'), ('H2', 'N2')]  # (H1,N1) weight 2
_BLOB2 = [('H3', 'N3'), ('H3', 'N4'), ('H4', 'N3'), ('H4', 'N4')]
_BRIDGE = [('H2', 'N3')]  # the only edge between blob1 and blob2


def _pos(rows):
    """Synthetic pos_with_ids: cluster_id_a/b + a unique pair_key per row."""
    return pd.DataFrame({
        'cluster_id_a': [a for a, _ in rows],
        'cluster_id_b': [b for _, b in rows],
        'pair_key': [f'{a}|{b}|{i}' for i, (a, b) in enumerate(rows)],
    })


def _two_blobs():
    return _pos(_BLOB1 + _BLOB2 + _BRIDGE)  # 10 rows; mega-CC = all 8 nodes


# --- build_pair_bigraph ------------------------------------------------------
def test_build_pair_bigraph_weight_is_row_count():
    H, edge_rows = build_pair_bigraph(_two_blobs())
    assert H.number_of_nodes() == 8 and H.number_of_edges() == 9
    assert H['a:H1']['b:N1']['weight'] == 2     # (H1,N1) carried 2 rows
    assert H.size(weight='weight') == 10         # sum of edge weights == #rows
    assert len(edge_rows[('a:H1', 'b:N1')]) == 2


def test_build_pair_bigraph_edge_rows_partition_rows():
    pos = _two_blobs()
    _H, edge_rows = build_pair_bigraph(pos)
    all_rows = [i for rows in edge_rows.values() for i in rows]
    assert sorted(all_rows) == list(pos.index)   # every row assigned to exactly one edge


# --- fragment_largest_cc -----------------------------------------------------
def test_fragment_largest_cc_finds_bridge():
    H, _ = build_pair_bigraph(_two_blobs())
    step = fragment_largest_cc(H, method='spectral', seed=1)
    assert step.dropped_pairs == 1 and len(step.cross_edges) == 1   # the weight-1 bridge
    assert frozenset(step.cross_edges[0]) == {'a:H2', 'b:N3'}
    part_b = step.cc_nodes - step.part_a
    assert step.part_a and part_b                                   # both sides non-empty
    assert step.part_a | part_b == step.cc_nodes                    # a partition of the CC
    assert step.part_a.isdisjoint(part_b)


def test_fragment_largest_cc_does_not_mutate():
    H, _ = build_pair_bigraph(_two_blobs())
    before = H.number_of_edges()
    fragment_largest_cc(H, seed=1)
    assert H.number_of_edges() == before                           # no graph mutation


# --- edges_to_row_index ------------------------------------------------------
def test_edges_to_row_index_canonicalizes_orientation():
    _H, edge_rows = build_pair_bigraph(_two_blobs())
    fwd = edges_to_row_index([('a:H2', 'b:N3')], edge_rows)
    rev = edges_to_row_index([('b:N3', 'a:H2')], edge_rows)          # reversed endpoints
    assert fwd == rev == edge_rows[('a:H2', 'b:N3')]


# --- fragment_once -----------------------------------------------------------
def test_fragment_once_splits_off_the_bridge():
    pos = _two_blobs()
    kept, dropped, step = fragment_once(pos, seed=1)
    assert len(dropped) == 1 and len(kept) == len(pos) - 1
    assert (dropped['cluster_id_a'].iloc[0], dropped['cluster_id_b'].iloc[0]) == ('H2', 'N3')
    assert step.dropped_pairs == 1


def test_fragment_once_deterministic():
    pos = _two_blobs()
    a = fragment_once(pos, seed=1)[2]
    b = fragment_once(pos, seed=1)[2]
    assert a.part_a == b.part_a and a.cross_edges == b.cross_edges


# --- apply_drop_budget_cut (synthetic no-cut path) ---------------------------
def test_apply_drop_budget_cut_none_is_noop():
    pos = _two_blobs()
    kept, audit = apply_drop_budget_cut(pos, cut_method='none')
    assert len(kept) == len(pos) and audit['n_cuts'] == 0 and audit['pairs_dropped'] == 0


# --- OOD integration (skip when the production clusters are absent) -----------
def _build_ood_pos_ids(threshold):
    """Uncapped production pos_ids for OOD nt_cds HA-NA at `threshold` (reuses
    dataset_pairs_cc._build_positives -- exactly what P1/P2 route)."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from src.datasets.dataset_pairs_cc import _build_positives, _resolve_spec
    from src.utils.config_hydra import get_virus_config_hydra

    bundle = 'flu_ha_na_cc_nt_cds_ood'
    cpath = f'data/processed/flu/July_2025/clusters_nt_cds_ood/{threshold}/combined_cluster.parquet'
    cfg = get_virus_config_hydra(bundle, config_path=str(PROJ / 'conf'))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(
        [f'dataset.split_strategy.cluster_id_path={cpath}']))
    args = SimpleNamespace(config_bundle=bundle, protein_final=None, override=None, out_dir=None)
    _df, pos_ids, _cooccur = _build_positives(cfg, _resolve_spec(args, cfg), args)
    return pos_ids


def test_fragment_once_ood_nt_cds_t095_reproduces_p1():
    """P1 anchor: one spectral cut of the OOD nt_cds t095 mega-CC drops 14 straddling
    pairs and splits 440 clusters into 297 / 143."""
    if not (OOD_CLUSTERS / 't095' / 'combined_cluster.parquet').exists():
        print('SKIP test_fragment_once_ood_nt_cds_t095: OOD clusters absent')
        return
    pos = _build_ood_pos_ids('t095')
    kept, dropped, step = fragment_once(pos, cut_method='spectral', seed=1)
    part_b = len(step.cc_nodes) - len(step.part_a)
    assert step.dropped_pairs == 14 and len(step.cross_edges) == 8
    assert len(step.cc_nodes) == 440 and {len(step.part_a), part_b} == {297, 143}
    assert len(dropped) == 14 and len(kept) == 78750


def test_apply_drop_budget_cut_ood_nt_cds_t099_golden():
    """Behavior-preserving guard: the budget loop on OOD nt_cds t099 reproduces the
    pre-refactor digest (cut count, dropped pairs, kept/dropped pair_key sets)."""
    if not (OOD_CLUSTERS / 't099' / 'combined_cluster.parquet').exists():
        print('SKIP test_apply_drop_budget_cut_ood_nt_cds_t099_golden: OOD clusters absent')
        return
    g = json.loads(GOLDEN.read_text())
    pos = _build_ood_pos_ids('t099')
    kept, audit = apply_drop_budget_cut(pos, cut_method='spectral', seed=1)

    def _sha(o):
        return hashlib.sha256(repr(o).encode()).hexdigest()

    assert len(kept) == g['n_kept']
    assert audit['n_cuts'] == g['n_cuts'] and audit['pairs_dropped'] == g['pairs_dropped']
    assert audit['n_atoms_after'] == g['n_atoms_after']
    assert _sha(sorted(kept['pair_key'].astype(str))) == g['kept_pairkeys_sha256']
    assert _sha(sorted(audit['dropped_pair_keys'])) == g['dropped_pairkeys_sha256']


if __name__ == '__main__':
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            print(f'{_name} ...', flush=True)
            _fn()
    print('Done. All tests passed.')
