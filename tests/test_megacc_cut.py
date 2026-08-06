"""Tests for the edge-min-cut primitives in `src/datasets/_megacc_cut.py`:
`build_pair_bigraph`, `fragment_largest_cc`, `edges_to_row_index`, `fragment_once`,
`fragment_until` and `fragment_to_targets`.

Fast synthetic tests run everywhere. The OOD tests pin the cut against the production
OOD nt_cds clusters and SKIP when that data is absent.

Run: pytest tests/test_megacc_cut.py   (or: python tests/test_megacc_cut.py)
"""
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._megacc_cut import (
    _largest_cc,
    build_pair_bigraph,
    edges_to_row_index,
    fragment_largest_cc,
    fragment_once,
    fragment_to_targets,
    fragment_until,
    stop_at_n_atoms,
)
from src.datasets._pair_helpers import cluster_ccs, sequence_ccs

FRAG_GOLDEN = PROJ / 'tests' / 'golden' / 'megacc_cut' / 'ood_nt_cds_t095_fragment_until.json'
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


def test_build_pair_bigraph_rejects_duplicate_index():
    """A repeated row label would make `.drop(index=...)` / `.loc[...]` over-drop silently."""
    pos = _two_blobs()
    pos.index = [0] * len(pos)
    try:
        build_pair_bigraph(pos)
    except AssertionError as e:
        assert 'unique' in str(e)
    else:
        raise AssertionError('build_pair_bigraph accepted a duplicated index')


# --- cluster_ccs vs sequence_ccs (edge-set agreement) ------------------------
# The two CC builders derive the same bigraph edge set by independent code paths: `cluster_ccs`
# via networkx (`build_pair_bigraph`), `sequence_ccs` via hand-rolled union-find. `cluster_ccs`
# labels the atoms the router packs and `build_pair_bigraph` builds the graph the edge min-cut
# bisects, so a disagreement would mean the routed atoms are not the components that were cut.
# These pin the agreement; `sequence_ccs`'s col_a/col_b mode exists for exactly this comparison.
def _partition(labels):
    """The row partition a labeling induces, ignoring the label values themselves."""
    return {frozenset(g.index) for _, g in labels.groupby(labels)}


def _assert_same_partition(pos):
    nx_labels, _s1 = cluster_ccs(pos, col_a='cluster_id_a', col_b='cluster_id_b')
    uf_labels, _s2 = sequence_ccs(pos, col_a='cluster_id_a', col_b='cluster_id_b')
    assert _partition(nx_labels) == _partition(uf_labels)


def test_cluster_ccs_matches_sequence_ccs_connected():
    _assert_same_partition(_two_blobs())            # bridge joins everything -> 1 CC


def test_cluster_ccs_matches_sequence_ccs_disconnected():
    _assert_same_partition(_pos(_BLOB1 + _BLOB2))   # no bridge -> 2 CCs


def test_cluster_ccs_matches_sequence_ccs_after_a_cut():
    """The flow `assign_atoms_prod` actually runs: re-label the KEPT rows after fragmenting."""
    kept, _dropped, _step = fragment_once(_two_blobs(), cut_method='spectral', seed=1)
    _assert_same_partition(kept)


def test_cluster_ccs_labels_are_order_invariant():
    """cc_id must not depend on row order: it feeds `_lpt_bin_pack`'s `(-size, cc_id)` tie-break,
    so an order-dependent id would silently move equal-sized CCs between splits."""
    pos = _two_blobs()
    base, _s = cluster_ccs(pos)
    shuffled = pos.sample(frac=1.0, random_state=7)
    shuf, _s2 = cluster_ccs(shuffled)
    assert (base.sort_index() == shuf.sort_index()).all()


def test_cluster_ccs_labels_rank_by_size():
    """cc_id=0 is the largest CC (ties break on the lowest node id)."""
    pos = _pos(_BLOB1 + _BLOB2)          # BLOB1 carries 5 rows, BLOB2 carries 4
    cc_id, summary = cluster_ccs(pos)
    assert summary['n_atoms'] == 2
    assert cc_id.value_counts()[0] == 5  # the 5-row component is labelled 0


def test_cluster_ccs_matches_sequence_ccs_ood_nt_cds_t095():
    """Same agreement on the production bigraph (670 nodes / 1,055 edges), natural and post-cut."""
    if not (OOD_CLUSTERS / 't095' / 'combined_cluster.parquet').exists():
        pytest.skip('OOD clusters absent')
    pos = _build_ood_pos_ids('t095')
    _assert_same_partition(pos)
    kept, _dropped, _audit = fragment_until(
        pos, cut_method='spectral', seed=1, stop_fn=stop_at_n_atoms(125), max_drop_frac=0.10)
    _assert_same_partition(kept.reset_index(drop=True))


# --- fragment_largest_cc -----------------------------------------------------
def test_fragment_largest_cc_finds_bridge():
    H, _ = build_pair_bigraph(_two_blobs())
    step = fragment_largest_cc(H, cut_method='spectral', seed=1)
    assert step.pairs_dropped == 1 and len(step.cross_edges) == 1   # the weight-1 bridge
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
    assert step.pairs_dropped == 1


def test_fragment_once_deterministic():
    pos = _two_blobs()
    a = fragment_once(pos, seed=1)[2]
    b = fragment_once(pos, seed=1)[2]
    assert a.part_a == b.part_a and a.cross_edges == b.cross_edges


# --- fragment_until / stop_at_n_atoms / _live_atom_count ---------------------
def test_live_atom_count_excludes_stranded_nodes():
    from src.datasets._megacc_cut import _live_atom_count
    H = nx.Graph()
    H.add_edge('a:1', 'b:1')                        # a real 2-node atom (>= 1 kept edge)
    H.add_node('a:9')                               # a stranded node (0 edges)
    assert nx.number_connected_components(H) == 2   # raw count includes the stranded node
    assert _live_atom_count(H) == 1                 # atom count excludes it (== cluster_ccs)


def test_fragment_until_reaches_target_atoms():
    pos = _two_blobs()                              # one CC of 8 nodes; bridge is the min-cut
    kept, dropped, audit = fragment_until(pos, stop_fn=stop_at_n_atoms(2), seed=1)
    assert audit['stopped_reason'] == 'stop_fn'
    assert audit['n_cuts'] == 1 and audit['n_atoms'] == 2
    assert len(dropped) == 1 and len(kept) == len(pos) - 1
    assert (dropped['cluster_id_a'].iloc[0], dropped['cluster_id_b'].iloc[0]) == ('H2', 'N3')


def test_fragment_until_already_satisfied_does_zero_cuts():
    pos = _two_blobs()                              # already one CC -> a target of 1 needs no cut
    kept, dropped, audit = fragment_until(pos, stop_fn=stop_at_n_atoms(1), seed=1)
    assert audit['stopped_reason'] == 'stop_fn' and audit['n_cuts'] == 0
    assert audit['n_atoms'] == 1 and len(dropped) == 0 and len(kept) == len(pos)


def test_fragment_until_drop_budget_caps():
    pos = _two_blobs()                              # 10 rows; only the weight-1 bridge is a cheap cut
    kept, dropped, audit = fragment_until(
        pos, stop_fn=stop_at_n_atoms(5), max_drop_frac=0.25, seed=1)   # target unreachable in budget
    assert audit['stopped_reason'] == 'max_drop_frac'  # a 2nd cut (into a dense blob) blows the budget
    assert audit['dropped_frac'] <= 0.25
    assert audit['n_atoms'] == 2 and len(dropped) == 1 and len(kept) == len(pos) - 1


def test_fragment_until_deterministic():
    pos = _two_blobs()
    a = fragment_until(pos, stop_fn=stop_at_n_atoms(2), seed=1)[1]
    b = fragment_until(pos, stop_fn=stop_at_n_atoms(2), seed=1)[1]
    assert list(a.index) == list(b.index)




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
    _df, pos_ids, _cooccur, _cc_sizes = _build_positives(cfg, _resolve_spec(args, cfg), args)
    return pos_ids


def test_fragment_once_ood_nt_cds_t095_reproduces_p1():
    """P1 anchor: one spectral cut of the OOD nt_cds t095 mega-CC drops 14 straddling
    pairs and splits 440 clusters into 297 / 143."""
    if not (OOD_CLUSTERS / 't095' / 'combined_cluster.parquet').exists():
        pytest.skip('OOD clusters absent')
    pos = _build_ood_pos_ids('t095')
    kept, dropped, step = fragment_once(pos, cut_method='spectral', seed=1)
    part_b = len(step.cc_nodes) - len(step.part_a)
    assert step.pairs_dropped == 14 and len(step.cross_edges) == 8
    assert len(step.cc_nodes) == 440 and {len(step.part_a), part_b} == {297, 143}
    assert len(dropped) == 14 and len(kept) == 78750




def test_fragment_until_ood_nt_cds_t095_golden():
    """Routing-B operating point on OOD nt_cds t095: fragmenting the mega-CC within a 2% drop
    budget grows the atom count from 108 to exactly 124 at the cheap knee before the edge-cut
    floor; a reachable atom target (115) instead stops via stop_fn.

    The spectral cut is a DIRECT dense eigensolve on a canonical node-order Laplacian
    (`_megacc_cut._bisect`), so it is bit-deterministic across processes -- this asserts the EXACT
    cut (n_cuts / pairs_dropped / n_atoms), not a range."""
    if not (OOD_CLUSTERS / 't095' / 'combined_cluster.parquet').exists():
        pytest.skip('OOD clusters absent')
    g = json.loads(FRAG_GOLDEN.read_text())
    pos = _build_ood_pos_ids('t095')
    _c0, summ0 = cluster_ccs(pos, col_a='cluster_id_a', col_b='cluster_id_b')
    assert summ0['n_atoms'] == g['natural_atoms']           # 108 natural atoms (union-find, exact)

    # budget-bound: an unreachable target -> the 2% drop budget is the stop (exact deterministic cut)
    gb = g['budget_bound']
    kept, dropped, audit = fragment_until(
        pos, cut_method='spectral', seed=1,
        stop_fn=stop_at_n_atoms(gb['target_atoms']), max_drop_frac=gb['max_drop_frac'])
    assert audit['stopped_reason'] == gb['stopped_reason'] == 'max_drop_frac'
    assert audit['n_cuts'] == gb['n_cuts']                      # exact (deterministic dense eigensolve)
    assert audit['pairs_dropped'] == gb['pairs_dropped']
    assert audit['n_atoms'] == gb['n_atoms'] and gb['n_atoms'] > g['natural_atoms']   # grew to 124
    assert audit['dropped_frac'] <= gb['max_drop_frac']         # within budget (guaranteed by the guard)
    assert len(kept) + len(dropped) == len(pos)                 # a partition of pos
    # the atom count fragment_until reports is exactly what the builder (cluster_ccs) sees
    _c1, summ1 = cluster_ccs(kept, col_a='cluster_id_a', col_b='cluster_id_b')
    assert audit['n_atoms'] == summ1['n_atoms']

    # target-bound: a reachable atom target stops via stop_fn (reaching 115 costs << the budget)
    tb = g['target_bound']
    _k2, _d2, audit2 = fragment_until(
        pos, cut_method='spectral', seed=1,
        stop_fn=stop_at_n_atoms(tb['target_atoms']), max_drop_frac=tb['max_drop_frac'])
    assert audit2['stopped_reason'] == tb['stopped_reason'] == 'stop_fn'
    assert audit2['n_atoms'] == tb['n_atoms'] and audit2['n_cuts'] == tb['n_cuts']
    assert audit2['pairs_dropped'] == tb['pairs_dropped']


# --- degenerate inputs -------------------------------------------------------
# Every entry point used to die with `IndexError: list index out of range` on an empty
# graph, from _largest_cc indexing an empty component list. These pin the clean exits.
_EMPTY = pd.DataFrame({'cluster_id_a': [], 'cluster_id_b': [], 'pair_key': []})


def test_largest_cc_on_empty_graph_says_what_is_wrong():
    with pytest.raises(ValueError, match='no nodes'):
        _largest_cc(nx.Graph())


def test_fragment_until_on_empty_input_returns_cleanly():
    kept, dropped, audit = fragment_until(_EMPTY, stop_fn=stop_at_n_atoms(100), max_drop_frac=0.05)
    assert len(kept) == 0 and len(dropped) == 0
    assert audit['n_cuts'] == 0 and audit['n_atoms'] == 0 and audit['pairs_dropped'] == 0




def test_fragment_to_targets_on_empty_graph_returns_cleanly():
    df, H, dropped_edges = fragment_to_targets(nx.Graph())
    assert len(dropped_edges) == 0 and H.number_of_nodes() == 0 and len(df) == 1




if __name__ == '__main__':
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            print(f'{_name} ...', flush=True)
            _fn()
    print('Done. All tests passed.')
