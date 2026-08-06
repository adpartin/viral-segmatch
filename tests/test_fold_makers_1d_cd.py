"""Unit tests for the 1D-CD production routing path (§5 P4.2 of
`docs/plans/2026-08-03_fold_maker_consolidation_plan.md`).

1D-CD had no unit tests at all: nothing exercised `cluster_disjoint_route_pos_df`, the LPT packer
it carves val with, or the feasibility check that gates its folds. `tests/test_production_splits.py`
guards the whole build end-to-end, but a digest mismatch does not say which function moved.

Synthetic frames throughout. The guarantee under test is the 1D one: the **constrained slot's**
clusters never span two splits, while the unconstrained slot is free to recur — that asymmetry is
what makes this 1D-CD rather than 2D-CD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._pair_helpers import _lpt_bin_pack  # noqa: E402
from src.datasets._split_helpers import (  # noqa: E402
    _compute_d3_check,
    attach_cluster_ids,
    cluster_disjoint_route_pos_df,
)

SPLITS = ('train', 'val', 'test')


def _pos(n_a_clusters=12, pairs_per_a=5):
    """Positives whose slot-A hashes fall into `n_a_clusters` groups.

    Slot-B hashes are deliberately drawn from a small shared pool, so the unconstrained slot
    recurs across splits -- the thing 1D-CD permits and 2D-CD does not.
    """
    rows = []
    for a in range(n_a_clusters):
        for i in range(pairs_per_a):
            rows.append({'prot_hash_a': f'ha_{a}_{i}', 'prot_hash_b': f'hb_{i % 3}'})
    pos = pd.DataFrame(rows)
    pos['pair_key'] = pos['prot_hash_a'] + '__' + pos['prot_hash_b']
    return pos


def _lookup(pos, n_a_clusters=12):
    """`(prot_hash, cluster_id)` for both slots: slot-A hashes grouped into per-A clusters,
    slot-B hashes each in their own cluster."""
    rows = [{'prot_hash': f'ha_{a}_{i}', 'cluster_id': f'A{a}'}
            for a in range(n_a_clusters) for i in range(20)]
    rows += [{'prot_hash': f'hb_{j}', 'cluster_id': f'B{j}'} for j in range(3)]
    return pd.DataFrame(rows).drop_duplicates('prot_hash')


def _route(pos, lookup, seed=1, **kw):
    kw.setdefault('single_slot', 'a')
    kw.setdefault('n_folds', 4)
    return cluster_disjoint_route_pos_df(pos, lookup, train_ratio=0.8, val_ratio=0.1,
                                         seed=seed, **kw)


# --- _lpt_bin_pack -----------------------------------------------------------
def test_lpt_assigns_each_group_to_exactly_one_bin():
    sizes = pd.Series({'g1': 50, 'g2': 30, 'g3': 20, 'g4': 10})
    packed = _lpt_bin_pack(sizes, {'train': 88.0, 'val': 11.0}, ['train', 'val'])
    assert set(packed) == set(sizes.index)
    assert set(packed.values()) <= {'train', 'val'}


def test_lpt_sends_the_largest_group_to_the_biggest_deficit():
    """Largest-first placement: with an empty board, the biggest target wins the biggest group."""
    sizes = pd.Series({'big': 100, 'small': 1})
    packed = _lpt_bin_pack(sizes, {'train': 80.0, 'test': 20.0}, ['train', 'test'])
    assert packed['big'] == 'train'


def test_lpt_breaks_size_ties_on_group_id_not_insertion_order():
    """Equal-sized groups must order by id, or fold membership would depend on row order --
    the failure the `(-size, cluster_id)` key exists to prevent."""
    sizes = pd.Series({'g2': 10, 'g1': 10, 'g3': 10})
    forward = _lpt_bin_pack(sizes, {'a': 20.0, 'b': 10.0}, ['a', 'b'])
    reversed_ = _lpt_bin_pack(sizes.iloc[::-1], {'a': 20.0, 'b': 10.0}, ['a', 'b'])
    assert forward == reversed_


def test_lpt_is_deterministic():
    sizes = pd.Series({f'g{i}': (i * 7) % 13 + 1 for i in range(20)})
    targets = {'train': 90.0, 'val': 12.0, 'test': 12.0}
    assert _lpt_bin_pack(sizes, targets, list(targets)) == _lpt_bin_pack(sizes, targets, list(targets))


# --- _compute_d3_check -------------------------------------------------------
def test_d3_passes_when_drift_and_test_size_are_within_bounds():
    check = _compute_d3_check({'train': 0.80, 'val': 0.10, 'test': 0.10},
                              {'train': 0.80, 'val': 0.10, 'test': 0.10},
                              max_acceptable_drift_pp=0.05, min_test_frac=0.05)
    assert check['all_pass'] is True
    assert check['max_acceptable_drift_pp']['achieved'] == 0.0


def test_d3_reports_the_worst_bin_not_the_average():
    """One bin far off must fail the check even when the others are exact."""
    check = _compute_d3_check({'train': 0.90, 'val': 0.05, 'test': 0.05},
                              {'train': 0.80, 'val': 0.10, 'test': 0.10},
                              max_acceptable_drift_pp=0.05, min_test_frac=0.01)
    assert check['max_acceptable_drift_pp']['achieved'] == pytest.approx(0.10)
    assert check['max_acceptable_drift_pp']['pass'] is False
    assert check['all_pass'] is False


def test_d3_fails_a_starved_test_split_even_when_drift_is_fine():
    """The two knobs are independent: a tiny test split fails on size, not drift."""
    check = _compute_d3_check({'train': 0.80, 'val': 0.18, 'test': 0.02},
                              {'train': 0.80, 'val': 0.18, 'test': 0.02},
                              max_acceptable_drift_pp=0.05, min_test_frac=0.05)
    assert check['max_acceptable_drift_pp']['pass'] is True
    assert check['min_test_frac']['pass'] is False
    assert check['all_pass'] is False


# --- attach_cluster_ids ------------------------------------------------------
def test_attach_cluster_ids_drops_pairs_whose_hash_is_unclustered():
    pos = _pos(n_a_clusters=3, pairs_per_a=2)
    lookup = _lookup(pos, n_a_clusters=3)
    lookup = lookup[lookup['prot_hash'] != 'ha_0_0']          # orphan one slot-A hash
    with_ids, audit = attach_cluster_ids(pos, lookup)
    assert audit['n_input'] == len(pos)
    assert audit['n_kept'] == len(with_ids) == len(pos) - 1
    assert with_ids['cluster_id_a'].notna().all()


# --- cluster_disjoint_route_pos_df, k-fold branch ---------------------------
def test_1d_cd_constrained_slot_never_spans_splits():
    """THE 1D-CD guarantee: no slot-A cluster appears in two splits of a fold."""
    pos = _pos()
    for train, val, test, _audit in _route(pos, _lookup(pos)):
        by_split = {s: set(df['cluster_id_a']) for s, df in
                    zip(SPLITS, (train, val, test))}
        assert not by_split['train'] & by_split['test']
        assert not by_split['train'] & by_split['val']
        assert not by_split['val'] & by_split['test']


def test_1d_cd_leaves_the_unconstrained_slot_free_to_recur():
    """The asymmetry that separates 1D-CD from 2D-CD: slot B may repeat across splits. If this
    ever failed, the split would silently have become 2D-CD."""
    pos = _pos()
    folds = _route(pos, _lookup(pos))
    recurs = any(set(train['cluster_id_b']) & set(test['cluster_id_b'])
                 for train, _val, test, _a in folds)
    assert recurs, 'slot B was held out too -- this is no longer a single-slot split'


def test_1d_cd_test_folds_partition_the_pairs():
    pos = _pos()
    folds = _route(pos, _lookup(pos))
    assert len(folds) == 4
    tested = [k for _tr, _va, te, _a in folds for k in te['pair_key']]
    assert sorted(tested) == sorted(pos['pair_key']), 'test folds must partition the pairs'


def test_1d_cd_each_fold_reconstructs_the_input():
    pos = _pos()
    for train, val, test, _audit in _route(pos, _lookup(pos)):
        got = sorted([*train['pair_key'], *val['pair_key'], *test['pair_key']])
        assert got == sorted(pos['pair_key']), 'a fold lost or duplicated pairs'


def test_1d_cd_is_seed_independent():
    """The 1D-CD router records `seed` but never consumes it: unshuffled GroupKFold and LPT are
    both deterministic. Different seeds must give identical folds."""
    pos = _pos()
    lookup = _lookup(pos)
    a = _route(pos, lookup, seed=1)
    b = _route(pos, lookup, seed=999)
    for (t1, v1, e1, _), (t2, v2, e2, _) in zip(a, b):
        assert list(t1['pair_key']) == list(t2['pair_key'])
        assert list(v1['pair_key']) == list(v2['pair_key'])
        assert list(e1['pair_key']) == list(e2['pair_key'])


def test_1d_cd_audit_carries_the_feasibility_check_per_fold():
    pos = _pos()
    for _tr, _va, _te, audit in _route(pos, _lookup(pos)):
        assert 'd3_check' in audit or 'feasibility' in audit or 'all_pass' in str(audit), \
            f'fold audit lacks a feasibility record: {sorted(audit)}'


def test_1d_cd_refuses_more_folds_than_atoms():
    """GroupKFold cannot make more folds than there are clusters on the constrained slot."""
    pos = _pos(n_a_clusters=3, pairs_per_a=4)
    with pytest.raises(NotImplementedError):
        _route(pos, _lookup(pos, n_a_clusters=3), n_folds=10)


def test_1d_cd_bilateral_kfold_is_not_implemented():
    """k-fold with `single_slot=None` is the case `dataset_pairs_cc.py` exists to serve."""
    pos = _pos()
    with pytest.raises(NotImplementedError):
        _route(pos, _lookup(pos), single_slot=None, n_folds=4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
