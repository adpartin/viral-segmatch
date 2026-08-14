"""Unit tests for the 2D-CD production fold-making functions (§5 P4.2 of
`docs/plans/2026-08-03_fold_maker_consolidation_plan.md`).

These decide what lands in train/val/test on the primary production path, and had no unit
coverage: `tests/test_production_splits.py` guards the whole build end-to-end, but a digest
mismatch says only that something moved, not which function moved it.

Synthetic frames throughout, so these run in the default suite. They pin properties rather than
values: atoms stay whole, val is measured against the whole set, negatives come only from their
own split, and the folds partition the positives exactly once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._pair_helpers import canonical_pair_key  # noqa: E402
from src.datasets.dataset_pairs_cc import (  # noqa: E402
    _SIDE_SRC,
    _carve_val_atoms,
    compute_negative_infeasible_ccs,
    groupkfold_by_atom,
    make_folds_within_fold,
    within_cc_negatives,
    within_fold_negatives,
)
from src.datasets.dataset_segment_pairs_v2 import _PAIR_COLUMNS  # noqa: E402

FA, FB = 'Hemagglutinin precursor', 'Neuraminidase protein'


def _pos(atom_sizes=(40, 30, 20, 6, 4)):
    """Positives with one atom per entry in `atom_sizes`, and the columns the fold-makers read.

    `prot_hash_a/b` are per-row so each pair has distinct endpoints; `row_id` tracks rows across
    splits. Atom `i` owns hashes prefixed `a{i}_` / `b{i}_`, which is what lets a test assert that
    a split's negatives were drawn only from that split's own positives.
    """
    rows = []
    for atom, size in enumerate(atom_sizes):
        for i in range(size):
            rows.append({'atom_id': atom, 'cc_id': atom,
                         'prot_hash_a': f'a{atom}_{i}', 'prot_hash_b': f'b{atom}_{i}'})
    pos = pd.DataFrame(rows)
    pos['row_id'] = range(len(pos))
    pos['pair_key'] = pos['prot_hash_a'] + '__' + pos['prot_hash_b']
    pos['label'] = 1
    for col in _PAIR_COLUMNS:
        if col not in pos.columns:
            pos[col] = pd.NA
    return pos


def _front_end(pos):
    """Minimal protein frame for negative enrichment: one row per (hash, slot function)."""
    rows = []
    for _, r in pos.iterrows():
        for h, func in ((r['prot_hash_a'], FA), (r['prot_hash_b'], FB)):
            rows.append({'assembly_id': f'iso_{h}', 'function': func, 'prot_hash': h,
                         'brc_fea_id': f'brc_{h}', 'genbank_ctg_id': f'ctg_{h}',
                         'prot_seq': f'SEQ{h}', 'ctg_dna_seq': f'ACGT{h}',
                         'canonical_segment': 'S4', 'ctg_dna_hash': f'd_{h}'})
    return pd.DataFrame(rows).drop_duplicates(['prot_hash', 'function'])[_SIDE_SRC]


def _cooccur(pos):
    return {canonical_pair_key(a, b) for a, b in zip(pos['prot_hash_a'], pos['prot_hash_b'])}


def _atoms(df):
    return set(df['atom_id'])


# --- groupkfold_by_atom: the shared routing core -----------------------------
def test_groupkfold_partitions_rows_exactly_once():
    """Every row is tested exactly once, and each fold's three splits reconstruct the input."""
    pos = _pos()
    folds = groupkfold_by_atom(pos, k_folds=4, val_ratio=0.1, seed=1)
    assert len(folds) == 4

    tested = [r for _tr, _va, te in folds for r in te['row_id']]
    assert sorted(tested) == sorted(pos['row_id']), 'test folds must partition the rows'
    for train, val, test in folds:
        got = sorted([*train['row_id'], *val['row_id'], *test['row_id']])
        assert got == sorted(pos['row_id']), 'a fold lost or duplicated rows'


def test_groupkfold_keeps_atoms_whole_in_all_three_splits():
    """The cluster-disjointness guarantee: the test fold shares no atom with train or val.

    Train and val DO share atoms -- `_carve_val_pairs` carves val at row level, so val is
    deliberately in-distribution. That is asserted positively, so a return to whole-atom carving
    fails here instead of passing quietly.
    """
    for train, val, test in groupkfold_by_atom(_pos(), k_folds=4, val_ratio=0.1, seed=1):
        assert not _atoms(train) & _atoms(test)
        assert not _atoms(val) & _atoms(test)
        assert _atoms(train) & _atoms(val)


def test_groupkfold_is_deterministic_for_a_seed():
    pos = _pos()
    a = groupkfold_by_atom(pos, 4, 0.1, seed=7)
    b = groupkfold_by_atom(pos, 4, 0.1, seed=7)
    for (t1, v1, e1), (t2, v2, e2) in zip(a, b):
        assert list(t1['row_id']) == list(t2['row_id'])
        assert list(v1['row_id']) == list(v2['row_id'])
        assert list(e1['row_id']) == list(e2['row_id'])


# --- _carve_val_atoms --------------------------------------------------------
def test_carve_val_targets_the_whole_set_not_the_non_test_pool():
    """val_ratio is a fraction of `n_total`, not of `tv`. At k=4 the non-test pool is ~75% of the
    data, so a carve measured against `tv` would come out ~25% short.

    Uses single-row atoms deliberately: the carve accumulates whole atoms until it crosses the
    target, so with coarse atoms both sizings can land on the same boundary and the assertion
    would hold either way.
    """
    pos = _pos(atom_sizes=(1,) * 100)
    tv = pos[pos['atom_id'] >= 25]                      # stand-in for one k=4 fold's non-test rows
    _train, val = _carve_val_atoms(tv, val_ratio=0.1, n_total=len(pos), seed=1)

    # Target is 10% of all 100 rows. Sizing against tv's 75 rows would give 8.
    assert len(val) == 10


def test_carve_val_takes_whole_atoms():
    tv = _pos()
    train, val = _carve_val_atoms(tv, val_ratio=0.2, n_total=len(tv), seed=3)
    assert not _atoms(train) & _atoms(val)
    assert sorted([*train['row_id'], *val['row_id']]) == sorted(tv['row_id'])


# --- within_fold_negatives: the production negative sampler ------------------
def test_within_fold_negatives_draw_only_from_their_own_split():
    """The invariant that keeps folds cluster-disjoint: both endpoints of every negative come
    from THIS split's positives, so a negative can never reference a held-out sequence."""
    pos = _pos()
    split = pos[pos['atom_id'].isin([0, 1])]
    neg = within_fold_negatives(split, _cooccur(pos), _front_end(pos), (FA, FB),
                                neg_to_pos_ratio=1.0, seed=1)
    assert len(neg) > 0
    assert set(neg['prot_hash_a']) <= set(split['prot_hash_a'])
    assert set(neg['prot_hash_b']) <= set(split['prot_hash_b'])


def test_within_fold_negatives_never_reproduce_a_positive():
    pos = _pos()
    cooccur = _cooccur(pos)
    neg = within_fold_negatives(pos, cooccur, _front_end(pos), (FA, FB),
                                neg_to_pos_ratio=1.0, seed=1)
    drawn = {canonical_pair_key(a, b) for a, b in zip(neg['prot_hash_a'], neg['prot_hash_b'])}
    assert not (drawn & cooccur), 'a sampled negative reconstructed an observed pair'
    assert len(drawn) == len(neg), 'duplicate negatives were emitted'


def test_within_fold_negatives_respect_the_ratio_and_label_zero():
    pos = _pos()
    neg = within_fold_negatives(pos, _cooccur(pos), _front_end(pos), (FA, FB),
                                neg_to_pos_ratio=0.5, seed=1)
    assert len(neg) <= round(0.5 * len(pos))            # reject sampling may under-fill
    assert (neg['label'] == 0).all()
    assert list(neg.columns) == list(_PAIR_COLUMNS)


def test_within_fold_negatives_shared_seen_stops_a_pair_being_drawn_twice():
    """One `seen` set across two calls keeps a pair out of both; separate sets do not.

    Both halves draw from one atom, so they share sequences and can reach the same pairs -- the
    situation train and val are in once val is carved at row level. The second assertion pins that
    the fixture reaches it, so the first cannot pass for want of any overlap to prevent.
    """
    pos = _pos_recurring_hashes(atom_sizes=(12,))
    cooccur, df = _cooccur(pos), _front_end(pos)
    first, second = pos.iloc[:6], pos.iloc[6:]
    kwargs = dict(neg_to_pos_ratio=2.0, hash_col='prot_hash')

    shared = set()
    a = within_fold_negatives(first, cooccur, df, (FA, FB), seed=1, seen=shared, **kwargs)
    b = within_fold_negatives(second, cooccur, df, (FA, FB), seed=2, seen=shared, **kwargs)
    assert not set(a['pair_key']) & set(b['pair_key'])

    solo_a = within_fold_negatives(first, cooccur, df, (FA, FB), seed=1, **kwargs)
    solo_b = within_fold_negatives(second, cooccur, df, (FA, FB), seed=2, **kwargs)
    assert set(solo_a['pair_key']) & set(solo_b['pair_key']), 'fixture must produce collisions'


# --- make_folds_within_fold: the production fold-maker ----------------------
def _within_fold_folds(pos, k=4):
    return make_folds_within_fold(pos, k, 0.1, seed=1, neg_to_pos_ratio=1.0,
                                  cooccur=_cooccur(pos), df=_front_end(pos),
                                  schema_pair_full=(FA, FB))


def test_make_folds_within_fold_returns_k_folds_of_positives_plus_negatives():
    pos = _pos()
    folds = _within_fold_folds(pos)
    assert len(folds) == 4
    for split in (s for fold in folds for s in fold):
        assert set(split['label']) <= {0, 1}
        assert (split['label'] == 1).sum() > 0, 'a split lost its positives'


def _pos_recurring_hashes(atom_sizes=(40, 30, 20, 6, 4)):
    """Positives whose sequences recur across rows, as a real corpus has them.

    `_pos` gives every row unique endpoints, so two splits can never draw the same negative and
    cross-split dedup has nothing to catch. Here each atom draws its rows from a small pool of
    sequences per slot -- pair_keys stay unique, but the hashes repeat, so the splits' negative
    draw pools overlap.

    The two pool sizes are consecutive, hence coprime, so `(i % n_a, i % n_b)` is unique over the
    atom, and their product is about twice `size` -- leaving roughly as many non-co-occurring
    combinations as positives, so negatives can actually be drawn.
    """
    rows = []
    for atom, size in enumerate(atom_sizes):
        n_a = int((2 * size) ** 0.5) + 1
        n_b = n_a + 1
        for i in range(size):
            rows.append({'atom_id': atom, 'cc_id': atom,
                         'prot_hash_a': f'a{atom}_{i % n_a}',
                         'prot_hash_b': f'b{atom}_{i % n_b}'})
    pos = pd.DataFrame(rows)
    pos['row_id'] = range(len(pos))
    pos['pair_key'] = pos['prot_hash_a'] + '__' + pos['prot_hash_b']
    pos['label'] = 1
    assert pos['pair_key'].is_unique, 'fixture must give one row per pair_key'
    for col in _PAIR_COLUMNS:
        if col not in pos.columns:
            pos[col] = pd.NA
    return pos


def test_make_folds_within_fold_draws_each_negative_at_most_once_per_fold():
    """No pair may appear in two splits of a fold.

    Since val is carved at row level, train and val share atoms and therefore share sequences, so
    both draw negatives from overlapping pools; only one `seen` set spanning the fold keeps a pair
    from being drawn twice. Test cannot collide with either -- GroupKFold holds its atoms out of
    both -- but it is checked so a regression there is not silent.
    """
    folds = _within_fold_folds(_pos_recurring_hashes())
    for train, val, test in folds:
        keys = {name: set(s['pair_key']) for name, s in
                (('train', train), ('val', val), ('test', test))}
        assert not keys['train'] & keys['val'], 'a pair landed in both train and val'
        assert not keys['train'] & keys['test'], 'a pair landed in both train and test'
        assert not keys['val'] & keys['test'], 'a pair landed in both val and test'


def test_make_folds_within_fold_partitions_positives_exactly_once():
    """Positives are routed, not resampled: each appears in the test split of exactly one fold."""
    pos = _pos()
    tested = [k for _tr, _va, te in _within_fold_folds(pos)
              for k in te.loc[te['label'] == 1, 'pair_key']]
    assert sorted(tested) == sorted(pos['pair_key'])


def test_make_folds_within_fold_negatives_stay_inside_their_split():
    """Cross-split negative leakage check: a negative's endpoints must belong to the split that
    holds it, which is what makes the whole fold cluster-disjoint."""
    pos = _pos()
    for fold in _within_fold_folds(pos):
        for split in fold:
            pos_rows = split[split['label'] == 1]
            neg_rows = split[split['label'] == 0]
            assert set(neg_rows['prot_hash_a']) <= set(pos_rows['prot_hash_a'])
            assert set(neg_rows['prot_hash_b']) <= set(pos_rows['prot_hash_b'])


# --- compute_negative_infeasible_ccs ----------------------------------------
def test_singleton_cc_is_negative_infeasible():
    """A CC holding one pair offers one a x one b, which is the positive itself."""
    pos = _pos(atom_sizes=(1,))
    assert compute_negative_infeasible_ccs(pos, _cooccur(pos)) == {0}


def test_cc_with_a_drawable_recombination_is_feasible():
    """Two pairs in one CC give four cross pairings, two of which are not co-occurrences."""
    pos = _pos(atom_sizes=(2,))
    assert compute_negative_infeasible_ccs(pos, _cooccur(pos)) == set()


def test_dense_cc_where_every_recombination_co_occurs_is_infeasible():
    """Feasibility is structural, not size-based: a CC can hold many pairs and still offer no
    drawable negative if every cross pairing is observed."""
    pos = _pos(atom_sizes=(1,))
    pos = pd.concat([pos, pos.assign(prot_hash_b='b0_1', pair_key='a0_0__b0_1', row_id=99)],
                    ignore_index=True)
    cooccur = _cooccur(pos)                             # both cross pairings observed
    assert compute_negative_infeasible_ccs(pos, cooccur) == {0}


# --- within_cc_negatives -----------------------------------------------------
def test_within_cc_negatives_stay_inside_their_cc():
    """The stricter scope: both endpoints come from one CC's isolate pool, so the negative
    carries that CC's atom_id and travels with it through GroupKFold."""
    pos = _pos(atom_sizes=(6, 6))
    # `cell` is the per-isolate metadata tuple the regime classifier compares, one entry per
    # DEFAULT_AXES (host, hn_subtype, year) -- a scalar here raises IndexError inside it.
    iso = pd.DataFrame([
        {'assembly_id': f'iso_{r.atom_id}_{i}', 'hash_a': f'a{r.atom_id}_{i}',
         'hash_b': f'b{r.atom_id}_{i}', 'cc_id': r.atom_id, 'atom_id': r.atom_id,
         'cell': ('duck', 'H5N1', '2010')}
        for i, r in enumerate(pos.itertuples())
    ])
    neg, cc_log = within_cc_negatives(pos, iso, _cooccur(pos), _front_end(pos), (FA, FB),
                                      neg_to_pos_ratio=1.0, seed=1)
    assert len(neg) > 0
    for _, row in neg.iterrows():
        assert row['prot_hash_a'].startswith(f"a{row['atom_id']}_")
        assert row['prot_hash_b'].startswith(f"b{row['atom_id']}_")
    assert set(cc_log['cc_id']) == set(pos['cc_id'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
