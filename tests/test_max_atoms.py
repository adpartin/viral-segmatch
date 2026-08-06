"""Unit tests for the max_atoms atom-count cap in the 2D-CD CC builder
(`dataset_pairs_cc._subsample_atoms`).

Run: pytest tests/test_max_atoms.py   (or: python tests/test_max_atoms.py)
"""
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets.dataset_pairs_cc import _subsample_atoms


def _pos(atom_ids):
    """Synthetic pos_ids: one column of atom_id + a payload column (rows per atom vary)."""
    return pd.DataFrame({'atom_id': atom_ids, 'row': range(len(atom_ids))})


def test_caps_atom_count():
    pos = _pos([0, 0, 1, 2, 3, 4, 4, 4])  # 5 atoms
    out = _subsample_atoms(pos, max_atoms=2, seed=0)
    assert out['atom_id'].nunique() == 2


def test_keeps_all_rows_of_kept_atoms():
    pos = _pos([0, 0, 1, 2, 3, 4, 4, 4])
    out = _subsample_atoms(pos, max_atoms=2, seed=0)
    # every kept atom keeps ALL its original rows (cap is on atoms, not rows)
    for a in out['atom_id'].unique():
        assert (out['atom_id'] == a).sum() == (pos['atom_id'] == a).sum()


def test_noop_when_none():
    pos = _pos([0, 1, 2])
    out = _subsample_atoms(pos, max_atoms=None, seed=0)
    assert out['atom_id'].nunique() == 3 and len(out) == len(pos)


def test_noop_when_within_budget():
    pos = _pos([0, 1, 2])
    out = _subsample_atoms(pos, max_atoms=5, seed=0)
    assert out['atom_id'].nunique() == 3


def test_noop_at_exact_budget():
    pos = _pos([0, 1, 2, 3, 4])  # exactly 5 atoms, cap 5 -> no-op
    out = _subsample_atoms(pos, max_atoms=5, seed=0)
    assert out['atom_id'].nunique() == 5


def test_deterministic_same_seed():
    pos = _pos(list(range(20)))
    a = _subsample_atoms(pos, max_atoms=5, seed=42)
    b = _subsample_atoms(pos, max_atoms=5, seed=42)
    assert set(a['atom_id']) == set(b['atom_id'])
    assert a['atom_id'].nunique() == 5


if __name__ == '__main__':
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn()
    print('Done. All tests passed.')
