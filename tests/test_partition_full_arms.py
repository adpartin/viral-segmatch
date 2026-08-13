"""Pin the leave-cc-out vs random arm contract in `dataset_pairs_cc._partition_full`.

The subtree under `_partition_full` (`pick_largest_atoms`, `make_folds_leave_cc_out`,
`make_folds_random`) runs only under `negative_scope: within_cc`, which no shipped bundle sets --
it is reached by overriding a 2D-CD bundle, as `ARM_OVERRIDES` does here. `_carve_val_pairs` is
shared with the production router and lives outside this block.
These tests are therefore the only coverage that path has. The property that matters most is the
paired experiment's premise: **both arms partition the SAME rows**, so any difference in results
is attributable to the split and nothing else.

Drives `_partition_full` with a synthetic `full` frame -- an end-to-end build would pull in the
within-CC negative pool and mostly exercise the negative sampler rather than the arms. Design and
the thresholds it was run at: docs/plans/2026-07-21_ood_vs_random_split_plan.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets.dataset_pairs_cc import _partition_full, _resolve_spec  # noqa: E402
from src.utils.config_hydra import get_virus_config_hydra  # noqa: E402

BUNDLE = 'flu_ha_na_cc_nt_cds_cm0'

# The paired arms are selected by config, not by a dedicated bundle: any 2D-CD bundle plus these
# four settings reaches _partition_full's leave_cc_out branch. `_resolve_spec` rejects
# leave_cc_out without within_cc, so the two must be set together.
ARM_OVERRIDES = [
    'dataset.n_folds=3',
    'dataset.split_strategy.negative_scope=within_cc',
    'dataset.split_strategy.fold_assignment=leave_cc_out',
    'dataset.split_strategy.paired_random=true',
]


def _spec(**overrides):
    """CCSpec for the paired arms: a live 2D-CD bundle plus `ARM_OVERRIDES`.

    Args:
        **overrides: extra `dataset.split_strategy.*` values, applied after ARM_OVERRIDES.

    Returns:
        The resolved CCSpec.
    """
    cfg = get_virus_config_hydra(BUNDLE, config_path=str(PROJ / 'conf'))
    dotlist = ARM_OVERRIDES + [f'dataset.split_strategy.{k}={v}' for k, v in overrides.items()]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    args = SimpleNamespace(config_bundle=BUNDLE, protein_final=None, override=None, out_dir=None)
    return _resolve_spec(args, cfg)


def _synthetic_full(atom_sizes=(40, 30, 20, 6, 4)):
    """A `full` frame shaped like the post-negative pool: one atom per entry in `atom_sizes`,
    half positive / half negative, with a unique `row_id` so rows can be tracked across arms."""
    rows = []
    for atom, size in enumerate(atom_sizes):
        for i in range(size):
            rows.append({'atom_id': atom, 'label': 1 if i % 2 == 0 else 0})
    full = pd.DataFrame(rows)
    full['row_id'] = range(len(full))
    return full


def test_overrides_select_the_arms():
    """`_resolve_spec` reads the knobs that switch `_partition_full` to the paired arms."""
    spec = _spec()
    assert spec.fold_assignment == 'leave_cc_out'
    assert spec.paired_random is True
    assert spec.negative_scope == 'within_cc'


def test_both_arms_partition_the_same_rows():
    """The experiment's premise: the two arms differ only in how identical rows are split."""
    arms = _partition_full(_synthetic_full(), _spec())
    assert set(arms) == {'ood', 'random'}

    def rows_seen(folds):
        # Every row each arm ever places, in any split of any fold.
        return {rid for train, val, test in folds for df in (train, val, test) for rid in df['row_id']}

    assert rows_seen(arms['ood']) == rows_seen(arms['random'])


def test_per_fold_test_sizes_match_between_arms():
    """Size-matching is what makes the arms comparable fold by fold."""
    arms = _partition_full(_synthetic_full(), _spec())
    ood_sizes = [len(test) for _, _, test in arms['ood']]
    random_sizes = [len(test) for _, _, test in arms['random']]
    assert ood_sizes == random_sizes


def test_ood_arm_tests_one_whole_atom_per_fold():
    """Leave-one-atom-out: k folds, each testing exactly one atom, no atom tested twice."""
    spec = _spec()
    arms = _partition_full(_synthetic_full(), spec)
    folds = arms['ood']
    assert len(folds) == spec.k_folds

    tested = []
    for train, val, test in folds:
        atoms = set(test['atom_id'])
        assert len(atoms) == 1, f"test split holds {len(atoms)} atoms, expected 1"
        atom = atoms.pop()
        assert atom not in set(train['atom_id']), "test atom leaked into train"
        assert atom not in set(val['atom_id']), "test atom leaked into val"
        tested.append(atom)
    assert len(set(tested)) == len(tested), "an atom was tested in more than one fold"


def test_random_arm_tests_each_row_once():
    """The control splits rows, not atoms: the test folds partition the main-atom rows."""
    arms = _partition_full(_synthetic_full(), _spec())
    tested = [rid for _, _, test in arms['random'] for rid in test['row_id']]
    assert len(tested) == len(set(tested)), "a row was tested in more than one fold"

    # Straddling atoms is the point of the control -- assert it happens, so a future change
    # that silently made this arm atom-aware would fail here instead of passing quietly.
    straddles = any(
        set(test['atom_id']) & set(train['atom_id'])
        for train, _, test in arms['random']
    )
    assert straddles, "random arm kept atoms whole; it is supposed to be the in-distribution control"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
