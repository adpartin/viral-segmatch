"""Tests for unique-sequence positive-pair selection and random CV."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._positive_pair_selection import select_positive_pairs
from src.datasets.dataset_segment_pairs_v2 import (
    _validate_v2_config,
    generate_all_cv_folds_v2,
)


def _selection_frame() -> pd.DataFrame:
    """Return a graph where A-first greedy is smaller than the maximum matching."""
    return pd.DataFrame(
        {
            "pair_key": ["p0", "p1", "p2"],
            "assembly_id_a": ["i0", "i1", "i2"],
            "prot_hash_a": ["a1", "a2", "a2"],
            "prot_hash_b": ["b1", "b1", "b2"],
        }
    )


def test_all_preserves_input_exactly():
    frame = _selection_frame()
    selected, audit = select_positive_pairs(
        frame,
        "all",
        "prot_hash_a",
        "prot_hash_b",
    )

    pd.testing.assert_frame_equal(selected, frame)
    assert audit["selected_pairs"] == 3
    assert audit["dropped_pairs"] == 0


def test_sequential_dedup_directions_differ_and_make_both_slots_unique():
    frame = _selection_frame()
    a_first, _ = select_positive_pairs(
        frame,
        "dedup_a_then_b",
        "prot_hash_a",
        "prot_hash_b",
    )
    b_first, _ = select_positive_pairs(
        frame,
        "dedup_b_then_a",
        "prot_hash_a",
        "prot_hash_b",
    )

    assert len(a_first) == 1
    assert len(b_first) == 2
    for selected in (a_first, b_first):
        assert selected["prot_hash_a"].is_unique
        assert selected["prot_hash_b"].is_unique


def test_hopcroft_karp_returns_known_maximum_and_correct_audit():
    selected, audit = select_positive_pairs(
        _selection_frame(),
        "hopcroft_karp",
        "prot_hash_a",
        "prot_hash_b",
    )

    assert selected["pair_key"].tolist() == ["p0", "p2"]
    assert audit["selected_pairs"] == 2
    assert audit["dropped_pairs"] == 1
    assert audit["selected_unique_a"] == 2
    assert audit["selected_unique_b"] == 2
    assert audit["unmatched_a"] == 0
    assert audit["unmatched_b"] == 0
    expected_checksum = hashlib.sha256(b"p0\np2").hexdigest()
    assert audit["selected_pair_keys_sha256"] == expected_checksum
    assert audit["networkx_version"]


def test_hopcroft_karp_handles_disconnected_graph():
    frame = pd.concat(
        [
            _selection_frame(),
            pd.DataFrame(
                {
                    "pair_key": ["p3"],
                    "assembly_id_a": ["i3"],
                    "prot_hash_a": ["a3"],
                    "prot_hash_b": ["b3"],
                }
            ),
        ],
        ignore_index=True,
    )
    selected, _ = select_positive_pairs(
        frame,
        "hopcroft_karp",
        "prot_hash_a",
        "prot_hash_b",
    )

    assert len(selected) == 3
    assert "p3" in set(selected["pair_key"])


@pytest.mark.parametrize(
    "method",
    ["dedup_a_then_b", "dedup_b_then_a", "hopcroft_karp"],
)
def test_active_selection_is_independent_of_input_row_order(method):
    frame = _selection_frame()
    expected, _ = select_positive_pairs(
        frame,
        method,
        "prot_hash_a",
        "prot_hash_b",
    )
    shuffled, _ = select_positive_pairs(
        frame.sample(frac=1, random_state=9),
        method,
        "prot_hash_a",
        "prot_hash_b",
    )

    assert shuffled["pair_key"].tolist() == expected["pair_key"].tolist()


def test_hopcroft_karp_is_stable_across_python_hash_seeds():
    code = """
import json
import pandas as pd
from src.datasets._positive_pair_selection import select_positive_pairs
frame = pd.DataFrame({
    'pair_key': ['p0', 'p1', 'p2'],
    'assembly_id_a': ['i0', 'i1', 'i2'],
    'prot_hash_a': ['a1', 'a2', 'a2'],
    'prot_hash_b': ['b1', 'b1', 'b2'],
})
selected, audit = select_positive_pairs(
    frame, 'hopcroft_karp', 'prot_hash_a', 'prot_hash_b')
print(json.dumps({
    'pair_keys': selected['pair_key'].tolist(),
    'checksum': audit['selected_pair_keys_sha256'],
}))
"""
    outputs = []
    for seed in ("1", "917"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJ,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads(result.stdout))

    assert outputs[0] == outputs[1]


def test_invalid_method_ordering_and_columns_raise():
    frame = _selection_frame()
    with pytest.raises(ValueError, match="method"):
        select_positive_pairs(
            frame,
            "unknown",
            "prot_hash_a",
            "prot_hash_b",
        )
    with pytest.raises(ValueError, match="ordering"):
        select_positive_pairs(
            frame,
            "hopcroft_karp",
            "prot_hash_a",
            "prot_hash_b",
            ordering="unknown",
        )
    with pytest.raises(ValueError, match="required columns"):
        select_positive_pairs(
            frame.drop(columns="prot_hash_b"),
            "hopcroft_karp",
            "prot_hash_a",
            "prot_hash_b",
        )


def test_active_selection_config_requires_random_cv():
    base = {
        "dataset": {
            "schema_pair": ["A", "B"],
            "pair_mode": "schema_ordered",
            "hard_partition_isolates": True,
            "n_folds": 3,
            "split_strategy": {"mode": "random"},
            "positive_pair_selection": {
                "method": "hopcroft_karp",
                "ordering": "pair_key",
            },
        }
    }
    _validate_v2_config(OmegaConf.create(base))

    single_split = OmegaConf.create(base)
    single_split.dataset.n_folds = None
    with pytest.raises(NotImplementedError, match="n_folds"):
        _validate_v2_config(single_split)

    wrong_mode = OmegaConf.create(base)
    wrong_mode.dataset.split_strategy.mode = "seq_disjoint"
    with pytest.raises(NotImplementedError, match="mode='random'"):
        _validate_v2_config(wrong_mode)


def _protein_frame(n_diagonal: int = 15) -> pd.DataFrame:
    """Build positive edges with one extra observed edge that matching must drop."""
    edges = [(f"a{i}", f"b{i}") for i in range(n_diagonal)]
    edges.append(("a0", "b1"))
    rows = []
    for edge_i, (hash_a, hash_b) in enumerate(edges):
        assembly_id = f"iso{edge_i:02d}"
        for side, function, seq_hash in (
            ("a", "A", hash_a),
            ("b", "B", hash_b),
        ):
            rows.append(
                {
                    "assembly_id": assembly_id,
                    "brc_fea_id": f"{assembly_id}_{side}",
                    "genbank_ctg_id": f"{assembly_id}_{side}_ctg",
                    "prot_seq": f"M{seq_hash}",
                    "ctg_dna_seq": f"ATG{seq_hash}",
                    "canonical_segment": function,
                    "function": function,
                    "prot_hash": seq_hash,
                    "ctg_dna_hash": f"ctg_{seq_hash}",
                    "cds_dna_hash": f"cds_{seq_hash}",
                }
            )
    return pd.DataFrame(rows)


def _selected_cv_folds(negative_scope: str = "within_fold") -> list[dict]:
    return list(
        generate_all_cv_folds_v2(
            df=_protein_frame(),
            n_folds=3,
            seed=42,
            neg_to_pos_ratio=1.0,
            val_ratio=0.2,
            schema_pair=("A", "B"),
            axes_for_flags=[],
            pair_key_alphabet="nt_cds",
            negative_scope=negative_scope,
            positive_selection_method="hopcroft_karp",
        )
    )


def test_selected_cv_enforces_split_invariants_and_test_coverage():
    folds = _selected_cv_folds()
    assert len(folds) == 3
    assert folds[0]["positive_selection_audit"]["input_pairs"] == 16
    assert folds[0]["positive_selection_audit"]["selected_pairs"] == 15

    test_positive_keys = []
    for fold in folds:
        splits = {
            "train": fold["train_pairs"],
            "val": fold["val_pairs"],
            "test": fold["test_pairs"],
        }
        audit = fold["duplicate_stats"]["positive_pair_selection_fold_audit"]
        assert all(
            count == 0
            for side in audit["full_pair_hash_overlap"].values()
            for count in side.values()
        )
        assert all(
            count == 0
            for split_counts in audit["negative_endpoints_outside_split"].values()
            for count in split_counts.values()
        )
        assert all(count == 0 for count in audit["duplicate_pair_keys"].values())
        assert all(
            count == 0
            for split_counts in audit["duplicate_positive_hashes"].values()
            for count in split_counts.values()
        )
        assert all(
            count == 0
            for split_counts in audit["missing_hashes"].values()
            for count in split_counts.values()
        )
        assert all(
            count == 0
            for count in audit[
                "observed_positive_pairs_labeled_negative"
            ].values()
        )
        assert all(
            split_counts["matches_requested"]
            for split_counts in audit["class_balance"].values()
        )

        for frame in splits.values():
            assert frame["pair_key"].is_unique
        test_positive_keys.extend(
            splits["test"].loc[
                splits["test"]["label"] == 1,
                "pair_key",
            ]
        )

    selected_keys = folds[0]["positive_selection_manifest"]["pair_key"].tolist()
    assert sorted(test_positive_keys) == sorted(selected_keys)
    assert len(test_positive_keys) == len(set(test_positive_keys))


def test_selected_cv_coverage_sampler_keeps_endpoints_in_split():
    folds = _selected_cv_folds(negative_scope="coverage")

    for fold in folds:
        audit = fold["duplicate_stats"]["positive_pair_selection_fold_audit"]
        assert all(
            count == 0
            for side in audit["full_pair_hash_overlap"].values()
            for count in side.values()
        )
        assert all(
            count == 0
            for split_counts in audit["negative_endpoints_outside_split"].values()
            for count in split_counts.values()
        )
        assert all(
            count == 0
            for count in audit[
                "observed_positive_pairs_labeled_negative"
            ].values()
        )


def test_discarded_observed_positive_is_never_a_negative():
    folds = _selected_cv_folds()
    manifest_keys = set(folds[0]["positive_selection_manifest"]["pair_key"])
    all_observed = {
        "__".join(sorted((f"cds_a{i}", f"cds_b{i}")))
        for i in range(15)
    }
    all_observed.add("__".join(sorted(("cds_a0", "cds_b1"))))
    discarded = all_observed - manifest_keys
    assert len(discarded) == 1

    for fold in folds:
        for split_name in ("train_pairs", "val_pairs", "test_pairs"):
            negatives = fold[split_name].loc[
                fold[split_name]["label"] == 0,
                "pair_key",
            ]
            assert discarded.isdisjoint(set(negatives))


def test_selected_cv_is_deterministic_for_fixed_seed():
    first = _selected_cv_folds()
    second = _selected_cv_folds()

    for first_fold, second_fold in zip(first, second):
        for split_name in ("train_pairs", "val_pairs", "test_pairs"):
            pd.testing.assert_frame_equal(
                first_fold[split_name],
                second_fold[split_name],
            )


def test_method_all_keeps_existing_cv_output():
    kwargs = {
        "df": _protein_frame(),
        "n_folds": 3,
        "seed": 42,
        "neg_to_pos_ratio": 1.0,
        "val_ratio": 0.2,
        "schema_pair": ("A", "B"),
        "axes_for_flags": [],
        "pair_key_alphabet": "nt_cds",
        "negative_scope": "within_fold",
    }
    default_folds = list(generate_all_cv_folds_v2(**kwargs))
    explicit_folds = list(
        generate_all_cv_folds_v2(
            **kwargs,
            positive_selection_method="all",
            positive_selection_ordering="pair_key",
        )
    )

    for default, explicit in zip(default_folds, explicit_folds):
        for split_name in ("train_pairs", "val_pairs", "test_pairs"):
            pd.testing.assert_frame_equal(
                default[split_name],
                explicit[split_name],
            )
