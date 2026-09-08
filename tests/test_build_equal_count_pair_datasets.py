import pandas as pd
import pytest

from src.analysis.build_equal_count_pair_datasets import sample_selected_pairs
from src.datasets._positive_pair_selection import _selected_pair_keys_sha256


def _selected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_key": [f"pair_{idx}" for idx in range(6)],
            "hash_a": [f"a_{idx}" for idx in range(6)],
            "hash_b": [f"b_{idx}" for idx in range(6)],
        }
    )


def _audit(selected: pd.DataFrame) -> dict:
    return {
        "method": "hopcroft_karp",
        "hash_col_a": "hash_a",
        "hash_col_b": "hash_b",
        "input_pairs": 10,
        "input_unique_a": 8,
        "input_unique_b": 9,
        "selected_pair_keys_sha256": _selected_pair_keys_sha256(
            selected["pair_key"]
        ),
    }


def test_sample_selected_pairs_is_deterministic_and_updates_audit():
    selected = _selected()

    sampled, audit = sample_selected_pairs(selected, _audit(selected), 4, 17)
    repeated, repeated_audit = sample_selected_pairs(
        selected, _audit(selected), 4, 17
    )

    pd.testing.assert_frame_equal(sampled, repeated)
    assert audit == repeated_audit
    assert len(sampled) == 4
    assert sampled["hash_a"].is_unique
    assert sampled["hash_b"].is_unique
    assert audit["selected_pairs"] == 4
    assert audit["dropped_pairs"] == 6
    assert audit["random_subsample"]["input_pairs"] == 6
    assert audit["random_subsample"]["selected_pairs"] == 4
    assert audit["random_subsample"]["seed"] == 17
    assert audit["selected_pair_keys_sha256"] == _selected_pair_keys_sha256(
        sampled["pair_key"]
    )


def test_sample_selected_pairs_rejects_oversized_target():
    selected = _selected()

    with pytest.raises(ValueError, match="cannot sample 7 positives"):
        sample_selected_pairs(selected, _audit(selected), 7, 17)


def test_sample_selected_pairs_requires_hopcroft_karp():
    selected = _selected()
    audit = _audit(selected)
    audit["method"] = "all"

    with pytest.raises(ValueError, match="requires Hopcroft-Karp"):
        sample_selected_pairs(selected, audit, 4, 17)
