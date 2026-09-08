"""Build fixed-size Hopcroft-Karp datasets without changing production config."""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence

import pandas as pd

from src.datasets import dataset_segment_pairs_v2
from src.datasets._positive_pair_selection import (
    _selected_pair_keys_sha256,
)


def sample_selected_pairs(
    selected: pd.DataFrame,
    audit: dict,
    n_positives: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Randomly sample selected positives and update their selection audit.

    Args:
        selected: Positives returned by Hopcroft-Karp selection.
        audit: Selection audit returned with `selected`.
        n_positives: Number of positives to retain.
        seed: Random seed used to choose the retained rows.

    Returns:
        The sampled positive rows and updated audit.
    """
    if audit.get("method") != "hopcroft_karp":
        raise ValueError("equal-count sampling requires Hopcroft-Karp selection")
    if n_positives <= 0:
        raise ValueError(f"n_positives must be positive; got {n_positives}")
    if n_positives > len(selected):
        raise ValueError(
            f"cannot sample {n_positives} positives from the "
            f"{len(selected)} selected by Hopcroft-Karp"
        )

    sampled = selected.sample(
        n=n_positives,
        replace=False,
        random_state=seed,
    )
    sampled = sampled.sort_values("pair_key", kind="stable").reset_index(drop=True)

    updated = dict(audit)
    hash_col_a = str(updated["hash_col_a"])
    hash_col_b = str(updated["hash_col_b"])
    input_pairs = int(updated["input_pairs"])
    input_unique_a = int(updated["input_unique_a"])
    input_unique_b = int(updated["input_unique_b"])
    selected_unique_a = int(sampled[hash_col_a].nunique())
    selected_unique_b = int(sampled[hash_col_b].nunique())

    updated["random_subsample"] = {
        "method": "random_without_replacement",
        "seed": int(seed),
        "input_pairs": int(len(selected)),
        "selected_pairs": int(len(sampled)),
        "input_pair_keys_sha256": updated["selected_pair_keys_sha256"],
    }
    updated["selected_pairs"] = int(len(sampled))
    updated["dropped_pairs"] = input_pairs - len(sampled)
    updated["selected_unique_a"] = selected_unique_a
    updated["selected_unique_b"] = selected_unique_b
    updated["unmatched_a"] = input_unique_a - selected_unique_a
    updated["unmatched_b"] = input_unique_b - selected_unique_b
    updated["retained_fraction"] = len(sampled) / input_pairs
    updated["selected_pair_keys_sha256"] = _selected_pair_keys_sha256(
        sampled["pair_key"]
    )
    return sampled, updated


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dataset CLI with fixed-size post-matching positive samples."""
    parser = argparse.ArgumentParser(
        description=(
            "Run dataset_segment_pairs after randomly sampling a fixed number "
            "of Hopcroft-Karp positives. Remaining arguments are passed to "
            "dataset_segment_pairs."
        )
    )
    parser.add_argument("--n_positives", type=int, required=True)
    parser.add_argument("--sample_seed", type=int, required=True)
    args, dataset_args = parser.parse_known_args(argv)
    if "--config_bundle" not in dataset_args:
        parser.error("dataset_segment_pairs requires --config_bundle")

    original_selector = dataset_segment_pairs_v2.select_positive_pairs

    def select_then_sample(*selector_args, **selector_kwargs):
        selected, audit = original_selector(*selector_args, **selector_kwargs)
        return sample_selected_pairs(
            selected,
            audit,
            n_positives=args.n_positives,
            seed=args.sample_seed,
        )

    original_argv = sys.argv
    dataset_segment_pairs_v2.select_positive_pairs = select_then_sample
    try:
        sys.argv = ["dataset_segment_pairs.py", *dataset_args]
        runpy.run_module("src.datasets.dataset_segment_pairs", run_name="__main__")
    finally:
        dataset_segment_pairs_v2.select_positive_pairs = original_selector
        sys.argv = original_argv


if __name__ == "__main__":
    main()
