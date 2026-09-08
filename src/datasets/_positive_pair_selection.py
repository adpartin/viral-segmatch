"""Select positive pairs whose sequence endpoints are unique within each slot."""

from __future__ import annotations

import hashlib

import networkx as nx
import pandas as pd

POSITIVE_PAIR_SELECTION_METHODS = {
    "all",
    "dedup_a_then_b",
    "dedup_b_then_a",
    "hopcroft_karp",
}
POSITIVE_PAIR_SELECTION_ORDERINGS = {"pair_key"}


def _selected_pair_keys_sha256(pair_keys: pd.Series) -> str:
    """Return the SHA-256 checksum of the sorted selected pair keys.

    Args:
        pair_keys: Selected pair keys.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    payload = "\n".join(sorted(pair_keys.astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hopcroft_karp_row_indices(
    ordered: pd.DataFrame,
    hash_col_a: str,
    hash_col_b: str,
) -> list[int]:
    """Return row indices for a deterministic maximum-cardinality matching.

    Args:
        ordered: Positive pairs in deterministic row order.
        hash_col_a: Sequence-hash column for slot A.
        hash_col_b: Sequence-hash column for slot B.

    Returns:
        Sorted row indices of the selected matching edges.
    """
    left_hashes = sorted(ordered[hash_col_a].astype(str).unique())
    right_hashes = sorted(ordered[hash_col_b].astype(str).unique())

    left_ids = {value: idx for idx, value in enumerate(left_hashes)}
    right_offset = len(left_ids)
    right_ids = {
        value: right_offset + idx
        for idx, value in enumerate(right_hashes)
    }

    graph = nx.Graph()
    graph.add_nodes_from(left_ids.values(), bipartite=0)
    graph.add_nodes_from(right_ids.values(), bipartite=1)

    edge_to_row: dict[tuple[int, int], int] = {}
    for row_idx, (hash_a, hash_b) in enumerate(
        zip(ordered[hash_col_a].astype(str), ordered[hash_col_b].astype(str))
    ):
        edge = (left_ids[hash_a], right_ids[hash_b])
        if edge in edge_to_row:
            raise ValueError(
                "positive-pair selection received duplicate slot-A/slot-B hash pairs"
            )
        edge_to_row[edge] = row_idx

    graph.add_edges_from(sorted(edge_to_row))
    matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(
        graph,
        top_nodes=set(left_ids.values()),
    )

    selected = [
        row_idx
        for (left_id, right_id), row_idx in edge_to_row.items()
        if matching.get(left_id) == right_id
    ]
    return sorted(selected)


def select_positive_pairs(
    pos_df: pd.DataFrame,
    method: str,
    hash_col_a: str,
    hash_col_b: str,
    ordering: str = "pair_key",
) -> tuple[pd.DataFrame, dict]:
    """Select positive pairs and report the retained sequence population.

    Args:
        pos_df: Globally deduplicated positive pairs.
        method: One of `all`, `dedup_a_then_b`, `dedup_b_then_a`, or
            `hopcroft_karp`.
        hash_col_a: Sequence-hash column for slot A.
        hash_col_b: Sequence-hash column for slot B.
        ordering: Deterministic ordering used before active selection. Only
            `pair_key` is supported.

    Returns:
        The selected positive rows and a selection audit.
    """
    if method not in POSITIVE_PAIR_SELECTION_METHODS:
        raise ValueError(
            f"positive_pair_selection.method must be one of "
            f"{sorted(POSITIVE_PAIR_SELECTION_METHODS)}; got {method!r}"
        )
    if ordering not in POSITIVE_PAIR_SELECTION_ORDERINGS:
        raise ValueError(
            f"positive_pair_selection.ordering must be one of "
            f"{sorted(POSITIVE_PAIR_SELECTION_ORDERINGS)}; got {ordering!r}"
        )

    required = ["pair_key", hash_col_a, hash_col_b]
    missing = [column for column in required if column not in pos_df.columns]
    if missing:
        raise ValueError(
            f"positive-pair selection is missing required columns: {missing}"
        )
    if pos_df[required].isna().any().any():
        raise ValueError(
            "positive-pair selection requires non-null pair keys and sequence hashes"
        )
    if not pos_df["pair_key"].is_unique:
        raise ValueError(
            "positive-pair selection requires globally deduplicated pair_key rows"
        )

    if method == "all":
        selected = pos_df.copy()
    else:
        ordered = pos_df.sort_values("pair_key", kind="stable").reset_index(
            drop=True
        )
        if method == "dedup_a_then_b":
            selected = (
                ordered
                .drop_duplicates(hash_col_a, keep="first")
                .drop_duplicates(hash_col_b, keep="first")
            )
        elif method == "dedup_b_then_a":
            selected = (
                ordered
                .drop_duplicates(hash_col_b, keep="first")
                .drop_duplicates(hash_col_a, keep="first")
            )
        else:
            row_indices = _hopcroft_karp_row_indices(
                ordered,
                hash_col_a,
                hash_col_b,
            )
            selected = ordered.iloc[row_indices]
        selected = selected.reset_index(drop=True)

        if not selected[hash_col_a].is_unique:
            raise RuntimeError(
                "positive-pair selection did not make slot A unique"
            )
        if not selected[hash_col_b].is_unique:
            raise RuntimeError(
                "positive-pair selection did not make slot B unique"
            )

    if not selected["pair_key"].is_unique:
        raise RuntimeError(
            "positive-pair selection produced duplicate pair keys"
        )
    if not set(selected["pair_key"]).issubset(set(pos_df["pair_key"])):
        raise RuntimeError(
            "positive-pair selection produced an unknown pair"
        )

    input_unique_a = int(pos_df[hash_col_a].nunique())
    input_unique_b = int(pos_df[hash_col_b].nunique())
    selected_unique_a = int(selected[hash_col_a].nunique())
    selected_unique_b = int(selected[hash_col_b].nunique())
    input_pairs = int(len(pos_df))
    selected_pairs = int(len(selected))

    audit = {
        "method": method,
        "hash_col_a": hash_col_a,
        "hash_col_b": hash_col_b,
        "input_pairs": input_pairs,
        "selected_pairs": selected_pairs,
        "dropped_pairs": input_pairs - selected_pairs,
        "input_unique_a": input_unique_a,
        "input_unique_b": input_unique_b,
        "selected_unique_a": selected_unique_a,
        "selected_unique_b": selected_unique_b,
        "unmatched_a": input_unique_a - selected_unique_a,
        "unmatched_b": input_unique_b - selected_unique_b,
        "retained_fraction": (
            selected_pairs / input_pairs if input_pairs else 0.0
        ),
        "ordering": ordering,
        "networkx_version": nx.__version__ if method == "hopcroft_karp" else None,
        "selected_pair_keys_sha256": _selected_pair_keys_sha256(
            selected["pair_key"]
        ),
    }
    return selected, audit


def positive_pair_selection_manifest(
    selected: pd.DataFrame,
    hash_col_a: str,
    hash_col_b: str,
) -> pd.DataFrame:
    """Return the selected pair keys, endpoint hashes, and representative isolate.

    Args:
        selected: Selected positive-pair rows.
        hash_col_a: Sequence-hash column for slot A.
        hash_col_b: Sequence-hash column for slot B.

    Returns:
        Manifest sorted by pair key.
    """
    required = ["pair_key", hash_col_a, hash_col_b, "assembly_id_a"]
    missing = [column for column in required if column not in selected.columns]
    if missing:
        raise ValueError(
            f"positive-pair selection manifest is missing required columns: {missing}"
        )
    if selected[required].isna().any().any():
        raise ValueError(
            "positive-pair selection manifest requires non-null pair keys, "
            "sequence hashes, and assembly IDs"
        )

    manifest = selected[required].copy()
    manifest = manifest.rename(columns={"assembly_id_a": "assembly_id"})
    return manifest.sort_values("pair_key", kind="stable").reset_index(drop=True)


def audit_positive_pair_selection_fold(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    hash_col_a: str,
    hash_col_b: str,
    cooccur_pairs: set,
    neg_to_pos_ratio: float,
) -> dict:
    """Verify sequence-disjoint selected splits and in-split negative endpoints.

    Args:
        train_pairs: Final training pairs.
        val_pairs: Final validation pairs.
        test_pairs: Final test pairs.
        hash_col_a: Sequence-hash column for slot A.
        hash_col_b: Sequence-hash column for slot B.
        cooccur_pairs: Full set of observed positive pair keys.
        neg_to_pos_ratio: Requested negative-to-positive ratio.

    Returns:
        Counts for cross-split overlap, endpoint scope, duplicates, observed
        positives labeled negative, and class balance.
    """
    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }
    positive_pools = {
        name: {
            "a": set(frame.loc[frame["label"] == 1, hash_col_a].astype(str)),
            "b": set(frame.loc[frame["label"] == 1, hash_col_b].astype(str)),
        }
        for name, frame in splits.items()
    }
    full_pools = {
        name: {
            "a": set(frame[hash_col_a].dropna().astype(str)),
            "b": set(frame[hash_col_b].dropna().astype(str)),
        }
        for name, frame in splits.items()
    }

    overlap = {}
    for side in ("a", "b"):
        overlap[side] = {
            "train_val": len(full_pools["train"][side] & full_pools["val"][side]),
            "train_test": len(full_pools["train"][side] & full_pools["test"][side]),
            "val_test": len(full_pools["val"][side] & full_pools["test"][side]),
        }
    if any(count for side in overlap.values() for count in side.values()):
        raise RuntimeError(
            "positive-pair selection produced cross-split sequence-hash overlap"
        )

    negative_endpoints_outside_split = {}
    duplicate_pair_keys = {}
    duplicate_positive_hashes = {}
    missing_hashes = {}
    observed_negatives = {}
    class_balance = {}
    for name, frame in splits.items():
        positives = frame[frame["label"] == 1]
        negatives = frame[frame["label"] == 0]
        n_positive = int(len(positives))
        n_negative = int(len(negatives))
        requested_negative = int(round(n_positive * neg_to_pos_ratio))
        class_balance[name] = {
            "positive": n_positive,
            "negative": n_negative,
            "requested_negative": requested_negative,
            "matches_requested": n_negative == requested_negative,
        }
        duplicate_positive_hashes[name] = {
            "a": int(positives[hash_col_a].duplicated().sum()),
            "b": int(positives[hash_col_b].duplicated().sum()),
        }
        missing_hashes[name] = {
            "a": int(frame[hash_col_a].isna().sum()),
            "b": int(frame[hash_col_b].isna().sum()),
        }
        outside_a = (
            set(negatives[hash_col_a].dropna().astype(str))
            - positive_pools[name]["a"]
        )
        outside_b = (
            set(negatives[hash_col_b].dropna().astype(str))
            - positive_pools[name]["b"]
        )
        negative_endpoints_outside_split[name] = {
            "a": len(outside_a),
            "b": len(outside_b),
        }
        duplicate_pair_keys[name] = int(frame["pair_key"].duplicated().sum())
        observed_negatives[name] = len(
            set(negatives["pair_key"].astype(str)) & cooccur_pairs
        )

    if any(
        count
        for split_counts in negative_endpoints_outside_split.values()
        for count in split_counts.values()
    ):
        raise RuntimeError(
            "a negative pair uses a sequence outside its assigned split"
        )
    if any(duplicate_pair_keys.values()):
        raise RuntimeError(
            "positive-pair selection produced duplicate pair keys within a split"
        )
    if any(
        count
        for split_counts in duplicate_positive_hashes.values()
        for count in split_counts.values()
    ):
        raise RuntimeError(
            "positive-pair selection reused a positive sequence within a split"
        )
    if any(
        count
        for split_counts in missing_hashes.values()
        for count in split_counts.values()
    ):
        raise RuntimeError(
            "a selected CV pair is missing a sequence hash"
        )
    if any(observed_negatives.values()):
        raise RuntimeError(
            "an observed positive co-occurrence was retained as a negative"
        )

    return {
        "full_pair_hash_overlap": overlap,
        "negative_endpoints_outside_split": negative_endpoints_outside_split,
        "duplicate_pair_keys": duplicate_pair_keys,
        "duplicate_positive_hashes": duplicate_positive_hashes,
        "missing_hashes": missing_hashes,
        "observed_positive_pairs_labeled_negative": observed_negatives,
        "class_balance": class_balance,
    }
