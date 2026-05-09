"""Pair feature loader for non-MLP baselines (logistic regression, k-NN, ...).

Returns dense numpy arrays (`X`, `y`) per split plus an optional fitted
``StandardScaler``. The MLP path keeps its torch ``Dataset``-based pipeline;
this module exists so sklearn-style baselines can avoid that machinery and
operate on plain matrices.

Supported feature sources
-------------------------
- ``feature_source='kmer'``: k-mer concat features (slot_transform='none',
  interaction='concat'). The k-mer path is intentionally narrow because
  k-mer features are interaction-agnostic in practice (per project
  findings: ``unit_diff ≈ concat`` on k-mer) and slot_norm doesn't
  benefit non-negative count vectors.
- ``feature_source='esm2'``: ESM-2 protein embeddings, with optional
  ``slot_transform`` (``'none'`` or ``'slot_norm'``) and any
  ``interaction`` from ``{concat, diff, unit_diff, prod}`` or
  ``+``-separated combinations (matches ``train_pair_classifier.py``'s
  ``parse_interaction_flags`` syntax).

Pipeline (ESM-2 path)
---------------------
1. Load per-pair (emb_a, emb_b) from the master HDF5 cache.
2. Apply ``slot_transform`` to each slot independently:
   - ``none``: identity.
   - ``slot_norm``: numpy LayerNorm (zero-mean / unit-variance per row).
     Uses the unparameterized form (gain=1, bias=0) — this differs from
     the trained MLP's ``nn.LayerNorm`` (which has learned gain/bias).
     The intent is to give the baseline the SAME pre-interaction
     normalization the MLP enjoys, without the MLP's training. Using
     trained gain/bias would couple the baseline's input to a specific
     MLP run, defeating the "MLP vs 1-NN, same input" comparison.
3. Apply ``interaction`` to the (possibly transformed) embeddings:
   - ``concat``: ``[emb_a, emb_b]`` (output dim = 2D).
   - ``diff``: ``|emb_a - emb_b|`` (output dim = D).
   - ``unit_diff``: ``(emb_a - emb_b) / ||emb_a - emb_b||`` (output dim = D).
   - ``prod``: ``emb_a * emb_b`` (output dim = D).
   - Combinations like ``concat+unit_diff`` concatenate the listed
     features along the feature axis.

The k-mer path applies ``slot_transform='none'`` and
``interaction='concat'`` only. Other combinations raise to keep the
baseline path predictable; if you need k-mer + unit_diff for an
experiment, extend ``_load_kmer_pair_features`` here.
"""
from pathlib import Path
from typing import Optional

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.kmer_utils import (
    get_kmer_pair_features,
    load_kmer_index,
    load_kmer_matrix,
)


# Numerical floor for unit_diff normalization (mirrors the MLP path).
_UNIT_DIFF_EPS = 1e-8
# LayerNorm denominator floor (mirrors torch.nn.LayerNorm default ``eps=1e-5``).
_LAYER_NORM_EPS = 1e-5


def _parse_interaction_flags(
    interaction: str,
) -> tuple[bool, bool, bool, bool]:
    """Mirror of ``train_pair_classifier.parse_interaction_flags``.

    Duplicated here to keep the baselines path free of imports from
    the MLP trainer (which pulls torch). Kept in lockstep with the MLP
    parser — if syntax changes there, mirror it here.
    """
    if interaction is None:
        return False, False, False, False
    tokens = [t.strip().lower() for t in str(interaction).split("+") if t.strip()]
    allowed = {"concat", "diff", "prod", "unit_diff"}
    unknown = [t for t in tokens if t not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown interaction tokens: {unknown} (allowed: {sorted(allowed)})"
        )
    return (
        "concat" in tokens,
        "diff" in tokens,
        "prod" in tokens,
        "unit_diff" in tokens,
    )


def _apply_slot_norm(emb: np.ndarray) -> np.ndarray:
    """Numpy unparameterized LayerNorm along the feature axis.

    Mirrors ``torch.nn.LayerNorm(D)(emb)`` with default gain=1, bias=0
    and ``eps=1e-5``. Operates row-wise: each row is rescaled to
    zero-mean, unit-variance across its feature dimension.

    Reserved for the ESM-2 baseline path where slot-level normalization
    matters before the interaction step. Not called for k-mer features
    (non-negative count vectors don't benefit from feature-axis
    standardization).
    """
    mu = emb.mean(axis=-1, keepdims=True)
    sd = emb.std(axis=-1, keepdims=True)
    return (emb - mu) / (sd + _LAYER_NORM_EPS)


def _interaction_block(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    interaction: str,
) -> np.ndarray:
    """Apply the requested interaction(s) to per-slot embeddings.

    Output column order matches the MLP trainer (concat, diff,
    unit_diff, prod) when multiple flags are set, so the resulting
    feature space is bit-for-bit comparable across MLP and baseline
    runs that use the same ``interaction`` string.
    """
    use_concat, use_diff, use_prod, use_unit_diff = _parse_interaction_flags(interaction)
    if not (use_concat or use_diff or use_prod or use_unit_diff):
        raise ValueError(
            f"interaction={interaction!r} produced no active flags. "
            f"Choose from concat / diff / unit_diff / prod or "
            f"+-separated combinations (e.g., 'concat+unit_diff')."
        )
    parts = []
    if use_concat:
        parts.append(emb_a)
        parts.append(emb_b)
    if use_diff:
        parts.append(np.abs(emb_a - emb_b))
    if use_unit_diff:
        diff_raw = emb_a - emb_b
        norms = np.maximum(
            np.linalg.norm(diff_raw, axis=1, keepdims=True), _UNIT_DIFF_EPS
        )
        parts.append(diff_raw / norms)
    if use_prod:
        parts.append(emb_a * emb_b)
    return np.concatenate(parts, axis=1)


def _load_esm2_pair_features(
    pairs_df: pd.DataFrame,
    embeddings_file: Path,
    id_to_row: dict,
    *,
    slot_transform: str,
    interaction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ESM-2 (emb_a, emb_b) per pair, apply slot_transform, then
    interaction. Returns ``(X, y)`` as float32 numpy arrays.

    Drops pairs whose ``brc_a`` or ``brc_b`` isn't in the embedding
    index; the caller's pair CSV row order is preserved among kept
    rows. Mirrors the missing-pair handling in
    ``embedding_utils.create_pair_embeddings_concatenation``.
    """
    if slot_transform not in {"none", "slot_norm"}:
        raise ValueError(
            f"slot_transform={slot_transform!r} not supported in baseline harness "
            f"(use 'none' or 'slot_norm'). The trainable variants "
            f"(shared, slot_specific, shared_adapter) live inside the MLP."
        )

    # Vectorized brc -> embedding-row mapping. .map yields NaN for
    # unknowns; we drop those before reading the HDF5 cache.
    a_idx = pairs_df["brc_a"].map(id_to_row)
    b_idx = pairs_df["brc_b"].map(id_to_row)
    valid = a_idx.notna() & b_idx.notna()
    if not valid.any():
        raise RuntimeError(
            "All pairs were missing from the ESM-2 embedding index — "
            "check feature_source / embeddings_file alignment."
        )
    if (~valid).any():
        n_drop = int((~valid).sum())
        print(f"  WARNING: {n_drop} pairs dropped (missing ESM-2 embedding for at least one slot)")

    a_rows = a_idx.loc[valid].astype(int).to_numpy()
    b_rows = b_idx.loc[valid].astype(int).to_numpy()
    labels = pairs_df.loc[valid, "label"].to_numpy()

    # h5py fancy indexing requires increasing/unique indices; collapse
    # to the unique set per slot and re-expand afterwards.
    with h5py.File(embeddings_file, "r") as f:
        if "emb" not in f:
            raise ValueError(
                f"Invalid embeddings file format: {embeddings_file}. "
                "Master format ('emb' dataset) required."
            )
        emb_data = f["emb"]
        a_unique, a_inverse = np.unique(a_rows, return_inverse=True)
        b_unique, b_inverse = np.unique(b_rows, return_inverse=True)
        emb_a = emb_data[a_unique][a_inverse]
        emb_b = emb_data[b_unique][b_inverse]

    # Apply slot_transform per slot before the interaction.
    if slot_transform == "slot_norm":
        emb_a = _apply_slot_norm(emb_a)
        emb_b = _apply_slot_norm(emb_b)

    X = _interaction_block(emb_a, emb_b, interaction).astype(np.float32, copy=False)
    return X, labels.astype(np.int64, copy=False)


def _load_esm2_brc_index(embeddings_file: Path) -> dict:
    """Load the ``brc_fea_id`` -> embedding-row mapping from the
    parquet index sidecar (``<embeddings_file>.parquet``).

    Mirrors ``ESMPairDataset.__init__`` in the MLP trainer — column
    names are pinned to ``'brc_fea_id'`` / ``'row'`` to match what
    Stage 2 (``compute_esm2_embeddings.py``) writes. If the schema
    changes there, mirror the change here.
    """
    index_file = Path(embeddings_file).with_suffix(".parquet")
    if not index_file.exists():
        raise FileNotFoundError(
            f"Embedding index parquet not found at {index_file}. "
            "Stage 2 (compute_esm2_embeddings.py) writes this alongside the .h5."
        )
    df = pd.read_parquet(index_file)
    return dict(zip(df["brc_fea_id"], df["row"]))


def load_pair_features_for_baselines(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    *,
    feature_source: str,
    feature_scaling: str,
    output_dir: Path,
    # k-mer specific (required when feature_source='kmer')
    kmer_dir: Optional[Path] = None,
    kmer_k: Optional[int] = None,
    # ESM-2 specific (required when feature_source='esm2')
    embeddings_file: Optional[Path] = None,
    # Interaction & slot transform (apply to either source where supported)
    interaction: str = "concat",
    slot_transform: str = "none",
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        Optional[StandardScaler],
    ]:
    """Materialize per-split pair features for sklearn baselines.

    Args:
        train_pairs / val_pairs / test_pairs: pair CSVs as DataFrames.
        feature_source: ``'kmer'`` or ``'esm2'``.
        feature_scaling: ``'none'`` or ``'standard'``. When
            ``'standard'``, fit ``StandardScaler`` on train, transform
            all splits, persist the scaler. Note: cosine-distance
            classifiers (e.g., the k-NN baseline) should use ``'none'``
            because StandardScaler shifts non-negative count features
            and changes cosine geometry.
        output_dir: where the fitted scaler is saved (only when
            scaling is ``'standard'``).
        kmer_dir / kmer_k: for the k-mer path. Required when
            feature_source='kmer'.
        embeddings_file: for the ESM-2 path. Required when
            feature_source='esm2'.
        interaction: ``'concat'`` (default), ``'diff'``,
            ``'unit_diff'``, ``'prod'``, or ``+``-separated. The
            k-mer path supports ``'concat'`` only — others raise.
        slot_transform: ``'none'`` (default) or ``'slot_norm'``. The
            k-mer path supports ``'none'`` only — others raise.

    Returns:
        ``((X_train, y_train), (X_val, y_val), (X_test, y_test),
        scaler)``. ``scaler`` is ``None`` when ``feature_scaling ==
        'none'``.
    """
    if feature_source not in {"kmer", "esm2"}:
        raise NotImplementedError(
            f"Baseline feature loader supports feature_source in "
            f"{{'kmer', 'esm2'}}; got {feature_source!r}."
        )
    if feature_scaling not in {"none", "standard"}:
        raise ValueError(
            f"feature_scaling must be 'none' or 'standard'; got {feature_scaling!r}"
        )

    if feature_source == "kmer":
        if interaction != "concat":
            raise NotImplementedError(
                f"k-mer baseline path supports interaction='concat' only "
                f"(got {interaction!r}). Project finding: k-mer is "
                f"interaction-agnostic in practice (unit_diff ≈ concat) "
                f"so a separate path hasn't been wired."
            )
        if slot_transform != "none":
            raise NotImplementedError(
                f"k-mer baseline path supports slot_transform='none' only "
                f"(got {slot_transform!r}). Non-negative count vectors "
                f"don't benefit from feature-axis LayerNorm."
            )
        if kmer_dir is None or kmer_k is None:
            raise ValueError("kmer_dir and kmer_k are required for feature_source='kmer'.")
        print(f"\nLoading k-mer pair features (k={kmer_k}) from {kmer_dir}")
        key_to_row = load_kmer_index(Path(kmer_dir), kmer_k)
        kmer_matrix = load_kmer_matrix(Path(kmer_dir), kmer_k)
        print(f"  k-mer matrix: {kmer_matrix.shape}")
        X_train, y_train = get_kmer_pair_features(train_pairs, kmer_matrix, key_to_row, interaction="concat")
        X_val, y_val = get_kmer_pair_features(val_pairs, kmer_matrix, key_to_row, interaction="concat")
        X_test, y_test = get_kmer_pair_features(test_pairs, kmer_matrix, key_to_row, interaction="concat")

    else:  # feature_source == "esm2"
        if embeddings_file is None:
            raise ValueError("embeddings_file is required for feature_source='esm2'.")
        embeddings_file = Path(embeddings_file)
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        print(f"\nLoading ESM-2 pair features from {embeddings_file}")
        print(f"  slot_transform={slot_transform}, interaction={interaction}")
        id_to_row = _load_esm2_brc_index(embeddings_file)
        X_train, y_train = _load_esm2_pair_features(
            train_pairs, embeddings_file, id_to_row,
            slot_transform=slot_transform, interaction=interaction,
        )
        X_val, y_val = _load_esm2_pair_features(
            val_pairs, embeddings_file, id_to_row,
            slot_transform=slot_transform, interaction=interaction,
        )
        X_test, y_test = _load_esm2_pair_features(
            test_pairs, embeddings_file, id_to_row,
            slot_transform=slot_transform, interaction=interaction,
        )

    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    scaler: Optional[StandardScaler] = None
    if feature_scaling == "standard":
        print("Fitting StandardScaler on training features (per-feature mean/std)...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_val = scaler.transform(X_val).astype(np.float32, copy=False)
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)
        scaler_path = Path(output_dir) / "feature_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"  fitted on {len(X_train):,} train rows ({X_train.shape[1]:,} features)")
        print(f"  saved scaler to: {scaler_path}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
