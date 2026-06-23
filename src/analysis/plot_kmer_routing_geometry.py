"""
Routing-geometry visualization for paired k-mer datasets.

Produces 2-D embedding plots that visualize how the train/val/test
partition separates as the Stage 3 split strategy moves from `random` →
`seq_disjoint` → `cluster_disjoint`. Inspired by DataSAIL Fig. 4
(two-input task; per-entity 2-D embeddings, color = split assignment).

Two complementary plot types per Stage 3 run:

- **Sequence-level** (`plot_kmer_sequence_geometry`): one point per
  unique protein sequence appearing in any pair, projected from its
  full k-mer feature vector. Two subplots side-by-side, one per
  function type (e.g. HA on the left, NA on the right). Color encodes
  which split the sequence appears in; for `random` routing a 4th
  color flags sequences that appear in multiple splits (leakage).

- **Pair-level** (`plot_kmer_pair_geometry`): one point per sampled
  (a, b) pair, projected from `concat(kmer_a, kmer_b)`. Single panel.
  Color = split. Marker fill = label (positive filled, negative hollow).

Both functions emit a TruncatedSVD panel and a UMAP panel (when
umap-learn is available). TruncatedSVD (not PCA) is used because k-mer
features are non-negative count vectors — see the
`src/utils/dim_reduction_utils` module docstring for the rule. The
convenience wrapper `plot_kmer_routing_geometry` produces all four PNGs
in one call.

Used by `visualize_dataset_stats.py`'s end-of-stage-3 orchestrator;
also callable standalone against any existing run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.utils.dim_reduction_utils import (
    UMAP_AVAILABLE,
    compute_truncated_svd_reduction,
    compute_umap_reduction,
)
from src.utils.kmer_utils import load_kmer_index, load_kmer_matrix
from src.utils.plot_config import (
    LABEL_SCATTER_STYLES,
    SPLIT_COLORS,
    SPLIT_MARKERS,
)

# Neutral color for sequences that appear in multiple splits (random-routing
# leakage). Mirrors DataSAIL Fig. 4's gray "drop" category.
LEAKAGE_COLOR = "#7f7f7f"
LEAKAGE_KEY = "multiple"

# Splits considered, in canonical order. Train / val / test colors come from
# `SPLIT_COLORS`; the 4th leakage color is appended only when present.
SPLITS = ("train", "val", "test")


# ============================================================================
# Top-level entry points
# ============================================================================

def plot_kmer_routing_geometry(
    run_dir: Path,
    kmer_dir: Path,
    output_dir: Path,
    function_to_short: dict[str, str],
    alphabet: Optional[str] = None,
    k: Optional[int] = None,
    max_per_label_per_split: int = 1000,
    max_sequences_per_function: int = 5000,
    umap_pre_pca_dim: int = 50,
    random_state: int = 42,
) -> None:
    """Generate both sequence-level and pair-level routing-geometry plots.

    Args:
        run_dir: Stage 3 run directory containing {train,val,test}_pairs.csv.
        kmer_dir: Directory holding `kmer_features_{alphabet}_k{k}.npz` +
            its index parquet + metadata JSON.
        output_dir: Where PNGs are written.
        function_to_short: Mapping of full function name → short name
            (e.g. `{"Hemagglutinin precursor": "HA",
            "Neuraminidase protein": "NA"}`). Used for sequence-level
            subplot labels.
        alphabet: 'nt' or 'aa'. If None, picked from a metadata JSON in
            `kmer_dir` — fine when only one cache lives there; otherwise
            pass explicitly to disambiguate.
        k: k-mer size. If None, picked from metadata alongside alphabet.
        max_per_label_per_split: Pair-level sub-sampling cap.
        max_sequences_per_function: Sequence-level sub-sampling cap (per
            protein type), stratified by split-membership so the
            "appears in multiple splits" category survives sampling under
            random routing.
        umap_pre_pca_dim: PCA dimensionality to which sequence-level
            features are reduced before UMAP runs. Keeps UMAP runtime
            tractable for 4000–16000-column k-mer features without
            qualitatively changing the embedding (UMAP's input still
            captures most of the variance). Pair-level UMAP uses the
            same setting.
        random_state: Seed for sampling, PCA, UMAP.
    """
    plot_kmer_sequence_geometry(
        run_dir=run_dir,
        kmer_dir=kmer_dir,
        output_dir=output_dir,
        function_to_short=function_to_short,
        alphabet=alphabet,
        k=k,
        max_sequences_per_function=max_sequences_per_function,
        umap_pre_pca_dim=umap_pre_pca_dim,
        random_state=random_state,
    )
    plot_kmer_pair_geometry(
        run_dir=run_dir,
        kmer_dir=kmer_dir,
        output_dir=output_dir,
        alphabet=alphabet,
        k=k,
        max_per_label_per_split=max_per_label_per_split,
        umap_pre_pca_dim=umap_pre_pca_dim,
        random_state=random_state,
    )
    plot_kmer_pair_negatives_by_regime(
        run_dir=run_dir,
        kmer_dir=kmer_dir,
        output_dir=output_dir,
        alphabet=alphabet,
        k=k,
        max_per_regime=500,
        umap_pre_pca_dim=umap_pre_pca_dim,
        random_state=random_state,
    )


def plot_kmer_sequence_geometry(
    run_dir: Path,
    kmer_dir: Path,
    output_dir: Path,
    function_to_short: dict[str, str],
    alphabet: Optional[str] = None,
    k: Optional[int] = None,
    max_sequences_per_function: int = 5000,
    umap_pre_pca_dim: int = 50,
    random_state: int = 42,
) -> None:
    """Sequence-level k-mer geometry: PCA + UMAP, one subplot per function.

    Each unique protein sequence (by `prot_hash`) appearing in any pair is
    rendered once. Color encodes which split(s) the sequence appears in;
    sequences that appear in more than one split get the leakage color.

    Sub-samples to at most `max_sequences_per_function` per protein type,
    stratified by split-membership (so the "appears in multiple splits"
    category survives sampling even when it's a small minority — critical
    for the random-routing panel to render its leakage signal).

    UMAP runs after a TruncatedSVD reduction to `umap_pre_pca_dim` for
    tractability on high-dim k-mer features (parameter name kept for API
    backwards-compatibility but the reduction is TruncatedSVD).

    Outputs (under `output_dir`):
        - `kmer_sequence_svd.png`
        - `kmer_sequence_umap.png`  (when UMAP is available)
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    k, alphabet = _resolve_k_and_alphabet(kmer_dir, alphabet=alphabet, k=k)
    kmer_matrix = load_kmer_matrix(kmer_dir, k=k, alphabet=alphabet)
    key_to_row = load_kmer_index(kmer_dir, k=k, alphabet=alphabet)

    pairs = _load_all_pairs(run_dir)
    pairs = _attach_short_names(pairs, function_to_short)

    seq_table = _build_sequence_split_table(pairs, key_to_row, alphabet)
    if seq_table.empty:
        print(
            f"WARNING: plot_kmer_sequence_geometry: no sequences with k-mer rows in {run_dir}"
        )
        return

    seq_table = _stratified_sample_sequences(
        seq_table, max_per_function=max_sequences_per_function,
        random_state=random_state,
    )

    rows = seq_table["kmer_row"].to_numpy()
    features = np.asarray(kmer_matrix[rows].todense(), dtype=np.float32)

    n_total = len(seq_table)
    n_multi = int((seq_table["color_key"] == LEAKAGE_KEY).sum())
    print(
        f"plot_kmer_sequence_geometry: {n_total:,} unique sequences "
        f"(in-multiple-splits: {n_multi:,}); features shape={features.shape}"
    )

    # 2-D panel: TruncatedSVD directly (k-mer features are non-negative
    # count vectors — see dim_reduction_utils module docstring for why
    # TruncatedSVD, not PCA, is the right tool here).
    #
    # Axis labels intentionally omit per-component variance ratios:
    # TruncatedSVD orders components by descending singular value, not by
    # projected variance. On uncentered count data the first singular
    # vector aligns with the corpus mean direction, so SVD1's projected
    # variance can be *smaller* than SVD2's — which would be confusing to
    # display ("SVD1 0.5% / SVD2 13% var" reads as if SVD2 is more
    # important when it really is in the TruncatedSVD ordering).
    svd_2d, _svd_model = compute_truncated_svd_reduction(
        features,
        n_components=2,
        return_model=True,
        algorithm="randomized",
        random_state=random_state,
    )
    _draw_sequence_panels(
        coords=svd_2d,
        seq_table=seq_table,
        xlab="SVD1",
        ylab="SVD2",
        sup_title=_sup_title("Sequence TruncatedSVD", alphabet, k, run_dir),
        output_path=output_dir / "kmer_sequence_svd.png",
    )

    # UMAP panel: pre-reduce with TruncatedSVD so UMAP's input is dense,
    # low-dim, and fast. Cap at the feature dim and N − 1.
    if UMAP_AVAILABLE:
        pre_dim = min(umap_pre_pca_dim, features.shape[0] - 1, features.shape[1])
        pre_svd, _ = compute_truncated_svd_reduction(
            features, n_components=pre_dim,
            algorithm="randomized", random_state=random_state,
        )
        umap_2d, _ = compute_umap_reduction(
            pre_svd, n_components=2, random_state=random_state,
        )
        _draw_sequence_panels(
            coords=umap_2d,
            seq_table=seq_table,
            xlab="UMAP 1",
            ylab="UMAP 2",
            sup_title=_sup_title(
                f"Sequence UMAP (TruncatedSVD-{pre_dim} pre-step)",
                alphabet, k, run_dir,
            ),
            output_path=output_dir / "kmer_sequence_umap.png",
        )
    else:
        print(
            "WARNING: plot_kmer_sequence_geometry: umap-learn not installed; "
            "skipping UMAP panel"
        )


def plot_kmer_pair_geometry(
    run_dir: Path,
    kmer_dir: Path,
    output_dir: Path,
    alphabet: Optional[str] = None,
    k: Optional[int] = None,
    max_per_label_per_split: int = 1000,
    umap_pre_pca_dim: int = 50,
    random_state: int = 42,
) -> None:
    """Pair-level k-mer geometry: PCA + UMAP, one panel.

    Feature vector per pair is `concat(kmer_a, kmer_b)`. Stratified
    sub-sampling caps the number of pairs per (split, label) cell.
    UMAP runs after a TruncatedSVD reduction to `umap_pre_pca_dim` so
    it stays tractable on the 2× alphabet^k feature dim (parameter name
    kept for API backwards-compatibility but the reduction is
    TruncatedSVD).

    Outputs (under `output_dir`):
        - `kmer_pair_svd.png`
        - `kmer_pair_umap.png`  (when UMAP is available)
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    k, alphabet = _resolve_k_and_alphabet(kmer_dir, alphabet=alphabet, k=k)
    kmer_matrix = load_kmer_matrix(kmer_dir, k=k, alphabet=alphabet)
    key_to_row = load_kmer_index(kmer_dir, k=k, alphabet=alphabet)

    pairs = _load_all_pairs(run_dir)
    pairs["label"] = pd.to_numeric(pairs["label"], errors="coerce")
    pairs = pairs[pairs["label"].isin([0, 1])].reset_index(drop=True)
    pairs = _stratified_sample_pairs(
        pairs, max_per_label_per_split, random_state
    )

    rows_a, valid_a = _lookup_kmer_rows(pairs, "a", key_to_row, alphabet)
    rows_b, valid_b = _lookup_kmer_rows(pairs, "b", key_to_row, alphabet)
    valid = (valid_a & valid_b).values

    n_in = len(pairs)
    pairs = pairs.loc[valid].reset_index(drop=True)
    rows_a = rows_a[valid]
    rows_b = rows_b[valid]
    dropped = n_in - len(pairs)
    if dropped:
        print(
            f"plot_kmer_pair_geometry: dropped {dropped:,}/{n_in:,} pairs "
            f"(missing k-mer rows)"
        )
    if len(pairs) == 0:
        print(f"WARNING: plot_kmer_pair_geometry: no valid pairs in {run_dir}")
        return

    emb_a = np.asarray(kmer_matrix[rows_a].todense(), dtype=np.float32)
    emb_b = np.asarray(kmer_matrix[rows_b].todense(), dtype=np.float32)
    features = np.concatenate([emb_a, emb_b], axis=1)
    print(
        f"plot_kmer_pair_geometry: {len(pairs):,} sampled pairs; "
        f"features shape={features.shape}"
    )

    # See sequence-level path above for why per-component variance
    # ratios are intentionally omitted from SVD axis labels.
    svd_2d, _svd_model = compute_truncated_svd_reduction(
        features,
        n_components=2,
        return_model=True,
        algorithm="randomized",
        random_state=random_state,
    )
    _draw_pair_panel(
        coords=svd_2d,
        pairs=pairs,
        xlab="SVD1",
        ylab="SVD2",
        title=_pair_title("Pair TruncatedSVD — concat", alphabet, k, run_dir),
        output_path=output_dir / "kmer_pair_svd.png",
    )

    if UMAP_AVAILABLE:
        pre_dim = min(umap_pre_pca_dim, features.shape[0] - 1, features.shape[1])
        pre_svd, _ = compute_truncated_svd_reduction(
            features, n_components=pre_dim,
            algorithm="randomized", random_state=random_state,
        )
        umap_2d, _ = compute_umap_reduction(
            pre_svd, n_components=2, random_state=random_state,
        )
        _draw_pair_panel(
            coords=umap_2d,
            pairs=pairs,
            xlab="UMAP 1",
            ylab="UMAP 2",
            title=_pair_title(
                f"Pair UMAP (TruncatedSVD-{pre_dim} pre-step) — concat",
                alphabet, k, run_dir,
            ),
            output_path=output_dir / "kmer_pair_umap.png",
        )
    else:
        print(
            "WARNING: plot_kmer_pair_geometry: umap-learn not installed; "
            "skipping UMAP panel"
        )


# ============================================================================
# Negatives-by-regime geometry
# ============================================================================

# 8-regime order (matches `_negative_regime_sampling._SPLIT_COMP_NEG_ORDER`
# in visualize_dataset_stats). Listed here too so the plotting module is
# self-contained.
_NEG_REGIMES = (
    "none_match",
    "host_only",
    "subtype_only",
    "year_only",
    "host_subtype_only",
    "host_year_only",
    "subtype_year_only",
    "host_subtype_year",
)


def plot_kmer_pair_negatives_by_regime(
    run_dir: Path,
    kmer_dir: Path,
    output_dir: Path,
    alphabet: Optional[str] = None,
    k: Optional[int] = None,
    max_per_regime: int = 500,
    umap_pre_pca_dim: int = 50,
    random_state: int = 42,
) -> None:
    """Pair-level k-mer geometry of *negatives only*, colored by `neg_regime`.

    Tests whether the 8 metadata-driven negative regimes (none_match,
    host_only, …, host_subtype_year) form distinct clusters in k-mer
    feature space, or are just metadata-level distinctions intermixed in
    feature space.

    Feature vector is the same `concat(kmer_a, kmer_b)` as the regular
    pair plot — only the row filter (label==0) and coloring scheme
    differ. Stratified sampling caps each regime at `max_per_regime`
    rows so under-represented regimes (e.g. `subtype_year_only`)
    survive sub-sampling.

    Outputs (under `output_dir`):
        - `kmer_pair_negatives_svd.png`
        - `kmer_pair_negatives_umap.png`  (when UMAP is available)
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    k, alphabet = _resolve_k_and_alphabet(kmer_dir, alphabet=alphabet, k=k)
    kmer_matrix = load_kmer_matrix(kmer_dir, k=k, alphabet=alphabet)
    key_to_row = load_kmer_index(kmer_dir, k=k, alphabet=alphabet)

    pairs = _load_all_pairs(run_dir)
    pairs["label"] = pd.to_numeric(pairs["label"], errors="coerce")
    pairs = pairs[pairs["label"] == 0].reset_index(drop=True)

    if "neg_regime" not in pairs.columns:
        print(
            f"WARNING: plot_kmer_pair_negatives_by_regime: no `neg_regime` "
            f"column in {run_dir} pair CSVs — skipping"
        )
        return

    pairs = _stratified_sample_by_regime(
        pairs, max_per_regime=max_per_regime, random_state=random_state,
    )
    if pairs.empty:
        print(
            f"WARNING: plot_kmer_pair_negatives_by_regime: no negative pairs "
            f"after sampling in {run_dir}"
        )
        return

    rows_a, valid_a = _lookup_kmer_rows(pairs, "a", key_to_row, alphabet)
    rows_b, valid_b = _lookup_kmer_rows(pairs, "b", key_to_row, alphabet)
    valid = (valid_a & valid_b).values

    n_in = len(pairs)
    pairs = pairs.loc[valid].reset_index(drop=True)
    rows_a = rows_a[valid]
    rows_b = rows_b[valid]
    if n_in - len(pairs):
        print(
            f"plot_kmer_pair_negatives_by_regime: dropped "
            f"{n_in - len(pairs):,}/{n_in:,} pairs (missing k-mer rows)"
        )
    if len(pairs) == 0:
        return

    emb_a = np.asarray(kmer_matrix[rows_a].todense(), dtype=np.float32)
    emb_b = np.asarray(kmer_matrix[rows_b].todense(), dtype=np.float32)
    features = np.concatenate([emb_a, emb_b], axis=1)
    print(
        f"plot_kmer_pair_negatives_by_regime: {len(pairs):,} sampled "
        f"negative pairs across {pairs['neg_regime'].nunique()} regimes; "
        f"features shape={features.shape}"
    )

    # See sequence-level path for why per-component variance ratios are
    # intentionally omitted from SVD axis labels.
    svd_2d, _svd_model = compute_truncated_svd_reduction(
        features, n_components=2, return_model=True,
        algorithm="randomized", random_state=random_state,
    )
    _draw_negatives_by_regime_panel(
        coords=svd_2d, pairs=pairs,
        xlab="SVD1", ylab="SVD2",
        title=_pair_title(
            "Negatives by regime — TruncatedSVD — concat",
            alphabet, k, run_dir,
        ),
        output_path=output_dir / "kmer_pair_negatives_svd.png",
    )

    if UMAP_AVAILABLE:
        pre_dim = min(umap_pre_pca_dim, features.shape[0] - 1, features.shape[1])
        pre_svd, _ = compute_truncated_svd_reduction(
            features, n_components=pre_dim,
            algorithm="randomized", random_state=random_state,
        )
        umap_2d, _ = compute_umap_reduction(
            pre_svd, n_components=2, random_state=random_state,
        )
        _draw_negatives_by_regime_panel(
            coords=umap_2d, pairs=pairs,
            xlab="UMAP 1", ylab="UMAP 2",
            title=_pair_title(
                f"Negatives by regime — UMAP (TruncatedSVD-{pre_dim} "
                f"pre-step) — concat",
                alphabet, k, run_dir,
            ),
            output_path=output_dir / "kmer_pair_negatives_umap.png",
        )
    else:
        print(
            "WARNING: plot_kmer_pair_negatives_by_regime: umap-learn not "
            "installed; skipping UMAP panel"
        )


def _stratified_sample_by_regime(
    pairs: pd.DataFrame, max_per_regime: int, random_state: int,
) -> pd.DataFrame:
    """Sub-sample at most `max_per_regime` rows per `neg_regime`."""
    chunks = []
    rng_seed = random_state
    for regime, grp in pairs.groupby("neg_regime", sort=False):
        if len(grp) <= max_per_regime:
            chunks.append(grp)
        else:
            chunks.append(grp.sample(n=max_per_regime, random_state=rng_seed))
            rng_seed += 1
    if not chunks:
        return pairs.iloc[0:0]
    return pd.concat(chunks, ignore_index=True)


def _regime_color_palette() -> dict[str, tuple]:
    """8-color palette keyed by regime, avoiding tab10's green (index 2)
    so the positive-bar color used elsewhere doesn't collide if these
    plots are stacked next to a split-composition figure."""
    tab10 = list(plt.get_cmap("tab10").colors)
    palette = [c for i, c in enumerate(tab10) if i != 2]
    return {r: palette[i % len(palette)] for i, r in enumerate(_NEG_REGIMES)}


def _draw_negatives_by_regime_panel(
    coords: np.ndarray, pairs: pd.DataFrame,
    xlab: str, ylab: str, title: str, output_path: Path,
) -> None:
    """Single-panel scatter: all negatives, color = `neg_regime`."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    color_for = _regime_color_palette()

    for regime in _NEG_REGIMES:
        mask = (pairs["neg_regime"] == regime).values
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            marker="o", s=18, alpha=0.7,
            facecolors=color_for[regime], edgecolors="none",
            label=f"{regime} (n={int(mask.sum()):,})",
        )

    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="neg_regime",
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=9, frameon=False,
    )
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Loading + sampling helpers
# ============================================================================

def _resolve_k_and_alphabet(
    kmer_dir: Path,
    alphabet: Optional[str] = None,
    k: Optional[int] = None,
) -> tuple[int, str]:
    """Resolve `(k, alphabet)` for the k-mer cache under `kmer_dir`.

    Priority:
      1. Explicit `alphabet` + `k` from the caller — used as-is.
      2. Metadata JSON `kmer_features_{alphabet}_k{k}_metadata.json` for
         the requested side (when only one of the two is given).
      3. The only metadata file present, if exactly one exists.

    Errors loudly if multiple caches coexist and the caller did not
    disambiguate — picking arbitrarily from a multi-cache dir was the
    source of an earlier subtle bug where aa k=3 was selected when the
    production config was nt k=6.
    """
    kmer_dir = Path(kmer_dir)
    if alphabet is not None and k is not None:
        meta_path = kmer_dir / f"kmer_features_{alphabet}_k{k}_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Requested k-mer cache not found: {meta_path}"
            )
        return int(k), str(alphabet)

    metas = sorted(kmer_dir.glob("kmer_features_*_metadata.json"))
    if not metas:
        raise FileNotFoundError(
            f"No kmer_features_*_metadata.json found under {kmer_dir}"
        )
    if len(metas) == 1:
        with open(metas[0]) as f:
            meta = json.load(f)
        return int(meta["k"]), str(meta.get("alphabet", "nt"))

    raise ValueError(
        f"Multiple k-mer caches in {kmer_dir} "
        f"({[m.name for m in metas]}); pass explicit `alphabet` and `k`"
    )


def _occ_col(alphabet: str, side: str) -> str:
    """Pair-table column for the kmer-occurrence id on one side."""
    suf = side.lower()
    if alphabet == "nt":
        return f"ctg_{suf}"
    if alphabet == "aa":
        return f"brc_{suf}"
    raise ValueError(f"alphabet must be 'nt' or 'aa'; got {alphabet!r}")


def _normalize_occurrence(series: pd.Series, alphabet: str) -> pd.Series:
    """Match `load_kmer_index`'s float-round-trip normalization for nt."""
    s = series.astype(str)
    if alphabet != "nt":
        return s
    return s.apply(
        lambda x: str(float(x)) if x.replace(".", "", 1).isdigit() else x
    )


def _load_all_pairs(run_dir: Path) -> pd.DataFrame:
    """Concatenate train/val/test pair CSVs, tagging each row with its split.

    Reads with `keep_default_na=False, na_values=['']` so the literal
    function-short value `'NA'` (Neuraminidase) is not silently nullified.
    """
    frames = []
    for split in SPLITS:
        path = Path(run_dir) / f"{split}_pairs.csv"
        if not path.exists():
            continue
        df = pd.read_csv(
            path, low_memory=False, keep_default_na=False, na_values=[""]
        )
        df["split"] = split
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No train_pairs.csv / val_pairs.csv / test_pairs.csv under {run_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def _attach_short_names(
    pairs: pd.DataFrame, function_to_short: dict[str, str]
) -> pd.DataFrame:
    """Add `function_short_a` and `function_short_b` columns by mapping
    `func_a` / `func_b` through the user-supplied dictionary.

    Any function not in the map is left as its full name (so the panel
    iteration still picks it up; the caller controls what's labeled).
    """
    pairs = pairs.copy()
    pairs["function_short_a"] = pairs["func_a"].map(
        lambda x: function_to_short.get(x, x)
    )
    pairs["function_short_b"] = pairs["func_b"].map(
        lambda x: function_to_short.get(x, x)
    )
    return pairs


def _stratified_sample_pairs(
    pairs: pd.DataFrame, max_per_label_per_split: int, random_state: int
) -> pd.DataFrame:
    """Sub-sample up to `max_per_label_per_split` rows per (split, label) cell."""
    out = []
    rng_seed = random_state
    for (_split, _label), grp in pairs.groupby(
        ["split", "label"], sort=False
    ):
        if len(grp) <= max_per_label_per_split:
            out.append(grp)
        else:
            out.append(
                grp.sample(n=max_per_label_per_split, random_state=rng_seed)
            )
            rng_seed += 1  # vary per cell so subsamples are independent
    return pd.concat(out, ignore_index=True)


def _lookup_kmer_rows(
    pairs: pd.DataFrame,
    side: str,
    key_to_row: dict,
    alphabet: str,
) -> tuple[np.ndarray, pd.Series]:
    """Resolve k-mer row indices for one side of a pair table.

    Returns `(rows, valid_mask)`. `rows` is length-N with -1 where the
    (assembly_id, occurrence_id) lookup failed; `valid_mask` is the
    corresponding boolean Series aligned with `pairs.index`.
    """
    asm = pairs[f"assembly_id_{side}"].astype(str)
    occ = _normalize_occurrence(pairs[_occ_col(alphabet, side)], alphabet)
    keys = list(zip(asm, occ))
    rows = np.array(
        [key_to_row.get(k, -1) for k in keys], dtype=np.int64
    )
    valid = pd.Series(rows >= 0, index=pairs.index)
    return rows, valid


# ============================================================================
# Sequence table
# ============================================================================

def _build_sequence_split_table(
    pairs: pd.DataFrame, key_to_row: dict, alphabet: str
) -> pd.DataFrame:
    """Per unique `prot_hash`: function_short, kmer_row, frozenset(splits), color_key.

    Sequences whose (assembly_id, occurrence_id) does not match any
    k-mer row are dropped. The `splits` field is the union of splits
    in which the sequence appears across all pairs.
    """
    occ_a, occ_b = _occ_col(alphabet, "a"), _occ_col(alphabet, "b")

    def _side(df: pd.DataFrame, side: str) -> pd.DataFrame:
        occ = occ_a if side == "a" else occ_b
        return df[
            [f"assembly_id_{side}", occ, f"prot_hash_{side}",
             f"function_short_{side}", "split"]
        ].rename(
            columns={
                f"assembly_id_{side}": "assembly_id",
                occ: "occ",
                f"prot_hash_{side}": "prot_hash",
                f"function_short_{side}": "function_short",
            }
        )

    long = pd.concat([_side(pairs, "a"), _side(pairs, "b")], ignore_index=True)
    long["assembly_id"] = long["assembly_id"].astype(str)
    long["occ"] = _normalize_occurrence(long["occ"], alphabet)

    keys = list(zip(long["assembly_id"], long["occ"]))
    long["kmer_row"] = [key_to_row.get(k, -1) for k in keys]
    long = long[long["kmer_row"] >= 0].copy()
    if long.empty:
        return long

    # Collapse to one row per prot_hash: take first function_short / kmer_row
    # (invariant for a given prot_hash by construction); union of splits.
    agg = long.groupby("prot_hash", sort=False).agg(
        function_short=("function_short", "first"),
        kmer_row=("kmer_row", "first"),
        splits=("split", lambda s: frozenset(s.unique())),
    ).reset_index()
    agg["color_key"] = agg["splits"].apply(_resolve_split_color_key)
    return agg


def _resolve_split_color_key(splits: frozenset) -> str:
    """Single-split → that split's name; multi-split → `LEAKAGE_KEY`."""
    if len(splits) == 1:
        return next(iter(splits))
    return LEAKAGE_KEY


def _stratified_sample_sequences(
    seq_table: pd.DataFrame,
    max_per_function: int,
    random_state: int,
) -> pd.DataFrame:
    """Sub-sample sequences per (function_short, color_key) cell.

    Caps each protein type at `max_per_function` sequences while keeping
    the per-color proportion roughly intact. The leakage category
    (`color_key == LEAKAGE_KEY`) gets a guaranteed minimum allocation so
    it survives sampling even when it's a small minority — this is what
    makes the random-routing leakage panel render at all.
    """
    out_chunks = []
    for fshort, panel in seq_table.groupby("function_short", sort=False):
        n_panel = len(panel)
        if n_panel <= max_per_function:
            out_chunks.append(panel)
            continue

        # Proportional allocation, but reserve at least
        # min(LEAKAGE_FLOOR, count(leakage)) for the leakage category so the
        # random-routing panel can render its signal even when it's rare.
        LEAKAGE_FLOOR = 500
        leakage_mask = panel["color_key"] == LEAKAGE_KEY
        n_leakage = int(leakage_mask.sum())
        leakage_target = min(n_leakage, LEAKAGE_FLOOR)

        non_leakage = panel.loc[~leakage_mask]
        non_leakage_budget = max(0, max_per_function - leakage_target)
        if len(non_leakage) > non_leakage_budget:
            non_leakage = non_leakage.sample(
                n=non_leakage_budget, random_state=random_state
            )
        leakage = panel.loc[leakage_mask]
        if len(leakage) > leakage_target:
            leakage = leakage.sample(
                n=leakage_target, random_state=random_state + 1
            )
        out_chunks.append(pd.concat([leakage, non_leakage], ignore_index=False))
    return pd.concat(out_chunks, ignore_index=True)


# ============================================================================
# Drawing — sequence panels
# ============================================================================

def _draw_sequence_panels(
    coords: np.ndarray,
    seq_table: pd.DataFrame,
    xlab: str,
    ylab: str,
    sup_title: str,
    output_path: Path,
) -> None:
    """One subplot per `function_short` in the table, sharing axes.

    Sequences are layered so the leakage color is drawn behind train /
    val / test for legibility.
    """
    function_shorts = list(
        seq_table["function_short"].drop_duplicates().tolist()
    )
    if not function_shorts:
        print(
            f"WARNING: _draw_sequence_panels: no function_short values in seq_table"
        )
        return
    n = len(function_shorts)

    color_for = {
        "train": SPLIT_COLORS["train"],
        "val": SPLIT_COLORS["val"],
        "test": SPLIT_COLORS["test"],
        LEAKAGE_KEY: LEAKAGE_COLOR,
    }
    has_leakage = bool((seq_table["color_key"] == LEAKAGE_KEY).any())
    order = [LEAKAGE_KEY, "train", "val", "test"]  # leakage drawn behind

    fig, axes = plt.subplots(
        1, n, figsize=(7.5 * n, 7), sharex=True, sharey=True, squeeze=False
    )
    axes = axes[0]
    for ax, fshort in zip(axes, function_shorts):
        sel_panel = (seq_table["function_short"] == fshort).values
        coords_panel = coords[sel_panel]
        table_panel = seq_table.loc[sel_panel].reset_index(drop=True)
        for key in order:
            sel = (table_panel["color_key"] == key).values
            if sel.sum() == 0:
                continue
            ax.scatter(
                coords_panel[sel, 0],
                coords_panel[sel, 1],
                s=10,
                alpha=0.55 if key == LEAKAGE_KEY else 0.8,
                c=color_for[key],
                edgecolors="none",
            )
        ax.set_title(
            f"{fshort} ({int(sel_panel.sum()):,} sequences)",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_xlabel(xlab, fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylab, fontsize=11)

    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=color_for["train"], markersize=8, label="Train"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=color_for["val"], markersize=8, label="Val"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=color_for["test"], markersize=8, label="Test"),
    ]
    if has_leakage:
        handles.append(
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=LEAKAGE_COLOR, markersize=8,
                label="In multiple splits",
            )
        )

    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(sup_title, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Drawing — pair panel
# ============================================================================

def _draw_pair_panel(
    coords: np.ndarray,
    pairs: pd.DataFrame,
    xlab: str,
    ylab: str,
    title: str,
    output_path: Path,
) -> None:
    """Single-panel scatter: color = split, fill/hollow = label."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    for split in SPLITS:
        for label in (1, 0):
            mask = (
                (pairs["split"] == split) & (pairs["label"] == label)
            ).values
            if mask.sum() == 0:
                continue
            style = LABEL_SCATTER_STYLES.get(int(label), {})
            face = (
                SPLIT_COLORS[split]
                if style.get("facecolors") == "auto"
                else style.get("facecolors", "none")
            )
            edge = (
                SPLIT_COLORS[split]
                if style.get("edgecolors") == "auto"
                else style.get("edgecolors", "none")
            )
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                marker=SPLIT_MARKERS[split],
                s=18,
                alpha=style.get("alpha", 0.7),
                facecolors=face,
                edgecolors=edge,
                linewidths=style.get("linewidths", 0.0),
            )

    split_handles = [
        Line2D(
            [0], [0], marker=SPLIT_MARKERS[s], color="none",
            markerfacecolor=SPLIT_COLORS[s], markeredgecolor="none",
            markersize=8, label=s.capitalize(),
        )
        for s in SPLITS
    ]
    label_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
               markeredgecolor="none", markersize=8, label="Positive"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="black", markeredgewidth=0.9,
               markersize=8, label="Negative"),
    ]
    leg1 = ax.legend(
        handles=split_handles, title="Split", loc="upper left",
        fontsize=9, frameon=True, framealpha=0.85,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=label_handles, title="Label", loc="upper right",
        fontsize=9, frameon=True, framealpha=0.85,
    )
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# Title helpers
# ============================================================================

def _sup_title(prefix: str, alphabet: str, k: int, run_dir: Path) -> str:
    return f"{prefix} — {alphabet} k={k}  ({Path(run_dir).name})"


def _pair_title(prefix: str, alphabet: str, k: int, run_dir: Path) -> str:
    return f"{prefix}({alphabet} k={k})  ({Path(run_dir).name})"
