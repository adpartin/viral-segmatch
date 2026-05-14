"""
Visualize dataset statistics for segment pair classification datasets.

Purpose:
    Generate distribution plots (host, HN subtype, year) and heatmaps (host × subtype)
    for train/val/test splits to visualize dataset composition and stratification.
    Optionally generate low-dimensional (UMAP/PCA) plots using the actual model inputs
    (concatenated embedding pairs) for split overlap sanity checks.

When to run:
    1. Automatically: Called from dataset_segment_pairs.py after dataset creation
    2. On-demand: Run standalone to regenerate plots for existing datasets

    Standalone usage:
        python src/analysis/visualize_dataset_stats.py --bundle flu_2024
        python src/analysis/visualize_dataset_stats.py --bundle flu_2024 flu_human_2024
        python src/analysis/visualize_dataset_stats.py --all

Data required:
    - dataset_stats.json file in the dataset run directory
    - Format: data/datasets/{virus}/{data_version}/runs/dataset_{bundle}_{timestamp}/

Output locations:
    - Dataset run directory: data/datasets/{virus}/{data_version}/runs/dataset_{bundle}_{timestamp}/plots/

Conditional plotting:
    - Host distribution: skipped if only 1 unique value
    - HN subtype distribution: skipped if only 1 unique value
    - Year distribution: skipped if only 1 year or very narrow range
    - Host × HN subtype heatmap: skipped if either dimension has ≤1 unique value

Generated plots:
    - host_distribution.png: Bar plot of host distribution (3 subplots: train/val/test)
    - subtype_distribution.png: Bar plot of HN subtype distribution (3 subplots: train/val/test)
    - year_distribution.png: Histogram of year distribution (3 subplots: train/val/test)
    - host_subtype_heatmap.png: Heatmap of host × HN subtype (3 subplots: train/val/test)
    - geo_location_distribution.png: Bar plot of geo_location_clean distribution (3 subplots)
    - passage_distribution.png: Bar plot of passage distribution (3 subplots)
    - pair_pca_concat.png, pair_pca_diff.png, pair_pca_prod.png, pair_pca_unit_diff.png:
      PCA scatter plots of pair embeddings under each interaction mode, colored/marked
      by split (train/val/test) and styled by label (positive/negative).
    - pair_interaction_diagnostics.json: per-interaction norm-vs-label correlations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config_hydra import get_virus_config_hydra, get_function_short_name_map
from src.utils.dim_reduction_utils import compute_pca_reduction
from src.utils.plot_config import (
    SPLIT_COLORS, SPLIT_MARKERS, LABEL_SCATTER_STYLES, apply_default_style,
)
from src.utils.embedding_utils import (
    create_pair_embeddings_concatenation,
    extract_unique_sequences_from_pairs,
    load_embedding_index,
    load_embeddings_by_ids,
    plot_embeddings_by_category,
    sample_pairs_stratified,
)
from src.utils.kmer_utils import load_kmer_index, load_kmer_matrix


def _collapse_to_top_n(series: pd.Series, top_n: int, other_label: str = 'Other') -> pd.Series:
    """Collapse a categorical series to its top-N values; everything else becomes `other_label`."""
    s = series.copy()
    s = s.replace({np.nan: 'null'})
    counts = s.value_counts(dropna=False)
    keep = set(counts.head(top_n).index.tolist())
    return s.apply(lambda x: x if x in keep else other_label)


def _load_pairs_minimal(run_dir: Path, split: str) -> Optional[pd.DataFrame]:
    """Load the necessary columns for embedding visualizations."""
    csv_path = run_dir / f"{split}_pairs.csv"
    if not csv_path.exists():
        return None
    desired = [
        'brc_a', 'brc_b', 'label',
        'seg_a', 'seg_b', 'func_a', 'func_b',
        'pair_key',
    ]
    try:
        # Prefer minimal IO: read header, then load only desired columns that exist.
        header_cols = list(pd.read_csv(csv_path, nrows=0).columns)
        usecols = [c for c in desired if c in header_cols]
        # Ensure we at least have the columns required for embedding construction.
        required = {'brc_a', 'brc_b', 'label'}
        if not required.issubset(set(usecols)):
            return pd.read_csv(csv_path, low_memory=False)
        return pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except Exception:
        # Fallback if file schema differs or any parsing issue occurs.
        return pd.read_csv(csv_path, low_memory=False)


_CORNER_TO_AXES_XY = {
    'upper left':  (0.02, 0.98, 'left',  'top'),
    'upper right': (0.98, 0.98, 'right', 'top'),
    'lower left':  (0.02, 0.02, 'left',  'bottom'),
    'lower right': (0.98, 0.02, 'right', 'bottom'),
}


def _pick_sparse_corner(points_2d: np.ndarray, used: set[str], region_frac: float = 0.25) -> str:
    """Pick a plot corner with few points to minimize legend/textbox overlap.

    Uses a quadrant density heuristic on coordinates normalized to [0, 1].
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    xr = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
    yr = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-12)

    corners = {
        'upper left':  (xr <= region_frac) & (yr >= 1 - region_frac),
        'upper right': (xr >= 1 - region_frac) & (yr >= 1 - region_frac),
        'lower left':  (xr <= region_frac) & (yr <= region_frac),
        'lower right': (xr >= 1 - region_frac) & (yr <= region_frac),
    }
    scores = []
    for name, mask in corners.items():
        if name in used:
            continue
        scores.append((int(np.sum(mask)), name))
    if not scores:
        return 'upper right'
    scores.sort(key=lambda t: t[0])
    return scores[0][1]


def _draw_corner_textbox(ax, points_2d: np.ndarray, used_locs: set[str], text: str) -> None:
    """Place a text box in a sparse corner of the axes (mutates `used_locs`)."""
    if not text:
        return
    loc = _pick_sparse_corner(points_2d, used_locs)
    used_locs.add(loc)
    x0, y0, ha, va = _CORNER_TO_AXES_XY.get(loc, (0.98, 0.98, 'right', 'top'))
    ax.text(
        x0, y0, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
    )


def _plot_pair_features_splits_2d(
    reduced_2d: np.ndarray,
    pairs: pd.DataFrame,
    splits: list[str],
    split_colors: dict,
    split_markers: dict,
    xlab: str,
    ylab: str,
    title: str,
    output_path: Path,
    filters_text: str = "",
    sample_text: str = "",
    ) -> None:
    """Scatter plot of 2D pair feature vectors, colored by split, styled by label.

    The "feature vector" can be ESM-2 pair embeddings (concat/diff/prod/unit_diff)
    or k-mer pair features — anything reducible to 2D. Color encodes split
    (train/val/test); marker fill encodes label (positive filled, negative hollow).
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    # Plot per (split × label) so we can keep split color semantics and still indicate pos/neg.
    for split in splits:
        for label in [1, 0]:
            mask = (pairs['split'] == split) & (pairs['label'] == label)
            if mask.sum() == 0:
                continue

            style = LABEL_SCATTER_STYLES.get(int(label), {})
            facecolors = style.get('facecolors', 'auto')
            edgecolors = style.get('edgecolors', 'none')
            linewidths = style.get('linewidths', 0.0)
            alpha = style.get('alpha', 0.65)

            # Resolve 'auto' to split color.
            if facecolors == 'auto':
                facecolors = split_colors[split]
            if edgecolors == 'auto':
                edgecolors = split_colors[split]

            ax.scatter(
                reduced_2d[mask.values, 0],
                reduced_2d[mask.values, 1],
                marker=split_markers[split],
                s=18,
                alpha=alpha,
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
            )

    # Legend: split markers/colors + label fill style.
    from matplotlib.lines import Line2D

    split_handles = [
        Line2D(
            [0], [0],
            marker=split_markers[s],
            color='none',
            markerfacecolor=split_colors[s],
            markeredgecolor='none',
            markersize=8,
            label=s.capitalize(),
        )
        for s in splits
    ]
    used_locs: set[str] = set()

    # Place Split legend inside the axes, in a sparse corner.
    split_loc = _pick_sparse_corner(reduced_2d, used_locs)
    used_locs.add(split_loc)
    leg1 = ax.legend(
        handles=split_handles,
        title='Split',
        loc=split_loc,
        fontsize=10,
        frameon=True,
        framealpha=0.85,
    )
    ax.add_artist(leg1)

    pos_style = LABEL_SCATTER_STYLES[1]
    neg_style = LABEL_SCATTER_STYLES[0]
    label_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='black' if pos_style.get('facecolors') == 'auto' else pos_style.get('facecolors', 'black'),
            markeredgecolor='none',
            markersize=8,
            label=pos_style.get('label', 'Positive (+)'),
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=neg_style.get('linewidths', 0.9),
            markersize=8,
            label=neg_style.get('label', 'Negative (-)'),
        ),
    ]
    # Place Label legend in a different sparse corner.
    label_loc = _pick_sparse_corner(reduced_2d, used_locs)
    used_locs.add(label_loc)
    ax.legend(
        handles=label_handles,
        title='Label',
        loc=label_loc,
        fontsize=10,
        frameon=True,
        framealpha=0.85,
    )
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)

    # Add dataset filter context in two places:
    # - A 2nd row in the title (good for quick scan in reports)
    # - A small textbox inside the axes (better for screenshots where titles can be clipped)
    #
    # NOTE/TODO: Today filters are mostly single-value exact matches. If we later support
    # multi-valued filters or ranges (e.g., year=[2010, 2019]), keep the plotting text
    # compact (summarize as "host: 7 values", "year: 2010–2019") rather than dumping long lists.
    # Title: keep clean (no 2nd-row filter text). Filters go into an in-axes textbox only.
    ax.set_title(title, fontweight='bold', fontsize=14)
    if filters_text:
        _draw_corner_textbox(ax, reduced_2d, used_locs, f"Filters:\n{filters_text}")
    if sample_text:
        _draw_corner_textbox(ax, reduced_2d, used_locs, sample_text)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def _derive_output_path(output_path: Path, suffix_tag: str) -> Path:
    """Derive a sibling output path by swapping *_umap.png → *_{suffix_tag}.png, or appending."""
    name = output_path.name
    if name.endswith('_umap.png'):
        name = name.replace('_umap.png', f'_{suffix_tag}.png')
    elif name.endswith('.png'):
        name = name.replace('.png', f'_{suffix_tag}.png')
    else:
        name = f"{name}_{suffix_tag}.png"
    return output_path.with_name(name)


def _load_filters_applied_from_run(run_dir: Path) -> dict:
    """Load filters_applied from run_dir/dataset_stats.json if present."""
    stats_path = run_dir / "dataset_stats.json"
    if not stats_path.exists():
        return {}
    try:
        with open(stats_path, "r") as f:
            d = json.load(f)
        return d.get("filters_applied", {}) or {}
    except Exception:
        return {}


def _format_filters_for_plot(
    filters_applied: Optional[dict],
    short_name_map: Optional[dict] = None,
    ) -> str:
    """Format dataset filters into a compact string for plot title/textbox.

    `short_name_map` (e.g., the `function_short_names` block from the virus
    config) is applied element-wise to any string filter value — so a filter
    like `schema_pair=['Hemagglutinin precursor', 'Neuraminidase protein']`
    renders as `schema_pair=['HA', 'NA']`.
    """
    if not filters_applied:
        return ""

    short_name_map = short_name_map or {}

    def _shorten(x):
        if isinstance(x, str):
            return short_name_map.get(x, x)
        return x

    parts: list[str] = []
    for k, v in filters_applied.items():
        if v is None:
            continue

        # Best-effort compact formatting for near-future multi-valued/range filters.
        if isinstance(v, (list, tuple, set)):
            vv = [_shorten(x) for x in list(v)]
            if len(vv) == 0:
                continue
            if len(vv) <= 4:
                parts.append(f"{k}={vv}")
            else:
                parts.append(f"{k}={len(vv)} values")
            continue
        if isinstance(v, dict):
            if 'min' in v and 'max' in v:
                parts.append(f"{k}={v['min']}–{v['max']}")
            else:
                parts.append(f"{k}={{...}}")
            continue

        parts.append(f"{k}={_shorten(v)}")

    return "; ".join(parts)


def _compute_interaction_features(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    interaction_spec: str,
) -> np.ndarray:
    """Compute interaction features from raw embedding pairs.

    Args:
        emb_a: (N, D) array of slot-A embeddings.
        emb_b: (N, D) array of slot-B embeddings.
        interaction_spec: One of "concat", "diff", "prod", "unit_diff",
            or combinations like "concat+diff", "diff+unit_diff", etc.

    Returns:
        (N, K) feature array.
    """
    tokens = [t.strip().lower() for t in interaction_spec.split('+')]
    features: list[np.ndarray] = []
    for t in tokens:
        if t == "concat":
            features.extend([emb_a, emb_b])
        elif t == "diff":
            features.append(np.abs(emb_a - emb_b))
        elif t == "unit_diff":
            diff = emb_a - emb_b
            norms = np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-8)
            features.append(diff / norms)
        elif t == "prod":
            features.append(emb_a * emb_b)
        else:
            raise ValueError(f"Unknown interaction token: {t!r}")
    if not features:
        raise ValueError(f"No valid interaction tokens in: {interaction_spec!r}")
    return np.concatenate(features, axis=1)


def _format_sample_counts(pairs: pd.DataFrame, per_split_totals: dict[str, int]) -> str:
    """Compact 'sampled vs available' description for in-axes textbox.

    `pairs` is the post-sampling concatenated frame with a `split` column;
    `per_split_totals` is the pre-sampling row count from each split CSV.
    """
    if 'split' not in pairs.columns or not per_split_totals:
        return ""
    sampled_counts = pairs.groupby('split').size().to_dict()
    total_sampled = int(sum(sampled_counts.values()))
    total_available = int(sum(per_split_totals.values()))
    lines = [f"Sampled: {total_sampled:,} / {total_available:,} pairs"]
    for split in ('train', 'val', 'test'):
        if split in per_split_totals:
            n_s = int(sampled_counts.get(split, 0))
            n_t = int(per_split_totals[split])
            lines.append(f"  {split}: {n_s:,} / {n_t:,}")
    return "\n".join(lines)


def _resolve_kmer_k(kmer_dir: Path, k: Optional[int] = None) -> int:
    """Resolve which k to use for k-mer PCA plots.

    Priority: explicit `k` argument > `kmer_features_k{k}_metadata.json` > filename glob.
    Errors if multiple k's are present and none was specified.
    """
    if k is not None:
        return int(k)
    metadata_files = sorted(kmer_dir.glob('kmer_features_k*_metadata.json'))
    if metadata_files:
        if len(metadata_files) > 1:
            ks = sorted({_extract_k_from_filename(p.name) for p in metadata_files} - {None})
            raise ValueError(
                f"Multiple k-mer caches in {kmer_dir} (k in {ks}); pass k= explicitly."
            )
        with open(metadata_files[0]) as f:
            return int(json.load(f)['k'])
    npz_files = sorted(kmer_dir.glob('kmer_features_k*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No k-mer cache found under {kmer_dir}")
    if len(npz_files) > 1:
        ks = sorted({_extract_k_from_filename(p.name) for p in npz_files} - {None})
        raise ValueError(
            f"Multiple k-mer caches in {kmer_dir} (k in {ks}); pass k= explicitly."
        )
    parsed = _extract_k_from_filename(npz_files[0].name)
    if parsed is None:
        raise ValueError(f"Could not parse k from {npz_files[0].name}")
    return parsed


def _extract_k_from_filename(name: str) -> Optional[int]:
    """Parse k from `kmer_features_k{k}{suffix}` filenames."""
    import re
    m = re.search(r'kmer_features_k(\d+)', name)
    return int(m.group(1)) if m else None


def _interaction_spec_to_filename(interaction_spec: str) -> str:
    """Convert interaction spec to a safe filename component.

    Examples: "concat" -> "concat", "concat+diff" -> "concat_diff",
    "unit_diff" -> "unit_diff".
    """
    return interaction_spec.replace('+', '_').replace(' ', '').lower()


def plot_pair_interactions(
    run_dir: Path,
    embeddings_file: Path,
    output_dir: Path,
    interactions: Optional[list] = None,
    max_per_label_per_split: int = 1000,
    random_state: int = 42,
    function_short_names: Optional[dict] = None,
    ) -> None:
    """Generate PCA scatter plots of pair embeddings for multiple interaction modes.

    For each interaction mode (e.g., concat, diff, prod, unit_diff), computes
    the corresponding feature representation from raw (emb_a, emb_b) pairs,
    projects to 2D via PCA, and plots colored by split (train/val/test) and
    styled by label (positive/negative).

    Raw embeddings are loaded ONCE and all interactions are computed from them,
    making this efficient for comparing multiple modes side by side.

    NOTE: This function visualizes data-level interactions only (computed from
    raw frozen embeddings).  Architecture-dependent transformations (e.g.,
    slot_norm, slot_transform) require trained model weights and are NOT visualized
    here -- those would be a separate post-training analysis.

    Args:
        run_dir: Dataset run directory containing {split}_pairs.csv files.
        embeddings_file: Path to master HDF5 embedding file.
        output_dir: Directory to save output plots and diagnostics JSON.
        interactions: List of interaction specs to visualize.  Each spec can be
            a single mode ("concat", "diff", "prod", "unit_diff") or a
            combination ("concat+diff", "diff+prod").
            Defaults to ["concat", "diff", "prod", "unit_diff"].
        max_per_label_per_split: Max pairs to sample per label per split.
        random_state: Random seed for reproducible sampling and PCA.

    Output files:
        - pair_pca_{spec}.png for each interaction spec
        - pair_interaction_diagnostics.json with per-interaction metrics
    """
    if interactions is None:
        interactions = ["concat", "diff", "prod", "unit_diff"]

    splits = ['train', 'val', 'test']
    split_colors = SPLIT_COLORS
    split_markers = SPLIT_MARKERS

    filters_applied = _load_filters_applied_from_run(run_dir)
    filters_text = _format_filters_for_plot(filters_applied, function_short_names)

    # ── 1. Load and sample pairs (once for all interactions) ──────────────
    sampled = []
    per_split_totals: dict[str, int] = {}
    for split in splits:
        df = _load_pairs_minimal(run_dir, split)  # load only necessary columns for embedding visualization
        if df is None or len(df) == 0:
            continue
        if 'label' in df.columns:
            df = df.copy()
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df[df['label'].isin([0, 1])]
        per_split_totals[split] = int(len(df))
        df_s = sample_pairs_stratified(
            df,
            max_per_label=max_per_label_per_split,
            label_col='label',
            random_state=random_state,
        )
        df_s = df_s.assign(split=split)
        sampled.append(df_s)

    if not sampled:
        print(f"WARNING: plot_pair_interactions: no pair CSVs found in {run_dir}")
        return

    pairs = pd.concat(sampled, ignore_index=True)
    sample_text = _format_sample_counts(pairs, per_split_totals)

    # ── 2. Load raw emb_a, emb_b (once, via concat then split) ───────────
    id_to_row = load_embedding_index(embeddings_file)
    concat_embs, labels, valid_mask = create_pair_embeddings_concatenation(
        pairs,
        embeddings_file,
        id_to_row=id_to_row,
        return_valid_mask=True,
        dtype=np.float32,
        use_concat=True,
        use_diff=False,
        use_prod=False,
        use_unit_diff=False,
    )
    if len(concat_embs) == 0:
        print("WARNING: plot_pair_interactions: no valid pair embeddings (missing IDs?)")
        return

    # Align pairs to valid mask
    n_in = len(pairs)
    pairs = pairs.loc[valid_mask].reset_index(drop=True)
    dropped = n_in - len(pairs)
    if dropped > 0:
        print(f"plot_pair_interactions: dropped {dropped:,}/{n_in:,} sampled pairs (missing embeddings)")
    assert len(pairs) == len(concat_embs), "pairs/embeddings misalignment"

    D = concat_embs.shape[1] // 2
    emb_a = concat_embs[:, :D]
    emb_b = concat_embs[:, D:]
    label_arr = pairs['label'].to_numpy().astype(float)

    # Optional: add seg_pair / func_pair columns for later use
    if 'seg_a' in pairs.columns and 'seg_b' in pairs.columns:
        pairs['seg_pair'] = pairs['seg_a'].astype(str).fillna('null') + '→' + pairs['seg_b'].astype(str).fillna('null')
    if 'func_a' in pairs.columns and 'func_b' in pairs.columns:
        pairs['func_pair'] = pairs['func_a'].astype(str).fillna('null') + '→' + pairs['func_b'].astype(str).fillna('null')

    # ── 3. Per-interaction: compute features → PCA → plot + diagnostics ───
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_diagnostics: dict = {
        'n_pairs': int(len(pairs)),
        'embedding_dim': int(D),
        'filters_applied': filters_applied,
        'interactions': {},
    }

    def _corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    for spec in interactions:
        safe_name = _interaction_spec_to_filename(spec)
        print(f"\n{'─'*60}")
        print(f"  Interaction: {spec}")
        print(f"{'─'*60}")

        try:
            features = _compute_interaction_features(emb_a, emb_b, spec)
        except ValueError as e:
            print(f"WARNING: skipping interaction '{spec}': {e}")
            continue

        # PCA → 2D
        pca_reduced, pca_model = compute_pca_reduction(
            features,
            n_components=2,
            return_model=True,
            svd_solver="randomized",
            random_state=random_state,
        )
        pca_2d = pca_reduced[:, :2]
        ev1 = pca_model.explained_variance_ratio_[0] if pca_model else 0
        ev2 = pca_model.explained_variance_ratio_[1] if pca_model else 0
        pc1_label = f"PC1 ({ev1:.1%} var)"
        pc2_label = f"PC2 ({ev2:.1%} var)"

        # Attach PCA coords temporarily for diagnostics
        pairs_tmp = pairs.copy()
        pairs_tmp['pc1'] = pca_2d[:, 0]
        pairs_tmp['pc2'] = pca_2d[:, 1]

        # ── Plot ──
        out_path = output_dir / f"pair_pca_{safe_name}.png"
        _plot_pair_features_splits_2d(
            reduced_2d=pca_2d,
            pairs=pairs_tmp,
            splits=splits,
            split_colors=split_colors,
            split_markers=split_markers,
            xlab=pc1_label,
            ylab=pc2_label,
            title=f"PCA: Pair Embeddings (features={spec})",
            output_path=out_path,
            filters_text=filters_text,
            sample_text=sample_text,
        )

        # ── Diagnostics ──
        feature_norm = np.linalg.norm(features, axis=1)
        corr_norm_label = _corr(feature_norm, label_arr)
        corr_norm_pc1 = _corr(feature_norm, pca_2d[:, 0])

        pos_mask = label_arr == 1
        neg_mask = label_arr == 0
        mean_norm_pos = float(np.mean(feature_norm[pos_mask])) if pos_mask.sum() > 0 else None
        mean_norm_neg = float(np.mean(feature_norm[neg_mask])) if neg_mask.sum() > 0 else None

        print(f"  PCA explained variance: PC1={ev1:.3f}, PC2={ev2:.3f}")
        print(f"  corr(||feature||, label) = {corr_norm_label:.4f}" if corr_norm_label is not None else "  corr(||feature||, label) = N/A")
        print(f"  corr(||feature||, PC1)   = {corr_norm_pc1:.4f}" if corr_norm_pc1 is not None else "  corr(||feature||, PC1)   = N/A")
        print(f"  mean ||feature|| positives = {mean_norm_pos:.4f}" if mean_norm_pos is not None else "  mean ||feature|| positives = N/A")
        print(f"  mean ||feature|| negatives = {mean_norm_neg:.4f}" if mean_norm_neg is not None else "  mean ||feature|| negatives = N/A")

        diag_entry: dict = {
            'feature_dim': int(features.shape[1]),
            'pca_explained_variance_top2': [float(ev1), float(ev2)],
            'corr_feature_norm_label': corr_norm_label,
            'corr_feature_norm_pc1': corr_norm_pc1,
            'mean_feature_norm_pos': mean_norm_pos,
            'mean_feature_norm_neg': mean_norm_neg,
        }

        # Seg-pair / func-pair summaries (if available)
        for cat_col in ('seg_pair', 'func_pair'):
            if cat_col in pairs_tmp.columns:
                grp = pairs_tmp.groupby(cat_col)[['pc1', 'pc2']].agg(['count', 'mean']).sort_values(('pc1', 'count'), ascending=False)
                top_cats = grp.head(5).index.tolist()
                diag_entry[f'{cat_col}_top5'] = top_cats

        all_diagnostics['interactions'][spec] = diag_entry

    # ── 4. Save diagnostics JSON ─────────────────────────────────────────
    try:
        diag_path = output_dir / "pair_interaction_diagnostics.json"
        with open(diag_path, 'w') as f:
            json.dump(all_diagnostics, f, indent=2)
        print(f"\nSaved interaction diagnostics: {diag_path}")
    except Exception as e:
        print(f"WARNING: failed to save diagnostics JSON: {e}")


# =============================================================================
# K-mer PCA plots
# =============================================================================

def _plot_kmer_scree(
    explained_variance_per_panel: dict[str, np.ndarray],
    output_path: Path,
    n_components: int = 12,
    ) -> None:
    """Multi-panel scree plot for k-mer PCA fits.

    Shows the top-N explained variance ratios for each panel. One subplot per
    fit; bars for individual PC variance, red line on the secondary axis for
    the running cumulative.
    """
    panels = list(explained_variance_per_panel.items())
    n = len(panels)
    if n == 0:
        return
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)

    for idx, (label, evr) in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        evr = np.asarray(evr, dtype=float)
        k = min(n_components, len(evr))
        x = np.arange(1, k + 1)
        bars = ax.bar(x, evr[:k], color='#4a90d9', edgecolor='#1f4c82', alpha=0.85)
        # Cumulative line on a secondary axis.
        cum = np.cumsum(evr[:k])
        ax2 = ax.twinx()
        ax2.plot(x, cum, color='#c0392b', marker='o', linewidth=1.5,
                 markersize=4, label='cumulative')
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel('Cumulative explained variance ratio', color='#c0392b', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='#c0392b')
        for bar, v in zip(bars, evr[:k]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.1%}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xlabel('Principal component', fontsize=10)
        ax.set_ylabel('Explained variance ratio (per PC)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(0.05, evr[:k].max() * 1.25))
        ax.grid(True, axis='y', alpha=0.3)

    # Hide any leftover empty subplots.
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    fig.suptitle('K-mer PCA: top-N explained variance ratio', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def _build_kmer_lookup_keys(assembly_ids: pd.Series, ctgs: pd.Series) -> pd.Series:
    """Build composite lookup keys matching `load_kmer_index` output.

    Mirrors the key formation in `kmer_utils.get_kmer_pair_features`:
    `f"{assembly_id}::{ctg}"` with both columns coerced to str.
    """
    return assembly_ids.astype(str) + '::' + ctgs.astype(str)


def _densify_kmer_rows(kmer_matrix, row_indices: np.ndarray) -> np.ndarray:
    """Densify a slice of the sparse k-mer CSR matrix to float32."""
    return np.asarray(kmer_matrix[row_indices].todense(), dtype=np.float32)


def plot_kmer_pca(
    run_dir: Path,
    kmer_dir: Path,
    output_dir: Path,
    virus_config,
    k: Optional[int] = None,
    max_per_label_per_split: int = 1000,
    random_state: int = 42,
    ) -> None:
    """Generate the pair-concatenated k-mer PCA plot and its companion scree.

    Produces two files:
      - `kmer_pca_concat.png` — one point per sample pair; vector is the
        concatenation of k-mer vectors for (assembly_id_a, ctg_a) and
        (assembly_id_b, ctg_b). Color = split, fill = label. Mirrors
        `pair_pca_concat.png`.
      - `kmer_pca_scree.png` — top-12 per-PC explained variance ratio (bars)
        + the cumulative line for those 12 components on a secondary axis.

    Callable both from `visualize_dataset_stats` and standalone after-the-fact.

    Args:
        run_dir: Dataset run directory containing {split}_pairs.csv files.
        kmer_dir: Directory containing kmer_features_k{k}.npz +
            kmer_features_k{k}_index.parquet (typically lives next to the
            ESM-2 master HDF5 cache).
        output_dir: Directory to save output PNGs.
        virus_config: `cfg.virus` Hydra sub-node; used to resolve protein
            short names for the filter-text caption.
        k: k-mer size. If None, autodetect from kmer_dir.
        max_per_label_per_split: Max pairs sampled per label per split for
            the scatter plot.
        random_state: Reproducible sampling/PCA seed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k = _resolve_kmer_k(kmer_dir, k)
    print(f"\nplot_kmer_pca: using k={k}, kmer_dir={kmer_dir}")

    kmer_matrix = load_kmer_matrix(kmer_dir, k)
    key_to_row = load_kmer_index(kmer_dir, k)

    func_short_names = get_function_short_name_map_from_config(virus_config)

    splits = ['train', 'val', 'test']
    filters_applied = _load_filters_applied_from_run(run_dir)
    filters_text = _format_filters_for_plot(filters_applied, func_short_names)

    # Per-panel explained-variance ratios are accumulated for the scree plot.
    scree_evr: dict[str, np.ndarray] = {}

    # ── Plot 1: concatenated k-mer pair PCA ──────────────────────────────
    evr = _plot_kmer_pair_concat(
        run_dir=run_dir,
        splits=splits,
        kmer_matrix=kmer_matrix,
        key_to_row=key_to_row,
        k=k,
        output_path=output_dir / 'kmer_pca_concat.png',
        max_per_label_per_split=max_per_label_per_split,
        random_state=random_state,
        filters_text=filters_text,
    )
    if evr is not None:
        scree_evr['Pair concat (all splits)'] = evr

    # ── Scree summary (top-12 per-PC variance ratio + cumulative line) ─
    if scree_evr:
        _plot_kmer_scree(
            explained_variance_per_panel=scree_evr,
            output_path=output_dir / 'kmer_pca_scree.png',
            n_components=12,
        )


def get_function_short_name_map_from_config(virus_config) -> dict[str, str]:
    """Bridge to the Hydra-aware helper without requiring the wrapping cfg.

    `get_function_short_name_map(cfg)` reads `cfg.virus.function_short_names`.
    Plot callers tend to already hold a `cfg.virus` sub-node, so accept it
    directly here and adapt.
    """
    raw = getattr(virus_config, 'function_short_names', None) if virus_config is not None else None
    if raw is None:
        return {}
    return {str(k): str(v) for k, v in dict(raw).items()}


def _plot_kmer_pair_concat(
    run_dir: Path,
    splits: list[str],
    kmer_matrix,
    key_to_row: dict,
    k: int,
    output_path: Path,
    max_per_label_per_split: int,
    random_state: int,
    filters_text: str,
    ) -> Optional[np.ndarray]:
    """Build and plot the concatenated-pair PCA across all splits (Plot 1).

    Returns the PCA explained_variance_ratio_ for use by the scree plot, or
    None if no plot was produced.
    """
    sampled = []
    per_split_totals: dict[str, int] = {}
    for split in splits:
        df = _load_pairs_minimal(run_dir, split)
        if df is None or len(df) == 0:
            continue
        # `ctg_a`/`ctg_b` are needed for k-mer key lookup but aren't in the
        # minimal-load column set, so fall back to a fresh read when missing.
        if 'ctg_a' not in df.columns or 'ctg_b' not in df.columns:
            df = pd.read_csv(run_dir / f'{split}_pairs.csv', low_memory=False)
        if 'label' in df.columns:
            df = df.copy()
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df[df['label'].isin([0, 1])]
        per_split_totals[split] = int(len(df))
        df_s = sample_pairs_stratified(
            df,
            max_per_label=max_per_label_per_split,
            label_col='label',
            random_state=random_state,
        ).assign(split=split)
        sampled.append(df_s)

    if not sampled:
        print(f"WARNING: plot_kmer_pca: no pair CSVs found in {run_dir}")
        return None
    pairs = pd.concat(sampled, ignore_index=True)

    # Build k-mer pair-concat features (drop pairs missing either side).
    keys_a = _build_kmer_lookup_keys(pairs['assembly_id_a'], pairs['ctg_a'])
    keys_b = _build_kmer_lookup_keys(pairs['assembly_id_b'], pairs['ctg_b'])
    rows_a = keys_a.map(key_to_row)
    rows_b = keys_b.map(key_to_row)
    valid = rows_a.notna() & rows_b.notna()
    n_in = len(pairs)
    n_valid = int(valid.sum())
    if n_valid == 0:
        print("WARNING: plot_kmer_pca: no valid pairs (no matching k-mer rows)")
        return None
    if n_in - n_valid > 0:
        print(f"plot_kmer_pca: dropped {n_in - n_valid:,}/{n_in:,} sampled pairs (missing k-mer rows)")
    pairs = pairs.loc[valid].reset_index(drop=True)
    rows_a = rows_a[valid].astype(int).to_numpy()
    rows_b = rows_b[valid].astype(int).to_numpy()

    emb_a = _densify_kmer_rows(kmer_matrix, rows_a)
    emb_b = _densify_kmer_rows(kmer_matrix, rows_b)
    features = np.concatenate([emb_a, emb_b], axis=1)

    # Request more PCs than the 2D scatter needs so the scree has enough data
    # for its top-12 bars + cumulative line; randomized SVD requires
    # n_components < min(M, N), so we cap defensively.
    n_components = min(12, features.shape[0] - 1, features.shape[1])
    pca_reduced, pca_model = compute_pca_reduction(
        features, n_components=n_components, return_model=True,
        svd_solver='randomized', random_state=random_state,
    )
    pca_2d = pca_reduced[:, :2]
    ev1 = pca_model.explained_variance_ratio_[0] if pca_model else 0
    ev2 = pca_model.explained_variance_ratio_[1] if pca_model else 0

    sample_text = _format_sample_counts(pairs, per_split_totals)

    _plot_pair_features_splits_2d(
        reduced_2d=pca_2d,
        pairs=pairs,
        splits=splits,
        split_colors=SPLIT_COLORS,
        split_markers=SPLIT_MARKERS,
        xlab=f"PC1 ({ev1:.1%} var)",
        ylab=f"PC2 ({ev2:.1%} var)",
        title=f"PCA: Pair K-mer Features (concat, k={k})",
        output_path=output_path,
        filters_text=filters_text,
        sample_text=sample_text,
    )
    return pca_model.explained_variance_ratio_ if pca_model is not None else None


def plot_pair_embeddings_splits_overlap(
    run_dir: Path,
    embeddings_file: Path,
    output_path: Path,
    max_per_label_per_split: int = 1000,
    random_state: int = 42,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    pre_pca_dim: Optional[int] = None,
    use_concat: bool = True,
    use_diff: bool = False,
    use_prod: bool = False,
    use_unit_diff: bool = False,
    ) -> None:
    """DEPRECATED: Use plot_pair_interactions() instead.

    This function is retained for backward compatibility with legacy bundles
    that use input_mode='interaction' and use_concat/use_diff/use_prod flags.
    New code should call plot_pair_interactions(), which generates PCA plots
    for multiple interaction modes in a single pass from raw embeddings.

    Plot sampled *pair embeddings* colored/marked by split.

    This uses the same feature construction as the model can be trained with:
    - concat:    [emb_a, emb_b]
    - diff:      |emb_a - emb_b|
    - unit_diff: (emb_a - emb_b) / ||emb_a - emb_b||  (direction only)
    - prod:      emb_a * emb_b

    Behavior:
    - Always computes and saves a PCA(2D) plot (fast, deterministic).
    - Additionally computes and saves a UMAP plot when UMAP is installed and succeeds.
      If `pre_pca_dim` is provided, UMAP runs on PCA-reduced embeddings for speed.
    """
    splits = ['train', 'val', 'test']
    split_markers = SPLIT_MARKERS  # consistent project-wide markers
    split_colors = SPLIT_COLORS    # consistent project-wide colors

    filters_applied = _load_filters_applied_from_run(run_dir)
    filters_text = _format_filters_for_plot(filters_applied)

    # Load and sample pairs per split (stratified by label for representativeness)
    sampled = []
    for split in splits:
        df = _load_pairs_minimal(run_dir, split)
        if df is None or len(df) == 0:
            continue

        # Ensure labels behave like {0,1} for stratified sampling (robust to
        # CSV schema/dtype drift).
        if 'label' in df.columns:
            df = df.copy()
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df[df['label'].isin([0, 1])]

        df_s = sample_pairs_stratified(df,
            max_per_label=max_per_label_per_split,
            label_col='label',
            random_state=random_state,
        )
        # Keep any optional metadata columns (seg_*/func_*/pair_key) for downstream diagnostics/plots.
        df_s = df_s.assign(split=split)
        sampled.append(df_s)

    if not sampled:
        print(f"WARNING: Skipping split-overlap plot: no pair CSVs found in {run_dir}")
        return

    pairs = pd.concat(sampled, ignore_index=True)

    if not (use_concat or use_diff or use_prod or use_unit_diff):
        raise ValueError("At least one of use_concat/use_diff/use_prod/use_unit_diff must be True.")

    # Create pair embeddings (match training feature construction)
    id_to_row = load_embedding_index(embeddings_file)
    pair_embeddings, labels, valid_mask = create_pair_embeddings_concatenation(
        pairs,
        embeddings_file,
        id_to_row=id_to_row,
        return_valid_mask=True,
        dtype=np.float32,
        use_concat=use_concat,
        use_diff=use_diff,
        use_prod=use_prod,
        use_unit_diff=use_unit_diff,
    )
    if len(pair_embeddings) == 0:
        print("WARNING: Skipping split-overlap plot: could not create any pair embeddings (missing embeddings?)")
        return

    # Align `pairs` to the returned embedding rows (critical when some IDs are
    # missing in embedding index)
    n_in = len(pairs)
    pairs = pairs.loc[valid_mask].reset_index(drop=True)
    dropped = n_in - len(pairs)
    if dropped > 0:
        print(f"Split-overlap plot: dropped {dropped:,}/{n_in:,} sampled pairs due to missing embeddings")
    assert len(pairs) == len(pair_embeddings), "pairs/embeddings misalignment: filter logic bug"

    # Feature descriptor for plot titles / logs.
    feature_tags = []
    if use_concat:
        feature_tags.append("concat")
    if use_diff:
        feature_tags.append("diff")
    if use_unit_diff:
        feature_tags.append("unit_diff")
    if use_prod:
        feature_tags.append("prod")
    feature_desc = "+".join(feature_tags) if feature_tags else "none"

    # Compute PCA (also serves as an optional UMAP pre-step for speed).
    pca_dim = 2 if pre_pca_dim is None else max(2, int(pre_pca_dim))
    pca_reduced, pca_model = compute_pca_reduction(
        pair_embeddings,
        n_components=pca_dim,
        return_model=True,
        svd_solver="randomized",
        random_state=random_state,
    )
    pca_2d = pca_reduced[:, :2]
    pc1 = f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%} var)" if pca_model is not None else "PC1"
    pc2 = f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%} var)" if pca_model is not None else "PC2"
    pca_path = _derive_output_path(output_path, "pca")

    # Attach PCA coordinates for order/metadata diagnostics and category plots.
    pairs = pairs.copy()
    pairs['pc1'] = pca_2d[:, 0]
    pairs['pc2'] = pca_2d[:, 1]

    # --- Ordering diagnostics: swap halves and project with the same PCA model ---
    diagnostics: dict = {
        'n_pairs': int(len(pairs)),
        'embedding_dim_total': int(pair_embeddings.shape[1]) if pair_embeddings.ndim == 2 else None,
        'feature_flags': {'use_concat': bool(use_concat), 'use_diff': bool(use_diff), 'use_unit_diff': bool(use_unit_diff), 'use_prod': bool(use_prod)},
        'pca_explained_variance_ratio_top2': (
            [float(pca_model.explained_variance_ratio_[0]), float(pca_model.explained_variance_ratio_[1])]
            if pca_model is not None and hasattr(pca_model, 'explained_variance_ratio_') else None
        ),
    }
    try:
        if pca_model is not None and pair_embeddings.ndim == 2 and use_concat:
            # We can only define a meaningful "swap (A,B)->(B,A)" when concat is present.
            #
            # Layout when use_concat is True:
            # - block0: emb_a (D)
            # - block1: emb_b (D)
            # - optional symmetric blocks: |a-b| (D), a*b (D)
            n_blocks = 2 + int(bool(use_diff)) + int(bool(use_prod))
            total_dim = int(pair_embeddings.shape[1])
            if total_dim % n_blocks != 0:
                raise ValueError(f"Unexpected feature dim: total_dim={total_dim} not divisible by n_blocks={n_blocks}")
            d = total_dim // n_blocks
            diagnostics['embedding_dim_half'] = int(d)

            blocks = [pair_embeddings[:, i*d:(i+1)*d] for i in range(n_blocks)]
            # Swap only A/B blocks; keep symmetric blocks unchanged.
            swapped = np.concatenate([blocks[1], blocks[0], *blocks[2:]], axis=1)
            swapped_pca = pca_model.transform(swapped)
            swapped_2d = swapped_pca[:, :2]

            pc1_ab = pca_reduced[:, 0]
            pc1_ba = swapped_pca[:, 0]
            pc2_ab = pca_reduced[:, 1]
            pc2_ba = swapped_pca[:, 1]

            # Sign-based flip rate (simple, but PC sign is arbitrary across *different* fits;
            # here we use the same fitted PCA model, so it's still informative).
            flip_rate = float(np.mean((pc1_ab >= 0) != (pc1_ba >= 0)))

            # More robust magnitude-based diagnostics.
            mean_abs_delta_pc1 = float(np.mean(np.abs(pc1_ab - pc1_ba)))
            mean_abs_delta_pc2 = float(np.mean(np.abs(pc2_ab - pc2_ba)))
            mean_l2_delta_2d = float(np.mean(np.linalg.norm(pca_2d - swapped_2d, axis=1)))

            corr_pc1 = None
            if np.nanstd(pc1_ab) > 0 and np.nanstd(pc1_ba) > 0:
                corr_pc1 = float(np.corrcoef(pc1_ab, pc1_ba)[0, 1])

            diagnostics.update({
                'order_diagnostics': {
                    'flip_rate_pc1_sign': flip_rate,
                    'mean_abs_delta_pc1': mean_abs_delta_pc1,
                    'mean_abs_delta_pc2': mean_abs_delta_pc2,
                    'mean_l2_delta_pca2d': mean_l2_delta_2d,
                    'corr_pc1_ab_vs_ba': corr_pc1,
                }
            })
    except Exception as e:
        diagnostics.update({'order_diagnostics_error': f"{type(e).__name__}: {e}"})

    # --- Metadata summaries (if available): seg_pair / func_pair + embedding half norms ---
    try:
        if 'seg_a' in pairs.columns and 'seg_b' in pairs.columns:
            seg_a = pairs['seg_a'].astype(str).fillna('null')
            seg_b = pairs['seg_b'].astype(str).fillna('null')
            pairs['seg_pair'] = seg_a + '→' + seg_b
            print("\n🔎 PCA clustering summary by seg_pair (top 15):")
            print(pairs['seg_pair'].value_counts().head(15).to_string())
            seg_grp = pairs.groupby('seg_pair')[['pc1', 'pc2']].agg(['count', 'mean', 'std']).sort_values(('pc1', 'count'), ascending=False)
            print("\n   seg_pair PCA means/stds (top 10 by count):")
            print(seg_grp.head(10).to_string())

        if 'func_a' in pairs.columns and 'func_b' in pairs.columns:
            func_a = pairs['func_a'].astype(str).fillna('null')
            func_b = pairs['func_b'].astype(str).fillna('null')
            pairs['func_pair'] = func_a + '→' + func_b
            print("\n🔎 PCA clustering summary by func_pair (top 15):")
            print(pairs['func_pair'].value_counts().head(15).to_string())
            func_grp = pairs.groupby('func_pair')[['pc1', 'pc2']].agg(['count', 'mean', 'std']).sort_values(('pc1', 'count'), ascending=False)
            print("\n   func_pair PCA means/stds (top 10 by count):")
            print(func_grp.head(10).to_string())

        def _corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                return None
            return float(np.corrcoef(x, y)[0, 1])

        label_arr = pairs['label'].to_numpy().astype(float)

        # Norm analysis depends on feature mode:
        # - concat mode: feature = [emb_a, emb_b], can extract halves
        # - diff-only mode: feature = emb_a - emb_b, feature norm IS diff norm
        # - prod-only mode: feature = emb_a * emb_b, similar to diff
        if use_concat and pair_embeddings.ndim == 2:
            # Concat mode: we can extract emb_a and emb_b halves
            n_blocks = 1 + int(bool(use_diff)) + int(bool(use_prod))  # concat + optional diff/prod
            total_dim = pair_embeddings.shape[1]
            d = total_dim // (1 + n_blocks)  # Each concat half is D
            if total_dim % (1 + n_blocks) == 0 or total_dim % 2 == 0:
                # Fallback: just split in half for concat
                d = total_dim // 2
            norm_a = np.linalg.norm(pair_embeddings[:, :d], axis=1)
            norm_b = np.linalg.norm(pair_embeddings[:, d:2*d], axis=1)
            norm_ratio = norm_a / (norm_b + 1e-12)

            corr_norm_a_pc1 = _corr(norm_a, pairs['pc1'].to_numpy())
            corr_norm_b_pc1 = _corr(norm_b, pairs['pc1'].to_numpy())
            corr_ratio_pc1 = _corr(norm_ratio, pairs['pc1'].to_numpy())

            print("\n🔎 Embedding-half norm correlations with PC1 (concat mode):")
            print(f"   corr(||emb_a||, PC1) = {corr_norm_a_pc1}")
            print(f"   corr(||emb_b||, PC1) = {corr_norm_b_pc1}")
            print(f"   corr(||emb_a||/||emb_b||, PC1) = {corr_ratio_pc1}")

            corr_norm_a_label = _corr(norm_a, label_arr)
            corr_norm_b_label = _corr(norm_b, label_arr)
            corr_ratio_label = _corr(norm_ratio, label_arr)

            # Diff norm: ||emb_a - emb_b|| (reconstruct from concat halves)
            diff_norm = np.linalg.norm(pair_embeddings[:, :d] - pair_embeddings[:, d:2*d], axis=1)
            corr_diff_norm_label = _corr(diff_norm, label_arr)
            corr_diff_norm_pc1 = _corr(diff_norm, pairs['pc1'].to_numpy())

            print("\n🔎 Embedding-half norm correlations with LABEL (leakage detection):")
            print(f"   corr(||emb_a||, label)           = {corr_norm_a_label:.4f}" if corr_norm_a_label else "   corr(||emb_a||, label)           = N/A")
            print(f"   corr(||emb_b||, label)           = {corr_norm_b_label:.4f}" if corr_norm_b_label else "   corr(||emb_b||, label)           = N/A")
            print(f"   corr(||emb_a||/||emb_b||, label) = {corr_ratio_label:.4f}" if corr_ratio_label else "   corr(||emb_a||/||emb_b||, label) = N/A")
            print(f"   corr(||emb_a - emb_b||, label)   = {corr_diff_norm_label:.4f}" if corr_diff_norm_label else "   corr(||emb_a - emb_b||, label)   = N/A")
            print(f"   corr(||emb_a - emb_b||, PC1)     = {corr_diff_norm_pc1:.4f}" if corr_diff_norm_pc1 else "   corr(||emb_a - emb_b||, PC1)     = N/A")

            # Mean diff norm by label
            pos_mask = label_arr == 1
            neg_mask = label_arr == 0
            mean_diff_norm_pos = float(np.mean(diff_norm[pos_mask])) if pos_mask.sum() > 0 else None
            mean_diff_norm_neg = float(np.mean(diff_norm[neg_mask])) if neg_mask.sum() > 0 else None
            print(f"\n🔎 Diff norm by label:")
            print(f"   mean ||emb_a - emb_b|| for positives = {mean_diff_norm_pos:.4f}" if mean_diff_norm_pos else "   mean ||emb_a - emb_b|| for positives = N/A")
            print(f"   mean ||emb_a - emb_b|| for negatives = {mean_diff_norm_neg:.4f}" if mean_diff_norm_neg else "   mean ||emb_a - emb_b|| for negatives = N/A")

            diagnostics.update({
                'norm_correlations': {
                    'corr_norm_a_pc1': corr_norm_a_pc1,
                    'corr_norm_b_pc1': corr_norm_b_pc1,
                    'corr_norm_ratio_pc1': corr_ratio_pc1,
                },
                'label_correlations': {
                    'corr_norm_a_label': corr_norm_a_label,
                    'corr_norm_b_label': corr_norm_b_label,
                    'corr_norm_ratio_label': corr_ratio_label,
                    'corr_diff_norm_label': corr_diff_norm_label,
                    'corr_diff_norm_pc1': corr_diff_norm_pc1,
                    'mean_diff_norm_pos': mean_diff_norm_pos,
                    'mean_diff_norm_neg': mean_diff_norm_neg,
                },
            })

        elif (use_diff or use_prod) and not use_concat and pair_embeddings.ndim == 2:
            # Diff-only or prod-only mode: the feature vector IS the transform
            # The norm of the feature vector is the norm of the diff/prod
            feature_norm = np.linalg.norm(pair_embeddings, axis=1)
            corr_feature_norm_label = _corr(feature_norm, label_arr)
            corr_feature_norm_pc1 = _corr(feature_norm, pairs['pc1'].to_numpy())

            mode_name = "diff" if use_diff else "prod"
            print(f"\n🔎 Feature norm correlations ({mode_name} mode, no concat):")
            print(f"   corr(||feature||, label) = {corr_feature_norm_label:.4f}" if corr_feature_norm_label else f"   corr(||feature||, label) = N/A")
            print(f"   corr(||feature||, PC1)   = {corr_feature_norm_pc1:.4f}" if corr_feature_norm_pc1 else f"   corr(||feature||, PC1)   = N/A")

            # Mean feature norm by label
            pos_mask = label_arr == 1
            neg_mask = label_arr == 0
            mean_norm_pos = float(np.mean(feature_norm[pos_mask])) if pos_mask.sum() > 0 else None
            mean_norm_neg = float(np.mean(feature_norm[neg_mask])) if neg_mask.sum() > 0 else None
            print(f"\n🔎 Feature norm by label ({mode_name} mode):")
            print(f"   mean ||feature|| for positives = {mean_norm_pos:.4f}" if mean_norm_pos else "   mean ||feature|| for positives = N/A")
            print(f"   mean ||feature|| for negatives = {mean_norm_neg:.4f}" if mean_norm_neg else "   mean ||feature|| for negatives = N/A")

            diagnostics.update({
                'label_correlations': {
                    f'corr_{mode_name}_norm_label': corr_feature_norm_label,
                    f'corr_{mode_name}_norm_pc1': corr_feature_norm_pc1,
                    f'mean_{mode_name}_norm_pos': mean_norm_pos,
                    f'mean_{mode_name}_norm_neg': mean_norm_neg,
                },
            })
    except Exception as e:
        diagnostics.update({'metadata_diagnostics_error': f"{type(e).__name__}: {e}"})

    # Save diagnostics JSON next to plots.
    try:
        diag_path = output_path.parent / "pair_embedding_order_diagnostics.json"
        with open(diag_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"\nSaved pair embedding ordering diagnostics: {diag_path}")
    except Exception as e:
        print(f"WARNING: Failed to save pair embedding ordering diagnostics JSON ({type(e).__name__}: {e})")

    # --- Extra PCA plots colored by seg_pair / func_pair (top-N, collapsed) ---
    def _plot_pca_by_category(
        category_col: str,
        top_n: int,
        out_name: str,
        title_prefix: str,
        ) -> None:
        if category_col not in pairs.columns:
            return
        cats = _collapse_to_top_n(pairs[category_col].astype(str), top_n=top_n, other_label='Other')
        unique = sorted(cats.unique().tolist())
        if len(unique) <= 1:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        cmap = plt.cm.get_cmap('tab20', max(3, len(unique)))
        for i, cat in enumerate(unique):
            mask = (cats == cat).to_numpy()
            ax.scatter(
                pca_2d[mask, 0],
                pca_2d[mask, 1],
                s=18,
                alpha=0.7,
                c=[cmap(i)],
                label=str(cat),
                edgecolors='none',
            )
        ax.set_xlabel(pc1)
        ax.set_ylabel(pc2)
        ax.set_title(f"{title_prefix} (top {top_n}; rest→Other)")
        if filters_text:
            ax.text(
                0.01, 0.01,
                f"Filters: {filters_text}",
                transform=ax.transAxes,
                ha='left', va='bottom',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            )
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True, framealpha=0.9)
        plt.tight_layout()
        out_path = output_path.parent / out_name
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    _plot_pca_by_category(
        category_col='seg_pair',
        top_n=12,
        out_name='pair_embeddings_pca_by_seg_pair.png',
        title_prefix='PCA: Pair embeddings colored by seg_pair',
    )
    _plot_pca_by_category(
        category_col='func_pair',
        top_n=12,
        out_name='pair_embeddings_pca_by_func_pair.png',
        title_prefix='PCA: Pair embeddings colored by func_pair',
    )

    _plot_pair_features_splits_2d(
        reduced_2d=pca_2d,
        pairs=pairs,
        splits=splits,
        split_colors=split_colors,
        split_markers=split_markers,
        xlab=pc1,
        ylab=pc2,
        title=f"PCA: Pair Embeddings Splits Overlap (features={feature_desc})",
        output_path=pca_path,
        filters_text=filters_text,
    )

    # Optionally compute UMAP if available.
    try:
        from src.utils.dim_reduction_utils import compute_umap_reduction  # optional dep

        umap_input = pca_reduced if pre_pca_dim is not None else pair_embeddings
        method = f"PCA({pca_dim})→UMAP" if pre_pca_dim is not None else "UMAP"
        umap_2d, _ = compute_umap_reduction(
            umap_input,
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
            return_model=False,
        )
        _plot_pair_features_splits_2d(
            reduced_2d=umap_2d,
            pairs=pairs,
            splits=splits,
            split_colors=split_colors,
            split_markers=split_markers,
            xlab="UMAP1",
            ylab="UMAP2",
            title=f"{method}:  Pair Embeddings Splits Overlap (features={feature_desc})",
            output_path=output_path,
            filters_text=filters_text,
        )
    except ImportError as e:
        print(f"UMAP not available ({e}). Saved PCA plot only: {pca_path}")
    except Exception as e:
        print(f"WARNING: UMAP failed ({type(e).__name__}: {e}). Saved PCA plot only: {pca_path}")


def plot_sequence_embeddings_by_confounders_from_pairs(
    run_dir: Path,
    embeddings_file: Path,
    protein_metadata_csv: Path,
    output_dir: Path,
    max_pairs_per_label_per_split: int = 1500,
    max_sequences: int = 5000,
    random_state: int = 42,
    top_n_categories: int = 12,
    ) -> None:
    """Project *single-sequence* embeddings (from sequences appearing in pairs) and color by confounders.

    Practical intent: check whether embeddings already separate by host/subtype/year/geo/passage
    (potential confounding signal), ideally within a single dominant function (e.g., HA).
    """
    # Load + sample pairs
    splits = ['train', 'val', 'test']
    sampled = []
    for split in splits:
        df = _load_pairs_minimal(run_dir, split)
        if df is None or len(df) == 0:
            continue
        df_s = sample_pairs_stratified(
            df,
            max_per_label=max_pairs_per_label_per_split,
            label_col='label',
            random_state=random_state,
        )
        sampled.append(df_s[['brc_a', 'brc_b', 'label']])
    if not sampled:
        return
    pairs = pd.concat(sampled, ignore_index=True)

    if not protein_metadata_csv.exists():
        print(f"Skipping confounder embedding plots (missing protein metadata: {protein_metadata_csv})")
        return

    # Load protein metadata (minimal columns if possible)
    usecols = [
        'brc_fea_id', 'function', 'canonical_segment', 'assembly_id',
        'host', 'year', 'hn_subtype', 'geo_location_clean', 'passage',
    ]
    try:
        meta = pd.read_csv(protein_metadata_csv, usecols=usecols, dtype=str, low_memory=False)
    except Exception:
        meta = pd.read_csv(protein_metadata_csv, dtype=str, low_memory=False)
        meta = meta[[c for c in usecols if c in meta.columns]]

    # Extract unique sequences from sampled pairs, with metadata merged
    unique_seqs = extract_unique_sequences_from_pairs(pairs, metadata_df=meta)
    if len(unique_seqs) == 0:
        return

    # Prefer focusing on a single dominant function to avoid "function clusters" dominating the plot
    target_function = None
    if 'function' in unique_seqs.columns:
        vc = unique_seqs['function'].dropna().astype(str).value_counts()
        if len(vc) > 0:
            target_function = vc.index[0]

    seqs_for_plot = unique_seqs.copy()
    if target_function is not None:
        seqs_for_plot = seqs_for_plot[seqs_for_plot['function'] == target_function].reset_index(drop=True)

    if len(seqs_for_plot) == 0:
        return

    if len(seqs_for_plot) > max_sequences:
        seqs_for_plot = seqs_for_plot.sample(n=max_sequences, random_state=random_state).reset_index(drop=True)

    # Load embeddings via composite (assembly_id, brc_fea_id) keys.
    id_to_row = load_embedding_index(embeddings_file)
    keys = list(zip(seqs_for_plot['assembly_id'].astype(str),
                    seqs_for_plot['brc_fea_id'].astype(str)))
    embeddings, valid_keys = load_embeddings_by_ids(
        keys,
        embeddings_file,
        id_to_row=id_to_row,
    )
    if len(embeddings) == 0:
        return
    valid_brc_ids = {brc for _, brc in valid_keys}
    seqs_for_plot = seqs_for_plot[seqs_for_plot['brc_fea_id'].isin(valid_brc_ids)].reset_index(drop=True)

    # Reduce dimensionality (UMAP preferred; PCA fallback)
    method = 'UMAP'
    try:
        from src.utils.dim_reduction_utils import compute_umap_reduction
        emb_2d, _ = compute_umap_reduction(embeddings, n_components=2, random_state=random_state, return_model=False)
        xlabel, ylabel = 'UMAP1', 'UMAP2'
    except Exception:
        method = 'PCA'
        emb_2d, pca = compute_pca_reduction(embeddings, n_components=2, return_model=True)
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)" if pca is not None else "PC1"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)" if pca is not None else "PC2"

    func_tag = f"{target_function}" if target_function is not None else "all_functions"
    func_title = f" (function={target_function})" if target_function is not None else ""

    # Derived year bins (pre/post 2020)
    if 'year' in seqs_for_plot.columns:
        y = pd.to_numeric(seqs_for_plot['year'], errors='coerce')
        seqs_for_plot['year_bin_prepost2020'] = np.where(y.isna(), 'Unknown', np.where(y < 2020, '<=2019', '2020+'))

    confounders: list[tuple[str, str]] = [
        ('host', 'Host'),
        ('hn_subtype', 'H/N Subtype'),
        ('year_bin_prepost2020', 'Year (<=2019 vs 2020+)'),
        ('geo_location_clean', 'Geography (geo_location_clean)'),
        ('passage', 'Passage'),
    ]

    for col, label in confounders:
        if col not in seqs_for_plot.columns:
            continue
        cats = seqs_for_plot[col].copy()
        cats = cats.replace({np.nan: 'null'})
        # Skip if no variation
        non_null_unique = cats[cats != 'null'].nunique()
        if non_null_unique < 2:
            continue
        cats2 = _collapse_to_top_n(cats, top_n=top_n_categories, other_label='Other')
        out_name = f"sequence_embeddings_{method.lower()}_{func_tag}_by_{col}.png"
        plot_embeddings_by_category(
            emb_2d,
            cats2,
            title=f"{method}: Sequence Embeddings by {label}{func_title}",
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=output_dir / out_name,
        )


def should_plot_distribution(distribution_dict: dict, min_unique: int = 2) -> bool:
    """
    Check if distribution has enough variation to be informative.
    
    Args:
        distribution_dict: Dictionary mapping values to counts
        min_unique: Minimum number of unique values required
    
    Returns:
        True if distribution should be plotted, False otherwise
    """
    if not distribution_dict:
        return False
    # Count non-zero entries (unique values with data)
    unique_values = len([k for k, v in distribution_dict.items() if v > 0 and k != 'null'])
    return unique_values >= min_unique


# Per-regime display order for the split-composition stack. Positive first
# (single segment), then negatives by ascending hardness — same order used by
# `_LEVEL1_REGIME_ORDER` in src/analysis/analyze_stage4_train.py.
_SPLIT_COMP_NEG_ORDER = (
    'none_match',
    'host_only',
    'subtype_only',
    'year_only',
    'host_subtype_only',
    'host_year_only',
    'subtype_year_only',
    'host_subtype_year',
)


def _load_regime_manifest_for_composition(
    manifest_csv: Path,
    ) -> Optional[dict]:
    """Read negative_regime_manifest.csv into {split -> {regime -> achieved}}.

    Returns None if the file is absent or unreadable; the caller falls back to
    deriving counts from the saved pair CSVs.
    """
    if not manifest_csv.exists():
        return None
    try:
        df = pd.read_csv(manifest_csv)
    except Exception as e:
        print(f"WARNING: failed to read regime manifest {manifest_csv}: {e}")
        return None
    required_cols = {'split', 'regime', 'achieved'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"WARNING: regime manifest missing columns {sorted(missing)}; "
              f"falling back to pair-CSV derivation")
        return None
    out: dict = {}
    for (split, regime), sub in df.groupby(['split', 'regime']):
        out.setdefault(str(split), {})[str(regime)] = int(sub['achieved'].sum())
    return out


def _derive_regime_counts_from_pairs(
    run_dir: Path,
    splits: list,
    ) -> dict:
    """Compute {split -> {regime -> count}} for negatives by classifying each
    saved pair via same_host / same_hn_subtype / same_year. Works whether or
    not regime-aware sampling was used at construction time.

    Returns an empty dict if no pair CSV is readable. Regimes with zero
    negatives are simply absent from the inner dict; callers should treat
    a missing key as "N/A" rather than as zero.
    """
    out: dict = {}
    for sp in splits:
        csv = run_dir / f'{sp}_pairs.csv'
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(
                csv, engine='python',
                usecols=['label', 'same_hn_subtype', 'same_host', 'same_year'],
            )
        except Exception as e:
            print(f"WARNING: failed reading {csv}: {e}; skipping {sp} in "
                  f"regime derivation")
            continue
        neg = df[df['label'] == 0]
        if len(neg) == 0:
            out[sp] = {}
            continue

        def _regime(row):
            h = bool(row['same_host'])
            s = bool(row['same_hn_subtype'])
            y = bool(row['same_year'])
            if not h and not s and not y: return 'none_match'
            if h and not s and not y:     return 'host_only'
            if not h and s and not y:     return 'subtype_only'
            if not h and not s and y:     return 'year_only'
            if h and s and not y:         return 'host_subtype_only'
            if h and not s and y:         return 'host_year_only'
            if not h and s and y:         return 'subtype_year_only'
            return 'host_subtype_year'

        regimes = neg.apply(_regime, axis=1)
        counts = regimes.value_counts().to_dict()
        out[sp] = {r: int(c) for r, c in counts.items()}
    return out


def _read_split_composition_config(run_dir: Path) -> tuple:
    """Extract (neg_to_pos_ratio, regime_targets) from resolved_config.yaml.

    Returns (None, None) on read failure. regime_targets is None when
    regime-aware sampling is disabled for the run.
    """
    cfg_path = run_dir / 'resolved_config.yaml'
    if not cfg_path.exists():
        return None, None
    try:
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"WARNING: failed reading {cfg_path}: {e}")
        return None, None
    ds = cfg.get('dataset', {}) or {}
    ratio = ds.get('neg_to_pos_ratio')
    ns = ds.get('negative_sampling') or {}
    targets = ns.get('regime_targets') if isinstance(ns, dict) else None
    return ratio, targets


def _format_regime_targets_lines(targets: dict, per_line: int = 4) -> list:
    """Multi-line summary of regime_targets as fractions in _SPLIT_COMP_NEG_ORDER.

    Returns a list of lines; first line is prefixed `regime_targets:` and the
    rest are indented for visual alignment. `per_line` controls how many
    `regime=value` pairs go on each line.
    """
    if not targets:
        return []
    items = [f'{r}={float(targets[r]):.2f}'
             for r in _SPLIT_COMP_NEG_ORDER if r in targets]
    chunks = [items[i:i + per_line] for i in range(0, len(items), per_line)]
    lines = []
    for i, chunk in enumerate(chunks):
        prefix = 'regime_targets: ' if i == 0 else '                '
        lines.append(prefix + ', '.join(chunk))
    return lines


def _split_composition_text_annotations(
    neg_to_pos_ratio,
    regime_targets,
    holdout_active: bool,
    split_sizes: dict,
    splits: list,
    ) -> list:
    """Build the upper-left text-annotation lines for a split-composition plot.

    Returns a list of (line, color, bg) tuples. Caller renders them in order.
    """
    lines = []
    if neg_to_pos_ratio is not None:
        lines.append((f'neg_to_pos_ratio (config) = {float(neg_to_pos_ratio):.2f}',
                      'black', None))
    for regime_line in _format_regime_targets_lines(regime_targets):
        lines.append((regime_line, 'black', None))
    if holdout_active:
        share_parts = []
        for s in splits:
            sh = split_sizes.get(s, {}).get('isolate_share')
            share_parts.append(f'{sh:.1%}' if sh is not None else '—')
        lines.append((
            f'Holdout active: actual = {" / ".join(share_parts)} (isolates); '
            f'train_ratio/val_ratio/test_ratio ignored',
            '#5a3a00', '#fff2cc',
        ))
    return lines


def _render_split_composition_vstacked(
    splits: list,
    pos_counts: np.ndarray,
    regime_counts_per_split: dict,
    by_regime: bool,
    annotations: list,
    bundle_name: str,
    output_path: Path,
    ) -> None:
    """Plot variant: vertical bars, positive at bottom + negative stacked on top.
    When `by_regime=True` the negative portion is split into 8 regime segments
    (ascending hardness, _SPLIT_COMP_NEG_ORDER).
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(splits))
    bar_width = 0.55

    # Positive segment at the bottom.
    ax.bar(x, pos_counts, width=bar_width, color='#2ca02c',
           edgecolor='white', linewidth=0.5, label='positive')

    if by_regime:
        palette = plt.get_cmap('tab10').colors
        regime_colors = {r: palette[(i + 1) % len(palette)]
                         for i, r in enumerate(_SPLIT_COMP_NEG_ORDER)}
        bottom = pos_counts.astype(float).copy()
        neg_totals = np.zeros(len(splits), dtype=int)
        for r in _SPLIT_COMP_NEG_ORDER:
            heights = np.array(
                [regime_counts_per_split.get(s, {}).get(r, 0) for s in splits],
                dtype=int,
            )
            if heights.sum() == 0:
                continue
            ax.bar(x, heights, width=bar_width, bottom=bottom,
                   color=regime_colors[r], edgecolor='white', linewidth=0.5,
                   label=f'neg: {r}')
            bottom = bottom + heights
            neg_totals += heights
        total_counts = pos_counts + neg_totals
    else:
        neg_counts = np.array(
            [sum(regime_counts_per_split.get(s, {}).values()) for s in splits],
            dtype=int,
        )
        ax.bar(x, neg_counts, width=bar_width, bottom=pos_counts,
               color='#d62728', edgecolor='white', linewidth=0.5,
               label='negative')
        total_counts = pos_counts + neg_counts

    # Total-count annotation above each bar.
    top = int(max(total_counts)) if len(total_counts) else 1
    for i, total in enumerate(total_counts):
        ax.text(x[i], total + top * 0.01,
                f'{int(total):,}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    # Achieved neg:pos centered inside each bar.
    for i in range(len(splits)):
        ach = (total_counts[i] - pos_counts[i]) / pos_counts[i] if pos_counts[i] > 0 else float('nan')
        txt = f'neg:pos (achieved) = {ach:.2f}' if not np.isnan(ach) else 'neg:pos (achieved) = —'
        ax.text(x[i], total_counts[i] * 0.5, txt,
                ha='center', va='center', fontsize=8, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#888888', alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    ax.set_ylabel('Pair count', fontsize=11)
    ax.set_title(f'Split composition — {bundle_name}', fontsize=12, fontweight='bold')
    # Add extra top margin to fit text annotations without colliding with bar
    # count labels. The margin scales with the number of annotation lines.
    n_anno_lines = len(annotations)
    ax.set_ylim(0, top * 1.12)
    ax.grid(True, axis='y', alpha=0.3)

    # Legend outside on the right; reverse order so positive sits at top.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left',
              bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
    fig.tight_layout()
    # Make room above the axes for the figure-level annotation lines.
    if annotations:
        fig.subplots_adjust(bottom=_annotation_bottom_reserved(len(annotations)))
        _render_text_annotations(fig, ax, annotations)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def _render_split_composition_grouped(
    splits: list,
    pos_counts: np.ndarray,
    regime_counts_per_split: dict,
    by_regime: bool,
    annotations: list,
    bundle_name: str,
    output_path: Path,
    ) -> None:
    """Plot variant: horizontal bars, NOT stacked. For each split there is
    a group of bars (positive + neg or positive + 8 regime bars); each bar
    is labeled with its count. Bars with zero count are rendered as a thin
    placeholder and labeled "N/A".
    """
    if by_regime:
        bar_categories = ['positive'] + [f'neg: {r}' for r in _SPLIT_COMP_NEG_ORDER]
        n_bars_per_group = 1 + len(_SPLIT_COMP_NEG_ORDER)
    else:
        bar_categories = ['positive', 'negative']
        n_bars_per_group = 2

    n_groups = len(splits)
    fig_height = max(4.5, 0.42 * n_bars_per_group * n_groups + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    pos_color = '#2ca02c'
    neg_blended_color = '#d62728'
    palette = plt.get_cmap('tab10').colors
    regime_colors = {r: palette[(i + 1) % len(palette)]
                     for i, r in enumerate(_SPLIT_COMP_NEG_ORDER)}

    # Layout: groups stacked top-to-bottom (train at top), bars within a group
    # also top-to-bottom. y-coordinates are negated so larger y is lower.
    y_positions = []
    bar_labels = []
    bar_colors = []
    bar_values = []
    bar_value_strs = []
    group_centers = []   # y-coord at the center of each group, for the split label

    intra_group_gap = 0.0
    inter_group_gap = 1.0

    cursor = 0.0
    for gi, sp in enumerate(splits):
        group_top = cursor
        for bi, cat in enumerate(bar_categories):
            y = cursor
            cursor += 1.0 + intra_group_gap

            if cat == 'positive':
                count = int(pos_counts[gi])
                color = pos_color
            elif cat == 'negative':
                count = int(sum(regime_counts_per_split.get(sp, {}).values()))
                color = neg_blended_color
            else:
                regime = cat.split('neg: ', 1)[1]
                count = int(regime_counts_per_split.get(sp, {}).get(regime, 0))
                color = regime_colors[regime]

            y_positions.append(y)
            bar_labels.append(cat)
            bar_colors.append(color)
            bar_values.append(count)
            bar_value_strs.append(f'{count:,}' if count > 0 else 'N/A')

        group_bottom = cursor - 1.0 - intra_group_gap
        group_centers.append((group_top + group_bottom) / 2.0)
        cursor += inter_group_gap

    y_arr = np.array(y_positions)
    val_arr = np.array(bar_values, dtype=float)
    # For "N/A" bars we draw a tiny visible nub so the row is locatable on
    # the y-axis; the label "N/A" carries the meaning.
    max_val = max(1.0, val_arr.max() if len(val_arr) else 1.0)
    display_widths = np.where(val_arr > 0, val_arr, max_val * 0.005)
    ax.barh(-y_arr, display_widths, color=bar_colors, edgecolor='white',
            linewidth=0.4, height=0.85)

    # Tick labels = the regime/positive/negative names, in the same order.
    ax.set_yticks(-y_arr)
    ax.set_yticklabels(bar_labels, fontsize=8)

    # Value labels at the bar tip.
    pad = max_val * 0.01
    for i, (v, vs) in enumerate(zip(val_arr, bar_value_strs)):
        ax.text(display_widths[i] + pad, -y_arr[i], vs,
                ha='left', va='center', fontsize=8,
                color='#555555' if v == 0 else 'black')

    # Group labels (split names) on the right, near the group center.
    for gi, sp in enumerate(splits):
        ax.text(1.005, -group_centers[gi], sp.capitalize(),
                transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=11, fontweight='bold',
                rotation=270)

    # Horizontal separator lines between groups, drawn at the midpoint
    # of the inter_group_gap.
    if len(splits) > 1:
        group_size = (1.0 + intra_group_gap) * n_bars_per_group + inter_group_gap
        for gi in range(1, len(splits)):
            sep_y = -(gi * group_size - inter_group_gap / 2.0 - 0.5)
            ax.axhline(sep_y, color='#cccccc', linewidth=0.7, linestyle=':')

    ax.set_xlabel('Pair count', fontsize=11)
    ax.set_title(f'Split composition — {bundle_name}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max_val * 1.18)
    ax.grid(True, axis='x', alpha=0.3)

    fig.tight_layout()
    # Make room above the axes for the figure-level annotation lines.
    if annotations:
        fig.subplots_adjust(bottom=_annotation_bottom_reserved(len(annotations)))
        _render_text_annotations(fig, ax, annotations)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


_ANNO_LINE_H = 0.028  # fraction of figure height per annotation line
_ANNO_XAXIS_PAD = 0.05  # extra room reserved above the annotation block for the x-axis tick labels and xlabel


def _annotation_bottom_reserved(n_lines: int) -> float:
    """Figure-fraction of bottom space to reserve for n annotation lines. Includes
    a small margin above the lines so the x-axis label and tick labels stay
    visible. The caller subtracts this via
    `fig.subplots_adjust(bottom=_annotation_bottom_reserved(n))`.
    """
    if n_lines <= 0:
        return 0.0
    return _ANNO_LINE_H * n_lines + _ANNO_XAXIS_PAD


def _render_text_annotations(fig, ax, annotations: list) -> None:
    """Render annotation lines (built by `_split_composition_text_annotations`)
    in the figure footer, below the x-axis label. This keeps them clear of
    both the title (top) and bars (interior).

    The caller must reserve enough bottom space via
    `fig.subplots_adjust(bottom=_annotation_bottom_reserved(n))` before
    invoking this; otherwise lines collide with the x-axis label.
    """
    if not annotations:
        return
    bbox = ax.get_position()
    # First annotation line sits just above figure y=0, lines stack upward
    # toward the axes. Reverse iteration so the original order reads
    # top-to-bottom (line 0 of the list ends up on top, closest to the axes).
    n = len(annotations)
    for i, (line, color, bg) in enumerate(annotations):
        y = 0.01 + (n - 1 - i) * _ANNO_LINE_H
        kw = dict(ha='left', va='bottom', fontsize=9, color=color)
        if bg:
            kw['bbox'] = dict(boxstyle='round,pad=0.3', facecolor=bg,
                              edgecolor='#bf9000', alpha=0.9)
        fig.text(bbox.x0, y, line, **kw)


def _plot_split_composition(
    split_sizes: dict,
    coverage: dict,
    regime_manifest_csv: Path,
    bundle_name: str,
    output_path: Path,
    holdout_active: bool = False,
    run_dir: Optional[Path] = None,
    ) -> None:
    """Top-level orchestrator: emits 4 split-composition PNGs.

    Output files (all in the same directory as `output_path`):
      - split_composition.png                       (plot 1: vstacked, blended neg)
      - split_composition_by_regime.png             (plot 2: vstacked, 8 regime segments)
      - split_composition_grouped.png               (plot 3: horizontal grouped, pos vs neg)
      - split_composition_grouped_by_regime.png     (plot 4: horizontal grouped, pos + 8 regimes)

    The first filename (`output_path`) determines the directory; the other
    three filenames are derived from it. Per-regime breakdown for plots 2
    and 4 is drawn from the regime manifest when available, otherwise
    derived from the saved pair CSVs (same_host / same_hn_subtype /
    same_year columns), so all 4 plots render even when regime-aware
    sampling is off.
    """
    splits = ['train', 'val', 'test']
    splits = [s for s in splits if s in split_sizes]
    if not splits:
        print("WARNING: _plot_split_composition: no split data found; skipping")
        return

    pos_counts = np.array(
        [int(split_sizes[s].get('positive_pairs', 0)) for s in splits],
        dtype=int,
    )

    # Per-split per-regime negative counts. Prefer the manifest (regime-aware
    # builds) and fall back to deriving from pair CSVs (regime-blind builds).
    manifest_counts = _load_regime_manifest_for_composition(regime_manifest_csv)
    if manifest_counts is not None:
        regime_counts_per_split = manifest_counts
    elif run_dir is not None:
        regime_counts_per_split = _derive_regime_counts_from_pairs(run_dir, splits)
    else:
        regime_counts_per_split = {}

    # Pull annotation inputs (config-level neg_to_pos_ratio + regime_targets).
    neg_to_pos_ratio, regime_targets = (None, None)
    if run_dir is not None:
        neg_to_pos_ratio, regime_targets = _read_split_composition_config(run_dir)
    annotations = _split_composition_text_annotations(
        neg_to_pos_ratio=neg_to_pos_ratio,
        regime_targets=regime_targets,
        holdout_active=holdout_active,
        split_sizes=split_sizes,
        splits=splits,
    )

    out_dir = output_path.parent
    primary_stem = output_path.stem  # usually "split_composition"
    suffix = output_path.suffix or '.png'
    paths = {
        'vstacked_blended':  out_dir / f'{primary_stem}{suffix}',
        'vstacked_regimes':  out_dir / f'{primary_stem}_by_regime{suffix}',
        'grouped_blended':   out_dir / f'{primary_stem}_grouped{suffix}',
        'grouped_regimes':   out_dir / f'{primary_stem}_grouped_by_regime{suffix}',
    }

    _render_split_composition_vstacked(
        splits, pos_counts, regime_counts_per_split,
        by_regime=False, annotations=annotations,
        bundle_name=bundle_name, output_path=paths['vstacked_blended'],
    )
    _render_split_composition_vstacked(
        splits, pos_counts, regime_counts_per_split,
        by_regime=True, annotations=annotations,
        bundle_name=bundle_name, output_path=paths['vstacked_regimes'],
    )
    _render_split_composition_grouped(
        splits, pos_counts, regime_counts_per_split,
        by_regime=False, annotations=annotations,
        bundle_name=bundle_name, output_path=paths['grouped_blended'],
    )
    _render_split_composition_grouped(
        splits, pos_counts, regime_counts_per_split,
        by_regime=True, annotations=annotations,
        bundle_name=bundle_name, output_path=paths['grouped_regimes'],
    )


def plot_distribution_by_split(
    metadata_distributions: dict,
    metadata_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    top_n: int = 30,
    split_sizes: Optional[dict] = None
    ) -> None:
    """
    Plot distribution across train/val/test splits as 3 vertical subplots.
    
    Args:
        metadata_distributions: Dict with 'train', 'val', 'test' keys, each containing
                               metadata distributions (e.g., host, hn_subtype, year)
        metadata_key: Key to extract from each split (e.g., 'host', 'hn_subtype', 'year')
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot
        top_n: Number of top values to show
        split_sizes: Optional dict with 'train'/'val'/'test' -> {'isolates': N}.
                     If provided, uses isolates as denominator for "X of N" (consistent across metadata keys).
    """
    splits = ['train', 'val', 'test']
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    for idx, split in enumerate(splits):
        ax = axes[idx]
        dist = metadata_distributions.get(split, {}).get(metadata_key, {})
        
        if not dist:
            ax.text(0.5, 0.5, f'No {metadata_key} data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split.capitalize()} Split', fontsize=12, fontweight='bold')
            continue
        
        # Convert to Series for easier handling
        series = pd.Series(dist)
        # Remove 'null' entries and sort
        if 'null' in series.index:
            series = series.drop('null')
        
        # Total isolates in this metadata distribution (may exclude nulls)
        dist_total = series.sum()
        # Use split isolate count as denominator when available
        total_isolates = None
        if split_sizes and split in split_sizes:
            total_isolates = split_sizes[split].get('isolates')
        if total_isolates is None:
            total_isolates = int(dist_total)
        
        # Filter to top_n for display
        series = series.sort_values(ascending=False).head(top_n)
        displayed_count = series.sum()
        
        if len(series) == 0:
            ax.text(0.5, 0.5, f'No {metadata_key} data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split.capitalize()} Split', fontsize=12, fontweight='bold')
            continue
        
        # Keep the styling consistent across all metadata keys.
        bars = ax.barh(range(len(series)), series.values, color='steelblue')
        
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index, fontsize=9)
        ax.set_xlabel(ylabel, fontsize=11)
        
        # Title: "X of N (P%)" where N = total isolates in split and P = isolate_share
        # across train+val+test. Surfaces holdout-driven imbalances at a glance.
        share = None
        if split_sizes and split in split_sizes:
            share = split_sizes[split].get('isolate_share')
        share_txt = f', {share:.1%} of dataset' if share is not None else ''
        if displayed_count < dist_total:
            ax.set_title(
                f'{split.capitalize()} Split (showing {displayed_count:,} of {total_isolates:,}{share_txt})',
                fontsize=12, fontweight='bold')
        else:
            ax.set_title(
                f'{split.capitalize()} Split (n={total_isolates:,}{share_txt})',
                fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.grid(axis='y', visible=False)
        
        # Add value labels on bars
        for i, v in enumerate(series.values):
            ax.text(v + max(series.values) * 0.01, i, f'{v:,}', 
                   va='center', fontsize=8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_year_distribution_by_split(
    metadata_distributions: dict,
    output_path: Path,
    start_year: int = 2000,
    split_sizes: Optional[dict] = None,
    ) -> None:
    """
    Plot year distribution as histograms across train/val/test splits.

    Args:
        metadata_distributions: Dict with 'train', 'val', 'test' keys
        output_path: Path to save the plot
        start_year: Minimum year to display
        split_sizes: Optional dict with 'train'/'val'/'test' -> {'isolate_share': ...}.
                     When provided, each subplot title shows the split's share of the
                     full dataset (post-filtering) — mirrors `plot_distribution_by_split`.
    """
    splits = ['train', 'val', 'test']
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for idx, split in enumerate(splits):
        ax = axes[idx]
        year_dist = metadata_distributions.get(split, {}).get('year', {})
        
        if not year_dist:
            ax.text(0.5, 0.5, 'No year data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split.capitalize()} Split', fontsize=12, fontweight='bold')
            continue
        
        # Calculate total count (all isolates, regardless of year filter)
        total_count = sum(int(count) for year_str, count in year_dist.items() if year_str != 'null')
        
        # Convert to list of years (repeat each year by its count)
        years = []
        for year_str, count in year_dist.items():
            if year_str == 'null':
                continue
            try:
                year = int(year_str)
                if year >= start_year:
                    years.extend([year] * int(count))
            except (ValueError, TypeError):
                continue
        
        if len(years) == 0:
            ax.text(0.5, 0.5, 'No year data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split.capitalize()} Split', fontsize=12, fontweight='bold')
            continue
        
        displayed_count = len(years)
        
        # Create histogram with bins aligned to year boundaries
        if len(years) > 0:
            year_min = min(years)
            year_max = max(years)
            
            # Create bins aligned to integer year boundaries
            # Bins should be [year_min, year_min+1, year_min+2, ..., year_max+1]
            # This ensures each bin represents exactly one year
            bin_edges = np.arange(int(year_min), int(year_max) + 2, 1)
            
            # Create histogram with aligned bins
            n, bins, patches = ax.hist(years, bins=bin_edges, color='steelblue', edgecolor='black', alpha=0.7)
            
            # Set x-ticks at reasonable intervals for better alignment
            year_range = year_max - year_min
            
            # Determine tick interval based on range
            if year_range <= 5:
                tick_interval = 1  # Every year
            elif year_range <= 15:
                tick_interval = 2  # Every 2 years
            elif year_range <= 30:
                tick_interval = 5  # Every 5 years
            else:
                tick_interval = 10  # Every 10 years

            # Calculate bin centers for count labels
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Create ticks at round year values, centered exactly at year + 0.5
            # Since bins are aligned to integer years [year, year+1), center is exactly year + 0.5
            tick_start = (int(year_min) // tick_interval) * tick_interval
            tick_end = ((int(year_max) // tick_interval) + 1) * tick_interval
            x_ticks_years = np.arange(tick_start, tick_end + 1, tick_interval)
            
            # Filter ticks to be within the actual data range (with small margin)
            x_ticks_years = x_ticks_years[(x_ticks_years >= int(year_min)) & (x_ticks_years <= int(year_max) + 1)]
            
            # Center ticks exactly at year + 0.5 (bin centers for integer-aligned bins)
            x_ticks_centered = x_ticks_years + 0.5
            x_tick_labels = [int(year) for year in x_ticks_years]
            
            ax.set_xticks(x_ticks_centered)
            ax.set_xticklabels(x_tick_labels, rotation=90, ha='right')
            
            # Set x-axis limits to match bin edges
            ax.set_xlim(int(year_min) - 0.5, int(year_max) + 1.5)
            
            # Add count labels at the top of bars (vertically rotated, positioned higher)
            max_count = max(n) if len(n) > 0 else 1
            for i, (count, center) in enumerate(zip(n, bin_centers)):
                if count > 0:  # Only label bars with counts > 0
                    ax.text(center, count + max_count * 0.05, f'{int(count):,}', 
                           ha='center', va='bottom', fontsize=7, rotation=90)
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Number of Isolates', fontsize=11)
        
        # Title showing displayed count out of total + dataset share (when known).
        share = None
        if split_sizes and split in split_sizes:
            share = split_sizes[split].get('isolate_share')
        share_txt = f', {share:.1%} of dataset' if share is not None else ''
        if displayed_count < total_count:
            ax.set_title(
                f'{split.capitalize()} Split (showing {displayed_count:,} of {total_count:,}{share_txt})',
                fontsize=12, fontweight='bold')
        else:
            ax.set_title(
                f'{split.capitalize()} Split (n={total_count:,}{share_txt})',
                fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add statistics
        if len(years) > 0:
            stats_text = f"Range: {min(years)} - {max(years)}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Year Distribution by Split', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def load_isolate_metadata(run_dir: Path, project_root: Path) -> Optional[pd.DataFrame]:
    """
    Try to load isolate metadata from the original protein data file.
    
    Args:
        run_dir: Dataset run directory
        project_root: Project root directory
    
    Returns:
        DataFrame with isolate metadata (assembly_id, host, hn_subtype, year), or None
    """
    # Try to find the original protein_final.csv file
    # Common locations:
    # 1. Check if there's a symlink or reference in the run directory
    # 2. Try standard location: data/processed/{virus}/{data_version}/protein_final.csv
    # 3. Try to infer from dataset_stats.json structure
    
    # For now, we'll try to load from a standard location
    # This is a simplified approach - in practice, we might need to store the input file path
    # in dataset_stats.json or have it passed as a parameter
    
    # Try common paths (this is a heuristic)
    possible_paths = [
        project_root / 'data' / 'processed' / 'flu' / 'July_2025' / 'protein_final.csv',
        # Add more paths as needed
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if 'assembly_id' in df.columns:
                    # Get one row per isolate with metadata
                    isolate_meta = df.groupby('assembly_id').first()
                    if all(col in isolate_meta.columns for col in ['host', 'hn_subtype']):
                        return isolate_meta[['host', 'hn_subtype', 'year']].reset_index()
            except Exception as e:
                continue
    
    return None


def compute_crosstab_from_pairs(
    pair_csv_path: Path,
    df_meta: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
    """
    Compute host × subtype cross-tabulation from pair CSV.
    
    Args:
        pair_csv_path: Path to train_pairs.csv, val_pairs.csv, or test_pairs.csv
        df_meta: DataFrame with isolate metadata (must have assembly_id, host, hn_subtype)
    
    Returns:
        DataFrame with host × subtype cross-tabulation (isolate counts), or None if not possible
    """
    if not pair_csv_path.exists():
        return None
    
    if df_meta is None:
        return None
    
    try:
        # We only need isolate IDs for the crosstab; avoid reading the full pairs CSV.
        try:
            pairs_df = pd.read_csv(pair_csv_path, usecols=['assembly_id_a', 'assembly_id_b'], dtype=str)
        except Exception:
            pairs_df = pd.read_csv(pair_csv_path, dtype=str)
        
        # Get unique isolates from pairs
        unique_isolates = set(pairs_df['assembly_id_a'].unique()) | set(pairs_df['assembly_id_b'].unique())
        
        # Get metadata for these isolates
        isolate_meta = df_meta[df_meta['assembly_id'].isin(unique_isolates)]
        isolate_meta = isolate_meta.groupby('assembly_id').first()  # One row per isolate
        
        if 'host' in isolate_meta.columns and 'hn_subtype' in isolate_meta.columns:
            # Create cross-tabulation (count isolates, not pairs)
            crosstab = pd.crosstab(isolate_meta['host'], isolate_meta['hn_subtype'])
            return crosstab
        
        return None
        
    except Exception as e:
        print(f"WARNING: Error computing crosstab from {pair_csv_path}: {e}")
        return None


def load_isolate_metadata_from_run(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load per-isolate metadata saved during dataset creation (run_dir/isolate_metadata.csv)."""
    path = run_dir / "isolate_metadata.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        if 'assembly_id' not in df.columns:
            return None
        return df
    except Exception:
        return None


def plot_host_subtype_heatmap_by_split(
    dataset_stats_path: Path,
    run_dir: Path,
    bundle_name: str,
    output_path: Path,
    top_hosts: int = 15,
    top_subtypes: int = 15,
    split_sizes: Optional[dict] = None,
    ) -> None:
    """
    Plot host × HN subtype heatmap across train/val/test splits.
    
    Computes cross-tabulation from pair CSVs by loading isolate metadata from original protein data.
    
    Args:
        dataset_stats_path: Path to dataset_stats.json (to check if metadata available)
        run_dir: Directory containing pair CSVs
        bundle_name: Bundle name (for title)
        output_path: Path to save the plot
        project_root: Project root directory (to find original protein data)
        top_hosts: Number of top hosts to show
        top_subtypes: Number of top subtypes to show
    """
    splits = ['train', 'val', 'test']
    fig, axes = plt.subplots(3, 1, figsize=(max(14, top_subtypes * 0.7), max(12, top_hosts * 0.6 * 3)))
    
    # Load isolate metadata from run directory (saved during dataset creation).
    df_meta = load_isolate_metadata_from_run(run_dir)
    
    if df_meta is None:
        # Can't generate heatmap without metadata
        for idx, split in enumerate(splits):
            ax = axes[idx]
            ax.text(0.5, 0.5, 'Metadata not available\n(cannot load original protein data)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{split.capitalize()} Split', fontsize=12, fontweight='bold')
        plt.suptitle(f'Host × HN Subtype Distribution by Split ({bundle_name})', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("Skipping host×subtype heatmap: could not load isolate metadata (missing/unreadable isolate_metadata.csv)")
        return
    
    all_crosstabs = {}
    for idx, split in enumerate(splits):
        ax = axes[idx]
        pair_csv = run_dir / f'{split}_pairs.csv'
        
        crosstab = compute_crosstab_from_pairs(pair_csv, df_meta)
        
        if crosstab is None or len(crosstab) == 0:
            ax.text(0.5, 0.5, f'No cross-tabulation data available for {split}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split.capitalize()} Split', fontsize=12, fontweight='bold')
            continue
        
        all_crosstabs[split] = crosstab
        
        # Get top hosts and subtypes across all splits for consistency
        # (we'll compute this after getting all crosstabs)
    
    # Determine top hosts and subtypes across all splits
    if all_crosstabs:
        all_hosts = set()
        all_subtypes = set()
        for crosstab in all_crosstabs.values():
            all_hosts.update(crosstab.index)
            all_subtypes.update(crosstab.columns)
        
        # Get top N by total count across all splits
        host_counts = {}
        subtype_counts = {}
        for crosstab in all_crosstabs.values():
            for host in crosstab.index:
                host_counts[host] = host_counts.get(host, 0) + crosstab.loc[host].sum()
            for subtype in crosstab.columns:
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + crosstab[subtype].sum()
        
        top_host_list = sorted(host_counts.items(), key=lambda x: x[1], reverse=True)[:top_hosts]
        top_host_list = [h[0] for h in top_host_list]
        top_subtype_list = sorted(subtype_counts.items(), key=lambda x: x[1], reverse=True)[:top_subtypes]
        top_subtype_list = [s[0] for s in top_subtype_list]
    else:
        return  # No data to plot
    
    # Now plot each split
    for idx, split in enumerate(splits):
        ax = axes[idx]
        
        if split not in all_crosstabs:
            continue
        
        crosstab = all_crosstabs[split]
        
        # Filter + reorder to top hosts/subtypes.
        #
        # IMPORTANT: A host/subtype can be in the global top list but absent in a
        # particular split. Use reindex(fill_value=0) to avoid KeyError like:
        #   "['Gull'] not in index"
        #
        # This keeps the axes consistent across splits while making missing
        # categories explicit as all-zero rows/columns.
        crosstab_filtered = crosstab.reindex(
            index=top_host_list,
            columns=top_subtype_list,
            fill_value=0,
        )
        
        if len(crosstab_filtered) == 0:
            ax.text(0.5, 0.5, f'No data for {split}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split.capitalize()} Split', fontsize=12, fontweight='bold')
            continue
        
        # Create heatmap
        sns.heatmap(
            crosstab_filtered,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            cbar_kws={'label': 'Number of Isolates'},
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'size': 8}
        )
        ax.set_xlabel('H/N Subtype', fontsize=11)
        ax.set_ylabel('Host', fontsize=11)
        share = None
        if split_sizes and split in split_sizes:
            share = split_sizes[split].get('isolate_share')
        share_txt = f', {share:.1%} of dataset' if share is not None else ''
        ax.set_title(
            f'{split.capitalize()} Split (n={crosstab_filtered.sum().sum():,} isolates{share_txt})',
            fontsize=12, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.suptitle(f'Host × HN Subtype Distribution by Split ({bundle_name})', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_dataset_stats(
    dataset_stats_path: Path,
    bundle_name: str,
    output_dir_dataset: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    generate_confounder_plots: bool = False,
    skip_esm_pca_plots: bool = False,
    skip_kmer_pca_plots: bool = False,
    ) -> None:
    """
    Generate visualization plots for a dataset.

    Args:
        dataset_stats_path: Path to dataset_stats.json file
        bundle_name: Bundle name (for plot titles)
        output_dir_dataset: Output directory in dataset run folder (optional)
        skip_esm_pca_plots: If True, skip the four pair_pca_*.png plots and the
            pair_interaction_diagnostics.json file produced by
            plot_pair_interactions().
        skip_kmer_pca_plots: If True, skip kmer_pca_concat.png + the scree
            summary produced by plot_kmer_pca().
    """
    # Load dataset statistics
    if not dataset_stats_path.exists():
        raise FileNotFoundError(f"Dataset stats file not found: {dataset_stats_path}")

    with open(dataset_stats_path, 'r') as f:
        stats = json.load(f)

    metadata_distributions = stats.get('metadata_distributions', {})
    split_sizes = stats.get('split_sizes', {})
    if not metadata_distributions:
        print(f"WARNING: no metadata_distributions found in {dataset_stats_path}")
        return
    
    # Determine run_dir and project_root if not provided
    if run_dir is None:
        run_dir = dataset_stats_path.parent
    
    if project_root is None:
        # IMPORTANT:
        # Do NOT infer project root from dataset_stats_path via parents[].
        # When called from a dataset run directory (data/datasets/.../runs/...),
        # parents[] points into data/datasets/... and can easily be off-by-N,
        # causing config paths like ".../data/datasets/conf" (wrong).
        # Use the repository root based on this file location instead.
        project_root = Path(__file__).resolve().parents[2]
    
    # Set up plotting style
    apply_default_style()
    sns.set_palette('Set2')
    
    # Output directory (single source of truth: dataset run dir)
    if not output_dir_dataset:
        raise ValueError("output_dir_dataset must be provided")
    plots_dir = output_dir_dataset / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GENERATING DATASET VISUALIZATIONS FOR: {bundle_name}")
    print(f"{'='*70}")
    
    # Check what metadata is available
    train_meta = metadata_distributions.get('train', {})
    has_host = should_plot_distribution(train_meta.get('host', {}))
    has_subtype = should_plot_distribution(train_meta.get('hn_subtype', {}))
    has_year = should_plot_distribution(train_meta.get('year', {}))
    has_geo = should_plot_distribution(train_meta.get('geo_location_clean', {}))
    has_passage = should_plot_distribution(train_meta.get('passage', {}))

    top_n = 20
    start_year = 2000
    # breakpoint()

    # 0. Split composition (4 PNGs: vstacked-blended, vstacked-by-regime,
    # grouped-blended, grouped-by-regime). All 4 render regardless of whether
    # regime-aware sampling was used; the by-regime variants derive the
    # negative breakdown from the regime manifest if present, otherwise from
    # the saved pair CSVs.
    coverage = stats.get('coverage', {})
    holdout_active = stats.get('metadata_holdout') is not None
    try:
        _plot_split_composition(
            split_sizes=split_sizes,
            coverage=coverage,
            regime_manifest_csv=run_dir / 'negative_regime_manifest.csv',
            bundle_name=bundle_name,
            output_path=plots_dir / 'split_composition.png',
            holdout_active=holdout_active,
            run_dir=run_dir,
        )
    except Exception as e:
        print(f"WARNING: failed to render split_composition plots ({type(e).__name__}: {e})")

    # 1. Host distribution
    if has_host:
        plot_distribution_by_split(
            metadata_distributions,
            'host',
            f'Host Distribution by Split ({bundle_name})',
            'Number of Isolates',
            plots_dir / 'host_distribution.png',
            top_n=top_n,
            split_sizes=split_sizes
        )
    else:
        print("Skipping host distribution (only 1 unique value)")

    # 2. HN subtype distribution
    if has_subtype:
        plot_distribution_by_split(
            metadata_distributions,
            'hn_subtype',
            f'HN Subtype Distribution by Split ({bundle_name})',
            'Number of Isolates',
            plots_dir / 'subtype_distribution.png',
            top_n=top_n,
            split_sizes=split_sizes
        )
    else:
        print("Skipping subtype distribution (only 1 unique value)")

    # 3. Geographic location distribution (geo_location_clean)
    if has_geo:
        plot_distribution_by_split(
            metadata_distributions,
            'geo_location_clean',
            f'Geographic Location Distribution by Split ({bundle_name})',
            'Number of Isolates',
            plots_dir / 'geo_location_distribution.png',
            top_n=top_n,
            split_sizes=split_sizes
        )
    else:
        print("Skipping geo_location distribution (insufficient variation)")

    # 4. Passage distribution
    if has_passage:
        plot_distribution_by_split(
            metadata_distributions,
            'passage',
            f'Passage Distribution by Split ({bundle_name})',
            'Number of Isolates',
            plots_dir / 'passage_distribution.png',
            top_n=top_n,
            split_sizes=split_sizes
        )
    else:
        print("Skipping passage distribution (insufficient variation)")
    
    # 5. Year distribution
    if has_year:
        # Check year range
        all_years = []
        for split in ['train', 'val', 'test']:
            year_dist = metadata_distributions.get(split, {}).get('year', {})
            for year_str, count in year_dist.items():
                if year_str != 'null':
                    try:
                        year = int(year_str)
                        all_years.extend([year] * int(count))
                    except (ValueError, TypeError):
                        pass
        
        if len(all_years) > 0:
            year_range = max(all_years) - min(all_years)
            if year_range > 1:  # More than 1 year difference
                plot_year_distribution_by_split(
                    metadata_distributions,
                    plots_dir / 'year_distribution.png',
                    start_year=start_year,
                    split_sizes=split_sizes,
                )
            else:
                print("Skipping year distribution (very narrow range)")
        else:
            print("Skipping year distribution (no year data)")
    else:
        print("Skipping year distribution (only 1 unique value)")

    # 6. Host × Subtype heatmap
    # Check if both host and subtype have sufficient variation
    if has_host and has_subtype:
        plot_host_subtype_heatmap_by_split(
            dataset_stats_path=dataset_stats_path,
            run_dir=run_dir,
            bundle_name=bundle_name,
            output_path=plots_dir / 'host_subtype_heatmap.png',
            top_hosts=15,
            top_subtypes=15,
            split_sizes=split_sizes,
        )
    else:
        print("Skipping host × subtype heatmap (insufficient variation in host or subtype)")

    # 7. Low-dimensional PCA plots for each interaction mode (pair embeddings)
    # Generates pair_pca_concat.png, pair_pca_diff.png, pair_pca_prod.png,
    # pair_pca_unit_diff.png + pair_interaction_diagnostics.json,
    # plus k-mer PCA plots (kmer_pca_concat.png + kmer_pca_scree.png).
    try:
        cfg = get_virus_config_hydra(bundle_name, config_path=str(project_root / 'conf'))
        virus_name = cfg.virus.virus_name
        data_version = cfg.virus.data_version
        embeddings_dir = project_root / 'data' / 'embeddings' / virus_name / data_version
        embeddings_file = embeddings_dir / 'master_esm2_embeddings.h5'

        if skip_esm_pca_plots:
            print("Skipping ESM-2 pair PCA plots (skip_esm_pca_plots=True)")
        elif embeddings_file.exists():
            plot_pair_interactions(
                run_dir=run_dir,
                embeddings_file=embeddings_file,
                output_dir=plots_dir,
                interactions=["concat", "diff", "prod", "unit_diff"],
                max_per_label_per_split=1000,
                random_state=42,
                function_short_names=get_function_short_name_map_from_config(cfg.virus),
            )
            # Task 8 (optional): single-embedding confounder plots (derived from sequences appearing in pairs).
            # Disabled by default since it can be slow and is not required for the split-overlap sanity check.
            if generate_confounder_plots:
                protein_metadata_csv = project_root / 'data' / 'processed' / virus_name / data_version / 'protein_final.csv'
                plot_sequence_embeddings_by_confounders_from_pairs(
                    run_dir=run_dir,
                    embeddings_file=embeddings_file,
                    protein_metadata_csv=protein_metadata_csv,
                    output_dir=plots_dir,
                    max_pairs_per_label_per_split=1500,
                    max_sequences=5000,
                    random_state=42,
                )
        else:
            print(f"Skipping ESM-2 pair PCA plots (missing embeddings file: {embeddings_file})")

        # K-mer PCA plots: cache lives next to ESM-2 embeddings.
        if skip_kmer_pca_plots:
            print("Skipping k-mer PCA plots (skip_kmer_pca_plots=True)")
        else:
            kmer_caches = list(embeddings_dir.glob('kmer_features_k*.npz'))
            if kmer_caches:
                plot_kmer_pca(
                    run_dir=run_dir,
                    kmer_dir=embeddings_dir,
                    output_dir=plots_dir,
                    virus_config=cfg.virus,
                    k=None,  # autodetect from kmer_features_k*_metadata.json
                    max_per_label_per_split=1000,
                    random_state=42,
                )
            else:
                print(f"Skipping k-mer PCA plots (no kmer_features_k*.npz under {embeddings_dir})")
    except Exception as e:
        print(f"WARNING: skipping pair interaction / k-mer plots ({type(e).__name__}: {e})")

    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Plots saved to: {plots_dir}")


def find_dataset_run_directory(base_dir: Path, bundle_name: str) -> Optional[Path]:
    """
    Find the dataset run directory for a bundle.
    
    Args:
        base_dir: Base directory containing runs/
        bundle_name: Bundle name (e.g., 'flu_2024')
    
    Returns:
        Path to the run directory, or None if not found
    """
    runs_dir = base_dir / 'runs'
    if not runs_dir.exists():
        return None
    
    prefix = f'dataset_{bundle_name}_'
    matching_dirs = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(prefix):
            remaining = d.name[len(prefix):]
            if remaining and remaining[0].isdigit():  # Timestamp starts with year
                matching_dirs.append(d)
    
    if not matching_dirs:
        return None
    
    # Return most recent
    return sorted(matching_dirs, key=lambda x: x.name)[-1]


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Visualize dataset statistics for segment pair classification datasets'
    )
    parser.add_argument(
        '--bundle',
        type=str,
        nargs='+',
        help='Bundle name(s) to visualize (e.g., flu_2024)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all bundles found in the datasets directory'
    )
    parser.add_argument(
        '--config_bundle',
        type=str,
        default=None,
        help='Config bundle for determining virus/data_version (default: infer from first bundle)'
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        default=None,
        help='Explicit dataset run directory (overrides bundle search)'
    )
    parser.add_argument(
        '--confounder_plots',
        action='store_true',
        help='Optional (Task 8): generate single-embedding confounder plots (can be slow). Default: off.'
    )
    parser.add_argument(
        '--skip_esm_pca_plots',
        action='store_true',
        help='Skip pair_pca_{concat,diff,prod,unit_diff}.png + pair_interaction_diagnostics.json.'
    )
    parser.add_argument(
        '--skip_kmer_pca_plots',
        action='store_true',
        help='Skip kmer_pca_concat.png + kmer_pca_scree.png.'
    )
    """
    python src/analysis/visualize_dataset_stats.py --bundle flu_ha_na_5ks --dataset_dir ./data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_20260119_144322
    """

    args = parser.parse_args()

    # Determine bundles to process
    if args.all:
        # Find all bundles in the datasets directory
        # This requires config to know virus/data_version
        if not args.config_bundle:
            raise ValueError("--config_bundle required when using --all")
        config_path = str(project_root / 'conf')
        config = get_virus_config_hydra(args.config_bundle, config_path=config_path)
        base_dir = project_root / 'data' / 'datasets' / config.virus.virus_name / config.virus.data_version
        runs_dir = base_dir / 'runs'
        if not runs_dir.exists():
            print(f"No runs directory found: {runs_dir}")
            return

        # Extract bundle names from directory names
        bundles = set()
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith('dataset_'):
                # Extract bundle name: dataset_{bundle}_{timestamp}
                parts = d.name.split('_')
                if len(parts) >= 3:
                    # Reconstruct bundle name (may have underscores)
                    # Find where timestamp starts (first part that's all digits)
                    bundle_parts = []
                    for i, part in enumerate(parts[1:], 1):  # Skip 'dataset'
                        if part.isdigit() and len(part) == 8:  # YYYYMMDD
                            break
                        bundle_parts.append(part)
                    if bundle_parts:
                        bundles.add('_'.join(bundle_parts))

        bundle_names = sorted(bundles)
        print(f"Found {len(bundle_names)} bundles: {bundle_names}")
    elif args.bundle:
        bundle_names = args.bundle
    else:
        parser.error("Must provide --bundle or --all")

    # Determine config if not provided
    if not args.config_bundle:
        args.config_bundle = bundle_names[0]

    # Load config to get virus/data_version
    config_path = str(project_root / 'conf')
    config = get_virus_config_hydra(args.config_bundle, config_path=config_path)
    virus_name = config.virus.virus_name
    data_version = config.virus.data_version

    base_dir = project_root / 'data' / 'datasets' / virus_name / data_version

    # Process each bundle
    for bundle_name in bundle_names:
        print(f"\n{'='*80}")
        print(f"Processing bundle: {bundle_name}")
        print(f"{'='*80}")

        if args.dataset_dir:
            run_dir = args.dataset_dir
        else:
            run_dir = find_dataset_run_directory(base_dir, bundle_name)

        if run_dir is None:
            print(f"WARNING: No run directory found for bundle '{bundle_name}'")
            continue

        dataset_stats_path = run_dir / 'dataset_stats.json'
        if not dataset_stats_path.exists():
            print(f"WARNING: dataset_stats.json not found in {run_dir}")
            continue

        # Determine output directories
        output_dir_dataset = run_dir

        visualize_dataset_stats(
            dataset_stats_path=dataset_stats_path,
            bundle_name=bundle_name,
            output_dir_dataset=output_dir_dataset,
            run_dir=run_dir,
            project_root=project_root,
            generate_confounder_plots=args.confounder_plots,
            skip_esm_pca_plots=args.skip_esm_pca_plots,
            skip_kmer_pca_plots=args.skip_kmer_pca_plots,
        )


if __name__ == '__main__':
    main()
