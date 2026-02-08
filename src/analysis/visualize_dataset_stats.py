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

from src.utils.config_hydra import get_virus_config_hydra
from src.utils.dim_reduction_utils import compute_pca_reduction
from src.utils.plot_config import SPLIT_COLORS, SPLIT_MARKERS, LABEL_SCATTER_STYLES, apply_default_style
from src.utils.embedding_utils import (
    create_pair_embeddings_concatenation,
    extract_unique_sequences_from_pairs,
    load_embedding_index,
    load_embeddings_by_ids,
    plot_embeddings_by_category,
    sample_pairs_stratified,
)


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


def _plot_pair_embedding_splits_2d(
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
    ) -> None:
    """Scatter plot of 2D embeddings, colored by split and styled by label (pos/neg)."""
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
    def _pick_sparse_corner(points_2d: np.ndarray, used: set[str], region_frac: float = 0.25) -> str:
        """Pick a plot corner with few points, to minimize legend/text overlap.

        We use a simple quadrant density heuristic on normalized coordinates.
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
        # Place Filters textbox in a sparse corner distinct from legends.
        filters_loc = _pick_sparse_corner(reduced_2d, used_locs)
        used_locs.add(filters_loc)

        # Map legend-like loc strings to axes coordinates.
        loc_to_xy = {
            'upper left': (0.02, 0.98, 'left', 'top'),
            'upper right': (0.98, 0.98, 'right', 'top'),
            'lower left': (0.02, 0.02, 'left', 'bottom'),
            'lower right': (0.98, 0.02, 'right', 'bottom'),
        }
        x0, y0, ha, va = loc_to_xy.get(filters_loc, (0.98, 0.98, 'right', 'top'))
        ax.text(
            x0, y0,
            f"Filters:\n{filters_text}",
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
        )
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


def _format_filters_for_plot(filters_applied: Optional[dict]) -> str:
    """Format dataset filters into a compact string for plot title/textbox."""
    if not filters_applied:
        return ""

    parts: list[str] = []
    for k, v in filters_applied.items():
        if v is None:
            continue

        # Best-effort compact formatting for near-future multi-valued/range filters.
        if isinstance(v, (list, tuple, set)):
            vv = list(v)
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

        parts.append(f"{k}={v}")

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
    slot_norm, pre_mlp) require trained model weights and are NOT visualized
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
    filters_text = _format_filters_for_plot(filters_applied)

    # ── 1. Load and sample pairs (once for all interactions) ──────────────
    sampled = []
    for split in splits:
        df = _load_pairs_minimal(run_dir, split)
        if df is None or len(df) == 0:
            continue
        if 'label' in df.columns:
            df = df.copy()
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df[df['label'].isin([0, 1])]
        df_s = sample_pairs_stratified(
            df,
            max_per_label=max_per_label_per_split,
            label_col='label',
            random_state=random_state,
        )
        df_s = df_s.assign(split=split)
        sampled.append(df_s)

    if not sampled:
        print(f"⚠️  plot_pair_interactions: no pair CSVs found in {run_dir}")
        return

    pairs = pd.concat(sampled, ignore_index=True)

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
        print("⚠️  plot_pair_interactions: no valid pair embeddings (missing IDs?)")
        return

    # Align pairs to valid mask
    n_in = len(pairs)
    pairs = pairs.loc[valid_mask].reset_index(drop=True)
    dropped = n_in - len(pairs)
    if dropped > 0:
        print(f"ℹ️  plot_pair_interactions: dropped {dropped:,}/{n_in:,} sampled pairs (missing embeddings)")
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
            print(f"⚠️  Skipping interaction '{spec}': {e}")
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
        _plot_pair_embedding_splits_2d(
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
        print(f"\n🧾 Saved interaction diagnostics: {diag_path}")
    except Exception as e:
        print(f"⚠️  Failed to save diagnostics JSON: {e}")


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
        print(f"⚠️  Skipping split-overlap plot: no pair CSVs found in {run_dir}")
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
        print("⚠️  Skipping split-overlap plot: could not create any pair embeddings (missing embeddings?)")
        return

    # Align `pairs` to the returned embedding rows (critical when some IDs are
    # missing in embedding index)
    n_in = len(pairs)
    pairs = pairs.loc[valid_mask].reset_index(drop=True)
    dropped = n_in - len(pairs)
    if dropped > 0:
        print(f"ℹ️  Split-overlap plot: dropped {dropped:,}/{n_in:,} sampled pairs due to missing embeddings")
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
        print(f"\n🧾 Saved pair embedding ordering diagnostics: {diag_path}")
    except Exception as e:
        print(f"⚠️  Failed to save pair embedding ordering diagnostics JSON ({type(e).__name__}: {e})")

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
        print(f"✅ Saved: {out_path}")

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

    _plot_pair_embedding_splits_2d(
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
        _plot_pair_embedding_splits_2d(
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
        print(f"ℹ️  UMAP not available ({e}). Saved PCA plot only: {pca_path}")
    except Exception as e:
        print(f"⚠️  UMAP failed ({type(e).__name__}: {e}). Saved PCA plot only: {pca_path}")


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
        print(f"⏭️  Skipping confounder embedding plots (missing protein metadata: {protein_metadata_csv})")
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

    # Load embeddings
    id_to_row = load_embedding_index(embeddings_file)
    embeddings, valid_ids = load_embeddings_by_ids(
        seqs_for_plot['brc_fea_id'].tolist(),
        embeddings_file,
        id_to_row=id_to_row,
    )
    if len(embeddings) == 0:
        return
    seqs_for_plot = seqs_for_plot[seqs_for_plot['brc_fea_id'].isin(valid_ids)].reset_index(drop=True)

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


def plot_distribution_by_split(
    metadata_distributions: dict,
    metadata_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    top_n: int = 30
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
        
        # Calculate total count before filtering to top_n
        total_count = series.sum()
        
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
        
        # Title showing displayed count out of total
        if displayed_count < total_count:
            ax.set_title(f'{split.capitalize()} Split (showing {displayed_count:,} of {total_count:,})', 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{split.capitalize()} Split (n={total_count:,})', 
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
    start_year: int = 2000
    ) -> None:
    """
    Plot year distribution as histograms across train/val/test splits.
    
    Args:
        metadata_distributions: Dict with 'train', 'val', 'test' keys
        output_path: Path to save the plot
        start_year: Minimum year to display
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
        
        # Title showing displayed count out of total
        if displayed_count < total_count:
            ax.set_title(f'{split.capitalize()} Split (showing {displayed_count:,} of {total_count:,})', 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{split.capitalize()} Split (n={total_count:,})', 
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
        print(f"⚠️  Warning: Error computing crosstab from {pair_csv_path}: {e}")
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
    top_subtypes: int = 15
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
        print("⏭️  Skipping host×subtype heatmap: could not load isolate metadata (missing/unreadable isolate_metadata.csv)")
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
        ax.set_title(f'{split.capitalize()} Split (n={crosstab_filtered.sum().sum():,} isolates)', 
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
    ) -> None:
    """
    Generate visualization plots for a dataset.
    
    Args:
        dataset_stats_path: Path to dataset_stats.json file
        bundle_name: Bundle name (for plot titles)
        output_dir_dataset: Output directory in dataset run folder (optional)
    """
    # Load dataset statistics
    if not dataset_stats_path.exists():
        raise FileNotFoundError(f"Dataset stats file not found: {dataset_stats_path}")
    
    with open(dataset_stats_path, 'r') as f:
        stats = json.load(f)
    
    metadata_distributions = stats.get('metadata_distributions', {})
    if not metadata_distributions:
        print(f"⚠️  Warning: No metadata_distributions found in {dataset_stats_path}")
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

    # 1. Host distribution
    if has_host:
        plot_distribution_by_split(
            metadata_distributions,
            'host',
            f'Host Distribution by Split ({bundle_name})',
            'Number of Isolates',
            plots_dir / 'host_distribution.png',
            top_n=top_n
        )
    else:
        print("⏭️  Skipping host distribution (only 1 unique value)")

    # 2. HN subtype distribution
    if has_subtype:
        plot_distribution_by_split(
            metadata_distributions,
            'hn_subtype',
            f'HN Subtype Distribution by Split ({bundle_name})',
            'Number of Isolates',
            plots_dir / 'subtype_distribution.png',
            top_n=top_n
        )
    else:
        print("⏭️  Skipping subtype distribution (only 1 unique value)")

    # 3. Geographic location distribution (geo_location_clean)
    if has_geo:
        plot_distribution_by_split(
            metadata_distributions,
            'geo_location_clean',
            f'Geographic Location Distribution by Split ({bundle_name})',
            'Number of Isolates',
            plots_dir / 'geo_location_distribution.png',
            top_n=top_n
        )
    else:
        print("⏭️  Skipping geo_location distribution (insufficient variation)")

    # 4. Passage distribution
    if has_passage:
        plot_distribution_by_split(
            metadata_distributions,
            'passage',
            f'Passage Distribution by Split ({bundle_name})',
            'Number of Isolates',
            plots_dir / 'passage_distribution.png',
            top_n=top_n
        )
    else:
        print("⏭️  Skipping passage distribution (insufficient variation)")
    
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
                    start_year=start_year
                )
            else:
                print("⏭️  Skipping year distribution (very narrow range)")
        else:
            print("⏭️  Skipping year distribution (no year data)")
    else:
        print("⏭️  Skipping year distribution (only 1 unique value)")

    # 6. Host × Subtype heatmap
    # Check if both host and subtype have sufficient variation
    if has_host and has_subtype:
        plot_host_subtype_heatmap_by_split(
            dataset_stats_path=dataset_stats_path,
            run_dir=run_dir,
            bundle_name=bundle_name,
            output_path=plots_dir / 'host_subtype_heatmap.png',
            top_hosts=15,
            top_subtypes=15
        )
    else:
        print("⏭️  Skipping host × subtype heatmap (insufficient variation in host or subtype)")

    # 7. Low-dimensional PCA plots for each interaction mode (pair embeddings)
    # Generates pair_pca_concat.png, pair_pca_diff.png, pair_pca_prod.png,
    # pair_pca_unit_diff.png + pair_interaction_diagnostics.json.
    try:
        cfg = get_virus_config_hydra(bundle_name, config_path=str(project_root / 'conf'))
        virus_name = cfg.virus.virus_name
        data_version = cfg.virus.data_version
        embeddings_file = project_root / 'data' / 'embeddings' / virus_name / data_version / 'master_esm2_embeddings.h5'
        if embeddings_file.exists():
            plot_pair_interactions(
                run_dir=run_dir,
                embeddings_file=embeddings_file,
                output_dir=plots_dir,
                interactions=["concat", "diff", "prod", "unit_diff"],
                max_per_label_per_split=1000,
                random_state=42,
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
            print(f"⏭️  Skipping pair interaction plots (missing embeddings file: {embeddings_file})")
    except Exception as e:
        print(f"⏭️  Skipping pair interaction plots (could not resolve embeddings/config): {e}")
 
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
            print(f"⚠️  Warning: No run directory found for bundle '{bundle_name}'")
            continue

        dataset_stats_path = run_dir / 'dataset_stats.json'
        if not dataset_stats_path.exists():
            print(f"⚠️  Warning: dataset_stats.json not found in {run_dir}")
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
        )


if __name__ == '__main__':
    main()
