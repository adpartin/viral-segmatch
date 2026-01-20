"""
Embedding loading and visualization utilities.

Provides functions for loading embeddings from master cache format,
extracting sequences from pairs, and creating visualizations.
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union
from sklearn.utils import resample

from src.utils.dim_reduction_utils import compute_pca_reduction


def load_embedding_index(embeddings_file: Path) -> Dict[str, int]:
    """Load brc_fea_id -> row index mapping from parquet index.
    
    The master format stores embeddings efficiently:
    - HDF5 file: Contains 'emb' dataset with shape (N, D) where N = num proteins
    - Parquet file: Maps brc_fea_id (string) → row index (int) in the HDF5
    
    This allows O(1) lookup of any protein's embedding by its BRC ID.
    
    Args:
        embeddings_file: Path to master HDF5 cache file
    
    Returns:
        dict: Mapping {brc_fea_id: row_index}
    
    Raises:
        FileNotFoundError: If parquet index file doesn't exist
    
    Example:
        >>> id_to_row = load_embedding_index(Path("data/embeddings/flu/July_2025/master_esm2_embeddings.h5"))
        >>> row_idx = id_to_row["BRC12345"]
    """
    parquet_file = embeddings_file.with_suffix('.parquet')
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet index not found: {parquet_file}")
    
    index_df = pd.read_parquet(parquet_file)
    return dict(zip(index_df['brc_fea_id'], index_df['row']))


def load_embeddings_by_ids(
    brc_ids: List[str],
    embeddings_file: Path,
    id_to_row: Optional[Dict[str, int]] = None
    ) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings for specific brc_fea_ids from master cache.
    
    Args:
        brc_ids: List of brc_fea_ids to load
        embeddings_file: Path to master HDF5 cache file
        id_to_row: Optional pre-loaded mapping (for efficiency if loading multiple times)
    
    Returns:
        embeddings: (N, D) array of embeddings where N = number of valid IDs found
        valid_ids: List of brc_ids that were successfully loaded
    
    Example:
        >>> embeddings, valid_ids = load_embeddings_by_ids(
        ...     ["BRC1", "BRC2", "BRC3"],
        ...     Path("data/embeddings/flu/July_2025/master_esm2_embeddings.h5")
        ... )
        >>> print(f"Loaded {len(valid_ids)} embeddings of shape {embeddings.shape}")
    """
    # Load index if not provided
    if id_to_row is None:
        id_to_row = load_embedding_index(embeddings_file)
    
    embeddings = []
    valid_ids = []
    
    with h5py.File(embeddings_file, 'r') as f:
        if 'emb' not in f:
            raise ValueError(f"Invalid embeddings file format: {embeddings_file}. Master format required.")
        
        emb_data = f['emb']
        for brc_id in brc_ids:
            if brc_id in id_to_row:
                row_idx = id_to_row[brc_id]
                embeddings.append(emb_data[row_idx])
                valid_ids.append(brc_id)
    
    if len(embeddings) == 0:
        return np.array([]), []
    
    return np.array(embeddings), valid_ids


def extract_unique_sequences_from_pairs(
    pairs_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
    """Extract unique sequences from pairs DataFrame.
    
    Extracts unique brc_fea_ids from 'brc_a' and 'brc_b' columns and optionally
    merges with metadata for segment/function information.
    
    Args:
        pairs_df: DataFrame with 'brc_a' and 'brc_b' columns
        metadata_df: Optional protein metadata DataFrame with 'brc_fea_id' column
                    and metadata columns like 'canonical_segment', 'function', etc.
    
    Returns:
        DataFrame with unique brc_fea_ids and metadata columns (if provided)
    
    Example:
        >>> unique_seqs = extract_unique_sequences_from_pairs(train_pairs, protein_metadata)
        >>> print(f"Found {len(unique_seqs)} unique sequences")
    """
    # Extract unique brc_ids from both columns
    brc_a = set(pairs_df['brc_a'].dropna())
    brc_b = set(pairs_df['brc_b'].dropna())
    unique_brc_ids = sorted(list(brc_a.union(brc_b)))
    
    # Create DataFrame with unique IDs
    unique_df = pd.DataFrame({'brc_fea_id': unique_brc_ids})
    
    # Merge with metadata if provided
    if metadata_df is not None and 'brc_fea_id' in metadata_df.columns:
        # Select relevant columns from metadata
        merge_cols = ['brc_fea_id']
        for col in ['canonical_segment', 'function', 'assembly_id']:
            if col in metadata_df.columns:
                merge_cols.append(col)
        
        unique_df = unique_df.merge(
            metadata_df[merge_cols],
            on='brc_fea_id',
            how='left'
        )
    
    return unique_df


def sample_sequences_stratified(
    sequences_df: pd.DataFrame,
    max_per_segment: int = 1000,
    segment_col: str = 'canonical_segment',
    random_state: int = 42
    ) -> pd.DataFrame:
    """Sample sequences with stratified sampling by segment.
    
    Ensures balanced representation across segments, regardless of total number
    of segments. Useful for visualization when dataset has varying numbers of
    segments (e.g., 2 segments vs 8 segments).
    
    Args:
        sequences_df: DataFrame with sequences and segment column
        max_per_segment: Maximum sequences to sample per segment
        segment_col: Column name for segment grouping
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame with balanced representation per segment
    
    Example:
        >>> sampled = sample_sequences_stratified(unique_seqs, max_per_segment=1000)
    """
    if segment_col not in sequences_df.columns:
        # If no segment column, just sample overall
        if len(sequences_df) > max_per_segment:
            return sequences_df.sample(n=max_per_segment, random_state=random_state)
        return sequences_df
    
    sampled_dfs = []
    for segment, group in sequences_df.groupby(segment_col):
        if len(group) > max_per_segment:
            sampled = group.sample(n=max_per_segment, random_state=random_state)
        else:
            sampled = group
        sampled_dfs.append(sampled)
    
    return pd.concat(sampled_dfs, ignore_index=True)


def sample_pairs_stratified(
    pairs_df: pd.DataFrame,
    max_per_label: int = 2500,
    label_col: str = 'label',
    random_state: int = 42
    ) -> pd.DataFrame:
    """Sample pairs with stratified sampling by label.
    
    Ensures balanced representation of positive and negative pairs.
    
    Args:
        pairs_df: DataFrame with pairs and label column
        max_per_label: Maximum pairs to sample per label (0 and 1)
        label_col: Column name for label
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame with balanced representation per label
    
    Example:
        >>> sampled = sample_pairs_stratified(train_pairs, max_per_label=2500)
    """
    if label_col not in pairs_df.columns:
        # If no label column, just sample overall
        if len(pairs_df) > max_per_label * 2:
            return pairs_df.sample(n=max_per_label * 2, random_state=random_state)
        return pairs_df
    
    sampled_dfs = []
    for label in [0, 1]:
        label_pairs = pairs_df[pairs_df[label_col] == label]
        if len(label_pairs) > max_per_label:
            sampled = label_pairs.sample(n=max_per_label, random_state=random_state)
        else:
            sampled = label_pairs
        sampled_dfs.append(sampled)
    
    return pd.concat(sampled_dfs, ignore_index=True)


def create_pair_embeddings_concatenation(
    pairs_df: pd.DataFrame,
    embeddings_file: Path,
    id_to_row: Optional[Dict[str, int]] = None,
    return_valid_mask: bool = False,
    dtype: Optional[np.dtype] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create pair embeddings using concatenation [emb_a, emb_b].
    
    This matches the default model input format (without interaction terms).
    
    Args:
        pairs_df: DataFrame with 'brc_a', 'brc_b', and 'label' columns
        embeddings_file: Path to master HDF5 cache file
        id_to_row: Optional pre-loaded mapping (for efficiency)
        return_valid_mask: If True, also return a boolean mask over the input pairs_df
            indicating which rows had both embeddings available (and therefore were kept).
        dtype: Optional dtype to cast the resulting pair embeddings (e.g., np.float32).
    
    Returns:
        pair_embeddings: (N, 2*D) array where D is embedding dimension
        labels: (N,) array of binary labels
        valid_mask: (len(pairs_df),) boolean array (only if return_valid_mask=True)
    
    Example:
        >>> pair_embs, labels = create_pair_embeddings_concatenation(
        ...     train_pairs,
        ...     Path("data/embeddings/flu/July_2025/master_esm2_embeddings.h5")
        ... )
    """
    # Load index if not provided
    if id_to_row is None:
        id_to_row = load_embedding_index(embeddings_file)

    if len(pairs_df) == 0:
        empty_embs = np.array([])
        empty_labels = np.array([])
        if return_valid_mask:
            return empty_embs, empty_labels, np.array([], dtype=bool)
        return empty_embs, empty_labels

    # Vectorized ID -> row mapping (fast, avoids Python iterrows loop)
    # Note: pandas .map returns NaN for missing keys.
    a_idx = pairs_df['brc_a'].map(id_to_row)
    b_idx = pairs_df['brc_b'].map(id_to_row)
    valid_mask = a_idx.notna() & b_idx.notna()

    if valid_mask.sum() == 0:
        empty_embs = np.array([])
        empty_labels = np.array([])
        if return_valid_mask:
            return empty_embs, empty_labels, valid_mask.to_numpy(dtype=bool)
        return empty_embs, empty_labels

    # Convert to integer row indices
    a_rows = a_idx.loc[valid_mask].astype(int).to_numpy()
    b_rows = b_idx.loc[valid_mask].astype(int).to_numpy()
    labels = pairs_df.loc[valid_mask, 'label'].to_numpy()

    with h5py.File(embeddings_file, 'r') as f:
        if 'emb' not in f:
            raise ValueError(f"Invalid embeddings file format: {embeddings_file}. Master format required.")
        
        emb_data = f['emb']
        # Bulk read; avoids Python loop.
        #
        # NOTE: h5py fancy indexing requires indices to be in increasing order
        # (often strictly increasing/unique).
        # Our pairs arrive in arbitrary order and may contain duplicates; we
        # therefore read unique row indices and then re-expand back to the original order
        # via `inverse` mapping.
        a_unique, a_inverse = np.unique(a_rows, return_inverse=True)
        b_unique, b_inverse = np.unique(b_rows, return_inverse=True)
        emb_a_unique = emb_data[a_unique]
        emb_b_unique = emb_data[b_unique]
        emb_a = emb_a_unique[a_inverse]
        emb_b = emb_b_unique[b_inverse]

    # Concatenate along feature axis: [emb_a, emb_b]
    pair_embeddings = np.concatenate([emb_a, emb_b], axis=1)
    if dtype is not None:
        pair_embeddings = pair_embeddings.astype(dtype, copy=False)

    if return_valid_mask:
        return pair_embeddings, labels, valid_mask.to_numpy(dtype=bool)
    return pair_embeddings, labels


def plot_embeddings_by_category(
    reduced_embeddings: np.ndarray,
    categories: pd.Series,
    category_colors: Optional[Dict] = None,
    title: str = "Embeddings",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_plot: bool = False
    ) -> None:
    """Plot reduced embeddings colored by category.
    
    Generic plotting function for any categorical coloring (segment, function, label, etc.).
    
    Args:
        reduced_embeddings: (N, 2) array from PCA/UMAP
        categories: Series with category labels (must match length of embeddings)
        category_colors: Optional dict mapping category -> color
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
        figsize: Figure size
        show_plot: If True, display plot (default: False)
    
    Example:
        >>> plot_embeddings_by_category(
        ...     pca_embeddings,
        ...     sequences_df['canonical_segment'],
        ...     title="PCA by Segment",
        ...     save_path=Path("pca_by_segment.png")
        ... )
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get unique categories
    unique_categories = sorted(categories.unique())
    
    # Generate colors if not provided
    if category_colors is None:
        cmap = plt.cm.tab10
        category_colors = {
            cat: cmap(i / max(len(unique_categories), 10))
            for i, cat in enumerate(unique_categories)
        }
    
    # Plot each category
    for category in unique_categories:
        mask = categories == category
        ax.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            c=[category_colors[category]],
            label=str(category),
            alpha=0.7,
            s=50
        )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pair_embeddings_by_label(
    reduced_pair_embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Pair Embeddings by Label",
    xlabel: str = "PC1",
    ylabel: str = "PC2",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_plot: bool = False
    ) -> None:
    """Plot pair embeddings colored by positive/negative label.
    
    Args:
        reduced_pair_embeddings: (N, 2) array from PCA
        labels: (N,) array of binary labels (0=negative, 1=positive)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
        figsize: Figure size
        show_plot: If True, display plot (default: False)
    
    Example:
        >>> plot_pair_embeddings_by_label(
        ...     pca_pairs,
        ...     pair_labels,
        ...     title="Train Pairs by Label",
        ...     save_path=Path("train_pairs_pca.png")
        ... )
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot negative pairs (label=0)
    neg_mask = labels == 0
    if neg_mask.sum() > 0:
        ax.scatter(
            reduced_pair_embeddings[neg_mask, 0],
            reduced_pair_embeddings[neg_mask, 1],
            c='#E74C3C',  # Red
            label=f'Negative (n={neg_mask.sum()})',
            alpha=0.7,
            s=50
        )
    
    # Plot positive pairs (label=1)
    pos_mask = labels == 1
    if pos_mask.sum() > 0:
        ax.scatter(
            reduced_pair_embeddings[pos_mask, 0],
            reduced_pair_embeddings[pos_mask, 1],
            c='#3498DB',  # Blue
            label=f'Positive (n={pos_mask.sum()})',
            alpha=0.7,
            s=50
        )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

