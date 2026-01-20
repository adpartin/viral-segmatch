"""
Dimensionality reduction utilities for embedding visualization.

Provides wrappers around PCA and UMAP for consistent usage across analysis scripts.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def compute_pca_reduction(
    embeddings: np.ndarray,
    n_components: int = 2,
    return_model: bool = False,
    svd_solver: str = "auto",
    random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[PCA]]:
    """Compute PCA reduction for embeddings.
    
    Args:
        embeddings: (N, D) array of embeddings
        n_components: Number of components (default: 2 for visualization)
        return_model: If True, return fitted PCA model
        svd_solver: sklearn PCA solver (e.g., "auto", "randomized")
        random_state: Random seed used by some solvers (e.g., "randomized")
    
    Returns:
        reduced_embeddings: (N, n_components) array
        pca_model: Optional fitted PCA model (if return_model=True)
    
    Example:
        >>> embeddings_2d, pca = compute_pca_reduction(embeddings, return_model=True)
        >>> variance_explained = pca.explained_variance_ratio_
    """
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    if return_model:
        return reduced_embeddings, pca
    else:
        return reduced_embeddings, None


def compute_umap_reduction(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    return_model: bool = False
    ) -> Tuple[np.ndarray, Optional[object]]:
    """Compute UMAP reduction for embeddings.
    
    Args:
        embeddings: (N, D) array of embeddings
        n_components: Number of components (default: 2)
        n_neighbors: UMAP parameter (default: 15)
        min_dist: UMAP parameter (default: 0.1)
        random_state: Random seed (default: 42)
        return_model: If True, return fitted UMAP model
    
    Returns:
        reduced_embeddings: (N, n_components) array
        umap_model: Optional fitted UMAP model (if return_model=True and UMAP available)
    
    Raises:
        ImportError: If UMAP is not installed
    
    Example:
        >>> embeddings_2d, umap_model = compute_umap_reduction(embeddings, return_model=True)
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP is not installed. Install with: pip install umap-learn"
        )
    
    umap_reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    
    if return_model:
        return reduced_embeddings, umap_reducer
    else:
        return reduced_embeddings, None

