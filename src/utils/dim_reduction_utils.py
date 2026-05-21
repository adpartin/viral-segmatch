"""
Dimensionality reduction utilities for embedding visualization.

Provides wrappers around PCA, TruncatedSVD, and UMAP for consistent usage
across analysis scripts.

**Choosing PCA vs TruncatedSVD**: both reduce dimensionality via SVD, but
they treat the input differently and the right choice depends on the
feature type. The rule used throughout this codebase:

- **Use `compute_pca_reduction`** for **dense, signed, real-valued**
  embeddings — e.g. ESM-2 (1280-dim, learned biology), trained model
  activations, distance metrics. PCA centers the data (X − mean) before
  SVD, which is the canonical reduction for these inputs.

- **Use `compute_truncated_svd_reduction`** for **sparse, non-negative,
  count-like** features — e.g. k-mer count vectors (4096-dim nt k=6,
  8000-dim aa k=3), TF-IDF, bag-of-words. TruncatedSVD does *not*
  center; it operates directly on the raw counts. Centering would
  destroy the non-negativity that makes these features meaningful and
  would densify the sparse representation. This is the canonical
  reduction used in LSI/LSA for text — k-mer features are mathematically
  the same shape.

Both routines support a `return_model=True` flag to expose the fitted
sklearn estimator for downstream inspection (e.g. explained variance).
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import PCA, TruncatedSVD

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


def compute_truncated_svd_reduction(
    features,
    n_components: int = 2,
    return_model: bool = False,
    algorithm: str = "randomized",
    random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[TruncatedSVD]]:
    """Reduce sparse / count-like features with TruncatedSVD.

    Mirror of `compute_pca_reduction` but using `sklearn.decomposition.TruncatedSVD`
    (no mean-centering, accepts sparse `X` directly). This is the
    appropriate reduction for non-negative count features like k-mer
    counts, TF-IDF, etc. — see the module docstring for the rule.

    Args:
        features: (N, D) array or sparse matrix. Sparse `X` is kept sparse
            internally by TruncatedSVD; dense `X` is also accepted.
        n_components: Number of components (default: 2).
        return_model: If True, return the fitted TruncatedSVD estimator
            (exposes `explained_variance_ratio_`, `components_`, …).
        algorithm: TruncatedSVD solver — "randomized" (default, fast) or
            "arpack" (deterministic for very small problems).
        random_state: Seed for the randomized solver.

    Returns:
        reduced: (N, n_components) dense ndarray.
        model: Optional fitted TruncatedSVD (when `return_model=True`).

    Example:
        >>> reduced, svd = compute_truncated_svd_reduction(
        ...     kmer_matrix, n_components=50, return_model=True
        ... )
        >>> variance_kept = svd.explained_variance_ratio_.sum()
    """
    svd = TruncatedSVD(
        n_components=n_components, algorithm=algorithm, random_state=random_state,
    )
    reduced = svd.fit_transform(features)
    if return_model:
        return reduced, svd
    return reduced, None


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

