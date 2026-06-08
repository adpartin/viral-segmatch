"""Shared k-mer feature + model machinery for the cluster-disjoint CV experiments.

Extracted (behavior-preserving) from `cluster_disjoint_cv_experiment.py` so the
m-sweep / fixed-N driver AND the regime-breakdown CV
(`cluster_disjoint_regime_cv.py`) import one feature builder + model fit from a
neutral module, rather than one script importing another's CLI module.

`fit_eval` is byte-for-byte the prior behavior (used by the m-sweep). The only
new surface is `fit_predict`, which returns the raw `(y_pred, y_prob)` that the
out-of-fold regime breakdown needs; `fit_eval` now calls it so the two share one
classifier construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score, average_precision_score, matthews_corrcoef

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.clustering_utils import compute_seq_hash  # noqa: E402

_KMER_DIR = PROJ / 'data/embeddings/flu/July_2025'
_PROTEIN_FINAL = PROJ / 'data/processed/flu/July_2025/protein_final.parquet'
# hash columns on the pair universe (load_pair_universe) per alphabet.
_HASH = {'aa': ('seq_hash_a', 'seq_hash_b'), 'nt_cds': ('dna_hash_a', 'dna_hash_b')}


def build_hash_to_row(kmer_k: int, functions_full: list[str]) -> dict:
    """{seq_hash -> k-mer matrix row} for aa, via the (assembly_id, brc_fea_id) join."""
    idx = pd.read_parquet(_KMER_DIR / f'kmer_features_aa_k{kmer_k}_index.parquet',
                          columns=['assembly_id', 'brc_fea_id', 'function', 'row'])
    idx = idx[idx['function'].isin(functions_full)].copy()
    prot = pd.read_parquet(_PROTEIN_FINAL, columns=['assembly_id', 'brc_fea_id', 'prot_seq', 'function'])
    prot = prot[prot['function'].isin(functions_full)].copy()
    prot['seq_hash'] = prot['prot_seq'].map(compute_seq_hash)
    for df in (idx, prot):
        df['assembly_id'] = df['assembly_id'].astype(str)
        df['brc_fea_id'] = df['brc_fea_id'].astype(str)
    m = prot.merge(idx[['assembly_id', 'brc_fea_id', 'row']], on=['assembly_id', 'brc_fea_id'], how='inner')
    return dict(zip(m['seq_hash'], m['row'].astype(int)))


def _interaction(ea: np.ndarray, eb: np.ndarray, interaction: str) -> np.ndarray:
    if interaction == 'concat':
        return np.hstack([ea, eb])
    if interaction == 'unit_diff':
        d = np.abs(ea - eb)
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
    raise ValueError(f"interaction must be 'concat' or 'unit_diff'; got {interaction!r}")


def pair_features(pairs: pd.DataFrame, ha: str, hb: str, hash_to_row: dict,
                  matrix: sparse.csr_matrix, interaction: str):
    """(X, y) for a labeled pair set; drops pairs missing a k-mer row."""
    ra = pairs[ha].map(hash_to_row)
    rb = pairs[hb].map(hash_to_row)
    valid = ra.notna() & rb.notna()
    ea = matrix[ra[valid].astype(int).to_numpy()].toarray().astype(np.float32)
    eb = matrix[rb[valid].astype(int).to_numpy()].toarray().astype(np.float32)
    return _interaction(ea, eb, interaction), pairs.loc[valid, 'label'].to_numpy()


def _labeled(pos: pd.DataFrame, neg: pd.DataFrame, ha: str, hb: str) -> pd.DataFrame:
    p = pos[[ha, hb]].copy(); p['label'] = 1
    n = neg[[ha, hb]].copy(); n['label'] = 0
    return pd.concat([p, n], ignore_index=True)


def _build_clf(model: str, n_train: int, seed: int):
    """Construct (but do not fit) the classifier for one model name.

    Shared by `fit_predict` / `fit_eval` so the m-sweep and the regime CV use
    identical model configs. `n_train` gates the MLP's early-stopping (the 10%
    val set is too noisy at small N).
    """
    if model == 'lgbm':
        from lightgbm import LGBMClassifier
        # n_jobs=-1 is pathological on many-core boxes (317s vs 7s here); colsample
        # subsampling is the speed lever on the 8000-dim k-mer features.
        return LGBMClassifier(n_estimators=100, min_child_samples=5, colsample_bytree=0.3,
                              random_state=seed, verbose=-1, n_jobs=16)
    if model == 'knn1':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=1, metric='cosine')
    if model == 'mlp':
        from sklearn.neural_network import MLPClassifier
        # early-stop only when abundant; the 10% val set is too noisy at small N
        # (tanked F1 at N~390), and MLP isn't the bottleneck after the LGBM fix.
        return MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                             early_stopping=(n_train > 3000), n_iter_no_change=8,
                             alpha=1e-3, random_state=seed)
    raise ValueError(f"unknown model {model!r}")


def fit_predict(model: str, x_tr, y_tr, x_te, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit one model; return (y_pred, y_prob) on the test fold.

    `y_prob` is the positive-class score; knn1's predict_proba is hard 0/1
    (degenerate but valid). This is the raw-prediction primitive the OOF
    regime breakdown collects.
    """
    clf = _build_clf(model, len(x_tr), seed)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    proba = clf.predict_proba(x_te)[:, 1] if hasattr(clf, 'predict_proba') else pred.astype(float)
    return pred, proba


def fit_eval(model: str, x_tr, y_tr, x_te, y_te, seed: int) -> dict:
    """Fit one model; return {auc_pr, f1_macro, mcc} on the test fold."""
    pred, proba = fit_predict(model, x_tr, y_tr, x_te, seed)
    return {
        'auc_pr': float(average_precision_score(y_te, proba)),
        'f1_macro': float(f1_score(y_te, pred, average='macro')),
        'mcc': float(matthews_corrcoef(y_te, pred)),
    }
