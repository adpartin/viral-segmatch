"""Cluster-disjoint K-fold CV experiment: F1-macro vs cluster threshold t.

Standalone harness (path B) for the score-vs-t curve under cluster-disjoint CV,
on k-mer features. Per (schema_pair, alphabet, t):
  1. build the pair universe -> assign each pair its connected-component (`_cv_sampling.assign_cc`),
  2. sample one pair per CC (cap-k-per-CC with k=1; `sample_positives`),
  3. GroupKFold(groups=cc_id) -> cluster-disjoint folds (no cut needed: capping
     bounds every group, so k=1 is a plain KFold and k>1 keeps each CC whole),
  4. per fold: 1:1 random within-fold negatives, k-mer pair features, fit
     {mlp, lgbm, knn1}, score F1-macro,
  5. plot F1-macro vs t (mean +/- std over folds) with N positives per t.

aa is clean (aa clusters + aa k-mers, seq-deduped). nt is NOT wired: the nt
k-mer cache is contig-level (`genbank_ctg_id`) while the clusters are CDS-level
(`cds_dna_hash`) -- a feature/cluster alphabet mismatch to resolve before running
nt (compute nt_cds k-mers, build nt_ctg clusters, or accept the inconsistency).

Models are lightweight sklearn/lightgbm fits on the k-mer pair features (not the
Stage-4 plumbing). Reuses kmer_utils' matrix/index and compute_seq_hash.

CLI:
    python -m src.analysis.cluster_disjoint_cv_experiment \\
        [--schema_pairs HA-NA] [--alphabets aa] \\
        [--thresholds t100 t099 t098 t097 t096 t095] \\
        [--kmer_k 3] [--interaction concat] [--k_folds 5] \\
        [--models mlp lgbm knn1] [--seed 0] \\
        [--out_dir results/flu/July_2025/runs/cluster_disjoint_cv]

Outputs (under --out_dir):
    scorecurve_{A}_{B}_{alphabet}.png      F1-macro vs t (one line per model)
    cluster_disjoint_cv_results.csv        schema_pair, alphabet, threshold, model,
                                           n_pos, n_folds, f1_mean, f1_std
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe  # noqa: E402
from src.analysis._cv_sampling import assign_cc, sample_positives, make_negatives  # noqa: E402
from src.utils.clustering_utils import compute_seq_hash  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402

_KMER_DIR = PROJ / 'data/embeddings/flu/July_2025'
_PROTEIN_FINAL = PROJ / 'data/processed/flu/July_2025/protein_final.parquet'
_HASH = {'aa': ('seq_hash_a', 'seq_hash_b'), 'nt_cds': ('dna_hash_a', 'dna_hash_b')}
_MODEL_COLOR = {'mlp': '#1f77b4', 'lgbm': '#2ca02c', 'knn1': '#d62728'}


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


def fit_score(model: str, x_tr, y_tr, x_te, y_te, seed: int) -> float:
    if model == 'lgbm':
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=200, min_child_samples=5, random_state=seed,
                             verbose=-1, n_jobs=-1)
    elif model == 'knn1':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    elif model == 'mlp':
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=False,
                            alpha=1e-3, random_state=seed)
    else:
        raise ValueError(f"unknown model {model!r}")
    clf.fit(x_tr, y_tr)
    return float(f1_score(y_te, clf.predict(x_te), average='macro'))


def _labeled(pos: pd.DataFrame, neg: pd.DataFrame, ha: str, hb: str) -> pd.DataFrame:
    p = pos[[ha, hb]].copy(); p['label'] = 1
    n = neg[[ha, hb]].copy(); n['label'] = 0
    return pd.concat([p, n], ignore_index=True)


def run_slice(universe, slot_a, slot_b, alphabet, threshold, *, hash_to_row, matrix,
              interaction, k_folds, models, seed):
    """One (pair, alphabet, t): GroupKFold-by-CC -> per-model F1-macro mean/std."""
    ha, hb = _HASH[alphabet]
    pairs = assign_cc(universe, slot_a, slot_b, alphabet, threshold)
    pos = sample_positives(pairs, max_per_cc=1, seed=seed)
    n_pos = len(pos)
    if n_pos < k_folds:
        print(f"  [{slot_a}-{slot_b} {alphabet} {threshold}] only {n_pos} CCs < k_folds; skipping.")
        return None

    gkf = GroupKFold(n_splits=k_folds)
    per_model = {m: [] for m in models}
    for train_idx, test_idx in gkf.split(pos, groups=pos['cc_id'].to_numpy()):
        tr_pos, te_pos = pos.iloc[train_idx], pos.iloc[test_idx]
        tr = _labeled(tr_pos, make_negatives(tr_pos, alphabet, seed), ha, hb)
        te = _labeled(te_pos, make_negatives(te_pos, alphabet, seed), ha, hb)
        x_tr, y_tr = pair_features(tr, ha, hb, hash_to_row, matrix, interaction)
        x_te, y_te = pair_features(te, ha, hb, hash_to_row, matrix, interaction)
        scaler = StandardScaler().fit(x_tr)  # fit on train only; fixes MLP on raw counts
        x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)
        for m in models:
            per_model[m].append(fit_score(m, x_tr, y_tr, x_te, y_te, seed))

    rows = []
    for m in models:
        s = np.array(per_model[m])
        rows.append({'schema_pair': f'{slot_a}-{slot_b}', 'alphabet': alphabet,
                     'threshold': threshold, 'model': m, 'n_pos': n_pos,
                     'n_folds': len(s), 'f1_mean': round(float(s.mean()), 4),
                     'f1_std': round(float(s.std()), 4)})
    print(f"  [{slot_a}-{slot_b} {alphabet} {threshold}] n_pos={n_pos}  "
          + "  ".join(f"{m}={r['f1_mean']:.3f}+/-{r['f1_std']:.3f}" for m, r in zip(models, rows)))
    return rows


def plot_scorecurve(df_pair: pd.DataFrame, pair_label, alphabet, models, out_png):
    sub = df_pair.sort_values('threshold')
    ts = sorted(sub['threshold'].unique(), reverse=True)  # t100 left
    x = np.arange(len(ts))
    npos = [int(sub[sub['threshold'] == t]['n_pos'].iloc[0]) for t in ts]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in models:
        mm = sub[sub['model'] == m].set_index('threshold')
        y = [mm.loc[t, 'f1_mean'] for t in ts]
        e = [mm.loc[t, 'f1_std'] for t in ts]
        ax.errorbar(x, y, yerr=e, marker='o', capsize=3, linewidth=1.6,
                    color=_MODEL_COLOR.get(m, None), label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(N={n:,})' for t, n in zip(ts, npos)], fontsize=8)
    ax.set_xlabel('cluster threshold t  (N = one-per-CC positives)', fontsize=10)
    ax.set_ylabel('F1-macro (mean +/- std over folds)', fontsize=10)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, linestyle=':', color='#888', linewidth=1, label='chance (0.5)')
    ax.grid(linestyle=':', alpha=0.5)
    ax.set_title(f'{pair_label} — {alphabet} — cluster-disjoint K-fold CV\n'
                 f'one-per-CC positives, GroupKFold-by-CC, k-mer features', fontsize=11)
    ax.legend(loc='lower left', fontsize=9, frameon=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final', default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'))
    p.add_argument('--schema_pairs', nargs='+', default=['HA-NA'])
    p.add_argument('--alphabets', nargs='+', default=['aa'], choices=['aa', 'nt_cds'])
    p.add_argument('--thresholds', nargs='+',
                   default=['t100', 't099', 't098', 't097', 't096', 't095'])
    p.add_argument('--kmer_k', type=int, default=3)
    p.add_argument('--interaction', default='unit_diff', choices=['concat', 'unit_diff'])
    p.add_argument('--k_folds', type=int, default=5)
    p.add_argument('--models', nargs='+', default=['mlp', 'lgbm', 'knn1'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/cluster_disjoint_cv')
    args = p.parse_args()

    if 'nt_cds' in args.alphabets:
        raise NotImplementedError(
            "nt_cds not wired: nt k-mer cache is contig-level (genbank_ctg_id) but the "
            "clusters are CDS-level (cds_dna_hash). Resolve the feature/cluster alphabet "
            "mismatch first (compute nt_cds k-mers, build nt_ctg clusters, or accept it).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s2f = load_function_metadata(PROJ / 'conf/virus/flu.yaml').short_to_function
    matrix = sparse.load_npz(_KMER_DIR / f'kmer_features_aa_k{args.kmer_k}.npz')

    all_rows: list[dict] = []
    for pair in args.schema_pairs:
        slot_a, slot_b = pair.split('-')
        print(f"=== {pair} ===")
        universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
        hash_to_row = build_hash_to_row(args.kmer_k, [s2f[slot_a], s2f[slot_b]])
        for alphabet in args.alphabets:
            pair_rows: list[dict] = []
            for t in args.thresholds:
                rows = run_slice(universe, slot_a, slot_b, alphabet, t,
                                 hash_to_row=hash_to_row, matrix=matrix,
                                 interaction=args.interaction, k_folds=args.k_folds,
                                 models=args.models, seed=args.seed)
                if rows:
                    pair_rows.extend(rows)
            if pair_rows:
                df_pair = pd.DataFrame(pair_rows)
                out_png = out_dir / f'scorecurve_{slot_a}_{slot_b}_{alphabet}.png'
                plot_scorecurve(df_pair, pair, alphabet, args.models, out_png)
                print(f"  wrote {out_png}")
                all_rows.extend(pair_rows)

    if all_rows:
        csv = out_dir / 'cluster_disjoint_cv_results.csv'
        pd.DataFrame(all_rows).to_csv(csv, index=False)
        print(f"\nwrote {csv} ({len(all_rows)} rows)")
    print("\nDone.")


if __name__ == '__main__':
    main()
