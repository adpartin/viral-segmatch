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
        # n_jobs=-1 is pathological on many-core boxes (317s vs 7s here); colsample
        # subsampling is the speed lever on the 8000-dim k-mer features.
        clf = LGBMClassifier(n_estimators=100, min_child_samples=5, colsample_bytree=0.3,
                             random_state=seed, verbose=-1, n_jobs=16)
    elif model == 'knn1':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    elif model == 'mlp':
        from sklearn.neural_network import MLPClassifier
        # early-stop only when abundant; the 10% val set is too noisy at small N
        # (tanked F1 at N~390), and MLP isn't the bottleneck after the LGBM fix.
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                            early_stopping=(len(x_tr) > 3000), n_iter_no_change=8,
                            alpha=1e-3, random_state=seed)
    else:
        raise ValueError(f"unknown model {model!r}")
    clf.fit(x_tr, y_tr)
    return float(f1_score(y_te, clf.predict(x_te), average='macro'))


def _labeled(pos: pd.DataFrame, neg: pd.DataFrame, ha: str, hb: str) -> pd.DataFrame:
    p = pos[[ha, hb]].copy(); p['label'] = 1
    n = neg[[ha, hb]].copy(); n['label'] = 0
    return pd.concat([p, n], ignore_index=True)


def run_cv(pairs, slot_a, slot_b, alphabet, threshold, max_per_cc, n_ccs, cc_sizes, *,
           hash_to_row, matrix, interaction, k_folds, models, seed):
    """One (pair, alphabet, t, m): cap-m-per-CC sample -> GroupKFold-by-CC F1-macro.

    `pairs` already carries cc_id (assigned once per t). `n_capped` = #CCs that hit
    the cap (|CC| >= m); the rest contribute their full (<m) size.
    """
    ha, hb = _HASH[alphabet]
    pos = sample_positives(pairs, max_per_cc=max_per_cc, seed=seed)
    n_pos = len(pos)
    n_capped = int((cc_sizes >= max_per_cc).sum())
    if n_pos < k_folds:
        return None

    # CV splitter: GroupKFold keeps whole CCs in one fold (cluster-disjoint). It is
    # deterministic + unshuffled, so f1_std is the fold-to-fold spread of ONE partition
    # (not repeated-CV variance); label balance comes from the per-fold 1:1 negatives
    # (make_negatives), not the splitter.
    # TODO(cv-variants): for seeded/repeated grouped CV, loop
    #   StratifiedGroupKFold(shuffle=True, random_state=seed) over seeds (sklearn has no
    #   RepeatedStratifiedGroupKFold; RepeatedStratifiedKFold is group-UNAWARE -> it would
    #   split CCs across folds and leak). To keep a covariate (e.g. hn_subtype) balanced
    #   across folds, pass it as `y` to StratifiedGroupKFold -- a deliberate stratification,
    #   distinct from the 1:1 label balance which is already created per fold.
    gkf = GroupKFold(n_splits=k_folds)
    per_model = {mdl: [] for mdl in models}
    for train_idx, test_idx in gkf.split(pos, groups=pos['cc_id'].to_numpy()):
        tr_pos, te_pos = pos.iloc[train_idx], pos.iloc[test_idx]
        tr = _labeled(tr_pos, make_negatives(tr_pos, alphabet, seed), ha, hb)
        te = _labeled(te_pos, make_negatives(te_pos, alphabet, seed), ha, hb)
        x_tr, y_tr = pair_features(tr, ha, hb, hash_to_row, matrix, interaction)
        x_te, y_te = pair_features(te, ha, hb, hash_to_row, matrix, interaction)
        scaler = StandardScaler().fit(x_tr)  # fit on train only; fixes MLP on raw counts
        x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)
        for mdl in models:
            per_model[mdl].append(fit_score(mdl, x_tr, y_tr, x_te, y_te, seed))

    rows = []
    for mdl in models:
        s = np.array(per_model[mdl])
        rows.append({'schema_pair': f'{slot_a}-{slot_b}', 'alphabet': alphabet,
                     'threshold': threshold, 'm': int(max_per_cc), 'model': mdl,
                     'n_ccs': int(n_ccs), 'n_capped': n_capped, 'n_pos': n_pos,
                     'n_folds': len(s), 'f1_mean': round(float(s.mean()), 4),
                     'f1_std': round(float(s.std()), 4)})
    print(f"  [{slot_a}-{slot_b} {alphabet} {threshold} m={max_per_cc}] "
          f"N={n_pos} ({n_capped}/{n_ccs} CCs capped)  "
          + "  ".join(f"{mdl}={r['f1_mean']:.3f}" for mdl, r in zip(models, rows)))
    return rows


def plot_m_sweep(df_pair: pd.DataFrame, pair_label, alphabet, models, ms, out_png):
    """2x2 grid: one F1-vs-t panel per model + an N-vs-t panel; m overlaid (viridis)."""
    ts = sorted(df_pair['threshold'].unique(), reverse=True)  # t100 left
    x = np.arange(len(ts))
    cmap = plt.get_cmap('viridis')
    colors = {mv: cmap(i / max(len(ms) - 1, 1)) for i, mv in enumerate(ms)}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    panels = list(models) + ['N']
    for ax, panel in zip(axes.flat, panels):
        for mv in ms:
            s = df_pair[df_pair['m'] == mv]
            if panel == 'N':
                y = [int(s[s['threshold'] == t]['n_pos'].iloc[0]) for t in ts]
                ax.plot(x, y, marker='o', color=colors[mv], label=f'm={mv}')
            else:
                mm = s[s['model'] == panel].set_index('threshold')
                y = [mm.loc[t, 'f1_mean'] for t in ts]
                e = [mm.loc[t, 'f1_std'] for t in ts]
                ax.errorbar(x, y, yerr=e, marker='o', capsize=2, color=colors[mv], label=f'm={mv}')
        ax.set_xticks(x)
        ax.set_xticklabels(ts, fontsize=8)
        ax.grid(linestyle=':', alpha=0.5)
        ax.legend(fontsize=8, title='cap m/CC')
        if panel == 'N':
            ax.set_ylabel('N positives (log)')
            ax.set_yscale('log')
            ax.set_title('dataset size N vs t')
        else:
            ax.set_ylabel('F1-macro (mean +/- std)')
            ax.set_ylim(0.4, 1.02)
            ax.axhline(0.5, linestyle=':', color='#999', linewidth=1)
            ax.set_title(f'{panel}')
    fig.suptitle(f'{pair_label} — {alphabet} — m-sweep (cap m pairs/CC): '
                 f'F1-macro and N vs cluster threshold t', fontsize=12, y=1.0)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
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
    p.add_argument('--max_per_cc', nargs='+', type=int, default=[1, 2, 5, 10],
                   help='cap of pairs sampled per CC (m); swept and overlaid.')
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
        ms = sorted(set(args.max_per_cc))
        for alphabet in args.alphabets:
            pair_rows: list[dict] = []
            for t in args.thresholds:
                pairs = assign_cc(universe, slot_a, slot_b, alphabet, t)  # CC structure: once per t
                n_ccs = int(pairs['cc_id'].nunique())
                cc_sizes = pairs.groupby('cc_id').size()
                for mv in ms:
                    rows = run_cv(pairs, slot_a, slot_b, alphabet, t, mv, n_ccs, cc_sizes,
                                  hash_to_row=hash_to_row, matrix=matrix,
                                  interaction=args.interaction, k_folds=args.k_folds,
                                  models=args.models, seed=args.seed)
                    if rows:
                        pair_rows.extend(rows)
            if pair_rows:
                df_pair = pd.DataFrame(pair_rows)
                out_png = out_dir / f'msweep_{slot_a}_{slot_b}_{alphabet}.png'
                plot_m_sweep(df_pair, pair, alphabet, args.models, ms, out_png)
                print(f"  wrote {out_png}")
                all_rows.extend(pair_rows)

    if all_rows:
        csv = out_dir / 'cluster_disjoint_cv_results.csv'
        pd.DataFrame(all_rows).to_csv(csv, index=False)
        print(f"\nwrote {csv} ({len(all_rows)} rows)")
    print("\nDone.")


if __name__ == '__main__':
    main()
