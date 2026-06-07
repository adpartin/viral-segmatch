"""Cluster-disjoint K-fold CV experiment: score (AUC-PR / F1-macro / MCC) vs threshold t.

Standalone harness (not the Stage-3/4 production pipeline) for the score-vs-t curve
under cluster-disjoint CV on k-mer features. Two experiment modes share the fold
machinery (GroupKFold by `atom_id`, 1:1 within-fold negatives, per-fold metrics):

  --strategy natural  (Exp 1-2, m-sweep): atom == bipartite CC; cap `m` pairs per CC
      (`--max_per_cc` swept); at low t the per-CC cap also bounds the mega-CC so the
      folds stay balanced. Plots `msweep_{A}_{B}_{alphabet}.png`.
  --strategy cut      (Exp 3, fixed-N): the mega-CC is edge-min-cut into atoms
      <= ~1/k_folds (`_cv_sampling.assign_atoms(strategy='cut')`); sample `m` pairs
      per CLUSTER PAIR (`--max_per_cluster_pair`) and subsample to a fixed N across t
      (`--fixed_n auto` = the floor over t). Holds N constant so the curve isolates
      the t-effect; the residual confound (dropped straddling pairs) is reported as
      `dropped_frac`. Plots `scorecurve_fixedN_{A}_{B}_{alphabet}.png`.

aa is clean (aa clusters + aa k-mers, seq-deduped). nt is NOT wired: the nt k-mer
cache is contig-level (`genbank_ctg_id`) while the clusters are CDS-level
(`cds_dna_hash`) -- resolve that mismatch before running nt.

CLI:
    python -m src.analysis.cluster_disjoint_cv_experiment \\
        [--schema_pairs HA-NA] [--alphabets aa] \\
        [--thresholds t100 t099 t098 t097 t096 t095] \\
        [--strategy natural|cut] [--cut_method spectral|kl] [--drift_pp 0.05] \\
        [--max_per_cc 1 2 5 10] [--max_per_cluster_pair 1] [--fixed_n auto] \\
        [--kmer_k 3] [--interaction unit_diff] [--k_folds 5] \\
        [--models mlp lgbm knn1] [--seed 0] [--verify] \\
        [--out_dir results/flu/July_2025/runs/cluster_disjoint_cv]

Outputs (under --out_dir):
    msweep_{A}_{B}_{alphabet}.png            (natural) F1/AUC-PR/MCC + N vs t, m overlaid
    cluster_disjoint_cv_results.csv          (natural)
    scorecurve_fixedN_{A}_{B}_{alphabet}.png (cut) per-metric + cut diagnostics vs t
    cluster_disjoint_cv_fixedN_results.csv   (cut)
  Result rows: schema_pair, alphabet, threshold, model, m, strategy, sample_unit,
    cut_method, fixed_n, n_cluster_pairs, n_atoms, largest_atom_frac, dropped_frac,
    n_capped, n_pos, n_folds, {auc_pr,f1_macro,mcc}_{mean,std}.
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
from sklearn.metrics import f1_score, average_precision_score, matthews_corrcoef
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe  # noqa: E402
from src.analysis._cv_sampling import assign_atoms, sample_positives, make_negatives  # noqa: E402
from src.utils.clustering_utils import compute_seq_hash  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402

_KMER_DIR = PROJ / 'data/embeddings/flu/July_2025'
_PROTEIN_FINAL = PROJ / 'data/processed/flu/July_2025/protein_final.parquet'
_HASH = {'aa': ('seq_hash_a', 'seq_hash_b'), 'nt_cds': ('dna_hash_a', 'dna_hash_b')}
_MODEL_COLOR = {'mlp': '#1f77b4', 'lgbm': '#2ca02c', 'knn1': '#d62728'}
_METRICS = ('auc_pr', 'f1_macro', 'mcc')
_METRIC_LABEL = {'auc_pr': 'AUC-PR', 'f1_macro': 'F1-macro', 'mcc': 'MCC'}


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


def fit_eval(model: str, x_tr, y_tr, x_te, y_te, seed: int) -> dict:
    """Fit one model; return {auc_pr, f1_macro, mcc} on the test fold."""
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
    pred = clf.predict(x_te)
    # AUC-PR needs a positive-class score; knn1's predict_proba is hard 0/1 (degenerate
    # AUC-PR but valid) -- drop knn1 from the cut/fixed-N path where AUC-PR matters.
    proba = clf.predict_proba(x_te)[:, 1] if hasattr(clf, 'predict_proba') else pred.astype(float)
    return {
        'auc_pr': float(average_precision_score(y_te, proba)),
        'f1_macro': float(f1_score(y_te, pred, average='macro')),
        'mcc': float(matthews_corrcoef(y_te, pred)),
    }


def _labeled(pos: pd.DataFrame, neg: pd.DataFrame, ha: str, hb: str) -> pd.DataFrame:
    p = pos[[ha, hb]].copy(); p['label'] = 1
    n = neg[[ha, hb]].copy(); n['label'] = 0
    return pd.concat([p, n], ignore_index=True)


def _assert_fold_disjoint(tr_pos: pd.DataFrame, te_pos: pd.DataFrame) -> None:
    """Train/test must share no cluster on either slot (the 2D-CD guarantee)."""
    for col in ('node_a', 'node_b'):
        overlap = set(tr_pos[col]) & set(te_pos[col])
        if overlap:
            raise AssertionError(
                f"cluster-disjoint fold violation: {len(overlap)} shared {col} "
                f"cluster(s) between train and test (e.g. {next(iter(overlap))})")


def run_cv(pos, slot_a, slot_b, alphabet, threshold, *, hash_to_row, matrix,
           interaction, k_folds, models, seed, meta, tag='', verify=False):
    """GroupKFold-by-atom CV on a PRE-SAMPLED positive set; per-metric mean±std.

    `pos` carries `atom_id` (the split group) + the hash columns. `meta` is the set
    of extra columns stamped on every result row (m, strategy, n_atoms, dropped_frac,
    fixed_n, ...). GroupKFold keeps whole atoms in one fold (cluster-disjoint), is
    deterministic + unshuffled, so the std is the fold-to-fold spread of ONE
    partition; label balance comes from the per-fold 1:1 negatives, not the splitter.
    With `verify`, asserts each fold's train/test clusters are disjoint on both slots.

    TODO(cv-variants): for seeded/repeated grouped CV, loop StratifiedGroupKFold(
    shuffle=True, random_state=seed) over seeds (sklearn has no
    RepeatedStratifiedGroupKFold; RepeatedStratifiedKFold is group-UNAWARE -> would
    split atoms across folds and leak).
    """
    ha, hb = _HASH[alphabet]
    if len(pos) < k_folds or pos['atom_id'].nunique() < k_folds:
        return None

    gkf = GroupKFold(n_splits=k_folds)
    per = {mdl: {mt: [] for mt in _METRICS} for mdl in models}
    for train_idx, test_idx in gkf.split(pos, groups=pos['atom_id'].to_numpy()):
        tr_pos, te_pos = pos.iloc[train_idx], pos.iloc[test_idx]
        if verify:
            _assert_fold_disjoint(tr_pos, te_pos)
        tr = _labeled(tr_pos, make_negatives(tr_pos, alphabet, seed), ha, hb)
        te = _labeled(te_pos, make_negatives(te_pos, alphabet, seed), ha, hb)
        x_tr, y_tr = pair_features(tr, ha, hb, hash_to_row, matrix, interaction)
        x_te, y_te = pair_features(te, ha, hb, hash_to_row, matrix, interaction)
        scaler = StandardScaler().fit(x_tr)  # fit on train only; fixes MLP on raw counts
        x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)
        for mdl in models:
            scores = fit_eval(mdl, x_tr, y_tr, x_te, y_te, seed)
            for mt in _METRICS:
                per[mdl][mt].append(scores[mt])

    rows = []
    for mdl in models:
        row = {'schema_pair': f'{slot_a}-{slot_b}', 'alphabet': alphabet,
               'threshold': threshold, 'model': mdl, 'n_pos': int(len(pos)),
               'n_folds': k_folds}
        row.update(meta)
        for mt in _METRICS:
            arr = np.array(per[mdl][mt])
            row[f'{mt}_mean'] = round(float(arr.mean()), 4)
            row[f'{mt}_std'] = round(float(arr.std()), 4)
        rows.append(row)
    summary = "  ".join(f"{mdl}:f1={r['f1_macro_mean']:.3f}/pr={r['auc_pr_mean']:.3f}/mcc={r['mcc_mean']:.3f}"
                        for mdl, r in zip(models, rows))
    print(f"  [{slot_a}-{slot_b} {alphabet} {threshold} {tag}] N={len(pos)}  {summary}")
    return rows


def plot_m_sweep(df_pair: pd.DataFrame, pair_label, alphabet, models, ms, out_png):
    """2x2 grid: one F1-macro-vs-t panel per model + an N-vs-t panel; m overlaid (viridis)."""
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
                y = [mm.loc[t, 'f1_macro_mean'] for t in ts]
                e = [mm.loc[t, 'f1_macro_std'] for t in ts]
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


def plot_fixedN_curve(df_pair: pd.DataFrame, pair_label, alphabet, models, out_png):
    """2x2 grid: AUC-PR / F1-macro / MCC vs t (one line per model) + a cut-diagnostics
    panel (drop% and largest-atom% per t, with atom counts)."""
    ts = sorted(df_pair['threshold'].unique(), reverse=True)
    x = np.arange(len(ts))
    dd = df_pair.drop_duplicates('threshold').set_index('threshold')

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    for ax, mt in zip(axes.flat[:3], _METRICS):
        for mdl in models:
            mm = df_pair[df_pair['model'] == mdl].set_index('threshold')
            y = [mm.loc[t, f'{mt}_mean'] for t in ts]
            e = [mm.loc[t, f'{mt}_std'] for t in ts]
            ax.errorbar(x, y, yerr=e, marker='o', capsize=2,
                        color=_MODEL_COLOR.get(mdl), label=mdl)
        ax.set_xticks(x); ax.set_xticklabels(ts, fontsize=8)
        ax.set_ylim(0.0, 1.02)
        ax.axhline(0.5, linestyle=':', color='#999', linewidth=1)
        ax.grid(linestyle=':', alpha=0.5); ax.legend(fontsize=8)
        ax.set_title(_METRIC_LABEL[mt]); ax.set_ylabel(f'{_METRIC_LABEL[mt]} (mean +/- std)')

    axd = axes.flat[3]
    axd.bar(x - 0.18, [100 * dd.loc[t, 'dropped_frac'] for t in ts], width=0.36,
            color='#d62728', alpha=0.75, label='cut drop %')
    axd.bar(x + 0.18, [100 * dd.loc[t, 'largest_atom_frac'] for t in ts], width=0.36,
            color='#1f77b4', alpha=0.75, label='largest atom %')
    for i, t in enumerate(ts):
        axd.annotate(f"{int(dd.loc[t, 'n_atoms'])} atoms", (i, 1.5), fontsize=7,
                     ha='center', va='bottom', rotation=90, color='#333')
    axd.set_xticks(x); axd.set_xticklabels(ts, fontsize=8)
    axd.grid(linestyle=':', alpha=0.5); axd.legend(fontsize=8)
    axd.set_title('cut diagnostics'); axd.set_ylabel('percent')

    n = int(df_pair['fixed_n'].iloc[0])
    fig.suptitle(f'{pair_label} — {alphabet} — fixed-N={n} (cut, cluster-pair sampled): '
                 f'score and cut-bias vs cluster threshold t', fontsize=12, y=1.0)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    plt.close(fig)


def _run_msweep(universe, slot_a, slot_b, alphabet, pair, args, hash_to_row, matrix, out_dir):
    """Exp 1-2: per-CC cap m (swept), natural atoms, GroupKFold by CC."""
    ms = sorted(set(args.max_per_cc))
    pair_rows: list[dict] = []
    for t in args.thresholds:
        pairs, audit = assign_atoms(universe, slot_a, slot_b, alphabet, t,
                                    strategy='natural', k_folds=args.k_folds,
                                    return_audit=True)
        cc_sizes = pairs.groupby('cc_id').size()
        for mv in ms:
            pos = sample_positives(pairs, max_per=mv, unit='cc', seed=args.seed)
            n_capped = int((cc_sizes >= mv).sum())
            meta = {'m': mv, 'strategy': 'natural', 'sample_unit': 'cc', 'cut_method': '',
                    'fixed_n': '', 'n_cluster_pairs': audit['n_cluster_pairs'],
                    'n_atoms': audit['n_atoms'], 'largest_atom_frac': audit['largest_atom_frac'],
                    'dropped_frac': 0.0, 'n_capped': n_capped}
            rows = run_cv(pos, slot_a, slot_b, alphabet, t, hash_to_row=hash_to_row,
                          matrix=matrix, interaction=args.interaction, k_folds=args.k_folds,
                          models=args.models, seed=args.seed, meta=meta,
                          tag=f'm={mv}({n_capped}/{audit["n_atoms"]}cap)', verify=args.verify)
            if rows:
                pair_rows.extend(rows)
    if pair_rows:
        df_pair = pd.DataFrame(pair_rows)
        out_png = out_dir / f'msweep_{slot_a}_{slot_b}_{alphabet}.png'
        plot_m_sweep(df_pair, pair, alphabet, args.models, ms, out_png)
        print(f"  wrote {out_png}")
    return pair_rows


def _run_fixed_n(universe, slot_a, slot_b, alphabet, pair, args, hash_to_row, matrix, out_dir):
    """Exp 3: fragment (cut) each t, sample per cluster pair, fix N across t, GroupKFold by atom."""
    # Pass 1: cut + sample per cluster pair; cache the atom-tagged pairs + measure availability.
    cache: dict = {}
    for t in args.thresholds:
        pairs, audit = assign_atoms(universe, slot_a, slot_b, alphabet, t, strategy='cut',
                                    k_folds=args.k_folds, cut_method=args.cut_method,
                                    drift_pp=args.drift_pp, seed=args.seed, return_audit=True)
        pos_full = sample_positives(pairs, max_per=args.max_per_cluster_pair,
                                    unit='cluster_pair', seed=args.seed)
        cache[t] = (pos_full, audit)
        print(f"  [cut {t}] drop {audit['dropped_frac']:.1%}  {audit['n_atoms']} atoms  "
              f"largest {audit['largest_atom_frac']:.1%}  avail {len(pos_full)} pos")

    avail = {t: len(pos) for t, (pos, _) in cache.items()}
    N = min(avail.values()) if args.fixed_n == 'auto' else int(args.fixed_n)
    floor_t = min(avail, key=avail.get)
    print(f"  fixed N = {N}  (floor {min(avail.values())} at {floor_t})")

    # Pass 2: subsample each t to N, run CV.
    pair_rows: list[dict] = []
    for t in args.thresholds:
        pos_full, audit = cache[t]
        pos = (pos_full if len(pos_full) <= N
               else pos_full.sample(n=N, random_state=args.seed).reset_index(drop=True))
        meta = {'m': args.max_per_cluster_pair, 'strategy': 'cut', 'sample_unit': 'cluster_pair',
                'cut_method': args.cut_method, 'fixed_n': N,
                'n_cluster_pairs': audit['n_cluster_pairs'], 'n_atoms': int(pos['atom_id'].nunique()),
                'largest_atom_frac': audit['largest_atom_frac'], 'dropped_frac': audit['dropped_frac'],
                'n_capped': ''}
        rows = run_cv(pos, slot_a, slot_b, alphabet, t, hash_to_row=hash_to_row, matrix=matrix,
                      interaction=args.interaction, k_folds=args.k_folds, models=args.models,
                      seed=args.seed, meta=meta, tag=f'N={N}', verify=args.verify)
        if rows:
            pair_rows.extend(rows)
    if pair_rows:
        df_pair = pd.DataFrame(pair_rows)
        out_png = out_dir / f'scorecurve_fixedN_{slot_a}_{slot_b}_{alphabet}.png'
        plot_fixedN_curve(df_pair, pair, alphabet, args.models, out_png)
        print(f"  wrote {out_png}")
    return pair_rows


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
    p.add_argument('--models', nargs='+', default=['mlp', 'lgbm', 'knn1'],
                   help="drop knn1 for the cut/fixed-N path (slow at high N, degenerate AUC-PR).")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--strategy', default='natural', choices=['natural', 'cut'],
                   help="natural = per-CC m-sweep (Exp 1-2); cut = fragmented fixed-N (Exp 3).")
    # Exp 1-2 (natural): cap per CC, swept + overlaid.
    p.add_argument('--max_per_cc', nargs='+', type=int, default=[1, 2, 5, 10],
                   help='natural: cap of pairs per CC (m); swept and overlaid.')
    # Exp 3 (cut): cap per cluster pair + a fixed N across t.
    p.add_argument('--cut_method', default='spectral', choices=['spectral', 'kl'],
                   help='cut: bisection heuristic (spectral drops fewer pairs than kl).')
    p.add_argument('--drift_pp', type=float, default=0.05,
                   help='cut: K-uniform LPT feasibility gate (max bin drift).')
    p.add_argument('--max_per_cluster_pair', type=int, default=1,
                   help='cut: cap of pairs sampled per cluster pair.')
    p.add_argument('--fixed_n', default='auto',
                   help="cut: target N positives held across t ('auto' = floor over t).")
    p.add_argument('--verify', action='store_true',
                   help='assert train/test cluster-disjointness on every fold (slower).')
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
    csv_name = ('cluster_disjoint_cv_results.csv' if args.strategy == 'natural'
                else 'cluster_disjoint_cv_fixedN_results.csv')

    all_rows: list[dict] = []
    for pair in args.schema_pairs:
        slot_a, slot_b = pair.split('-')
        print(f"=== {pair} (strategy={args.strategy}) ===")
        universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
        hash_to_row = build_hash_to_row(args.kmer_k, [s2f[slot_a], s2f[slot_b]])
        for alphabet in args.alphabets:
            runner = _run_msweep if args.strategy == 'natural' else _run_fixed_n
            all_rows.extend(runner(universe, slot_a, slot_b, alphabet, pair, args,
                                   hash_to_row, matrix, out_dir))

    if all_rows:
        csv = out_dir / csv_name
        pd.DataFrame(all_rows).to_csv(csv, index=False)
        print(f"\nwrote {csv} ({len(all_rows)} rows)")
    print("\nDone.")


if __name__ == '__main__':
    main()
