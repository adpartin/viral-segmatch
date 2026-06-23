"""Cluster-disjoint K-fold negative-regime breakdown (within-CC regime negatives).

Companion to the m-sweep (`cluster_disjoint_cv_experiment.py`). Where the m-sweep
measures score-vs-t, this measures **per-regime TPR/TNR** under cluster-disjoint
CV — does the model lean on a metadata shortcut? — using the v2 8-regime negative
taxonomy, now sampled WITHIN each CC (the cluster-disjoint analogue of the
production regime negatives).

Design (locked 2026-06-08; see the session design notes):
  - Atoms = natural bipartite CCs (`assign_atoms(strategy='natural')`). Each CC
    stays whole within a fold.
  - Positives: cap `m_pos` per CC (fold-size balance).
  - Negatives: WITHIN each CC, regime-targeted + availability-aware + redistribute
    on shortfall (`_cv_sampling.sample_regime_negatives`), budget =
    `neg_to_pos_ratio x sampled positives`. Singletons / metadata-homogeneous CCs
    yield few or none — best-effort, logged. (Subtype-differ regimes come only
    from the metadata-diverse mega-CC; pure satellites give subtype-match only.)
  - **Sample first, then split the full pos+neg set** by `atom_id` so GroupKFold
    balances actual fold sizes (the regime negative counts vary per CC). Negatives
    are within-CC, so `atom_id` keeps them with their CC and train/test negatives
    are automatically disjoint.
  - Repeated grouped CV: `GroupKFold(shuffle=True, random_state=seed)` x
    `n_repeats` (sklearn>=1.6 GroupKFold shuffles); each pair is OOF-predicted once
    per repeat -> per-regime TPR/TNR per repeat -> mean +/- std across repeats.

Reuses (no reimplementation): `assign_atoms` / `sample_positives` /
`build_isolate_context` / `sample_regime_negatives` (`_cv_sampling`); k-mer
features + `fit_predict` (`_cv_features`); the per-regime breakdown
`analyze_level1_neg_regimes` (`analyze_stage4_train`); the grouped-composition
renderer `_render_split_composition_grouped` (`visualize_dataset_stats`).

Settings match the m-sweep (aa k-mer k=3, `unit_diff`, LGBM). The production
config (nt k=6, slot_norm, unit_diff+prod) is a later port.

CLI:
    python -m src.analysis.cluster_disjoint_regime_cv \\
        [--schema_pair HA NA] [--alphabet aa] [--threshold t095] \\
        [--m_pos 100] [--neg_to_pos_ratio 1.0] [--k_folds 5] [--n_repeats 3] \\
        [--kmer_k 3] [--interaction unit_diff] [--seed 0] \\
        [--out_dir results/flu/July_2025/runs/cluster_disjoint_regime_cv]

Outputs (under --out_dir):
    level1_neg_regimes_agg_{slug}_{t}.csv   per-regime TPR/TNR mean+/-std over repeats
    level1_neg_regimes_agg_{slug}_{t}.png   bar plot with error bars
    composition_{slug}_{t}{,_by_regime}.png achieved neg:pos ratio + regime mix
    pos_metadata_{slug}_{t}.csv             sampled-positives metadata composition
    regimes/repeat{r}/level1_neg_regimes.*  per-repeat breakdown (reused producer)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold, train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe  # noqa: E402
from src.analysis._cv_sampling import (  # noqa: E402
    assign_atoms, sample_positives, build_isolate_context,
    sample_regime_negatives, sample_random_within_cc_negatives)
from src.analysis._cv_features import build_hash_to_row, _interaction, _KMER_DIR  # noqa: E402
from src.analysis.analyze_stage4_train import (  # noqa: E402
    analyze_level1_neg_regimes, _LEVEL1_REGIME_ORDER, _resolve_neg_regime_column,
    compute_basic_metrics, analyze_metrics_summary, plot_confusion_matrix)
from src.analysis.visualize_dataset_stats import _render_split_composition_grouped  # noqa: E402
from src.models.baselines.lgbm import get_estimator, fit as lgbm_fit  # noqa: E402
from src.datasets._negative_regime_sampling import REGIME_NAMES  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402

_HASH = {'aa': ('prot_hash_a', 'prot_hash_b'), 'nt_cds': ('cds_dna_hash_a', 'cds_dna_hash_b')}
_DEFAULT_REGIME_TARGETS = {
    'none_match': 0.05, 'host_only': 0.10, 'subtype_only': 0.10, 'year_only': 0.10,
    'host_subtype_only': 0.15, 'host_year_only': 0.15, 'subtype_year_only': 0.15,
    'host_subtype_year': 0.20,
}


def _features(df: pd.DataFrame, ha: str, hb: str, hash_to_row: dict,
              matrix: sparse.csr_matrix, interaction: str):
    """(X, df_valid): k-mer interaction features + the rows that have features.

    Like `_cv_features.pair_features` but also returns the surviving rows so the
    OOF predictions stay aligned with their `label` / `neg_regime`.
    """
    ra = df[ha].map(hash_to_row)
    rb = df[hb].map(hash_to_row)
    valid = ra.notna() & rb.notna()
    ea = matrix[ra[valid].astype(int).to_numpy()].toarray().astype(np.float32)
    eb = matrix[rb[valid].astype(int).to_numpy()].toarray().astype(np.float32)
    return _interaction(ea, eb, interaction), df.loc[valid].reset_index(drop=True)


def _balance(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample the majority class to 1:1 so the model isn't majority-biased.

    Applied to the TRAIN fold only; the test fold keeps all its negatives so the
    per-regime TNR is measured on the full (unbalanced) regime mix.
    """
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return df
    rng = np.random.RandomState(seed)
    pos_s = pos.sample(n=n, random_state=rng) if len(pos) > n else pos
    neg_s = neg.sample(n=n, random_state=rng) if len(neg) > n else neg
    return pd.concat([pos_s, neg_s], ignore_index=True)


def _fit_predict_prod_lgbm(x_tr, y_tr, x_te, seed):
    """Production LGBM (conf/baselines/default.yaml `baseline_lgbm` params) with
    val-driven early stopping. Reuses `src.models.baselines.lgbm.get_estimator/fit`;
    `config=None` selects the baseline_lgbm defaults. A 15% stratified val split is
    carved from train for early stopping (matches the production val-driven fit).
    """
    est = get_estimator(None, random_state=seed)
    if len(np.unique(y_tr)) > 1 and len(y_tr) >= 30:
        x_t, x_v, y_t, y_v = train_test_split(x_tr, y_tr, test_size=0.15,
                                              random_state=seed, stratify=y_tr)
        lgbm_fit(est, x_t, y_t, X_val=x_v, y_val=y_v, config=None)
    else:
        lgbm_fit(est, x_tr, y_tr, config=None)
    return est.predict(x_te), est.predict_proba(x_te)[:, 1]


def build_regime_dataset(universe, iso, cooccur, slot_a, slot_b, alphabet, threshold,
                         *, m_pos, m_neg, neg_to_pos_ratio, regime_targets, neg_strategy, seed):
    """Full labeled pos+neg set: cap-m positives per CC + within-CC regime negatives.

    Returns (full_df, sampling_log). `full_df` columns: prot_hash_a, prot_hash_b,
    label, atom_id, cc_id, neg_regime (pd.NA on positives). `sampling_log` carries
    per-CC achieved counts for the composition views.
    """
    ha, hb = _HASH[alphabet]
    pairs = assign_atoms(universe, slot_a, slot_b, alphabet, threshold, strategy='natural')
    pos = sample_positives(pairs, max_per=m_pos, unit='cc', seed=seed)
    pos = pos[[ha, hb, 'cc_id', 'atom_id']].copy()
    pos['label'] = 1
    pos['neg_regime'] = pd.NA

    # CC -> its atom (1:1 under natural); needed to tag negatives.
    cc_to_atom = dict(zip(pairs['cc_id'], pairs['atom_id']))
    iso_by_cc = {cc: g for cc, g in iso.groupby('cc_id')}
    pos_per_cc = pos.groupby('cc_id').size()

    neg_frames, log_rows = [], []
    for cc, n_pos in pos_per_cc.items():
        # m_neg (absolute per-CC cap) decouples negatives from the positive cap;
        # else fall back to a positive-tied budget (neg_to_pos_ratio x positives).
        budget = m_neg if m_neg is not None else int(round(neg_to_pos_ratio * n_pos))
        cc_iso = iso_by_cc.get(cc)
        if cc_iso is None or budget <= 0:
            log_rows.append({'cc_id': cc, 'n_pos': int(n_pos), 'budget': budget, 'n_neg': 0})
            continue
        if neg_strategy == 'random':
            neg = sample_random_within_cc_negatives(cc_iso, budget, cooccur, seed=seed + int(cc))
        else:
            neg = sample_regime_negatives(cc_iso, regime_targets, budget, cooccur,
                                          seed=seed + int(cc))
        if len(neg):
            neg = neg.rename(columns={'prot_hash_a': ha, 'prot_hash_b': hb})
            neg['cc_id'] = cc
            neg['atom_id'] = cc_to_atom[cc]
            neg['label'] = 0
            neg_frames.append(neg[[ha, hb, 'cc_id', 'atom_id', 'label', 'neg_regime']])
        log_rows.append({'cc_id': cc, 'n_pos': int(n_pos), 'budget': budget, 'n_neg': int(len(neg))})

    neg_all = (pd.concat(neg_frames, ignore_index=True) if neg_frames
               else pd.DataFrame(columns=[ha, hb, 'cc_id', 'atom_id', 'label', 'neg_regime']))
    full = pd.concat([pos[[ha, hb, 'cc_id', 'atom_id', 'label', 'neg_regime']], neg_all],
                     ignore_index=True)
    return full, pd.DataFrame(log_rows)


def oof_regime_breakdown(full, alphabet, *, hash_to_row, matrix, interaction,
                         k_folds, n_repeats, seed, out_dir, slug, threshold):
    """Repeated grouped-CV OOF -> per-regime TPR/TNR mean+/-std (reuses level1 producer)."""
    ha, hb = _HASH[alphabet]
    per_repeat_stats, all_oof = [], []
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=k_folds, shuffle=True, random_state=seed + r)
        oof = []
        for tr_idx, te_idx in gkf.split(full, groups=full['atom_id'].to_numpy()):
            tr, te = full.iloc[tr_idx], full.iloc[te_idx]
            tr = _balance(tr, seed + r)  # 1:1 train so the model isn't majority-biased
            x_tr, tr_v = _features(tr, ha, hb, hash_to_row, matrix, interaction)
            x_te, te_v = _features(te, ha, hb, hash_to_row, matrix, interaction)
            if len(tr_v) < 2 or te_v.empty or tr_v['label'].nunique() < 2:
                continue
            pred, prob = _fit_predict_prod_lgbm(x_tr, tr_v['label'].to_numpy(), x_te, seed)
            te_v = te_v.assign(pred_label=pred, pred_prob=prob)
            oof.append(te_v[['label', 'pred_label', 'pred_prob', 'neg_regime']])
        oof_df = pd.concat(oof, ignore_index=True)
        rep_dir = out_dir / 'regimes' / f'repeat{r}'
        rep_dir.mkdir(parents=True, exist_ok=True)
        stats = analyze_level1_neg_regimes(oof_df, rep_dir)  # reused producer (per repeat)
        per_repeat_stats.append(stats.set_index('regime'))
        all_oof.append(oof_df)

    # Aggregate TPR (positive) / TNR (negatives) across repeats: mean +/- std.
    rows = []
    for regime in _LEVEL1_REGIME_ORDER:
        metric = 'tpr' if regime == 'positive' else 'tnr'
        vals = [s.loc[regime, metric] for s in per_repeat_stats
                if regime in s.index and pd.notna(s.loc[regime, metric])]
        ns = [s.loc[regime, 'n_samples'] for s in per_repeat_stats if regime in s.index]
        rows.append({'regime': regime, 'metric': metric,
                     'mean': float(np.mean(vals)) if vals else np.nan,
                     'std': float(np.std(vals)) if vals else np.nan,
                     'n_mean': float(np.mean(ns)) if ns else 0.0,
                     'n_repeats': len(vals)})
    agg = pd.DataFrame(rows)
    agg.to_csv(out_dir / f'level1_neg_regimes_agg_{slug}_{threshold}.csv', index=False)
    _plot_regime_agg(agg, out_dir / f'level1_neg_regimes_agg_{slug}_{threshold}.png',
                     slug, threshold, n_repeats)
    return agg, pd.concat(all_oof, ignore_index=True)


def _plot_regime_agg(agg, out_png, slug, threshold, n_repeats):
    plot = agg[agg['n_mean'] > 0].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(plot))
    colors = ['seagreen' if r == 'positive' else 'crimson' for r in plot['regime']]
    ax.bar(x, plot['mean'], yerr=plot['std'], capsize=3, color=colors,
           edgecolor='black', linewidth=0.4)
    for xi, (m, n) in enumerate(zip(plot['mean'], plot['n_mean'])):
        if pd.notna(m):
            ax.annotate(f'{m:.2f}\nn={int(n):,}', (xi, m), textcoords='offset points',
                        xytext=(0, 3), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('host_subtype_year', 'host_sub_year') for r in plot['regime']],
                       rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('TPR (positive) / TNR (negatives)  ·  mean ± std')
    ax.axhline(1.0, linestyle=':', color='#999', linewidth=1)
    ax.set_title(f'Per-regime TPR/TNR — {slug} {threshold} — cluster-disjoint OOF '
                 f'({n_repeats}× repeated k-fold)', fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


def plot_prob_distribution(oof, out_dir, slug, threshold):
    """Per-regime OOF predicted-probability distribution — disambiguates low TNR.

    Probs clustered near 0.5 => the model is *uncertain* (can't tell within-CC
    negatives from positives -> no co-occurrence signal). Probs high (>0.5) on
    negatives => *confidently wrong* (the model 'believes' within-CC negatives are
    co-occurring pairs -- the cluster-membership shortcut). The 0.5 decision line
    is drawn for reference; the positive box anchors what a confident positive
    looks like.
    """
    df = oof.copy()
    df['regime'] = _resolve_neg_regime_column(df)
    order = [r for r in _LEVEL1_REGIME_ORDER if (df['regime'] == r).any()]
    data = [df.loc[df['regime'] == r, 'pred_prob'].to_numpy() for r in order]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black'))
    for patch, r in zip(bp['boxes'], order):
        patch.set_facecolor('seagreen' if r == 'positive' else 'crimson')
        patch.set_alpha(0.55)
    ax.axhline(0.5, linestyle='--', color='#333', linewidth=1.2, label='decision threshold (0.5)')
    for i, (r, d) in enumerate(zip(order, data), start=1):
        ax.annotate(f'n={len(d):,}\n>0.5:{100 * (d > 0.5).mean():.0f}%', (i, 1.03),
                    ha='center', va='bottom', fontsize=7, annotation_clip=False)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([r.replace('host_subtype_year', 'host_sub_year') for r in order],
                       rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('OOF predicted probability (positive class)')
    ax.set_title(f'Per-regime prediction-prob distribution — {slug} {threshold}\n'
                 f'low TNR = "uncertain ~0.5" or "confidently positive (shortcut)"?', fontsize=11)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    out_png = out_dir / f'prob_distribution_{slug}_{threshold}.png'
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    rows = [{'regime': r, 'n': len(d), 'mean_prob': float(d.mean()),
             'median_prob': float(np.median(d)), 'q10': float(np.quantile(d, 0.1)),
             'q90': float(np.quantile(d, 0.9)), 'frac_gt_0.5': float((d > 0.5).mean())}
            for r, d in zip(order, data)]
    pd.DataFrame(rows).to_csv(out_dir / f'prob_distribution_{slug}_{threshold}.csv', index=False)
    print(f'wrote {out_png}')


def plot_composition(full, sampling_log, out_dir, slug, threshold, *, m_pos, m_neg, neg_to_pos_ratio):
    """Achieved neg:pos ratio + regime mix (reuses the grouped-composition renderer)."""
    n_pos = int((full['label'] == 1).sum())
    n_neg = int((full['label'] == 0).sum())
    achieved = n_neg / n_pos if n_pos else 0.0
    regime_counts = {r: int((full['neg_regime'] == r).sum()) for r in REGIME_NAMES}
    rcps = {'sampled': regime_counts}
    pos_counts = np.array([n_pos])
    n_no_neg = int((sampling_log['n_neg'] == 0).sum())
    cfg = (f'm_pos={m_pos}, m_neg={m_neg}' if m_neg is not None
           else f'm_pos={m_pos}, neg_to_pos_ratio={neg_to_pos_ratio:.2f}')
    anno = [(f'{cfg}  |  achieved neg:pos = {achieved:.2f}'
             f'   ·   {n_no_neg:,} CCs yielded 0 negatives', '#333', None)]
    for by_regime, tag in [(False, ''), (True, '_by_regime')]:
        out_png = out_dir / f'composition_{slug}_{threshold}{tag}.png'
        _render_split_composition_grouped(
            splits=['sampled'], pos_counts=pos_counts, regime_counts_per_split=rcps,
            by_regime=by_regime, annotations=anno,
            bundle_name=f'regime-CV {slug} {threshold}', output_path=out_png)


def plot_pos_metadata(pos_full, iso, out_dir, slug, threshold):
    """Metadata composition (host / subtype / year) of the sampled positives.

    A positive pair maps to >=1 isolate; join each sampled positive pair to its
    isolates (via the seq-hash pair) and count the metadata-cell axes. Writes a
    3-panel bar plot + a CSV. (Counts isolate rows, so a pair shared across many
    isolates contributes its metadata once per isolate.)
    """
    iso2 = iso[['prot_hash_a', 'prot_hash_b', 'cell']].copy()
    j = pos_full.merge(iso2, on=['prot_hash_a', 'prot_hash_b'], how='inner')
    if j.empty:
        return
    cells = pd.DataFrame(j['cell'].tolist(), columns=['host', 'hn_subtype', 'year_bin'])

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (axis, color) in zip(axes, [('hn_subtype', '#4c72b0'), ('host', '#55a868'),
                                        ('year_bin', '#c44e52')]):
        vc = cells[axis].astype(str).value_counts().head(15)
        for val, n in vc.items():
            rows.append({'axis': axis, 'value': val, 'n_isolate_rows': int(n)})
        ax.bar(range(len(vc)), vc.values, color=color, edgecolor='black', linewidth=0.4)
        for xi, n in enumerate(vc.values):
            ax.annotate(f'{int(n):,}', (xi, n), textcoords='offset points', xytext=(0, 2),
                        ha='center', va='bottom', fontsize=6.5)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=45, ha='right', fontsize=7)
        ax.set_title(axis, fontsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
    axes[0].set_ylabel('isolate rows (per sampled positive pair)')
    fig.suptitle(f'Sampled-positive metadata composition — {slug} {threshold}', fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = out_dir / f'pos_metadata_{slug}_{threshold}.png'
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / f'pos_metadata_{slug}_{threshold}.csv', index=False)
    print(f'wrote {out_png} ({cells["hn_subtype"].nunique()} subtypes among sampled positives)')


def plot_metrics_confusion_level1(oof, out_dir, slug, threshold):
    """Overall metrics + confusion matrix + per-regime TPR/TNR, all via the
    production producers in `analyze_stage4_train` (matches the post-hoc plots in a
    `training_*` run): metrics_*.{png,csv}, confusion_matrix_*.png, and
    level1_neg_regimes.{png,csv}. Computed on the pooled OOF predictions.
    """
    y_true = oof['label'].to_numpy()
    y_pred = oof['pred_label'].to_numpy()
    y_prob = oof['pred_prob'].to_numpy()
    metrics = compute_basic_metrics(y_true, y_pred, y_prob)
    pd.DataFrame([metrics]).to_csv(out_dir / f'metrics_{slug}_{threshold}.csv', index=False)
    analyze_metrics_summary(metrics, out_dir, save_name=f'metrics_{slug}_{threshold}.png')
    plot_confusion_matrix(y_true, y_pred, out_dir / f'confusion_matrix_{slug}_{threshold}.png')
    analyze_level1_neg_regimes(oof, out_dir)  # production level1_neg_regimes.{png,csv}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final', default=str(PROJ / 'data/processed/flu/July_2025/cds_final.parquet'))
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'], metavar=('A', 'B'))
    p.add_argument('--alphabet', default='aa', choices=['aa', 'nt_cds'])
    p.add_argument('--threshold', default='t095')
    p.add_argument('--m_pos', type=int, default=100,
                   help='cap positives per CC (fold balance).')
    p.add_argument('--m_neg', type=int, default=None,
                   help='cap negatives per CC (best-effort), INDEPENDENT of m_pos; '
                        'overrides --neg_to_pos_ratio when set. Raise it to feed the '
                        'mega-CC subtype-differ regimes (which come only from the mega-CC); '
                        'per-fold training is rebalanced 1:1 so a high m_neg is fine.')
    p.add_argument('--neg_to_pos_ratio', type=float, default=1.0,
                   help='negatives per positive per CC when --m_neg is NOT set.')
    p.add_argument('--neg_strategy', default='regime', choices=['regime', 'random'],
                   help="within-CC negative sampler: 'regime' (target the 8-regime mix) or "
                        "'random' (uniform within-CC mixing; regimes labeled post-hoc).")
    p.add_argument('--k_folds', type=int, default=5)
    p.add_argument('--n_repeats', type=int, default=3)
    p.add_argument('--kmer_k', type=int, default=3)
    p.add_argument('--interaction', default='unit_diff', choices=['concat', 'unit_diff'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/cluster_disjoint_regime_cv')
    args = p.parse_args()
    if args.alphabet != 'aa':
        raise NotImplementedError("only aa wired (membership + k-mer); nt_cds is a later port.")

    slot_a, slot_b = args.schema_pair
    slug = f'{slot_a.lower()}_{slot_b.lower()}'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s2f = load_function_metadata(PROJ / 'conf/virus/flu.yaml').short_to_function

    print(f"=== regime-CV {slot_a}-{slot_b} {args.alphabet} {args.threshold} ===")
    universe = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
    u = assign_atoms(universe, slot_a, slot_b, args.alphabet, args.threshold, strategy='natural')
    iso, cooccur = build_isolate_context(u, universe, slot_a, slot_b, args.alphabet, args.threshold)
    hash_to_row = build_hash_to_row(args.kmer_k, [s2f[slot_a], s2f[slot_b]])
    matrix = sparse.load_npz(_KMER_DIR / f'kmer_features_aa_k{args.kmer_k}.npz')

    full, sampling_log = build_regime_dataset(
        universe, iso, cooccur, slot_a, slot_b, args.alphabet, args.threshold,
        m_pos=args.m_pos, m_neg=args.m_neg, neg_to_pos_ratio=args.neg_to_pos_ratio,
        regime_targets=_DEFAULT_REGIME_TARGETS, neg_strategy=args.neg_strategy, seed=args.seed)
    n_pos = int((full['label'] == 1).sum()); n_neg = int((full['label'] == 0).sum())
    print(f"  sampled {n_pos:,} pos + {n_neg:,} neg (achieved ratio {n_neg / max(n_pos, 1):.2f}); "
          f"{int((sampling_log['n_neg'] == 0).sum()):,} CCs with 0 negatives")

    ha, hb = _HASH[args.alphabet]
    plot_pos_metadata(full[full['label'] == 1][[ha, hb]], iso, out_dir, slug, args.threshold)
    plot_composition(full, sampling_log, out_dir, slug, args.threshold,
                     m_pos=args.m_pos, m_neg=args.m_neg, neg_to_pos_ratio=args.neg_to_pos_ratio)
    agg, oof_pooled = oof_regime_breakdown(
        full, args.alphabet, hash_to_row=hash_to_row, matrix=matrix,
        interaction=args.interaction, k_folds=args.k_folds, n_repeats=args.n_repeats,
        seed=args.seed, out_dir=out_dir, slug=slug, threshold=args.threshold)
    plot_prob_distribution(oof_pooled, out_dir, slug, args.threshold)
    plot_metrics_confusion_level1(oof_pooled, out_dir, slug, args.threshold)
    print("\nPer-regime TPR/TNR (mean ± std over repeats):")
    print(agg.round(3).to_string(index=False))
    print("\nDone.")


if __name__ == '__main__':
    main()
