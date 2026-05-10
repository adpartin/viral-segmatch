"""Stage 4 entry point for sklearn-style baselines (logistic regression, ...).

Runs ALONGSIDE the MLP trainer (it does not replace it). Reads the same
Stage 3 dataset directory and the same bundle config; writes one flat run
directory PER BASELINE under ``models/{virus}/{data_version}/runs/``.

Multi-baseline mode shares the (expensive) feature materialization across
baselines: features are loaded once with ``feature_scaling='none'``, then
each baseline's per-baseline scaling (StandardScaler or none) is applied
to a copy. The dominant kmer-densification cost (~minutes on full Flu A)
is paid once, regardless of how many baselines run.

Usage:
    # Single baseline, explicit
    python src/models/train_pair_baselines.py \\
        --config_bundle flu_ha_na_neg_regimes \\
        --baseline logistic \\
        --dataset_dir data/datasets/.../runs/dataset_flu_ha_na_neg_regimes_<TS>

    # Multiple baselines, explicit
    python src/models/train_pair_baselines.py \\
        --config_bundle flu_ha_na_neg_regimes \\
        --baseline logistic lgbm knn1_margin knn_vote \\
        --dataset_dir data/datasets/.../runs/dataset_flu_ha_na_neg_regimes_<TS>

    # Multiple baselines, from bundle (config.baselines.enabled)
    python src/models/train_pair_baselines.py \\
        --config_bundle flu_ha_na_neg_regimes \\
        --dataset_dir data/datasets/.../runs/dataset_flu_ha_na_neg_regimes_<TS>

Output (one run dir PER baseline):
    models/flu/{data_version}/runs/baseline_<name>_<bundle>_<TS>/
        ├── best_model.joblib            # fitted estimator
        ├── feature_scaler.joblib        # only when feature_scaling=='standard'
        ├── train_predicted.csv          # per-pair preds for the train split
        ├── val_predicted.csv
        ├── test_predicted.csv
        ├── optimal_threshold.txt
        ├── metrics_summary.json
        ├── training_info.json
        └── resolved_config.yaml
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary, save_config
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds
from src.utils.path_utils import build_training_paths, build_embeddings_paths
from src.models._pair_features import load_pair_features_for_baselines
from src.models._pair_metrics import compute_pair_metrics, find_optimal_threshold_pr


BASELINE_REGISTRY = {
    'logistic':    'src.models.baselines.logistic',
    'lgbm':        'src.models.baselines.lgbm',
    # knn1_margin: dedicated 1-NN cosine-margin diagnostic for leakage
    # detection (Plan Exp 2). knn_vote: generalized k-NN with configurable
    # k + weighting for the smoothed-neighborhood baseline comparison.
    'knn1_margin': 'src.models.baselines.knn1_margin',
    'knn_vote':    'src.models.baselines.knn_vote',
}


def _resolve_baseline_module(name: str):
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline {name!r}. Registered: {sorted(BASELINE_REGISTRY)}"
        )
    return importlib.import_module(BASELINE_REGISTRY[name])


def _resolve_kmer_k(config) -> int:
    if hasattr(config, 'kmer') and config.get('kmer') is not None:
        return int(config.kmer.get('k', 6))
    return 6


def _resolve_feature_scaling(config, baseline_module) -> str:
    """Resolution order: per-baseline bundle override → baseline module default."""
    cfg_key = f'baseline_{baseline_module.name()}'
    per_baseline_cfg = getattr(config, cfg_key, None)
    if per_baseline_cfg is not None and 'feature_scaling' in per_baseline_cfg:
        return str(per_baseline_cfg['feature_scaling'])
    if hasattr(baseline_module, 'feature_scaling_default'):
        return baseline_module.feature_scaling_default()
    return 'none'


def _resolve_baseline_list(args, config) -> list:
    """Resolve final list of baseline names from CLI + bundle.

    Precedence: --baseline (CLI, repeatable) wins; otherwise read
    config.baselines.enabled. Errors loudly when neither is provided.
    """
    if args.baseline:
        baselines = list(args.baseline)
    else:
        baselines_cfg = getattr(config, 'baselines', None)
        enabled = getattr(baselines_cfg, 'enabled', None) if baselines_cfg is not None else None
        if enabled is None:
            raise ValueError(
                "No baselines selected. Pass --baseline NAME [NAME ...] on the CLI "
                "or set `baselines.enabled` in the bundle config."
            )
        baselines = list(enabled)
    if not baselines:
        raise ValueError("Resolved baseline list is empty.")
    unknown = [b for b in baselines if b not in BASELINE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown baseline(s): {unknown}. Registered: {sorted(BASELINE_REGISTRY)}"
        )
    return baselines


def _resolve_output_dir(args, default_output_dir: Path, baseline_name: str,
                       config_bundle: str, batch_ts: str, n_baselines: int) -> Path:
    """Resolve output directory for one baseline.

    Single-baseline mode: --output_dir or --run_output_subdir override the
    default (preserves the prior single-baseline interface).
    Multi-baseline mode: those overrides are ambiguous (one path can't host
    N baselines) and we always use the per-baseline default
    `baseline_<name>_<bundle>_<batch_ts>`. Errors if a user passes them.
    """
    if n_baselines > 1:
        if args.output_dir or args.run_output_subdir:
            raise ValueError(
                "--output_dir / --run_output_subdir cannot be used with multiple "
                "baselines (they would all collide on one path). Either run "
                "baselines one at a time, or rely on the auto-derived per-baseline "
                "directories."
            )
        return default_output_dir / 'runs' / (
            f"baseline_{baseline_name}_{config_bundle}_{batch_ts}"
        )

    if args.output_dir:
        return Path(args.output_dir)
    if args.run_output_subdir:
        return default_output_dir / 'runs' / args.run_output_subdir
    return default_output_dir / 'runs' / (
        f"baseline_{baseline_name}_{config_bundle}_{batch_ts}"
    )


def _apply_per_baseline_scaling(
    feature_scaling: str,
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    output_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-baseline feature scaling on top of the shared raw matrices.

    For 'none': returns the raw arrays unchanged (no copy).
    For 'standard': fits a StandardScaler on the train split and transforms
    all three; persists the scaler under output_dir/feature_scaler.joblib so
    later inference matches.
    """
    if feature_scaling == 'none':
        return X_train_raw, X_val_raw, X_test_raw
    if feature_scaling != 'standard':
        raise ValueError(
            f"feature_scaling must be 'none' or 'standard'; got {feature_scaling!r}"
        )
    from sklearn.preprocessing import StandardScaler
    print('  Fitting StandardScaler on training features...')
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    Xtr = scaler.transform(X_train_raw).astype(np.float32, copy=False)
    Xvl = scaler.transform(X_val_raw).astype(np.float32, copy=False)
    Xte = scaler.transform(X_test_raw).astype(np.float32, copy=False)
    scaler_path = output_dir / 'feature_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f'  saved scaler to: {scaler_path}')
    return Xtr, Xvl, Xte


def _run_one_baseline(
    *,
    baseline_name: str,
    config,
    args,
    default_output_dir: Path,
    batch_ts: str,
    n_baselines: int,
    train_pairs: pd.DataFrame, val_pairs: pd.DataFrame, test_pairs: pd.DataFrame,
    X_train_raw: np.ndarray, y_train: np.ndarray,
    X_val_raw: np.ndarray, y_val: np.ndarray,
    X_test_raw: np.ndarray, y_test: np.ndarray,
    FEATURE_SOURCE: str, KMER_K: int,
    INTERACTION: str, SLOT_TRANSFORM: str,
    RANDOM_SEED, dataset_dir: Path,
    ) -> dict:
    """Fit one baseline on the shared materialized features, write per-baseline
    artifacts, and run post-hoc analysis. Returns a small summary dict.
    """
    per_timer = Timer()
    baseline_module = _resolve_baseline_module(baseline_name)
    feature_scaling = _resolve_feature_scaling(config, baseline_module)

    output_dir = _resolve_output_dir(
        args, default_output_dir, baseline_module.name(),
        args.config_bundle, batch_ts, n_baselines,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config alongside this baseline's artifacts. The
    # baseline-specific run dir gets its own copy so post-hoc / reruns can
    # treat it as standalone.
    save_config(config, str(output_dir / 'resolved_config.yaml'))

    print('\n' + '=' * 72)
    print(f'BASELINE: {baseline_module.name()}')
    print('=' * 72)
    print(f'Feature source:  {FEATURE_SOURCE}')
    print(f'Feature scaling: {feature_scaling}')
    print(f'Output dir:      {output_dir}')  # parsed by stage4_baselines.sh

    # ── Per-baseline scaling (cheap; ~5s for StandardScaler over ~9GB) ─────
    per_timer.begin_phase('scaling')
    X_train, X_val, X_test = _apply_per_baseline_scaling(
        feature_scaling, X_train_raw, X_val_raw, X_test_raw, output_dir,
    )
    per_timer.end_phase('scaling')

    # ── Train ──────────────────────────────────────────────────────────────
    per_timer.begin_phase('train')
    estimator = baseline_module.get_estimator(config, random_state=RANDOM_SEED)
    print(f'Fitting {type(estimator).__name__}: {estimator}')
    fit_fn = getattr(baseline_module, 'fit', None)
    if fit_fn is not None:
        fit_fn(estimator, X_train, y_train, X_val=X_val, y_val=y_val, config=config)
    else:
        estimator.fit(X_train, y_train)
    model_path = output_dir / 'best_model.joblib'
    joblib.dump(estimator, model_path)
    print(f'Saved fitted model to: {model_path}')
    per_timer.end_phase('train')

    # ── Inference (predictions + per-split metrics + CSVs) ─────────────────
    per_timer.begin_phase('inference')
    val_probs   = estimator.predict_proba(X_val)[:, 1]
    test_probs  = estimator.predict_proba(X_test)[:, 1]
    train_probs = estimator.predict_proba(X_train)[:, 1]

    if hasattr(estimator, 'decision_function'):
        train_logits = np.asarray(estimator.decision_function(X_train), dtype=np.float32)
        val_logits   = np.asarray(estimator.decision_function(X_val),   dtype=np.float32)
        test_logits  = np.asarray(estimator.decision_function(X_test),  dtype=np.float32)
    else:
        train_logits = val_logits = test_logits = None

    # ── Threshold selection (mirrors MLP) ──────────────────────────────────
    # TODO: per-baseline threshold override. Today every baseline + the MLP
    # read config.training.threshold_metric (a single bundle-level knob).
    # To allow e.g. LR uses val-F1 while LightGBM stays at 0.5, extend with
    # config.baselines.<name>.threshold_metric and resolve here. Low
    # priority until we see a concrete case where per-baseline thresholds
    # matter.
    THRESHOLD_METRIC = getattr(config.training, 'threshold_metric', None)
    if THRESHOLD_METRIC is None:
        threshold = 0.5
        threshold_score: Optional[float] = None
        print(f'\nUsing default threshold: {threshold:.4f} (threshold optimization disabled)')
    else:
        threshold, threshold_score = find_optimal_threshold_pr(
            y_val, val_probs, metric=THRESHOLD_METRIC,
        )
        print(f'\nOptimal threshold (optimizing {THRESHOLD_METRIC}): {threshold:.4f}')
        print(f'Best {THRESHOLD_METRIC} score on validation: {threshold_score:.4f}')
        with open(output_dir / 'optimal_threshold.txt', 'w') as f:
            f.write(f'{threshold}\n')
            f.write(f'metric: {THRESHOLD_METRIC}\n')
            f.write(f'best_score: {threshold_score:.4f}\n')

    summary = {}
    for split_name, pairs_df, y_true, probs, logits in (
        ('train', train_pairs, y_train, train_probs, train_logits),
        ('val',   val_pairs,   y_val,   val_probs,   val_logits),
        ('test',  test_pairs,  y_test,  test_probs,  test_logits),
    ):
        print(f'\n=== {split_name} ===')
        metrics, res_df = compute_pair_metrics(
            y_true, probs, threshold, pairs_df, logits=logits,
        )
        out_csv = output_dir / f'{split_name}_predicted.csv'
        res_df.to_csv(out_csv, index=False)
        print(f'{split_name} F1: {metrics["f1"]:.4f}, AUC: {metrics["auc"]:.4f}, '
              f'Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}')
        print(f'Saved predictions to: {out_csv}')
        summary[split_name] = metrics

    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    per_timer.end_phase('inference')

    training_info = {
        'config_bundle': args.config_bundle,
        'baseline': baseline_module.name(),
        'dataset_dir': str(dataset_dir),
        'feature_source': FEATURE_SOURCE,
        'feature_scaling': feature_scaling,
        'kmer_k': KMER_K if FEATURE_SOURCE == 'kmer' else None,
        'optimal_threshold': float(threshold),
        'threshold_metric': THRESHOLD_METRIC,
        'threshold_metric_score': float(threshold_score) if threshold_score is not None else None,
        'estimator': repr(estimator),
        'seed': RANDOM_SEED,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'interaction': INTERACTION,
        'slot_transform': SLOT_TRANSFORM,
        'hidden_dims': None,
        'dropout': None,
        'batch_size': None,
        'epochs': None,
        'patience': None,
        'learning_rate': None,
        'use_amp': None,
    }
    with open(output_dir / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f'\nSaved training provenance to: {output_dir / "training_info.json"}')

    per_timer.stop_timer()
    per_timer.display_timer()
    per_timer.save_timer(output_dir)

    # ── Post-hoc analysis ─────────────────────────────────────────────────
    if not args.skip_post_hoc:
        import subprocess
        post_hoc_log = output_dir / 'post_hoc_run.log'
        print(f'\nRunning post-hoc analysis -> {output_dir}/post_hoc/')
        post_hoc_cmd = [
            sys.executable,
            str(project_root / 'src' / 'analysis' / 'analyze_stage4_train.py'),
            '--config_bundle', args.config_bundle,
            '--model_dir', str(output_dir),
        ]
        try:
            with open(post_hoc_log, 'w') as logf:
                rc = subprocess.call(post_hoc_cmd, stdout=logf, stderr=subprocess.STDOUT)
            if rc != 0:
                print(f'WARNING: post-hoc analysis exited with code {rc} for {output_dir}.')
                print(f'         Baseline artifacts are unaffected. See log: {post_hoc_log}')
            else:
                print(f'Done. Post-hoc analysis complete. Log: {post_hoc_log}')
        except Exception as e:
            import traceback
            print(f'WARNING: post-hoc analysis subprocess failed for {output_dir}: {e}')
            traceback.print_exc()

    return {
        'baseline': baseline_module.name(),
        'output_dir': str(output_dir),
        'metrics': summary,
    }


def main() -> None:
    total_timer = Timer()

    parser = argparse.ArgumentParser(description='Train sklearn-style pair baseline(s)')
    parser.add_argument('--config_bundle', type=str, required=True,
                        help='Config bundle name (e.g., flu_ha_na_neg_regimes).')
    parser.add_argument('--baseline', type=str, nargs='+', default=None,
                        choices=sorted(BASELINE_REGISTRY.keys()),
                        help=f'One or more baselines (space-separated). '
                             f'If omitted, reads from `baselines.enabled` in the bundle. '
                             f'Registered: {sorted(BASELINE_REGISTRY)}.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Stage 3 dataset directory containing train/val/test_pairs.csv.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output dir. Single-baseline only.')
    parser.add_argument('--run_output_subdir', type=str, default=None,
                        help='Run-id subdirectory under default output_dir. Single-baseline only.')
    parser.add_argument('--cuda_name', type=str, default=None,
                        help='Accepted for shell-script symmetry; ignored by sklearn baselines.')
    parser.add_argument('--override', type=str, nargs='+', default=None,
                        help='Hydra dotlist overrides on top of the bundle.')
    parser.add_argument('--skip_post_hoc', action='store_true',
                        help='Skip the post-hoc analysis (analyze_stage4_train.py).')
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    config_path = str(project_root / 'conf')
    config = get_virus_config_hydra(args.config_bundle, config_path=config_path)
    if args.override:
        from omegaconf import OmegaConf
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
        print(f"Applied CLI overrides: {args.override}")
    print_config_summary(config)

    VIRUS_NAME = config.virus.virus_name
    DATA_VERSION = config.virus.data_version
    RANDOM_SEED = resolve_process_seed(config, 'training')
    FEATURE_SOURCE = getattr(config.training, 'feature_source', 'esm2')
    KMER_K = _resolve_kmer_k(config)
    INTERACTION = str(getattr(config.training, 'interaction', 'concat'))
    SLOT_TRANSFORM = str(getattr(config.training, 'slot_transform', 'none'))

    # Baselines and the MLP have different normalization concepts. The MLP
    # uses learned per-slot LayerNorm (training.slot_transform=slot_norm);
    # sklearn baselines use feature_scaling (StandardScaler / none) per
    # baseline. The kmer baseline path explicitly rejects slot_norm
    # (non-negative count vectors don't benefit). Silently coerce
    # slot_transform back to 'none' here so a bundle that sets
    # slot_norm=true for the MLP doesn't break the baseline runs.
    # ESM-2 baselines do support slot_norm — leave their setting alone.
    if FEATURE_SOURCE == 'kmer' and SLOT_TRANSFORM != 'none':
        print(
            f"NOTE: bundle sets training.slot_transform={SLOT_TRANSFORM!r} "
            "(for the MLP). The kmer baseline path uses each baseline's "
            "feature_scaling instead, so slot_transform is forced to 'none' "
            "for this run."
        )
        SLOT_TRANSFORM = 'none'

    if RANDOM_SEED is not None:
        set_deterministic_seeds(RANDOM_SEED, cuda_deterministic=False)
        print(f'Set deterministic seeds (seed: {RANDOM_SEED})')

    # ── Baseline list ───────────────────────────────────────────────────────
    baselines = _resolve_baseline_list(args, config)
    print(f'\nWill run {len(baselines)} baseline(s): {baselines}')

    # ── Paths ───────────────────────────────────────────────────────────────
    paths = build_training_paths(
        project_root=project_root, virus_name=VIRUS_NAME,
        data_version=DATA_VERSION, run_suffix="", config=config,
    )
    default_output_dir = paths['output_dir']
    batch_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    dataset_dir = Path(args.dataset_dir)
    print(f'\nConfig bundle:   {args.config_bundle}')
    print(f'Dataset dir:     {dataset_dir}')
    print(f'Batch timestamp: {batch_ts}')

    # ── Load data + materialize features ONCE (shared across baselines) ────
    # feature_scaling='none' here; per-baseline scalers are applied inside
    # _run_one_baseline() so each baseline can pick its own (e.g., LR =>
    # 'standard', LightGBM/k-NN => 'none').
    total_timer.begin_phase('load_data')
    CSV_ENGINE = 'python'
    train_pairs = pd.read_csv(dataset_dir / 'train_pairs.csv', engine=CSV_ENGINE)
    val_pairs   = pd.read_csv(dataset_dir / 'val_pairs.csv',   engine=CSV_ENGINE)
    test_pairs  = pd.read_csv(dataset_dir / 'test_pairs.csv',  engine=CSV_ENGINE)
    print(f'  train: {len(train_pairs):,} | val: {len(val_pairs):,} | test: {len(test_pairs):,}')

    if FEATURE_SOURCE == 'kmer':
        kmer_dir = build_embeddings_paths(
            project_root=project_root, virus_name=VIRUS_NAME,
            data_version=DATA_VERSION, run_suffix="", config=config,
        )['output_dir']
        embeddings_file = None
    else:  # esm2
        kmer_dir = None
        embeddings_file = build_training_paths(
            project_root=project_root, virus_name=VIRUS_NAME,
            data_version=DATA_VERSION, run_suffix="", config=config,
        )['embeddings_file']
        print(f'Embeddings file: {embeddings_file}')

    # output_dir for the SHARED feature load is a temporary scratch path
    # (the function may try to write a scaler there even though we pass
    # 'none'). We pass default_output_dir so any writes land in the runs
    # parent rather than the cwd.
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test), _ = (
        load_pair_features_for_baselines(
            train_pairs, val_pairs, test_pairs,
            feature_source=FEATURE_SOURCE,
            feature_scaling='none',  # raw; per-baseline scaling applied below
            kmer_dir=kmer_dir, kmer_k=KMER_K,
            embeddings_file=embeddings_file,
            interaction=INTERACTION,
            slot_transform=SLOT_TRANSFORM,
            output_dir=default_output_dir,
        )
    )
    total_timer.end_phase('load_data')
    print(f'\nMaterialized shared features: '
          f'X_train={X_train_raw.shape}, X_val={X_val_raw.shape}, X_test={X_test_raw.shape}')

    # ── Per-baseline runs ───────────────────────────────────────────────────
    n_baselines = len(baselines)
    results = []
    for baseline_name in baselines:
        t0 = time.time()
        try:
            res = _run_one_baseline(
                baseline_name=baseline_name,
                config=config, args=args,
                default_output_dir=default_output_dir,
                batch_ts=batch_ts, n_baselines=n_baselines,
                train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs,
                X_train_raw=X_train_raw, y_train=y_train,
                X_val_raw=X_val_raw,     y_val=y_val,
                X_test_raw=X_test_raw,   y_test=y_test,
                FEATURE_SOURCE=FEATURE_SOURCE, KMER_K=KMER_K,
                INTERACTION=INTERACTION, SLOT_TRANSFORM=SLOT_TRANSFORM,
                RANDOM_SEED=RANDOM_SEED, dataset_dir=dataset_dir,
            )
            res['wall_seconds'] = time.time() - t0
            results.append(res)
        except Exception as e:
            # Per-baseline isolation: one failure should not kill the others
            # in a multi-baseline batch. Emit the traceback and move on.
            import traceback
            print(f'\nERROR: baseline {baseline_name!r} FAILED after '
                  f'{time.time() - t0:.1f}s — {type(e).__name__}: {e}')
            traceback.print_exc()
            results.append({'baseline': baseline_name, 'error': repr(e),
                            'wall_seconds': time.time() - t0})

    # ── Final summary ──────────────────────────────────────────────────────
    total_timer.stop_timer()
    print('\n' + '=' * 72)
    print('BATCH SUMMARY')
    print('=' * 72)
    for r in results:
        if 'error' in r:
            print(f"  {r['baseline']:<14s}  FAILED  ({r['wall_seconds']:.1f}s)  {r['error']}")
        else:
            test_m = r['metrics']['test']
            print(f"  {r['baseline']:<14s}  test F1={test_m['f1']:.4f}  "
                  f"AUC={test_m['auc']:.4f}  ({r['wall_seconds']:.1f}s)  "
                  f"-> {r['output_dir']}")
    total_timer.display_timer()
    print(f'\nFinished {Path(__file__).name}!')


if __name__ == '__main__':
    main()
