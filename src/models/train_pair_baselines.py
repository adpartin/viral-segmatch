"""Stage 4 entry point for sklearn-style baselines (logistic regression, ...).

Runs ALONGSIDE the MLP trainer (it does not replace it). Reads the same
Stage 3 dataset directory and the same bundle config; writes its own flat
run directory under ``models/{virus}/{data_version}/runs/``.

Usage:
    python src/models/train_pair_baselines.py \\
        --config_bundle flu_ha_na \\
        --baseline logistic \\
        --dataset_dir data/datasets/.../runs/dataset_flu_ha_na_<TS>

Output (one run dir per call):
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
    'logistic': 'src.models.baselines.logistic',
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


def main() -> None:
    total_timer = Timer()
    phase_timers: dict[str, Timer] = {}

    def begin_phase(label: str) -> Timer:
        t = Timer()
        phase_timers[label] = t
        print(f'\n[phase {label}] starting...')
        return t

    def end_phase(label: str) -> None:
        t = phase_timers[label]
        t.stop_timer()
        h, m, s = t.hms
        print(f'[phase {label}] elapsed: {h:02}:{m:02}:{s:02} ({t.elapsed:.1f}s)')

    parser = argparse.ArgumentParser(description='Train sklearn-style pair baseline')
    parser.add_argument('--config_bundle', type=str, required=True,
                        help='Config bundle name (e.g., flu_ha_na).')
    parser.add_argument('--baseline', type=str, required=True,
                        choices=sorted(BASELINE_REGISTRY.keys()),
                        help=f'Baseline name. One of {sorted(BASELINE_REGISTRY)}.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Stage 3 dataset directory containing train/val/test_pairs.csv.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory. Default: derived from config + run_output_subdir.')
    parser.add_argument('--run_output_subdir', type=str, default=None,
                        help='Run-id subdirectory under default output_dir.')
    parser.add_argument('--cuda_name', type=str, default=None,
                        help='Accepted for shell-script symmetry; ignored by sklearn baselines.')
    parser.add_argument('--override', type=str, nargs='+', default=None,
                        help='Hydra dotlist overrides on top of the bundle.')
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

    if RANDOM_SEED is not None:
        set_deterministic_seeds(RANDOM_SEED, cuda_deterministic=False)
        print(f'Set deterministic seeds (seed: {RANDOM_SEED})')

    baseline_module = _resolve_baseline_module(args.baseline)
    feature_scaling = _resolve_feature_scaling(config, baseline_module)

    # ── Paths ───────────────────────────────────────────────────────────────
    paths = build_training_paths(
        project_root=project_root, virus_name=VIRUS_NAME,
        data_version=DATA_VERSION, run_suffix="", config=config,
    )
    default_output_dir = paths['output_dir']

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.run_output_subdir:
        output_dir = default_output_dir / 'runs' / args.run_output_subdir
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fallback_run_id = f"baseline_{baseline_module.name()}_{args.config_bundle}_{ts}"
        output_dir = default_output_dir / 'runs' / fallback_run_id
        print(f"WARNING: No run_output_subdir provided, using fallback: {fallback_run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, str(output_dir / 'resolved_config.yaml'))

    dataset_dir = Path(args.dataset_dir)

    print(f'\nConfig bundle:   {args.config_bundle}')
    print(f'Baseline:        {baseline_module.name()}')
    print(f'Feature source:  {FEATURE_SOURCE}')
    print(f'Feature scaling: {feature_scaling}')
    print(f'Dataset dir:     {dataset_dir}')
    print(f'Output dir:      {output_dir}')  # parsed by stage4_baselines.sh

    # ── Load data (CSVs + features + optional scaler) ──────────────────────
    begin_phase('load_data')
    CSV_ENGINE = 'python'  # match trainer (avoids C parser segfault on certain seqs)
    train_pairs = pd.read_csv(dataset_dir / 'train_pairs.csv', engine=CSV_ENGINE)
    val_pairs   = pd.read_csv(dataset_dir / 'val_pairs.csv',   engine=CSV_ENGINE)
    test_pairs  = pd.read_csv(dataset_dir / 'test_pairs.csv',  engine=CSV_ENGINE)
    print(f'  train: {len(train_pairs):,} | val: {len(val_pairs):,} | test: {len(test_pairs):,}')

    if FEATURE_SOURCE == 'kmer':
        kmer_dir = build_embeddings_paths(
            project_root=project_root, virus_name=VIRUS_NAME,
            data_version=DATA_VERSION, run_suffix="", config=config,
        )['output_dir']
    else:
        kmer_dir = None  # ESM-2 path will raise NotImplementedError inside the loader.

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = load_pair_features_for_baselines(
        train_pairs, val_pairs, test_pairs,
        feature_source=FEATURE_SOURCE,
        feature_scaling=feature_scaling,
        kmer_dir=kmer_dir, kmer_k=KMER_K,
        output_dir=output_dir,
    )
    end_phase('load_data')

    # ── Fit ─────────────────────────────────────────────────────────────────
    begin_phase('fit')
    estimator = baseline_module.get_estimator(config, random_state=RANDOM_SEED)
    print(f'Fitting {type(estimator).__name__}: {estimator}')
    estimator.fit(X_train, y_train)
    model_path = output_dir / 'best_model.joblib'
    joblib.dump(estimator, model_path)
    print(f'Saved fitted model to: {model_path}')
    end_phase('fit')

    # ── Inference ──────────────────────────────────────────────────────────
    begin_phase('inference')
    val_probs   = estimator.predict_proba(X_val)[:, 1]
    test_probs  = estimator.predict_proba(X_test)[:, 1]
    train_probs = estimator.predict_proba(X_train)[:, 1]

    # Optional decision_function (logits-equivalent for LR). Useful for diagnostics.
    if hasattr(estimator, 'decision_function'):
        train_logits = np.asarray(estimator.decision_function(X_train), dtype=np.float32)
        val_logits   = np.asarray(estimator.decision_function(X_val),   dtype=np.float32)
        test_logits  = np.asarray(estimator.decision_function(X_test),  dtype=np.float32)
    else:
        train_logits = val_logits = test_logits = None
    end_phase('inference')

    # ── Eval (threshold + metrics + per-split prediction CSVs) ─────────────
    begin_phase('eval')

    # ── Threshold selection (mirrors MLP) ───────────────────────────────────
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

    # ── Metrics + per-split prediction CSVs ─────────────────────────────────
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
    end_phase('eval')

    # ── Training info (parallels MLP's; null for inapplicable fields) ───────
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
        # MLP-only knobs, recorded as null for downstream tooling parity.
        'interaction': 'concat',
        'slot_transform': None,
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

    print(f'\nFinished {Path(__file__).name}!')
    total_timer.stop_timer()
    total_timer.display_timer()

    # Per-phase timing summary (also persisted via total_timer.save_timer below).
    phase_seconds = {label: round(t.elapsed, 3) for label, t in phase_timers.items()}
    print('Phase timings (seconds):')
    for label, secs in phase_seconds.items():
        print(f'  {label:<10s} {secs:>9.1f}s')
    total_timer.save_timer(output_dir, extra={'phases_seconds': phase_seconds})


if __name__ == '__main__':
    main()
