#!/usr/bin/env python3
"""
Cross-validation launcher for the Lambda 8-GPU cluster (no job scheduler).

Workflow
--------
1. Run Stage 3 (dataset generation) once → creates fold_0/ … fold_{N-1}/ inside
   a single run directory.
2. Run Stage 4 (training) for each fold in parallel, one subprocess per GPU.
   Each process receives --dataset_dir <run_dir>/fold_<k> and
   CUDA_VISIBLE_DEVICES=<gpu_k> so it uses exactly one GPU.
3. Wait for all training processes to finish.
4. Run aggregate_cv_results.py to compute mean±std across folds and write
   cv_summary.csv / cv_summary.json.

Usage
-----
  python scripts/run_cv_lambda.py \\
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5

  python scripts/run_cv_lambda.py \\
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 \\
      --n_folds 5 --gpus 0 1 2 3 4

  # Skip dataset generation (dataset already exists):
  python scripts/run_cv_lambda.py \\
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 \\
      --skip_dataset \\
      --dataset_run_dir data/datasets/flu/July_2025/runs/dataset_paper_flu_..._cv5_20260225_120000

  # Dry-run (print commands, do not execute):
  python scripts/run_cv_lambda.py \\
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 --dry_run

  # Serial mode (one fold at a time, useful for debugging):
  python scripts/run_cv_lambda.py \\
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 --serial
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd: list[str], env: Optional[dict] = None, dry_run: bool = False) -> Optional[subprocess.Popen]:
    """Launch a subprocess (or print cmd in dry-run mode)."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  $ {cmd_str}")
    if dry_run:
        return None
    merged_env = {**os.environ, **(env or {})}
    return subprocess.Popen(cmd, env=merged_env)


def wait_all(procs: list[subprocess.Popen], fold_ids: list[int]) -> dict[int, int]:
    """Wait for all subprocesses; return {fold_id: exit_code}."""
    exit_codes = {}
    for fold_id, proc in zip(fold_ids, procs):
        proc.wait()
        exit_codes[fold_id] = proc.returncode
    return exit_codes


def n_folds_from_config(config_bundle: str) -> Optional[int]:
    """Try to read dataset.n_folds from the Hydra config bundle."""
    try:
        from src.utils.config_hydra import get_virus_config_hydra
        config_path = str(PROJECT_ROOT / "conf")
        cfg = get_virus_config_hydra(config_bundle, config_path=config_path)
        val = getattr(cfg.dataset, "n_folds", None)
        return int(val) if val is not None else None
    except Exception as e:
        print(f"⚠️  Could not read n_folds from config: {e}")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="CV launcher for Lambda cluster")
    p.add_argument("--config_bundle", required=True,
                   help="Hydra bundle name (e.g. flu_schema_raw_slot_norm_unit_diff_cv5)")
    p.add_argument("--n_folds", type=int, default=None,
                   help="Number of CV folds (default: read from config dataset.n_folds)")
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)),
                   help="GPU indices to use (default: 0..7). Cycled if fewer than n_folds.")
    p.add_argument("--skip_dataset", action="store_true",
                   help="Skip stage 3 (dataset already generated). Requires --dataset_run_dir.")
    p.add_argument("--dataset_run_dir", type=str, default=None,
                   help="Existing CV dataset run directory (required with --skip_dataset).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands but do not execute them.")
    p.add_argument("--serial", action="store_true",
                   help="Run training folds serially (one at a time) instead of in parallel.")
    p.add_argument("--skip_aggregate", action="store_true",
                   help="Skip the final aggregation step.")
    return p.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Resolve n_folds ────────────────────────────────────────────────────
    n_folds = args.n_folds or n_folds_from_config(args.config_bundle)
    if n_folds is None or n_folds < 2:
        print("❌ n_folds must be >= 2. Set --n_folds or add dataset.n_folds to the bundle.")
        sys.exit(1)
    print(f"\n{'='*60}")
    print(f"CV launcher  |  bundle={args.config_bundle}  n_folds={n_folds}")
    print(f"Mode: {'dry-run' if args.dry_run else ('serial' if args.serial else 'parallel')}")
    print(f"{'='*60}\n")

    # ── Stage 3: dataset generation ────────────────────────────────────────
    if args.skip_dataset:
        if not args.dataset_run_dir:
            print("❌ --skip_dataset requires --dataset_run_dir")
            sys.exit(1)
        dataset_run_dir = Path(args.dataset_run_dir)
        print(f"Skipping Stage 3. Using existing dataset dir: {dataset_run_dir}")
    else:
        run_id = f"dataset_{args.config_bundle.replace('/', '_')}_{timestamp}"
        print(f"Stage 3: generating {n_folds}-fold dataset (run_id={run_id})...")
        stage3_cmd = [
            "python",
            str(PROJECT_ROOT / "src" / "datasets" / "dataset_segment_pairs.py"),
            "--config_bundle", args.config_bundle,
            "--run_output_subdir", run_id,
        ]
        proc = run_cmd(stage3_cmd, dry_run=args.dry_run)
        if proc is not None:
            proc.wait()
            if proc.returncode != 0:
                print(f"❌ Stage 3 failed (exit code {proc.returncode}). Aborting.")
                sys.exit(proc.returncode)
            print("✅ Stage 3 complete.")

        # Derive the dataset run dir from the config paths
        if not args.dry_run:
            from src.utils.config_hydra import get_virus_config_hydra
            config_path = str(PROJECT_ROOT / "conf")
            cfg = get_virus_config_hydra(args.config_bundle, config_path=config_path)
            virus_name   = cfg.virus.virus_name
            data_version = cfg.virus.data_version
            dataset_run_dir = (
                PROJECT_ROOT / "data" / "datasets" / virus_name / data_version / "runs" / run_id
            )
        else:
            dataset_run_dir = Path(f"<dataset_run_dir for {run_id}>")

    print(f"Dataset run dir: {dataset_run_dir}")

    # Verify fold directories exist
    if not args.dry_run:
        for fold_i in range(n_folds):
            fold_dir = dataset_run_dir / f"fold_{fold_i}"
            if not fold_dir.exists():
                print(f"❌ Expected fold directory not found: {fold_dir}")
                sys.exit(1)
        print(f"✅ All {n_folds} fold directories present.\n")

    # ── Stage 4: training (one process per fold) ───────────────────────────
    training_run_dirs = {}
    train_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    procs, fold_ids = [], []
    for fold_i in range(n_folds):
        gpu = args.gpus[fold_i % len(args.gpus)]
        fold_dir = dataset_run_dir / f"fold_{fold_i}"
        run_id_train = f"training_{args.config_bundle.replace('/', '_')}_fold{fold_i}_{train_timestamp}"
        training_run_dirs[fold_i] = run_id_train

        print(f"Stage 4 fold {fold_i}/{n_folds-1}: GPU={gpu}  run_id={run_id_train}")
        stage4_cmd = [
            "python",
            str(PROJECT_ROOT / "src" / "models" / "train_esm2_frozen_pair_classifier.py"),
            "--config_bundle", args.config_bundle,
            "--cuda_name", f"cuda:{gpu}",
            "--dataset_dir", str(fold_dir),
            "--run_output_subdir", run_id_train,
        ]
        env_override = {"CUDA_VISIBLE_DEVICES": str(gpu)}

        if args.serial:
            proc = run_cmd(stage4_cmd, env=env_override, dry_run=args.dry_run)
            if proc is not None:
                proc.wait()
                if proc.returncode != 0:
                    print(f"❌ Training fold {fold_i} failed (exit code {proc.returncode}). Aborting.")
                    sys.exit(proc.returncode)
                print(f"✅ Fold {fold_i} training done.")
        else:
            proc = run_cmd(stage4_cmd, env=env_override, dry_run=args.dry_run)
            if proc is not None:
                procs.append(proc)
                fold_ids.append(fold_i)

    if not args.serial and not args.dry_run and procs:
        print(f"\nWaiting for {len(procs)} parallel training processes...")
        exit_codes = wait_all(procs, fold_ids)
        failed = [fid for fid, ec in exit_codes.items() if ec != 0]
        if failed:
            print(f"❌ Training failed for folds: {failed}")
            sys.exit(1)
        print("✅ All folds training complete.\n")

    # ── Save CV manifest ───────────────────────────────────────────────────
    if not args.dry_run:
        manifest = {
            "config_bundle":      args.config_bundle,
            "n_folds":            n_folds,
            "dataset_run_dir":    str(dataset_run_dir),
            "training_run_ids":   training_run_dirs,
            "launched_at":        train_timestamp,
        }
        manifest_path = dataset_run_dir / "cv_run_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved CV run manifest to: {manifest_path}")

    # ── Aggregation ────────────────────────────────────────────────────────
    if not args.skip_aggregate and not args.dry_run:
        print("\nRunning CV results aggregation...")
        agg_cmd = [
            "python",
            str(PROJECT_ROOT / "scripts" / "aggregate_cv_results.py"),
            "--manifest", str(dataset_run_dir / "cv_run_manifest.json"),
        ]
        proc = run_cmd(agg_cmd, dry_run=args.dry_run)
        if proc is not None:
            proc.wait()
            if proc.returncode != 0:
                print(f"⚠️  Aggregation failed (exit code {proc.returncode}). Check manually.")
            else:
                print("✅ Aggregation complete.")
    elif args.dry_run:
        print("\n[dry-run] Would run aggregate_cv_results.py")

    print(f"\n{'='*60}")
    print("CV run complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
