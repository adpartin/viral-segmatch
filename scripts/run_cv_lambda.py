#!/usr/bin/env python3
"""
Cross-validation launcher for the Lambda 8-GPU cluster (no job scheduler).

Workflow
--------
1. Run Stage 3 (dataset generation) once → creates fold_0/ … fold_{N-1}/ inside
   a single run directory.
2. Run Stage 4 (training) for each fold using a GPU pool: at most one fold
   per GPU at a time. When a fold finishes, its GPU is released and the next
   pending fold is launched on it. Each process receives
   --dataset_dir <run_dir>/fold_<k> and CUDA_VISIBLE_DEVICES=<gpu_k>.
3. Wait for all training processes to finish.
4. Run aggregate_cv_results.py to compute mean±std across folds and write
   cv_summary.csv / cv_summary.json.

Usage
-----
  python scripts/run_cv_lambda.py \
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv10

  python scripts/run_cv_lambda.py \
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv10 \
      --n_folds 5 --gpus 0 1 2 3 4

  # Skip dataset generation (dataset already exists):
  python scripts/run_cv_lambda.py \
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv10 \
      --skip_dataset \
      --dataset_run_dir data/datasets/flu/July_2025/runs/dataset_paper_flu_..._cv5_20260225_120000

  # Stage 3 only (generate folds, skip training):
  python scripts/run_cv_lambda.py \
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv10 --skip_training

  # Stage 4 only (training on existing folds, skip dataset generation):
  python scripts/run_cv_lambda.py \
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv10 \
      --skip_dataset \
      --dataset_run_dir data/datasets/flu/July_2025/runs/dataset_..._cv10_20260225_120000

  # Dry-run (print commands, do not execute):
  python scripts/run_cv_lambda.py \
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv10 --dry_run

  # Serial mode (one fold at a time, useful for debugging):
  python scripts/run_cv_lambda.py \
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv10 --serial

  # Re-run only specific failed folds:
  #   --folds: which folds to re-run
  #   --skip_dataset + --dataset_run_dir: reuse existing folds
  #   --cv_runs_dir: merge re-run results into the original CV results directory
  #     so the manifest is updated in place (new fold training dirs replace the
  #     failed ones) and aggregation runs over all folds, not just the re-run.
  #     Without --cv_runs_dir, a new cv_runs directory is created containing
  #     only the re-run folds, which makes aggregation incomplete.
  python scripts/run_cv_lambda.py \
      --config_bundle flu_schema_raw_slot_norm_unit_diff_cv10 \
      --skip_dataset \
      --dataset_run_dir data/datasets/flu/.../dataset_..._cv10_20260225_120000 \
      --folds 0 4 \
      --cv_runs_dir models/flu/.../cv_runs/cv_..._20260225_120000
"""

import argparse
import io
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

from src.utils.timer_utils import Timer


class TeeWriter:
    """Write to both a stream (e.g., stdout) and a log file simultaneously."""

    def __init__(self, stream, log_file: Path):
        self._stream = stream
        self._fh = open(log_file, "w")

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def close(self):
        self._fh.close()


def relay_subprocess_output(proc: subprocess.Popen) -> None:
    """Read piped subprocess output line-by-line and write to sys.stdout.

    This ensures subprocess output is captured by the TeeWriter when active.
    Waits for the process to finish before returning.
    """
    for line in iter(proc.stdout.readline, b""):
        sys.stdout.write(line.decode("utf-8", errors="replace"))
    proc.wait()


def run_cmd(
    cmd: list[str],
    env: Optional[dict] = None,
    dry_run: bool = False,
    log_file: Optional[Path] = None,
    ) -> tuple[Optional[subprocess.Popen], Optional[open]]:
    """Launch a subprocess (or print cmd in dry-run mode).

    If *log_file* is given, stdout and stderr are redirected to that file.
    Returns (process, file_handle) — caller must close file_handle when the
    process finishes.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  $ {cmd_str}")
    if log_file and not dry_run:
        print(f"    log → {log_file}")
    if dry_run:
        return None, None
    merged_env = {**os.environ, **(env or {})}
    fh = None
    kwargs: dict = {}
    if log_file:
        fh = open(log_file, "w")
        kwargs["stdout"] = fh
        kwargs["stderr"] = subprocess.STDOUT
    else:
        # Pipe stdout so callers can relay output through TeeWriter
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.Popen(cmd, env=merged_env, **kwargs), fh


def wait_any(active: dict[int, tuple[subprocess.Popen, int, Optional[open]]]) -> tuple[int, int, int]:
    """Poll until any process finishes. Returns (fold_id, gpu, exit_code)."""
    while True:
        for fold_id, (proc, gpu, fh) in list(active.items()):
            ret = proc.poll()
            if ret is not None:
                if fh is not None:
                    fh.close()
                del active[fold_id]
                return fold_id, gpu, ret
        time.sleep(1)


def n_folds_from_config(config_bundle: str) -> Optional[int]:
    """Try to read dataset.n_folds from the Hydra config bundle."""
    try:
        from src.utils.config_hydra import get_virus_config_hydra
        config_path = str(PROJECT_ROOT / "conf")
        cfg = get_virus_config_hydra(config_bundle, config_path=config_path)
        val = getattr(cfg.dataset, "n_folds", None)
        return int(val) if val is not None else None
    except Exception as e:
        print(f"WARNING: Could not read n_folds from config: {e}")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="CV launcher for Lambda cluster")
    p.add_argument("--config_bundle", required=True,
                   help="Hydra bundle name (e.g. flu_schema_raw_slot_norm_unit_diff_cv5)")
    p.add_argument("--n_folds", type=int, default=None,
                   help="Number of CV folds (default: read from config dataset.n_folds)")
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)),
                   help="GPU indices to use (default: 0..7). Folds are pooled across GPUs.")
    p.add_argument("--skip_dataset", action="store_true",
                   help="Skip stage 3 (dataset already generated). Requires --dataset_run_dir.")
    p.add_argument("--dataset_run_dir", type=str, default=None,
                   help="Existing CV dataset run directory (required with --skip_dataset).")
    p.add_argument("--skip_training", action="store_true",
                   help="Run stage 3 (dataset generation) only, then exit. "
                        "Useful for generating folds before submitting training to HPC.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands but do not execute them.")
    p.add_argument("--serial", action="store_true",
                   help="Run training folds serially (one at a time) instead of in parallel.")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="Run only these fold indices (e.g. --folds 0 4). "
                        "Requires --skip_dataset. Default: all folds.")
    p.add_argument("--cv_runs_dir", type=str, default=None,
                   help="Existing CV results directory to merge into when re-running "
                        "failed folds (e.g. models/.../cv_runs/cv_..._20260312_100000). "
                        "If not provided, a new cv_runs directory is created. "
                        "Use with --folds to update the manifest and re-aggregate.")
    p.add_argument("--skip_aggregate", action="store_true",
                   help="Skip the final aggregation step.")
    return p.parse_args()


def main():
    args = parse_args()
    cv_timer = Timer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Tee stdout/stderr to a log file ───────────────────────────────────
    # Log starts in logs/cv/ (known immediately). At the end, a copy is
    # placed in cv_runs_dir for co-location with CV results.
    log_dir = PROJECT_ROOT / "logs" / "cv"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_cv_{args.config_bundle.replace('/', '_')}_{timestamp}.log"
    tee_out = TeeWriter(sys.stdout, log_file)
    tee_err = TeeWriter(sys.stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err

    # ── Resolve n_folds ────────────────────────────────────────────────────
    n_folds = args.n_folds or n_folds_from_config(args.config_bundle)
    if n_folds is None or n_folds < 2:
        print("ERROR: n_folds must be >= 2. Set --n_folds or add dataset.n_folds to the bundle.")
        sys.exit(1)

    # ── Resolve fold list ────────────────────────────────────────────────
    if args.folds is not None:
        if not args.skip_dataset:
            print("ERROR: --folds requires --skip_dataset (dataset must already exist).")
            sys.exit(1)
        for f in args.folds:
            if f < 0 or f >= n_folds:
                print(f"ERROR: fold {f} out of range [0, {n_folds - 1}].")
                sys.exit(1)
        folds_to_run = sorted(args.folds)
    else:
        folds_to_run = list(range(n_folds))

    print(f"\n{'='*60}")
    print(f"CV launcher  |  bundle={args.config_bundle}  n_folds={n_folds}")
    if args.folds is not None:
        print(f"Running folds: {folds_to_run}  (subset)")
    print(f"Mode: {'dry-run' if args.dry_run else ('serial' if args.serial else 'parallel')}")
    print(f"{'='*60}\n")

    # ── Stage 3: dataset generation ────────────────────────────────────────
    if args.skip_dataset:
        if not args.dataset_run_dir:
            print("ERROR: --skip_dataset requires --dataset_run_dir")
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
        proc, _ = run_cmd(stage3_cmd, dry_run=args.dry_run)
        if proc is not None:
            # Relay subprocess output through Python stdout so TeeWriter captures it
            relay_subprocess_output(proc)
            if proc.returncode != 0:
                print(f"ERROR: Stage 3 failed (exit code {proc.returncode}). Aborting.")
                sys.exit(proc.returncode)
            print("Done. Stage 3 complete.")

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
            # Copy launcher log (which contains Stage 3 output) into dataset dir
            import shutil
            shutil.copy2(log_file, dataset_run_dir / "stage3_dataset.log")
        else:
            dataset_run_dir = Path(f"<dataset_run_dir for {run_id}>")

    print(f"Dataset run dir: {dataset_run_dir}")

    # Verify fold directories exist
    if not args.dry_run:
        for fold_i in folds_to_run:
            fold_dir = dataset_run_dir / f"fold_{fold_i}"
            if not fold_dir.exists():
                print(f"ERROR: Expected fold directory not found: {fold_dir}")
                sys.exit(1)
        print(f"All {len(folds_to_run)} fold directories present.\n")

    # ── Early exit if --skip_training ─────────────────────────────────────
    if args.skip_training:
        cv_timer.stop_timer()
        print(f"{'='*60}")
        print(f"Stage 3 complete. Skipping training (--skip_training).")
        print(f"Dataset run dir: {dataset_run_dir}")
        cv_timer.display_timer()
        print(f"Log: {log_file}")
        print(f"{'='*60}")
        sys.stdout = tee_out._stream
        sys.stderr = tee_err._stream
        tee_out.close()
        tee_err.close()
        return

    # ── Stage 4: training (GPU pool — one fold per GPU at a time) ─────────
    training_run_dirs = {}
    train_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve models base dir for per-fold training dirs and CV results dir
    if not args.dry_run:
        from src.utils.config_hydra import get_virus_config_hydra
        config_path = str(PROJECT_ROOT / "conf")
        cfg = get_virus_config_hydra(args.config_bundle, config_path=config_path)
        virus_name   = cfg.virus.virus_name
        data_version = cfg.virus.data_version
        models_base  = PROJECT_ROOT / "models" / virus_name / data_version / "runs"

        # CV results dir: reuse existing (--cv_runs_dir) or create new
        if args.cv_runs_dir:
            cv_runs_dir = Path(args.cv_runs_dir)
            if not cv_runs_dir.exists():
                print(f"ERROR: --cv_runs_dir does not exist: {cv_runs_dir}")
                sys.exit(1)
        else:
            cv_run_id   = f"cv_{args.config_bundle.replace('/', '_')}_{train_timestamp}"
            cv_runs_dir = PROJECT_ROOT / "models" / virus_name / data_version / "cv_runs" / cv_run_id
        cv_runs_dir.mkdir(parents=True, exist_ok=True)
    else:
        models_base = None
        cv_runs_dir = None

    def make_fold_cmd(fold_i: int, gpu: int) -> tuple[list[str], dict, str, Optional[Path]]:
        """Build the command, env, run_id, and fold_log path for a fold."""
        fold_dir = dataset_run_dir / f"fold_{fold_i}"
        run_id_train = f"training_{args.config_bundle.replace('/', '_')}_fold{fold_i}_{train_timestamp}"
        cmd = [
            "python",
            str(PROJECT_ROOT / "src" / "models" / "train_pair_classifier.py"),
            "--config_bundle", args.config_bundle,
            "--cuda_name", f"cuda:{gpu}",
            "--dataset_dir", str(fold_dir),
            "--run_output_subdir", run_id_train,
        ]
        env_override = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        # Pre-create training output dir so we can write the log file there
        fold_log = None
        if models_base is not None:
            train_output_dir = models_base / run_id_train
            train_output_dir.mkdir(parents=True, exist_ok=True)
            fold_log = train_output_dir / f"fold_{fold_i}.log"
        return cmd, env_override, run_id_train, fold_log

    exit_codes: dict[int, int] = {}

    if args.serial:
        # Serial mode: one fold at a time on first GPU
        for fold_i in folds_to_run:
            gpu = args.gpus[0]
            cmd, env_override, run_id_train, fold_log = make_fold_cmd(fold_i, gpu)
            training_run_dirs[fold_i] = run_id_train
            print(f"Stage 4 fold {fold_i}/{n_folds-1}: GPU={gpu}  run_id={run_id_train}")
            proc, fh = run_cmd(cmd, env=env_override, dry_run=args.dry_run, log_file=fold_log)
            if proc is not None:
                proc.wait()
                if fh is not None:
                    fh.close()
                exit_codes[fold_i] = proc.returncode
                if proc.returncode != 0:
                    print(f"ERROR: Training fold {fold_i} failed (exit code {proc.returncode})."
                          f" See log: {fold_log}")
                    sys.exit(proc.returncode)
                print(f"Done. Fold {fold_i} training complete.")
    else:
        # Parallel mode: GPU pool — at most one fold per GPU
        available_gpus = list(args.gpus)
        # active: {fold_id: (Popen, gpu, file_handle)}
        active: dict[int, tuple[subprocess.Popen, int, Optional[open]]] = {}
        pending_folds = list(folds_to_run)

        # Pre-register all run IDs (needed for manifest even in dry-run)
        for fold_i in folds_to_run:
            run_id_train = f"training_{args.config_bundle.replace('/', '_')}_fold{fold_i}_{train_timestamp}"
            training_run_dirs[fold_i] = run_id_train

        while pending_folds or active:
            # Launch folds on available GPUs
            while pending_folds and available_gpus:
                fold_i = pending_folds.pop(0)
                gpu = available_gpus.pop(0)
                cmd, env_override, run_id_train, fold_log = make_fold_cmd(fold_i, gpu)
                print(f"Stage 4 fold {fold_i}/{n_folds-1}: GPU={gpu}  run_id={run_id_train}")
                proc, fh = run_cmd(cmd, env=env_override, dry_run=args.dry_run, log_file=fold_log)
                if proc is not None:
                    active[fold_i] = (proc, gpu, fh)
                else:
                    # dry-run: gpu goes back immediately
                    available_gpus.append(gpu)

            if not active:
                break

            # Wait for any fold to finish, then reclaim its GPU
            fold_id, gpu, rc = wait_any(active)
            exit_codes[fold_id] = rc
            if rc == 0:
                print(f"Done. Fold {fold_id} training complete (GPU {gpu} now free).")
            else:
                fold_log = models_base / training_run_dirs[fold_id] / f"fold_{fold_id}.log" if models_base else None
                print(f"WARNING: Fold {fold_id} failed (exit code {rc}) on GPU {gpu}."
                      f" See log: {fold_log}")
            available_gpus.append(gpu)

        failed = [fid for fid, ec in exit_codes.items() if ec != 0]
        if failed:
            print(f"ERROR: Training failed for folds: {failed}")
            sys.exit(1)
        print(f"Done. All {len(folds_to_run)} folds training complete.\n")

    # ── Save CV manifest (in cv_runs dir, not dataset dir) ─────────────────
    if not args.dry_run:
        manifest_path = cv_runs_dir / "cv_run_manifest.json"

        # When re-running a subset of folds, merge with existing manifest
        # so we don't lose the other folds' training run IDs.
        if manifest_path.exists() and args.folds is not None:
            with open(manifest_path) as f:
                existing = json.load(f)
            existing_ids = existing.get("training_run_ids", {})
            # Update only the re-run folds (keys are strings in JSON)
            for fold_i, run_id in training_run_dirs.items():
                existing_ids[str(fold_i)] = run_id
            manifest = {
                "config_bundle":      args.config_bundle,
                "n_folds":            n_folds,
                "dataset_run_dir":    str(dataset_run_dir),
                "training_run_ids":   existing_ids,
                "launched_at":        train_timestamp,
            }
        else:
            manifest = {
                "config_bundle":      args.config_bundle,
                "n_folds":            n_folds,
                "dataset_run_dir":    str(dataset_run_dir),
                "training_run_ids":   {str(k): v for k, v in training_run_dirs.items()},
                "launched_at":        train_timestamp,
            }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"CV results dir: {cv_runs_dir}")
        print(f"Saved CV run manifest to: {manifest_path}")

    # ── Aggregation ────────────────────────────────────────────────────────
    if not args.skip_aggregate and not args.dry_run:
        print("\nRunning CV results aggregation...")
        agg_cmd = [
            "python",
            str(PROJECT_ROOT / "scripts" / "aggregate_cv_results.py"),
            "--manifest", str(cv_runs_dir / "cv_run_manifest.json"),
        ]
        proc, _ = run_cmd(agg_cmd, dry_run=args.dry_run)
        if proc is not None:
            relay_subprocess_output(proc)
            if proc.returncode != 0:
                print(f"WARNING: Aggregation failed (exit code {proc.returncode}). Check manually.")
            else:
                print("Done. Aggregation complete.")
    elif args.dry_run:
        print("\n[dry-run] Would run aggregate_cv_results.py")

    cv_timer.stop_timer()
    print(f"\n{'='*60}")
    print("CV run complete.")
    cv_timer.display_timer()
    if cv_runs_dir is not None and not args.dry_run:
        cv_timer.save_timer(cv_runs_dir)
    print(f"Log: {log_file}")
    print(f"{'='*60}")

    # ── Close tee and copy log to cv_runs_dir ─────────────────────────────
    sys.stdout = tee_out._stream
    sys.stderr = tee_err._stream
    tee_out.close()
    tee_err.close()
    if cv_runs_dir is not None and not args.dry_run:
        import shutil
        shutil.copy2(log_file, cv_runs_dir / "cv_run.log")


if __name__ == "__main__":
    main()
