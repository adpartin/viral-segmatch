#!/usr/bin/env python3
"""Parsl port of the all-pairs CV launcher (Phase 3; see the plan doc).

Replaces the bespoke mpiexec dispatch + run_cv_lambda.py hand-rolled `wait_any`
GPU pool with a Parsl HighThroughputExecutor: each fold is one `bash_app`
(reusing train_pair_classifier.py unchanged), and Parsl handles node/GPU
placement (`available_accelerators=4` -> one GPU per worker, CUDA_VISIBLE_DEVICES
set automatically) plus retries.

Why Parsl:
- ALCF-native: the Polaris config bakes in MpiExecLauncher(--depth=64) + a
  NUMA-aware `cpu_affinity` -- i.e. it encodes the CPU-binding fix the mpiexec
  launcher was missing (plan Phase 0c). Default binding made it ~5x slower.
- retries / failure isolation replace the manual re-run commands.
- one documented engine instead of ~250 lines of bash + a hand-rolled pool.

Validated on 2 pairs / 2 nodes vs the mpiexec numbers (~5 s/epoch). The
PBSProProvider submits its own PBS job -- run this from a login node.

Example:
  python3 scripts/run_allpairs_parsl.py \
    --pairs flu_28p_ha_na flu_28p_pb2_pb1 --nodes 2 --queue debug --epochs 20 \
    --tag parslval --dataset_manifest <RUN>/dataset_manifest.json
"""

import argparse
import json
from pathlib import Path

import parsl
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import PBSProProvider

PROJECT = Path("/lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch")
VENV = "/lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/cepi_polaris"

# Batch shells don't source dotfiles; bring up conda + venv + proxy + TMPDIR.
WORKER_INIT = (
    "module use /soft/modulefiles; module load conda; conda activate base; "
    f"source {VENV}/bin/activate; "
    "export TMPDIR=/tmp; "  # ALCF: avoids AF_UNIX 'path too long' on 1-node blocks
    "export http_proxy=http://proxy.alcf.anl.gov:3128; "
    "export https_proxy=http://proxy.alcf.anl.gov:3128"
)

# One GPU per worker, 4 workers/node, each pinned to a NUMA-local core block --
# the ALCF-recommended Polaris affinity, and exactly the CPU-binding the mpiexec
# launcher lacked (its default binding starved the host-bound folds ~5x).
CPU_AFFINITY = "list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39"


def make_config(nodes_per_block, account, queue, walltime, run_dir):
    return Config(
        run_dir=str(run_dir),
        retries=2,  # failure isolation: a killed fold re-runs automatically
        executors=[
            HighThroughputExecutor(
                label="htex_polaris",
                available_accelerators=4,  # one GPU/worker (sets CUDA_VISIBLE_DEVICES)
                max_workers_per_node=4,
                cpu_affinity=CPU_AFFINITY,
                provider=PBSProProvider(
                    account=account,
                    queue=queue,
                    walltime=walltime,
                    nodes_per_block=nodes_per_block,
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                    cpus_per_node=64,
                    select_options="ngpus=4",
                    # Polaris requires the filesystems directive:
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                    ),
                    worker_init=WORKER_INIT,
                ),
            )
        ],
    )


@bash_app
def train_fold(
    pair_bundle,
    fold,
    dataset_dir,
    run_subdir,
    project,
    epochs=None,
    stdout=None,
    stderr=None,
):
    """One fold = one training process on the node+GPU Parsl assigns.

    `project` is passed in (not a module global) so the app is self-contained
    on the worker; cuda:0 is the assigned GPU.
    """
    epoch_ovr = f"--override training.epochs={epochs}" if epochs else ""
    return (
        f"cd {project} && "
        f"python3 src/models/train_pair_classifier.py "
        f"--config_bundle {pair_bundle} --cuda_name cuda:0 "
        f"--dataset_dir {dataset_dir}/fold_{fold} "
        f"{epoch_ovr} --skip_post_hoc "
        f"--run_output_subdir {run_subdir}"
    )


def main():
    ap = argparse.ArgumentParser(description="Parsl all-pairs CV launcher (Phase 3)")
    ap.add_argument(
        "--dataset_manifest",
        required=True,
        help="JSON mapping bundle -> dataset dir (reuse; skips Stage 3)",
    )
    ap.add_argument(
        "--pairs", nargs="+", default=None, help="bundles; default = all in manifest"
    )
    ap.add_argument("--n_folds", type=int, default=12)
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--queue", default="debug")
    ap.add_argument("--account", default="IMPROVE_Aim1")
    ap.add_argument("--walltime", default="00:50:00")
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override training.epochs (else bundle default)",
    )
    ap.add_argument("--tag", default="parsl")
    args = ap.parse_args()

    with open(args.dataset_manifest) as f:
        manifest = json.load(f)
    pairs = args.pairs or sorted(manifest.keys())
    log_dir = PROJECT / "logs" / "parsl"
    log_dir.mkdir(parents=True, exist_ok=True)

    parsl.load(
        make_config(
            args.nodes, args.account, args.queue, args.walltime, log_dir / "runinfo"
        )
    )

    futures = []
    for pair in pairs:
        if pair not in manifest:
            print(f"WARNING: {pair} not in manifest; skipping")
            continue
        ds = manifest[pair]
        for fold in range(args.n_folds):
            sub = f"{pair}_{args.tag}_fold{fold}"
            fut = train_fold(
                pair,
                fold,
                ds,
                sub,
                str(PROJECT),
                epochs=args.epochs,
                stdout=str(log_dir / f"{sub}.out"),
                stderr=str(log_dir / f"{sub}.err"),
            )
            futures.append((pair, fold, fut))

    print(f"submitted {len(futures)} folds across {len(pairs)} pairs; waiting...")
    ok = fail = 0
    for pair, fold, fut in futures:
        try:
            fut.result()
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAILED {pair} fold{fold}: {e}")
    print(f"Parsl all-pairs done: {ok} ok / {fail} failed ({len(futures)} folds)")
    parsl.dfk().cleanup()


if __name__ == "__main__":
    main()
