"""Train one baseline fit per seed directory of a frozen-fold balanced negative redraw.

Each `seed_{s}` directory under `--dataset-root`, as `resample_balanced_train_negatives.py` writes
it, is fitted once, with `--neg_redraw_seed {s}` recorded in its `training_info.json`. Fits are
launched through `src/models/baseline_fits.py`, the code every training runner uses.

How?

- `--override` entries are passed to the trainer and composed into the planned config, so a
  redraw of a 5-fold dataset can be fitted with the same config as the dataset's own folds.
- The dataset's `resolved_config.yaml` defaults to the one next to the source fold that
  `redraw_summary.json` names; its checksum is recorded with the fit's inputs.
- An existing run directory is skipped only when it is the finished fit of the planned job: its
  saved config must equal the planned one, and its recorded input checksums and commit must equal
  the files on disk and the current commit. Anything else stops the script before that fit, since
  the trainer would overwrite it.

CLI:
    python scripts/run_balanced_train_redraw.py \
        --dataset-root data/datasets/flu/July_2025/runs/<redraw>/fold_0 \
        --config-bundle flu_28p_codon_pb2_ha --run-prefix lgbm_<redraw>_fold0 \
        --seeds 42 43 --override dataset.n_folds=5

Outputs:
    models/{virus}/{version}/runs/{run_prefix}_seed{s}/    one fit per seed, with redraw_train.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_fits import existing_fit_problems, fit_command, run_fit  # noqa: E402
from src.utils.config_hydra import compose_config  # noqa: E402
from src.utils.dataset_manifest import SPLIT_FILES, site_cache_files  # noqa: E402
from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402


def current_input_checksums(seed_dir: Path, dataset_resolved_config: Path, config) -> dict:
    """Checksums of the files a fit of this seed directory reads, as they are on disk now.

    Args:
      seed_dir: one redraw seed directory.
      dataset_resolved_config: the dataset's `resolved_config.yaml`.
      config: the planned fit config, which names the site-feature caches.

    Returns:
      Absolute path -> SHA-256, in the form the trainer records.
    """
    paths = [seed_dir / name for name in SPLIT_FILES if (seed_dir / name).exists()]
    paths += [dataset_resolved_config, *site_cache_files(config)]
    checksums = {str(Path(path).resolve()): file_sha256(path) for path in paths}
    return checksums


def default_dataset_config(dataset_root: Path) -> Path:
    """The dataset `resolved_config.yaml` next to the source fold of a redraw.

    Args:
      dataset_root: a redraw's fold directory, holding `redraw_summary.json`.

    Returns:
      `{source_fold}/../resolved_config.yaml`.
    """
    summary = json.loads((dataset_root / 'redraw_summary.json').read_text())
    path = Path(summary['source_fold']).parent / 'resolved_config.yaml'
    return path


def main(argv: Optional[list] = None) -> None:
    """Fit one model for each seed directory and retain its training log.

    Args:
      argv: command-line arguments; None reads `sys.argv`.

    Returns:
      None. Exits non-zero when a fit fails or an existing run directory is not the planned fit.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--config-bundle', required=True)
    parser.add_argument('--run-prefix', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--override', nargs='+', default=[],
                        help='dotlist overrides passed to the trainer, e.g. dataset.n_folds=5')
    parser.add_argument('--dataset-resolved-config', type=Path, default=None,
                        help="the dataset's resolved_config.yaml; default: the one next to the "
                             "source fold named in redraw_summary.json")
    parser.add_argument('--baseline', default='lgbm')
    args = parser.parse_args(argv)

    dataset_config = args.dataset_resolved_config or default_dataset_config(args.dataset_root)
    planned_config = compose_config(args.config_bundle, args.override)
    model_root = (PROJECT_ROOT / 'models' / planned_config.virus.virus_name
                  / planned_config.virus.data_version / 'runs')
    commit = get_git_info()['commit']
    for seed in args.seeds:
        dataset_dir = args.dataset_root / f'seed_{seed}'
        run_name = f'{args.run_prefix}_seed{seed}'
        output_dir = model_root / run_name
        expected_inputs = current_input_checksums(dataset_dir, dataset_config, planned_config)
        if output_dir.exists():
            problems = existing_fit_problems(output_dir, planned_config, expected_inputs, commit,
                                             neg_redraw_seed=seed)
            if problems:
                raise SystemExit(f'ERROR: {run_name} exists but is not the planned fit: {problems}')
            print(f'Done. {run_name} already finished; skipping.', flush=True)
            continue
        command = fit_command(args.config_bundle, args.baseline, dataset_dir, output_dir,
                              args.override, dataset_config, neg_redraw_seed=seed)
        print(f'Running {run_name}', flush=True)
        exit_code = run_fit(command, output_dir, log_name='redraw_train.log')
        problems = ([f'exit {exit_code}'] if exit_code else
                    existing_fit_problems(output_dir, planned_config, expected_inputs, commit,
                                          neg_redraw_seed=seed))
        if problems:
            raise SystemExit(f'ERROR: {run_name} failed: {problems}; see '
                             f'{output_dir / "redraw_train.log"}')
        print(f'Done. {run_name}', flush=True)


if __name__ == '__main__':
    main()
