"""Train LightGBM once per frozen-fold balanced negative redraw."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Fit one model for each seed directory and retain its training log.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--config-bundle', required=True)
    parser.add_argument('--run-prefix', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    model_root = project / 'models/flu/July_2025/runs'
    for seed in args.seeds:
        dataset_dir = args.dataset_root / f'seed_{seed}'
        run_name = f'{args.run_prefix}_seed{seed}'
        output_dir = model_root / run_name
        if (output_dir / 'metrics_summary.json').exists():
            print(f'Done. {run_name} already has metrics; skipping.', flush=True)
            continue
        command = [
            sys.executable, str(project / 'src/models/train_pair_baselines.py'),
            '--config_bundle', args.config_bundle, '--baseline', 'lgbm',
            '--dataset_dir', str(dataset_dir), '--run_output_subdir', run_name,
            '--skip_post_hoc',
        ]
        print(f'Running {run_name}', flush=True)
        result = subprocess.run(command, cwd=project, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        if output_dir.exists():
            (output_dir / 'redraw_train.log').write_text(result.stdout)
        if result.returncode or not (output_dir / 'metrics_summary.json').exists():
            print(result.stdout[-4000:], flush=True)
            raise RuntimeError(f'{run_name} failed with exit code {result.returncode}')
        print(f'Done. {run_name}', flush=True)


if __name__ == '__main__':
    main()
