"""Back-compat shim: set-cover cluster builder (easy-linclust / easy-cluster).

The builder was unified into `src/preprocess/build_clusters.py`; this module preserves the old
entry point, defaulting `--method linclust`. Pass `--method cluster` (or `--method search`) to
override. See `build_clusters.py` for the full CLI, the unified driver, and the arg matrix.

Breaking changes vs the pre-unification CLI: `--algorithm {linclust,cluster}` is gone (use
`--method`); the `--results_md` / `redundancy_summary.md` report was dropped; the stats file is
now `cluster_stats.csv` (with `method` + `cluster_mode` columns) instead of `redundancy_stats.csv`.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.build_clusters import main  # noqa: E402

if __name__ == '__main__':
    main(default_method='linclust')
