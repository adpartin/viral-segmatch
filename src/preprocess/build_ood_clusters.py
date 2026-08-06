"""Back-compat shim: OOD cluster builder (mmseqs easy-search all-vs-all -> union-find CCs).

The builder was unified into `src/preprocess/build_clusters.py`; this module preserves the old
entry point, defaulting `--method search`. See `build_clusters.py` for the full CLI, the unified
driver, and the arg matrix.

Method (unchanged): per-(function, threshold) clusters are the connected components of the
per-function SIMILARITY graph (nodes = unique sequences; edge = an mmseqs easy-search hit at
>= t identity AND >= -c coverage), computed by union-find. Putting whole components on one fold
then guarantees no test sequence links to any train sequence -- the across-cluster separation
("across clusters: different") that a cluster-disjoint / OOD split needs. Component here =
single-segment similarity-graph CC (a *cluster* / *mega-cluster*), NOT the CC / mega-CC
of 2D-CD routing (docs/methods/glossary.md).

Contrast with `--method linclust`/`cluster` (set-cover), which do NOT give the guarantee -- and
with `--method cluster --cluster-mode 1`, whose connected-component assignment also lacks it (the
prefilter can miss edges). search is the slower path that carries the guarantee.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.build_clusters import main  # noqa: E402

if __name__ == '__main__':
    main(default_method='search')
