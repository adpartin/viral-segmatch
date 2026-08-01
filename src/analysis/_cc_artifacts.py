"""Read the persisted CC-structure artifacts (the Gen-2 input for the `bigraph_*` analyses).

`build_cc_structure.py` persists, per (cluster source, schema pair, threshold):

    data/processed/{virus}/{version}/cc_{source}/{pair}/tXXX/
        pairs_with_cc.parquet      pair_key + cluster_id_a/b + cc_id
        cc_sizes.csv               cc_id, n_pairs
        cc_cluster_composition.csv cc_id, slot, cluster_id, n_pairs, pct_of_cc
        cc_summary.json            n_ccs, largest-CC fraction, per-slot floor, max_balanced_k
        fragmented/                the same four, after the mega-CC edge cut

Analyses read those instead of rebuilding the pair universe and re-deriving clusters. Two reasons,
both concrete:

  - **Correctness.** The artifacts come from the production positive path
    (`create_positive_pairs_v2` via `dataset_pairs_cc.assign_atoms_prod`), so the analysis sees the
    pairs the splitter actually routes. The analysis-side `cluster_pair_weight_topk.
    load_pair_universe` defaults to an aa-keyed dedup, which for nt_cds collapses each protein pair
    onto one arbitrary CDS representative -- 58,826 HA-NA pairs against the production 78,764.
  - **Cost.** No ~100s Stage-3 front-end per run.

`load_cc_bigraph` returns the same weighted simple bigraph every other consumer uses, via the one
shared builder (`_bigraph.build_pair_bigraph`), so a Gen-2 analysis and the splitter agree by
construction rather than by coincidence.

Natural vs fragmented: pass the `tXXX/` dir for the pre-cut CCs, `tXXX/fragmented/` for the
post-edge-cut ones. Anything that *performs* a cut (e.g. `bigraph_min_cut`) wants the natural dir --
handing it `fragmented/` would cut an already-cut graph.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._bigraph import build_pair_bigraph  # noqa: E402

# Where build_cc_structure.py writes. `source` is the cluster root minus the `clusters_` prefix
# (e.g. clusters_nt_cds_cm0 -> nt_cds_cm0), so the cluster provenance is in the path.
PROCESSED = PROJ / 'data/processed/flu/July_2025'


def cc_dir(source: str, pair: str, threshold: str, *, fragmented: bool = False) -> Path:
    """Path to one artifact slice: `cc_{source}/{pair}/{threshold}[/fragmented]`.

    Args:
        source: cluster source, e.g. 'nt_cds_cm0' (the `clusters_` prefix dropped).
        pair: schema pair as written on disk, e.g. 'HA-NA'.
        threshold: 'tXXX'.
        fragmented: return the post-edge-cut slice instead of the natural one.
    """
    d = PROCESSED / f'cc_{source}' / pair / threshold
    return d / 'fragmented' if fragmented else d


def load_cc_pairs(d: Path) -> pd.DataFrame:
    """`pairs_with_cc.parquet` from one artifact dir, with a directive error if it is missing."""
    f = Path(d) / 'pairs_with_cc.parquet'
    if not f.exists():
        raise SystemExit(
            f"ERROR: no CC artifact at {f}.\n"
            f"Build it first, e.g.:\n"
            f"  python src/datasets/build_cc_structure.py \\\n"
            f"      --config_bundle flu_ha_na_cc_nt_cds_cm0_wf --thresholds {Path(d).name} --fragment"
        )
    return pd.read_parquet(f)


def load_cc_summary(d: Path) -> dict:
    """`cc_summary.json` from one artifact dir ({} if absent -- it is a convenience, not required)."""
    f = Path(d) / 'cc_summary.json'
    return json.loads(f.read_text()) if f.exists() else {}


def load_cc_bigraph(d: Path) -> tuple[nx.Graph, pd.DataFrame]:
    """Rebuild the cluster-level bigraph from a persisted CC slice.

    Returns `(H, pairs)`: the weighted simple bigraph (one node per cluster, slot-prefixed
    `a:`/`b:`, edge `weight` = positive pairs) and the underlying `pairs_with_cc` frame, whose
    `pair_key` joins the pair universe for anything needing per-pair columns.

    The graph is built by the shared `build_pair_bigraph`, not re-derived here, so its node order
    and edge weights match the splitter's exactly. `cc_id` in `pairs` is the recorded labelling
    from build time; recomputing components on `H` gives the same partition (labels may differ if
    the artifact predates the canonical `ranked_ccs` ordering).
    """
    pairs = load_cc_pairs(d)
    H, _edge_rows = build_pair_bigraph(
        pairs.reset_index(drop=True), col_a='cluster_id_a', col_b='cluster_id_b')
    return H, pairs


def add_cc_source_args(p, *, default_source: str = 'nt_cds_cm0',
                       default_pair: str = 'HA-NA') -> None:
    """Register the standard Gen-2 artifact selectors on an ArgumentParser.

    `--cc_source` / `--pair` / `--threshold(s)` locate the slice; `--cc_dir` overrides the whole
    path for an artifact tree outside the default layout.
    """
    p.add_argument('--cc_source', default=default_source,
                   help=f"cluster source under data/processed/.../cc_<source> "
                        f"(default {default_source}; the `clusters_` prefix dropped).")
    p.add_argument('--pair', default=default_pair,
                   help=f'schema pair directory name (default {default_pair}).')
    p.add_argument('--cc_dir', type=Path, default=None,
                   help='explicit artifact dir, overriding --cc_source/--pair/--threshold.')
