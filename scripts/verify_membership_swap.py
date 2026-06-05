"""Verify the membership-backed cluster_map swap is bit-identical to the parquet path.

For every (alphabet, major protein, threshold), compares:
  - cluster_source.cluster_map(...)        -> membership-backed map
  - cluster_source._parquet_cluster_map(.) -> direct cluster-parquet read (correct key)
and asserts the two {hash: cluster_id} dicts are exactly equal. Also sanity-checks
that the HA-NA pair_key universe derived from the aa membership table matches the
`cds_final` universe (load_pair_universe), since the bigraph scripts still build the
universe from cds_final.

Exit code 0 iff every comparison matches.

    python -m scripts.verify_membership_swap
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis import cluster_source
from src.analysis.cluster_pair_weight_topk import load_pair_universe
from src.utils.config_hydra import load_function_metadata
from src.datasets._pair_helpers import canonical_pair_key

_ROOTS = {
    'aa': PROJ / 'data/processed/flu/July_2025/clusters_aa',
    'nt_cds': PROJ / 'data/processed/flu/July_2025/clusters_nt_cds',
}


def _thresholds(root: Path) -> list[str]:
    return sorted([d.name for d in root.iterdir()
                   if d.is_dir() and d.name.startswith('t') and d.name[1:].isdigit()],
                  reverse=True)


def main() -> None:
    majors = list(load_function_metadata(PROJ / 'conf/virus/flu.yaml').selected_short_names)
    checked = mism = 0
    for alphabet, root in _ROOTS.items():
        if not cluster_source.membership_available(alphabet):
            print(f"ERROR: membership table missing for {alphabet}; build it first.")
            sys.exit(2)
        for t in _thresholds(root):
            for P in majors:
                memb = cluster_source.cluster_map(alphabet, P, t)
                parq = cluster_source._parquet_cluster_map(root, P, t)
                checked += 1
                if memb != parq:
                    mism += 1
                    print(f"  MISMATCH [{alphabet} {P} {t}] "
                          f"memb={len(memb):,} parq={len(parq):,} "
                          f"keys_eq={set(memb)==set(parq)}")
        print(f"[{alphabet}] checked {checked} (protein x threshold) maps so far, {mism} mismatch")

    # Pair-universe sanity (HA-NA): membership-derived pair_key set vs cds_final.
    m = cluster_source.load_membership('aa')
    s2f = load_function_metadata(PROJ / 'conf/virus/flu.yaml').short_to_function
    ha = m.loc[m['function'] == s2f['HA'], ['assembly_id', 'seq_hash']].rename(columns={'seq_hash': 'a'})
    na = m.loc[m['function'] == s2f['NA'], ['assembly_id', 'seq_hash']].rename(columns={'seq_hash': 'b'})
    pr = ha.merge(na, on='assembly_id')
    memb_pk = {canonical_pair_key(x, y) for x, y in zip(pr['a'], pr['b'])}
    cds_pk = set(load_pair_universe(PROJ / 'data/processed/flu/July_2025/cds_final.parquet', 'HA', 'NA')['pair_key'])
    pk_ok = memb_pk == cds_pk
    print(f"\nHA-NA pair_key universe: membership={len(memb_pk):,} cds_final={len(cds_pk):,} identical={pk_ok}")

    print(f"\n{'PASS' if (mism == 0 and pk_ok) else 'FAIL'}: "
          f"{checked} cluster maps compared, {mism} mismatch; pair_key universe identical={pk_ok}")
    sys.exit(0 if (mism == 0 and pk_ok) else 1)


if __name__ == '__main__':
    main()
