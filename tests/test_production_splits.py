"""End-to-end guard for the two production split paths (§5 P4.1 of
`docs/plans/2026-08-03_fold_maker_consolidation_plan.md`).

Each test rebuilds one production bundle in full and diffs its per-fold `pair_key` digests
against `tests/golden/production_splits/`. The build and the digest come from
`scripts/production_split_harness.py`, invoked as a subprocess, so this file cannot drift from
the script you run by hand.

**Deselected by default** via the `production_split` marker: each build takes minutes and needs
the Stage-1 corpus. Run them with:

    pytest -m production_split

A failure means either the split code moved pairs (the regression this catches) or the corpus was
rebuilt (legitimate — re-capture the goldens). The harness prints which, and how.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
HARNESS = PROJ / 'scripts' / 'production_split_harness.py'
GOLDEN_DIR = PROJ / 'tests' / 'golden' / 'production_splits'
PROCESSED = PROJ / 'data' / 'processed' / 'flu' / 'July_2025'

# Stage-1 outputs and the cluster parquet both production bundles route on. Absent on a machine
# that has not run Stage 1, in which case these tests skip rather than fail.
CORPUS = (
    PROCESSED / 'protein_final.parquet',
    PROCESSED / 'cds_dna_final.parquet',
    PROCESSED / 'clusters_nt_cds_cm0' / 't099' / 'combined_cluster.parquet',
)


def _run_guard(name: str):
    """Run `production_split_harness.py check --only <name>`; assert it reports bit-exact."""
    # Report paths relative to the repo where possible, but never let the skip message itself
    # raise: a CORPUS entry outside PROJ makes relative_to throw.
    missing = []
    for path in CORPUS:
        if path.exists():
            continue
        try:
            missing.append(str(path.relative_to(PROJ)))
        except ValueError:
            missing.append(str(path))
    if missing:
        pytest.skip(f'Stage-1 corpus absent: {missing}')
    if not (GOLDEN_DIR / f'{name}.json').exists():
        pytest.skip(f'no golden for {name}; capture it with the harness first')

    proc = subprocess.run([sys.executable, str(HARNESS), 'check', '--only', name],
                          cwd=PROJ, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f'{name} is no longer bit-exact\n'
                    f'--- harness stdout ---\n{proc.stdout}\n'
                    f'--- harness stderr ---\n{proc.stderr}')


@pytest.mark.production_split
def test_2d_cd_t099_is_bit_exact():
    """2D-CD, nt_cds t099 HA-NA (`flu_ha_na_cc_nt_cds_cm0_wf`)."""
    _run_guard('2d_cd_t099')


@pytest.mark.production_split
def test_1d_cd_ha_t099_is_bit_exact():
    """1D-CD on the HA axis, nt_cds t099 HA-NA (`flu_ha_na_1dcd_nt_cds`)."""
    _run_guard('1d_cd_ha_t099')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'production_split'])
