"""Synthetic validation tests for DataSAIL's eval_split / L(π).

DataSAIL's `__init__.py` doesn't export `eval_split` and the package
doesn't ship a documented example for the function. This script runs
three small hand-computable tests to confirm we are calling
`eval_split` correctly and interpreting its output the way we think
we are. Run before trusting the production wrapper
`datasail_lpi_measure.py` on real data.

**Why a precomputed similarity matrix, not `similarity='mmseqs'`?**
DataSAIL's `cluster_param_binary_search` hardcodes a target window of
`10 < n_clusters <= 100`. With fewer than 10 entities, the binary
search returns a **zero similarity matrix** (`cluster/utils.py:76`:
`np.zeros((len(dataset.names), len(dataset.names)))`), giving
total = 0 and ratio = nan. So `similarity='mmseqs'` cannot be
validated on tiny synthetic inputs. Instead, we pass a precomputed
similarity matrix as a `(names, matrix)` tuple — `read_matrix_input`
(reader/utils.py:229) accepts this format directly, and
`clustering.py:48-53` short-circuits the binary search when
similarity is already an ndarray.

Using a 3×3 all-1s similarity matrix (every entity "fully similar"
to every other entity) gives clean hand-computable ratios:

    Test 1 (all in train):           0 / 9 = 0.000
    Test 2 (2 train, 1 test):        4 / 9 ≈ 0.444
    Test 3 (1 train, 1 val, 1 test): 6 / 9 ≈ 0.667

The leakage formula reduces to:
    ratio = (sum of cross-split cells in 1-mask) × weight^2 / (N^2 × weight^2)
          = (number of cross-split off-diagonal cells in 3×3 matrix) / 9

If `eval_split` returns these three ratios exactly, the function is
behaving as documented and our wrapper can be trusted.

Run with:

    conda run -n datasail python src/analysis/datasail_lpi_validate.py

Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from datasail.eval import eval_split
except ImportError as e:
    sys.stderr.write(
        f"ERROR: cannot import datasail ({e}). This script must run in the "
        f"'datasail' conda env. Try:\n"
        f"    conda run -n datasail python {sys.argv[0]}\n"
    )
    sys.exit(1)


# 3 entities with all-1s similarity matrix (every pair fully similar).
# Sequences don't matter because we pass similarity precomputed.
ENTITY_NAMES = ['entity_K', 'entity_A', 'entity_D']
SIMILARITY_MATRIX = np.ones((3, 3), dtype=float)
DUMMY_SEQUENCES = {name: 'M' for name in ENTITY_NAMES}  # data dict required by reader


def run_eval(split_assignment: dict, verbose: bool = False
              ) -> tuple[float, float, float]:
    """Thin wrapper around eval_split with our synthetic data + precomputed sim."""
    ratio, absolute, total = eval_split(
        datatype='P',
        data=dict(DUMMY_SEQUENCES),
        weights=None,
        similarity=(list(ENTITY_NAMES), SIMILARITY_MATRIX.copy()),
        distance=None,
        dist_conv=None,
        split_assignment=split_assignment,
    )
    if verbose:
        print(f"    ratio={ratio:.6f}  absolute={absolute:.4f}  total={total:.4f}")
    return float(ratio), float(absolute), float(total)


def assert_close(actual: float, expected: float, tol: float = 1e-6,
                  context: str = '') -> bool:
    """Return True if |actual - expected| <= tol; print PASS/FAIL line."""
    ok = abs(actual - expected) <= tol
    status = 'PASS' if ok else 'FAIL'
    diff = abs(actual - expected)
    print(f"  [{status}] {context}: actual={actual:.6f}  expected={expected:.6f}  "
          f"|diff|={diff:.2e}  tol={tol:.0e}")
    return ok


def main():
    print("=" * 70)
    print("DataSAIL eval_split — synthetic validation")
    print("=" * 70)
    print(f"Inputs: 3 entities with precomputed 3x3 all-1s similarity matrix,")
    print(f"        uniform weights, datatype='P'.")
    print()

    all_pass = True

    # Test 1: all entities in train -> no leakage
    print("Test 1: all 3 entities in 'train'")
    print("  Expected ratio = 0/9 = 0.000  (everything in one split, no leakage)")
    split1 = {'entity_K': 'train', 'entity_A': 'train', 'entity_D': 'train'}
    ratio1, abs1, total1 = run_eval(split1, verbose=True)
    all_pass &= assert_close(ratio1, 0.0,
                              context='Test 1 (all-train)')
    print()

    # Test 2: 2 train, 1 test -> ratio = 4/9
    print("Test 2: 2 entities in 'train', 1 in 'test'")
    print("  Expected ratio = 4/9 ≈ 0.444  (4 cross-split off-diagonal cells)")
    split2 = {'entity_K': 'train', 'entity_A': 'train', 'entity_D': 'test'}
    ratio2, abs2, total2 = run_eval(split2, verbose=True)
    all_pass &= assert_close(ratio2, 4.0 / 9.0,
                              context='Test 2 (2-train/1-test)')
    print()

    # Test 3: 1 train, 1 val, 1 test -> ratio = 6/9
    print("Test 3: 1 entity per split (train/val/test)")
    print("  Expected ratio = 6/9 ≈ 0.667  (6 cross-split off-diagonal cells)")
    split3 = {'entity_K': 'train', 'entity_A': 'val', 'entity_D': 'test'}
    ratio3, abs3, total3 = run_eval(split3, verbose=True)
    all_pass &= assert_close(ratio3, 6.0 / 9.0,
                              context='Test 3 (1/1/1)')
    print()

    # Ordering sanity check
    print("Ordering check: ratio(Test1) <= ratio(Test2) <= ratio(Test3)")
    ord_ok = (ratio1 <= ratio2 <= ratio3)
    status = 'PASS' if ord_ok else 'FAIL'
    print(f"  [{status}] {ratio1:.4f} <= {ratio2:.4f} <= {ratio3:.4f}")
    all_pass &= ord_ok
    print()

    # Summary
    print("=" * 70)
    if all_pass:
        print("OVERALL: PASS — eval_split returns expected ratios exactly.")
        print()
        print("Note: this validates the math on a precomputed similarity matrix.")
        print("Real runs use similarity='mmseqs', which requires >10 entities for")
        print("the internal binary search; that path is not validated here, but")
        print("the L(π) computation downstream of the similarity matrix is the same.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — see individual test results above. Investigate before")
        print("                trusting datasail_lpi_measure.py on real data.")
        sys.exit(1)


if __name__ == '__main__':
    main()
