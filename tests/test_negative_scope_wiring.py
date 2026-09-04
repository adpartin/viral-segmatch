"""Every path that builds a dataset must forward `negative_scope`.

Regression test for a real defect: the single-split branch of `dataset_segment_pairs.py` called
`split_dataset_v2` without `negative_scope`, so a bundle asking for `within_fold` silently got the
default `coverage` sampler instead. On H3N2 2024 that produced 492 test negatives against 358
requested, which moved precision by about 0.039 through class balance alone. The CV branch passed
the argument correctly, so the two call sites disagreed and nothing failed loudly.

The check is on the source rather than on a built dataset, because building one takes about a
minute and needs the full corpus, while the defect is entirely a matter of which keyword reaches
the call. Parsing the module is enough to catch a dropped argument, and it costs nothing.

`dataset_segment_pairs.py` runs its work at import, so it cannot be imported here. It is read and
parsed instead.

Covers:
  1. Every `split_dataset_v2(...)` call in the orchestrator passes `negative_scope`
  2. Every `generate_all_*_cv_folds_v2(...)` call does too
  3. `split_dataset_v2` still declares the parameter, so the tests above are checking a real name

Run: python tests/test_negative_scope_wiring.py
"""
import ast
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

ORCHESTRATOR = PROJ / 'src/datasets/dataset_segment_pairs.py'
BUILDER = PROJ / 'src/datasets/dataset_segment_pairs_v2.py'

# Every callee that takes negative_scope and decides which negative sampler runs.
ROUTING_CALLS = (
    'split_dataset_v2',
    'generate_all_cv_folds_v2',
    'generate_all_cluster_disjoint_cv_folds_v2',
)


def _calls_named(tree: ast.AST, name: str) -> list:
    """Every Call node in `tree` whose callee is the bare function `name`."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            found.append(node)
    return found


def test_every_routing_call_forwards_negative_scope():
    tree = ast.parse(ORCHESTRATOR.read_text())
    checked = 0
    for name in ROUTING_CALLS:
        for call in _calls_named(tree, name):
            checked += 1
            keywords = {kw.arg for kw in call.keywords}
            assert 'negative_scope' in keywords, (
                f"{ORCHESTRATOR.name}:{call.lineno}: {name}(...) does not pass negative_scope, so "
                f"a bundle that sets split_strategy.negative_scope would silently fall back to "
                f"the default sampler.")
    # Guard the guard: if the call sites are ever renamed, this test must not pass vacuously.
    assert checked >= 3, (
        f"expected at least 3 dataset-routing calls in {ORCHESTRATOR.name}, found {checked}. "
        f"Update ROUTING_CALLS if the orchestrator was restructured.")


def test_split_dataset_v2_still_takes_negative_scope():
    tree = ast.parse(BUILDER.read_text())
    signatures = {node.name: node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name in ROUTING_CALLS}
    assert set(signatures) == set(ROUTING_CALLS), (
        f"missing definitions in {BUILDER.name}: {set(ROUTING_CALLS) - set(signatures)}")
    for name, node in signatures.items():
        parameters = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        assert 'negative_scope' in parameters, (
            f"{BUILDER.name}: {name} no longer takes negative_scope; the wiring test above is "
            f"checking a name that does not exist.")


if __name__ == '__main__':
    tests = [
        test_every_routing_call_forwards_negative_scope,
        test_split_dataset_v2_still_takes_negative_scope,
    ]
    failed = 0
    for t in tests:
        try:
            print(f'... {t.__name__}')
            t()
            print('    OK')
        except Exception as e:
            failed += 1
            print(f'    FAIL: {e}')
    if failed:
        print(f'\n{failed} test(s) failed')
        sys.exit(1)
    print(f'\nAll {len(tests)} tests passed.')
