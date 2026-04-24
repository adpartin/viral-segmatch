# Plan: Subtype Balancing in Stage 3 (PR 1 of 2)

**Status: IMPLEMENTED**

Landed in `src/datasets/dataset_segment_pairs.py` (helper
`select_balanced_isolate_pool` + CLI wiring with three mutual-exclusion
error checks), `conf/dataset/default.yaml` (new `subtype_selection` block,
defaults to `mode: natural`), and `conf/bundles/README.md` ("Running
balanced subtype sweeps" section with `--override` recipe and caveats).

Smoke tests on the 100-isolate dataset (subtype counts H3N2=32, H1N1=30,
H5N1=13):

1. **Natural regression** -- mode=natural runs unchanged, no manifest
   written.
2. **Balanced happy path** -- N=min=13; manifest written with per-subtype
   counts and full `selected_assembly_ids`; 39 isolates after balancing;
   full Stage 3 completes to pair CSVs.
3. **Error: balanced + max_isolates_to_process=10000** -- clear error
   naming both config keys and the resolution.
4. **Error: balanced + dataset.hn_subtype=H3N2** -- clear error.
5. **Error: target_count_per_subtype=999999** -- clear error listing
   per-subtype available counts and pointing at 'min'.
6. **Determinism** -- two runs with seed=42 produce bit-identical
   manifests (`diff` empty).
7. **Seed override** -- `subtype_selection.seed=99` produces a different
   `selected_assembly_ids` tuple as expected.

## Motivation

Roadmap v2 Task 2 ("subtype-balanced training and test sets", Jim's #1
priority from 2026-03-12 meeting): construct subtype-balanced train/val/test
splits so per-subtype test performance is a clean measurement of model
capability rather than confounded by training-frequency effects.

Definitions (from `roadmap_v2.md`):
- **subtype-balanced** — explicitly resampled so HN-subtype representation is
  equalized across subtypes (downsampling dominant subtypes).
- **subtype-filtered** — restricted to one subtype (e.g., H3N2-only).
- **natural / unfiltered** — sampled proportional to BV-BRC frequencies.

This PR implements **Approach 2** from the roadmap ("integrate balancing into
dataset generation"), chosen over Approach 1 ("sub-sample from existing
folds") because it's the right place for the pair-distribution ledger (PR 2)
and keeps balancing logic co-located with pair generation.

## Scope boundaries

**In scope (PR 1):**
- Config-driven isolate-level balancing step, runs once before CV fold
  generation.
- Downsampling only (by roadmap decision; explicitly rejects upsampling).
- Deterministic: bit-identical given `(master_seed, subtype_list, N, df)`.
- Backward compatible: `mode: natural` is the default, current bundles
  unchanged.
- Explicit error paths for incompatible config combinations.

**Out of scope (deferred to PR 2):**
- Pair-distribution ledger (per-fold × per-split × per-subtype-pair accounting).
- Extraction of shared `classify_pair_type` helper.

**Out of scope (upstream decision):**
- Upsampling.
- Per-fold strict subtype balance (KFold's approximate balance is acceptable).
- New bundle files. First balanced sweep uses the existing `--override`
  mechanism on an existing bundle.
- Option A (parallel master files). Single-master + config toggle is
  preferred for simplicity; migration path to Option A remains open.

## Design decisions

**Architecture choice:** single master bundle
(`flu_28_major_protein_pairs_master.yaml`) + config flag toggled via
`--override`. The existing `--override` plumbing in `run_cv_lambda.py` and
`dataset_segment_pairs.py` already handles this pattern cleanly. Concurrent
sweeps (natural + balanced_big3 in parallel Polaris jobs) are supported by
invoking the same bundle with different overrides. `resolved_config.yaml`
snapshot per run documents what actually ran.

**Call site:** between `filter_by_metadata` and the
`MAX_ISOLATES_TO_PROCESS` block in the CLI section of
`src/datasets/dataset_segment_pairs.py`. This placement guarantees:
- `hn_subtype` column is populated (metadata enrichment has run).
- Balancing operates on metadata-filtered `prot_df`.
- Selected-functions narrowing (which uses `config.virus.selected_functions`)
  happens after balancing, so the balanced isolate pool is identical across
  the 28 pair bundles even though each pair's `schema_pair` differs.

**Seed path:** `random.Random(subtype_selection.seed or RANDOM_SEED)`. Local
RNG only — no reads from the process-global random stream. Consistent with
the hygiene established by the negative-pair RNG fix.

**Determinism via sorted iteration:** sort both the included subtypes and
each subtype's `assembly_id` list before sampling. Removes any dependence on
df row order.

## Config schema

Add under `dataset:` in `conf/dataset/default.yaml`:

```yaml
subtype_selection:
  mode: natural              # natural | balanced
  included_subtypes: null    # list[str], required when mode=balanced
  target_count_per_subtype: min   # "min" or int; int > min(available) -> error
  seed: null                 # null -> inherit RANDOM_SEED
```

## Helper signature

```python
def select_balanced_isolate_pool(
    prot_df: pd.DataFrame,
    subtype_selection_cfg,
    master_seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Downsample isolates to equal per-subtype representation.

    No-op when mode=natural. Emits subtype_selection_manifest.json to
    output_dir when mode=balanced.
    """
```

## Interaction rules (error matrix)

| Config combination | Behavior |
|---|---|
| `mode=natural` | no-op (backward compatible) |
| `mode=balanced` + `max_isolates_to_process=null` + `hn_subtype=null` + `included_subtypes` valid | runs |
| `mode=balanced` + `max_isolates_to_process != null` | ERROR |
| `mode=balanced` + `dataset.hn_subtype != null` | ERROR |
| `mode=balanced` + empty/null `included_subtypes` | ERROR |
| `mode=balanced` + `target_count_per_subtype=int > min(available)` | ERROR (no upsampling) |
| `mode=balanced` + `included_subtypes` references a value not in df | ERROR (lists available) |
| `mode` not in {natural, balanced} | ERROR |

## Manifest (written only when `mode=balanced`)

`subtype_selection_manifest.json` at the dataset-run-dir root:

```json
{
  "mode": "balanced",
  "included_subtypes": ["H1N1", "H3N2", "H5N1"],
  "target_count_per_subtype": "min",
  "N": <int>,
  "seed": <int>,
  "per_subtype": {
    "H1N1": {"available": <int>, "selected": <int>},
    ...
  },
  "isolates_before": <int>,
  "isolates_after": <int>,
  "selected_assembly_ids": [...]
}
```

## Deliverables

1. `src/datasets/dataset_segment_pairs.py` — helper (~70 lines) + call site
   + error paths.
2. `conf/dataset/default.yaml` — `subtype_selection` default block.
3. `conf/bundles/README.md` — "Running balanced subtype sweeps" section
   showing the `--override` pattern.
4. This plan doc, marked `IMPLEMENTED` at merge.

## Smoke tests

Run against `flu_pb2_pb1` (10K isolates):

1. **Natural regression.** Existing behavior unchanged.
   ```
   python scripts/run_cv_lambda.py --config_bundle flu_pb2_pb1 --skip_training --dry_run
   ```
2. **Balanced happy path.**
   ```
   python scripts/run_cv_lambda.py --config_bundle flu_pb2_pb1 --skip_training \
     --override dataset.max_isolates_to_process=null \
                dataset.subtype_selection.mode=balanced \
                'dataset.subtype_selection.included_subtypes=[H1N1,H3N2,H5N1]'
   ```
3. **Error path:** balanced + `max_isolates_to_process=10000`.
4. **Error path:** balanced + `dataset.hn_subtype=H3N2`.
5. **Error path:** `target_count_per_subtype=999999` (upsampling attempt).

Also: rerun (2) with same seed → bit-identical `selected_assembly_ids`;
different seed → different pool.

## Acceptance criteria

- `mode=natural` regression against `flu_pb2_pb1`: bit-identical outputs vs
  current master.
- `mode=balanced` deterministic across reruns with the same seed.
- All five smoke-test error paths raise with actionable messages.
- Smoke run (case 2) completes Stage 3 end-to-end with manifest written.
- `conf/bundles/README.md` has the `--override` recipe.

## Follow-up (not this PR)

- **PR 2** — pair-distribution ledger + extract shared `classify_pair_type`
  from `analyze_stage4_train.py:752-758`.
- **Audit re-run** — re-evaluate `docs/audits/class_balance_audit.md` against
  the first balanced sweep. Expected: pair_key cross-split drop rate
  worsens as the 3-subtype pool is denser in shared sequences than the
  natural pool.
