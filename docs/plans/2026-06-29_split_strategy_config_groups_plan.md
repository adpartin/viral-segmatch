# Split-strategy config groups (mode-scoped knobs)

**Status: PROPOSED** — draft for review; not approved or started.

## Goal

Move per-mode `dataset.split_strategy` knobs out of one flat shared block (and out of
code-level `getattr(ss, 'X', 'X') or 'X'` fallbacks) into **mode-scoped Hydra config
groups**, so that:
- a mode's knobs **only exist when that mode is selected** (no CC knobs leaking into
  seq_disjoint);
- each knob's **default lives in conf/**, not buried in Python (closes the
  `cluster_alphabet` default TODO and the split-brain where `default.yaml` *comments*
  `# cluster_alphabet: aa (default)` while the code actually enforces it);
- (optional) the **default mode** can become `cluster_disjoint(_cc)` cleanly.

Scope: config plumbing only. **No change to split semantics** — every bundle must produce
byte-identical datasets pre/post.

## Current state (verified 2026-06-29)

- `conf/dataset/default.yaml` `split_strategy` is **flat**: `mode: seq_disjoint`,
  `hash_key: seq`, plus `cluster_alphabet`/`cluster_id_path`/`single_slot` as **commented
  doc lines** (not active keys). The comment `'aa' | 'nt'` is itself stale (`nt` predates
  the `nt_cds`/`nt_ctg` split).
- Defaults for `cluster_alphabet` / `pair_key_alphabet` / `negative_scope` / `m_pos_per_cc`
  / `drop_negative_infeasible_ccs` are **code fallbacks** in `dataset_pairs_cc.py`
  (`getattr(...) or ...`), not conf.
- **Cross-cutting resolver:** `config_hydra.py` (~L57, L81-82) has a `dataset.molecule`
  master knob + axis-consistency guard (§13) that **fills `cluster_alphabet` /
  `pair_key_alphabet` where absent**. Any group refactor must keep this working.

## Knob → readers → scope (the migration map, verified)

| knob | readers (file) | scope |
|---|---|---|
| `mode` | `dataset_pairs_cc` (requires `cluster_disjoint_cc`); v2 CLI `_validate_v2_config` | all |
| `hash_key` | `_pair_helpers.bipartite_components` (legacy seq/dna); v2 CLI | seq / legacy |
| `cluster_alphabet` | `_split_helpers` (v2 routing); `dataset_pairs_cc`; `config_hydra` resolver | **shared** (cluster + cc) |
| `pair_key_alphabet` | `_pair_helpers.create_positive_pairs_v2` (shared); `config_hydra` resolver; `dataset_pairs_cc` | **shared** (cluster + cc) |
| `cluster_id_path` | v2 CLI; `dataset_pairs_cc` | **shared** (cluster + cc) |
| `cds_final_path` | v2 CLI; `dataset_pairs_cc` (nt_cds) | **shared** (cluster + cc) |
| `single_slot` | `_split_helpers` (v2 routing only) | cluster (v2 only) |
| `feasibility.{max_acceptable_drift_pp,min_test_frac}` | v2 cluster_disjoint | cluster (v2 only) |
| `negative_scope` | `dataset_pairs_cc` only | **CC-only** |
| `m_pos_per_cc` | `dataset_pairs_cc` only | **CC-only** |
| `drop_negative_infeasible_ccs` | `dataset_pairs_cc` only | **CC-only** |

> v2 CLI (`dataset_segment_pairs.py`) reads these via `SPLIT_STRATEGY_CFG` and forwards to
> `_split_helpers`/`_pair_helpers`; confirm the exact set in step 1.

## Proposed structure

```
conf/split_strategy/
  seq_disjoint.yaml          # mode: seq_disjoint; hash_key: seq
  cluster_disjoint.yaml      # mode; cluster_alphabet: aa; pair_key_alphabet: null;
                             #   cluster_id_path: null; cds_final_path: null;
                             #   single_slot: null; feasibility: {...}
  cluster_disjoint_cc.yaml   # defaults: [cluster_disjoint] (inherit shared) +
                             #   m_pos_per_cc: 1; negative_scope: within_cc;
                             #   drop_negative_infeasible_ccs: true
  random.yaml                # if still used
```
- `conf/dataset/default.yaml`: `defaults: [- split_strategy: <chosen>]` (see Decision 1).
- Bundles: replace the inline `split_strategy: {mode: ..., ...}` with
  `defaults: [- split_strategy: cluster_disjoint_cc]` + only experiment-specific overrides
  (`cluster_id_path`, `cluster_alphabet`).

## Open decisions

1. **Default group.** `seq_disjoint` = no behavior change. `cluster_disjoint(_cc)` =
   rigor-forward, but `cluster_id_path` has no universal default → becomes effectively
   **required** (builders already error clearly), and flipping it **changes behavior for any
   bundle relying on the current default** → audit those first.
2. **`cluster_disjoint_cc` inherits `cluster_disjoint`** (DRY on the shared knobs) — verify
   it does NOT thereby pull in `single_slot`/`feasibility`, which are v2-routing-only.
3. **Resolver interaction.** The `dataset.molecule` filler in `config_hydra.py` must still
   find/fill `cluster_alphabet`/`pair_key_alphabet` once they live in the selected group.
4. **Code cleanup (closes the TODO).** Once conf provides the keys, simplify
   `getattr(ss,'X','X') or 'X'` → `ss.X` in `dataset_pairs_cc` (and the v2 readers).
   Validation (`if alphabet not in _ENABLED_ALPHABETS: raise`) still guards bad/null.
5. **OmegaConf struct mode** (optional): makes a typo'd bundle key error instead of silently
   defaulting — but requires every key declared in the group.

## Migration steps (incremental, each byte-verified)

1. Create `conf/split_strategy/*.yaml` mirroring **current** effective defaults (no behavior
   change). Enumerate the v2 CLI reader set precisely.
2. Point `default.yaml` `defaults` at the chosen group; confirm one existing bundle is
   unchanged (resolved-config diff).
3. Convert **one** bundle (`flu_ha_na_cc_aa`) to the group form → regenerate dataset →
   **byte-identical** vs a pre-migration baseline.
4. Convert the remaining `*_cc_*` + v2 `cluster_disjoint` bundles; byte-verify each.
5. Simplify the code `getattr`-fallbacks → `ss.X`; re-verify byte-identical (TODO closes).
6. Audit + convert bundles that rely on the default `mode`.
7. (Optional) enable struct mode for typo-safety.

## Risks

- The `dataset.molecule` resolver in `config_hydra.py` (cross-cutting; test explicitly).
- Bundles relying on the default `mode` (behavior change if the default flips).
- Hydra `defaults`-list / `_self_` ordering when a bundle both selects a group and overrides.
- v2's `SPLIT_STRATEGY_CFG` access path must still resolve every key.

## Verification

Per-bundle **byte-identical dataset** (the established check) + AST/import/ruff, run after
each step on: a seq_disjoint bundle, a v2 cluster_disjoint bundle, and a CC bundle.

## Out of scope

Split semantics, negative sampling, the molecule master-knob design (only its interaction).
