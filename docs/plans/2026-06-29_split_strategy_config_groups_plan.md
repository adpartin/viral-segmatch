# Split-strategy config groups (mode-scoped knobs)

**Status: SPIKE DONE — config plumbing verified byte-safe (2026-06-29).** Steps 1–3
(create groups, point `default.yaml` at the group, byte-verify one bundle) are complete
and byte-verified; the plumbing changes nothing behaviorally fleet-wide. The remaining
steps (per-bundle group conversion 4–6, and the optional `cluster_disjoint`-default flip)
are still PROPOSED / unstarted. See **Spike findings (2026-06-29)** below.

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

## Current state (pre-spike baseline, corrected 2026-06-29)

- `conf/dataset/default.yaml` `split_strategy` was **one flat shared block** carrying
  **eight active keys**: `mode: seq_disjoint`, `hash_key: seq`, the `feasibility:`
  sub-block (`max_acceptable_drift_pp: 0.05`, `min_test_frac: 0.05`), and the three
  CC-builder knobs `negative_scope: within_cc`, `drop_negative_infeasible_ccs: true`,
  `m_pos_per_cc: 1`. `cluster_alphabet`/`cluster_id_path`/`single_slot` are present only
  as **commented doc lines**. The comment `'aa' | 'nt'` is stale (`nt` predates the
  `nt_cds`/`nt_ctg` split).
  - Correction: an earlier draft of this plan said `negative_scope`/`m_pos_per_cc`/
    `drop_negative_infeasible_ccs` were "code fallbacks … not conf." They are **active
    keys in `default.yaml`** (and *also* have matching code fallbacks). Because the block
    is shared, **every bundle inherits these CC + feasibility keys regardless of mode** —
    exactly the leakage this refactor removes.
- `cluster_alphabet` / `pair_key_alphabet` are the only soft knobs whose default is a
  pure code fallback in `dataset_pairs_cc.py` (`getattr(...) or ...`) / the v2 CLI inference
  (no active conf key).
- **Cross-cutting resolver:** `config_hydra.py` (~L57, L81-82) has a `dataset.molecule`
  master knob + axis-consistency guard (§13) that **fills `cluster_alphabet` /
  `pair_key_alphabet` where absent**. Any group refactor must keep this working.
  Spike-verified intact (molecule bundle still resolves correctly post-migration).

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

## Spike findings (2026-06-29)

Steps 1–3 executed on branch `feature/split-strategy-config-groups`.

**Built:** `conf/dataset/split_strategy/{seq_disjoint,cluster_disjoint,cluster_disjoint_cc}.yaml`
(the last `defaults:`-inherits `cluster_disjoint`), and `conf/dataset/default.yaml` now
selects the group (`defaults: [- split_strategy: seq_disjoint, - _self_]`) with the inline
block replaced by an 8-line pointer. No bundle files changed.

**Key byte-safety fact (verified in source):** every soft-knob *reader code-default*
already equals the value the shared block carried — v2 CLI `feasibility` → 0.05/0.05
(`dataset_segment_pairs.py` L609-610); CC builder `negative_scope` → `within_cc` (L488),
`m_pos_per_cc` → 1 (L492), `drop_negative_infeasible_ccs` → True (L485-486); `hash_key`
→ `seq`, `cluster_alphabet` → `aa`. The **only** value `default.yaml` changes from the
code default is `mode` (`'random'` → `seq_disjoint`). So moving the soft knobs into
mode-scoped groups removes them from where they don't belong **without changing behavior** —
the readers backstop the now-absent keys with the same values.

**Verification (decisive):**
- *Fleet compose diff*, all 17 distinct bundle shapes (the 28 `flu_28/` children are
  `schema_pair`-only clones of the master and don't compose standalone in the harness due
  to a pre-existing subdir path quirk — represented by the master): **zero value-changes**
  on any key still present. The only diffs are *removed* inert keys (seq_disjoint /
  cluster_disjoint(v2) bundles shed `feasibility` + the three CC knobs; CC bundles shed
  only `negative_scope`), each backstopped by a verified-matching code-default.
- *Byte-identical dataset*: regenerated `flu_ha_na_cc_aa` (seed 42, within_cc, m_pos=1) →
  all 15 fold CSVs (5 folds × train/val/test) **identical** to the pre-migration
  `within_cc_smoke` baseline (md5 match).

**Hydra gotcha (found + fixed):** an explicit `# @package dataset.split_strategy` header
on a *nested* group at `conf/dataset/split_strategy/` **mis-routes** the content — Hydra
treats `@package X.Y` relative to the already-`dataset` package, so `split_strategy`
silently vanishes for bundles with no inline block (`flu_base`, the master, `bunya_base`
resolved to `<none>`). Fix: omit the header; the nested directory auto-packages to
`dataset.split_strategy` correctly. The group files carry **no** `@package` line.

**Measured blast radius of the two remaining escalations:**
1. *Per-bundle group conversion (steps 4–6) is OPTIONAL for the primary goal* and is where
   it balloons. Converting `flu_ha_na_cc_aa` to *select* the `cluster_disjoint_cc` group
   collides with `cluster_id_path`: the group defaults it `null`, but the live value is
   inherited from `flu_ha_na_cluster_t099` on a different layer — so the whole cluster
   inheritance chain must convert at once. Not required: the inline-override bundles already
   resolve byte-identically (proven above).
2. *Flipping the default to `cluster_disjoint` (Open Decision 1) is the genuinely large
   change.* Only **6 of 45** bundles set `mode` explicitly (`flu_ha_na`, `flu_pb2_pb1` →
   seq_disjoint; 4 cluster/cc bundles → cluster_disjoint(_cc)); **39 have no inline
   `split_strategy`** and ride the default. `cluster_disjoint` requires a `cluster_id_path`
   with no universal default, so flipping breaks all 39. To do it safely, make
   `seq_disjoint` explicit wherever it is currently implicit **first** — most efficiently
   on the shared parents (set it on `flu_28_major_protein_pairs_master` to cover all 28
   `flu_28/` children in one edit; then `flu_base`, `bunya_base`, and a few `flu_ha_na_*`
   variants). This is a separate, deliberate audit — **not** byte-safe by construction.

**Recommendation:** land the plumbing (3 group files + `default.yaml`) as the byte-safe
foundation; treat (1) per-bundle group conversion and (2) the `cluster_disjoint`-default
flip as separate, explicitly-scoped follow-ups.

## Structure (as built in the spike)

The group lives **nested under `dataset/`**, not at top-level `conf/split_strategy/` as the
original draft proposed — nested auto-packages to `dataset.split_strategy` with no
`@package` header (see the Hydra gotcha in Spike findings; a top-level group would need the
header, which mis-routes).

```
conf/dataset/split_strategy/
  seq_disjoint.yaml          # mode: seq_disjoint; hash_key: seq
  cluster_disjoint.yaml      # mode; hash_key; cluster_alphabet: aa; cluster_id_path: null;
                             #   cluster_id_threshold: null; cds_final_path: null;
                             #   single_slot: null; feasibility: {...}
  cluster_disjoint_cc.yaml   # defaults: [- cluster_disjoint, - _self_] (inherit shared) +
                             #   mode: cluster_disjoint_cc; negative_scope: within_cc;
                             #   m_pos_per_cc: 1; drop_negative_infeasible_ccs: true
  # random.yaml              # not created — no active bundle selects mode=random
```
- `conf/dataset/default.yaml`: `defaults: [- split_strategy: seq_disjoint, - _self_]`
  (default kept at `seq_disjoint` — see Decision 1). **Done in spike.**
- Bundles: **unchanged in spike.** They keep their inline `split_strategy: {mode: ..., ...}`
  overrides, which merge byte-identically over the selected group. Converting them to
  `defaults: [- /dataset/split_strategy: <group>]` form (steps 4–6) is optional polish, not
  required for byte-safety.

## Open decisions

1. **Default group.** Kept `seq_disjoint` in the spike (no behavior change). Flipping to
   `cluster_disjoint(_cc)` is rigor-forward, but `cluster_id_path` has no universal default
   → becomes effectively **required** (builders already error clearly). **Measured cost:**
   39 of 45 bundles ride the implicit default; all would break on the flip. Make
   `seq_disjoint` explicit first (master covers its 28 `flu_28/` children in one edit; then
   `flu_base`/`bunya_base`/a few `flu_ha_na_*`). **Open — user wants `cluster_disjoint`
   default eventually; deferred to a separate audited step.**
2. **`cluster_disjoint_cc` inherits `cluster_disjoint`** (DRY on the shared knobs).
   **Spike result:** it DOES inherit `single_slot`/`feasibility`, but those are
   v2-routing-only and the CC builder never reads them, so they are inert — byte-identical
   CC dataset confirms no effect. Acceptable as-is.
3. **Resolver interaction.** The `dataset.molecule` filler in `config_hydra.py` must still
   find/fill `cluster_alphabet`/`pair_key_alphabet` once they live in the selected group.
   **Spike-verified:** `flu_ha_na_molecule_nt_cds` resolves correctly post-migration.
4. **Code cleanup (closes the TODO).** Once conf provides the keys, simplify
   `getattr(ss,'X','X') or 'X'` → `ss.X` in `dataset_pairs_cc` (and the v2 readers).
   Validation (`if alphabet not in _ENABLED_ALPHABETS: raise`) still guards bad/null.
   **NOTE:** only safe once *every* bundle delivers the key via a group (steps 4–6); today
   the soft knobs are intentionally absent from seq_disjoint/cluster_disjoint bundles and
   the `getattr` fallback is load-bearing. Do this LAST.
5. **OmegaConf struct mode** (optional): makes a typo'd bundle key error instead of silently
   defaulting — but requires every key declared in the group.

## Migration steps (incremental, each byte-verified)

1. **[DONE]** Create `conf/dataset/split_strategy/*.yaml` mirroring **current** effective
   defaults (no behavior change). v2 CLI reader set enumerated (see migration map).
2. **[DONE]** Point `default.yaml` `defaults` at `seq_disjoint`; fleet resolved-config diff
   = zero value-changes (only inert keys removed).
3. **[DONE]** Regenerated `flu_ha_na_cc_aa` (inline form, not converted) → **byte-identical**
   (15/15 fold CSVs) vs the `within_cc_smoke` pre-migration baseline.
4. **[not started, OPTIONAL]** Convert `*_cc_*` + v2 `cluster_disjoint` bundles to
   group-*select* form; byte-verify each. Balloon risk: the `cluster_id_path` layer collision
   (see Spike findings #1) means converting the whole cluster inheritance chain together.
5. **[not started — do LAST]** Simplify the code `getattr`-fallbacks → `ss.X`; re-verify
   byte-identical. Only safe after step 4 makes every bundle deliver the keys via a group.
6. **[not started — separate audit]** Make `seq_disjoint` explicit on the 39 default-riding
   bundles, *then* (if desired) flip the default to `cluster_disjoint`.
7. (Optional) enable struct mode for typo-safety.

## Risks

- The `dataset.molecule` resolver in `config_hydra.py` (cross-cutting). **Spike-verified intact.**
- Bundles relying on the default `mode` (behavior change if the default flips) — **39 of 45;
  quantified in Decision 1.**
- Hydra `defaults`-list / `_self_` ordering when a bundle both selects a group and overrides
  — the **`@package` mis-routing** gotcha (Spike findings) is the concrete instance found;
  nested group + no header avoids it.
- v2's `SPLIT_STRATEGY_CFG` access path must still resolve every key — **spike-verified**
  (all readers use `getattr(..., default)`, so absent keys fall to matching code-defaults).

## Verification

Per-bundle **byte-identical dataset** (the established check) + AST/import/ruff, run after
each step on: a seq_disjoint bundle, a v2 cluster_disjoint bundle, and a CC bundle.

## Out of scope

Split semantics, negative sampling, the molecule master-knob design (only its interaction).
