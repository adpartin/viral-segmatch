# conf/bundles/

One YAML file per named experiment. Bundle name encodes the experiment:
`flu_{proteins}_{n_isolates}[_{modifiers}]`

Bundles are loaded via `config_hydra.py` → `hydra.compose(config_name="bundles/{name}")`.

---

## Bundle Inheritance (Hydra defaults chains)

Several bundles are **base files** — other bundles inherit from them via the Hydra `defaults`
list. Moving a base to a subdirectory would break its children.

```
flu.yaml  ──────────────────► flu_concat.yaml
                               flu_diff.yaml
                               flu_concat_no_canon.yaml
                               flu_h3n2.yaml ──► flu_h3n2_concat.yaml
                                                  flu_h3n2_diff.yaml

flu_schema.yaml ─────────────► flu_schema_diff.yaml
  (base for ALL Gen 2+3)        flu_schema_human.yaml
                                flu_schema_h3n2.yaml ──► flu_schema_h3n2_diff.yaml
                                flu_schema_raw_slot_norm_unit_diff.yaml ──► _h3n2, _human, _illinois, _2024, _temporal
                                flu_schema_raw_slot_norm_concat.yaml    ──► _h3n2, _human, _illinois, _2024
                                flu_schema_raw_none_unit_diff.yaml      ──► _h3n2
                                flu_schema_raw_kmer_k6_slot_norm_unit_diff.yaml ──► _h3n2
                                flu_schema_raw_kmer_k6_slot_norm_concat.yaml    ──► _h3n2
                                flu_schema_raw_slot.yaml
                                flu_schema_raw_shared.yaml
                                flu_schema_raw_adapter.yaml
```

---

## Status Table

### Active — current best model and paper experiments

| Bundle | Proteins | Filter | slot_transform | interaction | Notes |
|--------|----------|--------|-------------|-------------|-------|
| `flu_schema_raw_slot_norm_unit_diff` | HA, NA | none | slot_norm | unit_diff | **Best overall model** |
| `flu_schema_raw_slot_norm_unit_diff_h3n2` | HA, NA | H3N2 | slot_norm | unit_diff | patience=40 (delayed learning) |
| `flu_schema_raw_slot_norm_unit_diff_human` | HA, NA | Human | slot_norm | unit_diff | |
| `flu_schema_raw_slot_norm_unit_diff_illinois` | HA, NA | Illinois | slot_norm | unit_diff | |
| `flu_schema_raw_slot_norm_unit_diff_2024` | HA, NA | year=2024 | slot_norm | unit_diff | |
| `flu_schema_raw_slot_norm_concat` | HA, NA | none | slot_norm | concat | Comparison to unit_diff |
| `flu_schema_raw_slot_norm_concat_h3n2` | HA, NA | H3N2 | slot_norm | concat | patience=100; known to fail (AUC≈0.5) |
| `flu_schema_raw_slot_norm_concat_human` | HA, NA | Human | slot_norm | concat | |
| `flu_schema_raw_slot_norm_concat_illinois` | HA, NA | Illinois | slot_norm | concat | |
| `flu_schema_raw_slot_norm_concat_2024` | HA, NA | year=2024 | slot_norm | concat | |
| `flu_schema_raw_kmer_k6_slot_norm_unit_diff` | HA, NA | none | slot_norm | unit_diff | K-mer (k=6) baseline; feature_source=kmer |
| `flu_schema_raw_kmer_k6_slot_norm_concat` | HA, NA | none | slot_norm | concat | K-mer + concat comparison |
| `flu_schema_raw_kmer_k6_slot_norm_unit_diff_h3n2` | HA, NA | H3N2 | slot_norm | unit_diff | K-mer H3N2; AUC=0.988 |
| `flu_schema_raw_kmer_k6_slot_norm_concat_h3n2` | HA, NA | H3N2 | slot_norm | concat | K-mer H3N2; AUC=0.985 (no concat collapse) |
| `flu_pb2_pb1_pa_5ks` | PB2, PB1, PA | none | — | — | Conserved segments; paper result |
| `flu_pb2_ha_na_5ks` | PB2, HA, NA | none | — | — | Mixed segments; paper result |

### Ablations — controlled comparisons

| Bundle | Purpose |
|--------|---------|
| `flu_schema_raw_none_unit_diff` | No LayerNorm ablation (compare to slot_norm) |
| `flu_schema_raw_none_unit_diff_h3n2` | No LayerNorm on H3N2; confirms LayerNorm is critical |
| `flu_pb2_pb1_pa_5ks_label_shuffle` | Label shuffle sanity check (should converge to ~50% F1) |
| `flu_concat_no_canon` | No canonical pair orientation (compare to default) |

### Experimental — paper experiments in progress

| Bundle | Notes |
|--------|-------|
| `flu_schema_raw_slot_norm_unit_diff_temporal` | Temporal holdout: train 2021-2023, test 2024 |

### Experimental / abandoned — unfinished slot transform exploration

| Bundle | Notes |
|--------|-------|
| `flu_schema_raw_shared` | Shared slot transform (slot_transform=shared); not completed |
| `flu_schema_raw_adapter` | Adapter slot transform (slot_transform=shared_adapter); not completed |
| `flu_schema_raw_slot` | Slot-specific slot transform (slot_transform=slot_specific); not completed |

### Legacy Gen 2 — superseded by slot_norm + unit_diff

| Bundle | Notes |
|--------|-------|
| `flu_schema` | Schema-ordered base; slot_transform=none, concat |
| `flu_schema_diff` | Gen 2 interaction test |
| `flu_schema_h3n2` | Gen 2 H3N2 filter |
| `flu_schema_h3n2_diff` | Gen 2 H3N2 + diff |
| `flu_schema_human` | Gen 2 human filter; uses concat (known to fail on homogeneous data) |

### Legacy Gen 1 — pre-schema, earliest experiments

| Bundle | Notes |
|--------|-------|
| `flu` | Gen 1 base; HA/NA, no schema ordering |
| `flu_concat` | Gen 1 concat interaction |
| `flu_diff` | Gen 1 diff interaction |
| `flu_concat_no_canon` | Gen 1 no canonical ordering |
| `flu_h3n2` | Gen 1 H3N2 filter base |
| `flu_h3n2_concat` | Gen 1 H3N2 + concat |
| `flu_h3n2_diff` | Gen 1 H3N2 + diff |

### Not maintained

| Bundle | Notes |
|--------|-------|
| `bunya` | Bunyavirales; pipeline exists but lags current Flu A conventions |

### Paper experiments

Bundles for publication experiments must stay **flat** in `conf/bundles/` (not in a subdirectory).

**Reason**: Hydra's package resolution double-nests inherited configs from subdirectories.
For a bundle in `conf/bundles/paper/`, the composed config lands at `full_config.bundles.paper.*`
instead of `full_config.bundles.*`, breaking `get_virus_config_hydra`. No Hydra package directive
(`@package bundles`, `@_global_`, etc.) fully resolves this without changing the loader.

**Convention**: use a `paper_` prefix or an unambiguous descriptive name for paper bundles:
- `flu_schema_raw_slot_norm_unit_diff_cv5.yaml` — 5-fold CV (active)
- Future: `flu_schema_raw_slot_norm_unit_diff_temporal.yaml`, etc.

`conf/bundles/paper/` directory is kept for non-Hydra documentation (e.g., notes, experiment
descriptions) but YAML bundle files must live in the flat `conf/bundles/` directory.

---

## Running balanced subtype sweeps (no new bundle required)

Roadmap v2 Task 2 ("subtype-balanced training and test sets") uses the
existing single-master setup plus Hydra CLI overrides -- no parallel master
files. The `subtype_selection` config block in `conf/dataset/default.yaml`
defaults to `mode: natural` (no-op, current behavior). To run a balanced
sweep, pass overrides via `scripts/run_cv_lambda.py --override ...` (or
directly to `dataset_segment_pairs.py`). Every child bundle inheriting from
`flu_28_major_protein_pairs_master` picks up the same override.

**big3 balanced sweep (H1N1 / H3N2 / H5N1, downsampled to the smallest):**

```bash
python scripts/run_cv_lambda.py --config_bundle flu_28p_ha_na \
  --override dataset.subtype_selection.mode=balanced \
             'dataset.subtype_selection.included_subtypes=[H1N1,H3N2,H5N1]'
```

**Notes:**

- `subtype_selection.target_count_per_subtype` defaults to `"min"` (use the
  smallest included-subtype count as N). Override to an int to cap lower;
  exceeding the smallest count is rejected (no upsampling).
- `subtype_selection.seed` defaults to the dataset process seed (derived
  from `master_seed`). Override only when you want a distinct balanced pool
  for ablation.
- Incompatible config combinations error loudly in Stage 3:
  - `subtype_selection.mode=balanced` + `dataset.max_isolates_to_process`
  - `subtype_selection.mode=balanced` + `dataset.hn_subtype`
- Stage 3 writes `subtype_selection_manifest.json` at the dataset-run-dir
  root documenting the per-subtype counts, N, seed, and full list of
  selected `assembly_id`s. Audit trail for the paper methods section.
- For concurrent natural + balanced sweeps on the same bundle, launch two
  separate jobs with different overrides; timestamps + `resolved_config.yaml`
  in each run dir disambiguate.
- See `docs/plans/subtype_balancing_plan.md` for the full design.

---

## Key Findings (inform new bundle design)

- **ESM-2: `unit_diff` + `slot_norm`** is the current best ESM-2 combination. ESM-2 `concat`
  fails on homogeneous data (H3N2-only: AUC≈0.50); `unit_diff` succeeds (AUC≈0.96).
- **K-mer `concat` does NOT fail on H3N2** (AUC=0.985). The concat collapse is specific to
  ESM-2's embedding geometry (protein-type subspace offset), not concatenation itself.
- **K-mer dominates ESM-2 on homogeneous data**: H3N2-only AUC 0.988 vs 0.957 (both unit_diff).
  K-mer features are interaction-agnostic (unit_diff ≈ concat).
- **`slot_norm` (LayerNorm per slot) is critical for ESM-2** on homogeneous subsets. Without it,
  slot offset dominates the unit_diff direction and the model cannot learn.
- **H3N2-only runs** require `patience≥40` due to a plateau-then-breakthrough learning
  pattern (see `_ongoing_work.md` for analysis).
- **High FP rate** on filtered datasets (year, host, geography) is likely a population-level
  confounder issue. Hard negative sampling or strict multi-field filtering may help.
