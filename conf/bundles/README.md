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
                                flu_schema_raw_slot_norm_unit_diff.yaml ──► _h3n2, _human, _illinois, _2024
                                flu_schema_raw_slot_norm_concat.yaml    ──► _h3n2, _human, _illinois, _2024
                                flu_schema_raw_none_unit_diff.yaml      ──► _h3n2
                                flu_schema_raw_slot.yaml
                                flu_schema_raw_shared.yaml
                                flu_schema_raw_adapter.yaml
```

---

## Status Table

### Active — current best model and paper experiments

| Bundle | Proteins | Filter | pre_mlp_mode | interaction | Notes |
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
| `flu_pb2_pb1_pa_5ks` | PB2, PB1, PA | none | — | — | Conserved segments; paper result |
| `flu_pb2_ha_na_5ks` | PB2, HA, NA | none | — | — | Mixed segments; paper result |

### Ablations — controlled comparisons

| Bundle | Purpose |
|--------|---------|
| `flu_schema_raw_none_unit_diff` | No LayerNorm ablation (compare to slot_norm) |
| `flu_schema_raw_none_unit_diff_h3n2` | No LayerNorm on H3N2; confirms LayerNorm is critical |
| `flu_pb2_pb1_pa_5ks_label_shuffle` | Label shuffle sanity check (should converge to ~50% F1) |
| `flu_concat_no_canon` | No canonical pair orientation (compare to default) |

### Experimental / abandoned — unfinished pre-MLP mode exploration

| Bundle | Notes |
|--------|-------|
| `flu_schema_raw_shared` | Shared pre-MLP (pre_mlp_mode=shared); not completed |
| `flu_schema_raw_adapter` | Adapter pre-MLP (pre_mlp_mode=shared_adapter); not completed |
| `flu_schema_raw_slot` | Slot-specific pre-MLP (pre_mlp_mode=slot_specific); not completed |

### Legacy Gen 2 — superseded by slot_norm + unit_diff

| Bundle | Notes |
|--------|-------|
| `flu_schema` | Schema-ordered base; pre_mlp_mode=none, concat |
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

### Paper experiments (upcoming)

See `paper/` subdirectory. Reserved for bundles that will be reported in the publication.
Cross-validation, temporal holdout, and large-dataset runs will go here.

---

## Key Findings (inform new bundle design)

- **`unit_diff` + `slot_norm`** is the current best combination. `concat` fails on homogeneous
  data (H3N2-only: AUC≈0.5); `unit_diff` succeeds (AUC≈0.96).
- **`slot_norm` (LayerNorm per slot) is critical** for homogeneous subsets. Without it,
  slot offset dominates the unit_diff direction and the model cannot learn.
- **H3N2-only runs** require `patience≥40` due to a plateau-then-breakthrough learning
  pattern (see `_ongoing_work.md` for analysis).
- **High FP rate** on filtered datasets (year, host, geography) is likely a population-level
  confounder issue. Hard negative sampling or strict multi-field filtering may help.
