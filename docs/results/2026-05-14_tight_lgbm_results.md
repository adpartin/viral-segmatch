# Tight bundles + LGBM — results

**Date.** 2026-05-14.
**Scope.** Two new bundles (`flu_ha_na_tight`, `flu_pb2_pb1_tight`) filtered to
Human+Pig × H3N2+H1N1 × year≥2008, trained with **LightGBM only** on nt 6-mer
features at `neg_to_pos_ratio=1.5`, `seq_disjoint` routing with `hash_key=seq`.
Methodology consistent with all 2026-05-13 runs (slot_transform=unit_norm
implicitly inherited by the k-mer feature loader; LGBM is scale-invariant
so the transform doesn't apply at the model level).

## Datasets

| Bundle | Isolates passing filters | Pos pair_keys | Train | Val | Test |
|---|---:|---:|---:|---:|---:|
| `flu_ha_na_tight`   | 61,295 (57% of 107,524) | 31,186 | 62,372 | 7,797 | 7,795 |
| `flu_pb2_pb1_tight` | 61,295 (57% of 107,524) | **26,928** | 53,855 | 6,732 | 6,732 |

Filter cascade identical for both (host → year_range → hn_subtype). Dedup
rate on positives differs: HA/NA 49% (consistent with HA/NA's diversity),
PB2/PB1 **56%** (higher conservation → more cross-isolate duplicates), so
PB2/PB1 ends up with fewer unique pair_keys.

## Test-set metrics

| Bundle | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flu_ha_na_tight` LGBM   | **0.947** | 0.916 | 0.980 | **0.984** | 0.962 | **0.911** | 0.036 |
| `flu_pb2_pb1_tight` LGBM | 0.814 | 0.732 | 0.917 | 0.908 | 0.841 | 0.680 | 0.124 |

**HA/NA tight beats PB2/PB1 tight by 13.3 pp F1, 7.6 pp AUC, 23.1 pp MCC.**
That's far larger than the corresponding gap on the unfiltered HA/NA vs PB2/PB1
(F1 0.931 vs 0.930 LGBM at ratio 1.5 — within noise). The tight filter HURT
PB2/PB1 substantially more than HA/NA.

## Per-regime TPR/TNR

| Bundle | positive | none_match | host_only | subtype_only | year_only | host_sub | host_yr | sub_yr | **host_sub_yr** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flu_ha_na_tight` LGBM   | 0.980 | 1.000 | 1.000 | 0.988 | 1.000 | 0.840 | 0.998 | 0.947 | **0.698** |
| `flu_pb2_pb1_tight` LGBM | 0.917 | 0.988 | **0.662** | 0.983 | 0.945 | **0.613** | 0.699 | 0.916 | **0.523** |

n_samples per regime on test (negatives), for context:

| Bundle | none_match | host_only | subtype_only | year_only | host_sub | host_yr | sub_yr | host_sub_yr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HA/NA tight   | 776 | 775 | 810 | 353 | 755 | 406 | 378 | 424 |
| PB2/PB1 tight | 600 | 745 | 590 | 309 | 727 | 398 | 263 | 407 |

PB2/PB1 tight collapses most visibly on **host-matching but subtype-mismatch
regimes** (`host_only` 0.66, `host_subtype_only` 0.61, `host_year_only` 0.70).
HA/NA tight hits 1.00 TNR on `host_only` — meaning when two HA/NA pairs come
from the same host but different subtype-or-year, the model trivially says
"different isolate." On PB2/PB1 it can't.

## Interpretation

PB2/PB1 is under strong purifying selection (polymerase subunits — any change
in active site or interaction surface is costly). Restricting to Human+Pig ×
H3N2+H1N1 × ≥2008 strips out exactly the demographic diversity that *could*
have produced sequence-level signal. What's left is a pool where many isolates
carry essentially-identical PB2 and PB1 proteins; the model has very little
signal to tell different isolates apart, *especially* when metadata also
matches.

HA/NA stays distinguishable in the same tight pool because HA and NA are
under positive selection (immune escape) — even within Human-H3N2 since 2008,
sequence drift is visible.

**Operational read.** The tight bundle is a stress test for "can the model
extract co-occurrence signal when demographic shortcuts are nearly
unavailable?" HA/NA passes it convincingly (TNR ≥ 0.70 on every regime,
including `host_subtype_year`). PB2/PB1 fails it on multiple regimes —
which is consistent with the prior finding that PB2/PB1 model performance
on the full corpus relied substantially on cluster-leakage / near-neighbor
lookup, not on learning representations.

## Caveats

- **Single seed** (master_seed=42). No CV / seed variance.
- **LightGBM only** — no MLP comparison on the tight bundles, no 1-NN
  baseline. The MLP-vs-1-NN biology-learning criterion isn't checked here.
- **No regime-aware sampling on these tight bundles.** With only 2 hosts
  and 2 subtypes the natural neg distribution already favors hard regimes
  (75% of train negs match on ≥1 axis on HA/NA tight); regime-aware
  sampling at ratio 1.5 would compete with this naturally-occurring
  hard-regime composition. Worth testing as a follow-up.
- **Cluster leakage not addressed.** `seq_disjoint` bounds the exact-hash
  case; mmseqs2 cluster-disjoint splits would further test biology learning
  on the PB2/PB1 tight pool specifically.

## Artifacts

Datasets:
- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_tight_20260513_235556/`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_tight_20260514_090927/`

Models:
- `models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_tight_20260514_091347/`
- `models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_tight_20260514_091349/`

## Related

- `docs/results/2026-05-13_full_8cell_matrix.md` — the 8-cell matrix on the
  unfiltered corpus.
- `conf/bundles/flu_ha_na_tight.yaml`, `conf/bundles/flu_pb2_pb1_tight.yaml`
  — the tight-bundle configs.
- `docs/methods/leakage_definitions.md` — modes #4 (cluster) and #5
  (demographic shortcut) are the two leakage modes this experiment exercises.
