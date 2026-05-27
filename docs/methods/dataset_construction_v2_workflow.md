# Dataset construction (Stage 3, v2) — workflow walkthrough

End-to-end walkthrough of `dataset_segment_pairs_v2.py` on one
concrete production run. Every count here is verified from the run's
`dataset_stats.json`. For the design spec see
[`../plans/done/2026-05-11_design_dataset_gen_v2.md`](../plans/done/2026-05-11_design_dataset_gen_v2.md);
for the broader pipeline framing see
[`pipeline_overview.md`](pipeline_overview.md) §4–§6.

## The example run

- **Bundle:** `flu_ha_na_regimes` (HA / NA, regime-aware negative
  sampling, `neg_to_pos_ratio: 2.0`, `seq_disjoint` routing with
  `hash_key: seq`).
- **Run directory:** `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_20260512_114205/`.
- **Builder version:** `v2`.
- **HEAD commit at run time:** `45a2ed3` ("docs(results): drop stale
  aggregator outputs; add Tests 1-4 baselines-vs-MLP heatmaps", 2026-05-12
  ~11:42). Relevant earlier commits in scope: `a1b8301`
  (`flu_ha_na` → Test 3 + `seq_disjoint`), `6233902` (centralized
  baseline defaults), `f8af35f` (Stage-3 split-composition viz).
- **Frozen copy of the run's full stats:**
  [`../examples/dataset_flu_ha_na_regimes_20260512_114205_dataset_stats.json`](../examples/dataset_flu_ha_na_regimes_20260512_114205_dataset_stats.json)
  (100 KB JSON; copied here so the example remains reproducible even if
  the original `data/datasets/` directory is cleared).
- **Total runtime:** 3 min 55 s (per `runtime.json`).

## Essential config (relevant knobs)

The bundle inherits from `flu_base` and `flu_ha_na`, then layers on the
regime-aware overrides. The knobs that gate each phase below:

| Knob | Value in this run | Phase gated |
|---|---|---|
| `dataset.pair_builder_version` | `v2` | dispatch |
| `dataset.schema_pair` | `["Hemagglutinin precursor", "Neuraminidase protein"]` | Phase 1, 2 |
| `dataset.pair_mode` | `schema_ordered` | Phase 2 |
| `virus.selected_functions` | 8 major proteins (HA, NA, …) | Phase 1 |
| `dataset.host / year / hn_subtype` | all `null` (no metadata filter) | Phase 1 |
| `dataset.max_isolates_to_process` | `null` (full Flu-A) | Phase 1 |
| `dataset.hard_partition_isolates` | `true` | Phase 3 |
| `dataset.split_strategy.mode` | `seq_disjoint` | Phase 3 |
| `dataset.split_strategy.hash_key` | `seq` (protein-level, default) | Phase 3 |
| `dataset.train_ratio` / `val_ratio` | `0.8` / `0.1` (so test = 0.1) | Phase 3 |
| `dataset.neg_to_pos_ratio` | `2.0` | Phase 4 + 5 |
| `dataset.negative_sampling.regime_targets` | 8-regime mix (see Phase 5) | Phase 5 |
| `master_seed` | `42` | negative sampling (routing is deterministic given atoms + ratios) |

See [`../conf_guide.md`](../conf_guide.md) for the Hydra inheritance
chain and per-knob semantics.

---

## Phase 0 — Load inputs and enrich with metadata

Before any pairing logic runs, the builder loads Stage 1 outputs and
joins per-isolate metadata onto them. This phase is upstream of the v2
builder itself (lives in `dataset_segment_pairs.py` CLI before the v2
dispatch) but determines what the v2 builder sees as input.

### What happens

1. **Load `protein_final`.** Read
   `data/processed/flu/July_2025/protein_final.parquet` (or `.csv`) —
   the Stage 1 output of `preprocess_flu.py`. One row per CDS,
   ~1,793,563 rows in the full corpus.
2. **Load `genome_final`.** Read
   `data/processed/flu/July_2025/genome_final.parquet` — one row per
   contig, 868,240 rows.
3. **Attach DNA + `dna_hash` to protein rows.**
   `_pair_helpers.attach_dna_to_prot_df` joins on
   `(assembly_id, genbank_ctg_id)` and computes
   `dna_hash = md5(dna_seq)` per protein row. (The protein-level
   `seq_hash = md5(prot_seq)` is already on `protein_final` from Stage 1.)
4. **Enrich with parsed metadata.**
   `metadata_enrichment` joins the per-isolate columns from
   `data/processed/flu/metadata_eda/flu_genomes_metadata_parsed.csv`
   — `host`, `hn_subtype`, `year`, `geo_location_clean`, `passage`,
   etc. — onto every protein row.

### Outputs at end of Phase 0

A single working dataframe with one row per protein, every row
carrying:
- assembly_id, brc_fea_id, genbank_ctg_id, canonical_segment
- function, prot_seq, dna_seq
- seq_hash, dna_hash
- per-isolate metadata (host, hn_subtype, year, geo_location_clean, …)

### Why this matters

Everything downstream is a row-level operation on this dataframe. If
metadata enrichment fails for an isolate (e.g., no matching row in the
parsed metadata CSV), that isolate's pairs are still constructable but
the regime-aware sampler in Phase 5 will see null on the missing axes
(and the existing 8-tuple mapping classifies null on an axis as
no-match on that axis — see
[`leakage_definitions.md`](leakage_definitions.md) mode #5).

---

## Phase 1 — Filtering / isolate-pool narrowing

Stage 3 narrows the per-isolate dataframe in this order, before pair
construction. Each step shrinks the pool the pair builder sees.

### Steps

1. **Metadata filtering** — `dataset.host`, `dataset.year`,
   `dataset.hn_subtype`, `dataset.geo_location` filter to matching
   isolates. **None applied in this run** (`filters_applied.host =
   None`, etc., per dataset_stats.json).
2. **Subtype balancing** (optional) — downsample to equal per-subtype
   representation. Not applied here.
3. **`max_isolates_to_process`** — optional random isolate cap. `null`
   here, so no cap.
4. **Function selection** — keep only `virus.selected_functions`
   (the 8 major proteins; see
   [`gto_format_reference.md`](gto_format_reference.md) §6.4).
5. **`schema_pair` narrowing (v2-only)** — narrow to the two functions
   in `schema_pair`. For this run: HA + NA only.

### Result

After filtering, the pool that pair construction sees is:

- **107,524 isolates** that have **both** HA and NA proteins (verified
  from `total.isolates` in dataset_stats.json — Stage 3 considers an
  "isolate" as one that contributes to the positive pair set).
- Of the corpus's 108,530 total assemblies, ~1,006 (~0.93%) are
  dropped for ambiguous-subtype reasons upstream (see
  `conf/dataset/default.yaml` for the `drop_ambiguous_hn_subtype`
  knob). The remainder pass Phase 1 because nothing is filtered out
  here.

---

## Phase 2 — Positive pair construction + dedup

### Step 2a — Cross-product within each isolate

For each isolate with **both** `func_left` (HA) and `func_right` (NA),
take the cross-product of slot-A rows × slot-B rows. In Flu-A this is
usually 1 × 1 = 1 pair per isolate (one HA, one NA).

`pair_mode = schema_ordered`: slot A always carries the
`schema_pair[0]` protein (HA), slot B always carries `schema_pair[1]`
(NA). This removes input direction as a free variable for directed
features like `unit_diff`. (Note: `unit_diff` is abs-based and
symmetric, but the schema ordering is kept for consistency across
bundles.)

### Step 2b — Dedup on `pair_key`

After construction, deduplicate on
`pair_key = canonical(seq_hash_a, seq_hash_b)`. Two isolates that
carry sequence-identical HA + NA proteins collapse to one `pair_key`
— keeping both would let the model see the same example twice and
inflate metrics.

### Verified numbers

From `pos_dedup` in dataset_stats.json:

| Quantity | Value |
|---|---:|
| Pre-dedup pairs (`n_before`) | 107,524 |
| Post-dedup pairs (`n_after`) | 58,388 |
| Dropped duplicates (`n_dropped`) | 49,136 |
| Dedup rate | 45.7% |

The 49,136 dropped is the conservation signal: many isolates share
sequence-identical HA + NA pairs (especially within tight
subtype/year clusters like seasonal H3N2). The PB2/PB1 schema pair
hits ~51% dedup for the same reason on more-conserved proteins.

### Cross-isolate collision signal

`co_occurrence_blocking` in dataset_stats.json reports the same set
viewed as "co-occurrence":

| Quantity | Value | Meaning |
|---|---:|---|
| `total_cooccur_pairs` | 58,388 | unique (HA, NA) `pair_keys` (matches post-dedup positive count) |
| `pairs_in_multiple_isolates` | 9,965 | `pair_keys` that appear in ≥2 isolates |
| `max_isolates_per_pair` | 901 | the single most-replicated pair appears in 901 isolates |

The max=901 reflects a single HA/NA pair deposited 901 times — a
conservation pocket. The dedup-collapsed pair set more broadly drives
the **negative blocking** in Phase 4/5: a candidate negative
`(seq_hash_a, seq_hash_b)` is rejected if that pair appears anywhere
in `total_cooccur_pairs`, because if it co-occurs in any isolate,
calling it "negative" anywhere else would be a contradictory label.

---

## Phase 3 — Split routing (seq_disjoint with hash_key=seq)

### What happens

With `split_strategy.mode = seq_disjoint` and `hash_key = seq`, the
post-dedup positive pairs are routed to train/val/test such that **no
`seq_hash` appears in two splits** on either slot. The algorithm
(`_pair_helpers.seq_disjoint_route_pos_df`):

1. Build a bipartite graph with slot-A `seq_hash`es as one side,
   slot-B `seq_hash`es as the other, and an edge per positive pair.
2. Find connected components (each component is a set of pairs that
   share at least one `seq_hash` on either side, transitively).
3. Bin-pack components into 3 buckets sized to the requested
   train/val/test ratios using LPT-greedy (largest-first).
4. Hard-fail if the resulting routing produces nonzero overlap on the
   active hash family — `seq_disjoint_audit.json` records both
   `seq_hash_overlap` (the guarantee) and `dna_hash_overlap` (a
   diagnostic, since synonymous-codon DNA variants of the same protein
   could in principle diverge — though under `hash_key=seq` they
   stay together).

### Verified numbers

From `seq_disjoint_audit.json`:

| Quantity | Value | Notes |
|---|---:|---|
| Algorithm | `bipartite_cc_lpt_greedy` | |
| `n_components` | 21,719 | distinct connected components on 58,388 pairs |
| `largest_component_pairs` | 11,748 | the biggest single CC holds 20.1% of pairs |
| `singleton_components` | 17,724 | components of size 1 (a unique HA seq paired with a unique NA seq) |
| `top_10_sizes` | 11748, 4177, 2734, 2172, 722, 545, 287, 234, 201, 170 | tail drops off fast — long-tail distribution |
| `n_hashes_a` (unique HA seq_hashes) | 41,629 | |
| `n_hashes_b` (unique NA seq_hashes) | 37,207 | |

| Quantity | Target | Achieved |
|---|---:|---:|
| Train pairs | 46,710 (80%) | 46,710 |
| Val pairs | 5,839 (10%) | 5,839 |
| Test pairs | 5,839 (10%) | 5,839 |

**`pairs_dropped = 0`.** LPT-greedy hit the integer-pair targets
*exactly* because the largest component (11,748 pairs, 20.1% of the
total 58,388) is well inside the train bucket (46,710 pairs).

The audit JSON reports `max_target_deviation_pct = 0.0007%` — that is
a percentage-display artifact, not a real deviation. The fractional
target for train is `0.8 × 58,388 = 46,710.4` pairs; the bin-packer
must choose an integer and picked 46,710, giving an achieved
percentage of `46,710 / 58,388 × 100 = 79.9993%` instead of the ideal
80.0000%. At integer-pair granularity the targets and the achieved
counts are identical.

Cross-split overlap on the **active hash family** (`seq`):

| Hash family | Slot | train↔val | train↔test | val↔test |
|---|---|---:|---:|---:|
| `seq_hash` | A | **0** | **0** | **0** |
| `seq_hash` | B | **0** | **0** | **0** |
| `dna_hash` (diagnostic) | A | 0 | 0 | 0 |
| `dna_hash` (diagnostic) | B | 0 | 0 | 0 |

Under `hash_key=seq` the seq-level overlap is **0 by construction**
(the routing enforces it). The dna-level overlap is reported as a
diagnostic and happens to also be 0 because in this corpus
synonymous-codon DNA variants of HA/NA proteins routed with their
parent protein. Under `hash_key=dna` the partitioning would instead
be on `dna_hash` and the seq-level overlap would be the diagnostic;
that's the looser routing variant (synonymous-codon variants of the
same protein may end up in different splits).

### Other routings

This walkthrough exercises `seq_disjoint` with `hash_key=seq`. Stage 3
also supports `random` (plain shuffle-split, leakage-blind),
`cluster_disjoint` bilateral (BiCC + LPT on mmseqs2 cluster ids), and
`cluster_disjoint single_slot='a'|'b'` (per-cluster atoms on one slot
only — unlocks idXX thresholds past the bilateral feasibility ceiling).
See [`leakage_definitions.md`](leakage_definitions.md) § "Routing
equivalence" for the side-by-side and
[`clustering_overview.md`](clustering_overview.md) § 7 for the
bin-packer algorithm shared across all cluster_disjoint variants.

---

## Phase 4 — Negative sampling: coverage phase

Negative pairs are sampled across isolates with **three guarantees**:

1. **Co-occurrence blocking.** A candidate `(seq_hash_a, seq_hash_b)`
   is rejected if it appears in `total_cooccur_pairs` (any positive
   anywhere). Counted in `co_occurrence_blocking.{train,val,test}_blocked`.
2. **Coverage.** Every sequence (per slot, per `seq_hash` and per
   `dna_hash`) that appears in positives must appear in ≥1 negative
   in the same split.
3. **Regime mix** (Phase 5; opt-in).

The **coverage phase** walks every `(slot, dna_hash)` target and
finds at least one valid negative partner before the fill phase
starts. This guarantees the protein-level imbalance fix (mode #2
leakage; see [`leakage_definitions.md`](leakage_definitions.md)) and
the DNA-level best-effort variant.

### Verified numbers

From `coverage` in dataset_stats.json:

| Split | `requested_negatives` (= 2 × pos) | `min_required_for_coverage` | `coverage_phase_pairs` | `n_seqs_with_zero_negatives` |
|---|---:|---:|---:|---:|
| Train | 93,420 | 42,247 | 57,162 | **0** |
| Val | 11,678 | 5,839 | 7,988 | **0** |
| Test | 11,678 | 5,839 | 7,987 | **0** |

Notes:
- `min_required_for_coverage` = number of distinct `(slot, seq_hash)`
  tuples that need at least one negative. Train has 42,247 — more than
  half of which are covered by a single negative that simultaneously
  fills multiple `(slot, seq_hash)` cells.
- `coverage_phase_pairs` (57,162 train) **overshoots** the strict
  minimum because the sampler often picks negatives that aren't yet
  blocked even after the minimum is met — natural slack as it iterates.
- **`n_seqs_with_zero_negatives = 0`** in every split: the
  protein-level coverage invariant held without any exceptions. The
  v2 builder hard-raises at `dataset_segment_pairs_v2.py:819` if this
  were nonzero.
- DNA-level coverage (`rejection_stats.n_dna_uncovered`) is tracked
  internally; this run had 0 there too (logged at WARNING level
  if > 0; absence of warning = 0).

The 663 train-blocked negatives in `co_occurrence_blocking.train_blocked`
are candidates the sampler proposed that landed in
`total_cooccur_pairs` and got rejected — a small fraction (0.7% of
93,420 train negatives) reflecting how rarely the random sampler
hits a real co-occurring pair.

---

## Phase 5 — Negative sampling: regime-aware fill phase

Once coverage is satisfied, the sampler **fills** to `num_negatives
= round(neg_to_pos_ratio × |pos|)`, biasing toward the configured
per-regime mix. This phase only runs when
`negative_sampling.regime_targets` is set.

### The 8 regimes

For each candidate `(isolate_i, isolate_j)`, classify by which of
{host, hn_subtype, year_bin} match:

| Regime | Match condition |
|---|---|
| `none_match` | host≠ AND subtype≠ AND year≠ |
| `host_only` / `subtype_only` / `year_only` | exactly one axis matches |
| `host_subtype_only` / `host_year_only` / `subtype_year_only` | exactly two axes match |
| `host_subtype_year` | all three match (hardest) |

(Null on an axis classifies as no-match.)

### `regime_targets` mix used in this bundle

From `flu_ha_na_regimes.yaml` (matches the standard mix in
`conf/dataset/default.yaml`):

```
none_match:           0.20
host_only:            0.10
subtype_only:         0.10
year_only:            0.10
host_subtype_only:    0.10
host_year_only:       0.05
subtype_year_only:    0.05
host_subtype_year:    0.30   # heaviest weight on the hardest regime
```

### Verified numbers

From `coverage.<split>.fill_phase_pairs`:

| Split | `coverage_phase_pairs` | `fill_phase_pairs` | `achieved_negatives` (sum) |
|---|---:|---:|---:|
| Train | 57,162 | 36,258 | 93,420 |
| Val | 7,988 | 3,690 | 11,678 |
| Test | 7,987 | 3,691 | 11,678 |

The fill phase ran on every split because `neg_to_pos_ratio = 2.0`
asked for more negatives than the coverage phase produced. The per-
regime achieved mix vs target is logged in
`negative_regime_manifest.{csv,json}` (per split); shortfalls on the
hardest `host_subtype_year` regime are common in val/test on full
Flu-A because the per-cell isolate count in any dominant
host × subtype × year cell is finite.

`coverage_overrode_ratio = False` in every split means the coverage
phase did not need to exceed `requested_negatives` to satisfy the
minimum — i.e. the requested-vs-coverage relationship was healthy.

---

## Phase 6 — Save + audit

### Outputs written

To the run directory `dataset_flu_ha_na_regimes_20260512_114205/`:

- `train_pairs.{csv,parquet}`, `val_pairs.{csv,parquet}`, `test_pairs.{csv,parquet}` — the pair tables.
- `dataset_stats.json` — the full stats snapshot referenced throughout this doc.
- `seq_disjoint_audit.json` — the routing audit (Phase 3,
  `seq_disjoint` mode only). Under `cluster_disjoint` mode the
  routing audit lives in `cluster_disjoint_audit.json` and includes
  `slot_a`/`slot_b` per-family (cluster_id / seq_hash / dna_hash)
  leakage blocks (also surfaced as `slot_leakage_summary` in
  `dataset_stats.json`).
- `negative_regime_manifest.{csv,json}` — per-regime achieved counts per split (Phase 5).
- `duplicate_stats.json` — internal rejection stats (Phase 2 + 4).
- `sequence_exposure_{train,val,test}.csv` — per-`(slot, seq_hash)` exposure (`pos_only`/`neg_only`/`dual`).
- `split_overlap_stats.csv` — per-split unique `seq_hash` / `dna_hash` counts and cross-split overlaps.
- `metadata_coverage.json` — per-axis coverage of the metadata
  axes used by the regime sampler.
- `isolate_metadata.csv` — flat per-isolate metadata for downstream
  analysis tools.
- `resolved_config.yaml` — the full Hydra-resolved config.
- `cooccurring_sequence_pairs.csv` — the `pair_key`s that were
  blocked from negatives in Phase 4.
- `runtime.json` — wall-clock.
- `plots/` — split-composition figures (added by the
  `f8af35f` viz commit).
- `stage3_dataset.log` — stdout/stderr.

### Hard-fail audits at save time

1. **Cross-split `pair_key` overlap** — must be 0 across all 3 split
   pairs. The v2 builder raises if not.
2. **Cross-split `seq_hash` overlap on the active hash family** — must
   be 0. Recorded in `seq_disjoint_audit.json::seq_hash_overlap` and
   in `*_overlap_full_pairs_<side>` (which counts on the full
   pairs table after negatives are added — even tighter than the
   pre-negative routing audit). Raises if not.
3. **`n_seqs_with_zero_negatives` per split** — must be 0 at the
   protein level. Raises with message at line 819 of
   `dataset_segment_pairs_v2.py` otherwise.
4. **Cross-split `cluster_id` overlap on the constrained slot(s)**
   (`cluster_disjoint` mode only) — must be 0 by construction.
   Recorded in `cluster_disjoint_audit.json::cluster_id_overlap`;
   bilateral mode checks both slots, single-slot mode checks the
   constrained slot only. Raises if not
   (`dataset_segment_pairs_v2.py:2020-2042`).

Audits 1–3 passed cleanly in this run (audit 4 does not apply under
`seq_disjoint` routing). The fact that `dataset_stats.json` exists at
all is itself proof since the save step runs after the audits.

### Final shapes

From `split_sizes`:

```
train: 140,130 pairs   (pos  46,710, neg  93,420, ratio 2.00)
val  :  17,517 pairs   (pos   5,839, neg  11,678, ratio 2.00)
test :  17,517 pairs   (pos   5,839, neg  11,678, ratio 2.00)
total: 175,164 pairs   (across 107,524 isolates)
```

`isolate_share`: 80% / 10% / 10% (hard-partitioned).
`pair_share`: 80% / 10% / 10% (matches isolate_share because
`pair_keys` are dedup'd before routing, so the share is bookended by
isolate identity).

---

## Cross-references

- [`pipeline_overview.md`](pipeline_overview.md) §4-§6 — broader
  framing of the same flow with multi-audience prose.
- [`../plans/done/2026-05-11_design_dataset_gen_v2.md`](../plans/done/2026-05-11_design_dataset_gen_v2.md) —
  function-by-function design spec for the v2 builder.
- [`../plans/done/2026-05-10_seq_disjoint_routing_plan.md`](../plans/done/2026-05-10_seq_disjoint_routing_plan.md) —
  Phase 3 implementation plan.
- [`../plans/done/2026-05-09_metadata_aware_negatives_plan.md`](../plans/done/2026-05-09_metadata_aware_negatives_plan.md) —
  Phase 5 design.
- [`leakage_definitions.md`](leakage_definitions.md) — modes #2
  (sequence-level label imbalance, Phase 4 fix), #3 (sequence-level
  leakage, Phase 3 fix), #5 (demographic shortcut, Phase 5
  mitigation).
- [`gto_format_reference.md`](gto_format_reference.md) — Stage 1 input
  schema that Phase 0 reads from.
- [`../conf_guide.md`](../conf_guide.md) — Hydra config system.
- [`../examples/dataset_flu_ha_na_regimes_20260512_114205_dataset_stats.json`](../examples/dataset_flu_ha_na_regimes_20260512_114205_dataset_stats.json) —
  frozen copy of the full run stats this walkthrough references.
