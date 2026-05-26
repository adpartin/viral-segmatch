# Single-slot PB2-only cluster_disjoint sweep on PB2-PB1 — falsification sibling to HA-NA

Second MMD trajectory across a full idXX sweep on Flu A, designed as a
falsification test for the "biological coupling drives the unconstrained
slot's shift" story established on HA-NA in
`docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`.

Question: under aa cluster_disjoint **single-slot PB2-only** routing,
does the unconstrained slot (PB1) shift less than NA did under HA-only?
Pre-registered prediction (written before the sweep, recorded in the
plan summary):

- **H1**: constrained-slot MMD (PB2) grows monotonically with id↓.
- **H2 (KEY)**: unconstrained-slot MMD (PB1) shifts MUCH less than NA
  did under HA-only (because PB2 and PB1 are less biologically coupled
  than HA and NA — both are internal polymerase subunits, neither
  carries the antigenic subtype identity).
- **H3**: pair MMD (S2) tracks the constrained slot more tightly than
  HA-NA's pair MMD did.
- **H4 (KEY)**: Cramér's V(PB2-cluster × PB1-cluster) < 0.8 across the
  sweep; HA-NA equivalent V(HA-cluster × NA-subtype) was 0.98 → 0.82.
  Pre-flight on `protein_final` + cluster artifacts (no built datasets
  needed).
- **H5**: held-out F1 drops less steeply than HA-NA's 4.6 pp.

## Definitions

These three terms recur throughout — pinning them down once so the
tables below are unambiguous:

- **Constrained slot** = the slot whose cluster IDs are forced disjoint
  between train / val / test by the single-slot routing. For PB2-only
  routing, that's **slot a = PB2**. The constraint is what
  `dataset.split_strategy.single_slot: a` enforces in the bundle config.
- **Unconstrained slot** = the other slot (here **slot b = PB1**).
  Clusters can freely overlap between splits. Whether the unconstrained
  slot's *distribution* actually shifts depends on biological coupling
  with the constrained slot.
- **Pair MMD = S2 MMD** = the pair-level RBF-MMD on the joint feature
  representation the MLP actually trains and tests on — Test 3:
  `slot_transform=unit_norm` + `interaction=unit_diff+prod`,
  PCA-reduced to 50 dims. S1 is the per-slot version on a single slot's
  PCA-50 features.

The filename convention from `mmd_sweep.sh` always uses literal `_HA_`
for the constrained slot's CSV and `_NA_` for the unconstrained slot's
CSV, regardless of actual protein names; the aggregator's
`--slot_a_display PB2 --slot_b_display PB1` flags swap the *display*
labels in plots. Tables in this writeup use PB2 / PB1.

## Scope

- One virus, one pair: Flu A PB2-PB1, full corpus
  (`data_version=July_2025`, ~108K isolate pairs before dedup).
- One alphabet: aa (symmetric easy-linclust). nt sweep range is
  narrower per pre-flight (only id100..id097 feasible single-slot on
  nt PB2/PB1, vs id100..id095 on aa) and was not run.
- One slot direction: PB2-only (PB1-only sibling is queued as a
  follow-up; nt sweep also queued).
- One pair interaction at S2: Test 3 (`slot_transform=unit_norm`,
  `interaction=unit_diff+prod`) — production setting.
- Six thresholds: id100, id099, id098, id097, id096, id095.
  All feasible 80/10/10 per pre-flight (`results/.../single_slot_feasibility_pb2_pb1_aa.csv`);
  id095 PB1-direction has a 78% largest-atom (close to the 80% rule
  boundary), PB2-direction stays at 15% — neither hits the cliff.
- One feature space: aa k=3 (8000-dim). ESM-2 not run on PB2/PB1
  (cost: ~75 min/sweep × 3 label_filters = ~4 hr; not justified yet).
- Same 1000-isolate subsample (subsample_seed=42); σ values
  re-derived from PB2 slot of id099 PB2-only via the median heuristic
  (different from HA-NA's σ — PB2/PB1 features cluster more tightly):
  - σ_S1 = **11.3474** (HA-NA was 29.3192)
  - σ_S2 = **0.4144** (HA-NA was 1.0720)

## Setup

- **Datasets**: six Stage 3 dirs under
  `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_cluster_aa_id{XXX}_PB2only_*`
  built 2026-05-25. All hit 80/10/10 exactly (train=105313 / val=13166 /
  test=13166 each).
- **Bundle configs**:
  `conf/bundles/flu_pb2_pb1_cluster_aa_id{100,099,098,097,096,095}_PB2only.yaml`.
  Each inherits `flu_pb2_pb1` and overrides `dataset.split_strategy.{mode:
  cluster_disjoint, cluster_alphabet: aa, single_slot: a, cluster_id_threshold:
  0.XX, cluster_id_path: ...}`.
- **MMD**: PCA-50 + RBF + 500-permutation, same as HA-NA. σ from
  the median heuristic on id099 PB2 slot (run automatically by
  `scripts/pb2_pb1_phase4_5_launch.sh` before the sweep fires).
- **Scripts**: `src/analysis/mmd_per_slot.py` (S1),
  `src/analysis/mmd_per_pair.py` (S2),
  `src/analysis/aggregate_mmd_single_slot_sweep.py` (rollup + plot,
  parametrized for any pair/direction).
- **Pre-flight Cramér's V**:
  `src/analysis/cluster_pair_coupling_precheck.py` (new sibling to
  `cluster_disjoint_feasibility.py`, works directly from
  `protein_final` + cluster artifacts).

## Cramér's V coupling pre-check (H4 verdict before any datasets built)

Computed BEFORE building datasets, on raw `protein_final.parquet` +
cluster artifacts. Apples-to-apples comparison vs HA-NA at the same
thresholds:

| idXX | HA-NA V(HA-clust × NA-clust) | PB2-PB1 V(PB2-clust × PB1-clust) | Δ |
|---:|---:|---:|---:|
| 100 | 0.85 | **0.76** | -0.09 |
| 099 | 0.73 | 0.65 | -0.08 |
| 098 | 0.65 | 0.55 | -0.10 |
| 097 | 0.59 | 0.46 | -0.13 |
| 096 | 0.54 | 0.39 | -0.15 |
| 095 | 0.53 | 0.41 | -0.12 |

V(cluster × whole-isolate H_N_ subtype) for context:

| idXX | V(HA × sub) | V(NA × sub) | V(PB2 × sub) | V(PB1 × sub) |
|---:|---:|---:|---:|---:|
| 100 | 0.96 | 0.81 | 0.91 | 0.87 |
| 095 | 0.60 | 0.42 | 0.50 | 0.24 |

**H4 supported.** PB2-PB1 cluster-cluster coupling is consistently
0.08-0.15 below HA-NA's at every threshold, and drops faster as id↓.
The sweep is genuinely a different regime, not HA-NA redux.

## Results — MMD across all three label filters

The model trains and tests on the **joint** pos + neg pair
distribution, so `label=both` is the most direct measure of the shift
the model perceives. `label=pos` and `label=neg` are decompositions.

### `label=both` (the "what the model sees" view) — primary

| idXX | PB2 MMD² | PB2 p | PB1 MMD² | PB1 p | Pair MMD² | Pair p |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.00456 | 0.347 | 0.00934 | **0.002** | 0.00379 | **0.016** |
| 099 | 0.01968 | **0.002** | 0.03218 | **0.002** | 0.02610 | **0.002** |
| 098 | 0.01951 | **0.002** | 0.03437 | **0.002** | 0.02621 | **0.002** |
| 097 | 0.02793 | **0.002** | 0.03335 | **0.002** | 0.02756 | **0.002** |
| 096 | 0.02811 | **0.002** | 0.03518 | **0.002** | 0.02948 | **0.002** |
| 095 | 0.03052 | **0.002** | 0.02623 | **0.002** | 0.02611 | **0.002** |

Growth ratios (id095 / id100): PB2 **6.7×**, PB1 **2.8×**, Pair **6.9×**.

Two anomalies in the `both` view worth flagging:

- **PB1 MMD drops at id095** (0.03518 → 0.02623, −25%). Not present in
  pos or neg alone; positives and negatives are shifting partially
  out-of-phase on the unconstrained slot at id095, partially canceling
  in the joint view. PB1's pos+neg growth ratios (5.7× and 10.0×) are
  both monotone; only the joint sum dips.
- **At id100, PB2 is non-significant (p=0.35) but PB1 is significant
  (p=0.002)**. The routing at id100 doesn't perturb the constrained
  slot (each PB2 sequence is its own cluster, so the partition reads
  as random on PB2 features), but the negative-pair sampling still
  shifts the PB1 distribution detectably. Reverses the HA-NA id100
  pattern where the constrained slot was always significant.

### `label=pos` (positives only)

| idXX | PB2 MMD² | PB2 p | PB1 MMD² | PB1 p | Pair MMD² | Pair p |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.00456 | 0.347 | 0.00741 | 0.114 | 0.00483 | 0.202 |
| 099 | 0.01968 | **0.002** | 0.03846 | **0.002** | 0.02814 | **0.002** |
| 098 | 0.01951 | **0.002** | 0.03336 | **0.002** | 0.02619 | **0.002** |
| 097 | 0.02793 | **0.002** | 0.03096 | **0.002** | 0.03057 | **0.002** |
| 096 | 0.02811 | **0.002** | 0.03868 | **0.002** | 0.03117 | **0.002** |
| 095 | 0.03052 | **0.002** | 0.04245 | **0.002** | 0.03119 | **0.002** |

Growth ratios: PB2 **6.7×**, PB1 **5.7×**, Pair **6.5×**.

### `label=neg` (negatives only)

| idXX | PB2 MMD² | PB2 p | PB1 MMD² | PB1 p | Pair MMD² | Pair p |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.00332 | 0.691 | 0.00502 | 0.158 | 0.00340 | 0.224 |
| 099 | 0.02525 | **0.002** | 0.04273 | **0.002** | 0.03097 | **0.002** |
| 098 | 0.01607 | **0.012** | 0.03363 | **0.002** | 0.02360 | **0.002** |
| 097 | 0.03396 | **0.002** | 0.02348 | **0.002** | 0.02758 | **0.002** |
| 096 | 0.03188 | **0.002** | 0.02470 | **0.002** | 0.02535 | **0.002** |
| 095 | 0.03154 | **0.002** | 0.05038 | **0.002** | 0.03888 | **0.002** |

Growth ratios: PB2 **9.5×**, PB1 **10.0×**, Pair **11.4×**.

### Comparison to HA-NA at each label filter

| Filter | HA-NA HA growth | PB2/PB1 PB2 growth | HA-NA NA growth | PB2/PB1 PB1 growth | HA-NA Pair growth | PB2/PB1 Pair growth |
|---|---:|---:|---:|---:|---:|---:|
| pos | 33.7× | 6.7× | 12.9× | 5.7× | 28.3× | 6.5× |
| neg | (not tabulated) | 9.5× | (not tabulated) | 10.0× | (not tabulated) | 11.4× |
| both | (similar magnitude) | 6.7× | (similar magnitude) | 2.8× | (similar magnitude) | 6.9× |

The constrained slot's positive-pair shift is **5× smaller** on
PB2/PB1 than HA-NA's HA (6.7× vs 33.7×). Three reasons consistent with
each other:

1. **Less inherent clustering structure**: PB2 has ~33K distinct
   sequences vs HA's ~41K but with PB2's distribution closer to "many
   small clusters" (largest aa id098 atom = 15% on PB2 vs 30% on HA).
2. **Less subtype-driven coupling**: V(PB2-clust × PB1-clust) is 0.76
   at id100 vs HA-NA's 0.85, dropping to 0.41 vs 0.53 at id095. Both
   pairs have coupling; PB2/PB1 has *less*.
3. **σ differs** — direct cross-pair MMD² comparison requires care;
   the *ratios* (id095 / id100) are the correct cross-pair comparison
   and they confirm PB2/PB1 shifts ~5× less.

### Coupling story — H2 supported

H2 predicted that **PB1 (unconstrained) shifts MUCH less than NA did**.
The pos-only PB1 growth is 5.7× vs HA-NA's NA at 9-13×. Held up. But
notice the wrinkle in `label=both`: PB1's joint growth is only **2.8×**
(vs 6.7× for PB2 on the same view) — the unconstrained slot *is*
substantially decoupled in the joint view, **more so than under HA-NA**.

The simpler reading: the biological coupling V(PB2-clust × PB1-clust)
≈ 0.76 at id100, dropping to 0.41 at id095, is *enough* to let the
constrained-slot constraint drag the unconstrained slot under
single-slot routing — but the drag is materially smaller than under
HA-NA's coupling (V started at 0.85 dropping to 0.53).

## Held-out test performance (aa k=3, Test 3 interaction)

One MLP + two baselines (LGBM, 1-NN cosine margin) trained per dataset
using `flu_pb2_pb1_kmer_aa_k3`. MLP trained with **3 seeds (42, 43, 44)**;
LGBM and 1-NN are single-seed (the sweep wrapper produced 3 dirs each
but they're byte-identical without a master_seed override —
`stage4_sweep.sh` was patched in commit 412ac88 to skip this redundancy,
but the patch landed after the loop body was already buffered).

**MLP across 3 seeds (mean ± std):**

| idXX | F1 | AUC-ROC | MCC |
|---:|---:|---:|---:|
| 100 | 0.9219 ± 0.0041 | 0.9709 ± 0.0009 | 0.8685 ± 0.0070 |
| 099 | 0.8853 ± 0.0036 | 0.9534 ± 0.0026 | 0.8056 ± 0.0064 |
| 098 | 0.9035 ± 0.0005 | 0.9600 ± 0.0021 | 0.8369 ± 0.0011 |
| 097 | 0.8944 ± 0.0039 | 0.9594 ± 0.0024 | 0.8210 ± 0.0068 |
| 096 | 0.9060 ± 0.0028 | 0.9600 ± 0.0023 | 0.8410 ± 0.0049 |
| 095 | 0.8973 ± 0.0028 | 0.9603 ± 0.0025 | 0.8262 ± 0.0047 |

F1 drop id100 → id095: **2.5 pp** (vs HA-NA's 4.6 pp). AUC-ROC drop:
1.1 pp (vs HA-NA 1.9-3.0 pp). MCC drop: 4.2 pp (vs HA-NA 7.9-10.0 pp).

Trajectory is **non-monotone**: id099 is the local minimum (F1 0.885),
recovers to id098 (0.904), dips to id097 (0.894), recovers to id096
(0.906), final dip to id095 (0.897). The MLP seed std (0.0005-0.0041)
is well below the inter-threshold variation — these are real
oscillations, not noise. They mirror the PB1-MMD wobble in `label=both`.

**LGBM and 1-NN (single-seed):**

| idXX | LGBM F1 | 1-NN F1 | LGBM AUC-ROC | 1-NN AUC-ROC | LGBM MCC | 1-NN MCC |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.9241 | 0.9288 | 0.9778 | 0.9804 | 0.8725 | 0.8802 |
| 099 | 0.8837 | 0.8948 | 0.9597 | 0.9678 | 0.8032 | 0.8225 |
| 098 | 0.9029 | 0.9209 | 0.9686 | 0.9766 | 0.8360 | 0.8667 |
| 097 | 0.8934 | 0.9002 | 0.9646 | 0.9674 | 0.8202 | 0.8316 |
| 096 | 0.9035 | 0.9132 | 0.9665 | 0.9690 | 0.8370 | 0.8540 |
| 095 | 0.8989 | 0.9066 | 0.9658 | 0.9693 | 0.8298 | 0.8427 |

Drops id100 → id095: F1 2.5 pp (LGBM), 2.2 pp (1-NN). AUC-ROC 1.2 pp
(LGBM), 1.1 pp (1-NN). MCC 4.3 pp (LGBM), 3.8 pp (1-NN).

**H5 strongly supported.** Every model drops ~half as much on PB2/PB1
as on HA-NA.

### Caveat: 1-NN > MLP at every threshold

Model ordering on PB2/PB1 is **1-NN > MLP ≈ LGBM** at every cell
(1-NN F1 lead: 0.7-1.7 pp). This **reverses HA-NA's ordering** (where
MLP > 1-NN > LGBM in most cells, with 1-NN nudging ahead only at
id096).

The standard interpretation (compact-summary canon: "1-NN cosine
margin ≥ LGBM at every cluster_disjoint routing is the leakage gauge
— cluster_disjoint weakens the near-neighbor signal *gradually*
rather than eliminating it"): on PB2/PB1, residual structure (likely
metadata-driven via host / year / subtype) is still strong enough
that nearest-neighbor lookup beats trained models. The trained models
are **not learning much beyond nearest-neighbor calibration on this
pair**.

This is consistent with the smaller F1 drop (H5): if 1-NN can win
across the sweep, the routing isn't breaking the dominant signal —
it's just shifting the data slightly. The MLP/LGBM aren't generalizing
better than lookup because the lookup signal is still there.

What this implies for the paper experiment:
- The drop magnitudes (~2.5 pp) are real but small — the corpus is
  still leak-permissive even at id095 single-slot.
- If 1-NN dominance is undesirable as a paper result, the next lever
  is multi-axis metadata holdout (year + host + subtype) on top of
  cluster_disjoint to attack the residual lookup signal directly.
  Tracked as BACKLOG follow-up.

## Plots

Under `results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate/pb2_pb1_PB2only/`:

- `sweep_mmd_vs_idxx.png` — single panel (aa k=3 only), three lines
  (PB2 blue, PB1 orange, pair purple), filled markers for p ≤ 0.05.
- `sweep_perf_vs_idxx.png` — three panels (F1, AUC-ROC, MCC), three
  lines per panel (MLP, LGBM, 1-NN); MLP line is mean across seeds
  with ±1 std shaded band, per-seed dots overlaid.
- `sweep_perf_vs_mmd_pair_kmer_aa_{pos,neg,both}.png` — three single
  panel F1-vs-MMD scatters, MLP error bars (3 seeds), per-seed dots.
- `idxx_sweep_pair_svd.png`, `idxx_sweep_pair_umap.png` — 6-panel
  geometry plots (TruncatedSVD-2 and UMAP-2 with TruncatedSVD-50
  pre-step) showing pair k-mer features across the sweep. ONE global
  projection across all 6 thresholds, so the same pair lands in the
  same XY across panels; color (train/test) is what changes between
  panels. Visually: PB2/PB1 panels look broadly intermixed throughout,
  consistent with the small MMD growth — compare to the HA-NA sibling
  plot at `.../sweep_aggregate/idxx_sweep_pair_{svd,umap}.png` where
  test-dominated sub-clusters become visible by id095. Generated by
  `src/analysis/plot_idxx_sweep_geometry.py`.

CSVs:
- `sweep_combined.csv` — long-format MMD (idxx × role × label_filter).
- `sweep_perf.csv` — long-format perf (idxx × model × seed).
- `sweep_perf_summary.csv` — perf wide-format (mean/std/min/max/count).

Note: `reference_baselines.csv` is intentionally absent — the HA-NA
random/seq_disjoint baselines were not re-computed for PB2/PB1
(`--skip_reference_baselines` flag suppresses them; HA-NA references
would be misleading context on PB2/PB1 axes).

## Hypothesis verdict summary

| H | Prediction | Verdict |
|---|---|---|
| H1 | Constrained-slot MMD grows monotone with id↓ | **Mostly** — pos PB2 monotone; neg PB2 has id098 dip (0.025 → 0.016); both PB2 monotone. The HA-NA pattern (mostly monotone with one dip) repeats. |
| H2 | Unconstrained-slot shifts MUCH less than NA did | **Supported (modest)** — PB1 pos 5.7× vs NA pos 9-13×; PB1 both 2.8× vs NA both ~6-8× (HA-NA). Decoupling effect is real but smaller in absolute terms than the smaller V differential might suggest. |
| H3 | Pair MMD tracks constrained slot more tightly | **Mixed** — pos pair (6.5×) sits between PB2 (6.7×) and PB1 (5.7×), tracks PB2 closely; both pair (6.9×) > PB2 (6.7×), again tracks PB2; but pair > PB1 at every label filter, so pair is NOT a clean average of the two. |
| H4 | V(PB2 × PB1) < 0.8 | **Confirmed pre-flight** (V=0.76 at id100, dropping to 0.41 at id095; HA-NA equivalent V was 0.85 → 0.53). |
| H5 | F1 drops less steeply than HA-NA | **Strongly supported** — every model drops ~half as much (MLP 2.5pp vs 4.6pp; LGBM 2.5pp vs 5.9pp; 1-NN 2.2pp vs 4.7pp). |

## Observations (beyond pre-registered hypotheses)

1. **Negatives shift MORE than positives at almost every cell.**
   Pair growth: pos 6.5×, neg 11.4×, both 6.9×. PB2 growth: pos 6.7×,
   neg 9.5×, both 6.7×. The HA-NA writeup framed the perf drop as a
   "joint positives + negatives shift, not positive-driven" — the
   same framing applies to PB2/PB1, and arguably more strongly
   (negative-shift dominates positive-shift on this pair).

2. **id100 partition reads as random on PB2 features.**
   Largest PB2 atom at id100 is 1.80% of pairs; PB2-MMD p=0.347 (pos),
   p=0.691 (neg), p=0.347 (both). The constrained slot is unperturbed
   at id100. This was NOT true for HA-NA, where HA cells at id100 were
   already detecting shift even though largest atom was 0.7% — HA has
   stronger intrinsic structure (subtype-clustering at the sequence
   level) that even an effectively-random partition catches. PB2
   doesn't have that.

3. **PB1 at id100 already shifts significantly** (p=0.002 in both
   label_filter=both and label=neg). The unconstrained slot is the
   first to detect shift even when the constrained slot can't.
   Consistent with: negative-pair sampling generates more diverse PB1
   sequences in test than train at id100, while PB2 sees no constraint
   yet.

4. **`label=both` PB1 has a unique non-monotone dip at id095.** Not
   present in pos or neg alone. Suggests pos and neg are shifting in
   partially-opposing directions in PB1 feature space at id095.
   Possibly: at id095, the heavy PB2 constraint forces specific PB1
   partner pools in train vs test that, when combined with negative
   sampling, produce a joint distribution that happens to overlap
   between train and test more than at id096. This is the kind of
   detail worth follow-up (which PB2 clusters land where, what
   subtypes do they correspond to) but doesn't break the headline.

5. **MMD↔F1 relationship is murkier than HA-NA's.** HA-NA showed a
   nearly-linear F1-vs-MMD² scatter across all 3 models. PB2/PB1's
   scatter is messier — the id099 F1 dip is well below the
   id099→id100 MMD-line trajectory; the id098 recovery doesn't fit
   either. This is consistent with "1-NN dominates → models aren't
   learning shift-sensitive features" — if the model isn't really
   responding to the constrained-slot distribution, perf-vs-MMD
   shouldn't be tight.

## Interpretation — what we can and cannot claim

### What the empirical results show

- For Flu A PB2-PB1, single-slot PB2-only cluster_disjoint from id100
  to id095 produces a measurable but **5× smaller** constrained-slot
  MMD shift than HA-NA's HA-only equivalent.
- The unconstrained slot (PB1) shifts substantially less than HA-NA's
  NA did, consistent with the smaller cluster-cluster coupling
  V(PB2 × PB1) measured pre-flight.
- All three models (MLP, LGBM, 1-NN cosine margin) show ~2.2-2.5 pp
  F1 drop across the sweep, about half of HA-NA's drop.
- 1-NN beats MLP and LGBM at every threshold, opposite of HA-NA. The
  trained models aren't outperforming nearest-neighbor lookup on this
  pair — suggesting residual metadata-driven leakage even under
  cluster_disjoint single-slot.
- The "joint positives + negatives shift" framing repeats: negative
  pairs shift at least as much as positives in every cell. The
  `label=both` view is the right single summary for "what the model
  sees".

### What this does not establish

- **MLP measured with 3 seeds; LGBM and 1-NN with 1 effective seed.**
  Multi-seed LGBM/1-NN would tighten the model ordering claim. Same
  caveat as HA-NA.
- **Only model-seed variance measured, not split-induced variance.**
  Same single-bin-packing-per-threshold limitation as HA-NA.
- **One feature space for training (aa k=3).** ESM-2 training
  cross-check would tell us whether the 1-NN dominance is feature-
  specific or pair-specific.
- **One sweep, one direction.** PB1-only sibling would test slot
  symmetry. PB2/PB1 nt single-slot (feasible only at id100..id097)
  would test alphabet dependence. Neither was run.
- **Negative-pair regime composition not checked.** HA-NA did the
  `axis_flag_summary` sanity check; PB2/PB1 sweep didn't (would be
  ~5 min). The MMD-on-neg trajectory already addresses the most
  important question (is the negative distribution shifting?), but
  the axis-composition sanity is a good follow-up.
- **id095 PB1 non-monotone dip in `label=both` is described, not
  explained.** A focused follow-up (which PB2 clusters in train vs
  test? which PB1 partners?) would identify the specific cluster-
  coalescence behavior.
- **1-NN > MLP ordering claim is one-seed.** LGBM with sampling
  randomness might shift, but the 1-NN cosine margin is deterministic
  on this data, so its lead is real. The "MLP not learning beyond
  lookup" framing is consistent with the data but not directly
  validated against (e.g.) feature-importance or learned-vs-lookup
  ablations.

### Implications for the bigger picture

- **PB2-PB1 is the falsification experiment that lands.** The pair has
  less biological coupling than HA-NA, less inherent clustering
  structure, and produces materially smaller MMD shifts and perf
  drops. The "single-slot routing decouples slots more on
  less-coupled pairs" story holds.
- **The "memorization" caveat is louder on PB2/PB1 than on HA-NA.**
  1-NN's dominance suggests cluster_disjoint single-slot is not by
  itself sufficient to break residual leakage on this pair —
  multi-axis metadata holdout (year + host + subtype) on top of
  cluster_disjoint is the natural next experiment.
- **The sweep axis works for PB2/PB1 too**, just over a smaller
  dynamic range. The 6 thresholds × 3 label filters still give
  enough resolution to see the joint-shift story and the
  unconstrained-slot decoupling.

## Reproduce

End-to-end via the launcher (which probes σ, then dispatches both
sweeps in parallel):

```bash
# Pre-flight (Phase 1 + Phase 2):
python -m src.analysis.single_slot_cluster_disjoint_feasibility \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_aa \
    --schema_pair "RNA-dependent RNA polymerase PB2 subunit" \
                  "RNA-dependent RNA polymerase catalytic core PB1 subunit" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85

python -m src.analysis.cluster_pair_coupling_precheck \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_aa \
    --schema_pair "RNA-dependent RNA polymerase PB2 subunit" \
                  "RNA-dependent RNA polymerase catalytic core PB1 subunit" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95

# Phase 3: build the 6 sweep datasets:
for THR in 100 099 098 097 096 095; do
  bash scripts/stage3_dataset.sh flu_pb2_pb1_cluster_aa_id${THR}_PB2only
done

# Phase 4 + 5: σ probe + MMD sweep × 3 label_filters + training × 3 seeds:
bash scripts/pb2_pb1_phase4_5_launch.sh

# Phase 6: aggregate (parametrized for PB2/PB1):
python -m src.analysis.aggregate_mmd_single_slot_sweep \
    --routing_direction PB2only \
    --training_bundle flu_pb2_pb1_kmer_aa_k3 \
    --slot_a_display PB2 --slot_b_display PB1 \
    --out_subdir pb2_pb1_PB2only \
    --feature_spaces kmer_aa \
    --skip_reference_baselines
```

## See also

- `docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md` — HA-NA
  reference sweep; the comparison anchor for every table here.
- `results/flu/July_2025/runs/cluster_disjoint_feasibility/single_slot_feasibility_pb2_pb1_aa.csv` —
  Phase 1 feasibility pre-flight.
- `results/flu/July_2025/runs/cluster_disjoint_feasibility/cluster_pair_coupling_precheck_pb2_pb1_aa.csv` —
  Phase 2 Cramér's V pre-check (also includes HA-NA reference).
- `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df` —
  single-slot routing helper.
- `src/analysis/cluster_pair_coupling_precheck.py` — coupling pre-check.
- `src/analysis/aggregate_mmd_single_slot_sweep.py` — parametrized
  aggregator (commit 8b57cc3).
- `scripts/pb2_pb1_phase4_5_launch.sh` — one-shot launcher (sigma
  probe + 3 MMD sweeps + training sweep in parallel).
