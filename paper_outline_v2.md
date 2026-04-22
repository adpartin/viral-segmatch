# Paper Outline: Viral Segment Matching

**Working title:** Predicting Same-Isolate Origin of Genome Segments from Protein and Nucleotide Features

**Status:** Draft outline — updated to align with 2026-03-12 group meeting decisions.
See `roadmap_v2.md` for implementation details, task status, effort estimates, and execution order.

---

## Abstract

_One paragraph. Update after results are finalized._

> **[2026-03-12 meeting]** Scope: Influenza A only. K-mers primary; ESM-2 as comparison baseline. 8×8 all-segment heatmap and stratified evaluation are central results. Use cases deferred until model validation is complete.

Problem: Influenza A viruses have genomes split across eight RNA segments.
Linking segments to the same isolate is essential for genomic surveillance, but public
databases often lack reliable metadata for this linkage. We frame segment matching as
binary classification: given two sequence representations, predict whether they originate
from the same viral isolate. We evaluate k-mer frequency features (primary) and protein
language model representations (ESM-2, comparison baseline) with an MLP classifier on a
["subtype-balanced dataset of ~50–100K" --> update this based on what we actually run] Flu A isolates across all 28 segment-pair
combinations. We assess model performance stratified by HN subtype, host, and isolation
year, and demonstrate the model's utility for data remediation and wastewater surveillance
applications.

---

## 1. Introduction

### Motivation

> **[2026-03-12 meeting]** Scoped to Influenza A only. Bunyavirales moved to Future Directions.

Influenza A viruses have genomes split across eight RNA segments. Each segment is often
sequenced and deposited independently, making it difficult to determine which segments
originated from the same viral isolate. Accurate linking of genome segments to the same
isolate is critical for:

1. **Data remediation.** Public genomic databases (BV-BRC, NCBI GenBank) ingest records
   without enforcing metadata fields that link segments to the same isolate. Records may
   lack shared isolate IDs or have inconsistent naming. Computational segment matching
   can re-link orphaned records, improving database quality and utility.

2. **Wastewater surveillance.** Metagenomic sequencing of wastewater recovers mixed viral
   fragments from entire communities with no isolate-level metadata. Segment matching can
   reconstruct which fragments likely came from the same circulating strain, turning
   fragmented metagenomic signal into actionable surveillance data (Wolfe et al. 2024 —
   CDC influenza A H5N1 monitoring in US wastewater during the 2024 avian flu outbreak).

### Problem formulation

Given representations of two segments from Influenza A, predict whether they originate
from the same isolate (binary classification). We evaluate all 28 pairwise combinations
of the 8 major segment gene products (PB2, PB1, PA, HA, NP, NA, M1, NS1).

### Contributions

- Frame viral segment-to-isolate linkage as a binary classification task using sequence
  representations (first to our knowledge)
- Evaluate k-mer frequency features as the primary representation and compare against
  protein language model (ESM-2) embeddings as a baseline
- Demonstrate segment matching across all 28 pairwise segment combinations (8×8 heatmap)
  on a large, subtype-balanced Influenza A dataset (~50–100K isolates)
- Evaluate model performance stratified by HN subtype, host, and isolation year to
  characterize where the model succeeds and where it fails
- Provide calibrated uncertainty estimates enabling users to assess prediction reliability
- Demonstrate data remediation and wastewater surveillance applications

> **[2026-03-12 meeting]** Low priority for this paper: The ESM-2 concat failure mode finding. Only relevant if ESM-2 is included as a comparison baseline.

---

## 2. Methods

### 2.1 Data

- **Source:** ~111K Flu A isolates from BV-BRC (curated by Jim Davis). Each GTO contains
  genome and protein sequences, and various metadata.
- **Preprocessing:** Protein sequences mapped to 8 major proteins (one per segment): PB2,
  PB1, PA, HA, NP, NA, M1, NS1. Alternative reading frame products (M2, NEP) excluded.
  Nucleotide genome sequences extracted per segment for k-mer features. Quality filters
  applied to both protein and genome sequences.
- **Dataset scale:** ~50–100K isolates (see `roadmap_v2.md` Task 2 for scaling plan).

#### Terminology

- **Subtype-balanced** — explicitly resampled to equalize HN-subtype representation.
- **Subtype-filtered** — restricted to one subtype.
- **Natural** / **unfiltered** — sampled proportional to the (BV-BRC) data frequencies.

#### 2.1.1 Negative pair taxonomy and pair-distribution analysis

> **[2026-03-12 meeting]** New from group discussion. The **analysis framework** is the unconditional priority; balanced training is a conditional data-centric lever.

**Negative pair taxonomy.** A negative pair consists of segments from two different
isolates. Two regimes:

1. **Within-subtype negatives** (e.g., H3 from H3N2 isolate A + N2 from H3N2 isolate B):
   Hard negatives. Both isolates share the same HN subtype, so the model must use
   isolate-specific signal, not population-level signal.
2. **Cross-subtype negatives** (e.g., H3 from H3N2 isolate A + N1 from H5N1 isolate C):
   Easy negatives. The isolates come from different HN subtypes, so the model can exploit
   population-level differences.

This two-regime taxonomy generalizes to all 28 segment pairs — "subtype" is an
isolate-level property, so it applies regardless of which segments are paired. For
non-HA/NA pairs (internal/conserved segments), within-subtype negatives may be even harder
because internal proteins are more conserved across isolates within a subtype. The 8×8
heatmap (Section 3.7) crossed with this stratification will reveal this.

Random sampling heavily overrepresents easy cross-subtype negatives, which inflates
apparent accuracy.

**Priority order:**

1. **Stratified error analysis by pair type (unconditional).** Classify every test
   prediction into the two categories + positives, report metrics per category. The
   analysis framework is the real deliverable, not any specific fix.
2. **If the model performs well across both pair types** — no rebalancing needed.
3. **If performance degrades on within-subtype negatives** — choose interventions
   (data-centric and model-centric in parallel; see Section 3.5).

**Metadata confounders.** HN subtype is correlated with host, geography, and year in the
natural data. Report marginal stratified metrics per independent axis and acknowledge
confounders explicitly. Fully disentangling these factors would require controlled subsets
that may not exist in the natural data. See Section 3.5.1 for the hierarchical analysis
framework.

**Subtype-balanced sampling (data-centric lever, conditional):** If stratified analysis
reveals subtype-dependent failures, balance positive pairs by subtype and control the
within-subtype vs cross-subtype negative ratio explicitly.

**Test set:** Report performance on within-subtype and cross-subtype negatives separately.

> See `roadmap_v2.md` Appendix for biology context on H-type/N-type/HN subtype terminology.

#### 2.1.2 Pair-distribution ledger

For every experiment, maintain a full accounting of pair distributions:

| Pair type | Subtype(s) | Label | Count |
|-----------|-----------|-------|-------|
| H3 + N2 | H3N2 × H3N2 | pos | ... |
| H3 + N2 | H3N2 × H3N2 | neg (within-subtype) | ... |
| H3 + N1 | H3N2 × H5N1 | neg (cross-subtype) | ... |
| H5 + N1 | H5N1 × H5N1 | pos | ... |
| H5 + N1 | H5N1 × H5N1 | neg (within-subtype) | ... |
| ... | ... | ... | ... |

Enables: (a) verifying balance, (b) stratified metric reporting, (c) using negative
distribution as a data-centric lever.

### 2.2 Feature representations

| Feature | Source | Dimensionality | Description | Role |
|---------|--------|---------------|-------------|------|
| K-mer (k=6) | Nucleotide sequences | 4096 | Sparse frequency vectors over 6-mer vocabulary | **Primary** |
| ESM-2 | `esm2_t33_650M_UR50D` | 1280 | Frozen protein language model mean-pool embeddings | Comparison baseline |

### 2.3 Interaction approaches

| Interaction | Formula | Output dim | Properties |
|-------------|---------|-----------|------------|
| `concat` | [A, B] | 2D | Preserves both embeddings; order-sensitive |
| `unit_diff` | (A − B) / ‖A − B‖₂ | D | L2-normalized signed difference; retains direction only |

### 2.4 Slot transform architecture variants

Five `slot_transform` variants:

| Mode | Slot A path | Slot B path | Key property |
|------|------------|------------|-------------|
| `none` | pass-through | pass-through | Raw embeddings; no learned transform |
| `shared` | Transform(a) | Transform(b) | Same transform for both slots |
| `slot_specific` | Transform_A(a) | Transform_B(b) | Independent per-slot transforms |
| `shared_adapter` | Transform(a) + Adapter_A(·) | Transform(b) + Adapter_B(·) | Shared transform + per-slot residual adapters |
| `slot_norm` | [Transform(a)→] LN_A(·) | [Transform(b)→] LN_B(·) | Per-slot normalization; neutralizes subspace offset |

`slot_norm` is the current best configuration.

```
(a) none — pass-through

  emb_a ─────────────────────────┐
                                 ├─→ Interaction ──→ MLP Classifier ──→ P(same isolate)
  emb_b ─────────────────────────┘


(b) shared — single shared transform

  emb_a ──→ [ Shared Transform ] ──→ a' ─┐
                                          ├─→ Interaction ──→ MLP Classifier ──→ P(same isolate)
  emb_b ──→ [ Shared Transform ] ──→ b' ─┘
                (same weights)


(c) slot_specific — independent transform per slot

  emb_a ──→ [ Transform_A ] ──→ a' ─┐
                                     ├─→ Interaction ──→ MLP Classifier ──→ P(same isolate)
  emb_b ──→ [ Transform_B ] ──→ b' ─┘
            (separate weights)


(d) shared_adapter — shared transform + per-slot residual adapters

  emb_a ──→ [ Shared Transform ] ──→ z_a ──→ z_a + Adapter_A(z_a) ──→ a' ─┐
                                                                            ├─→ Interaction ──→ MLP ──→ P
  emb_b ──→ [ Shared Transform ] ──→ z_b ──→ z_b + Adapter_B(z_b) ──→ b' ─┘
                (same weights)               (separate adapter weights)


(e) slot_norm — optional shared transform + per-slot LayerNorm  ** current best **

  emb_a ──→ [ Shared Transform ]? ──→ [ LayerNorm_A ] ──→ a' ─┐
                                                                ├─→ Interaction ──→ MLP ──→ P
  emb_b ──→ [ Shared Transform ]? ──→ [ LayerNorm_B ] ──→ b' ─┘
                (optional)            (separate LN params)
```

### 2.5 Classifier

MLP binary classifier on interaction features. Trained with BCEWithLogitsLoss, early
stopping on validation loss.

### 2.6 Evaluation metrics

F1, AUC-ROC, precision, recall, Brier score. Optimal threshold selected on validation set.

---

## 3. Experiments & Results

> **[2026-03-12 meeting]** Decision tree: scale up → 8×8 heatmap → stratified eval → fix only where performance dips. Mitigation is conditional.

### 3.1 Baseline performance (random split)

- ESM-2 + unit_diff on mixed-subtype HA/NA: AUC ~0.966, F1 ~0.917
- K-mer + unit_diff on mixed-subtype HA/NA: AUC ~0.982, F1 ~0.957
- Needs re-run at ~50–100K scale.

### 3.2 Cross-validation (N=5 folds)

> **[2026-03-12 meeting]** Not discussed. Default/must-have.

Report mean ± std of AUC, F1, precision, recall across 5 folds.

### 3.3 Temporal holdout (train 2021–2023, test 2024)

> **[2026-03-12 meeting]** Jim reframes temporal as a post-hoc stratification axis. Keep both: explicit temporal holdout + post-hoc year stratification.

Preliminary results (known dedup artifact — needs re-run):

| Metric | ESM-2 (unit_diff) | K-mer k=6 (unit_diff) |
|--------|-------------------|----------------------|
| AUC-ROC | 0.891 | **0.941** |
| F1 | 0.734 | **0.832** |

### 3.4 ESM-2 embedding geometry: concat failure on homogeneous populations

> **[2026-03-12 meeting]** Not discussed. Low priority for Paper 1. Brief treatment if ESM-2 included.

- ESM-2 concat fails on H3N2-only (AUC=0.498); unit_diff succeeds (AUC=0.957)
- K-mer concat does NOT fail on H3N2 (AUC=0.985)
- Explanation: ESM-2 HA/NA embeddings occupy distinct subspaces → learnable shortcut

### 3.5 Stratified evaluation and error analysis

> **[2026-03-12 meeting]** Stratified error analysis by pair type is **unconditional.** Interventions are conditional and can be explored in parallel (data-centric + model-centric).

#### 3.5.1 Stratified performance by pair type and metadata (UNCONDITIONAL)

The core analytical deliverable. Must be done for every experiment.

**Level 1: Pair-type regime (always report).** Metrics per category:

| Category | Description | Expected difficulty |
|----------|-------------|-------------------|
| Positive pairs | Same isolate | — |
| Within-subtype negatives | Different isolate, same HN subtype | Hard |
| Cross-subtype negatives | Different isolate, different HN subtype | Easy |

**Level 2: Per-category marginals (report where data supports).** Stratify by each
metadata axis independently, collapsing over all other axes:
- **HN subtype:** Top 5–8 subtypes + "other/rare" bucket
- **Host:** human, avian, swine, other
- **Geographic region:** continent-level
- **Isolation year:** per-year or per-era

Minimum ~50–100 test pairs per category; pool smaller categories into "other."

**Level 3: Cross-tabulation heatmaps (diagnostic, not publication).** Full subtype×subtype
FP rate matrix. Spot unexpected patterns; guide targeted drill-downs.

**Metadata confounders.** HN subtype correlates with host, geography, and year. Report
marginal metrics per independent axis (not crossed factors). Acknowledge confounders
explicitly. Targeted controlled analyses only where data supports it.

**Non-HA/NA pairs.** Within-subtype negatives may be harder for internal segment pairs
because internal proteins are more conserved. The 8×8 heatmap crossed with pair-type
stratification will reveal this.

#### 3.5.2 Pair-level error characterization

Known issue: high FP/FN ratio (64:1 ESM-2 temporal, 11.6:1 k-mer temporal). Model
over-predicts same-isolate origin. Characterize which pairs are wrong and why using
pair-level metadata matrix and FP/FN concentration analysis.

#### 3.5.3 Diagnostics (conditional — only if 3.5.1 reveals failures)

1. Embedding distance distributions (positive vs within-subtype neg vs cross-subtype neg)
2. Predicted probability histograms (TPs, FPs, FNs, TNs)
3. Pair-level FP rate matrix by (subtype_a × subtype_b)

#### 3.5.4 Interventions (conditional — parallel data-centric + model-centric)

> **[2026-03-12 meeting]** Only if stratified eval reveals failures. Both approaches are **parallel options**, not sequential.

Fix val/test sets. Modify training approach only. Evaluate with same stratified analysis.

**Data-centric:** Hard negative mining, controlled negative ratio, subtype-balanced
training, curriculum learning, negative-to-positive ratio.

**Model-centric:** XGBoost/LightGBM (different inductive bias), contrastive learning,
focal loss, weighted BCE, custom loss / multi-task learning.

The stratified analysis framework (Section 3.5.1) is the constant across all
interventions, enabling direct comparison.

#### 3.5.5 Stress testing across metadata axes (optional)

| Experiment | Train on | Test on | Tests |
|-----------|----------|---------|-------|
| Cross-subtype | H3N2 | H1N1 | Subtype generalization |
| Cross-geography | USA | China | Geographic generalization |
| Cross-host | Human | Avian | Host generalization |
| Temporal | 2021–2023 | 2024 | Already in Section 3.3 |

### 3.6 Mixed-subtype discrimination test (Carla's wastewater-motivated test)

> **[2026-03-12 meeting]** New. Tests whether the model handles a soup of segments from multiple subtypes.

**Test set:** Segments from multiple HN subtypes:
- Positive: same-isolate pairs from H3N2, H5N1, H1N1, etc.
- Within-subtype negatives: different-isolate within same subtype
- Cross-subtype negatives: segments from different subtypes

**Key question (Jim):** Can the model work out of the box without knowing HN type? If not,
consider pre-filtering or per-pair classifiers.

**Training options:** (a) one subtype → test mixed (cross-subtype generalization);
(b) subtype-balanced training → test on subtype-balanced mix (Jim's formulation)

### 3.7 All protein-pair combinations (8×8 heatmap)

> **[2026-03-12 meeting]** **ELEVATED TO HIGH PRIORITY.** Jim's #2 ask.

28 pairwise combinations of 8 major proteins. Results as 8×8 AUC/F1 heatmap. Shows which
pairs are easiest/hardest; whether model generalizes beyond HA/NA.

> Biology context: HA/NA most variable (immune targets); internal segments more conserved → expect performance gradient. See `roadmap_v2.md` Appendix.

**Preliminary results (Polaris, 2026-04-08).** 12-fold CV on the full Flu A dataset
(~111K isolates), k-mer k=6 + slot_norm + concat, 100 epochs per fold.
334/336 folds completed (2 launcher races on `pb1_ha/fold11` and `pb2_pa/fold6`; not
re-run). All 28 pairs achieve val AUC ∈ [0.9924, 0.9958] (median 0.9944) and
val F1 ∈ [0.957, 0.982] (median 0.971). Per-pair fold std σ_AUC ≈ 0.0005–0.0009 — CV
is highly stable. M1-containing pairs are easiest (PA·M1, HA·M1, PB2·M1, PB1·M1 ≈
0.9956); NA-containing surface pairs are hardest (NA·NS1 0.9924, PB2·NA 0.9927). The
variability gradient predicted from biology (HA/NA most variable, internals more
conserved) is **not** strongly visible — the model handles all 28 pair types well above
chance, and the AUC spread across pairs is only ~0.0034. This suggests k-mer features
capture isolate-specific signal across both variable and conserved segments.
Heatmap PNG: `models/flu/July_2025/allpairs_prod_20260408_063203/heatmap_auc_roc.png`.

---

## 4. Application: Data Remediation and Wastewater Surveillance

> **[2026-03-12 meeting]** Deferred until model validation is solid. Wastewater data sourcing initiated (Carla → Rachel Paretsky; Jim has ~20K assembled mixed samples; SRA backup).

### Concept

Train on curated dataset, apply in inference mode to: (1) BV-BRC records lacking isolate
linkage, (2) wastewater metagenomes.

### 4.1 Large-scale inference

Batch inference at BV-BRC scale. Candidate pruning by metadata overlap. HPC on Polaris.

### 4.2 Uncertainty quantification (UQ) — REQUIRED

**Rationale:** UQ is essential for two complementary reasons:

1. **Operational necessity.** Users must know which predictions to trust. A probability of
   0.95 is actionable; 0.55 is not.

2. **Complements stratified analysis, addresses the long tail.** The stratified evaluation
   (Section 3.5.1) characterizes performance on well-populated metadata categories. But the
   real world includes rare subtypes and unusual hosts where stratified metrics are
   unreliable due to small samples. UQ sidesteps this: a well-calibrated model flags
   uncertain predictions at inference time, regardless of *why* it's uncertain. Stratified
   analysis tells you *where* the model fails (population-level); UQ tells the *user*
   *when* it's failing on their specific input (instance-level). They are complementary.

**Approaches:**
- **Deep ensemble (primary):** N=5 CV fold models are already an ensemble. Report mean ±
  std across members. No additional training required.
- **Calibration analysis (required):** Reliability diagram + ECE. Temperature/Platt scaling
  if needed.
- **MC Dropout (optional):** Alternative for single-model inference.
- **Conformal prediction (optional):** Distribution-free coverage guarantees.

### TODO

- [ ] Quantify unlinked BV-BRC records (scope Section 4)
- [ ] Inference script
- [ ] UQ implementation (ensemble + calibration)
- [ ] Wastewater data sourcing

---

## 5. Discussion

### Key findings

- K-mers match or exceed ESM-2 on all configurations. Simpler, lower compute, better
  generalization.
- 8×8 heatmap reveals variability gradient across segment pairs.
- Stratified evaluation characterizes where the model succeeds and where caution is needed.
- Calibrated uncertainty enables users to assess individual prediction reliability.

### Limitations

- High FP/FN ratio (model over-predicts same-isolate origin); Section 3.5 characterizes
- Negative pair distribution may overrepresent easy cross-subtype negatives if not balanced
- Metadata confounders (subtype correlates with host/geography/year) limit causal claims
- Requires fully assembled segments; reads-based approach out of scope

### Future directions

- **Reads-based approach:** From reads rather than assembled segments (Jim raised, Carla
  recommended deferring)
- **Other segmented viruses:** Bunyavirales, Rotavirus (out of scope for Paper 1)
- **Reassortment detection:** Flag unusual segment combinations
- **Wastewater at scale:** With upstream demixing (e.g., Freyja)

> GenSLM genome-level representations deferred to Paper 2.

---

## 6. Planned Figures and Tables

| # | Type | Content | Priority |
|---|------|---------|----------|
| 1 | Figure | Pipeline schematic | Core |
| 2 | Table | Dataset statistics + pair-distribution ledger | Core |
| 3 | Figure | ROC curves: k-mer vs ESM-2 | Core |
| 4 | Table | Cross-validation results (mean ± std) | Core |
| 5 | Table | Temporal holdout results | Core |
| 6 | Figure | Subtype distribution shift 2021–23 → 2024 | Supporting |
| 7 | Fig/Table | Concat vs unit_diff on H3N2 | Low (if ESM-2 included) |
| 8 | Figure | PCA of interaction features | Low (if ESM-2 included) |
| 9 | Table | FP/FN metadata breakdown | Conditional |
| 10 | Fig/Table | **Stratified metrics by subtype, host, year** | **Core** |
| 11 | Table | Mitigation results | Conditional |
| 12 | **Figure** | **8×8 protein-pair AUC heatmap** — available: `allpairs_prod_20260408_063203/heatmap_auc_roc.png` | **Core** |
| 13 | Figure | Delayed learning curve (H3N2) | Low (Paper 2) |
| 14 | Fig/Table | Data remediation / wastewater inference | Deferred |
| 15 | Figure | **Reliability diagram + ECE** | **Required** |
| 16 | Fig/Table | **Ensemble uncertainty vs correctness** | **Required** |
| NEW | Table | **Pair-distribution ledger** | **Core** |
| NEW | Fig/Table | **Mixed-subtype discrimination results** | **Core** |

---

## 7. Experiment Status Tracker

Maps paper sections to `roadmap_v2.md` tasks.

| Section | Roadmap Task | Status | Priority | Blocking? |
|---------|-------------|--------|----------|-----------|
| 2.1.1 Pair-type analysis | New | Not implemented | **High (unconditional)** | Yes |
| 3.1 Baseline at scale | Task 2 | Needs re-run | **High** | Yes |
| 3.2 Cross-validation | Task 1 | Code ready | High | Yes |
| 3.3 Temporal holdout | Task 3 | Needs dedup fix | High | Yes |
| 3.4 Embedding geometry | Existing | Done | Low | No |
| 3.5.1 Stratified eval | New | Not implemented | **High (unconditional)** | Yes |
| 3.5.2–3 Error/diagnostics | Task 12 | Conditional | Conditional | Conditional |
| 3.5.4 Interventions | Task 12 + 4b | Conditional | Conditional | Conditional |
| 3.6 Mixed-subtype test | New (Carla) | Not started | **High** | No |
| 3.7 All protein pairs | Task 11 | **Complete (12-fold CV, full dataset; 2026-04-08)** | **High** | No |
| 4.2 UQ | New | Not started | **Required (last)** | Yes |
| 4. Use cases | Deferred | Not started | Deferred | Deferred |

**Minimum viable paper:** Sections 3.1 (at scale), 3.2 (CV), 3.5.1 (stratified eval),
3.7 (8×8 heatmap), 4.2 (UQ) + Discussion.

---

## 8. Target Venues

**Computational biology / bioinformatics (best fit):**
Bioinformatics (Oxford), PLOS Computational Biology, BMC Bioinformatics.

**Higher impact (if remediation/wastewater story is strong):**
Nature Methods, Genome Biology.

### Publication strategy

**Paper 1 (biology-focused, primary):** Segment matching for data remediation and genomic
surveillance. K-mers primary. Target: Bioinformatics, PLOS Comp Bio, Genome Biology.

**Paper 2 (ML-focused, follow-up):** ESM-2 embedding geometry + GenSLM comparison. Target:
NeurIPS/ICML CompBio workshop. Depends on `roadmap_v2.md` Task 10.

**Paper 3 (HPC, exploratory):** Scaling segment matching on leadership-class HPC. Target:
SC, PASC. Viability TBD — depends on whether Task 11 and scaling experiments produce
interesting computational findings.

Potential angles:
- **Data scaling:** Preliminary evidence exists. AUC improves 0.972→0.993 (5K→111K isolates)
  with 4x tighter variance; per-epoch time scales linearly (4.7s→97s). Two data points so
  far — need intermediate points (10K, 25K, 50K) to characterize the scaling curve shape
  (log-linear? saturating? power law?). A clear scaling law for biological sequence matching
  would be novel.
- **Task scaling:** 28 protein-pair experiments × 10 CV folds = 280 independent tasks.
  Task packing strategies (ensemble packing within nodes, job arrays across nodes), GPU
  utilization, scheduling overhead, queue strategy comparison (capacity vs preemptable vs prod).
- **Compute scaling:** Wall-clock vs nodes/GPUs at fixed problem size. Efficiency curves.
  Whether the embarrassingly parallel structure (no inter-task communication) achieves
  near-linear speedup in practice on Polaris, or whether I/O and scheduling overhead degrades it.
- **Model scaling (if explored):** MLP width/depth vs performance at fixed data size. Not
  yet systematically studied — could be added as an ablation axis if the HPC paper materializes.

To strengthen the HPC case, consider running Task 11 at multiple data scales (5K, 25K, 50K,
111K) rather than jumping straight to full dataset. This produces the scaling curves needed
for both Paper 1 (does more data help?) and Paper 3 (how does compute cost scale?).

---

## Appendix: Delayed Learning Phenomenon

On H3N2-only data with ESM-2 + unit_diff, training exhibits a characteristic
plateau-then-breakthrough pattern (~epochs 10–32, seed-dependent). The model shows near-random
performance for many epochs before suddenly learning. This is likely related to the difficulty of
extracting isolate-specific signal from homogeneous-population embeddings after the subspace offset
has been removed by unit_diff.

**Practical implication:** Increase `patience` to 40+ for H3N2-only or other single-subtype runs
to avoid early stopping during the plateau phase.
