# Paper Outline: Viral Segment Matching

**Working title:** Predicting Same-Isolate Origin of Genome Segments from Protein and Nucleotide Features

**Status:** Draft outline — circulate to team before committing to experiment scope.

---

## Abstract

_One paragraph. Update after results are finalized._

Problem: Segmented viruses like influenza have genomes split across multiple RNA segments.
Linking segments to the same isolate is essential for genomic surveillance, but public
databases often lack reliable metadata for this linkage. We frame segment matching as
binary classification: given two sequence representations, predict whether they originate
from the same viral isolate. We compare protein language model representations (ESM-2)
and k-mer frequency features with an MLP classifier, evaluating on cross-validation,
temporal holdout, and metadata-restricted settings (e.g., host, HN subtype, geographic
location). We demonstrate the model's utility for data remediation by applying it in
inference mode to unlinked BV-BRC records with calibrated uncertainty estimates.

---

## 1. Introduction

### Motivation

Segmented viruses (Influenza A, Bunyavirales) have genomes split across multiple RNA
segments. Each segment is often sequenced and deposited independently, making it
difficult to determine which segments originated from the same viral isolate. Accurate
linking of genome segments to the same isolate is critical for:

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

Given representations of two protein segments from a segmented virus, predict whether
they originate from the same isolate (binary classification). This work focuses on
Influenza A HA and NA segments, with extensions to other segment pairs.

### Contributions

- Frame viral segment-to-isolate linkage as a binary classification task using sequence
  representations (first to our knowledge)
- Compare protein language model (ESM-2) and k-mer feature representations
- Identify a geometry-specific failure mode of ESM-2 representations under concatenation
  on homogeneous populations, and show that unit-normalized differences resolve it
- Demonstrate temporal generalization (train on 2021–2023, test on 2024 flu season)
- Demonstrate data remediation application on BV-BRC records at scale with calibrated uncertainty estimates

---

## 2. Methods

### 2.1 Data

- **Source:** A set of ~111K Flu A isolates from BV-BRC (curated by Jim Davis). Each GTO
  contains genome and protein sequences, and various metadata.
- **Preprocessing:** Protein sequences extracted from GTO replicon functions and mapped
  to 8 major proteins (one primary gene product per segment): PB2, PB1, PA, HA, NP, NA,
  M1, NS1. Alternative reading frame products (M2, NEP) excluded. Nucleotide genome
  sequences extracted per segment for k-mer feature computation. Quality filters applied
  to both protein and genome sequences.
- **Pair construction:** Positive pairs = segments from the same isolate. Negative pairs =
  segments from different isolates, with co-occurrence blocking (no contradictory labels).
  Train is balanced (1:1 pos/neg); val/test reflect natural imbalance.

### 2.2 Feature representations

| Feature | Source | Dimensionality | Description |
|---------|--------|---------------|-------------|
| ESM-2 | `esm2_t33_650M_UR50D` | 1280 | Frozen protein language model mean-pool embeddings |
| K-mer (k=6) | Nucleotide sequences | 4096 | Sparse frequency vectors over 6-mer vocabulary |

### 2.3 Interaction approaches

Given embeddings A (slot 1) and B (slot 2), with schema-ordered assignment (HA→slot 1,
NA→slot 2):

| Interaction | Formula | Output dim | Properties |
|-------------|---------|-----------|------------|
| `concat` | [A, B] | 2D | Preserves both embeddings; order-sensitive |
| `unit_diff` | (A − B) / ‖A − B‖₂ | D | L2-normalized signed difference; retains direction only |

### 2.4 Slot transform architecture variants

The pipeline transforms per-slot embeddings before the interaction function. Five
`slot_transform` variants control this transformation:

```
(a) none — pass-through

  emb_a ─────────────────────────┐
                                 ├─→ Interaction ──→ MLP Classifier ──→ P(same isolate)
  emb_b ─────────────────────────┘


(b) shared — single shared MLP

  emb_a ──→ [ Shared MLP ] ──→ a' ─┐
                                    ├─→ Interaction ──→ MLP Classifier ──→ P(same isolate)
  emb_b ──→ [ Shared MLP ] ──→ b' ─┘
              (same weights)


(c) slot_specific — independent MLP per slot

  emb_a ──→ [  MLP_A  ] ──→ a' ─┐
                                 ├─→ Interaction ──→ MLP Classifier ──→ P(same isolate)
  emb_b ──→ [  MLP_B  ] ──→ b' ─┘
            (separate weights)


(d) shared_adapter — shared MLP + per-slot residual adapters

  emb_a ──→ [ Shared MLP ] ──→ z_a ──→ z_a + Adapter_A(z_a) ──→ a' ─┐
                                                                      ├─→ Interaction ──→ MLP ──→ P
  emb_b ──→ [ Shared MLP ] ──→ z_b ──→ z_b + Adapter_B(z_b) ──→ b' ─┘
              (same weights)           (separate adapter weights)


(e) slot_norm — optional shared MLP + per-slot LayerNorm  ** current best **

  emb_a ──→ [ Shared MLP ]? ──→ [ LayerNorm_A ] ──→ a' ─┐
                                                          ├─→ Interaction ──→ MLP ──→ P
  emb_b ──→ [ Shared MLP ]? ──→ [ LayerNorm_B ] ──→ b' ─┘
              (optional)        (separate LN params)
```

**Summary:**

| Mode | Slot A path | Slot B path | Key property |
|------|------------|------------|-------------|
| `none` | pass-through | pass-through | Raw embeddings; no learned transform |
| `shared` | Shared_MLP(a) | Shared_MLP(b) | Same projection for both slots |
| `slot_specific` | MLP_A(a) | MLP_B(b) | Independent per-slot projections |
| `shared_adapter` | Shared_MLP(a) + Adapter_A(·) | Shared_MLP(b) + Adapter_B(·) | Shared backbone + per-slot residual adapters |
| `slot_norm` | [Shared_MLP(a)→] LN_A(·) | [Shared_MLP(b)→] LN_B(·) | Per-slot normalization; neutralizes subspace offset |

`slot_norm` (without the optional shared MLP) is the current best configuration. Per-slot
LayerNorm is critical for ESM-2 on homogeneous subsets — it neutralizes the systematic
offset between HA and NA embedding subspaces, allowing `unit_diff` to capture genuine
biological signal rather than slot identity.

### 2.5 Classifier

MLP binary classifier on interaction features. Architecture: [interaction_dim] → hidden
layers → sigmoid. Trained with BCEWithLogitsLoss, early stopping on validation loss.

### 2.6 Evaluation metrics

F1, AUC-ROC, precision, recall, Brier score. Optimal threshold selected on validation set.

---

## 3. Experiments & Results

### 3.1 Baseline performance (random split)

_Roadmap Task: existing results_

- ESM-2 + unit_diff on mixed-subtype HA/NA: AUC ~0.966, F1 ~0.917
- K-mer + unit_diff on mixed-subtype HA/NA: AUC ~0.982, F1 ~0.957
- K-mer matches or exceeds ESM-2 across all configurations tested

### 3.2 Cross-validation (N=5 folds)

_Roadmap Task 1 — IMPLEMENTED, needs end-to-end run_

- Report mean ± std of AUC, F1, precision, recall across 5 folds
- Both ESM-2 and k-mer feature sources
- **Status:** Code ready, awaiting run

### 3.3 Temporal holdout (train 2021–2023, test 2024)

_Roadmap Task 3 — IMPLEMENTED, needs dedup fix and re-run_

- Assesses generalization to future flu seasons
- Notable subtype distribution shift: H5N1 24% → 41% in 2024 (avian flu surge)
- Preliminary results (with known dedup artifact):

| Metric | ESM-2 (unit_diff) | K-mer k=6 (unit_diff) |
|--------|-------------------|----------------------|
| AUC-ROC | 0.891 | **0.941** |
| F1 | 0.734 | **0.832** |

- Both show AUC drop vs random splits (~0.97), confirming genuine temporal difficulty
- K-mers generalize better than ESM-2 across flu seasons
- **Status:** Needs re-run with dedup fix for clean metrics

### 3.4 ESM-2 embedding geometry: concat failure on homogeneous populations

_Existing results — key mechanistic finding_

- ESM-2 concat completely fails on H3N2-only data (AUC=0.498, random chance)
- ESM-2 unit_diff succeeds on H3N2 (AUC=0.957)
- K-mer concat does NOT fail on H3N2 (AUC=0.985)
- **Explanation:** ESM-2 embeddings for HA and NA occupy distinct subspaces. On diverse
  data, concat exploits this offset as a shortcut. On homogeneous data (single subtype),
  the shortcut is uninformative. unit_diff strips magnitude, retaining only directional
  signal. K-mers lack the subspace structure entirely.
- LayerNorm (slot_norm) is critical: without it, ESM-2 unit_diff also fails on H3N2
- Delayed learning phenomenon on H3N2 + unit_diff (plateau-then-breakthrough, seed-dependent)

### 3.5 Error analysis and robustness

_Partially existing (FP/FN metadata available), analysis NOT IMPLEMENTED_

The model consistently shows high FP/FN ratio (e.g., 64:1 for ESM-2 temporal, 11.6:1 for
k-mer temporal). This means the model over-predicts same-isolate origin — it says "yes, these
segments belong together" too often. Understanding and addressing this is critical for
the remediation application, where a false link is worse than a missed link.

#### 3.5.1 Error diagnosis

Characterize *which* pairs the model gets wrong and *why*. Crucially, error analysis
must account for the **pair-level metadata profile**, not just individual segment metadata:

- **Positive pairs** (same isolate): both segments share subtype, host, year, geography.
  Errors stratified by the pair's subtype (e.g., FN rate on H1N1 vs H3N2 positives).
- **Negative pairs** (different isolates): each segment may come from a different population.
  Two distinct error regimes:
  - **Within-subtype negatives** (e.g., HA from H1N1 isolate A + NA from H1N1 isolate B):
    these are hard negatives — same population, different isolate. FPs likely concentrated here.
  - **Cross-subtype negatives** (e.g., HA from H1N1 + NA from H3N2): easy negatives —
    population-level differences make them trivially distinguishable.

Analysis approach:
- **Pair-level metadata matrix:** For each negative pair, record (subtype_a, subtype_b,
  host_a, host_b, year_a, year_b). Compute FP rate per (subtype_a × subtype_b) cell.
  Same for host and year axes.
- **FP concentration:** Confirm hypothesis that FPs are concentrated in within-subtype
  negatives. If so, the high FP/FN ratio is explained: most negatives in a mixed-subtype
  training set are cross-subtype (easy), inflating apparent precision, while the minority
  of within-subtype negatives drive the FP count.
- **FN analysis:** Are FNs enriched for underrepresented subtypes or unusual host/year
  combinations? If so, the model hasn't seen enough examples from those populations.

#### 3.5.2 Stress testing across metadata axes

Systematically evaluate generalization across population dimensions:

| Experiment | Train on | Test on | Tests |
|-----------|----------|---------|-------|
| Cross-subtype | H3N2 | H1N1 | Subtype generalization |
| Cross-geography | USA | China (or other region) | Geographic generalization |
| Cross-host | Human | Avian | Host generalization |
| Temporal | 2021–2023 | 2024 | Already in Section 3.3 |

For each axis, report how performance degrades and whether the FP/FN ratio shifts.
This tells users: "the model is reliable for X populations but should be used with
caution for Y populations."

#### 3.5.3 Mitigation strategies

**Experimental design:** Fix the val and test sets from the initial run. Modify only the
training set composition, retrain, and evaluate on the same held-out data. This isolates
the effect of training data changes from test set variation.

Mitigation covers both data-centric and model-centric approaches, applied in order of
increasing complexity. See `_roadmap.md` Task 12 for the full detailed plan, including
diagnostics (embedding distance distributions, predicted probability histograms,
pair-level metadata matrix), data-centric fixes (hard negative mining, negative ratio,
curriculum learning, subtype balancing), model-centric fixes (focal loss, weighted BCE,
contrastive learning, two-stage model), and recommended execution order.

**Summary of approaches (see `_roadmap.md` Task 12 for detail):**

- **Data-centric:** Hard negative mining (highest priority), increased negative-to-positive
  ratio, curriculum learning on negatives, subtype-balanced training, population-specific
  models, per-population threshold tuning.
- **Model-centric:** Focal loss (one-line change, try early), weighted BCE, contrastive
  learning (heavier intervention, try if simpler approaches fail), two-stage model.

**Iterative refinement loop:**
1. Train on initial dataset → evaluate on fixed test set → error diagnosis (3.5.1)
2. Identify dominant failure mode (e.g., high FP on within-subtype H1N1 negatives)
3. Apply mitigation (data-centric first, then model-centric) → retrain on modified data
4. Re-evaluate on same fixed test set → compare metrics → iterate

**Connection to applications:** For data remediation, these results translate directly
into usage guidelines — e.g., "apply the general model with confidence for H3N2 pairs;
for H5N1 pairs, use the H5N1-specific model or apply a stricter confidence threshold."

- **Status:** FP/FN metadata files exist from training runs. Analysis scripts not
  implemented. Stress testing requires new bundles + runs. Mitigation experiments
  require dataset modifications + fixed test set infrastructure.

### 3.6 All protein-pair combinations

_Roadmap Task 11 — NOT IMPLEMENTED_

- 28 pairwise combinations of 8 major Flu A proteins (C(8,2)): PB2, PB1, PA, HA, NP,
  NA, M1, NS1 — one primary gene product per genome segment. Alternative reading frame
  products (M2, NEP) are excluded.
- Results presented as 8×8 AUC/F1 heatmap
- Shows which segment pairs are easiest/hardest to match; whether model generalizes
  beyond HA/NA
- **Status:** Not started. HPC (Polaris) job array needed.

---

## 4. Application: Data Remediation at Scale

_New experiment — inference on BV-BRC_

### Concept

Train the model on the curated 111K-isolate dataset (where segment-isolate links are
known), then apply in inference mode to the broader BV-BRC Influenza A collection where
segment-isolate metadata is missing or unreliable.

### Approach

1. Train on curated dataset (best model from Section 3 experiments)
2. Identify BV-BRC records lacking reliable isolate linkage (records beyond Jim's curated
   111K set — need to quantify how many such records exist)
3. For candidate segment pairs, compute features and predict same-isolate probability
4. Evaluate: compare model predictions against known links (where available) as ground
   truth; report precision/recall on the remediation task

### 4.1 Large-scale inference

Running pairwise inference at BV-BRC scale (potentially millions of segment records)
requires an efficient inference pipeline:

- **Batch inference script:** Load trained model, iterate over candidate pairs, output
  predicted probabilities. Builds on Roadmap Task 8 (inference mode).
- **Candidate pruning:** Full pairwise is O(N²). Prune by metadata overlap (same year,
  same geographic region, same host) to reduce to tractable candidate sets.
- **HPC execution:** Polaris for large-scale inference. Embarrassingly parallel — partition
  candidate pairs across nodes.
- **Output:** Per-pair CSV with (segment_a_id, segment_b_id, predicted_prob, threshold_decision).

### 4.2 Uncertainty quantification (UQ)

For data remediation to be actionable, predictions need calibrated confidence estimates.
Approaches to consider:

- **MC Dropout:** Enable dropout at inference time, run N forward passes per pair, report
  mean prediction and predictive variance. Lightweight to implement (add dropout layers
  if not already present; toggle `model.train()` at inference).
- **Deep ensemble:** Train M independent models (different seeds), average predictions.
  Natural fit with cross-validation — the N=5 fold models from Section 3.2 are already
  an ensemble. Report mean ± std across ensemble members.
- **Calibration analysis:** Reliability diagram (predicted probability vs observed frequency).
  If predictions are well-calibrated, the raw sigmoid output is itself a usable confidence
  score. Report Expected Calibration Error (ECE).
- **Conformal prediction:** Distribution-free coverage guarantees. Given a calibration set,
  produce prediction sets (e.g., "same-isolate origin with 95% coverage"). Attractive for a
  remediation tool where users need to know "how much can I trust this link?"

**Recommended approach for the paper:** Deep ensemble (from CV folds) + calibration analysis.
This requires no architectural changes — just run inference with each fold's model and
aggregate. Add MC Dropout or conformal prediction if time permits.

### Key questions to resolve

- How many unlinked/poorly-linked records exist in BV-BRC Flu A beyond the curated set?
- What is the computational cost of pairwise inference at full BV-BRC scale? (N segments
  → O(N²) pairs per protein type, but can be pruned by metadata overlap)
- Can we validate remediation predictions against an independent source (e.g., GISAID)?

### Status

Not implemented. Requires: (1) quantifying the unlinked record population in BV-BRC,
(2) inference script (Roadmap Task 8 lays groundwork), (3) evaluation framework,
(4) UQ implementation.

### TODO

- [ ] **Quantify unlinked BV-BRC records.** Ask Jim or query BV-BRC directly: how many
  Influenza A segment records exist in total vs. the 111K curated isolates? What fraction
  lack reliable isolate linkage? This determines the scale and impact of the remediation
  demo. Without this number, we cannot scope Section 4.
- [ ] **Inference script.** Add inference-only mode to training script or create standalone
  script. Load trained model + threshold, predict on arbitrary pair dataset.
- [ ] **UQ implementation.** Start with deep ensemble from CV fold models. Add calibration
  plot (reliability diagram + ECE) to analysis scripts.

---

## 5. Discussion

### Key findings

- K-mer frequency features match or exceed ESM-2 protein embeddings on all tested
  configurations, including temporal holdout. Simpler features, lower compute, better
  generalization.
- ESM-2 concat failure on homogeneous populations reveals a fundamental property of
  protein language model embedding geometry: protein types occupy distinct subspaces,
  creating learnable shortcuts that fail to generalize.
- unit_diff (L2-normalized signed difference) extracts directional signal robust to
  population homogeneity. The direction of embedding difference carries genuine
  biological signal (coordinated mutations between co-evolving segments).
- Temporal generalization is harder than random splits (AUC drops from ~0.97 to ~0.89–0.94),
  reflecting real distributional shift in circulating subtypes.

### Limitations

- High FP/FN ratio indicates the model over-predicts same-isolate origin; Section 3.5
  characterizes this and proposes mitigations, but it remains a deployment concern
- Currently tested on HA/NA pairs only (Section 3.6 addresses this)
- Negative pairs are constructed by cross-isolate mixing; may be "easy" due to
  subtype/host/year confounders in mixed-population training (Section 3.5.3 tests this)
- Data remediation application demonstrated but not rigorously validated against
  independent ground truth

### Future directions

- **Reassortment detection:** The model learns normal co-occurrence patterns. Segments
  predicted as non-co-occurring despite being found together could flag reassortment
  events. Requires known reassortant sequences for validation.
- **Extension to other segmented viruses** (Bunyavirales, Rotavirus)
- **Wastewater application:** Paired with metagenomic assembly, apply to real wastewater
  sequencing data
- **GenSLM genome-level representations** as alternative to protein-level features

---

## 6. Planned Figures and Tables

| # | Type | Content | Source |
|---|------|---------|--------|
| 1 | Figure | Pipeline schematic (preprocessing → embeddings → pairs → classifier) | New |
| 2 | Table | Dataset statistics (isolates, pairs, subtypes, years) | Stage 3 stats |
| 3 | Figure | ROC curves: ESM-2 vs k-mer, random split | Existing |
| 4 | Table | Cross-validation results (mean ± std, N=5) | Task 1 (pending) |
| 5 | Table | Temporal holdout results (train 2021–23, test 2024) | Task 3 (pending re-run) |
| 6 | Figure | Subtype distribution shift 2021–2023 → 2024 | Temporal analysis |
| 7 | Figure/Table | Concat vs unit_diff on H3N2: ESM-2 fails, k-mer succeeds | Existing |
| 8 | Figure | PCA of pair interaction features (concat vs unit_diff) | Existing plots |
| 9 | Table | FP/FN metadata breakdown (subtype, host, year overlap) | Error analysis (pending) |
| 10 | Figure/Table | Stratified metrics by subtype, host, geography | Stress testing (pending) |
| 11 | Table | Mitigation results (balanced training, population-specific models) | Mitigation (pending) |
| 12 | Figure | 8×8 protein-pair AUC heatmap | Task 11 (pending) |
| 13 | Figure | Delayed learning curve (H3N2 + unit_diff plateau-breakthrough) | Existing |
| 14 | Table/Figure | Data remediation inference results on BV-BRC | New (pending) |
| 15 | Figure | Reliability diagram (calibration) + ECE | UQ analysis (pending) |
| 16 | Figure/Table | Ensemble prediction uncertainty vs correctness | UQ analysis (pending) |

---

## 7. Experiment Status Tracker

Maps paper sections to roadmap tasks and their readiness.

| Section | Roadmap Task | Status | Blocking? |
|---------|-------------|--------|-----------|
| 3.1 Baseline | Existing | Done | No |
| 3.2 Cross-validation | Task 1 | Code ready, needs run | Yes |
| 3.3 Temporal holdout | Task 3 | Needs dedup fix + re-run | Yes |
| 3.4 Embedding geometry | Existing | Done | No |
| 3.5 Error analysis | Task 12 | Diagnostics not implemented | Yes |
| 3.5.2 Stress testing | Task 12 | Needs new bundles + runs | Yes |
| 3.5.3 Mitigation | Task 12 | Data-centric + model-centric; see `_roadmap.md` | Yes |
| 3.6 All protein pairs | Task 11 | Not started (HPC) | No (strengthens paper) |
| 4. Data remediation | New + Task 8 | Not started | Yes |
| 4.2 UQ | New | Not started | No (enhances Section 4) |

**Minimum viable paper:** Sections 3.1, 3.2, 3.3, 3.4 + Discussion. Sections 3.5 and 4
strengthen the paper substantially but could be deferred to a follow-up if timeline is tight.

---

## 8. Target Venues

Discuss with team — the venue determines how much weight goes to the application demo
vs. the ML analysis.

**Computational biology / bioinformatics (best fit):**

| Venue | Notes |
|-------|-------|
| **Bioinformatics** (Oxford) | Short applications papers welcome; strong fit for a tool/method with biological application |
| **PLOS Computational Biology** | Open access; good for interdisciplinary ML + bio work |
| **BMC Bioinformatics** | Open access; accepts methods papers with biological validation |

**Higher impact (if remediation/wastewater story is strong):**

| Venue | Notes |
|-------|-------|
| **Nature Methods** | Requires real-world remediation at scale (Section 4) |
| **Genome Biology** | Open access; high visibility in genomics community |

**ML-leaning (if ESM-2 geometry finding is the main story):**

| Venue | Notes |
|-------|-------|
| **Bioinformatics** | Also works for ML-focused papers |
| **NeurIPS/ICML CompBio workshop** | Shorter format; good for the ESM-2 concat failure finding specifically |

### Publication strategy (decided)

**Paper 1 (biology-focused, primary):** Segment matching as a tool for data remediation
and genomic surveillance. Target: biology/bioinformatics venue (Bioinformatics, PLOS Comp
Bio, Genome Biology). Teammates provide biological framing and domain insight. Emphasize
applications (Sections 1, 4), temporal generalization (3.3), and all-protein-pair results
(3.5). The concat collapse and embedding geometry findings are supporting evidence, not
the main story.

**Paper 2 (ML-focused, follow-up):** ESM-2 embedding geometry — concat collapse on
homogeneous populations, interaction function design, and comparison with GenSLM
genome-level representations. Target: NeurIPS/ICML CompBio workshop or similar ML venue.
Deeper dive into why protein language model embeddings fail under concatenation, the role
of subspace offset, and whether genome-level foundation models (GenSLM) exhibit the same
geometry. This paper depends on Roadmap Task 10 (GenSLM embeddings).
