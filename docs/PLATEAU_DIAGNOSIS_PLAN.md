# Segment Pair Prediction: Plateau Diagnosis Plan

**Date**: December 2, 2025  
**Status**: In Progress  
**Primary Concern**: Immediate validation plateau in Flu A model training  
**Related Docs**: [EMBEDDINGS_BRANCH_CHANGES.md](./EMBEDDINGS_BRANCH_CHANGES.md)

---

## TODO Status (Updated: December 2, 2025)

### Completed Tasks
| Task | Status | Notes |
|------|--------|-------|
| Train model on Bunya | ✅ Complete | Val F1: 0.75→0.91, Test AUC: 0.956 |
| Analyze Bunya training results | ✅ Complete | 100% same-func negative accuracy |
| Run embedding similarity on Bunya | ✅ Complete | Results in embedding_similarity/ |
| Re-run Bunya with allow_same_func_negatives=false | ✅ Complete | Still achieves 90.6% accuracy, 0.936 AUC |
| Document embeddings branch changes | ✅ Complete | See docs/EMBEDDINGS_BRANCH_CHANGES.md |

### Pending Tasks
| Task | Priority | Notes |
|------|----------|-------|
| Run embedding similarity on Flu A | P1 | Critical diagnostic |
| Compare Bunya vs Flu A similarity distributions | P1 | Quantify conservation impact |
| Segment-specific models for Flu A | P2 | HA-NA vs PB1-PB2 comparison |
| Try use_diff=True, use_prod=True on Flu A | P2 | If diagnostics show promise |
| Contrastive fine-tuning of ESM-2 | P3 | If current approach insufficient |
| Explore genome foundation models | P4 | GenSLM, Evo2 for nucleotide signal |
| Decide on dataset run directory structure | P5 | Currently using runs/ subdirs |

---

## 1. Problem Summary

### What We Observed in Flu A Training

| Observation | Value | Implication |
|-------------|-------|-------------|
| Test F1/AUC | ~84% | Better than random (50%) |
| Validation plateau | After epoch 1 | Model stops learning immediately |
| Training continues | Loss decreases | Model memorizes training data |

**Key insight**: The model learns *something* quickly, then stops improving. This suggests either:
- The model quickly exhausts all learnable signal
- There's fundamental biological limitation
- Data leakage gives artificially good initial performance (partially addressed)

### Biological Reality: Why Flu A May Be Hard

#### Influenza Protein Conservation (Scientific Evidence)

Studies show influenza internal proteins are highly conserved ([PMC3036627](https://pmc.ncbi.nlm.nih.gov/articles/PMC3036627/)):

| Segment | Protein | Conservation (Human Strains) |
|---------|---------|------------------------------|
| 2 | PB1 | **98.1%** (highest) |
| 1 | PB2 | ~95% |
| 5 | NP | ~95% |
| 3 | PA | ~94% |
| 4 | HA | 70-85% (immune pressure) |
| 6 | NA | 80-90% (immune pressure) |

**Key insight**: Internal proteins (polymerase complex) are highly conserved; surface proteins (HA/NA) have more variation due to antigenic drift.

#### Why ESM-2 Struggles with This Task

1. **ESM-2 was trained on masked language modeling (MLM)** — predicting amino acids from context. This captures:
   - Protein structure and function
   - Evolutionary relationships
   - Amino acid properties

2. **ESM-2 was NOT trained to distinguish isolate origin**. Two HA proteins from different isolates produce nearly identical embeddings because they're functionally equivalent.

3. **The information-theoretic limit**: If sequences are >98% identical, and ESM-2 generalizes over evolutionary variation, the embeddings will be nearly indistinguishable. No downstream model can recover signal that isn't in the representation.

**In simpler terms**: If I hand you two PB1 protein embeddings and ask "are these from the same isolate?" — if all PB1 proteins are 98% identical and map to essentially the same embedding, the task is impossible regardless of model architecture.

### Data Science Issues (Status)

| Issue | Status | Notes |
|-------|--------|-------|
| Contradictory labels (duplicate sequences) | ✅ Addressed | Blocked negatives implemented |
| Data leakage (same pairs in train/test) | ✅ Addressed | Pair-key validation added |
| Near-identical non-duplicate sequences | ❌ Not addressed | May be the core issue |
| Low effective diversity | ❓ Unknown | Need to quantify |

---

## 2. The Critical Diagnostic

**The key question we need to answer:**

```
If cosine_similarity(positive_pairs) ≈ cosine_similarity(negative_pairs)
Then → ESM-2 embeddings fundamentally cannot solve this task
```

### What This Means

- **Positive pairs**: Two proteins from the SAME isolate
- **Negative pairs**: Two proteins from DIFFERENT isolates
- **If distributions overlap heavily**: No amount of model training will help - the signal simply isn't there in the embeddings

---

## 3. Diagnostic Plan

### Phase 1: Technical Validation

| Step | Task | Status |
|------|------|--------|
| 1a | Fix Bunya embedding loading | ✅ Done |
| 1b | Run analyze_stage3_embeddings.py on Bunya | ✅ Done |
| 1c | Train model on Bunya | ✅ Done |
| 1d | Analyze Bunya model results | ✅ Done |
| 1e | Run analyze_stage3_embeddings.py on Flu A | ⬜ Pending |
| 1f | Create pair similarity diagnostic script | ⬜ Pending |

### Phase 2: Pair Similarity Analysis (PRIORITY)

Create a focused diagnostic script that:
1. Loads train_pairs.csv (has positive/negative labels)
2. Loads embeddings for both proteins in each pair
3. Computes cosine similarity for each pair
4. Plots positive vs negative distributions side-by-side
5. Computes separation metrics (KL divergence, overlap %)

### Phase 3: Comparative Analysis

Run for both Bunya and Flu A:

| Diagnostic | What It Tells Us |
|------------|------------------|
| Positive vs negative similarity distributions | Can embeddings distinguish at all? |
| Within-function similarity | How conserved is each protein? |
| Cross-function similarity | Is there protein-specific signal? |
| Sequence clustering at 95%/99% identity | What's the actual diversity? |

### Phase 4: Decision Point

Based on diagnostics:

| Finding | Conclusion | Next Action |
|---------|------------|-------------|
| Distributions separate well for Bunya, not Flu | Biology limits Flu, not our approach | Try different features for Flu |
| Distributions overlap for both | ESM-2 embeddings wrong choice | Try different embeddings |
| Distributions separate for both | Bug in our pipeline | Debug training |

---

## 4. KEY FINDING: Bunya vs Flu A Comparison

### Training Dynamics Comparison

| Metric | Bunya | Flu A |
|--------|-------|-------|
| Val F1 (epoch 1) | 0.754 | 0.74 |
| Val F1 (best) | **0.910** (epoch 13) | 0.74 (epoch 1) |
| Val F1 improvement | **+0.156** | 0 (plateau) |
| Learning continues? | ✅ Yes (13 epochs) | ❌ No |
| Test F1 | 0.859 | ~0.84 |
| Test AUC | **0.956** | ~0.84 |

### Bunya Model Performance (December 1, 2025)

**Training run**: `data/models/bunya/April_2025/runs/training_bunya_20251201_111543`

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.922 |
| Test F1 | 0.859 |
| Test AUC | 0.956 |
| Precision (positive) | 0.79 |
| Recall (positive) | 0.95 |

**Segment Pair Performance**:
| Segment Pair | F1 | AUC | Accuracy |
|--------------|-----|-----|----------|
| L-M | 0.876 | 0.947 | 0.912 |
| L-S | 0.865 | 0.945 | 0.903 |
| M-S | 0.836 | 0.927 | 0.872 |

**Same-Function Negative Pairs**: 100% accuracy (all 240 correctly classified)  
**Different-Function Negative Pairs**: 87.2% accuracy

### Conclusion from Comparison

**✅ The pipeline works.** Bunya demonstrates:
1. Continuous validation improvement over 13 epochs
2. High test AUC (0.956)
3. Perfect discrimination for same-function negative pairs (100%)

**❌ The Flu A plateau is biology-specific**, not a pipeline bug. The problem is likely:
- Higher sequence conservation in Flu A proteins
- ESM-2 embeddings capturing function, not isolate-specific variation

---

## 5. Experimental Ideas (If Diagnostics Show Promise)

### 5.1 Quick Win: Interaction Features

**Hypothesis**: Raw concatenation loses relational information

**Experiment**: Run with `use_diff=True` and `use_prod=True`

```yaml
# In flu_a_plateau_analysis.yaml
training:
  use_diff: true   # |emb_a - emb_b| - captures differences
  use_prod: true   # emb_a * emb_b - captures interactions
```

**Why this might help**:
- `use_diff`: Highlights WHERE embeddings differ (signal for different origin)
- `use_prod`: Captures interaction patterns (multiplication amplifies similar dimensions)

**Note**: Bunya already uses `use_diff=True, use_prod=True` and achieves good results.

**Command** (doesn't require regenerating dataset):
```bash
./scripts/run_training.sh flu_a_plateau_analysis --cuda_name cuda:7 \
    --dataset_dir <EXISTING_DATASET_DIR> --skip_postprocessing
```

### 5.2 Cosine Similarity as Explicit Feature

**Hypothesis**: A single scalar (cosine sim) might be more informative than MLP on concatenated embeddings

```python
# Add cosine similarity as an explicit input feature
cos_sim = F.cosine_similarity(emb_a, emb_b)
features = torch.cat([emb_a, emb_b, cos_sim.unsqueeze(-1)], dim=-1)
```

### 5.3 Alternative Task Formulations

**Contrastive Learning Approach**:
Instead of binary classification, use metric learning:
- Pull together proteins from same isolate
- Push apart proteins from different isolates

**Siamese Network with Distance Learning**:
Learn an embedding space where same-isolate pairs are close.

---

## 6. Baseline Experiments

### 6.1 Random Label Baseline
**Goal**: Confirm model isn't learning spurious patterns

```python
# Shuffle labels randomly
train_pairs['label'] = np.random.permutation(train_pairs['label'])
```
**Expected**: F1 ~0.50 (random chance). If higher, something is wrong.

### 6.2 Logistic Regression on Cosine Similarity
**Goal**: Check if a simpler model can learn the signal

```python
from sklearn.linear_model import LogisticRegression
# Use cosine similarity as single feature
cos_sim = cosine_similarity(emb_a, emb_b)
clf = LogisticRegression().fit(cos_sim.reshape(-1, 1), labels)
```

**Interpretation**:
- If logistic regression ~= MLP: Signal is simple, complex model unnecessary
- If logistic regression << MLP: MLP is learning something beyond raw similarity
- If both ~84%: That's the ceiling from embeddings

### 6.3 Same-Protein Baseline
**Goal**: Is the task easier for specific protein pairs?

Subset to only PB1-PB2 pairs and check if signal is stronger. Some protein combinations may have more isolate-specific variation.

---

## 7. Error Analysis (Post-Training)

### 7.1 Confusion Matrix Analysis
- Which pairs are misclassified?
- Are certain protein functions harder?
- Is there a pattern (e.g., all HA-NA pairs fail)?

### 7.2 Per-Pair Analysis
- For incorrectly classified pairs, what's their sequence similarity?
- Are borderline cases (prediction ~0.5) biologically meaningful?
- Do false positives come from highly similar isolates?

---

## 8. Expected Outputs

### Quantitative Metrics (per virus)

| Metric | Description |
|--------|-------------|
| Mean positive similarity | Average cosine similarity for same-isolate pairs |
| Mean negative similarity | Average cosine similarity for different-isolate pairs |
| Similarity AUC | Using raw similarity as classifier (upper bound) |
| Distribution overlap % | Proportion of overlap between distributions |
| Within-function variance | How similar are proteins of same function? |

### Visualizations

1. **Similarity distribution plot**: Overlaid histograms for positive/negative pairs
2. **ROC curve**: Using cosine similarity as classifier
3. **Function-specific heatmaps**: Similarity within/across protein functions
4. **Sequence diversity plots**: Cluster counts at different identity thresholds

---

## 9. Honest Assessment

### What 84% F1 in Flu A Might Mean

1. **Best case**: There IS signal, but model architecture limits learning → Try better architectures
2. **Worst case**: 84% IS the ceiling determined by embedding overlap → Need different approach
3. **Middle case**: Some protein functions have signal, others don't → Stratified analysis needed

### Why Bunya Succeeds

Based on the December 1, 2025 analysis:
- Bunya has only 3 segments (vs Flu's 8)
- Likely higher sequence diversity between isolates
- ESM-2 embeddings capture enough isolate-specific variation
- Same approach achieves 0.956 AUC

### Don't Fool Ourselves

If the pair similarity diagnostic shows positive/negative distributions heavily overlap for Flu A (but not Bunya), we should accept that ESM-2 concatenation cannot solve segment-pairing for Flu A specifically.

### But High Overlap Doesn't Mean "Give Up"

The embedding similarity analysis tells us about RAW embedding space, not what's recoverable through:

1. **Non-linear transformations**: MLPs can learn complex decision boundaries
2. **Interaction features**: `|emb_a - emb_b|` and `emb_a * emb_b` can highlight differences
3. **Contrastive fine-tuning**: Reshape the embedding space for our task
4. **Genome-level models**: Nucleotide sequences may contain more signal

### Research Roadmap (Ranked by Expected Impact)

| Rank | Approach | Expected Impact | Effort |
|------|----------|-----------------|--------|
| 1 | **Segment-specific models** | High | Low |
| 2 | **Contrastive fine-tuning ESM-2** | High | Medium |
| 3 | **Interaction features (diff, prod)** | Medium | Low |
| 4 | **Genome foundation models (GenSLM, Evo2)** | Medium-High | High |
| 5 | **Cross-attention architecture** | Medium | Medium |

### Segment-Specific Model Hypothesis

**Core idea**: Train separate models for different segment pairs to test the conservation hypothesis.

**Experiments**:
- **Model A**: HA-NA pairs only (high variability → expect high accuracy)
- **Model B**: PB1-PB2-PA pairs only (high conservation → expect lower accuracy)
- **Model C**: All pairs (current approach)

**Expected outcome**: If HA-NA model significantly outperforms polymerase model, this confirms conservation as the primary limiting factor.

**Why this matters**: It would show we can accurately predict segment pairs for less-conserved proteins, even if conserved proteins remain challenging.

---

## 10. Current Results

### Bunya (Completed - December 1, 2025)

**Embedding Analysis**:
- **Location**: `results/bunya/April_2025/embeddings_analysis/`
- **Embedding coverage**: 100% (2963/2963 proteins)

**Model Training**:
- **Location**: `data/models/bunya/April_2025/runs/training_bunya_20251201_111543/`
- **Val F1**: 0.754 → 0.910 (continuous improvement over 13 epochs)
- **Test F1**: 0.859, **Test AUC**: 0.956

**Model Analysis**:
- **Location**: `results/bunya/April_2025/bunya/`
- **Key finding**: 100% accuracy on same-function negative pairs
- **Generated plots**: confusion_matrix.png, roc_curve.png, precision_recall_curve.png, etc.

### Flu A (Partially Complete)

**Embedding Analysis**:
- **Location**: `results/flu_a/July_2025/embeddings_analysis/`
- **Status**: Not yet run

**Model Training**:
- **Previous runs**: Val F1 plateaus at 0.74 after epoch 1
- **Test F1**: ~0.84, **Test AUC**: ~0.84

---

## 11. Next Steps (Updated)

| Priority | Task | Status |
|----------|------|--------|
| 1 | Create pair similarity diagnostic script | ⬜ Pending |
| 2 | Run pair similarity on Bunya | ⬜ Pending |
| 3 | Run pair similarity on Flu A | ⬜ Pending |
| 4 | Compare distributions | ⬜ Pending |
| 5 | If overlap differs → Flu is biology-limited | Decision point |
| 6 | Run Flu A with use_diff=True, use_prod=True | ⬜ Pending |
| 7 | Baseline experiments if needed | ⬜ Pending |

---

## 12. Success Criteria

- **Short-term**: ✅ Achieved for Bunya (Val F1 improves beyond 0.74)
- **Medium-term**: Quantify the theoretical ceiling for Flu A based on embedding similarity
- **Long-term**: Establish robust pipeline for viral segment matching (or document biological limitations for Flu A)

---

## 13. Files Referenced

| File | Purpose |
|------|---------|
| `src/analysis/analyze_stage3_embeddings.py` | Individual embedding analysis |
| `src/analysis/analyze_stage4_model_results.py` | Model results analysis |
| `src/datasets/dataset_segment_pairs.py` | Dataset creation with duplicate handling |
| `conf/bundles/bunya.yaml` | Bunya config (use_diff=True, use_prod=True) |
| `conf/bundles/flu_a_plateau_analysis.yaml` | Flu A plateau investigation config |
| `docs/DUPLICATE_SEQUENCE_HANDLING.md` | Duplicate problem documentation |

---

## Appendix: Key Commands

```bash
# Run embedding analysis
python src/analysis/analyze_stage3_embeddings.py --virus bunya --data_version April_2025
python src/analysis/analyze_stage3_embeddings.py --virus flu_a --data_version July_2025

# Train model
./scripts/run_training.sh bunya --cuda_name cuda:7 --dataset_dir data/datasets/bunya/April_2025
./scripts/run_training.sh flu_a_plateau_analysis --cuda_name cuda:7 --dataset_dir <path_to_dataset>

# Analyze model results
python src/analysis/analyze_stage4_model_results.py --config_bundle bunya --model_dir <model_dir> --create_performance_plots
python src/analysis/analyze_stage4_model_results.py --config_bundle flu_a_plateau_analysis --model_dir <model_dir> --create_performance_plots
```
