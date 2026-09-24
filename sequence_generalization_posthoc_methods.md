# Post-hoc assessment of sequence-model generalization with fixed train/test splits

This guide covers nucleotide and protein prediction tasks. All eight methods preserve the existing train/test assignments and can be applied without retraining the original model. Some require only sequences; others require labels and saved predictions.

The goal is to establish where predictive performance holds up, not to assign a percentage of the model's behavior to memorization. Generalization is relative to the intended use: predicting close relatives and predicting distant sequences are different evaluation goals.

A useful distinction:

- Method: what you do, such as a 1-nearest-neighbor (1-NN) baseline.
- Similarity measure: how you compare examples, such as alignment percent identity with a coverage requirement.
- Tool: software that supports the analysis, such as BLASTP, BLASTN, or MMseqs2.

The ease and robustness assessments below are practical judgments. “Common” refers to published use of the approach, not merely availability in a software package.

## 1. Sequence overlap and nearest-training-similarity audit

Method and what it measures

Two complementary dataset-only checks:

Exact duplicate detection: compare sequence strings between train and test, for example using a Python set or dictionary.

Nearest-neighbor similarity audit: search each test sequence against the training sequences and record its highest detected similarity under a specified alignment and coverage rule.

Concrete tools include BLASTP for proteins, BLASTN for nucleotide sequences, and MMseqs2 for large searches. These perform the sequence comparisons; they do not themselves establish whether the prediction model memorizes. [9, 10]

A useful output has one row per test sequence: test ID, exact-match flag, closest training ID, percent identity, and alignment coverage. Plot a histogram or empirical cumulative distribution function (ECDF) of closest-training identity. An ECDF shows the fraction at or below each observed identity. Keep no-hit cases as a separate category rather than assigning them an arbitrary identity.

Ease of applying

Easy for exact matches; moderate for similarity searches. The main cost is comparing sequences. Apply only sequence normalization justified by the task; do not silently replace ambiguous bases or residues.

How robust is it?

Strong for describing overlap, not model behavior. Percent identity must be interpreted with alignment coverage. Search tools can miss matches: “no hit detected” is not zero similarity. Report search settings and the definition of identity. Do not assume that different cluster IDs guarantee low train–test similarity.

Common across papers?

Yes. Train–test sequence-similarity assessment is established in biological benchmarking; GraphPart and SpanSeq discuss its importance. [1, 2]

Recently introduced?

No. The general practice predates those papers and tools.

## 2. Similarity-stratified performance evaluation

Method and what it measures

Also called performance by sequence-identity bin or performance versus nearest-training similarity.

Join the results of method 1 with the test labels and model predictions. Group examples by closest-training identity, then calculate performance separately in each group. This measures where the existing model performs well within the available similarity range.

Concrete implementation options:

pandas cut to assign specified identity ranges.

scikit-learn roc_auc_score or average_precision_score for suitable classification tasks.

mean_absolute_error for regression.

Matplotlib for performance versus identity, including group counts and uncertainty intervals. [11]

Use task-appropriate ranges and keep exact duplicates identifiable. Show actual identity ranges; a “lowest-similarity” group could still contain only very close relatives. Average precision is a particular precision–recall summary and should be named explicitly.

Ease of applying

Easy once method 1 is complete. Saved predictions are sufficient; additional model inference is unnecessary.

How robust is it?

One of the most informative analyses here. Report sample counts and class proportions, or regression-target distributions, for every group. Differences in difficulty or composition can explain performance changes. Binary ROC-AUC is undefined in a group containing only one class.

Common across papers?

Yes. DeepFRI and CPEC provide direct examples of evaluation by similarity to training sequences. [3, 4]

Recently introduced?

No. Those examples appeared in 2021 and 2024; the evaluation principle is older.

## 3. 1-NN or k-NN similarity-based prediction baseline

Method and what it measures

1-nearest-neighbor (1-NN) predicts using the single closest training example:

Classification: copy its label.

Regression: copy its measured target value.

For example, if the closest training sequence has label 1, the 1-NN prediction is 1. A k-NN extension uses several neighbors: if four of five neighbors are positive, an unweighted binary class-fraction score is 0.8.

For sequence data, this is also called similarity-based label transfer or, when supported by biological relatedness, homology-based annotation transfer. Transfer labels directly from a BLAST/MMseqs2 hit table. Alternatively, scikit-learn's KNeighborsClassifier and KNeighborsRegressor accept prepared numerical representations or precomputed distances; they do not align raw sequence strings. [12]

Compare the baseline with your model on identical test examples, both overall and within method 2's groups.

Ease of applying

Moderate. Reuse method 1's search results. Fix the ranking rule, coverage requirements, tie handling, and no-hit fallback without tuning on test outcomes. Paired inputs need an explicitly defined pair comparison.

How robust is it?

Strong complementary evidence about improvement over simple label transfer. A poorly configured baseline weakens the comparison. A 1-NN classifier produces hard labels; k-NN class fractions provide a more informative baseline for probability-ranking metrics. Neither equality nor superiority establishes the model's internal mechanism.

Common across papers?

Yes—widely used. DeepGO and CAFA evaluations include sequence-similarity baselines, although their precise label-transfer rules differ. [5, 6]

Recently introduced?

No. Nearest-neighbor prediction and biological annotation transfer are long established.

## 4. Equal-cluster-weight performance

Method and what it measures

Use inverse cluster-size weighting to check whether large groups of similar test sequences dominate the result.

Obtain sequence groups from justified annotations or a tool such as MMseqs2 clustering. Give each test example weight 1 / n_c, where n_c is the number of test examples in its cluster. Each cluster then has total weight 1.

For example, each member of a 100-example cluster receives weight 0.01, while each member of a 10-example cluster receives weight 0.1. For mean absolute error, this is equivalent to averaging errors within each cluster and then averaging across clusters.

Tools: pandas groupby for group sizes and the sample_weight argument of suitable scikit-learn metrics, including mean_absolute_error. [11]

Ease of applying

Moderate. Requires group definitions and a metric with an appropriate weighted form.

How robust is it?

Useful as a sensitivity analysis. Report it beside ordinary example-weighted performance: the two summaries answer different questions. Results depend on cluster definitions. Do not average within-cluster ROC-AUC when some clusters contain only one class.

Common across papers?

Weighting and group averaging are established statistical ideas. I have not verified this exact sequence-evaluation diagnostic as common across multiple papers. Present it as a supplementary robustness check rather than a standardized memorization test.

Recently introduced?

Not a newly published named method. This is an adaptation of general weighting principles.

## 5. Cluster-bootstrap confidence intervals

Method and what it measures

The cluster bootstrap, also called a group bootstrap, estimates uncertainty while keeping related observations together.

For example, if the test set has 80 defensible groups, sample 80 group IDs with replacement, retaining every observation in each selected group and its selection multiplicity. Recalculate the metric for each resample. A 95% percentile bootstrap interval uses the 2.5th and 97.5th percentiles of the resulting metric distribution.

For a model–baseline comparison, use the same sampled groups for both predictors and calculate their performance difference in each resample. This gives a paired comparison.

Implementation: a short NumPy/Python routine that samples group IDs. Ordinary row-wise resampling does not automatically implement this procedure. Grouped and hierarchical resampling are established statistical approaches. [7]

Ease of applying

Moderate. Choosing defensible groups is harder than implementing resampling.

How robust is it?

Useful when groups are reasonably independent and sufficiently numerous. Sequence-cluster membership alone does not establish independence. Paired examples sharing either member may need a more tailored approach. Cluster count is not automatically the effective sample size. These intervals describe uncertainty for the fitted model; they do not include training-run variability.

Common across papers?

Established in statistics. I have not established that it is routine across sequence-prediction benchmarks. [7]

Recently introduced?

No. Established resampling methodology. It resamples observations; it does not shuffle labels.

## 6. Training–test performance gap

Method and what it measures

Also called the train–test gap or, with appropriate qualification, an empirical generalization gap.

Evaluate the same metric on training and test data under comparable inference settings. For a higher-is-better metric, report training score minus test score. For a loss, report test loss minus training loss.

For example, training ROC-AUC of 0.98 and test ROC-AUC of 0.82 gives a gap of 0.16. This illustrates a performance difference, not the percentage attributable to memorization.

Tools: the same scikit-learn metrics used for test evaluation. [11]

Ease of applying

Very easy if both sets of predictions are available; otherwise, run inference on training examples.

How robust is it?

Limited alone. A large gap can reflect overfitting, differences between train and test populations, or both. A small gap can occur when test sequences closely resemble training sequences.

Common across papers?

Yes. Standard ML evaluation, including biological sequence modeling. [1]

Recently introduced?

No.

## 7. Reliability diagrams by training-sequence similarity

Method and what it measures

For probabilistic classifiers, use probability calibration analysis, specifically a reliability diagram, separately within method 2's similarity groups.

For binary classification, predictions near 0.8 should correspond to roughly 80% positive outcomes. Compare average predicted positive-class probability with observed positive-class frequency in probability bins.

Concrete tools: scikit-learn calibration_curve and CalibrationDisplay. These assess an existing model; fitting a recalibration model is a separate action. [13]

An optional numerical summary is expected calibration error (ECE), a weighted average of discrepancies across probability bins, with the precise definition stated. A Brier score is another probability-quality measure, but it is not a pure calibration measure.

Ease of applying

Easy to moderate. Requires probability predictions, labels, and nearest-training similarities. There are two separate groupings: sequence-similarity groups and probability bins within them.

How robust is it?

Useful for reliability, indirect for memorization. Small groups and probability-bin choices can make estimates unstable. Good calibration can coexist with poor predictive discrimination.

Common across papers?

Calibration analysis is common in ML. The prevalence of this exact sequence-similarity-specific extension has not been established here. [8]

Recently introduced?

No for calibration itself. Guo et al. (2017) is a well-known modern reference, not its origin. [8]

## 8. DataSAIL evaluation of existing splits

Method and what it measures

Use DataSAIL's split-evaluation function, documented as datasail.eval.eval_split, to summarize similarity crossing the existing split boundaries. [14]

The tool reports an absolute score and a normalized score; the normalized version divides the cross-split score by the total pairwise similarity under the chosen formulation. The authors call it a leakage score. It describes the partition, not the percentage of predictions explained by memorization.

Supply the existing split assignments and the required similarity information. Running this evaluation does not require using DataSAIL to generate new splits.

Ease of applying

Moderate. Easier if the required similarities are available. A nearest-hit-only search output is generally insufficient to reproduce a score defined over all relevant pairwise similarities.

How robust is it?

Useful as a supplementary dataset summary. Results depend on the similarity definition, weights, and split configuration. Compare scores under consistent definitions. It cannot replace model-performance analysis.

Common across papers?

Newer; not established here as a broadly adopted evaluation standard. Demonstrations by the tool's authors are not the same as widespread independent use.

Recently introduced?

Relatively recent: the main journal paper appeared in 2025; a 2026 addendum explicitly evaluated predefined splits. These are publication dates, not a claim that cross-split similarity assessment originated then. [15, 16]

Practical scope and reporting

Recommended starting set: methods 1–3, with method 5 where the dependence structure supports it. Add method 4 when test redundancy is substantial.

Keep the primary model frozen: do not change decision thresholds or baseline settings based on favorable test results. Report the analysis as post hoc.

Single sequences versus pairs: for pairs, distinguish overlap of each component from overlap of the complete input pair. Nearest matches for two separate components need not belong to the same training pair.

Pretraining: similarity to supervised training data does not capture prior exposure in a pretrained model. State which training corpus was audited.

Unrepresented cases: if the test set contains no sufficiently distant sequences, these analyses cannot establish performance on them.

Shuffling: none of the eight methods requires label shuffling. Shuffled-label retraining is a separate permutation control; it is not a direct test separating biological learning from shortcuts.

## References and tool documentation

1. Paper examples support the stated approach; they do not validate every proposed adaptation.

GraphPart: homology partitioning for biological sequence analysis (2023). Paper.

SpanSeq: similarity-based sequence data splitting method for improved development and assessment of deep learning projects (2024). Paper.

Structure-based protein function prediction using graph convolutional networks — DeepFRI (2021). Paper. See performance grouped by maximum training-sequence identity.

Leveraging conformal prediction to annotate enzyme function space with limited false positives — CPEC (2024). Paper.

DeepGO: predicting protein functions from sequence and interactions using a deep ontology-aware classifier (2018 journal issue). Paper.

The CAFA challenge reports improved protein function prediction and new functional annotations for hundreds of genes through experimental screens (2019). Paper.

Application of the hierarchical bootstrap to multi-level data in neuroscience (2020). Paper. General statistical precedent, not a sequence-specific memorization study.

On Calibration of Modern Neural Networks (2017). Paper.

MMseqs2 user guide. Search, alignment, identity, and clustering documentation.

NCBI BLAST command-line applications manual. Documentation.

Evaluation and grouping tools: pandas cut, ROC-AUC, average precision, mean absolute error and sample weights.

scikit-learn nearest neighbors: KNeighborsClassifier, KNeighborsRegressor.

scikit-learn probability calibration: User guide, calibration_curve.

DataSAIL: evaluation of existing splits. Documentation.

Data splitting to avoid information leakage with DataSAIL (2025). Paper.

16. Addendum: Data splitting against information leakage with DataSAIL (2026). Paper.

Documentation checked: 24 September 2026. Record the versions and settings actually used in any implementation.
