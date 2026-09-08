# Per-site nucleotide features for HA-NA segment matching

**Status: IN PROGRESS**

## Goal

Find out which sequence positions the model uses to make predictions. K-mer counts record how often a subsequence occurs, but not where it occurs. Using one feature per nucleotide, codon, or amino acid preserves position, so feature importance can be mapped directly along the sequence.

Scope: HA-NA, H3N2, 2024. Idea and prior results from Jamie Overbeek (see `notes.md`, chat of
2026-05-12), who used a very similar approach with a RF regressor to predict collection date.

## Files touched

### New scripts

| file | step | what it does |
|---|---|---|
| `src/analysis/plot_site_entropy.py` | 2 | Shannon entropy per CDS position; the conservation map and the alignment sanity check |
| `src/embeddings/compute_site_features.py` | 3 | builds the per-site feature cache (nt / codon / aa) |
| `src/utils/site_utils.py` | 3-4, 7a | reads the cache, builds pair feature matrices, entropy helper, one-side slot selection |
| `src/analysis/plot_site_importance.py` | 6, 7b(i) | gain / SHAP / permutation importance, plus the conventional `plot_importance` bar charts |
| `src/analysis/plot_site_group_permutation.py` | 7b(ii) | shuffle the top N sites together, no retrain |
| `src/analysis/plot_site_retrain_ablation.py` | 7b(iii) | corrupt the top N sites, then refit from scratch |
| `src/analysis/plot_seen_sequence_effect.py` | 7c | test AUC split by whether a sequence was seen in training |
| `src/analysis/plot_negative_pair_ambiguity.py` | Post-hoc | Relates false-positive rate to the minimum single-slot Hamming distance from an observed positive |
| `src/analysis/compare_negative_pair_distances.py` | Post-hoc | Compares the single-slot distance with unrestricted whole-pair distance and reports FPR jointly by both slot distances |
| `src/analysis/_importance_helpers.py` | 6, 7b | shared readers for the importance table: load, rank by a chosen measure, average a permutation curve |
| `src/analysis/plot_site_importance_trace.py` | 6 | one importance measure along the CDS, drawn from the saved table |
| `src/analysis/plot_confusion_folds.py` | Post-hoc | confusion matrix pooled over the CV folds, with the per-fold spread |
| `src/datasets/_positive_pair_selection.py` | 8 | selects unique-sequence positives and audits the resulting CV splits |
| `src/analysis/build_equal_count_pair_datasets.py` | Four-pair experiment | samples a fixed number of Hopcroft-Karp positives before the existing CV and negative-generation pipeline runs |

Plus one new config group, `conf/site/default.yaml` (`unit`, `encoding`, `slots`), 10
per-site experiment bundles, 4 step-8 positive-selection bundles, and 4 Human H3N2 2024
four-pair experiment bundles.

### Updated (existing files, extended for this plan)

| file | what changed |
|---|---|
| `src/utils/protein_utils.py` | added `starts_with_m` |
| `src/utils/gto_utils.py` | dedup key now includes `function`, so it can no longer merge two different proteins that share a sequence |
| `src/preprocess/extract_cds_dna.py` | carries `starts_with_m` / `has_terminal_stop` / `has_internal_stop` / `is_complete_cds` into `cds_dna_final` |
| `src/utils/cds_utils.py` | added `check_cds_length` |
| `conf/virus/flu.yaml` | added the `cds_length` pin table |
| `conf/dataset/default.yaml` | added `require_complete_cds_at_pinned_length` and `positive_pair_selection` |
| `src/datasets/_pair_helpers.py` | added `filter_complete_cds_at_pinned_length` |
| `src/datasets/dataset_segment_pairs.py` | wired the CDS filter and positive selection into the v2 builder |
| `src/datasets/dataset_segment_pairs_v2.py` | added selected-positive random CV and its fold audits |
| `conf/bundles/flu_base.yaml` | registered the `/site` config group |
| `docs/methods/glossary.md` | added the Site / Site unit / Site encoding / Pinned CDS length terms |
| `src/models/_pair_features.py` | added the `site` feature-source branch |
| `src/models/train_pair_baselines.py` | resolves the site cache dir and slot proteins |
| `src/models/baselines/lgbm.py` | added `categorical_feature` |
| `src/analysis/plot_site_group_permutation.py`, `src/analysis/plot_site_retrain_ablation.py` | added `--rank_by`, so the top-N sets can be ordered by gain, SHAP or permutation |

## What we found (steps 0-8 done; step 9 open)

Steps 0-7 used H3N2 HA–NA pairs collected in 2024. Step 8 also includes PB2–PA with the same metadata filters, four random-split folds and LightGBM. Within each feature comparison, the models used the same folds and positive population. Reported values are the mean and std across folds.

**Per-site nucleotide features performed similarly to k-mer features.** The nucleotide-site model obtained an F1 macro of 0.9192 ± 0.0134 using 3,111 features. The k-mer model obtained 0.9094 ± 0.0145 using 8,192 features. The nucleotide-site model scored higher in all 4 folds, but the difference was not statistically significant (p=0.128). With only 4 folds, this result does not establish either superiority or equivalence.

**Codon features retained similar performance with fewer features.** The codon model used 1,037 features and obtained an F1 macro of 0.9159. Its performance did not differ significantly from the nucleotide-site model (p=0.509). Codon features therefore match nucleotide-site performance at a third of the width, although the experiment has limited power to detect a difference.

**Nucleotide identity provided information that amino-acid identity did not preserve.** Codon and amino-acid (aa) features represent the same 1,037 positions, but codons retain nucleotide changes that do not alter the translated amino acid. The codon model obtained an F1 macro of 0.9159, compared with 0.8091 for the aa model. The mean difference was 0.107, occurred in the same direction in every fold, and had p=0.002. This result shows that information discarded during translation contributes substantially to prediction. It does not show that aa sequence contains no useful information or evaluate ESM-2 features.

**High performance persisted when every retained positive used each CDS only once.** On the
maximum-cardinality Hopcroft-Karp population, the k-mer model obtained F1 macro 0.8768 ± 0.0070
and AUC-ROC 0.9376 ± 0.0069. Per-site nt features remained slightly higher at 0.8894 ± 0.0100
and 0.9427 ± 0.0101. Per-site codon features obtained 0.8716 ± 0.0240 and 0.9276 ± 0.0227,
while per-site aa fell to 0.7331 ± 0.0068 and 0.8173 ± 0.0100. Thus, exact sequence reuse is not
required for strong performance, and the codon-aa gap remains when exact sequence reuse is
removed. The uniqueness-controlled datasets contain about half as many positives as the original
dataset and represent selected populations, so their lower scores cannot be attributed solely to
the removal of sequence reuse.

**PB2–PA was less predictable than HA–NA under the same uniqueness-controlled design.**
With Hopcroft-Karp matching, PB2–PA obtained F1 macro 0.8147 ± 0.0236 and AUC-ROC
0.9045 ± 0.0205 with k-mers. Per-site nt obtained 0.8123 ± 0.0545 and 0.8977 ± 0.0447.
Neither representation consistently outperformed the other: each won 2 of 4 folds. Both remained
predictive, but their AUC-ROC was 0.03-0.05 lower than for HA–NA under the same design. Precision
also remained substantially lower than recall with both representations.

**The fitted model concentrated importance on a small number of sites, but other sites contained overlapping predictive information.** The top-10 codon sites (ranked by importance) accounted for 28% of total mean absolute SHAP importance. Shuffling these sites without re-fitting removed 49.5% of the model’s above-chance AUC-ROC. When a new model was trained after the same sites were corrupted, the loss was 15.8%. The smaller loss after re-fitting indicates that the remaining sites contain information that can partly replace the corrupted sites.

The following checks support this interpretation:
- Models using only HA or only NA produced mean AUC-ROC values of 0.5007 and 0.4979, respectively, compared with 0.9547 when both proteins were used. Thus, neither slot alone predicted the pair label in this dataset.
- After retraining on data with 100 randomly selected sites corrupted, only 1.7-2.0% of above-chance AUC-ROC was lost, against 86-89% when the top-100 ranked sites were corrupted. This supports the importance ranking.
- On 2,953 test pairs for which neither exact sequence occurred in training, the nucleotide-site model obtained an AUC-ROC of 0.9544, compared with 0.9597 over all test pairs. The AUC-ROC difference between pairs with two previously seen sequences and pairs with no previously seen sequences was 0.0202 for nucleotide-site features and 0.0276 for k-mer features. These results do not support exact sequence reuse as the main explanation for performance. They do not rule out effects from closely related sequences or population structure.

**The entropy results support consistent reading frames.** 3rd codon positions were 2.8 times more variable than 1st positions and 3.8 times more variable than 2nd positions. Randomly shifting each sequence by 0 to 2 nucleotides removed this codon-position pattern and increased mean entropy by a factor of 19. These checks show that the method can detect reading-frame disruption and that the retained sequences have the expected codon-phase pattern. They do not by themselves prove that every site is homologously aligned.

**The main importance measures produced similar top rankings.** SHAP and gain importance had a correlation of 0.97 and shared 12 of their 15 top-N sites. Permutation importance shared 12 of 15 top sites with SHAP for HA and 11 of 15 for NA. Split-count importance produced a different ranking and was not used for the biological interpretation.

## Naming

These are called **per-site features**, not *positional encodings*. A *positional encoding* represents
token order within a model (like transformers). Here, the sequence values at fixed positions are the input features.

- `feature_source: site` selects per-site features.
- `site.unit: nt | codon | aa` specifies what each site contains.
- `site.encoding: ordinal | onehot` specifies how site values are represented.
  - `ordinal` uses one integer-coded categorical column per site. The integers are category labels,
    not ordered measurements.
  - `onehot` uses one binary column for each possible value at each site.
- `site.slots: both | a | b` selects whether features are loaded from both proteins or from only
  one protein.

The settings are grouped under the top-level `site:` configuration block, parallel to the existing
`kmer:` block. The corresponding terms are defined in `docs/methods/glossary.md`.

### Feature units

The three units preserve different levels of sequence information:

| unit | source sequence | sites per HA-NA pair | possible codes | ordinal width | one-hot width |
|---|---|---:|---:|---:|---:|
| `nt` | CDS DNA | 3,111 | 5 | 3,111 | 15,555 |
| `codon` | CDS DNA | 1,037 | 65 | 1,037 | 67,405 |
| `aa` | protein | 1,037 | 22 | 1,037 | 22,814 |

HA contributes 1,701 nucleotide sites or 567 codon/aa sites. NA contributes 1,410
nucleotide sites or 470 codon/aa sites.

Codon and aa features describe the same 1,037 translated positions. Codon features retain
the nucleotide identity of each codon, including synonymous differences. Amino-acid features retain
only the translated residue. Nucleotide features represent each of the 3 positions within a
codon separately.

Each unit includes one catch-all code:

- `nt`: 4 standard bases plus `other`;
- `codon`: 64 standard codons plus `unk`;
- `aa`: 20 standard amino acids, the terminal stop character, and `other`.

Only CDS DNA is supported. Contig DNA (`nt_ctg`) is excluded because contigs include untranslated
regions and have variable lengths, so the same index does not consistently represent the same CDS
position.

### Codon codes

Codon features use the 64 codon IDs from `genslm_vocab/tokenizer_config.json`. A codon containing a
non-ACGT character uses GenSLM's `<unk>` ID.

The feature builder does not add `<cls>`, `<eos>`, `<pad>`, or `<mask>` tokens. Those tokens are
relevant to GenSLM model input, not to the per-site LightGBM features. Input sequences are converted
to uppercase before encoding.

The numerical order of the codon IDs does not affect the current models because every site column
is declared categorical. The IDs identify categories; they do not represent numerical magnitudes.

## CDS completeness and length before filtering

These measurements were made on unique CDS sequences from H3N2 HA and NA records collected in
2024. The source was `cds_dna_final.parquet`.

| protein | unique CDS | complete CDS | complete CDS at the pinned length |
|---|---:|---:|---:|
| HA | 2,792 | 2,785 | 2,785 at 1,701 nt |
| NA | 2,415 | 2,306 | 2,301 at 1,410 nt |

A CDS is classified as complete when its corresponding protein starts with `M`, ends with `*`,
and has no internal `*`. These conditions are stored as `starts_with_m`, `has_terminal_stop`,
and `has_internal_stop`; `is_complete_cds` is their conjunction. The protein markers correspond
to the CDS start and stop codons, as described in the Background section.

All the HA sequences dropped (7 in this case; 2,792-2,785) were incomplete: 6
lacked the terminal stop marker, and one lacked both the start and terminal stop markers.

NA had 114 sequences dropped. Of these, 109 were incomplete: 90 lacked the
terminal stop marker, 17 lacked the start marker, and 2 lacked both. The remaining 5 passed
the completeness check but had lengths of 1,407 nt (3 sequences), 1,413 nt, or 1,416 nt.
Completeness and length must therefore be checked separately.

No HA or NA sequence in this population contained an internal stop marker. This observation does
not by itself rule out every possible frameshift or alignment error.

Filtering both proteins for completeness and pinned length retained `3,580` of 3,723 unique positive
HA-NA pairs, or 96.2%.

Across all 8 protein functions in `cds_dna_final`, 98.6-99.8% of unique CDS sequences began
with `ATG`. All 3 standard stop codons occurred, although one stop codon predominated for each
protein—for example, 98% of M1 sequences ended with `TGA`, while 97% of PA sequences ended with
`TAG`. The completeness check must therefore accept all 3 standard stop codons.

The regenerated `protein_final` contains 18 protein functions. `cds_dna_final` contains the 8
selected modeling functions, one primary protein product per segment.

## Design decisions

### Pair representation

HA and NA have different sequence lengths and different biological positions. HA site 500 and NA
site 500 do not represent corresponding variables. Elementwise interactions such as `diff`,
`unit_diff`, and `prod` would therefore compare unrelated positions and, because the slot vectors
have different lengths, cannot be computed directly.

The HA and NA feature vectors are concatenated in a fixed order: all HA sites followed by all NA
sites. For per-site features, the configuration must use:

```yaml
training:
  interaction: concat
  slot_transform: none
  feature_scaling: none
```

The loader validates these settings and rejects unsupported combinations.

Ordinal site values are category labels. Their numerical values do not represent magnitudes, so
normalization or standard scaling would change arbitrary code values without adding biological
meaning. One-hot features also do not need these transformations; each sequence has one active
value per site, giving every fixed-length sequence the same number of active columns.

### Model and categorical features

The experiments use LightGBM with ordinal site encoding. Every ordinal site column is passed to
LightGBM through `categorical_feature`, so nt, codon, and aa codes are treated as
unordered categories.

The one-hot encoding path is implemented but was not evaluated in these experiments. The reported
results therefore apply only to ordinal encoding with categorical LightGBM features.

A standard scikit-learn RF was not used for the primary comparison because it does not
natively treat integer-coded predictors as unordered categories. Without one-hot encoding, it
would split the category codes numerically.

### Comparable k-mer baseline

The completeness and pinned-length filter changes the HA-NA pair population. The existing 2024
folds were built before this filter and cannot provide a matched comparison.

A new 4-fold dataset was therefore built after filtering. The nucleotide-site, codon,
aa, and k-mer models use the same pair rows and fold assignments. The matched k-mer baseline
has an F1 macro of 0.9094 ± 0.0145. The earlier value of 0.9177 ± 0.0086 was measured on the
unfiltered population and is not used for comparison with the per-site models.

### Split strategy

The experiments use 4-fold CV. A balanced 4-fold cluster-disjoint split is not
feasible for this H3N2 2024 HA-NA population at `t099`: one NA cluster contains 94.6% of the
positive-pair mass, and the reported `max_balanced_k` is 1.

Random splitting allows the same sequence, or a closely related sequence, to occur in both training
and test data. The results therefore measure prediction within the H3N2 2024 population; they do not
establish performance on cluster-disjoint, future-year, or other-subtype data. The step 7c analysis
evaluates reuse of exact sequences but does not remove this broader limitation.

## Steps

0. **Preprocessing prerequisite — DONE (2026-09-01).**

   **Goal.** Record CDS completeness (one combined flag and the 3 components it's built from).

   **Implementation.**
   - Stage 1 now records `starts_with_m` in `protein_final`, next to the existing
     `has_terminal_stop` and `has_internal_stop` flags.
   - Stage 1.5 copies those three flags into `cds_dna_final` and adds:

         is_complete_cds = starts_with_m & has_terminal_stop & ~has_internal_stop

   - The 3 source flags are retained so that later experiments can choose their own
     completeness rule.
   - `handle_assembly_duplicates` now uses `[prot_seq, assembly_id, function]` as its key.
     The previous key omitted `function` and could collapse two different protein annotations
     from the same assembly when their sequences were identical.

   **Regenerated outputs.**
   - `protein_final`: 1,793,563 -> 1,793,572 rows, with one new column. The 1,793,563
     retained rows have identical values in all 28 pre-existing columns.
   - The 9 recovered rows are exactly the rows listed as removed in the archived duplicate
     report. They are all NEP or NS3 annotations; none of the 8 selected major proteins is
     affected.
   - `ctg_dna_final`: unchanged. Its CSV and parquet files are byte-identical to the archive.
   - `cds_dna_final`: unchanged at 868,240 rows, with four new boolean columns. All 11
     pre-existing columns are unchanged. The completeness counts are:
     `starts_with_m` 864,444; `has_terminal_stop` 858,776; `has_internal_stop` 0; and
     `is_complete_cds` 855,695 (98.56%).
   - Every CDS flag matches its source value in `protein_final`, and every
     `is_complete_cds` value matches the formula above. The start flag agrees with the 1st DNA
     codon on all rows after normalizing the stored lowercase DNA. The terminal-stop flag
     also agrees after translating the final codon; six final codons are the unambiguous IUPAC
     stop `TAR` rather than a literal `TAA`, `TAG`, or `TGA`.
   - The two GTO aggregate parquets are byte-identical to the pre-change archive at
     `data/processed/flu/July_2025/archive_09_01_2026/`.

   **Filtering policy.** Preprocessing records completeness but does not filter on it. A later
   experiment may need different inclusion rules, and k-mer features do not require positions
   to align. Rows are still dropped when CDS extraction or translate-back validation fails,
   because those rows have no validated CDS to retain.

   **Output organization.** The aggregate and final-data artifacts remain at their fixed paths.
   Reports from this run were moved manually to `preprocess_qc_20260901/`. This is not an
   implemented output-path change: rerunning `preprocess_flu.py` will write the reports to the
   top-level processed-data directory again.

1. **Filter — DONE (2026-09-01).**

   **Goal.** Keep only complete CDS sequences that are the pinned length.

   - Completeness alone does not guarantee equal length. 5 NA sequences are complete but a
     different length from the rest: 3 at 1,407 nt, 1 at 1,413 nt, and 1 at 1,416 nt.
   - Length is a property of the population, not of a single record, so it cannot be decided
     during preprocessing. Thus, the filter checks both conditions together, on the protein
     rows, before pairs are built: 1) a record must be complete, 2) it must be at the pinned
     length.

   **Result on H3N2 2024.**
   - `HA`: 2,792 unique CDS -> 2,785 kept. 7 were dropped because they were incomplete; none were
     dropped for being a different length.
   - `NA`: 2,415 unique CDS -> 2,301 kept. 114 were dropped: 109 were incomplete, and
     another 5 were complete but different length (3 at 1,407 nt, 1 at 1,413 nt, and 1
     at 1,416 nt).
   - Protein rows: 10,964 -> 10,787, counting one row per isolate per protein (not unique sequences).
   - Unique positive pairs (HA-NA): 3,723 -> 3,580, or 96.2%.
   - The CV folds contain `2,732` unique HA sequences and `2,298` unique NA sequences, fewer than the
     2,785 and 2,301 kept above. The cause is 169 isolates that carry only one of the two
     proteins meeting the filtering criteria.

   **Matched k-mer baseline.** The k-mer model was re-run on the filtered folds and obtained
   an F1 macro of 0.9094 ± 0.0145. The earlier value of 0.9177 ± 0.0086 was measured on the
   unfiltered population (0.008 difference). 0.9094 is the baseline that per-site features are
   compared against in step 5.

   **Config.** The filter is controlled by `dataset.require_complete_cds_at_pinned_length` in
   `conf/dataset/default.yaml`, and defaults to `off`. If it were `on` by default, every existing
   nt_cds dataset would change the next time it was rebuilt, and previously reported results would no longer be reproducible.

   **Naming.** For the length, we use the word "pinned" (`check_cds_length(..., pinned_nt)`, and
   the comment in `flu.yaml`) rather than "canonical" in code, because that word already names
   `canonical_segment` (the segment label) and `canonical_pair_key`. In plain sentences, we may
   still use "canonical length".

   **Implementation.** The filter is implemented in
   `_pair_helpers.filter_complete_cds_at_pinned_length` and is called from
   `dataset_segment_pairs.py` immediately before the `cds_dna_hash` attach step.
   `dataset_pairs_cc.py` (2D-CD builder) does not read this flag yet.

   **Source of the target length.** The target length is read from `conf/virus/flu.yaml`
   `cds_length` (HA 1,701 nt, NA 1,410 nt), not computed as the most common length in the current
   run, because a per-run value can drift between populations and make two importance maps
   impossible to compare without raising any error. `src.utils.cds_utils.check_cds_length`
   re-derives the most common length from the complete CDS actually loaded and raises an error if
   that value disagrees with the pinned value, or if fewer than 90% of records reach that length.
   Both failure conditions were tested and do fire: PB1 has no pinned length, and H5N1 HA's real
   length (1,704 nt) does not match the H3N2/H1N1 pin of 1,701 nt. The pinned-length table
   currently covers H3N2 and H1N1 only; PB1 and NS1 are excluded because neither has one fixed
   length across subtypes and years.

   **Bundle.** The filtered config bundle lives in `flu_ha_na_h3n2_2024_random_cv4_pinned_length.yaml`,
   which inherits the unfiltered bundle and adds the flag. This keeps the 0.9177 result reproducible
   from its own, unchanged bundle.

   **Regression check.** Rebuilding the unfiltered dataset (i.e., the flag is `off`) with the modified
   code reproduced the existing run byte-for-byte across all 12 fold splits.

2. **Entropy map — DONE (2026-09-01).** `src/analysis/plot_site_entropy.py`.

   **Goal.** Compute per-position entropy on the CDS nucleotide sequence, as a conservation map
   and as a check that positions are aligned across sequences.

   - The script stacks complete, same-length nucleotide sequences (`cds_dna_seq`) into a matrix
     (rows: unique CDS; columns: positions) and computes Shannon entropy down each column.
   - Output: `site_entropy.png` and one `site_entropy_{protein}.csv` per protein, written to
     `results/flu/July_2025/dataset_ha_na_h3n2_2024_random_cv4_pinned_length/site_entropy/`.
     Step 6 reads the CSV against the importance map.
   - It uses unique CDS (not pair rows) across all splits (train, val, and test).
   - Nucleotide only. `column_entropy` (the underlying function) also works on the codon and
     aa site caches from step 3, but this script has only ever been run on nucleotides;
     no codon- or aa-level entropy has been computed.

   **Conservation.** Entropy at a position is `H = −Σᵥ p_v · log2(p_v)`, summed over each
   value `v` seen at that position across the `n` unique CDS sequences, where `p_v` is the
   fraction of those sequences carrying value `v` there. H is in bits and ranges from 0 (every
   sequence agrees at that position) to log2(k), where `k` is the alphabet size (4 for a
   nucleotide: A, C, G, T).
   - `HA`: 2,732 unique CDS, mean entropy 0.0577 bits. 550 of 1,701 positions never vary
     (32.3%).
   - `NA`: 2,298 unique CDS, mean entropy 0.0580 bits. 506 of 1,410 positions never vary
     (35.9%).
   - The observed means are under 3% of the 2-bit ceiling given above, so on average positions
     are nowhere near random: both proteins are strongly conserved, and what variation exists is
     concentrated in a few spots rather than spread evenly.

   **Checking the positions really line up.** "Line up" means position `i` is the same physical
   base in the gene in every sequence, e.g. position 200 in one HA record is the same base as
   position 200 in every other HA record, not just the 200th character of whatever string that
   record happens to hold. Per-site features assume this once every sequence is the same length;
   the two checks below test whether that assumption actually holds on this data. The 2nd check
   is the sharper test.

   The `1st`/`2nd`/`3rd` columns below are the position WITHIN a codon (base 1, 2, or 3 of
   every codon), averaged over all codons in the sequence, not the 1st/2nd/3rd codon of the
   sequence.

   | | mean bits | 1st codon pos. | 2nd | 3rd | 3rd/1st ratio |
   |---|---|---|---|---|---|
   | HA, as built | 0.0577 | 0.0383 | 0.0283 | 0.1065 | 2.78x |
   | NA, as built | 0.0580 | 0.0376 | 0.0281 | 0.1084 | 2.89x |
   | HA, each seq. shifted 0-2 nt | 1.0971 | 1.0970 | 1.0981 | 1.0963 | 1.00x |

   - The 3rd position within a codon is the most variable and the 2nd position the least, in
     both proteins. This is the expected pattern, since most changes at the 3rd position of a
     codon do not change the amino acid, and most changes at the 2nd position do. The pattern
     shows the reading frame is correct; a flat entropy trace would not.
   - The last row is the negative control. Each sequence is shifted by a random 0, 1, or 2
     nucleotides, so positions no longer line up. Mean entropy jumps 19-fold and the
     codon-position pattern flattens to 1.00x, showing that the check would actually catch
     a misalignment if one existed.
   - A separate sanity check confirms the metric itself: shuffling each column's values
     independently, instead of shifting sequences, leaves the entropy numbers identical to
     4 decimals. This is expected, since entropy is computed per column.
   - This check catches wholesale misalignment, not a small subset of shifted sequences.
     The completeness-and-length filter from step 1 rules those out, since an internal
     shift would need an insertion and a deletion that exactly cancel.

3. **Feature builder — DONE (2026-09-01).** `src/embeddings/compute_site_features.py`.

   **Goal.** Build one per-site feature cache per protein and per unit (`nt`, `codon`, `aa`), so
   training and analysis can look up an ordinal code for every site of every complete,
   pinned-length CDS.

   **Implementation.**
   - New config group `conf/site/default.yaml` (`unit`, `encoding`; `slots` added later in 7a), registered in
     `conf/bundles/flu_base.yaml` next to `/kmer`. The 4 new terms are defined in
     `docs/methods/glossary.md`.
   - For each protein and unit, the builder writes 3 files to the embeddings directory:
     `site_features_{unit}_{protein}.npz` (the codes, uint8), `_index.parquet` (maps
     `cds_dna_hash` to a row number), and `_metadata.json` (code map, site count, kept/dropped
     counts).
   - Caching is existence-check based, per protein. `--force_recompute` rebuilds.
   - Only complete CDS at the pinned length take part in the cache. The matrix width equals the
     CDS length, which differs by protein, so the builder writes one matrix per protein rather
     than one for the whole corpus.
   - The cache stores ordinal codes only. `site.encoding: onehot` is expanded later, at load
     time (step 4), so switching encoding does not require rebuilding the cache. Storing
     one-hot directly would make the cache 5-65x larger for no benefit.
   - Every unit, `aa` included, is keyed by `cds_dna_hash`. Two different DNA sequences that
     translate to the same protein get two separate but identical `aa` rows. This provides one join key and one row order shared across all three
     units, so codon site *i* and aa site *i* are guaranteed to be the same position by
     construction.

   **Built for HA and NA in all three units:**

   | | unique CDS | complete | at pinned length | nt sites | codon/aa sites |
   |---|---|---|---|---|---|
   | HA | 65,414 | 64,125 | 44,202 | 1,701 | 567 |
   | NA | 58,887 | 57,278 | 46,175 | 1,410 | 470 |

   - The builder reads `cds_dna_final`, the shared preprocessed file for all subtypes and
     years, not a file scoped to H3N2 2024. The pinned length (1,701 nt for HA) only covers
     H3N2 and H1N1, so HA from other subtypes fails the length filter: H5N1 HA is 1,704 nt, and
     H9 and H7 HA is 1,683 nt. That is why HA loses 19,923 of its 64,125 complete CDS (31%) to
     the length filter. Those sequences are complete, but a different length from this pin, so
     they cannot be lined up position-by-position against a 1,701-nt reference. This loss is
     expected, not a bug; a per-site run on those other subtypes would need its own pinned
     length for their HA. It also does not affect the H3N2 2024 experiment itself: the cache
     still holds every one of the 2,732 HA and 2,298 NA hashes that experiment needs.

   **Verification.**
   - Every build decodes a sample of rows back to the source sequence and fails on any
     mismatch, so a wrong code map cannot reach a model silently.
   - Three one-time checks also passed: codon IDs rebuilt from the `nt` codes matched exactly
     on 400 rows per protein; codon-to-aa translation matched NCBI translation table 1 exactly,
     checked against three independent sources (GenSLM's tokenizer, `prot_seq`, and
     `cds_utils._CODON_TABLE_1`); and site counts line up across units (nt sites = 3 x codon
     sites = 3 x aa sites).
   - Codon IDs come from GenSLM's own vocabulary (`genslm_vocab/tokenizer_config.json`, read at
     build time, not hand-copied): GGC=33, GCC=34, ATC=35, GAC=36, the three stop codons =
     93/95/96, `<unk>` = 3.

4. **Loader and training — DONE (2026-09-02).**

   **Goal.** Wire the per-site cache into the pair-feature builder and training pipeline, and
   confirm the change does not affect other feature sources.

   **Implementation.**
   - `src/utils/site_utils.py` reads the cache, as a sibling of `kmer_utils.py`.
   - `src/models/_pair_features.py` gained a `site` branch. It used to reject
     `feature_source: site`.
   - `train_pair_baselines.py` now resolves the cache directory and figures out which protein
     is in which slot.
   - `baselines/lgbm.py` now accepts a `categorical_feature` argument.
   - New bundle `flu_ha_na_h3n2_2024_random_cv4_site_nt` inherits the pinned-length dataset
     bundle and only swaps the feature source, so any difference from the k-mer result comes
     from the features, not the underlying population.
   - Ordinal codes are labels, not numbers with meaning: code 7 is not "more" than code 3.
     Without telling LightGBM this, it would split on `<=` and read an order into the codes
     that is not really there. Under `encoding: ordinal`, every column is one site, so every
     column is declared categorical, checked against the fitted model directly, with all 3,111
     columns confirmed. One-hot columns are already 0/1, so they are left as ordinary numeric
     columns. Every other feature source (k-mer, ESM-2) passes `None`, which LightGBM treats as
     `'auto'`.
   - A file records which column is which position: `site_feature_columns_{unit}_{encoding}.csv`,
     written at load time. Columns: `column`, `slot`, `protein`, `site`, and (for one-hot)
     `code`. For example, column 0 is HA site 1, column 1700 is HA site 1701, and column 1701
     is NA site 1. Step 6 reads this file instead of re-deriving the layout.

   **Verification.**
   - Column counts match the table in step 3, confirmed on fold 0: nt 3,111 ordinal / 15,555
     one-hot; codon 1,037 / 67,405; aa 1,037 / 22,814. One-hot rows always sum to the site
     count, so exactly one code fires per site. One-hot width comes from the code map the cache
     declares, not from which values happen to appear in a given split, so train, val, and test
     always come out with identical widths.
   - Twelve error checks all fire as expected: `interaction` other than `concat`,
     `slot_transform` other than `none`, `feature_scaling` other than `none`, missing
     `site_dir`, missing or malformed `site_proteins`, slots given in the wrong order, a
     protein with no cache, an unrecognized `feature_source`, a `cds_dna_hash` not in the
     cache, the two slots built with different units, a pair table missing the hash columns,
     and an unrecognized encoding. The wrong-order check matters most, because nothing else
     confirms that the cache addressed by short name (e.g. "HA") actually matches the full
     function name the pair table carries in that slot. Without it, a run could silently
     featurize NA into slot A.
   - Regression check. The k-mer baseline on fold 0 reproduces to six decimal places after this
     change, so the shared loader and the new `categorical_feature` argument have no effect on
     the other feature sources.
   - Quick smoke test on fold 0 only: site nt F1 macro 0.9246 vs. k-mer 0.9219 on the same fold.
     One fold proves nothing by itself; step 5 is the real comparison.

5. **Train and Compare — DONE (2026-09-02).** LightGBM was trained and evaluated on all four
   pinned-length folds. Every arm uses the same folds, so the comparisons are paired.

   **Goal.** Compare per-site `nt`/`codon`/`aa` features against the k-mer baseline, and against
   each other, under matched folds, so any difference in score comes from the features rather
   than the split.

   | arm | columns | F1 macro | AUC-ROC |
   |---|---|---|---|
   | k-mer k=6 (`nt_cds`) | 8,192 | 0.9094 ± 0.0145 | 0.9564 ± 0.0064 |
   | per-site `nt` | 3,111 | **0.9192 ± 0.0134** | 0.9597 ± 0.0087 |
   | per-site `codon` | 1,037 | 0.9159 ± 0.0087 | 0.9547 ± 0.0056 |
   | per-site `aa` | 1,037 | 0.8091 ± 0.0228 | 0.8842 ± 0.0200 |

   **Paired comparison, F1 macro, across the four folds:**

   | comparison | mean difference | folds where the 1st arm wins | p-value |
   |---|---|---|---|
   | `nt` vs k-mer | +0.0098 | 4 of 4 | 0.128 |
   | `codon` vs k-mer | +0.0065 | 3 of 4 | 0.397 |
   | `codon` vs `aa` | **+0.1067** | 4 of 4 | **0.002** |
   | `nt` vs `codon` | +0.0033 | 3 of 4 | 0.509 |

   The p-values are from two-sided paired t-tests over the four fold scores. With only four
   folds, they are descriptive and provide limited statistical evidence.

   **Codon features outperform amino-acid features.** Both representations cover the same
   1,037 positions in the same records, but the amino-acid representation removes synonymous
   codon differences. Codon features increase mean F1 macro by 0.1067 and outperform
   amino-acid features on all four folds (p=0.002). This shows that synonymous nucleotide
   variation contributes substantial predictive signal. One possible explanation is that
   synonymous changes retain lineage information, but this experiment does not test that
   explanation. This result applies to the per-site amino-acid representation, not ESM-2
   embeddings.

   **Per-site nucleotide and codon features perform similarly to or slightly better than the
   k-mer baseline.** Per-site nucleotide features have the highest mean F1 macro and win on all
   four folds, but the four-fold comparison provides limited statistical evidence (mean
   difference +0.0098; p=0.128).

   **Random CV permits individual-sequence overlap.** Depending on the fold, 18-21% of test HA
   sequences and 22-26% of test NA sequences also occur in training; both sequences were seen
   for 7-10% of test rows. Exact `pair_key` overlap is zero. Step 7 directly tests whether this
   overlap explains the results.

6. **Feature Importance — DONE (2026-09-02).** `src/analysis/plot_site_importance.py`, run on the
   `codon` model. Each of its 1,037 categorical features represents one codon position and
   therefore one residue position.

   **Goal.** Rank the positions used by the fitted model and check whether the ranking agrees
   across importance measures and folds. Step 7 tests whether predictions actually depend on
   the highest-ranked positions.

   **Outputs.** Four files:
   - `site_importance_codon.png`: importance by CDS position and against entropy.
   - `site_importance_codon_barplot.png`: split-count, gain, and SHAP rankings.
   - `site_importance_codon.csv`: mean importance, variability, rank, and site metadata.
   - `site_importance_codon_per_fold.csv`: importance values for each fold.

   **Importance measures.** The script computes four:
   - **Gain** is the total reduction in training loss from tree splits using a feature. It comes
     directly from the fitted trees and describes what the model found useful during training.
   - **SHAP** is computed on each fold's held-out test rows. Mean absolute SHAP measures how
     strongly a feature affects those predictions, without retaining the direction of the
     effect. The script verifies that the SHAP contributions reconstruct the model's raw score.
   - **Split count** records how often a feature is used. It produced a substantially different
     ranking and is retained only as a diagnostic.
   - **Permutation importance** measures the test AUC-ROC loss after shuffling a feature. It is
     discussed in step 7b(i).

   Gain and SHAP are normalised within each fold before averaging because the folds contain
   different numbers of fitted trees.

   **Results.** Gain and SHAP rankings were strongly correlated (Spearman 0.971 for HA and
   0.974 for NA), and 12 of their top 15 sites overlapped. SHAP rankings were also reasonably
   stable across folds: mean fold-to-fold Spearman correlation was 0.715 for HA and 0.681 for
   NA, and 9 of each protein's top 15 sites appeared in every fold's top 15.

   The top three sites overall were HA 544, NA 310, and NA 244 by gain, and NA 284, HA 544,
   and NA 310 by SHAP.

   These measures identify positions used by the model; they do not establish biological
   importance or distinguish biological signal from lineage or sequence identification. Step 7
   tests the model's dependence on these positions.

7. **Masking and Shuffling based on Feature Importance  — DONE (2026-09-02).** Five analyses test different possible
   explanations for the model's performance:

   | analysis | question | model retrained? |
   |---|---|---|
   | 7a | Can either protein predict the label alone? | yes |
   | 7b(i) | Does the fitted model depend on individual sites? | no |
   | 7b(ii) | Does it depend on groups of top sites? | no |
   | 7b(iii) | Can a new model compensate after those sites are corrupted? | yes |
   | 7c | Does performance depend on exact sequences seen in training? | no |

   **7a. One side alone — DONE (2026-09-02).** Models were trained with HA alone, NA alone,
   or both proteins.

   | input | columns | AUC-ROC |
   |---|---|---|
   | HA + NA | 1,037 | 0.9547 ± 0.0056 |
   | HA only | 567 | 0.5007 ± 0.0076 |
   | NA only | 470 | 0.4979 ± 0.0084 |

   HA alone and NA alone perform at chance. Prediction therefore requires information from both
   slots. This test rules out a one-sided shortcut; by itself, it does not rule out recognition
   based on both sequences.

   **7b(i). Shuffle one site at a time, no retraining — DONE (2026-09-02).** In
   `src/analysis/plot_site_importance.py`, each site is shuffled across held-out test rows five
   times. The fitted model is not changed, and permutation importance is the mean AUC-ROC loss.

   SHAP and permutation importance share 12 of the top 15 HA sites and 11 of the top 15 NA
   sites. The whole-ranking correlations are lower (0.555 for HA and 0.536 for NA), largely
   because most sites have little measurable individual effect. Baseline AUC-ROC is 0.9547, or
   0.4547 above chance.

   | | value |
   |---|---|
   | sites with AUC-ROC loss greater than 0.001 | 43 of 1,037 |
   | sites with AUC-ROC loss greater than 0.005 | 10 |
   | largest single-site loss | 0.0288 (6.3% of above-chance AUC-ROC) |
   | sum of all individual-site losses | 0.2523 (55.5% of above-chance AUC-ROC) |
   | sum of top-10 individual-site losses, ranked by SHAP | 0.1388 (30.5%) |

   The largest individual-site loss is 6.3% of above-chance AUC-ROC. Individual permutation
   losses are not additive because sites may be correlated or substitute for one another.
   Section 7b(ii) therefore tests groups of sites directly.

   **7b(ii). Shuffle the top N sites together, no retraining — DONE (2026-09-02).**
   `src/analysis/plot_site_group_permutation.py` compares the top N sites by SHAP with N random
   sites. The fitted model is unchanged. Results are reported as
   `(clean AUC − shuffled AUC) / (clean AUC − 0.5)`, the share of above-chance AUC-ROC lost.

   | N | top, test | random, test | top, train | random, train |
   |---|---|---|---|---|
   | 1 | 0.049 | 0.000 | 0.039 | 0.000 |
   | 5 | 0.293 | 0.000 | 0.256 | 0.003 |
   | 10 | **0.495** | 0.008 | 0.456 | 0.004 |
   | 50 | 0.801 | 0.031 | 0.747 | 0.033 |
   | 100 | 0.892 | 0.062 | 0.851 | 0.070 |
   | 200 | 0.989 | 0.135 | 0.961 | 0.189 |
   | 500 | 1.002 | 0.532 | 0.998 | 0.541 |
   | 1,037 | 0.980 | 1.000 | 1.003 | 0.999 |

   Shuffling all 1,037 sites reduces AUC-ROC to approximately 0.5, as expected. On test data,
   shuffling the top 10 sites removes 49.5% of above-chance AUC-ROC, compared with 0.8% for 10
   random sites. About 500 random sites are needed to produce a similar loss.

   The summed individual losses of the top 10 sites are 30.5%, compared with 49.5% when those
   sites are shuffled together. Although individual permutation losses are not additive, this
   difference is consistent with correlated or substitutable information among the top sites.

   **7b(iii). Shuffle the top N sites together, then retrain — DONE (2026-09-02).**
   `src/analysis/plot_site_retrain_ablation.py` refits the model after corrupting the selected
   sites in train, validation, and test data. It compares two corruption units:

   - **Row-level:** values are shuffled independently across pair rows. The same sequence may
     receive different corrupted values in different rows.
   - **Sequence-level:** each unique sequence receives one corrupted value at each selected
     site, used consistently wherever that sequence appears.

   | N | row, top | row, random | sequence, top | sequence, random |
   |---|---|---|---|---|
   | 1 | 0.004 | 0.000 | 0.006 | 0.000 |
   | 5 | 0.042 | -0.004 | 0.035 | 0.003 |
   | 10 | **0.158** | 0.008 | **0.126** | 0.005 |
   | 25 | 0.333 | 0.004 | 0.255 | 0.002 |
   | 50 | 0.561 | 0.015 | 0.409 | 0.008 |
   | 100 | 0.892 | 0.020 | 0.865 | 0.017 |
   | 1,037 | 1.007 | - | 0.994 | - |

   Corrupting all 1,037 sites reduces AUC-ROC to approximately 0.5 in both modes. For the top
   10 sites under row-level corruption, retraining reduces the loss from 49.5% in the fixed
   model to 15.8%. Sequence-level corruption followed by retraining produces a 12.6% loss. The
   remaining sites therefore contain information that a new model can use. At 100 sites,
   row-level corruption produces the same 89.2% loss with or without retraining, so little
   additional performance can be recovered. In contrast, corrupting 100 random sites costs
   only 1.7-2.0%.

   Sequence-level corruption is milder than row-level corruption at N=10 and N=50. This is
   consistent with a sequence retaining a stable identifier after sequence-level corruption,
   but the difference does not distinguish exact-sequence recognition from lineage or other
   population structure.

   **7c. Performance by exact-sequence reuse — DONE (2026-09-02).**
   `src/analysis/plot_seen_sequence_effect.py` groups test rows according to whether each exact
   sequence also appears in that fold's training split. Train-test `pair_key` overlap is zero in
   every fold, so no exact test pair was used for training.

   | arm | all rows | neither seen | slot a seen | slot b seen | both seen |
   |---|---|---|---|---|---|
   | k-mer k=6 | 0.9564 | 0.9493 | 0.9608 | 0.9528 | 0.9769 |
   | per-site nt | 0.9597 | **0.9544** | 0.9620 | 0.9594 | 0.9746 |
   | per-site codon | 0.9547 | 0.9489 | 0.9622 | 0.9511 | 0.9701 |
   | rows, all folds | 7,160 | 2,953 | 1,348 | 2,261 | 598 |

   On the 2,953 test pairs for which neither exact sequence appeared in training, per-site
   nucleotide AUC-ROC is 0.9544, compared with 0.9597 overall. The difference between the
   "both seen" and "neither seen" groups is 0.0276 for k-mer, 0.0202 for per-site nucleotide,
   and 0.0213 for per-site codon features. The seen-sequence advantage is therefore not larger
   for per-site features than for k-mers.

   These results do not support exact-sequence reuse as the main explanation for performance.
   They do not rule out effects from related sequences, lineage, or population structure.
   Subgroup differences are descriptive because label composition and difficulty may also
   differ among the seen-status groups.

8. **Unique-sequence positive matching with CV — DONE (2026-09-07).** Test whether performance
   remains high after each CDS is used in at most one retained positive pair. This removes exact
   sequence reuse from both slots while preserving cross-validation. It is different from the
   current `seq_disjoint` split: the positive graph is matched before random folds are assigned,
   rather than grouped into sequence-connected components and routed by LPT-greedy.

   **Population.** Start with HA-NA H3N2 2024. Do not widen the metadata range for the first run.
   The saved population has 3,580 deduplicated positives, 2,732 HA sequences and 2,298 NA
   sequences. Preliminary selection retained 1,687 or 1,703 positives with sequential
   deduplication and 1,782 with Hopcroft-Karp, about 445 test positives per fold in 4-fold CV.
   The sequential-dedup methods are order-dependent and need not produce a maximal matching;
   only Hopcroft-Karp gives the maximum number of retained positive pairs. This is enough for the
   initial comparison.

   **Implementation requirements.** Add three configurable selectors:
   `drop_duplicates(HA)` then `drop_duplicates(NA)`, the reverse order, and maximum-cardinality
   Hopcroft-Karp matching. Apply selection before fold assignment. Keep the full observed-positive
   set as the negative exclusion universe, so discarded positives can never be sampled as
   negatives. Generate negatives within each fold and allow each endpoint to use only sequences
   assigned to that fold. Audit one-use-per-slot, exact 4-fold test coverage, zero train-test CDS
   hash overlap, zero positive-negative overlap, and class balance.

   **Run in stages.**
   1. **DONE (2026-09-07):** implemented the selectors, selected-positive CV routing, run-level
      manifest, and fold audits. The selectors reproduce the measured counts of 1,687, 1,703 and
      1,782 positives. The full test suite passes (236 passed; 2 production tests deselected).
   2. **DONE (2026-09-07):** built and audited all three HA-NA datasets. Sequential deduplication
      retained 1,687 pairs in HA-then-NA order and 1,703 in NA-then-HA order; Hopcroft-Karp
      retained 1,782. Pair-key Jaccard overlap was 0.920 between the two dedup populations and
      0.922-0.925 between each dedup population and Hopcroft-Karp. All 12 fold audits found unique
      sequences in both slots, exact one-time test coverage, zero cross-split CDS-hash overlap,
      zero out-of-split negative endpoints, zero observed positives labeled as negatives, and
      exact 1:1 class balance.
   3. **DONE (2026-09-07):** trained the nucleotide k-mer baseline on all three selectors, then
      trained k-mer, per-site nt, codon and aa models on the same Hopcroft-Karp folds. All 24
      model fits completed successfully. Summaries are in
      `results/flu/July_2025/positive_pair_matching_cv/`.

   **K-mer comparison across positive populations.** These rows are descriptive, not paired,
   because each selector retains a different population.

   | positive population | positives | F1 macro | AUC-ROC | precision | recall |
   |---|---:|---:|---:|---:|---:|
   | original | 3,580 | 0.9094 ± 0.0145 | 0.9564 ± 0.0064 | 0.8600 ± 0.0220 | 0.9798 ± 0.0083 |
   | dedup HA then NA | 1,687 | 0.8577 ± 0.0116 | 0.9335 ± 0.0083 | 0.8048 ± 0.0169 | 0.9484 ± 0.0102 |
   | dedup NA then HA | 1,703 | 0.8750 ± 0.0203 | 0.9284 ± 0.0099 | 0.8191 ± 0.0261 | 0.9665 ± 0.0137 |
   | Hopcroft-Karp | 1,782 | 0.8768 ± 0.0070 | 0.9376 ± 0.0069 | 0.8244 ± 0.0133 | 0.9602 ± 0.0079 |

   All three uniqueness-controlled populations remain highly predictive, but score below the
   original population. This experiment changes three things together: exact sequence reuse,
   training-set size and which positive pairs are retained. It therefore shows that exact reuse
   is not required, but does not measure how much of the score decrease is caused by removing
   reuse. A size-matched random-positive control would be needed to separate the reuse effect from
   the smaller training population.

   **TODO:** Compare Hopcroft-Karp with size-matched random subsets of the original HA-NA
   positives. This secondary control does not block the next schema experiment.

   **Paired feature comparison on the Hopcroft-Karp folds.**

   | feature representation | columns | F1 macro | AUC-ROC |
   |---|---:|---:|---:|
   | k-mer k=6 (`nt_cds`) | 8,192 | 0.8768 ± 0.0070 | 0.9376 ± 0.0069 |
   | per-site `nt` | 3,111 | **0.8894 ± 0.0100** | **0.9427 ± 0.0101** |
   | per-site `codon` | 1,037 | 0.8716 ± 0.0240 | 0.9276 ± 0.0227 |
   | per-site `aa` | 1,037 | 0.7331 ± 0.0068 | 0.8173 ± 0.0100 |

   Per-site nt exceeded k-mer by 0.0126 mean F1 macro and won 3 of 4 folds (p=0.182). Codon was
   0.0052 below k-mer on average despite winning 3 folds, because codon fold 2 was much weaker;
   its validation AUC was also low, and all 1,037 categorical columns were configured correctly.
   Codon exceeded aa in all four folds by 0.1385 mean F1 macro (p=0.0009). As above, paired
   t-test p-values over four folds are descriptive and provide limited statistical evidence.

   **PB2-PA replication — DONE (2026-09-07).** Complete-CDS and pinned-length filtering retained
   3,958 observed positive pairs, with 2,915 unique PB2 and 2,814 unique PA sequences. The
   maximum-cardinality matching retained 2,127 positives (53.7%). All four fold audits found no
   sequence reuse within the retained positives, no cross-split CDS-hash overlap, no out-of-split
   negative endpoints, no observed positive labeled as negative, no duplicate pair keys and exact
   1:1 class balance.

   | feature representation | columns | F1 macro | AUC-ROC | precision | recall |
   |---|---:|---:|---:|---:|---:|
   | k-mer k=6 (`nt_cds`) | 8,192 | 0.8147 ± 0.0236 | 0.9045 ± 0.0205 | 0.7674 ± 0.0262 | 0.9093 ± 0.0085 |
   | per-site `nt` | 4,431 | 0.8123 ± 0.0545 | 0.8977 ± 0.0447 | 0.7634 ± 0.0533 | 0.9149 ± 0.0416 |

   PB2-PA was less predictable than HA-NA under the same filters and matching design. Per-site nt
   was 0.0024 lower in mean F1 macro and 0.0069 lower in AUC-ROC than k-mer; each representation
   won 2 of 4 folds (paired p=0.947 for F1 macro and p=0.812 for AUC-ROC). Per-site nt also varied
   more across folds. The precision-recall gap remained large for both representations. These
   results provide a schema-specific weakening, not a complete failure, and four folds cannot
   establish equivalence between the feature representations.

   Scores from different selectors describe different retained positive populations and are not
   paired fold comparisons. Feature representations trained on the same Hopcroft-Karp folds are
   paired. The proposed collection-date-gap false-positive check could not be run: predictions
   retain assembly IDs, but the local parsed metadata retains only collection year, and every
   sequence in this experiment is from 2024. Restore the exact collection date upstream before
   attempting that diagnostic; a year-gap calculation would be identically zero.

   If the one-year results remain uncertain, widen the years while keeping subtype fixed and report
   the year composition. Detailed implementation notes are in `positive_pair_matching_cv_design.md`.

9. **Cross-protein interactions — OPEN.** HA alone and NA alone perform at chance, while the
   combined model reaches 0.9547 AUC-ROC. This motivates testing which HA and NA sites the model
   uses together.

   **Plan.**
   1. Screen all HA–NA site pairs using LightGBM split/path co-occurrence statistics. Use this
      only to select candidates, because frequently used features have more opportunities to
      occur together.
   2. Compute TreeSHAP interaction values on a manageable sample of held-out test rows in each
      fold, then retain the cross-protein candidate pairs from step 1.
   3. Report each candidate's interaction strength by fold and compare its stability with random
      HA–NA pairs matched on individual feature importance.
   4. Validate the strongest stable pairs by shuffling or ablating both sites together on held-out
      data and comparing the result with the corresponding single-site perturbations.

   These results would describe interactions used by the model, not biological epistasis or
   coevolution. Random CV does not remove effects from related sequences, lineage, or population
   structure; stronger biological interpretation requires a sequence-disjoint or
   uniqueness-controlled evaluation.


## Human-H3N2-2024 four pairs experiment

Related: `docs/results/2026-09-08_cds_pair_capacity.md`.

  1. Define the comparison — DONE.
      - HA–NA, PB2–PA, PB2–NA, and PA–HA.
      - Human H3N2 collected in 2024.
      - Each pair schema is filtered independently to keep complete CDS at the pair's pinned lengths.
      - No common (six-protein) cohort is imposed. Therefore, eligible isolates can differ among schemas.
      - Target: 1,698 positives per schema. After independent Human H3N2 2024 completeness and
        pinned-length filtering, HA-NA has the smallest Hopcroft-Karp matching of the four schemas
        at 1,698 positives, so it sets the common sample size. The common six-protein cohort is not
        used here; under that restriction, HA-NA would retain 1,686 positives.

  2. Build and audit equal-count datasets — DONE (2026-09-08).
      - `src/analysis/build_equal_count_pair_datasets.py` wraps the existing selector for this
        experiment. It runs Hopcroft-Karp, samples without replacement using seed 42, and then
        returns control to the existing CV and negative-generation pipeline. No production
        configuration option was added.
      - Sampling occurs before CV assignment and negative generation. The full observed-positive
        universe remains the negative-blocking set. The selected pair-key manifest and audit record
        both the pre-sampling and final checksums.

        | pair | observed positives | Hopcroft-Karp | retained |
        |---|---:|---:|---:|
        | HA-NA | 3,466 | 1,698 | 1,698 |
        | PB2-PA | 3,837 | 2,030 | 1,698 |
        | PB2-NA | 3,532 | 1,745 | 1,698 |
        | PA-HA | 3,805 | 1,944 | 1,698 |

      - All 16 fold audits passed: unique positive endpoints, exact one-time test coverage, zero
        cross-split sequence-hash overlap, negative endpoints confined to their split's positive
        sequence pool, no observed positives labeled as negatives, no duplicate pair keys, and
        exact 1:1 class balance. The datasets use the ratio-driven `within_fold` negative sampler,
        not the coverage-first sampler.
      - The focused test run passed 18 tests, and Ruff passed for the driver and its tests.

  3. Train the four feature representations — OPEN.
      - Nucleotide 6-mers.
      - Per-site nucleotide.
      - Per-site codon.
      - Per-site amino acid.
      - Reuse the same dataset and folds across all four representations within each schema.
      - Define matching identity using nucleotide CDS. Note that distinct nucleotide sequences can collapse to identical amino-acid sequences.

  4. Compare schemas and representations — OPEN.
      - Report F1 macro, AUC-ROC, precision, and recall across four folds.
      - Report eligible-isolate counts and isolate overlap among schemas.
      - Treat cross-schema differences as descriptive: equal positive counts do not equalize sequence diversity, negative difficulty, feature width, or isolate membership.
      - If an interesting difference appears, we should later (not now) consider repeating the sampling with additional seeds and consider adding a production `max_positives` option.


## Post-hoc: where the false positives sit

**Goal.** Determine whether false positives are concentrated among negative pairs that closely
resemble observed positive pairs.

**The observation.** In every fold, precision is lower than recall for both per-site `nt` and
k-mer features. Averaged over the folds, precision/recall is 0.8760/0.9776 for per-site `nt` and
0.8600/0.9798 for k-mer. On fold 0 of the per-site `nt` arm, this corresponds to 127 false
positives and 9 false negatives.

The decision threshold is 0.5 because `training.threshold_metric` is unset; it is not tuned on the
validation split. The folds are balanced at `neg_to_pos_ratio=1.0`, but neither balance nor a 0.5
threshold requires symmetric errors. For illustration, a threshold of about 0.70 gives 312 false
positives and 310 false negatives when the per-site `nt` predictions are pooled across folds. That
threshold was found from the test predictions and must not be used operationally. It only shows
that the precision-recall asymmetry depends on the threshold.

**Near-duplicate negatives.** Negative pairs combine slot-A and slot-B sequences that were not
observed together. Both samplers reject exact observed co-occurrences, but neither rejects pairs
that are close to them in sequence. These are called **near-duplicate negatives**; see
`docs/methods/glossary.md`. Their label means "recombined and not observed co-occurring", not
"biologically incompatible".

**The measurement.** For a negative pair `(A, B)`:

- `distance_slot_b` is the minimum Hamming distance between B and any slot-B sequence observed
  with A.
- `distance_slot_a` is the minimum Hamming distance between A and any slot-A sequence observed
  with B.
- `distance_min` is the smaller of those two distances and is used for binning.

Hamming distance counts mismatched sites between equal-length, aligned sequences. Thus,
`distance_min` is the distance to the nearest observed positive that shares one sequence with the
negative pair; it is not a search over all positive pairs. Both slot distances remain in the CSV.
The comparison uses every observed co-occurrence, not only test positives.
`src/analysis/plot_negative_pair_ambiguity.py` reads the saved `test_predicted.csv` files and
does not retrain the model.

**Result**, pooled over the 4 random-CV folds of the per-site `nt` arm (3,580 negatives):

| minimum single-slot Hamming distance | negatives | FPR (=FP/N) |
|---|---:|---:|
| 0-2 nt | 176 | 0.824 |
| 3-5 nt | 847 | 0.234 |
| 6-10 nt | 1,748 | 0.078 |
| 11-20 nt | 547 | 0.027 |
| >20 nt | 262 | 0.008 |

The raw share of false positives depends on how many negatives fall within each distance.
Enrichment accounts for that:

| within | share of negatives | share of FPs | enrichment |
|---|---:|---:|---:|
| 2 nt | 4.9% | 29.2% | 5.95x |
| 5 nt | 28.6% | 69.2% | 2.42x |
| 10 nt | 77.4% | 96.6% | 1.25x |

Within 5 nt, 28.6% of negatives account for 69.2% of false positives, corresponding to 2.42x
enrichment. The 10 nt result is less informative because 77.4% of all negatives are already within
that distance.

Interpret the two quantities separately. Distances and bin sizes describe the sampler and
population. FPR and enrichment also depend on the fitted model and its 0.5 threshold. The k-mer arm
shows the same trend, so the association is not specific to one feature representation.

**Sensitivity to the distance definition.** `distance_min` searches only observed positives that
share HA or NA exactly with the negative. It is therefore an upper bound on the unrestricted
nearest-positive distance

`min[Hamming(HA, HA') + Hamming(NA, NA')]`

over all observed positive pairs `(HA', NA')`.
`src/analysis/compare_negative_pair_distances.py` computed both measures for all 3,580 negatives.
The unrestricted distance is smaller for 757 negatives (21.1%). The median gap is 0 nt over all
negatives and 1 nt among the 757 that change; the median distance is 7 nt under both definitions.
Per-negative distances, summary tables, and the slot-distance heatmap are in
`results/flu/July_2025/dataset_ha_na_h3n2_2024_random_cv4_pinned_length/negative_pair_distance_comparison/`.

The unrestricted search changes true negatives more often than false positives (22.4% versus
13.5%), so the near-distance enrichment becomes slightly weaker:

| distance measure | within 2 nt | within 5 nt | within 10 nt |
|---|---:|---:|---:|
| `distance_min` | 5.95x | 2.42x | 1.25x |
| unrestricted whole-pair distance | 5.84x | 2.33x | 1.23x |

The conclusion does not change. Under the unrestricted measure, FPR still decreases monotonically
from 0.810 at 0-2 nt to 0.008 beyond 20 nt.

**Both slot distances matter.** The two-dimensional table retains information hidden by
`distance_min`. All 56 negatives with both slot distances at 0-2 nt are false positives. When
only one slot is at 0-2 nt and the other is at 3-10 nt, 85 of 108 are false positives (78.7%).
FPR falls to 0.560 when both slots are at 3-5 nt and to 0.122 when both are at 6-10 nt. Thus, the
minimum distance captures the main trend, while the pair of slot distances describes its strength
more completely.

**Limits.** No negative has distance 0: an exact match would reproduce an observed `pair_key` and
would be rejected by the sampler. Near-duplicate negatives are therefore distinct sequences, not
retained duplicates, and per-side sequence deduplication would not remove them.

These negatives are hard, not demonstrably unlabelable. The per-site `nt` model correctly rejects
31 of the 176 negatives in the 0-2 nt bin and 77% of those in the 3-5 nt bin. This analysis also
cannot explain why precision is lower than recall because it examines only negative rows.

`distance_min` is slot-sensitive: NA supplies the minimum for 75% of negatives and HA for 25%.
Different sequence lengths may contribute, but sequence diversity also matters. Hamming distance
weights every site equally, unlike the fitted model, and measures sequence similarity rather than
phylogeny. The result therefore shows an association between short sequence distance and false
positives; it does not establish biological compatibility or show that the model is wrong.

## Open questions for Jamie

Checked against the 2026-05-12 chat in `notes.md`.

1. **Did you filter to complete CDS, or did your GenBank pull already contain only complete
   records?** Asked twice, never directly answered. She said "mostly a single length" early and
   "the single length holds for all segments" later. Our data has 4.7% of NA off-length, nearly all
   incomplete records, so either the datasets differ or something filtered them. She sampled 300
   unique sequences per season; that sampling may have picked complete ones.
2. **Did you compare ordinal codes against one-hot?** Not covered in the chat.

Already answered: masking meant dropping the columns, not shuffling. She said "I would just exclude those
features with high importance value".

## Risks

- **Memorisation — checked in step 7c, and it is not what carries the result.** The worry was that
  with ~1,700 positions per segment a handful is enough to identify a sequence exactly, so a
  per-site vector nearly names the sequence it came from, while k-mer counts do not. Splitting the
  test rows by whether their sequences appear in training: on the 2,953 rows where neither
  sequence was ever seen, per-site nt scores 0.9544 against its overall 0.9597. The gap between
  "both seen" and "neither seen" is +0.0202 for per-site nt and +0.0276 for the k-mer baseline --
  so per-site leans on recall LESS than k-mers do, the opposite of the concern.
- **Filter changes the population.** The filter drops 3.8% of pairs, so no earlier 2024 number
  is directly comparable. Settled in step 1: the k-mer baseline was re-run on the filtered folds
  and scores 0.9094 ± 0.0145. Compare against that, not against 0.9177.
- **One year, one subtype.** Nothing here shows the importance map generalises to other years or
  subtypes. Treat it as a description of H3N2 2024.

---

## Background: what "complete" means and why we filter on it

### The background you need

A CDS (coding sequence) is the stretch of DNA that codes for one protein. The cell reads it 3
letters at a time, and each group of 3 becomes one letter of a protein. So a 1,410-letter CDS makes
a 469-letter protein, plus one group at the end that means "stop here" (stop codon) — the CDS is
always exactly 3 times the length of the protein record it came from.

Two markers tell you where a CDS begins and ends:

- **Starts** with `ATG` in the DNA, which becomes `M` as the 1st protein letter.
- **Ends** with one of three specific 3-letter groups meaning "stop", written as `*` at the end
  of the protein.

A record that has both markers covers the whole CDS. That is what "complete" means. It describes
the RECORD we hold, not the organism, and it is the same property whether you look at the protein
or the DNA (measured below to agree).

### What "not complete" means

The record covers only part of the CDS. Someone sequenced part of it, or the assembly ran out of
data before reaching the end. The letters that are there are correct. There are just fewer of them
than the whole CDS has.

A stop in the MIDDLE is a different problem, not a version of this one. A short record shifts every
position after the cut, which is exactly what breaks per-site features. A mid-sequence stop shifts
nothing — it means the read is bad or that copy of the CDS is non-functional. In this data version
there are zero of them: 0 of 868,240 rows in `cds_dna_final` and 0 of 1,793,572 in `protein_final`.
Keeping the flag therefore removes no rows today and will catch the case in a future data version,
but it is not what makes the positions line up. If one ever appears, drop the record.

Confirmed: for every one of these, the DNA length exactly matches the protein length it came from.
Our extraction never disagrees with its source. So nothing is corrupted.

It matters here because per-site features number the positions 1, 2, 3... and compare position 200
across sequences. If one record is missing the first 20 letters, its position 200 is a different
place than everyone else's, and the comparison is meaningless.

### Do the protein check and the DNA check agree?

You can look for these markers in either the protein or the DNA. Both ways compared on all 868,240
rows of `cds_dna_final`:

| question | asked of the protein | asked of the DNA | agree |
|---|---|---|---|
| does it have a proper start? | starts with `M` | starts with `ATG` | **100%** |
| does it have a proper end? | ends with `*` | ends with `TAA`, `TAG`, or `TGA` | 99.999% |
| is there a stop in the middle? | `*` in the middle | stop group in the middle | **100%** |

The literal DNA check misses six final `TAR` codons. In the IUPAC alphabet, `R` means A or G,
so `TAR` represents either `TAA` or `TAG`; both are stop codons. A translation-aware DNA check
therefore agrees with the protein flag on all 868,240 rows.

Stage 1.5 reuses the protein flags because Stage 1 already computed them and translate-back
validation checks the protein against the extracted DNA. This also avoids implementing a 2nd
IUPAC-aware stop-codon check.

---

## Known weak points in Stage 1 / 1.5 (audit, 2026-09-01)

Found while auditing `preprocess_flu.py` and `extract_cds_dna.py` before step 0. None is firing
today. Each is recorded here because the evidence took a while to gather and is easy to lose.

The audit's headline result: `extract_cds_dna.py` reproduces its archived output exactly —
868,240 rows, all 11 columns byte-identical, zero rows dropped. Joins, location parsing,
coordinates and translation are correct on this corpus. Also verified: `brc_fea_id` unique in both
outputs, `(assembly_id, function)` unique in both, no nulls in any critical column, `prot_hash` and
`cds_dna_hash` equal to md5 of their sequences, `assembly_id -> file` 1:1, and no duplicate
`(assembly_id, genbank_ctg_id)` contigs, which matters because `extract_cds_dna` builds a dict on
that key and would otherwise overwrite silently.

- **The minus-strand path is never exercised.** All 2,070,209 features in the corpus are on the `+`
  strand, so `extract_cds_dna`'s reverse-complement branch has never run on real data. If it ever
  does with a multi-exon feature, check the exon order first: reverse-complementing a concatenation
  reverses exon order, and the code assumes the order in `location` is already correct. Single-exon
  minus-strand is unambiguous and safe.
- **`genetic_code` is 11 on every row, but translation uses NCBI table 1.** This is correct —
  tables 1 and 11 have identical codon-to-residue maps and differ only in permitted start codons —
  but nothing checks or documents it. Translate-back validation would catch a genuinely different
  table, so the guard exists indirectly.
- **`extract_cds_dna.py` reads `.csv` when Stage 1 writes both `.csv` and `.parquet`.** Slower, and
  it round-trips `location` and `prot_seq` through text for no benefit. Switching to parquet should
  be verified against the archive rather than assumed, since the byte-identical reproduction above
  was measured on the CSV path.
- **The ESM-2 readiness filter gates a shared output.** `preprocess_flu.py` drops rows whose
  `esm2_ready_seq` is null from `protein_final`, which every experiment reads. 199 rows today, all
  auxiliary proteins (M2, M42, PB1-F2, NS3, PA-X), so nothing the nt_cds path needs is lost. But an
  ESM-2-specific rule is deciding the contents of a shared artifact.
- **Dead code.** The "drop unassigned replicons" filter in `apply_protein_basic_filters` removes 0
  rows and cannot fire: canonical-segment assignment already requires a mapped replicon, and that
  filter runs first. It still writes an empty CSV.
- **Module globals in functions.** `validate_protein_counts` reads `core_functions` and
  `analyze_protein_counts_per_file` reads `output_dir` from module scope rather than taking them as
  parameters.
