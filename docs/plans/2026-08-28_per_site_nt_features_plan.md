# Per-site feature importance for HA-NA segment matching

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

Plus one new config group, `conf/site/default.yaml` (`unit`, `encoding`, `slots`), and 6 new experiment bundles (`..._pinned_length`, `..._site_nt`, `..._site_codon`, `..._site_aa`, `..._site_codon_slot_a`, `..._site_codon_slot_b`).

### Updated (existing files, extended for this plan)

| file | what changed |
|---|---|
| `src/utils/protein_utils.py` | added `starts_with_m` |
| `src/utils/gto_utils.py` | dedup key now includes `function`, so it can no longer merge two different proteins that share a sequence |
| `src/preprocess/extract_cds_dna.py` | carries `starts_with_m` / `has_terminal_stop` / `has_internal_stop` / `is_complete_cds` into `cds_dna_final` |
| `src/utils/cds_utils.py` | added `check_cds_length` |
| `conf/virus/flu.yaml` | added the `cds_length` pin table |
| `conf/dataset/default.yaml` | added `require_complete_cds_at_pinned_length` |
| `src/datasets/_pair_helpers.py` | added `filter_complete_cds_at_pinned_length` |
| `src/datasets/dataset_segment_pairs.py` | wired the filter in |
| `conf/bundles/flu_base.yaml` | registered the `/site` config group |
| `docs/methods/glossary.md` | added the Site / Site unit / Site encoding / Pinned CDS length terms |
| `src/models/_pair_features.py` | added the `site` feature-source branch |
| `src/models/train_pair_baselines.py` | resolves the site cache dir and slot proteins |
| `src/models/baselines/lgbm.py` | added `categorical_feature` |

## What we found (steps 0-7 done; step 8 open)

All experiments used H3N2 HA–NA pairs collected in 2024, four random-split folds, and LightGBM. Each feature representation used the same folds. Reported values are the mean and std across folds.

**Per-site nucleotide features performed similarly to k-mer features.** The nucleotide-site model obtained an F1 macro of 0.9192 ± 0.0134 using 3,111 features. The k-mer model obtained 0.9094 ± 0.0145 using 8,192 features. The nucleotide-site model scored higher in all 4 folds, but the difference was not statistically significant (p=0.128). With only 4 folds, this result does not establish either superiority or equivalence.

**Codon features retained similar performance with fewer features.** The codon model used 1,037 features and obtained an F1 macro of 0.9159. Its performance did not differ significantly from the nucleotide-site model (p=0.509). This makes codon features the smaller representation among the two tested per-site nucleotide representations, although the experiment has limited power to detect a difference.

**Nucleotide identity provided information that amino-acid identity did not preserve.** Codon and amino-acid features represent the same 1,037 positions, but codons retain nucleotide changes that do not alter the translated amino acid. The codon model obtained an F1 macro of 0.9159, compared with 0.8091 for the amino-acid model. The mean difference was 0.107, occurred in the same direction in every fold, and had p=0.002. This result shows that information discarded during translation contributes substantially to prediction. It does not show that amino-acid sequence contains no useful information or evaluate ESM-2 features.

**The fitted model concentrated importance on a small number of sites, but other sites contained overlapping predictive information.** The top-10 codon sites (ranked by importance) accounted for 34% of total mean absolute SHAP importance. Shuffling these sites without re-fitting removed 49.5% of the model’s above-chance AUC-ROC. When a new model was trained after the same sites were corrupted, the loss was 15.9%. The smaller loss after re-fitting indicates that the remaining sites contain information that can partly replace the corrupted sites.

The following checks support this interpretation:
- Models using only HA or only NA produced mean AUC-ROC values of 0.5007 and 0.4979, respectively, compared with 0.9547 when both proteins were used. Thus, neither slot alone predicted the pair label in this dataset.
- Shuffling 100 randomly selected sites removed 1.7–2.0% of above-chance AUC-ROC, compared with 86–89% when the top-100 ranked sites were shuffled. This supports the importance ranking.
- On 2,953 test pairs for which neither exact sequence occurred in training, the nucleotide-site model obtained an AUC-ROC of 0.9544, compared with 0.9597 over all test pairs. The AUC-ROC difference between pairs with two previously seen sequences and pairs with no previously seen sequences was 0.0202 for nucleotide-site features and 0.0276 for k-mer features. These results do not support exact sequence reuse as the main explanation for performance. They do not rule out effects from closely related sequences or population structure.

**The entropy results support consistent reading frames.** 3rd codon positions were 2.8 times more variable than 1st positions and 3.8 times more variable than 2nd positions. Randomly shifting each sequence by 0 to 2 nucleotides removed this codon-position pattern and increased mean entropy by a factor of 19. These checks show that the method can detect reading-frame disruption and that the retained sequences have the expected codon-phase pattern. They do not by themselves prove that every site is homologously aligned.

**The main importance measures produced similar top rankings.** SHAP and gain importance had a correlation of 0.97 and shared 12 of their 15 top-N sites. Permutation importance also shared 12 of its 15 top-N sites with SHAP. Split-count importance produced a different ranking and was not used for the biological interpretation.

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

HA contributes 1,701 nucleotide sites or 567 codon/amino-acid sites. NA contributes 1,410
nucleotide sites or 470 codon/amino-acid sites.

Codon and amino-acid features describe the same 1,037 translated positions. Codon features retain
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

All the HA sequences outside the pinned length (7 in this case; 2,792-2,785) were incomplete: 6
lacked the terminal stop marker, and one lacked both the start and terminal stop markers.

NA had 114 sequences outside the pinned length. Of these, 109 were incomplete: 90 lacked the
terminal stop marker, 17 lacked the start marker, and 2 lacked both. The remaining 5 passed
the completeness check but had lengths of 1,407 nt (3 sequences), 1,413 nt, or 1,416 nt.
Completeness and length must therefore be checked separately.

No HA or NA sequence in this population contained an internal stop marker. This observation does
not by itself rule out every possible frameshift or alignment error.

Filtering both proteins for completeness and pinned length retained 3,580 of 3,723 unique positive
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
amino-acid, and k-mer models use the same pair rows and fold assignments. The matched k-mer baseline
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
   - Result: `protein_final` went from 1,793,563 to 1,793,572 rows (+9 rows, one new column). Every pre-existing row is byte-identical on every column it already had.
   - `ctg_dna_final`: unchanged.
   - `cds_dna_final`: still 868,240 rows, four new columns added; 855,695 rows (98.56%) are complete.
   - The +9 new rows are exactly the rows an old duplicate-key bug used to remove (see the last bullet below).
   - Added `starts_with_m` in Stage 1 (`src/utils/protein_utils.py`), next to the existing `has_terminal_stop` and `has_internal_stop`. This is the one completeness fact nothing recorded before. It is computed from the protein, not the DNA, because the protein version is more reliable (see Background).
   - Copied all three flags into `cds_dna_final` in Stage 1.5. That file already holds `prot_seq`, so this copies a value instead of recomputing it — same rule the repo already uses for hashes.
   - Added one derived column in Stage 1.5:

         is_complete_cds = starts_with_m & has_terminal_stop & ~has_internal_stop

     Kept the three underlying flags too, so an experiment can combine them differently. Per-site features strictly need only the first two — an internal stop does not shift positions — but step 1 filters on the combined column.
   - **Flag, don't drop.** Preprocessing is shared by every experiment, so dropping a record here would remove it from experiments that don't care about completeness (k-mer features never need positions to line up, for example). Recording a fact and letting each experiment apply its own rule is also what Stage 1 already does with `has_terminal_stop`, `has_ambiguities`, `x_count_ratio`, and the rest — the drops that already exist in `extract_cds_dna.py` are a different case and stay, because those rows failed extraction and there is nothing to record. The rule: drop only when there is no data to record; flag when there is. Don't name a column "invalid" — that is a judgment about use, not a fact about the record.
   - Re-ran both stages and diffed the output against the pre-change archive (`data/processed/flu/July_2025/archive_09_01_2026/`): same row count, same columns plus the new flag, every other value identical. Do not proceed past this step until that check passes.
   - Output layout: top level still keeps five files — `protein_final`, `ctg_dna_final`, `cds_dna_final`, and the two GTO aggregates `protein_agg_from_GTOs` / `genome_agg_from_GTOs` (read back by fixed path, so their names and location cannot change). The ~13 other report files moved to `preprocess_qc_20260901/`; the analysis scripts that read them were updated to match.
   - Kept `extract_cds_dna.py` separate from `preprocess_flu.py`, and kept the two aggregate file names as they are, so the archive diff stays a plain file-for-file comparison.
   - **Where the +9 rows come from:** `handle_assembly_duplicates` used to key on `[prot_seq, assembly_id]` only, which could merge two different proteins from one segment when they happened to share a sequence. `function` is now part of the key. This did not touch any of the 8 major proteins this plan uses — only auxiliary proteins were affected.

1. **Filter — DONE (2026-09-01).** Two conditions are needed, not one.
   - `is_complete_cds` alone does not guarantee equal length: 5 NA sequences are complete but the wrong length (1407 nt x3, 1413, 1416).
   - Length is a property of the whole population, not of one record, so it cannot be decided during preprocessing. Both conditions — complete AND at the pinned length — are applied together, on the protein rows, before pairs are built.

   **Result on H3N2 2024:**
   - HA: 2,792 unique CDS → 2,785 kept (7 dropped for incomplete, 0 for wrong length).
   - NA: 2,415 → 2,301 kept (109 incomplete, 5 complete but wrong length).
   - Protein rows: 10,964 → 10,787.
   - Unique positive pairs: 3,723 → 3,580 (96.2% kept).
   - Every CDS in the built folds is now exactly one length: HA went from 5 different lengths to 1, NA from 15 to 1, and both are 100% complete. That is what this step exists to guarantee.
   - The folds end up with 2,732 unique HA and 2,298 unique NA (fewer than the 2,785 / 2,301 kept above), because 169 isolates carry only one of the two proteins and so never form a pair.

   **K-mer baseline re-run on the filtered folds:** F1 macro 0.9094 +/- 0.0145, against 0.9177 +/- 0.0086 on the unfiltered folds. The 0.008 drop is smaller than the normal spread across folds, so the filter itself does not change what k-mers can do on this population — the wider spread is the cost of having 4% fewer pairs. **0.9094 is the number per-site features have to beat.**

   - **Config.** New setting `dataset.require_complete_cds_at_pinned_length` in `conf/dataset/default.yaml`, off by default. Off by default because turning it on changes which rows survive; leaving it on would silently change every nt_cds dataset built before it existed and make old results impossible to reproduce.
   - **Naming.** Not called "canonical" in code. That word is already used for two other things in this repo — `canonical_segment` (the segment label) and `canonical_pair_key` (the order-invariant dedup key, `'__'.join(sorted([hash_a, hash_b]))`). A 3rd meaning as an identifier would be ambiguous. The word used here is "pinned" (`check_cds_length(..., pinned_nt)`, and the comment in `flu.yaml`). In plain sentences "canonical length" is still fine and is still used in `flu.yaml` and in `check_cds_length`'s error text — the naming rule is about identifiers, not prose.
   - **Implementation.** `_pair_helpers.filter_complete_cds_at_pinned_length`, called from `dataset_segment_pairs.py` right before the `cds_dna_hash` attach step. It checks `(assembly_id, function)` membership rather than doing a merge, so a duplicate key in `cds_dna_final` cannot silently multiply protein rows. `dataset_pairs_cc.py` (the 2D-CD builder) does NOT read this flag yet.
   - **Where the target length comes from.** `conf/virus/flu.yaml` `cds_length` (HA 1701, NA 1410) — not "the most common length in this run", because that value can drift between different populations and quietly make two importance maps impossible to compare. `src.utils.cds_utils.check_cds_length` re-derives the most common length from the complete CDS this run actually loaded and fails if it disagrees with the pinned value, or if fewer than 90% of records hit that length. Both failure cases were tested and do fire: PB1 (no pinned length exists for it) and H5N1 HA (real length 1704, against the H3N2/H1N1 pin of 1701). The pinned-length table only covers H3N2 and H1N1; PB1 and NS1 are left out because neither has one fixed length across subtypes and years.
   - **Bundle.** `flu_ha_na_h3n2_2024_random_cv4_pinned_length` — a new bundle that inherits the unfiltered one and adds the flag, rather than editing the unfiltered bundle in place. This keeps the 0.9177 result reproducible from its own unchanged recipe.
   - **Regression check.** Rebuilding the unfiltered dataset with the modified code reproduces the existing run byte-for-byte across all 12 fold splits — so the change has no effect when the flag is off.

2. **Entropy map — DONE (2026-09-01).** `src/analysis/plot_site_entropy.py`.
   - What it does: stacks the kept sequences into a matrix (rows = unique CDS, columns = positions) and computes Shannon entropy down each column.
   - Output: `site_entropy.png` and one `site_entropy_{SHORT}.csv` per protein, written to `results/flu/July_2025/dataset_ha_na_h3n2_2024_random_cv4_pinned_length/site_entropy/`. Step 6 reads the CSV against the importance map.
   - Uses unique CDS, not pair rows — a heavily sampled strain would otherwise dominate the answer.
   - Uses all splits (train + val + test), because nothing is being fit here. If entropy is ever used to select which positions to keep, it must be recomputed on the training split only.

   **Conservation.**
   - HA: 2,732 unique CDS, mean entropy 0.0577 bits, 550 of 1,701 positions never vary (32.3%).
   - NA: 2,298 unique CDS, mean entropy 0.0580 bits, 506 of 1,410 positions never vary (35.9%).
   - The maximum possible entropy for 4 bases is 2 bits, so both proteins are strongly conserved with a few variable spots.

   **Checking the positions really line up.** Two checks; the 2nd is the sharper one.

   | | mean bits | 1st codon pos. | 2nd | 3rd | 3rd/1st ratio |
   |---|---|---|---|---|---|
   | HA, as built | 0.0577 | 0.0383 | 0.0283 | 0.1065 | 2.78x |
   | NA, as built | 0.0580 | 0.0376 | 0.0281 | 0.1084 | 2.88x |
   | HA, each sequence shifted 0-2 nt | 1.0971 | 1.0970 | 1.0981 | 1.0963 | 1.00x |

   - 3rd codon positions are the most variable and 2nd the least, in both proteins — the expected pattern, since most changes at the 3rd position of a codon do not change the amino acid, and most changes at the 2nd position do. This pattern is what shows the reading frame is correct; a flat entropy trace would not.
   - The last row is the negative control: shift each sequence by a random 0, 1, or 2 nucleotides, so positions no longer line up. Mean entropy jumps 19-fold and the codon-position pattern flattens to 1.00x. This shows the check would actually catch a misalignment if one existed.
   - Separate sanity check: shuffling each column's values independently (instead of shifting sequences) leaves the entropy numbers identical to four decimals, as it must — entropy is computed per column.
   - This check catches wholesale misalignment, not one or two shifted sequences. The completeness-and-length filter from step 1 is what rules those out, since an internal shift would need an insertion and a deletion that exactly cancel.

3. **Feature builder — DONE (2026-09-01).** `src/embeddings/compute_site_features.py`, alongside `compute_esm2_embeddings.py` and `compute_kmer_features.py`.
   - New config group `conf/site/default.yaml` (`unit`, `encoding`), registered in `conf/bundles/flu_base.yaml` next to `/kmer`. The four new terms are defined in `docs/methods/glossary.md`.
   - For each protein and unit, writes three files to the embeddings directory:
     - `site_features_{unit}_{SHORT}.npz` — the codes (uint8)
     - `_index.parquet` — maps `cds_dna_hash` to a row number
     - `_metadata.json` — code map, site count, kept/dropped counts
   - Caching is existence-check based, per protein; `--force_recompute` rebuilds.

   - **One matrix per protein, not one for the corpus.** The matrix width is the CDS length, which differs by protein, so one matrix cannot hold every protein. Only complete CDS at the pinned length take part.
   - **The cache stores ordinal codes only.** `site.encoding: onehot` is expanded later, at load time (step 4), so switching encoding does not require rebuilding the cache. Storing one-hot directly would make the cache 5-65x larger for no benefit.
   - **Keyed by `cds_dna_hash` in every unit, `aa` included.** Two different DNA sequences that translate to the same protein get two separate (but identical) `aa` rows. That costs a little extra space and buys one join key and one row order shared across all three units — so codon site *i* and amino-acid site *i* are guaranteed to be the same position, by construction rather than by convention.

   **Built for HA and NA in all three units:**

   | | unique CDS | complete | at pinned length | nt sites | codon/aa sites |
   |---|---|---|---|---|---|
   | HA | 65,414 | 64,125 | 44,202 | 1,701 | 567 |
   | NA | 58,887 | 57,278 | 46,175 | 1,410 | 470 |

   - 37 MB on disk for all six matrices, against 140 MB if `nt` alone were stored uncompressed. About 15 seconds to build both proteins for one unit.
   - HA loses 19,923 complete CDS (31%) to the length filter, because the full corpus spans subtypes the pin does not cover — H5N1 HA is 1704 nt, H9 and H7 are 1683 nt. That is expected, not a bug: those sequences cannot be lined up position-by-position against a 1,701-nt reference. A per-site run on those subtypes would need its own pinned length. The H3N2 2024 dataset itself is unaffected — the cache holds every one of the 2,732 HA and 2,298 NA hashes it needs.

   **Verification.**
   - Every build decodes a sample of rows back to the source sequence and fails on any mismatch, so a wrong code map cannot reach a model silently. Positions on the catch-all code are checked the other way round, since those cannot round-trip exactly.
   - Three further checks, run once:
     - The `nt` codes rebuild the correct codon IDs exactly — 0 mismatches over 400 rows per protein.
     - Each codon translates to the correct amino-acid code (NCBI translation table 1) — 0 mismatches. This is an independent check: codon IDs come from GenSLM's tokenizer, amino-acid codes come from `prot_seq`, and the translation rule comes from `cds_utils._CODON_TABLE_1` — three separate sources, all agreeing.
     - The three units share one index, and site counts line up: nt sites = 3 x codon sites = 3 x aa sites.
   - Codon IDs come from GenSLM's own vocabulary (`genslm_vocab/tokenizer_config.json`, read at build time, not hand-copied): GGC=33, GCC=34, ATC=35, GAC=36, the three stop codons = 93/95/96, `<unk>` = 3.

4. **Loader and training — DONE (2026-09-02).**
   - `src/utils/site_utils.py` reads the cache (a sibling of `kmer_utils.py`).
   - `src/models/_pair_features.py` gained a `site` branch — it used to reject `feature_source: site`.
   - `train_pair_baselines.py` now resolves the cache directory and figures out which protein is in which slot.
   - `baselines/lgbm.py` now accepts a `categorical_feature` argument.
   - New bundle `flu_ha_na_h3n2_2024_random_cv4_site_nt`: inherits the pinned-length dataset bundle and only swaps the feature source, so any difference from the k-mer result is due to the features, not the underlying population.

   - **Categorical columns are declared.** Ordinal codes are labels, not numbers with meaning — code 7 is not "more" than code 3. Without telling LightGBM this, it would split on `<=` and read an order into the codes that is not really there. Under `encoding: ordinal`, every column is one site, so every column is declared categorical; checked against the fitted model directly — all 3,111 columns confirmed. One-hot columns are already 0/1, so they are left as ordinary numeric columns. Every other feature source (k-mer, ESM-2) passes `None`, which LightGBM treats as `'auto'`.
   - **Column counts match the table in step 3, confirmed on fold 0:** nt 3,111 ordinal / 15,555 one-hot; codon 1,037 / 67,405; aa 1,037 / 22,814. One-hot rows always sum to the site count, so exactly one code fires per site. One-hot width comes from the code map the cache declares, not from which values happen to appear in a given split, so train, val and test always come out with identical widths.
   - **A file records which column is which position:** `site_feature_columns_{unit}_{encoding}.csv`, written at load time. Columns: `column`, `slot`, `protein`, `site`, and (for one-hot) `code`. Example: column 0 = HA site 1, column 1700 = HA site 1701, column 1701 = NA site 1. Step 6 reads this file instead of re-deriving the layout.
   - **Twelve error checks, all tested and confirmed to fire:** `interaction` other than `concat`, `slot_transform` other than `none`, `feature_scaling` other than `none`, missing `site_dir`, missing or malformed `site_proteins`, slots given in the wrong order, a protein with no cache, an unrecognized `feature_source`, a `cds_dna_hash` not in the cache, the two slots built with different units, a pair table missing the hash columns, and an unrecognized encoding. The wrong-order check matters most: nothing else confirms that the cache addressed by short name (e.g. "HA") actually matches the full function name the pair table carries in that slot — without it, a run could silently featurize NA into slot A.
   - **Regression check.** The k-mer baseline on fold 0 reproduces to six decimal places after this change, so the shared loader and the new `categorical_feature` argument have no effect on the other feature sources.
   - Quick smoke test on fold 0 only: site nt F1 macro 0.9246 vs. k-mer 0.9219 on the same fold. One fold proves nothing by itself — step 5 is the real comparison.

5. **Train and compare — DONE (2026-09-02).** LGBM trained on all four pinned-length folds; every arm uses the same folds, so the comparisons are paired.

   | arm | columns | F1 macro | F1 | AUC-ROC |
   |---|---|---|---|---|
   | k-mer k=6 (nt_cds) | 8,192 | 0.9094 +/- 0.0145 | 0.9159 +/- 0.0122 | 0.9564 +/- 0.0064 |
   | per-site `nt` | 3,111 | **0.9192 +/- 0.0134** | 0.9239 +/- 0.0121 | 0.9597 +/- 0.0087 |
   | per-site `codon` | 1,037 | 0.9159 +/- 0.0087 | 0.9211 +/- 0.0077 | 0.9547 +/- 0.0056 |
   | per-site `aa` | 1,037 | 0.8091 +/- 0.0228 | 0.8257 +/- 0.0185 | 0.8842 +/- 0.0200 |

   **Paired comparison, F1 macro, across the four folds:**

   | comparison | mean difference | folds where the 1st arm wins | p-value |
   |---|---|---|---|
   | `nt` vs k-mer | +0.0098 | 4 of 4 | 0.128 |
   | `codon` vs k-mer | +0.0065 | 3 of 4 | 0.397 |
   | `codon` vs `aa` | **+0.1067** | 4 of 4 | **0.002** |
   | `nt` vs `codon` | +0.0033 | 3 of 4 | 0.509 |

   - **Per-site features at least match k-mers.** `nt` wins on every fold, but with only 4 folds the difference is not statistically significant (p=0.128), so the fair claim is "matches", not "beats" — and it does that with 3,111 columns instead of 8,192. `codon` (1,037 columns, a third of `nt`'s width) is statistically indistinguishable from `nt` (p=0.509).
   - **Silent (synonymous) changes carry most of the signal.** `codon` and `aa` cover the exact same 1,037 positions in the exact same records — the only difference is whether a DNA change that does not change the amino acid is visible to the model. Removing that information costs **0.107 F1 macro**, on every fold, p=0.002 — by far the largest effect measured, roughly an order of magnitude above the k-mer-vs-per-site gap.
     - One interpretation, not separately tested: synonymous changes are close to neutral, so they drift with lineage and can act like a lineage marker, while amino-acid changes are shaped by selection and can end up similar across different lineages. Under that reading, the matching signal is carried mostly in the DNA rather than the protein — at least at the level of detail a per-site categorical feature can capture.
     - **This has not been tested for ESM-2**, which reads 1,280 continuous dimensions rather than 22 categories per site. It should not be assumed to hold there.
   - **The memorisation risk is bounded here, but not yet ruled out at this step.** Under a random split, some sequences appear in both train and test, and a per-site vector could in principle almost identify which exact sequence it is. Measured: 18-21% of test HA sequences and 22-26% of test NA sequences also appear somewhere in training, but only **7-10% of test rows have BOTH sequences already seen in training**, and `pair_key` overlap between train and test is 0 in every fold — no test pair was trained on. So recalling a specific pairing could explain at most about a tenth of the test set. The `aa` result by itself cannot separate "the model is relying on memory" from "removing synonymous positions removes real signal", because both would look the same here. Step 7 is what separates them.

6. **Importance map — DONE (2026-09-02).** `src/analysis/plot_site_importance.py`, run on the `codon` arm (1,037 columns — statistically the same as `nt` at a third of the width, and one column per amino-acid position, so a site number IS a residue number).

   Four output files:
   - `site_importance_codon.png` — importance plotted along the CDS, and importance plotted against entropy.
   - `site_importance_codon_barplot.png` — LightGBM's own `plot_importance` bar charts (`split` and `gain`) next to the SHAP ranking.
   - `site_importance_codon.csv` — one row per column: `column, slot, protein, site, shap_frac, shap_frac_std, gain_frac, gain_frac_std, folds_used, split_count, entropy_bits, n_values, shap_rank, gain_rank`.
   - `site_importance_codon_per_fold.csv` — each fold's own numbers separately, so fold-to-fold agreement can be recomputed.
   - Both figures record the script that made them and the run name.

   **Three importance measures, for three different reasons.**
   - **Split count** — how often a feature was used to split. This is what `lightgbm.plot_importance` shows by default. On fold 0, its top list (NA 154, HA 162, HA 325, NA 233, NA 349) does not overlap at all with the gain or SHAP top-15 lists. A feature used in many shallow splits scores high here even if none of those splits mattered much. Shown for comparison only, not used to rank sites.
   - **Gain** — total reduction in loss from every split that used a feature. Read directly off the fitted trees, so on its own it describes what the trees were built on, not necessarily what they are worth on new data.
   - **SHAP** — exact TreeSHAP, computed via LightGBM's own `pred_contrib` (no extra library needed), on each fold's **held-out test split**. The script checks that SHAP values plus the base value reconstruct the model's raw margin score, rather than assuming this holds. Where gain and SHAP disagree, SHAP is treated as correct, because it is measured on data the model did not fit.
   - **The two agree closely, so the ranking survives an out-of-sample check.** Spearman +0.971 (HA) and +0.974 (NA), with 12 of the top 15 sites shared between them.
   - Two known differences: gain overstates how concentrated the signal is (HA top 10 = 34.1% of SHAP but 46.4% of gain; top 50 = 65.3% vs 79.5%), and gain undervalues sites with many possible codon values — e.g. HA site 161 (10 codon values) ranks 35th by gain but 9th by SHAP; HA 363 ranks 69th and 13th; NA 400 (11 values) ranks 22nd and 3rd; NA 385 (10 values) ranks 116th and 11th. This is the opposite of the usual warning that gain favours high-cardinality features — measured here, it does the reverse. Gain also splits credit between HA and NA slightly differently than SHAP does (57.2%/42.8% vs 52.3%/47.7%).
   - Both gain and SHAP are normalised to sum to 1 within each fold before averaging, because early stopping gives each fold a different number of trees (411 to 998), so raw totals are not comparable across folds.

   **The model concentrates on a small number of positions.**
   - HA: top 10 sites hold 34.1% of HA's total SHAP; top 50 hold 65.3%. Only 344 of 567 HA sites get any gain at all, though 97.5% of HA sites vary at least somewhat.
   - NA: top 10 hold 47.8%; top 50 hold 75.4%. Only 253 of 470 NA sites get any gain, though 96.6% vary at all.

   **The ranking is consistent enough across folds to trust.** Fold-to-fold Spearman on SHAP is 0.715 for HA (range 0.699-0.731 across the six fold pairs) and 0.681 for NA. 9 of the top-15 sites for each protein are in every single fold's top 15, and every site on both top-15 lists is used by all four folds. Ranks further down the list move around more, so treat the top list as a group of important sites rather than a precise ordering.

   Top sites by SHAP: HA 544, 36, 129, 239, 531, 286, 87, 451, 161; NA 284, 310, 400, 244, 223, 24, 140.

   **A site has to vary to matter, but varying is not enough on its own.** Spearman correlation between SHAP and entropy, restricted to sites that vary at all, is +0.516 (HA) and +0.579 (NA) — positive, since a site that never changes cannot help the model decide anything, but far from a perfect correlation. Looking at the scatter: every top site sits at entropy 0.5-1.0 bits, while most sites in that same range contribute nothing. So conservation sets an upper bound on how important a site could be, without predicting which variable sites actually matter — that gap is what makes the importance map worth having.

   **What this step alone cannot tell you.** A handful of sites carrying most of the model's decision is equally consistent with those positions carrying real lineage signal and with their being the most efficient way to identify a sequence — the memorisation risk. Step 7 is what separates them.

7. **Masking and shuffling — DONE (2026-09-02), in four passes.**
   - The original plan was one check: retrain with the top-ranked positions removed, and separately with their values shuffled. It grew into four checks, because the first attempts kept answering a narrower question than the one asked:
     - **7a** — can one side alone predict anything?
     - **7b(i)** — does the fitted model depend on one site at a time?
     - **7b(ii)** — does it depend on a group of top sites together, still without retraining?
     - **7b(iii)** — same as 7b(ii), but the model is retrained on the corrupted data.
     - **7c** — does the score depend on having seen a sequence before?

   **7a. One side alone — DONE (2026-09-02), passes.** Run first, because a failure here would mean the importance map in step 6 is not worth interpreting.
   - New config option `site.slots: a | b | both` (`conf/site/default.yaml`, default `both`) keeps one slot's columns and drops the other. New bundles `flu_ha_na_h3n2_2024_random_cv4_site_codon_slot_a` and `..._slot_b`.

   | arm | columns | F1 macro | AUC-ROC | precision |
   |---|---|---|---|---|
   | both slots (HA+NA) | 1,037 | 0.9159 +/- 0.0087 | 0.9547 +/- 0.0056 | 0.8708 |
   | slot a only (HA) | 567 | 0.4942 +/- 0.0086 | **0.5007 +/- 0.0076** | 0.4996 |
   | slot b only (NA) | 470 | 0.4819 +/- 0.0214 | **0.4979 +/- 0.0084** | 0.4992 |

   - Both one-sided models score at chance, to three decimal places: per-fold AUC-ROC never leaves 0.4914-0.5113, and precision sits on the 0.50 base rate. Read AUC-ROC here, not F1 — F1 is not centred on 0.5, so a model that just guesses "match" can still score 0.48-0.59 on F1 while having learned nothing.
   - Why this matters: the label describes a PAIR, and the negative sampler deliberately pairs one isolate's segment with a different isolate's segment, so the same HA sequence shows up in both correct and incorrect pairings across the dataset. One sequence alone therefore cannot answer the question — unless something in how the pairs were built leaked information one-sidedly, for instance if common strains ended up disproportionately in positive pairs. This test shows that did not happen: essentially all of the 0.9159 score comes from relating the two sides, which is also why the mixed HA/NA top-15 list in step 6 (8 HA sites, 7 NA sites) is meaningful rather than a coincidence.
   - This test does NOT rule out memorisation — a model recalling "this exact HA goes with this exact NA" also needs both sides to do that. It rules out a one-sided shortcut only.

   **7b(i). Shuffle one site at a time, no retraining — DONE (2026-09-02).**
   - Method: in each fold's held-out test split, shuffle one column's values across the rows, keep the already-fitted model unchanged, re-predict, and record the AUC-ROC drop. Repeated 5 times per column and averaged; 1,037 columns x 4 folds runs in about 1.5 minutes. Row-level shuffling is correct here because the model is fixed and scores one row at a time — it has no way to notice that the same sequence now carries different values in different rows. (That changes once the model is retrained — see 7b(iii).)
   - Added to `plot_site_importance.py`, so gain, SHAP, and this permutation score all sit in one table.
   - The top-ranked sites hold up under this test too: HA site 544 is rank 1 on all three measures, costing 0.0288 AUC when shuffled. NA sites 284 and 244 cost 0.0219 and 0.0218. SHAP and permutation share 12 of the top 15 sites for HA and 11 of 15 for NA.
   - Rank correlation between SHAP and permutation over ALL 1,037 columns is only +0.555 (HA) / +0.536 (NA), much weaker than the +0.97 between SHAP and gain — but this is a fact about the tail of the list, not the head. Most columns have no measurable effect when shuffled, so their permutation ranks are noise, and a whole-list correlation is dominated by that noise. The overlap at the top of the list is the number that matters.
   - **No single site is essential on its own, and the signal is spread across many sites.** Baseline AUC-ROC is 0.9547, i.e. 0.4547 above the chance level of 0.5.

     | | value |
     |---|---|
     | sites whose shuffle costs more than 0.001 AUC | 43 of 1,037 |
     | sites whose shuffle costs more than 0.005 AUC | 10 |
     | largest single-site cost | 0.0288 (6.3% of the total signal) |
     | sum of every single-site cost | 0.2523 (55.5% of the total signal) |
     | sum of just the top 10 sites' costs (ranked by SHAP) | 0.1388 (30.5%) |

   - Shuffling the single most important site removes only 6.3% of what the model knows. Adding up every site's individual cost reaches only 55.5% — meaning almost half of what the model uses is invisible when sites are tested one at a time, because sites can substitute for each other. That gap is exactly what the group test in 7b(ii) measures.
   - This also bears on memorisation: a model that was memorising specific sequences would be expected to depend heavily on a small number of very informative positions. This model does not depend heavily on any single position.

   **7b(ii). Shuffle the top N sites together, no retraining — DONE (2026-09-02).**
   - `src/analysis/plot_site_group_permutation.py`. Same idea as 7b(i), but N columns are shuffled together instead of one at a time; the model still is not retrained.
   - Two groups compared at every N: the top N sites by SHAP, and N sites drawn at random (the control).
   - 10 set sizes x 2 arms x 4 folds x 5 repeats, measured on both test and train.
   - Reported as "share of the signal lost" = (clean AUC − shuffled AUC) / (clean AUC − 0.5), which puts test (clean AUC 0.9547) and train (clean AUC 0.9859) on the same scale.

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

   - Sanity check: shuffling every column (N=1,037) removes 0.98-1.00 of the signal, i.e. AUC-ROC ~0.5, in both arms — expected, since at N=1,037 "top" and "random" are the same set of columns.
   - **The top sites are far more important as a group than random sites.** 10 well-chosen sites remove about half the signal; 10 random sites remove essentially none (0.008). It takes roughly 500 random sites — about half of all 1,037 columns — to lose as much signal as just the top 10.
   - **Shuffling sites together costs more than shuffling them one at a time and adding it up — this is the answer to whether sites substitute for each other.** From 7b(i), adding up the individual costs of the top 10 sites gives 30.5% of the signal. Shuffling all 10 together gives **49.5%** — 1.6x more. So the top sites were substituting for each other, and testing them one at a time understated every one of them. The redundancy the single-site test exposed is therefore partly INSIDE the top set, not only spread across the rest of the sites.
   - **This does not look like memorisation.** If the model had learned training-specific detail through these positions, shuffling them should hurt the training score more than the test score. Instead it hurts training LESS at every N tested. In raw AUC terms the top 10 sites are worth almost the same on both splits — 0.225 on test, 0.222 on train — so the model's higher score on training data (0.9859 vs. 0.9547 on test) is not coming from these particular sites.
   - **Shuffling disables a column more thoroughly than filling it with a constant value.** Replacing a column with its single most common value, instead of shuffling it, loses 0.08-0.15 less signal at N=10 and N=50, on both splits — filling with the most common value still lets most rows go down the tree branch they would normally take, while shuffling actively puts wrong values everywhere. Measured directly, not assumed.

   **7b(iii). Shuffle the top N sites together, then RETRAIN — DONE (2026-09-02).**
   - `src/analysis/plot_site_retrain_ablation.py`. Same idea as 7b(ii), but the model is refit from scratch on the corrupted data (same settings as the normal baseline), instead of staying fixed.
   - Two ways to corrupt the columns, because they test different things:
     - **row** — shuffle values across rows independently in each split, so a single sequence can end up with different values at the same site in different rows.
     - **sequence** — shuffle values across UNIQUE sequences, so each sequence gets one consistent (but wrong) value everywhere it appears, in train, val and test alike.
   - Both modes corrupt train, val and test together. Corrupting only train would leave test with real values the model was trained to ignore, which would confound the result.
   - 7 set sizes x 2 modes x 2 arms x 4 folds; 116 model fits total, about 6 minutes.

   | N | row, top | row, random | sequence, top | sequence, random |
   |---|---|---|---|---|
   | 1 | 0.004 | 0.000 | 0.006 | -0.000 |
   | 5 | 0.042 | -0.004 | 0.035 | 0.003 |
   | 10 | **0.159** | 0.008 | **0.126** | 0.005 |
   | 25 | 0.333 | 0.004 | 0.255 | 0.002 |
   | 50 | 0.561 | 0.015 | 0.409 | 0.008 |
   | 100 | 0.892 | 0.020 | 0.865 | 0.017 |
   | 1,037 | 1.007 | - | 0.994 | - |

   - Sanity check: corrupting all 1,037 columns still removes essentially all the signal in both modes (1.007 and 0.994, i.e. AUC-ROC ~0.5).
   - **The information is recoverable elsewhere — a retrained model can partly work around losing the top sites.** With the ORIGINAL fitted model (7b-ii), shuffling the top 10 sites cost 49.5% of the signal. After retraining on the corrupted data, the cost falls to **15.9%** — about two-thirds of that loss is recovered from the other 1,027 columns. The same pattern holds at N=25 and N=50. Only at N=100 do the fixed-model and retrained-model results agree (0.892 both ways): once 100 positions are gone, there is nothing left elsewhere to recover from.
   - This means "the top 10 sites hold half the signal" (from step 6 / 7b-ii) is a statement about THAT fitted model, not about where the information lives. The information is spread widely; a boosted tree concentrates on a few positions because that is a cheap way to fit the data, not because the rest are uninformative.
   - The random-site control stays flat after retraining too: 100 random sites cost only 1.7-2.0%, against 86-89% for the top 100. The ranking is still picking out something real.
   - **Sequence-level corruption is consistently milder than row-level corruption** — 0.126 vs. 0.159 at N=10, 0.409 vs. 0.561 at N=50, converging by N=100. A plausible reading: row-level corruption turns a column into pure noise within one sequence, so a retrained model can learn to drop it; sequence-level corruption still gives the model one consistent (if wrong) value per sequence, which it can still use to tell that sequence apart from others.
   - **That gap between the two corruption modes is smaller than the framing this plan used to give it, and it does not, by itself, separate memorisation from real signal.** Two explanations both fit: (1) the model is recognising specific training sequences, or (2) the corrupted column still correlates with which sequence it is, and sequence identity itself correlates with lineage, which is real signal. This test alone cannot choose between them. 7c is the check that can — it is directly bounded already by the numbers in step 5: only 7-10% of test rows have both slots seen in training, and `pair_key` overlap is 0.

   **7c. Compare scores on sequences seen in training vs. never seen — DONE (2026-09-02). Memorisation is not what carries the result.**
   - `src/analysis/plot_seen_sequence_effect.py`. The model is left completely alone; the test rows are split instead, by whether each slot's sequence also appears somewhere in that fold's training split.
   - Reads the already-saved `test_predicted.csv` files, so any feature source (k-mer, per-site nt, per-site codon) can be compared on the exact same rows.
   - Checked first, before anything else: zero `pair_key` overlap between train and test in every fold. So this test is strictly about whether the INDIVIDUAL SEQUENCES were seen before, never about whether the exact PAIR (the answer) was seen before.

   | arm | all rows | neither seen | slot a seen | slot b seen | both seen |
   |---|---|---|---|---|---|
   | k-mer k=6 | 0.9564 | 0.9493 | 0.9608 | 0.9528 | 0.9769 |
   | per-site nt | 0.9597 | **0.9544** | 0.9620 | 0.9594 | 0.9746 |
   | per-site codon | 0.9547 | 0.9489 | 0.9622 | 0.9511 | 0.9701 |
   | rows, all folds | 7,160 | 2,953 | 1,348 | 2,261 | 598 |

   - **The result holds even when neither sequence was ever seen during training.** On those 2,953 rows, per-site nt scores 0.9544 AUC-ROC against its overall average of 0.9597 — only 0.005 lower. Nothing collapses when recall is impossible.
   - **Per-site features rely on having seen a sequence before LESS than k-mers do, not more.** The gap between "both sequences seen" and "neither seen" is +0.0276 for k-mer, +0.0202 for per-site nt, and +0.0213 for per-site codon. The concern flagged earlier in this plan was the opposite — that a per-site vector could almost identify its exact sequence while a k-mer count could not, so per-site should be MORE prone to relying on recall. That is not what was measured: k-mer shows the largest seen-advantage of the three.
   - **The per-site advantage over k-mer from step 5 is not explained by memorisation.** Per-site nt (0.9544) still leads k-mer (0.9493) and per-site codon (0.9489) on the hardest rows, where neither sequence was ever seen in training.
   - Caveat on how to read the "both seen" numbers: having seen HA_x paired with NA_y in training helps the model correctly reject a test negative that wrongly pairs HA_x with some other NA — but it works against the model on a test positive where HA_x's real partner is different from what it saw in training. So +0.02 to +0.03 is the net of both effects pulling in different directions, not a clean measure of memorisation alone.

8. **Interactions** (only if steps 5-7 look sound).
   - Needs either SHAP interaction values or LightGBM's split-pair statistics.
   - Computing every feature's main-effect SHAP value together is already cheap — step 6 measures it at about 0.3 seconds per fold, via `pred_contrib` — but computing INTERACTION values costs `n_features` times more work per row. At 1,037 codon features that is a different order of magnitude, so this should be budgeted and planned as its own piece of work.

## Open questions for Jamie

Checked against the 2026-05-12 chat in `notes.md`.

1. **Did you filter to complete CDS, or did your GenBank pull already contain only complete
   records?** Asked twice, never directly answered. She said "mostly a single length" early and
   "the single length holds for all segments" later. Our data has 4.7% of NA off-length, nearly all
   incomplete records — so either the datasets differ or something filtered them. She sampled 300
   unique sequences per season; that sampling may have picked complete ones.
2. **Did you compare ordinal codes against one-hot?** Not covered in the chat.

Already answered: masking meant dropping the columns, not shuffling — "I would just exclude those
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
  and scores 0.9094 +/- 0.0145. Compare against that, not against 0.9177.
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
data before reaching the end. The letters that are there are correct — there are just fewer of them
than the whole CDS has.

A stop in the MIDDLE is a different problem, not a version of this one. A short record shifts every
position after the cut, which is exactly what breaks per-site features. A mid-sequence stop shifts
nothing — it means the read is bad or that copy of the CDS is non-functional. In this data version
there are zero of them: 0 of 868,240 rows in `cds_dna_final` and 0 of 1,793,563 in `protein_final`.
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
| does it have a proper end? | ends with `*` | ends with a stop group | 99.999% |
| is there a stop in the middle? | `*` in the middle | stop group in the middle | **100%** |

Same answer essentially always. The six exceptions are worth knowing about: their last DNA group is
`TAR`, where `R` means "this letter is either A or G — the sequencing wasn't sure". Either way it is
a stop, so the protein correctly says `*`, but a literal DNA text match does not recognise it.

So the two checks mean the same thing, and **the protein version is the more reliable one** — it
copes with uncertain letters, the DNA text match does not.

---

## Known weak points in Stage 1 / 1.5 (audit, 2026-09-01)

Found while auditing `preprocess_flu.py` and `extract_cds_dna.py` before step 0. None is firing
today. Each is recorded here because the evidence took a while to gather and is easy to lose.

The audit's headline result: `extract_cds_dna.py` reproduces its archived output exactly —
868,240 rows, all 11 columns byte-identical, zero rows dropped. Joins, location parsing,
coordinates and translation are correct on this corpus. Also verified: `brc_fea_id` unique in both
outputs, `(assembly_id, function)` unique in both, no nulls in any critical column, `prot_hash` and
`cds_dna_hash` equal to md5 of their sequences, `assembly_id -> file` 1:1, and no duplicate
`(assembly_id, genbank_ctg_id)` contigs — which matters because `extract_cds_dna` builds a dict on
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
