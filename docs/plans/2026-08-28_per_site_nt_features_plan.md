# Per-site nucleotide features for HA-NA segment matching

**Status: IN PROGRESS**

## Goal

Find out which parts of the sequence the model uses to decide match or no-match. A k-mer count
records how many times a subsequence occurs but not where, so a k-mer importance score cannot be
traced back to a place in the CDS. One feature per position keeps the position, so importance can
be reported per position along the CDS.

Scope: HA-NA, H3N2, 2024. Idea and prior results from Jamie Overbeek (see `notes.md`, chat of
2026-05-12), who used the same features with a random forest to predict collection date.

## What we found (steps 0-7 done; step 8 open)

Scope throughout: HA-NA, H3N2, collected 2024, four random-split folds, LGBM. Every arm runs on
the same folds, so the comparisons are paired.

**Per-site features work, at a third of the width.** F1 macro 0.9192 +/- 0.0134 for `nt` (3,111
columns) against 0.9094 +/- 0.0145 for k-mer k=6 (8,192). `nt` wins all four folds but at k=4 that
is p=0.128, so the honest claim is "matches", not "beats". `codon` is indistinguishable from `nt`
(p=0.509) at 1,037 columns and is the efficient choice. [step 5]

**Silent changes carry most of the signal — the largest effect in the whole study.** `codon` and
`aa` are the same 1,037 positions on the same records and differ only in whether a change that
leaves the protein unchanged is visible. Removing those costs **0.107 F1 macro**, every fold,
p=0.002. The matching signal lives in the DNA, not the protein. That says nothing about ESM-2,
which reads 1,280 continuous dimensions rather than 22 categories per site. [step 5]

**The model leans on few positions, but the information is spread widely.** The top 10 codon sites
hold 34% of SHAP and, with the model fixed, half the signal. Refit after corrupting them and the
cost falls from 49.5% to 15.9% -- two thirds comes back from the other 1,027 columns. So "the top
10 hold half the signal" describes this fitted model, not where the information lives: a boosted
tree is greedy. [steps 6, 7b(ii), 7b(iii)]

**Three checks that could have invalidated the interpretation, all passed.**
- One side alone scores at chance: AUC-ROC 0.5007 (HA only) and 0.4979 (NA only) against 0.9547
  with both. No one-sided shortcut in how pairs were built, so the mixed HA/NA top-15 is
  meaningful. [7a]
- Corrupting random sites costs nothing: 1.7-2.0% at N=100 against 86-89% for the top 100. The
  ranking picks out something real. [7b(ii), 7b(iii)]
- Memorisation is not carrying the result, and per-site leans on recall LESS than k-mers do. On
  the 2,953 test rows whose sequences never appear in training, per-site `nt` scores 0.9544
  against its overall 0.9597. The seen-minus-unseen gap is +0.0202 for per-site `nt` and +0.0276
  for k-mer. [7c]

**The positions genuinely line up**, which everything above depends on. Third codon positions are
2.8x more variable than first and 3.8x more than second, the expected order under selection.
Shifting each sequence by a random 0-2 nt destroys that ordering and raises mean entropy 19-fold,
so the check has teeth. [step 2]

**Three importance measures agree at the top.** SHAP and gain correlate at +0.97 with 12 of 15 top
sites shared; permutation shares 12 of 15 with SHAP. Split count -- what `lightgbm.plot_importance`
shows by default -- ranks differently and is reported for contrast, not used. [step 6]

## Naming

Not "positional encoding" — in ML that means the sinusoidal position signal added to token
embeddings in a transformer, which is a different thing. Published work splits this into two
choices, and we follow that split:

- `feature_source: site` — one feature per position. A position in an alignment is a **site**;
  "per-site" is the standard term (also used as "column" in the MSA literature).
- `site.unit: nt | codon | aa` — what one site is.
- `site.encoding: ordinal | onehot` — how each site's value is coded. Both names are standard.

The config block is `site:` at top level, matching the existing `kmer:` block that goes with
`feature_source: kmer`. Same word in both places.

Add the terms to `docs/methods/glossary.md` before using them anywhere else.

### Which unit to use

`codon` and `aa` are the SAME positions, not just the same count: 3 x protein length = CDS length
exactly (HA 1,701 nt = 567 codons = 567 protein characters; NA 1,410 = 470 = 470). Codon site 200
and amino-acid site 200 are the same place. Only the value differs — the 3-letter DNA group (64
possible) or the protein letter it becomes (21). So the choice between them is one thing: **codon
keeps silent changes, aa discards them.**

| unit | reads | sites per pair | codes | ordinal width | one-hot width |
|---|---|---|---|---|---|
| `nt` | CDS DNA | 3,111 | 5 | 3,111 | 15,555 |
| `codon` | CDS DNA | 1,037 | 65 | 1,037 | **67,405** |
| `aa` | protein | 1,037 | 22 | 1,037 | 22,814 |

(Current k-mer setup is 8,192 wide.) Note codon is narrower than nt only under `ordinal`; under
`onehot` it is the widest of the three, because each site needs one column per codon.

The code counts include one catch-all, which the plain alphabet sizes (4 / 64 / 21) leave out: a
site whose character is an IUPAC ambiguity code has no clean value. Rare -- 0.006% of nt sites and
0.010% of aa sites on the built cache -- but it is a real code, and a one-hot expansion needs a
column for it. The aa count is 22, not 21, for the same reason: 20 residues, the stop character,
and `X` (residue unknown) sharing the catch-all.

`nt_ctg` (contig DNA) is not a valid source for any unit: contigs include the untranslated ends and
vary in length, so positions do not line up. Measured on H3N2 2024 segment 4, contigs span
1,672-1,762 with the most common length covering 58%, against 99.7% for the CDS.

Order of work: start with `nt` (finest view, and the direct comparison to Jamie's results), then
add `codon` and `aa` on the same positions. Comparing those two says whether silent changes carry
any signal, and needs only a second feature cache once the loader exists.

**Answered (step 5): they carry most of it.** `codon` scores 0.9159 F1 macro against `aa`'s
0.8091 -- same positions, same records, +0.107 on every fold, p=0.002. `nt` and `codon` are
indistinguishable (p=0.509), so `codon` is the efficient choice at a third of `nt`'s width.

### Codon numbering

Use GenSLM's codon-to-integer map. `genslm_vocab/tokenizer_config.json` has all 64
codons at ids 33-96, five special tokens (`<cls>` 0, `<pad>` 1, `<eos>` 2, `<unk>` 3, `<mask>` 32),
tokenizer class `EsmTokenizer`, max length 2,048 tokens. The order is NOT alphabetical — it starts
GGC, GCC, ATC, GAC and puts the three stops last (93, 95, 96), which looks like frequency order.

For our own models the numbering does not matter, since the column is declared categorical and the
model treats the values as unordered labels. It matters only if we later feed sequences to GenSLM
for embeddings, where the ids must be theirs. Adopting their map now avoids
rebuilding the feature cache later. Two things to confirm against their loader first: whether each
sequence needs the `<cls>` / `<eos>` wrappers, and whether input must be upper-cased (ours is stored
lowercase). Length is not a constraint — 567 and 470 codons against GenSLM's 2,048 context window.

## What we already know (measured 2026-08-28)

Counts are unique CDS in H3N2 + 2024 from `cds_dna_final.parquet`.

**The sequences are nearly all one length per segment, once incomplete records are removed.**

| | unique CDS | complete CDS | complete and at the pinned length |
|---|---|---|---|
| HA | 2,792 | 2,785 | 2,785 (length 1701) |
| NA | 2,415 | 2,306 | 2,301 (length 1410) |

"Complete" means: starts `ATG`, ends in a stop group, no stop in the middle. (A fourth test,
length divisible by 3, passes on 100% of rows and so is dropped — see Background.)

The off-length sequences are almost all incomplete records, not real length variation. Of 7
off-length HA, none is a complete CDS (6 missing the stop, 1 missing both ends). Of 114
off-length NA, only 5 are (90 missing the stop, 17 missing the start, 2 missing both). No sequence in either
segment has an internal stop, so nothing is frameshifted.

**Filter yield.** Keeping only pairs where both slots are complete and at the pinned length leaves
**3,580 of 3,723 positive pairs (96.2%)**.

**Codons.** Across all 8 proteins in `cds_dna_final`, 98.6-99.8% of unique CDS start with `ATG`,
and all three standard stop codons are used. Each segment prefers one stop codon (e.g. M1 is 98%
`TGA`, PA is 97% `TAG`), so the filter must accept all three.

**Only 8 of the 18 functions in `protein_final` appear in `cds_dna_final`** — the 8 majors, one per
segment.

## Design decisions

**Only `interaction: concat` is valid.** The two slots live in different spaces: slot A is HA
position 1..1701, slot B is NA position 1..1410. HA position 500 and NA position 500 have nothing
to do with each other, and the vectors are not even the same length. `diff`, `unit_diff` and `prod`
are therefore meaningless here and must be rejected, not silently allowed. `slot_transform` must be
`none` for the same reason — normalising a vector of category codes has no meaning.

**Start with LGBM and declared categoricals.** Ordinal codes are nominal, not ordered, so the model
must be told. LightGBM supports this through `categorical_feature`; sklearn's random forest does
not, which is why Jamie's ordinal codes were treated as ordered. Keep `onehot` as the fallback if
declared categoricals underperform.

**Rebuild the dataset, then re-run the k-mer baseline on it.** The 2024 folds
(`dataset_ha_na_h3n2_2024_random_cv4`) were built before the filter existed, so they contain pairs
this plan drops — nothing about them can be reused. Once rebuilt, the old k-mer number describes a
different population, so it has to be re-run on the new folds for the comparison to mean anything.
Done in step 1: the rebuild takes ~55 s and the four LGBM folds ~5 s each.

**The split stays random.** A cluster-disjoint split is not available for a single year: at t099
one NA cluster holds 94.6% of the pairs, so `max_balanced_k` is 1
(`cc_nt_cds_cm0_h3n2_2024/HA-NA/t099/cc_summary.json`).

## Steps

0. **Preprocessing prerequisite — DONE (2026-09-01).** Implemented and verified; the sub-points
   below record what was built. Outcome: `protein_final` 1,793,563 -> 1,793,572 rows (+9), one new
   column, every pre-existing row byte-identical on every shared column; `ctg_dna_final`
   unchanged; `cds_dna_final` 868,240 rows unchanged with four new columns, 855,695 complete
   (98.56%). The +9 are exactly the rows the old duplicate key removed — see the last sub-point.
   - Add `starts_with_m` in Stage 1, next to `has_terminal_stop` and `has_internal_stop` in
     `src/utils/protein_utils.py`. This is the one part of completeness nothing records today, and
     it belongs on the protein because the protein version is the more reliable of the two (see
     Background).
   - Carry all three flags into `cds_dna_final` in Stage 1.5. That file already holds `prot_seq`,
     so this copies a value rather than recomputing it — same rule the repo already uses for hashes.
   - Add one derived column in Stage 1.5 for convenience:

         is_complete_cds = starts_with_m & has_terminal_stop & ~has_internal_stop

     Keep the three underlying flags as well, so an experiment can use a different combination.
     Per-site features strictly need only the first two — an internal stop does not shift positions
     — but the combined column is what step 1 filters on.
   - **Flag, do not drop.** Preprocessing is shared across every experiment, so a record dropped
     here is lost to experiments that would have been fine with it (k-mer features never line
     positions up, so they do not care about completeness). Recording a fact and letting each
     experiment apply its own rule is also what Stage 1 already does with `has_terminal_stop`,
     `has_ambiguities`, `x_count_ratio` and the rest. The existing drops in `extract_cds_dna.py`
     are a different case and stay: those rows failed extraction, so there is nothing to record.
     The rule is — drop when there is no data to record, flag when there is. Avoid naming a column
     "invalid": that is a verdict that depends on the use, not a fact about the record.
   - Re-run both stages and diff the output against
     `data/processed/flu/July_2025/archive_09_01_2026/`: same row count, same columns plus the new
     flag, identical values everywhere else. Do not proceed until that passes.
   - Output layout. Top level keeps five files — `protein_final`, `ctg_dna_final`, `cds_dna_final`,
     and the two GTO aggregates `protein_agg_from_GTOs` / `genome_agg_from_GTOs` (a cache
     `preprocess_flu.py` reads back by fixed path, so their names and location must not change).
     The remaining ~13 report files move to `preprocess_qc_20260901/`. A few analysis scripts read
     some of those reports, so update their paths in the same change.
   - Do NOT merge `extract_cds_dna.py` into `preprocess_flu.py`, and do not rename the two
     aggregates. Keeping the names makes the archive diff a straight file-for-file comparison.
   - **Also fixed while here** (found by auditing Stage 1): `handle_assembly_duplicates` keyed on
     `[prot_seq, assembly_id]`, which collapses two DIFFERENT products of one segment when they
     share a sequence — 334 such groups exist in the raw corpus (PB1 with PB1-N40, PA with
     PA-N155/PA-N182, HA with its mature subunit, NS3 with NEP). `function` is now part of the key.
     No group contains two of the 8 majors, so nothing important was ever at risk, but the
     protection came from an unrelated filter running first. Stage 1 now reports "No duplicates
     found": every removal this function made on this corpus was a cross-function one.

1. **Filter — DONE (2026-09-01).** Two conditions, not one. `is_complete_cds` does not by itself
   give equal length: 5 NA sequences are complete but off-length (1407 x3, 1413, 1416). Length is
   a property of a population, not of a record, so it cannot live in preprocessing. Both
   conditions are applied in the dataset builder, on the protein rows, before pairs are made.

   Outcome on H3N2 2024: HA 2,792 unique CDS -> 2,785 (7 incomplete, 0 off-length); NA 2,415 ->
   2,301 (109 incomplete, 5 complete but off-length). Protein rows 10,964 -> 10,787; unique
   positive pairs 3,723 -> 3,580 (96.2%). Every CDS in the built folds is now one length — HA
   went from 5 distinct lengths to 1, NA from 15 to 1, both 100% complete. That is what the step
   exists to guarantee. The folds carry 2,732 unique HA and 2,298 unique NA rather than the
   2,785 / 2,301 kept, because 169 isolates hold only one of the two proteins and never enter a
   pair.

   K-mer LGBM re-run on the new folds: F1 macro **0.9094 +/- 0.0145**, against 0.9177 +/- 0.0086
   on the unfiltered folds. The 0.008 drop is smaller than the fold spread, so the filter does
   not change what k-mers can do on this population; the wider spread is what 4% fewer pairs
   buys. 0.9094 is the number per-site features have to beat.

   - **Config.** `dataset.require_complete_cds_at_pinned_length`, default false, in
     `conf/dataset/default.yaml`. Off by default because the filter drops rows: always-on would
     change every nt_cds dataset built before it existed and make those results irreproducible.
   - **Not "canonical" in a name.** The repo already uses that word for two other things —
     `canonical_segment` (the segment label) and `canonical_pair_key` (the order-invariant
     dedup key, `'__'.join(sorted([hash_a, hash_b]))`) — so a third meaning in an identifier
     would be ambiguous. "Pinned" is the word already used for this idea, in
     `check_cds_length(..., pinned_nt)` and in the `flu.yaml` comment. In prose "canonical
     length" is unambiguous and stays, which is why `conf/virus/flu.yaml` and
     `check_cds_length`'s error text still say it. (A third use, the
     `canonicalize_pair_orientation` config flag, was removed after this: it was v1-only and
     v1 went on 2026-06-03.)
   - **Implementation.** `_pair_helpers.filter_complete_cds_at_pinned_length`, called from
     `dataset_segment_pairs.py` just before the `cds_dna_hash` attach. It matches on
     `(assembly_id, function)` membership rather than a merge, so a duplicate key in
     `cds_dna_final` cannot silently multiply protein rows. `dataset_pairs_cc.py` is NOT wired
     yet: the 2D-CD builder ignores the flag.
   - **Where the length comes from.** `conf/virus/flu.yaml` `cds_length` (HA 1701, NA 1410), not
     the most common length per run — a per-run value can differ between populations and quietly
     make two importance maps non-comparable. `src.utils.cds_utils.check_cds_length` re-derives
     the most common length from the complete CDS this run actually loaded and raises if it
     disagrees with the pin, or if the pin holds for under 90% of them. Both guards were tested
     and fire: PB1 (no pinned length) and H5N1 HA (real length 1704 against a 1701 pin). The
     table is scoped to H3N2 and H1N1; PB1 and NS1 are absent because neither has one length
     across subtypes and years.
   - **Bundle.** `flu_ha_na_h3n2_2024_random_cv4_pinned_length`, inheriting the unfiltered
     bundle plus the flag. Kept separate rather than switching the flag on in place, so the
     0.9177 result stays reproducible from its own recipe.
   - **Regression.** Rebuilding the unfiltered dataset with the modified driver reproduces the
     existing run byte-identically across all 12 fold splits, so the change is inert when the
     flag is off.
2. **Entropy map — DONE (2026-09-01).** `src/analysis/plot_site_entropy.py`. Stacks the kept
   sequences into a matrix (rows = unique CDS, columns = positions) and computes Shannon entropy
   down each column. Writes `site_entropy.png` and one `site_entropy_{SHORT}.csv` per protein to
   `results/flu/July_2025/dataset_ha_na_h3n2_2024_random_cv4_pinned_length/site_entropy/`; step 6
   reads the CSV against the importance map.

   Unique CDS, not pair rows -- a heavily sampled strain would otherwise decide the answer. All
   splits, because nothing is fitted. If entropy is ever used to *select* positions, it must be
   recomputed on train alone.

   **Conservation.** HA: 2,732 unique CDS, mean 0.0577 bits, 550 of 1,701 positions invariant
   (32.3%). NA: 2,298 unique CDS, mean 0.0580 bits, 506 of 1,410 invariant (35.9%). Against a
   ceiling of 2 bits for four bases, both are strongly conserved with isolated variable sites.

   **The positions line up.** Two checks, and the second is the sharper one:

   | | mean bits | 1st | 2nd | 3rd | 3rd/1st |
   |---|---|---|---|---|---|
   | HA as built | 0.0577 | 0.0383 | 0.0283 | 0.1065 | 2.78x |
   | NA as built | 0.0580 | 0.0376 | 0.0281 | 0.1084 | 2.88x |
   | HA, each sequence shifted 0-2 nt | 1.0971 | 1.0970 | 1.0981 | 1.0963 | 1.00x |

   Third codon positions are the most variable and second the least, in both proteins -- the
   expected order, since most third-base changes are silent and most second-base changes are not.
   That ordering is what says the reading frame is right; a flat entropy trace would not.

   The last row is the negative control: shifting each sequence by a random 0-2 nt, so the
   positions no longer correspond, raises mean entropy 19-fold and flattens the codon-position
   ordering to 1.00x. The diagnostic has teeth. (Shuffling each column independently instead
   leaves the numbers identical to four decimals, as it must -- entropy is per column.)

   This catches wholesale misalignment, not one or two shifted sequences. The completeness plus
   pinned-length filter is what rules those out, since an internal shift would need an insertion
   and a deletion that cancel.
3. **Feature builder — DONE (2026-09-01).** `src/embeddings/compute_site_features.py`, alongside
   `compute_esm2_embeddings.py` and `compute_kmer_features.py`. Config group `conf/site/default.yaml`
   (`unit`, `encoding`), registered in `conf/bundles/flu_base.yaml` next to `/kmer`. The four terms
   are in `docs/methods/glossary.md`.

   Per protein and unit it writes `site_features_{unit}_{SHORT}.npz` (uint8 codes),
   `_index.parquet` (`cds_dna_hash` -> row) and `_metadata.json` (code map, site count, kept and
   dropped counts) to the embeddings dir. Existence-check caching per protein, `--force_recompute`
   to rebuild.

   **One matrix per protein, not one for the corpus.** The width is the CDS length, which differs
   by protein, so a single matrix cannot hold them. Only complete CDS at the pinned length take
   part.

   **The cache stores ordinal codes only.** `site.encoding: onehot` is expanded at load time
   (step 4), so switching encoding does not rebuild the cache. Storing one-hot would be 5-65x
   larger for nothing.

   **Keyed by `cds_dna_hash` for every unit, `aa` included.** Two CDS that translate to the same
   protein get two identical aa rows. That costs a little space and buys one join key and one row
   order across all three units -- so codon site *i* and aa site *i* are the same place for the
   same row, by construction rather than by convention.

   Built for HA and NA in all three units:

   | | unique CDS | complete | at pinned length | nt sites | codon/aa sites |
   |---|---|---|---|---|---|
   | HA | 65,414 | 64,125 | 44,202 | 1,701 | 567 |
   | NA | 58,887 | 57,278 | 46,175 | 1,410 | 470 |

   37 MB on disk for all six matrices, against 140 MB uncompressed for `nt` alone. 15 s to build
   both proteins for one unit.

   HA loses 19,923 complete CDS (31%) to the length filter, because the corpus spans subtypes the
   pin does not cover -- H5N1 HA is 1704 nt, H9 and H7 are 1683. That is correct, not a bug: those
   sequences cannot be placed position by position against a 1,701-nt frame. A per-site run on
   them needs its own pinned length. The H3N2 2024 dataset is unaffected: the cache covers all
   2,732 HA and 2,298 NA hashes it needs.

   **Verification.** Every build decodes a sample of rows back to the source and raises on any
   mismatch, so a wrong code map cannot reach a model silently; the catch-all code is checked the
   other way round, since those positions cannot round-trip. Three further checks, run once:

   - The `nt` codes rebuild the codon ids exactly -- 0 mismatches over 400 rows per protein.
   - Each codon translates to its amino-acid code under NCBI table 1 -- 0 mismatches. This is an
     independent path: the codon ids come from GenSLM's tokenizer, the aa codes from `prot_seq`,
     and the translation from `cds_utils._CODON_TABLE_1`. All three agree.
   - The three units share one index, and nt sites == 3 x codon sites == 3 x aa sites.

   Codon ids are GenSLM's, read from `genslm_vocab/tokenizer_config.json` at build time rather
   than copied: GGC 33, GCC 34, ATC 35, GAC 36, stops 93 / 95 / 96, `<unk>` 3.
4. **Loader and training — DONE (2026-09-02).** `src/utils/site_utils.py` reads the cache
   (sibling of `kmer_utils.py`); `src/models/_pair_features.py` gained the `site` branch it used
   to reject; `train_pair_baselines.py` resolves the cache dir and the two slot proteins;
   `baselines/lgbm.py` takes `categorical_feature`. Bundle
   `flu_ha_na_h3n2_2024_random_cv4_site_nt` inherits the pinned-length dataset bundle and only
   swaps the features, so a difference against the k-mer number is the feature set and not the
   population.

   **Categoricals are declared.** Ordinal codes are labels, not magnitudes -- code 7 is not "more"
   than code 3, and without the declaration LightGBM splits on `<=` and reads an order that is not
   there. Under `encoding: ordinal` every column is one site, so every column is categorical; the
   fitted booster confirms all 3,111. One-hot columns are already binary and are left numeric.
   Every other feature source passes `None`, which becomes LightGBM's own `'auto'`.

   **Widths match the table above exactly**, measured on fold 0: nt 3,111 ordinal / 15,555
   one-hot; codon 1,037 / 67,405; aa 1,037 / 22,814. One-hot rows sum to the site count, so
   exactly one code fires per site. One-hot width comes from the code map the cache declares, not
   from the values a split happens to hold, so train, val and test cannot come out different
   widths.

   **Which column is which position** is written to
   `site_feature_columns_{unit}_{encoding}.csv` at load time: column, slot, protein, site and (for
   one-hot) code. Column 0 is HA site 1, column 1700 is HA site 1701, column 1701 is NA site 1.
   Step 6 reads this instead of re-deriving the layout.

   **Guards.** Twelve, all tested and firing: `interaction` other than `concat`, `slot_transform`
   other than `none`, `feature_scaling` other than `none`, missing `site_dir`, missing or
   malformed `site_proteins`, slots given in the wrong order, a protein with no cache, an unknown
   `feature_source`, a `cds_dna_hash` the cache does not hold, the two slots built with different
   units, a pair table without the hash columns, and an unknown encoding. The wrong-order one
   matters most: nothing else ties the short names the cache is addressed by to the full function
   strings the pair table carries, so without it a run could featurize NA into slot A and never
   say so.

   **Regression.** The k-mer baseline on fold 0 reproduces to six decimals after the change, so
   the shared loader and the new `categorical_feature` keyword are inert for the other sources.

   Smoke result, fold 0 only: site nt F1 macro 0.9246 against k-mer 0.9219 on the same fold. One
   fold decides nothing; step 5 is the comparison.
5. **Train and compare — DONE (2026-09-02).** LGBM on the four pinned-length folds, all arms on
   the same folds so the differences are paired.

   | arm | columns | F1 macro | F1 | AUC-ROC |
   |---|---|---|---|---|
   | k-mer k=6 (nt_cds) | 8,192 | 0.9094 +/- 0.0145 | 0.9159 +/- 0.0122 | 0.9564 +/- 0.0064 |
   | per-site `nt` | 3,111 | **0.9192 +/- 0.0134** | 0.9239 +/- 0.0121 | 0.9597 +/- 0.0087 |
   | per-site `codon` | 1,037 | 0.9159 +/- 0.0087 | 0.9211 +/- 0.0077 | 0.9547 +/- 0.0056 |
   | per-site `aa` | 1,037 | 0.8091 +/- 0.0228 | 0.8257 +/- 0.0185 | 0.8842 +/- 0.0200 |

   Paired on F1 macro across the four folds:

   | comparison | mean difference | folds won | p |
   |---|---|---|---|
   | `nt` vs k-mer | +0.0098 | 4/4 | 0.128 |
   | `codon` vs k-mer | +0.0065 | 3/4 | 0.397 |
   | `codon` vs `aa` | **+0.1067** | 4/4 | **0.002** |
   | `nt` vs `codon` | +0.0033 | 3/4 | 0.509 |

   **Per-site at least matches k-mers, so the plan continues.** `nt` wins every fold, but at k=4
   the paired test does not reach significance (p=0.128), so the honest reading is "matches", not
   "beats" -- and it does it with 3,111 columns against 8,192, or 1,037 for `codon`, which is
   indistinguishable from `nt` (p=0.509) at a third of the width.

   **Silent changes carry most of the signal.** `codon` and `aa` are the same 1,037 positions on
   the same records and differ in one thing: whether a change that leaves the protein unchanged is
   visible. Removing those costs **0.107 F1 macro**, on every fold, p=0.002 -- far the largest
   effect in the comparison, an order above the k-mer gap. Synonymous sites are nearly neutral, so
   they drift with lineage and act as a lineage fingerprint; amino-acid changes are under
   selection and can converge across lineages. This says the matching signal lives in the DNA and
   not in the protein, at least at the resolution a per-site categorical column can see. It does
   NOT say the same about ESM-2, which reads 1,280 continuous dimensions rather than 22 categories
   per site; that comparison has not been run.

   **The memorisation risk is bounded but not settled.** Under a random split some sequences recur
   across splits, and a per-site vector nearly names the sequence it came from. Measured on these
   folds: 18-21% of test HA sequences and 22-26% of test NA also appear in train, but only
   **7-10% of test rows have BOTH slots seen in training**, and `pair_key` overlap is 0 in every
   fold -- no test pair was trained on. So at most about a tenth of the test set could be answered
   by recalling a combination. The `aa` result is consistent with memorisation and with real
   signal alike, since collapsing silent variants removes identities and signal together, so it
   does not separate the two. Step 7 is what does.
6. **Importance map — DONE (2026-09-02).** `src/analysis/plot_site_importance.py`, on the `codon`
   arm: 1,037 columns, statistically indistinguishable from `nt` at a third of the width, and one
   column per amino-acid position, so a site number is a residue number. Writes
   four files: `site_importance_codon.png` (importance along the CDS, plus importance against
   entropy), `site_importance_codon_barplot.png` (LightGBM's own `plot_importance` bar charts for
   `split` and `gain` beside the held-out SHAP ranking), `site_importance_codon.csv` (column, slot,
   protein, site, shap_frac, shap_frac_std, gain_frac, gain_frac_std, folds_used, split_count,
   entropy_bits, n_values, shap_rank, gain_rank) and `site_importance_codon_per_fold.csv` (each
   fold's own shares, so the fold agreement can be recomputed). Both figures carry the producing
   script and the run name.

   **Split is the third measure, and it ranks differently.** `lightgbm.plot_importance` defaults
   to split -- how often a feature was used -- and on fold 0 its top list holds NA 154, HA 162,
   HA 325, NA 233 and NA 349, none of which appear in the gain or SHAP top 15. Counting splits
   rewards a position consulted in many shallow splits; it does not say what the position was
   worth. Shown for comparison, not used for ranking.

   **Two importance measures, not one.** Gain is a training-time quantity read off the fitted
   trees, so on its own it says what the trees were built on rather than what they are worth on
   new data. SHAP is exact TreeSHAP through LightGBM's own `pred_contrib` -- no extra dependency,
   and the script checks that contributions plus base value reconstruct the raw margin rather than
   assuming it -- measured on each fold's **held-out test split**. The split count that
   `lightgbm.plot_importance` shows by default is in the CSV; counting splits says how often a
   position was consulted, not what it was worth.

   **The two agree, so the ranking survives an out-of-sample check.** Spearman(SHAP, gain) is
   +0.971 for HA and +0.974 for NA, with 12 of the top 15 sites shared. Where they differ, believe
   SHAP.

   Two differences worth knowing. Gain **overstates concentration**: the HA top 10 holds 34.1% of
   SHAP against 46.4% of gain, and the top 50 hold 65.3% against 79.5%. And gain **undersells the
   many-valued sites** -- HA 161 (10 codon values) is gain rank 35 but SHAP rank 9, HA 363 is 69
   and 13, NA 400 (11 values) is 22 and 3, NA 385 (10 values) is 116 and 11. Note this is the
   opposite of the usual warning that split-gain favours high-cardinality features; measured here,
   it does the reverse. Gain also splits the two slots differently: HA 57.2% / NA 42.8% by gain
   against 52.3% / 47.7% by SHAP.

   Both measures are normalised per fold before averaging -- early stopping gives the folds 411 to
   998 trees, so raw totals are not comparable across them. A site's number is its share of the
   model's total.

   **The model uses few positions.** The HA top 10 sites hold 34.1% of HA's SHAP and the top 50
   hold 65.3%; for NA, 47.8% and 75.4%. Only 344 of 567 HA sites and 253 of 470 NA sites get any
   gain at all, against 97.5% and 96.6% that vary.

   **The ranking is stable enough to read.** Fold-to-fold Spearman on SHAP is 0.715 for HA
   (0.699-0.731 over the six fold pairs) and 0.681 for NA; 9 of the top 15 sites are in every
   fold's top 15 for both proteins, and every site in both top-15 lists is used by all four folds.
   Individual ranks past the head move, so read the list as a set, not an order.

   Top sites by SHAP: HA 544, 36, 129, 239, 531, 286, 87, 451, 161; NA 284, 310, 400, 244, 223,
   24, 140.

   **Variability is necessary, not sufficient.** Spearman(SHAP, entropy) over varying sites is
   +0.516 (HA) and +0.579 (NA) -- positive, since an invariant column cannot separate anything,
   but far from 1. The scatter shows the shape: every top site sits at 0.5-1.0 bits, while most
   sites in that same range contribute nothing. So conservation bounds importance and does not
   predict it, which is what makes the map worth having.

   **What this does not yet say.** A handful of sites holding half the gain is equally consistent
   with those positions carrying real lineage signal and with their being the most efficient way
   to identify a sequence -- the memorisation risk. Step 7 separates them.
7. **Masking and shuffling — DONE (2026-09-02), in four passes.** The original plan was to retrain
   with the top-ranked positions removed and separately with their values shuffled. That grew into
   four checks, because the first attempts kept answering a narrower question than the one asked:
   7a one side alone, 7b(i) one site at a time, 7b(ii) the top N together, 7b(iii) the same with a
   refit, and 7c seen against unseen sequences.

   **7a. One side alone — DONE (2026-09-02), passes.** Run first, because it decides whether the
   importance map is worth interpreting at all. `site.slots: a | b | both` (`conf/site/default.yaml`,
   default `both`) keeps one slot's columns and drops the other; bundles
   `flu_ha_na_h3n2_2024_random_cv4_site_codon_slot_{a,b}`.

   | arm | columns | F1 macro | AUC-ROC | precision |
   |---|---|---|---|---|
   | both slots (HA+NA) | 1,037 | 0.9159 +/- 0.0087 | 0.9547 +/- 0.0056 | 0.8708 |
   | slot a only (HA) | 567 | 0.4942 +/- 0.0086 | **0.5007 +/- 0.0076** | 0.4996 |
   | slot b only (NA) | 470 | 0.4819 +/- 0.0214 | **0.4979 +/- 0.0084** | 0.4992 |

   Chance, to three decimals, on both sides: per-fold AUC-ROC spans 0.4914-0.5113 and precision
   sits on the 0.50 base rate. Read AUC-ROC here, not F1 -- F1 is not centred on 0.5, so a model
   that guesses "match" often still scores 0.48-0.59 while learning nothing.

   Why it matters. The label is a fact about a PAIR, and the negative sampler pairs one isolate's
   segment with someone else's, so the same sequence appears in both matched and mismatched rows.
   One sequence alone therefore cannot answer the question -- unless the pair construction leaked
   something one-sided, for instance sequences from heavily sampled strains landing
   disproportionately among positives. It did not. Every bit of the 0.9159 comes from relating the
   two sides, which is what makes the mixed top-15 (8 HA, 7 NA) meaningful rather than incidental.

   This does not address memorisation: a model recalling "this HA goes with this NA" also needs
   both sides. It rules out one-sided leakage only.

   **7b(i). Permutation, one site at a time — DONE (2026-09-02).** In each fold's held-out test
   split, one column's values are shuffled among the rows, the SAME fitted model re-predicts, and
   the AUC-ROC drop is recorded. Nothing is retrained. 5 shuffles per column, averaged; 1,037
   columns x 4 folds takes about 1.5 minutes. Added to `plot_site_importance.py` alongside SHAP
   and gain, so all three sit in one table.

   Row-level shuffling is right for this pass: the model is fixed and predicts one row at a time,
   so it cannot notice that a sequence appearing in several rows now carries different values in
   each. That changes for 7b(ii), which retrains.

   **The top sites hold up.** HA 544 is rank 1 on all three measures, costing 0.0288 AUC. NA 284
   and NA 244 cost 0.0219 and 0.0218. SHAP and permutation share 12 of the top 15 sites for HA and
   11 of 15 for NA.

   Their rank correlation over all 1,037 columns is only +0.555 (HA) and +0.536 (NA), against
   +0.97 between SHAP and gain -- but that is a fact about the tail, not the head. Most columns
   have no measurable drop, so their permutation ranks are noise, and a whole-list Spearman is
   dominated by it. Read the top overlap.

   **No single position is load-bearing, and the signal is redundant.** Against a clean AUC-ROC of
   0.9547, i.e. 0.4547 above chance:

   | | value |
   |---|---|
   | sites whose shuffle costs > 0.001 AUC | 43 of 1,037 |
   | sites whose shuffle costs > 0.005 AUC | 10 |
   | largest single-site cost | 0.0288 (6.3% of the signal) |
   | all single-site costs added up | 0.2523 (55.5%) |
   | top 10 by SHAP, added up | 0.1388 (30.5%) |

   Scrambling the single most important position costs 6.3% of what the model knows. Every
   single-site cost added together reaches 55.5%, so nearly half the signal is invisible to
   one-at-a-time removal -- positions cover for each other. That gap is what a group ablation
   measures and this pass cannot, which is the case for 7b(ii).

   It also bears on memorisation. A model identifying sequences would lean hard on a few
   high-resolution positions; this one does not lean hard on any.

   **7b(ii). The top N shuffled together, no retrain — DONE (2026-09-02).**
   `src/analysis/plot_site_group_permutation.py`. Same idea as 7b(i) but N whole columns at once,
   still with the fitted model. Two arms at every N: the top N by SHAP, and N sites drawn at
   random as the control. 10 set sizes x 2 arms x 4 folds x 5 repeats, on test and on train.
   Reported as share of the signal lost, `(clean AUC - shuffled AUC) / (clean AUC - 0.5)`, so
   train and test sit on one scale despite different clean scores (test 0.9547, train 0.9859).

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

   The anchor holds: shuffling every column loses 0.98-1.00 of the signal, i.e. AUC-ROC 0.5. Both
   arms meet there, since "top 1,037" and "random 1,037" are the same set.

   **The top sites are special, by a wide margin.** Ten well-chosen sites cost half the signal;
   ten random sites cost nothing (0.008). It takes about 500 random sites -- half of all columns --
   to reach what the top 10 do.

   **The group is worth more than the sum of its parts, which settles the masking question.**
   Shuffling the top 10 one at a time and adding up the drops gives 30.5% of the signal (7b(i)).
   Shuffling the same 10 together gives **49.5%** -- 1.6x more. So the top sites were covering for
   each other and single-site permutation understated every one of them. The redundancy the
   single-site pass exposed is therefore partly INSIDE the top set, not only spread across the
   tail.

   **Nothing here looks like memorisation.** If the model had fitted those positions to
   training-specific detail, shuffling them would hurt train more than test. It hurts train
   *less* at every N. In absolute AUC the two are the same: the top 10 are worth 0.225 on test and
   0.222 on train. The model's train-to-test gap (0.9859 against 0.9547) is not located in the
   sites it leans on.

   **Constant fill understates, so shuffling is the right choice.** Replacing a column with its
   most common value instead of scrambling it loses 0.08-0.15 less signal at both N=10 and N=50,
   on both splits. Filling with the mode sends every row down the branch most rows already take;
   shuffling actively puts wrong values in, which disables the column more thoroughly. Measured
   rather than argued.

   **7b(iii). Corrupt, then refit — DONE (2026-09-02).**
   `src/analysis/plot_site_retrain_ablation.py`. N columns are corrupted in train, val and test
   alike and the model is fitted from scratch on them, using the same estimator and fit as the
   baselines. Two corruption modes and two arms, 7 set sizes, 4 folds; 116 refits in about 6
   minutes. The anchor holds in both modes: corrupting all 1,037 columns loses 0.99-1.01 of the
   signal.

   | N | row, top | row, random | sequence, top | sequence, random |
   |---|---|---|---|---|
   | 1 | 0.004 | 0.000 | 0.006 | -0.000 |
   | 5 | 0.042 | -0.004 | 0.035 | 0.003 |
   | 10 | **0.159** | 0.008 | **0.126** | 0.005 |
   | 25 | 0.333 | 0.004 | 0.255 | 0.002 |
   | 50 | 0.561 | 0.015 | 0.409 | 0.008 |
   | 100 | 0.892 | 0.020 | 0.865 | 0.017 |
   | 1,037 | 1.007 | - | 0.994 | - |

   **The signal is redundant: a fresh model recovers most of what the top sites carried.** Take
   the top 10. With the model fixed, scrambling them costs 49.5% of the signal (7b(ii)). Refit
   afterwards and the cost falls to **15.9%** -- two thirds of the loss comes back from the other
   1,027 columns. The same holds at 25 and 50. Only by N=100 do the two agree (0.892 against
   0.892): once a hundred positions are gone there is nothing left to recover from.

   So "the top 10 sites hold half the signal" is a statement about THIS fitted model, not about
   where the information lives. The information is spread widely; the model concentrates on a few
   positions because a boosted tree is greedy, not because the rest are uninformative.

   **The random control stays flat.** Corrupting 100 random sites costs 1.7-2.0% after refitting,
   against 86-89% for the top 100. The ranking picks out something real.

   **Sequence-level corruption is consistently gentler than row-level** -- 0.126 against 0.159 at
   N=10, 0.409 against 0.561 at N=50, converging by N=100. Row-level makes a column noise within a
   sequence, so a refit model discards it; sequence-level leaves each sequence one consistent
   wrong value, which a refit model can still read as a (re-coded) property of that sequence.

   That gap is smaller than the framing this plan used to give it, and it does not cleanly separate
   memorisation from signal. Two readings fit: the model recovers by identifying sequences it saw
   in training, or the corrupted column still correlates with sequence identity and so, indirectly,
   with lineage. Nothing here chooses between them. The seen-versus-unseen test is what would --
   split the test rows by whether their sequences appear in training and compare scores. Only
   7-10% of test rows have both slots seen and pair_key overlap is 0 (step 5), which already bounds
   how much memorisation could be worth.

   **7c. Seen against unseen sequences — DONE (2026-09-02), memorisation is not what carries the
   result.** `src/analysis/plot_seen_sequence_effect.py`. The model is left alone; the test rows
   are split instead, by whether each slot's sequence also appears in that fold's training split.
   Reads the saved `test_predicted.csv`, so any feature source can be compared on the same rows.
   `pair_key` overlap between train and test is 0 in every fold, checked before anything else, so
   this is about sequences and never about repeated answers.

   | arm | all | neither seen | slot a seen | slot b seen | both seen |
   |---|---|---|---|---|---|
   | k-mer k=6 | 0.9564 | 0.9493 | 0.9608 | 0.9528 | 0.9769 |
   | per-site nt | 0.9597 | **0.9544** | 0.9620 | 0.9594 | 0.9746 |
   | per-site codon | 0.9547 | 0.9489 | 0.9622 | 0.9511 | 0.9701 |
   | rows, all folds | 7,160 | 2,953 | 1,348 | 2,261 | 598 |

   **The result holds on sequences the model has never seen.** On the 2,953 rows where neither
   sequence appears in training, per-site nt scores 0.9544 against its overall 0.9597 -- a
   difference of 0.005. Nothing collapses when recall is impossible.

   **Per-site features memorise LESS than k-mers, not more.** The gap between "both seen" and
   "neither seen" is +0.0276 for k-mer, +0.0202 for per-site nt and +0.0213 for per-site codon.
   The risk this plan flagged was that a per-site vector nearly names its sequence while k-mer
   counts do not, so per-site would lean on recall. Measured, the k-mer baseline shows the larger
   seen-advantage of the three.

   **The ranking survives on the hardest rows.** per-site nt (0.9544) still leads k-mer (0.9493)
   and codon (0.9489) where neither sequence was ever seen, so per-site's edge in step 5 is not
   bought with recall.

   Caveat on direction: having seen HA_x with NA_y helps reject a test negative pairing HA_x with
   something else, but works against a test positive where HA_x has a different true partner. The
   +0.02 to +0.03 is the net of the two, not a pure memorisation term.

8. **Interactions** (only if steps 5-7 look sound). Pairwise interaction strength needs SHAP
   interaction values or LightGBM split-pair statistics. Main-effect SHAP is already cheap --
   step 6 measures it in 0.3 s per fold through `pred_contrib` -- but interaction values are
   `n_features` times more work per row, so at 1,037 codon features that is a different scale.
   Budget for it as its own piece of work.

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

- **Starts** with `ATG` in the DNA, which becomes `M` as the first protein letter.
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
