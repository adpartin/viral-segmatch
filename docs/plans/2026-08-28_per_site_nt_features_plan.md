# Per-site nucleotide features for HA-NA segment matching

**Status: IN PROGRESS**

## Goal

Find out which parts of the sequence the model uses to decide match or no-match. A k-mer count
records how many times a subsequence occurs but not where, so a k-mer importance score cannot be
traced back to a place in the CDS. One feature per position keeps the position, so importance can
be reported per position along the CDS.

Scope: HA-NA, H3N2, 2024. Idea and prior results from Jamie Overbeek (see `notes.md`, chat of
2026-05-12), who used the same features with a random forest to predict collection date.

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
   `site_importance_codon.png` and `site_importance_codon.csv` (column, slot, protein, site,
   gain_frac, gain_frac_std, folds_used, split_count, entropy_bits, n_values, rank).

   Score is LightGBM gain, normalised per fold before averaging -- early stopping gives the folds
   411 to 998 trees, so raw gain is not comparable across them. A site's number is its share of
   the model's total gain.

   **The model uses few positions.** Slot A (HA) holds 57.2% of total gain, slot B (NA) 42.8%.
   Within each, the top 10 sites hold 46.4% (HA) and 53.6% (NA) of that protein's gain, and the
   top 50 hold 79.5% and 86.6%. Only 344 of 567 HA sites and 253 of 470 NA sites get any gain at
   all, against 97.5% and 96.6% that vary.

   **The ranking is stable enough to read.** Fold-to-fold Spearman on gain is 0.730 for HA
   (0.711-0.746 across the six fold pairs) and 0.701 for NA; 8 of the top 15 HA sites and 7 of 15
   NA sites are in every fold's top 15, and every site in both top-15 lists is used by all four
   folds. Individual ranks past the top few move, so read the head of the list, not its order.

   Top sites: HA 544, 36, 129, 239, 286, 87, 531; NA 310, 244, 284, 24, 462, 223.

   **Variability is necessary, not sufficient.** Spearman(gain, entropy) over varying sites is
   +0.485 (HA) and +0.529 (NA) -- positive, since an invariant column cannot separate anything,
   but far from 1. The scatter shows the shape: every top site sits at 0.5-1.0 bits, while most
   sites in that same range contribute nothing. So conservation bounds importance and does not
   predict it, which is what makes the map worth having.

   **What this does not yet say.** A handful of sites holding half the gain is equally consistent
   with those positions carrying real lineage signal and with their being the most efficient way
   to identify a sequence -- the memorisation risk. Step 7 separates them.
7. **Masking and shuffling.** Retrain with the top-ranked positions removed, and separately with
   their values shuffled between isolates. If the score is unchanged, the model was not using those
   positions. If it drops sharply, those positions carry the signal. Shuffling is the better control
   of the two because the feature count stays the same.
8. **Interactions** (only if steps 5-7 look sound). Pairwise interaction strength needs SHAP
   interaction values or LightGBM split-pair statistics. At 1,000-3,100 features depending on unit,
   an all-pairs pass is costly in time and memory, so budget for it as its own piece of work.

Stop after step 5 if per-site features do not at least match k-mers. The goal is interpretability,
but a feature set that scores clearly worse is not worth interpreting.

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

- **Memorisation.** With ~1,700 positions per segment, a small number of them is usually enough to
  identify one sequence exactly, so a per-site vector nearly names the sequence it came from. K-mer
  counts do not, because many sequences give the same counts. Under a random split the same
  sequences appear in train and test, so the model can score well by memorising which HA goes with
  which NA. Step 7 is the check; without it a higher score than k-mers cannot be interpreted.
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
