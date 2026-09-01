# Per-site nucleotide features for HA-NA segment matching

**Status: IN PROGRESS**

## Goal

Find out which parts of the sequence drive the match/no-match prediction. K-mer counts throw
position away, so a k-mer feature importance score cannot be traced back to a place in the CDS.
Encoding each position as its own feature keeps that link, so feature importance becomes a map
along the CDS.

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

| unit | reads | sites per pair | values | ordinal width | one-hot width |
|---|---|---|---|---|---|
| `nt` | CDS DNA | 3,111 | 4 | 3,111 | 12,444 |
| `codon` | CDS DNA | 1,037 | 64 | 1,037 | **66,368** |
| `aa` | protein | 1,037 | 21 | 1,037 | 21,777 |

(Current k-mer setup is 8,192 wide.) Note codon is narrower than nt only under `ordinal`; under
`onehot` it is the widest of the three, because each site needs 64 columns.

`nt_ctg` (contig DNA) is not a valid source for any unit: contigs include the untranslated ends and
vary in length, so positions do not line up. Measured on H3N2 2024 segment 4, contigs span
1,672-1,762 with the most common length covering 58%, against 99.7% for the CDS.

Order of work: start with `nt` (finest view, and the direct comparison to Jamie's results), then
add `codon` and `aa` on the same positions. Comparing those two says whether silent changes carry
any signal, which is nearly free once the machinery exists.

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

| | unique CDS | complete CDS | complete and at modal length |
|---|---|---|---|
| HA | 2,792 | 2,785 | 2,785 (length 1701) |
| NA | 2,415 | 2,306 | 2,301 (length 1410) |

"Complete" means: starts `ATG`, ends in a stop group, no stop in the middle. (A fourth test,
length divisible by 3, passes on 100% of rows and so is dropped — see Background.)

The off-length sequences are almost all incomplete records, not real length variation. Of 7
off-modal HA, none is a complete CDS. Of 114 off-modal NA, only 5 are. The rest are cut off at one
end (90 NA missing the stop, 17 missing the start). No sequence in either segment has an internal
stop, so nothing is frameshifted.

**Filter yield.** Keeping only pairs where both slots are complete and at modal length leaves
**3,580 of 3,723 positive pairs (96.2%)**.

**Feature width.** 1701 + 1410 = **3,111** features per pair under `ordinal`, **12,444** under
`onehot`. For comparison the current k-mer setup uses 8,192 (2 x 4096).

**Codons.** Across all 8 proteins in `cds_dna_final`, 98.6-99.8% of unique CDS start with `ATG`,
and all three standard stop codons are used. Each segment prefers one stop codon (e.g. M1 is 98%
`TGA`, PA is 97% `TAG`), so the filter must accept all three.

**Only 8 of the 18 functions in `protein_final` appear in `cds_dna_final`.** These are the 8 major proteins (one per segment).

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

**Reuse the split, not the dataset.** The 2024 folds
(`dataset_ha_na_h3n2_2024_random_cv4`) were built before the completeness filter existed, so they
contain pairs this plan drops. Rebuild the dataset with the filter, then re-run the k-mer baseline
on the new folds so both feature types see the same population. That re-run is 4 x ~35 s.

**The split stays random.** A cluster-disjoint split is not available for a single year: at t099
one NA cluster holds 94.6% of the pairs, so `max_balanced_k` is 1
(`cc_nt_cds_cm0_h3n2_2024/HA-NA/t099/cc_summary.json`).

## Steps

0. **Preprocessing prerequisite** (do this first; it is the only change outside the per-site work).
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

1. **Completeness filter.** Filter on `is_complete_cds` in the front end (the per-experiment stage,
   not preprocessing) and rebuild the 2024 dataset. Record how many sequences and pairs it removes. Re-run the k-mer LGBM baseline on the new folds; the current
   number on the unfiltered folds is F1 macro 0.9177 +/- 0.0086.
2. **Entropy map.** Stack the kept sequences into a matrix (rows = sequences, columns = positions)
   and compute Shannon entropy down each column. Two purposes: a conservation map along each CDS,
   and a check that positions are comparable. If the sequences were not truly aligned, entropy
   would be high and flat across the whole length. Note this catches wholesale misalignment, not
   one or two shifted sequences — the completeness plus equal-length filter is what rules those
   out, since an internal shift would need an insertion and a deletion that cancel.
3. **Feature builder.** `src/embeddings/compute_site_features.py`, matching the existing
   `compute_esm2_embeddings.py` and `compute_kmer_features.py` in that directory. Write
   a cache keyed by `cds_dna_hash`, same pattern as the k-mer cache. Verify a few sequences decode
   back to the original.
4. **Loader and training.** Add a `site` branch to `src/models/_pair_features.py`, which today
   rejects anything outside `{kmer, esm2}` (line 326). Wire `categorical_feature` into the LGBM
   baseline, which does not pass it today. Reject any `interaction` other than `concat` and any
   `slot_transform` other than `none`.
5. **Train and compare.** LGBM on the new folds, against the re-run k-mer number. Report both.
6. **Importance map.** Per-position importance along each CDS, read against the entropy map from
   step 2.
7. **Masking and shuffling.** Retrain with the top-ranked positions removed, and separately with
   their values shuffled between isolates. If the score holds up, the model was not relying on
   those positions. If it collapses, they carry the signal. Shuffling is the better control of the
   two because it keeps the feature count fixed.
8. **Interactions** (only if steps 5-7 look sound). Pairwise interaction strength needs SHAP
   interaction values or LightGBM split-pair statistics. At ~3,100 features an all-pairs pass is
   expensive, so this needs its own design; do not assume it comes for free.

Stop after step 5 if per-site features do not at least match k-mers. The point is interpretability,
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

- **Memorisation.** A per-site vector is close to a fingerprint: with ~1,700 positions, a handful
  of them usually identify a sequence exactly. K-mer counts blur this. Under a random split the
  same sequences appear in train and test, so the model can score well by memorising which HA goes
  with which NA. Step 7 is the check; without it a win over k-mers cannot be interpreted.
- **Filter changes the population.** The completeness filter drops 3.8% of pairs, so results are
  not directly comparable to any earlier 2024 number until the k-mer baseline is re-run (step 1).
- **One year, one subtype.** Nothing here shows the importance map generalises to other years or
  subtypes. Treat it as a description of H3N2 2024.

---

## Background: what "complete" means and why we filter on it

### The background you need

A CDS (coding sequence) is the stretch of DNA that codes for one protein. The cell reads it 3
letters at a time, and each group of 3 becomes one letter of a protein. So a 1,410-letter CDS makes
a 469-letter protein, plus one group at the end that means "stop here" (stop codon).

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
So the flag costs nothing to keep and guards future data versions, but it is not what makes the
positions line up. If one ever appears, drop the record.

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

Also, the extracted CDS records are 3 times longer than their equivalent protiens records.
