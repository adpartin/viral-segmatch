# Alphabet-specific pair_key — scoping plan

**Status: IN PROGRESS** (scoping only; no implementation pending agreement on scope)

Source: BACKLOG.md "Methodology ideas — possible paper contributions" #3.
Inline TODO: `docs/methods/splits.md` § 2.2.

This plan scopes the change. It does NOT design the implementation or
authorize any code/data work. Adoption requires explicit agreement on
items in § 6 (decision criteria) and § 7 (couplings).

---

## 1. The problem

Today, `pair_key` is constructed exclusively from PROTEIN sequence hashes:

```python
# src/datasets/_pair_helpers.py:36
def canonical_pair_key(seq_hash_a: str, seq_hash_b: str) -> str:
    return "__".join(sorted([seq_hash_a, seq_hash_b]))
```

`seq_hash` is `md5(prot_seq)` (Stage 1 invariant; see CLAUDE.md
"Sequence hashes"). Every downstream surface — v2 within-input
dedup, cluster-disjoint routing input, MMD pair construction,
negative-sampling forbidden set, leakage audit (protein level) —
operates on this protein-only universe regardless of which
cluster alphabet (`aa`, `nt_cds`) the routing is using or which
feature space (ESM-2 protein embedding, aa k-mer, nt k-mer) the
model trains on.

Concrete consequence on Flu A HA-NA: ONE shared pair universe of
**58,826 canonical pairs**, regardless of alphabet. Two distinct
synonymous-codon CDS variants encoding the same HA protein paired
with the same NA protein collapse to a single pair_key. The
splitter, the dedup logic, and the audit all see one pair where
the nt-feature model sees two distinct training rows (different
k-mer vectors).

The proposal: derive `pair_key` from the hash family appropriate to
the model's view of "what is a unique pair":

| Alphabet view | Hash column source | Pair-key construction |
|---|---|---|
| Protein (aa) | `seq_hash` (Stage 1) | `canonical(seq_hash_a, seq_hash_b)` (current) |
| nt CDS | `cds_dna_hash` (Stage 1.5, `attach_cds_dna_hash_to_pos_df`) | `canonical(cds_dna_hash_a, cds_dna_hash_b)` |
| nt full contig | `dna_hash` (Stage 1, `attach_dna_to_prot_df`) | `canonical(dna_hash_a, dna_hash_b)` |

Three universes, three sizes. The protein universe is always a lower
bound on the others (one protein → ≥1 CDS → ≥1 contig); the gap
quantifies how much synonymous-codon and UTR/intronic variation
collapses under the current convention.

---

## 2. Why this matters (bias direction)

Pair-key dedup propagates into every "what counts as a unique pair"
decision the pipeline makes:

1. **v2 positive-pair dedup** (`dataset_segment_pairs_v2.py:248`):
   `drop_duplicates(subset=['pair_key'])`. Under protein-only, a HA
   seq with 10 synonymous CDS variants paired with one NA seq
   contributes 1 pair (the CDS variants merge); under nt-CDS, it
   contributes 10. **Bias: nt training is undersampling its own
   variation.** Verified empirically (memory.md "Level-0 multiplicity
   does NOT skew class balance under v2 pipeline, 2026-05-31"):
   correlation copy_count vs n_pairs is sub-linear because copies of
   the same HA seq tend to pair with the same NA across isolates,
   and they all collapse under the protein pair_key. Under nt-CDS
   pair_key, that collapse weakens.

2. **Negative-sampling forbidden set** (`dataset_segment_pairs_v2.py:592, 598`):
   `cooccur_pairs` and `forbidden_pair_keys` are protein-keyed. A
   sampled negative `(cds_dna_hash_a', cds_dna_hash_b')` that's
   protein-equivalent to a positive will be rejected by the
   protein-keyed forbidden set. Under nt-CDS pair_key, those
   "synonymous-of-positive" negatives become legal — more genuine
   negatives, possibly easier negatives (depending on regime).

3. **Cluster-disjoint splitter** (`_split_helpers.py::cluster_disjoint_route_pos_df`):
   The atoms (1D-CD: per-slot cluster; 2D-CD/2D-CD-test: bipartite
   CC) are derived from a JOIN of the pair universe against the
   cluster parquet on `pos_hash_col` (default `seq_hash`, override
   `cds_dna_hash` for nt routing). The JOIN ITSELF is already
   alphabet-aware. **What's NOT alphabet-aware is the pair universe
   feeding the JOIN** — pair-universe rows are dedupped on protein
   pair_key first, so the atoms cover a protein-collapsed view of
   the corpus even under `cluster_alphabet=nt`.

4. **MMD pair-set construction** (`mmd_per_pair.py:115`): already
   parametrized by hash column names (`hash_a_col`, `hash_b_col`),
   so the per-pair MMD calculation itself is alphabet-flexible. But
   the pair-universe input is the same protein-collapsed table — so
   even the alphabet-aware MMD is measuring the protein-collapsed
   pair set.

5. **Leakage audit** (`audit_split_leakage.py:78-79`): already
   computes BOTH `prot_key` and `dna_key` levels and reports against
   both. **Strongest precedent that alphabet-specific pair_key is
   tractable** — the audit script's `_canonical` helper duplicates
   `canonical_pair_key` logic precisely to enable the nucleotide-
   level check. Promoting nt pair_key from "audit-only diagnostic"
   to "first-class universe" is the proposal.

The current convention is **biased toward aa-feature training**: the
pair count matches what aa training sees (each unique protein-pair
is one training row), but undercount what nt training sees. Trained-
model interpretation has implicitly assumed this match; the bias is
silent in published numbers.

---

## 3. Inventory of pair_key code surfaces

| Site | File:line | Current behavior | Proposed change | Risk |
|---|---|---|---|---|
| Canonical helper | `_pair_helpers.py:36` | One function on protein hashes | Generalize to `canonical_pair_key(hash_a, hash_b)` (already accepts any strings; only docstring needs update) OR add `canonical_pair_key_dna()` siblings | Low — fn is hash-agnostic; signature already general |
| v2 vectorized column build | `dataset_segment_pairs_v2.py:205-212` | `seq_hash_{a,b}` → `pair_key` | Select hash source by config | Medium — touches the hot path of every Stage 3 run |
| v2 within-input dedup | `dataset_segment_pairs_v2.py:248` | `drop_duplicates(subset=['pair_key'])` | Same operation, different universe size | Low — mechanical |
| v2 cooccur set | `_pair_helpers.py::compute_cooccur_pairs (line 458)` | Builds protein-keyed `cooccur_pairs` for neg-sampling | Build alphabet-keyed equivalent | Medium — couples to negative sampler |
| v2 neg-sampling | `dataset_segment_pairs_v2.py:566, 591-598` | Checks candidate pair_key against protein-keyed sets | Switch keys to match alphabet of forbidden set | Medium — must stay consistent with cooccur |
| Cluster lookup attach | `_split_helpers.py::attach_cluster_ids (line 60+)` | `pos_hash_col` param already chooses join column | No change — already alphabet-aware | None |
| seq_disjoint routing | `_pair_helpers.py::seq_disjoint_route_pos_df (line 603)` | `hash_key='seq'\|'dna'` already routes on chosen hash | No change — already alphabet-aware | None |
| bipartite_components | `_pair_helpers.py:496` | `hash_key='seq'\|'dna'` or explicit `col_a`/`col_b` | No change | None |
| cluster_disjoint route | `_split_helpers.py::cluster_disjoint_route_pos_df` | Operates on `(cluster_id_a, cluster_id_b)` derived from the JOIN | No change — atom derivation is downstream of pair-universe choice | None directly; **changes upstream when pair universe shifts** |
| MMD per-slot | `mmd_per_slot.py` | Operates on per-slot sequence features (hash-key independent at the slot level) | No change | None |
| MMD per-pair | `mmd_per_pair.py:115` | Parametrized by `hash_a_col`, `hash_b_col` | No change — already alphabet-aware | None |
| Leakage audit | `audit_split_leakage.py:78-79` | Already computes both protein and dna levels | No change — proposal makes the audit's nt level the primary universe rather than diagnostic | None |
| Cluster pair-weight rollup | `cluster_pair_weight_topk.py:131` | Protein-keyed | Add alphabet parameter | Low — analysis-only |
| Cluster-disjoint feasibility | `cluster_disjoint_feasibility.py` | References pair_key in pre-flight comment; operates on `pos_df` rows | Verify any pair_key dependency; likely just inherits universe choice | Low |
| Bipartite graph properties | `bipartite_graph_properties.py` | Builds graph from "pair_key-deduped pair universe" | Inherit universe choice from caller | Low |

**Surfaces summary**: 4 sites construct or dedup `pair_key`
(v1 + v2 builders + helper). ~8 sites consume the universe
downstream and would inherit the change passively. 2 sites already
have alphabet-aware machinery (`seq_disjoint`, `audit_split_leakage`).
The v2 builder is the structural choke point — change there
propagates everywhere.

**v1 is untouched** in this proposal: `dataset_segment_pairs.py` is
the legacy builder. If retained for any active bundle, it stays
protein-only. (Confirm with Alex: is v1 still in use? CLAUDE.md
says "v2 builder (default since 2026-05-11)" — v1 may be dead code.)

---

## 4. Universe-size deltas (measured 2026-06-02)

Measured by `/tmp/measure_pair_key_universes.py`: load `cds_final.parquet`
(seq_hash + cds_dna_hash) and `genome_final.parquet` (compute
`dna_hash = md5(dna_seq)`), inner-join on `assembly_id` per schema
pair, count distinct canonical pair_keys under each hash family.

| Schema pair | protein (seq_hash) | nt_cds (cds_dna_hash) | nt_ctg (dna_hash) | Δ nt_cds | Δ nt_ctg |
|---|---:|---:|---:|---:|---:|
| HA + NA | 58,826 | 79,347 | 83,364 | **+34.9 %** | **+41.7 %** |
| PB2 + PB1 | 53,078 | 83,331 | 86,424 | **+57.0 %** | **+62.8 %** |

Per-slot unique-hash inflation (for context):

| Slot | unique seq_hash | unique cds_dna_hash (×) | unique dna_hash (×) |
|---|---:|---:|---:|
| HA | 41,896 | 65,414 (1.56×) | 70,051 (1.67×) |
| NA | 37,488 | 58,887 (1.57×) | 64,329 (1.72×) |
| PB2 | 33,663 | 67,341 (2.00×) | 71,716 (2.13×) |
| PB1 | 31,226 | 67,034 (2.15×) | 71,243 (2.28×) |

**Observations**:

1. **HA+NA protein universe 58,826 matches the glossary baseline**
   exactly — validates the measurement methodology.
2. **Both schema pairs blow past the 10 % adoption threshold** in
   § 6 criterion 1. The bias is material, not negligible.
3. **PB2-PB1 has nearly 2× the nt_cds delta of HA-NA** (+57.0 %
   vs +34.9 %). Polymerase proteins are more conserved at the
   protein level (33,663 unique PB2 vs 41,896 unique HA on the
   same corpus) but carry MORE synonymous-codon variation per
   unique protein (2.00–2.15× nt-vs-aa inflation per slot, vs
   1.56–1.57× for HA/NA). PB2/PB1 nt experiments are
   currently undercounting their training-row diversity by ~37 %.
4. **The nt_ctg incremental delta over nt_cds is small** (+5
   pp on HA-NA, +4 pp on PB2-PB1). Most of the nt-vs-aa variation
   sits in the coding region (synonymous codons), not in UTR/intronic
   regions. This is methodologically interesting: it suggests
   `nt_ctg` would primarily add **assembly-artifact noise**
   (polyA, primer-trim variability per memory.md §117) on top of
   `nt_cds`, with modest biological signal gain. Reinforces § 7.2's
   tentative position to reserve the `nt_ctg` enum value without
   adopting it operationally in this migration.
5. **Side-A vs side-B asymmetry is small** within each pair —
   both slots inflate by roughly the same factor. Adopting
   alphabet-specific pair_key does not introduce a per-slot bias.

---

## 5. What gets re-validated under adoption

Pair_key change invalidates every artifact downstream of pair-
universe construction. Inventory of what would need regeneration:

### 5.1 Bundles (config files)
- HA-NA cluster_aa: 6 × HAonly (id100..id095), 6 × NAonly (if built),
  + 4 base bundles (`flu_ha_na.yaml`, `flu_ha_na_cluster_id99.yaml`,
  `flu_ha_na_random.yaml`, `flu_ha_na_seq_disjoint.yaml`).
- HA-NA cluster_nt: 2 (id100, id099).
- PB2-PB1 cluster_aa: 6 × PB2only (id100..id095) + 2 base.
- PB2-PB1 cluster_nt: 2.
- + experimental/H3N2/race bundles.
- **~25-30 bundles total** would need a new key
  (`split_strategy.pair_key_alphabet: aa|nt_cds|nt_ctg`) defaulting
  to `aa` for backward compatibility, or an explicit override.

### 5.2 Datasets (`data/datasets/flu/July_2025/runs/`)
- All cluster-disjoint dataset runs under the affected bundles.
- All seq_disjoint and random runs (the dedup step itself shifts).
- **~50-80 dataset runs** under the active bundle set.
- Existing runs stay readable (pair_key column still populated),
  but split partitions are no longer equivalent under the new key.

### 5.3 Trained models (`models/flu/July_2025/runs/`)
- Every trained MLP, LGBM, 1-NN baseline whose dataset would be
  regenerated. **~100+ run dirs** (single-slot HA-NA sweep alone is
  18 MLP + 12 baseline = 30 dirs).
- Holding pattern: leave existing runs in place; mark in
  `training_info.json` whether they're built under "protein
  pair_key" or "alphabet-specific pair_key" so retrospective
  comparisons can filter.

### 5.4 Published numerical findings
- All Key Findings in CLAUDE.md "Key Experimental Findings"
  derived from cluster-disjoint or seq_disjoint datasets carry
  implicit "under protein pair_key" caveat.
- The 2026-05-24 single-slot HA-only sweep + 2026-05-26 PB2-PB1
  PB2-only sweep — both rely on per-dataset pair counts that would
  shift under alphabet-specific pair_key.
- The 2026-06-01 cut-node analysis (mega-CC = 57,526 pairs at HA-NA
  aa id095) operates on the protein-pair multigraph; nt-CDS would
  give a multigraph with more parallel edges.
- **Decision**: do not re-run old experiments. New experiments
  conducted under the new convention; old experiments stay
  pinned with explicit "protein pair_key" labels.

### 5.5 MMD baselines
- The 2026-05-24 S1 + S2 MMD baselines (random / seq_disjoint /
  cluster_id099) sit on protein-keyed pair universes. Pair MMD
  values would shift under nt_cds pair_key (more pairs → different
  sample distributions).
- **Decision**: keep current baselines as reference; recompute
  baselines under nt_cds for any new sweep using nt pair_key.

### 5.6 Cluster artifacts
- Cluster parquets at `data/processed/flu/July_2025/clusters_{aa,nt}/idXXX/`
  are unaffected (they are mmseqs output, independent of pair_key).
- Cluster-pair-weight rollups, bipartite-graph-property reports,
  cluster_disjoint_feasibility CSVs ARE affected — they consume
  the pair universe.
- **~3-5 analysis re-runs** at HA-NA aa idXXX (cheap; ~10-30 s each).

### 5.7 Cluster sweeps (mmseqs runs)
- **Not affected by pair_key change.** Cluster parquets are
  alphabet-source-side artifacts, indexed by `seq_hash` (or by
  `cds_dna_hash` for nt clustering — see `clusters.md`). They live
  upstream of pair construction. No re-clustering needed for this
  change alone.

---

## 6. Decision criteria

Adopt **alphabet-specific pair_key** if all of:

1. **Empirical magnitude**: nt_cds universe ≥10 % larger than
   protein universe on at least one active schema pair.
   **STATUS: SATISFIED.** HA+NA: +34.9 %; PB2+PB1: +57.0 % (§ 4).
2. **Methodological cost** of NOT adopting is non-trivial: at least
   one publication-track finding is meaningfully affected by the
   bias (e.g., k-mer nt training metrics interpreted as comparable
   to k-mer aa training metrics when their pair universes silently
   differ in size).
   **STATUS: PROBABLY SATISFIED.** The 2026-05-15 Experiment B-nt
   ("1-NN cosine margin ≥ LGBM at every cluster_disjoint routing")
   compares nt vs aa routings on protein-collapsed universes; under
   alphabet-specific pair_key, the nt routing universes inflate
   by 35–57 % and the comparison becomes ambiguous. The
   2026-05-31 multiplicity-skew finding rests on the protein
   pair_key collapse mechanism explicitly. Adopting changes the
   interpretive frame of both results — needs explicit
   methodological discussion.
3. **Rebuild capacity**: ~1-2 weeks of compute + reviewer
   bandwidth to regenerate the active bundle/dataset set and
   re-establish the relevant baselines.
   **STATUS: TBD — Alex.**

Two of three criteria satisfied empirically. Decision point pivots
to (3) plus the couplings in § 7.

**Cheap intermediate** (no full adoption): add an optional
audit-only column `pair_key_nt_cds` to dataset writes,
parallel to `pair_key`. Costs ~50 LOC in v2, gives every
downstream a way to see the delta without changing routing
behavior. Lets the field decide whether the full migration
is justified after another quarter of experiments. Given the
measured delta magnitudes (§ 4), this intermediate is
likely undersized — the bias is large enough that audit-only
visibility may just generate pressure for the full migration
within a few weeks anyway.

**Cheap intermediate** (no full adoption): add an optional
audit-only column `pair_key_nt_cds` to dataset writes,
parallel to `pair_key`. Costs ~50 LOC in v2, gives every
downstream a way to see the delta without changing routing
behavior. Lets the field decide whether the full migration
is justified after another quarter of experiments.

---

## 7. Couplings

Three other changes touch the same artifact set (bundles,
datasets, models, docs). The question is: bundle into one
migration, or stage separately?

### 7.1 `idXXX` → `tXXX` notation migration

Per CLAUDE.md "Threshold notation" (adopted 2026-05-29, docs-only):
the canonical user-facing form is `tXXX` (e.g., `t095`);
code/dirs/bundles still use `idXXX`. Full code migration
deferred to "next cluster-sweep regeneration (option B)".

**Argument for bundling with pair_key migration**: pair_key
adoption regenerates ~25-30 bundles + ~50-80 datasets + ~100+
model dirs. Doing the `idXXX → tXXX` rename in the same migration
costs marginal extra effort (rename strings in templates) and
avoids a second cycle of cross-ref breakage.

**Argument against bundling**: two unrelated changes increase
the blast radius and complicate rollback. Pair_key migration is
a semantic change (different pair counts; different splits);
threshold notation is a cosmetic rename. Mixing them muddies the
commit history.

**Tentative position**: bundle, BUT do them as two distinct
commits in the same migration branch — pair_key change first
(semantic), threshold rename second (mechanical). If the
pair_key change is later judged wrong, threshold rename can be
cherry-picked forward independently. If we don't bundle, the
threshold rename has no natural trigger and may keep slipping.

**Open question to settle**: Alex's preference.

### 7.2 `nt_ctg` adoption as an operational alphabet

Per memory.md "Considered but not yet pivoted: contig-level
(nt_ctg) clustering" — currently piloted only (2026-05-30,
2026-05-31). Adoption would require a new cluster sweep on
`dna_hash` (full contig), new `clusters_nt_ctg/` parquets, and
a new `cluster_alphabet=nt_ctg` value.

**Argument for bundling**: if alphabet-specific pair_key lands,
nt_ctg becomes the third pair-key family from day one — but
remains unusable without contig-level cluster parquets. Adding
the cluster-sweep work to the same migration would make all
three alphabets first-class on day one.

**Argument against bundling**: nt_ctg has known
**assembly-artifact contamination** (memory.md §117) — polyA
tails and primer-trim differences inflate the "biological"
variation. The contig-trimming preprocessor needed to clean this
up is a separate work item. Adopting nt_ctg before that fix
risks publishing numbers that conflate biology with
assembly-pipeline noise.

**Tentative position**: pair_key migration adds the *capability*
to use nt_ctg (the `ctg` value is reserved in the enum from day
one) but does NOT generate nt_ctg cluster artifacts or
bundles. Operationalizing nt_ctg is a separate plan.

### 7.3 v1 builder deprecation

`src/datasets/dataset_segment_pairs.py` is the legacy v1 builder.
v2 has been the default since 2026-05-11. v1 still ships
`canonical_pair_key` and a parallel `pair_key` construction path.

**Argument for bundling**: pair_key migration is a clean moment
to delete v1 — fewer surfaces to update, no concern about
"alphabet-specific pair_key in v2 but protein-only in v1" forking.

**Argument against bundling**: deletion is its own work item.
Check if any active script or bundle still imports from v1.

**Tentative position**: pre-flight check before the migration.
If v1 is dead code (no imports outside tests; no active
bundles), delete it as a same-PR cleanup. If still in use,
leave alone and update its pair_key construction too.

---

## 8. Out of scope for this plan

- **ESM-2 retraining**: ESM-2 is a protein-level feature; its
  pair_key would naturally stay seq_hash-based even under
  alphabet-specific pair_key (alphabet=aa). No retraining
  triggered by this change.
- **Single-slot HA-NA + PB2-PB1 sweep re-publication**: the
  2026-05-24 / 2026-05-26 results stand as-is, pinned to the
  protein pair_key convention.
- **DataSAIL C1 / 2D-CD-test design**: orthogonal. 2D-CD-test
  (BACKLOG item, separately planned) is a routing-rule change;
  pair_key alphabet is a universe-definition change. They
  compose cleanly but don't depend on each other.
- **k-fold infrastructure** (`docs/plans/2026-05-28_kfold_remaining.md`):
  unaffected. k-fold operates on whatever universe the splitter
  is given.
- **Implementation of the pair_key change itself**: a follow-up
  plan, conditional on agreement to adopt.

---

## 9. Open questions for Alex

1. Should we **measure the universe deltas** (§ 4) as the next
   concrete step, before deciding to adopt?
2. **idXXX → tXXX rename** (§ 7.1) — bundle with this migration
   or separate?
3. **v1 builder** (§ 7.3) — still in use, or safe to delete in
   the same PR?
4. **Intermediate audit-only column** (§ 6 cheap intermediate)
   — would that be useful as a stepping stone, or jump straight
   to full adoption if the deltas warrant it?
5. Any **publication deadline** that constrains when this can
   land? (Rebuild bandwidth is ~1-2 weeks of light load.)

---

## See also

- BACKLOG.md "Methodology ideas — possible paper contributions" #3
- `docs/methods/splits.md` § 2.2 (inline TODO referencing this plan)
- `docs/methods/glossary.md` — Pair universe, Cluster pair,
  Bipartite multigraph definitions
- `src/analysis/audit_split_leakage.py` — existing precedent for
  alphabet-specific pair_key (audit-level)
- CLAUDE.md "Threshold notation", "Sequence hashes", "aa/nt vs
  protein/DNA" — surrounding conventions
