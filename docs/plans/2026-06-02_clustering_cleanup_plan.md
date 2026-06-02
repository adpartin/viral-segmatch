# Clustering utilities — cleanup and alphabet extension

**Status: IN PROGRESS** (Phase 1 IMPLEMENTED; Phases 2-3 pending gates)

Three-phase plan for the cluster-export → mmseqs → parse → parquet
pipeline (`src/utils/clustering_utils.py`, called from
`src/analysis/seq_redundancy_per_function.py`).

Phase 1 is independent cleanup, lands now. Phases 2-3 extend the
alphabet vocabulary and are gated on decisions in the pair_key
alphabet plan (`docs/plans/2026-06-02_pair_key_alphabet_plan.md`).

Source: 2026-06-02 conversation reviewing user's manual edits to the
clustering scripts. The 11-question review surfaced four
independent cleanup items and seven plan-coupled items.

---

## Phases and commit boundaries

| Phase | Scope | Gate | Commit boundary | Status |
|---|---|---|---|---|
| **1. Cleanup** | API + naming cleanup; user TODOs | None | One commit on its own; ship anytime | **IMPLEMENTED 2026-06-02 (commit `27cc0d0` on `feature/clustering-cleanup-phase1`)** |
| **2. Alphabet enum + column rename** | `nt → nt_cds`; cluster parquet column rename; nt_ctg enum value reserved | Pair_key plan § 6 adoption + § 7.1 rename bundling | Co-committed with pair_key migration in same PR; cluster parquets fully regenerated | Pending gates |
| **3. nt_ctg operationalization** | Contig exporter; `clusters_nt_ctg/` dir; per-function contig filter | Pair_key plan § 7.2 (operationalize vs reserve) | Separate commit, only if § 7.2 says go | Pending gates |

Each phase is independently testable (Phase 1 doesn't require Phase
2; Phase 2 doesn't require Phase 3). Phases 2 and 3 are
**ordered** — Phase 3 builds on the enum and parquet-column
conventions Phase 2 establishes.

---

## Phase 1 — Cleanup (no gate)

**Status: IMPLEMENTED 2026-06-02** (commit `27cc0d0` on
`feature/clustering-cleanup-phase1`).

Implementation summary, deviations, and validation outcome are at the
end of this section: [Phase 1 implementation notes](#phase-1-implementation-notes).
Subsections 1.1-1.5 below are the original scoping content, retained
as-is for the design record.

### 1.1 Collapse the two FASTA exporters (Q3)

Today: `export_function_fasta(prot_df, ...)` (aa) and
`export_function_cds_fasta(cds_df, ...)` (nt) do the same logical
operation with seven incidental differences (input validation
shape, hash source [computed vs read], cleaning step, dedup column,
sanity-check shape, stat-key name, local DataFrame naming).

Replace with a single
`export_function_fasta(df, function_name, alphabet, out_path)`:

- Dispatch source/hash columns via a module-level dict:
  ```python
  _COLS_BY_ALPHABET = {
      'aa':     {'seq_col': 'prot_seq',  'hash_col': 'seq_hash'},
      'nt_cds': {'seq_col': 'cds_dna',   'hash_col': 'cds_dna_hash'},
      # 'nt_ctg' added in Phase 3
  }
  ```
- Always read the pre-computed hash from the input DataFrame (no
  on-the-fly md5). Caller pre-hashes — matches the nt path today,
  removes the footgun where a stale `seq_hash` column would
  silently disagree with the FASTA header.
- One `_clean_for_mmseqs(seq, alphabet)` dispatcher: strip
  trailing `*` for aa; no-op for nt_cds.
- One ambiguity-count stat keyed by alphabet
  (`n_with_ambiguity` regex `[^A-Z]` for aa, `[^ACGTacgt]` for nt).
  The current aa `n_with_x` becomes the special case; emit both
  for one release cycle if any caller depends on `n_with_x`.

### 1.2 Rename `clean_for_mmseqs` → `clean_aa_for_mmseqs` (Q4)

Add `clean_nt_for_mmseqs(dna_seq) -> str` as a sibling no-op
pass-through. This is the right home for future nt-specific
cleaning (memory.md §117 polyA / primer-trim trimming for
nt_ctg slots in here cleanly when Phase 3 lands).

### 1.3 Rename `n_unique_sequences` → `n_uniq_seqs` (Q5)

Output dict key only. Two call sites in `seq_redundancy_per_function.py`
(print statements). Sweep both.

### 1.4 Default `threads=8` in `run_mmseqs_easy_clust` (Q8)

Today: `threads=None` → mmseqs uses all cores. Rude on the shared
Lambda dev box. Set default to `8` with docstring note: "default
8 (shared-machine-friendly); override with `--threads` on
dedicated runs."

`cluster_one_function_one_threshold` already exposes `threads` —
the default propagates naturally.

### 1.5 User TODOs (already in the working tree)

- `seq_redundancy_per_function.py` + `cluster_analysis_summary.py`:
  pull `FUNCTION_TO_SHORT` / `SHORT_CANONICAL_ORDER` /
  `--functions` default from `conf/virus/flu.yaml` keys
  (`function_short_names`, `protein_order`, `selected_functions`).
  Eliminates the in-script duplication; downstream changes to the
  config propagate automatically.
- `_threshold_label`: prefix with `t` (e.g., `t095`) instead of
  `id` (e.g., `id095`). Per CLAUDE.md "Threshold notation" (docs-
  adopted 2026-05-29, code migration "deferred to next cluster-
  sweep regeneration"). Phase 1 changes the helper but NOT the
  on-disk dir names — those flip in Phase 2 when cluster parquets
  get regenerated anyway. Pre-flight: the helper has one call
  site (Phase 1 only changes the new-cluster output path; existing
  `clusters_aa/idXXX/` parquets are read by other scripts under
  the old name until Phase 2).
- **Defer**: renaming `seq_redundancy_per_function.py` itself
  (Alex's TODO comment). The script name is referenced in 6+ docs
  and 3 plan files; rename is its own micro-PR.

### Phase 1 validation

Pick one (function, threshold) cell — e.g., `(HA, t099)` aa. Run
the existing pipeline, save the cluster parquet. Run the
post-Phase-1 pipeline against the same input. **Byte-diff the
parquets.** Should be identical: Phase 1 changes only API surface
+ stat-key names, not clustering behavior.

If a diff appears, the candidate is `n_uniq_seqs` rename leaking
into a written column — must stay in the print-only dict.

### Phase 1 commit shape

One commit, descriptive title:
`refactor(clustering): unify exporters, alphabet-aware cleaning, sensible defaults`

Lands independently. No downstream artifacts need regeneration —
all changes are upstream of the on-disk parquet schema.

### Phase 1 implementation notes

**Commit**: `27cc0d0` on branch `feature/clustering-cleanup-phase1`
(`refactor(clustering): Phase 1 — unify exporters, alphabet-aware
cleaning, config-driven constants`). 4 files changed, +295/-168.

**What landed (§ 1.1-§ 1.5):**

- § 1.1 unified exporter: implemented as scoped, with one design
  variation — `_COLS_BY_ALPHABET` uses `'nt'` (not `'nt_cds'`) for
  Phase 1; the `nt → nt_cds` rename moves to Phase 2 alongside the
  bundle/dir flip. Module-level dispatch dict + caller pre-hash +
  exporter recompute-and-assert safety net all in place.
- § 1.2 `clean_for_mmseqs` → `clean_aa_for_mmseqs` + new
  `clean_nt_for_mmseqs` no-op + `_clean_for_mmseqs` dispatcher.
  Implemented as scoped.
- § 1.3 `n_unique_sequences` → `n_uniq_seqs`; `n_with_x` and
  `n_with_ambiguity` unified to alphabet-aware `n_with_ambiguity`.
  Two callers updated.
- § 1.4 Default `threads=16` (not 8 as plan-scoped — Alex bumped
  during review for the shared-box use case). Aligned across CLI +
  `cluster_one_function_one_threshold` + `run_mmseqs_easy_clust`.
- § 1.5 Config-driven function lists: `FUNCTION_TO_SHORT` /
  `SHORT_CANONICAL_ORDER` / `--functions` default sourced from
  `conf/virus/flu.yaml` via new `load_function_metadata(virus_yaml)`
  helper. Helper placed in `src/utils/config_hydra.py` (not
  `clustering_utils.py` as initially scoped) — `config_hydra.py`
  already hosts `get_function_short_name_map(config)`, the
  bundle-based companion; both loaders now co-locate. Both call
  sites (`seq_redundancy_per_function.py` and
  `cluster_analysis_summary.py`) updated. `cluster_analysis_summary`
  scopes its constants to the 8 majors (`selected_short_names`) to
  preserve prior filter behavior; `seq_redundancy_per_function`
  gets the full 20-protein vocabulary.

**Deferred from § 1.5 (moved to Phase 2):**

- `_threshold_label` `id` → `t` prefix change. Plan pre-flight
  said "helper has one call site" — actual count is 12+ source
  files reading the `idXXX` pattern directly
  (`cluster_disjoint_feasibility`, `aa_nt_cluster_crosstab`,
  `dataset_segment_pairs_v2`, `aggregate_mmd_single_slot_sweep`,
  `bipartite_graph_properties`, `coupling_visualizations`,
  `cluster_pair_weight_topk`, `cluster_pair_coupling_precheck`,
  `plot_idxx_sweep_geometry`, …). Flipping just the helper would
  break every reader. Phase 2 regenerates cluster parquets + flips
  all readers in one shot. TODO comment retained at
  `seq_redundancy_per_function.py:116`.

**Deferred (separate micro-PR):**

- Rename `seq_redundancy_per_function.py` → `build_mmseqs_clusters.py`
  (per Alex's TODO at line 3). Script name is referenced in 6+ docs
  + 3 plan files; cross-ref sweep is its own PR.

**Validation outcome:**

End-to-end byte-diff against pre-Phase-1 cluster parquets on 6
(function, alphabet) cells × t099:

| Cell | Unique seqs | Clusters | Match |
|---|---:|---:|---|
| aa M1 t099 | 4,771 | 1,764 | identical (hash set + every seq_hash → same cluster_rep) |
| aa HA t099 | 41,896 | 22,679 | identical |
| aa NA t099 | 37,488 | 9,369 | identical |
| nt M1 t099 | 32,413 | 10,227 | identical |
| nt HA t099 | 65,414 | 12,150 | identical |
| nt NA t099 | 58,887 | 12,092 | identical |

Phase 1 is end-to-end behavior-preserving: same FASTAs → same
mmseqs runs → same parquet cluster assignments. Validation
scripts at `/tmp/validate_phase1_*.py` (gitignored; reproducible).

---

## Phase 2 — Alphabet enum + cluster-parquet column rename

**Gate: pair_key plan § 6 adoption decision + § 7.1 rename
bundling decision.** Phase 2 is rebuild-heavy (every active
cluster parquet regenerates) and should land in the same PR as
the pair_key migration so users never see a half-migrated state.

### 2.1 Alphabet enum: `{aa, nt} → {aa, nt_cds, nt_ctg}` (Q1)

`'nt'` becomes `'nt_cds'` everywhere it appears as a string value:

- Function signatures: `run_mmseqs_easy_clust(alphabet=)`,
  `cluster_one_function_one_threshold(alphabet=)`,
  `parse_cluster_tsv(alphabet=)` (new param, see § 2.2).
- Bundle YAML keys: `split_strategy.cluster_alphabet: aa|nt`
  becomes `aa|nt_cds|nt_ctg`. ~6 bundles touched (the four
  `cluster_nt_id{100,099}` bundles for HA-NA + PB2-PB1 plus any
  derivatives).
- Dataset run dirs: `clusters_nt/` → `clusters_nt_cds/` under
  `data/processed/flu/<version>/`. Coordinate with the
  `clusters_aa/idXXX/` → `clusters_aa/tXXX/` rename in the same
  pass (Phase 1 § 1.5 only flipped the helper; here the on-disk
  dirs flip too because we're regenerating anyway).
- `--dbtype` dispatch stays unchanged: `1` for aa, `2` for nt_cds
  AND nt_ctg (both are nucleotide alphabets).
- `nt_ctg` enum value is **reserved** in Phase 2 (passes
  validation) but raises `NotImplementedError` until Phase 3.

### 2.2 `parse_cluster_tsv` writes alphabet-specific hash column (Q9)

Today: writes `seq_hash` column regardless of alphabet — wrong
for nt_cds clusters (values are `cds_dna_hash` semantically).

Change: add `alphabet` parameter, dispatch column name:

| alphabet | Output column name |
|---|---|
| `aa` | `seq_hash` |
| `nt_cds` | `cds_dna_hash` |
| `nt_ctg` | `dna_hash` (added Phase 3) |

Drop the rename hack at `_split_helpers.py::attach_cluster_ids` line
102: `lookup.rename(columns={'seq_hash': pos_col_a, ...})`. The
join column now matches naturally.

### 2.3 Re-build all active cluster parquets

The parquet schema column rename means every parquet under
`clusters_{aa,nt_cds}/<threshold>/<function>_cluster.parquet` must
be regenerated. Validation: re-run `cluster_disjoint_feasibility`
on HA-NA aa t099 — bipartite-CC sizes should be identical, only
the column name changes.

Approximate compute: ~30 min sequential for 8 functions × 11
thresholds × 2 alphabets on the segmatch env.

### 2.4 Update CLAUDE.md "Threshold notation" entry

Today: "code/dirs/bundles still use `idXXX`. Full `idXXX → tXXX`
migration … deferred until the next cluster-sweep regeneration."
Post-Phase-2: this IS that regeneration. Update the entry to
reflect code/dir adoption.

### Phase 2 validation

- Cluster-disjoint feasibility CSV on (HA-NA, PB2-PB1) × aa × all
  thresholds: bipartite-CC sizes identical to pre-Phase-2.
- Cluster-disjoint feasibility CSV on (HA-NA, PB2-PB1) × nt_cds ×
  {t100, t099}: same.
- `attach_cluster_ids` smoke test on one Stage 3 dataset
  rebuild: cluster_id assignments byte-identical.

### Phase 2 commit shape

Two commits in the same PR, ordered:
1. `refactor(clustering): alphabet enum {aa,nt_cds,nt_ctg}; parse_cluster_tsv writes alphabet-specific hash column` — code only, no parquet regeneration.
2. `data(clusters): regenerate cluster parquets under new column convention; flip clusters_nt → clusters_nt_cds; flip idXXX → tXXX dirs` — data-only commit (or git-LFS / external if parquets gitignored; check current convention).

Both bundle with the pair_key migration PR.

---

## Phase 3 — nt_ctg operationalization

**Gate: pair_key plan § 7.2 decision.** If § 7.2 says "reserve
the enum only" (likely, given the +4-5 pp marginal delta over
nt_cds in pair_key plan § 4 + the assembly-artifact contamination
concern), Phase 3 does not happen. If § 7.2 says "operationalize",
Phase 3 follows.

### 3.1 New exporter `export_function_ctg_fasta`

Signature different from Phase 1's unified exporter because the
input is per-contig (genome_final), not per-function:

```python
def export_function_ctg_fasta(
    genome_df: pd.DataFrame,
    prot_df: pd.DataFrame,
    function_name: str,
    out_path: Path,
) -> dict
```

Logic:
1. Filter `prot_df` to rows where `function == function_name`.
2. Inner-join to `genome_df` on `(assembly_id, genbank_ctg_id)` to
   get the contigs that *carry* this function's protein.
3. Compute or read `dna_hash` (md5 of full contig); dedup.
4. Write FASTA with `dna_hash` as header, raw contig DNA as body.
5. No mmseqs cleaning (contig DNA is raw; ambiguity codes left
   in place for mmseqs to score natively).

**Decision needed before implementation**: contigs that carry
multiple proteins (M1+M2 share segment-7 contig; NS1+NEP share
segment-8; PA+PA-X share segment-3). The same contig
participates in clustering twice — once filtered for M1, once
filtered for M2 — and produces two cluster assignments for the
same dna_hash. The bipartite-CC routing on (M1_ctg, M2_ctg)
needs the per-function cluster views to be **distinct enough**
that the routing constraint is meaningful. Two options:

- (a) **Naive per-function**: cluster the same contig set twice
  with the function tag. Cluster IDs differ across functions
  (different rep selection per run) so the bipartite is
  well-defined, but the underlying sequence space is identical.
  Risk: cluster_disjoint constraint becomes vacuous for shared-
  contig pairs.
- (b) **Pre-filter to function-specific genomic windows**:
  trim each contig to the protein's location range before
  hashing. Loses the contig-vs-CDS distinction (becomes a
  function-specific window with UTRs). Probably defeats the
  point of nt_ctg over nt_cds.

This decision is the load-bearing risk in Phase 3 and is the
strongest argument for deferring nt_ctg operationalization.

### 3.2 Directory convention

`data/processed/flu/<version>/clusters_nt_ctg/t<XXX>/<function>_cluster.parquet`.

### 3.3 Validation

Cluster-disjoint feasibility on (HA, NA) × nt_ctg × {t100, t099}.
Compare bipartite-CC sizes to nt_cds at the same thresholds:
the pair_key plan § 4 measurement says nt_ctg pair universe is
+4-5 pp over nt_cds; feasibility numbers should track.

### Phase 3 commit shape

One commit:
`feat(clustering): nt_ctg cluster path (full-contig dna_hash)`

Plus data commit if parquets are tracked.

---

## Risks

- **Phase 2 parquet regeneration churn**: ~30 min wall time, but
  if cluster_disjoint_feasibility numbers shift unexpectedly,
  diagnosing root cause (column rename vs mmseqs nondeterminism
  vs upstream input drift) is annoying. Mitigation: regen
  pre-Phase-2 with no code changes first, save those parquets as
  oracle; then regen post-Phase-2 and diff.
- **Phase 3 shared-contig ambiguity** (§ 3.1 decision): unclear
  whether nt_ctg routing on M1-M2 / NS1-NEP / PA-PA-X pairs is
  meaningful. None of the production schema pairs (HA-NA,
  PB2-PB1) hit this; segment-7/8/3 pairs would.
- **Phase 1 unified exporter regressions**: aa-side currently
  computes `seq_hash` on the fly. Switching to "read from input"
  silently passes if the caller pre-hashes correctly; silently
  produces wrong FASTA headers if not. The dedup sanity-check
  catches mismatches between hash and cleaned-seq, but only when
  two ROWS have the same hash but different sequences. A single
  row with a stale hash slips through. Mitigation: at exporter
  entry, recompute hash from `seq_col` and assert against the
  `hash_col` value. Costs one md5 per row; safe.

---

## Out of scope

- **`mmseqs_bin` default to a hardcoded path** (Q6): rejected.
  Current 3-tier resolution (arg → `$MMSEQS_BIN` → PATH) is
  correct; the fix for "I forget to set MMSEQS_BIN" is shell-rc
  level, not source.
- **Renaming `seq_redundancy_per_function.py`**: separate
  micro-PR (6+ docs reference the current name).
- **Replacing mmseqs entirely** (alternative clustering tools):
  not on this plan's radar.
- **Generalizing to non-Flu corpora**: enum and dispatch stay
  Flu-shaped; Bunyavirales is not maintained.

---

## Decision points before execution

1. **Phase 1 — green.** Confirm with Alex, then execute.
2. **Phase 2 — gated on pair_key plan adoption + rename
   bundling.** Do NOT start until pair_key plan § 6 says
   "adopt" and § 7.1 says "bundle the idXXX→tXXX rename in
   this migration."
3. **Phase 3 — gated on pair_key plan § 7.2** AND on the §3.1
   shared-contig decision. If § 7.2 says "reserve enum only",
   Phase 3 is closed; § 3.1 decision is moot.

---

## See also

- `docs/plans/2026-06-02_pair_key_alphabet_plan.md` — the
  alphabet-extension parent decision.
- `src/utils/clustering_utils.py`,
  `src/analysis/seq_redundancy_per_function.py` — the files
  touched.
- CLAUDE.md "Threshold notation" — the docs-only `idXXX → tXXX`
  adoption (2026-05-29) that Phase 2 finishes.
- `.claude/memory.md` "2026-05-22 symmetric easy-linclust"
  bullet — most recent precedent for a cluster-pipeline
  regeneration (5 commits; reference scope for Phase 2 effort).
