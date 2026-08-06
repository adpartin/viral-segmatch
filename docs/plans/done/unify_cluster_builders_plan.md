# Unify cluster builders (linclust / cluster / search) into one file

**Status: IMPLEMENTED**

## Context
`src/preprocess/build_mmseqs_clusters.py` (set-cover: easy-linclust / easy-cluster) and
`src/preprocess/build_ood_clusters.py` (easy-search + union-find) are the **same pipeline
skeleton with one swappable core** (the clustering step). ~150–200 lines are duplicated:
main() preamble, the threshold sweep, the cluster-parquet cache wrapper + stats-dict, the
FASTA-export cache, the input arg group, and the stats-CSV merge (`build_ood` already uses the
shared `write_or_merge_stats_csv`; `build_mmseqs` inlines a copy). We want **one file** that can
invoke `easy-linclust`, `easy-cluster`, or `easy-search`, so the shared skeleton lives once and
each method is a thin injected function. Also lets us run set-cover `--cluster-mode 0` vs `1`
A/B experiments trivially.

## Design

### Two orthogonal axes — never a combined enum
- **`--method {linclust, cluster, search}`** — the mmseqs subcommand / clustering paradigm.
- **`--cluster-mode {0,1,2}`** — a parameter of `linclust`/`cluster` only (rejected for `search`).
`search` builds CCs via Python union-find (`connected_components_from_hits`), so it takes no
`--cluster-mode`. Keep the two args separate; record BOTH in outputs.

### Arg matrix (verified against `mmseqs <sub> --help`; source of truth = the binary, not web docs)
| arg | linclust | cluster | search | kind |
|---|:-:|:-:|:-:|---|
| `--min-seq-id` `-c` `--cov-mode` `--seq-id-mode` `-e` `--max-seqs` `--threads` `--dbtype` | ✓ | ✓ | ✓ | mmseqs2 |
| `-s` (sensitivity) | — | ✓ | ✓ | mmseqs2 |
| `--cluster-mode` | ✓ | ✓ | ✗ | mmseqs2 |
| `--similarity-type` | ✓ | ✓ | ✗ | mmseqs2 |
| `--single-step-clustering` | ✗ | ✓ | ✗ | mmseqs2 (easy-cluster only) |
| `--prefilter-mode` (via `--exhaustive`) `--search-type` | ✗ | ✗ | ✓ | mmseqs2 |
| `--gpu` | ✗ | ✗ | ✓ | mmseqs2 |
| `--delete_hits` `--scratch_dir` | ✗ | ✗ | ✓ | builder |
| `--functions` `--thresholds` `--out_root` `--force` `--no_combined` `--method` | ✓ | ✓ | ✓ | builder |

### Help / docstring convention (requested)
Every arg's `help=` and the docstring where it's passed states either
**`[mmseqs2: --flag]`** (passthrough — note which subcommands accept it) or **`[builder]`**
(our orchestration). `--seq-id-mode` and `--similarity-type` are genuine mmseqs flags (in the
binary help) even though the web docs omit them — mark them `[mmseqs2]`.

### Shared driver (the collapsed duplication) — lives in `build_clusters.py`
- `add_input_args(parser)` — the mutually-exclusive `--protein_final/--cds_dna_final/--ctg_dna_final`
  (+ `--function_source`) group.
- `load_and_filter(args) -> (df, alphabet, functions)` — `load_sequence_frame` + `mkdir` +
  `filter_present_functions` + skip-warning + no-functions exit (byte-identical today).
- `build_or_load_clusters(cluster_parquet, force, meta, cluster_fn) -> dist` — the cache wrapper:
  cache short-circuit, timing, the 4 column attaches (`function/function_short/threshold/alphabet`),
  `to_parquet`, cached/fresh print, then `cluster_size_distribution` + `dist.update({... method,
  cluster_mode ...})`. `cluster_fn(fasta) -> lookup df` is the ONLY method-specific piece.
- threshold sweep: `for threshold: for short: build_or_load_clusters(...)`; then
  `aggregate_combined_lookup` (unless `--no_combined`) + `write_runtime_json`; finally
  `write_or_merge_stats_csv(out_root, all_stats, 'cluster_stats.csv')`.

### Per-method `cluster_fn` (the real difference, kept per-method)
- `linclust`/`cluster`: `run_mmseqs_easy_clust(...)` → `parse_cluster_tsv(...)`.
- `search`: `read_fasta_hashes` → `run_mmseqs_search(...)` → `connected_components_from_hits(...)`
  (+ the hits-persistence block gated by `--delete_hits`).
The `run_mmseqs_*` wrappers in `clustering_utils.py` are unchanged.

### Outputs
- **One stats file: `cluster_stats.csv`** (drops set-cover's old `redundancy_stats.csv` name),
  with new **`method`** and **`cluster_mode`** columns so a cm0-vs-cm1 (or method) A/B is
  distinguishable in the data.
- `runtime.json` per threshold: method-aware config dict (`method`, `cluster_mode`, sensitivity,
  coverage, exhaustive/gpu as relevant).
- **Drop `write_results_markdown`** + `--results_md` + `SHORT_TO_SEGMENT` + `SHORT_CANONICAL_ORDER`
  (~130 lines).

### `--cluster-mode 1` docstring note (requested, short)
On the `--cluster-mode` help: "1 = connected-component assignment; unlike the `search` method's
CCs this carries **no exhaustive-search across-cluster-separation guarantee** (the linclust
prefilter can miss edges). `search` is far slower than linclust cluster-mode 1 but is the one
that gives the guarantee."

## Files to change
- **New:** `src/preprocess/build_clusters.py` — the unified builder (shared driver + both cluster_fns).
- **Shims (thin, back-compat):** `build_mmseqs_clusters.py` → default `--method linclust`;
  `build_ood_clusters.py` → default `--method search`; each forwards to `build_clusters.main()`
  and still allows an explicit `--method` override.
- `src/utils/clustering_utils.py` — no logic change; already exports `write_or_merge_stats_csv`,
  `write_runtime_json`, `run_mmseqs_easy_clust`, `run_mmseqs_search`, `connected_components_from_hits`,
  `aggregate_combined_lookup`, `parse_cluster_tsv`. (Shared driver helpers may live in
  `build_clusters.py` rather than here — keep clustering_utils as the mmseqs/graph layer.)
- **Docs/docstrings:** update references in `docs/architecture.md`, `docs/project_changelog.md`,
  and the prose mentions in `src/analysis/verify_ood_clusters.py`, `plot_clusters.py`,
  `cluster_disjoint_feasibility.py`, `cluster_analysis_summary.py`.

## Verification (refactor-first, then compare)
The two methods have different reproducibility, so different bars:

**A. linclust/cluster (set-cover) — compare on INVARIANTS (not bit-reproducible).**
Run unified `--method linclust` AND current-HEAD `build_mmseqs` on **HA t095 nt_cds** at the same
`--threads`; compare `n_clusters`, `largest_cluster`, size distribution, and the emitted `# CMD:`
line. Expect a match. Don't require byte-identical parquets — linclust has thread nondeterminism;
the on-disk 06-02 artifact is a weaker secondary check (confounded by code-drift + nondeterminism).

**B. search (OOD) — compare EXACTLY (deterministic).**
`connected_components_from_hits` gives canonical, size-ranked, hash-tie-broken labels
(clustering_utils.py:930–932), so given the same hits the CCs are byte-identical. Test on **M1 @
t099** (smallest protein, tightest threshold = cheapest OOD run, ~minutes; big proteins take
hours). Run unified `--method search` vs current-HEAD `build_ood` into two temp out_roots, compare:
  1. emitted easy-search `# CMD:` — byte-identical (invocation + `--search-type 3` + flags unchanged),
  2. **`<hash> → cluster_id` mapping — byte-identical** (the strict bar the determinism buys),
  3. hits parquet / `n_hits` — identical if easy-search is run-to-run stable (drift = search's own
     prefilter nondeterminism, isolable from the refactor).
Plus the search-only paths linclust doesn't exercise:
  4. `--delete_hits` file outcomes (TSV+parquet gone) vs default (compact `*_hits.parquet` kept),
  5. `--scratch_dir` routing (transient TSV to scratch, cluster parquet to out_root, scratch cleaned),
  6. arg-gating: `--cluster-mode` with `--method search` → `SystemExit` (new unified logic),
  7. threshold-label collision check preserved (build_ood has it; build_mmseqs doesn't).

**C. Shims & help.** Confirm old invocations still run (`python -m ...build_ood_clusters
--functions HA ...`; `...build_mmseqs_clusters ...`), and the `[mmseqs2]/[builder]` help renders.

## Out of scope (separate follow-ups, not this refactor)
- Regenerating `clusters_nt_cds` and the **cm0/cm1 A/B** into `clusters_nt_cds_cm0` /
  `clusters_nt_cds_cm1` (a *use* of the unified builder: `--method linclust --cluster-mode {0,1}
  --out_root <dir>`; analysis writeup notes cm1 ≠ search-CC guarantee).
- The OOD `combined_cluster.parquet` regen for the just-finished `clusters_aa_ood`.
- The `verify_ood_clusters.py` `--search-type 3` nt bug.
