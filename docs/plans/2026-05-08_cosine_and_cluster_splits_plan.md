# Cluster-disjoint and cosine-controlled splits — Plan

**Status: PROPOSED** (Experiment B is the priority — Carla's suggestion)
**Date:** 2026-05-08; refreshed 2026-05-14.
**Parent plan:** `docs/plans/2026-05-07_leakage_diagnostics_plan.md` Exp 5.

## One-line framing

Extend our current `seq_disjoint` routing — which is implicitly
cluster-disjoint at the **100% identity threshold** — to a tunable
threshold like 95% via mmseqs2, so that pairs across splits don't just
have **non-identical** sequences but **non-similar** sequences. The
threshold trades partitioning freedom (and hence dataset size) for the
strength of the generalization test.

## Context

`seq_disjoint` with `hash_key=seq` (production default since 2026-05-12)
guarantees that no two pairs across splits share a `seq_hash` (protein-sequence
hash). That's exact-match deduplication — it addresses mode #3
(sequence-level leakage) but leaves **mode #4 (cluster leakage)** open:
a test pair whose protein is 1 aa away from a training pair's protein
has a different `seq_hash` and is therefore allowed.

The 2026-05-13 aa-vs-nt similarity diagnostic
(`docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md`) confirmed
this is real. **The diagnostic measures the residual leakage that
`hash_key=seq` doesn't catch**: for every unique test protein it
finds the maximum aa-identity to any training protein (and the same
at the nt level for DNA sequences). On the HA/NA `*_regimes` dataset
under `hash_key=seq`:

- 47.8% of unique test HA proteins have a training neighbor at **≥99.5% aa identity** (slot a). At the nt level for the same dataset, only 41.6% of unique test HA contigs hit the ≥99.5% nt-identity bucket — so the high-similarity tail is **6 pp denser when measured on protein than on DNA**.
- For NA (slot b): 55.1% aa vs 44.9% nt → **+10 pp denser on aa**.

In words: under current `seq_disjoint` routing, ~half of the test
proteins are essentially "almost identical" to *something* in train,
and that residual near-neighbor density is higher at the aa level
than at the nt level (which matches the biological expectation:
synonymous codons let DNA diverge while keeping the protein constant,
so aa is the "stickier" similarity measure). The current model
results are therefore upper-bounded by some unknown amount of
near-neighbor lookup — and **this plan quantifies how much**.

Two split-construction experiments address mode #4 from different
sides:

| | Cluster-disjoint (Experiment B) | Cosine-controlled (Experiment A) |
|---|---|---|
| **Space** | Biological sequence space | Model's feature space (k-mer or ESM-2 concat) |
| **What "similar" means** | Pairwise aa identity (e.g. ≥ 95%) | Cosine on the joint pair feature vector |
| **Threshold semantics** | % aa identity — biology-standard, representation-invariant | Cosine ∈ [-1, 1], unitless, representation-dependent |
| **What it catches** | Test protein biologically near a training protein | Test pair feature vector cosine-near a training pair's |
| **What it can miss** | Cases where features overlap for non-biological reasons (compositional bias) | Phylogenetic neighbors whose features happen to differ |
| **Tool** | `mmseqs easy-cluster` (external binary) | sklearn / numpy on cached feature matrix |
| **Priority** | **High — Carla's proposal** | Secondary — complementary diagnostic |

If both experiments drop AUC similarly at corresponding thresholds, the
model's features track biology. If cluster-disjoint drops AUC harder,
the model relies on biological neighbors more than on the feature
representation; if cosine-controlled drops harder, the feature space
has artifacts beyond biology.

---

## Experiment B — mmseqs2 cluster-disjoint splits (priority)

### Goal

Build alternative train/val/test such that no two pairs across splits
share a protein cluster at a chosen aa-identity threshold (default 95%).
Re-train and compare to `seq_disjoint` baseline.

### Method

1. **Pre-clustering redundancy assessment** (do FIRST, before the
   routing change). For each major protein function in `protein_final`,
   run mmseqs2 at multiple thresholds (1.00, 0.95, 0.90, 0.80) and
   report the cluster-size distribution: number of clusters, largest
   cluster size, median cluster size, fraction of sequences in singletons.
   This characterizes the per-function redundancy — which is the data
   property that determines how aggressive a cluster-disjoint split can
   be without dropping unacceptable numbers of pairs. Output:
   `docs/results/<date>_protein_redundancy_per_function.md` with a
   per-function table per threshold.
2. **Cluster all unique proteins.** Run `mmseqs easy-cluster` on the
   unique-sequence FASTA exported from `protein_final.csv`.
3. **Cluster → seq_hash mapping.** Parse the output TSV; emit a
   parquet `(seq_hash, cluster_id)` for fast lookup.
4. **Cluster-disjoint routing.** Partition `cluster_id`s across
   train/val/test. Each pair routes to the split where BOTH its
   sequences' clusters land (cluster-disjoint on **both slot a and
   slot b**, since each side of a pair contributes its own protein).
   Pairs spanning split-mixed clusters are dropped.

### Interaction with negative-sampling phases

Cluster-disjoint routing operates at the **positive-pair routing
stage** (Phase 3) — it partitions positive pairs into splits *before*
negative sampling starts. The downstream phases (Phase 4 coverage and
Phase 5 regime-aware fill) are unchanged in their *logic* but operate
on the cluster-disjoint partition:

- **Coverage (Phase 4)** still iterates every `(slot, seq_hash)` in
  the partition and finds a covering negative. The partner-isolate
  pool is the full corpus minus the excluded-by-`seq_hash` set —
  cluster boundaries are NOT enforced on negative partners (only on
  positive routing). This is a deliberate choice: the leakage we're
  preventing is *within-pair* train↔test similarity, and a negative
  pair's two sequences are by construction from different isolates;
  cluster-disjoint on the positive side is sufficient.
- **Regime-aware coverage** (planned in
  `docs/plans/2026-05-14_regime_aware_coverage_plan.md`) interacts
  with cluster-disjoint routing the same way the existing coverage
  does: cluster boundaries shape the *positive routing*, then the
  regime priority chain picks negative partners. If cluster-disjoint
  shrinks the per-split positive count substantially, the regime
  sampler may face a smaller candidate pool for hard regimes; this is
  measurable as a follow-up but doesn't change the design.
- **Negative regimes (8-tuple match counts)** are computed from the
  pair's metadata, independent of cluster membership. The
  `regime_targets` knob from the bundle continues to drive the fill
  phase as designed.

### Why aa, not nt

mmseqs2 supports both, but protein clustering is the right choice here:

- Carla's framing — "biological cluster" — naturally maps to aa
  identity. Two viral isolates with identical proteins but
  synonymous-codon-different DNA are biologically the same lineage;
  DNA clustering would unnecessarily split them.
- The 2026-05-13 aa-vs-nt similarity diagnostic showed the
  high-similarity tail is denser on aa than on nt (see Context),
  so aa is where the residual leakage to mitigate lives.
- mmseqs2 sensitivity is highest on protein input.

**Will aa clustering drop more sequences than nt clustering at the
same threshold?** Generally yes, in a specific sense: at any given
X% identity threshold, aa clustering captures *more* sequences inside
each cluster (because synonymous nt variation collapses to identical
aa). That makes individual aa-clusters *larger* — and a single
oversize cluster can force a partitioning failure when no split has
room. The expected pattern is **fewer clusters but larger ones on
aa**, which can translate to more dropped pairs at aggressive
thresholds. The pre-clustering assessment (step 1) measures this
directly per major protein.

If DNA clustering is ever needed as a control, it must operate on
**CDS DNA** (extracted via the `location` field — helper sketched in
`gto_format_reference.md` §9; not currently in the codebase), not the
full contig.

### Preprocessing for mmseqs2

mmseqs2 input requirements:
- Canonical alphabet preferred (`ACDEFGHIKLMNPQRSTVWY`).
- **`X` residues are accepted natively** — mmseqs2 treats them as
  unknown and skips them during similarity scoring. No imputation
  needed (and arguably better avoided — `X → G` substitution introduces
  artificial signal that a clustering tool would treat as real).
- Sequences must not contain internal stop codons (`*` characters mid-sequence).
- Terminal stop codons (trailing `*`) should be stripped.

This means **using the raw `prot_seq` column (with light cleaning) is
preferable to using `esm2_ready_seq`** for clustering:

| Field in `protein_final` | Suitable for mmseqs2? | Why |
|---|---|---|
| `prot_seq` (raw, includes terminal `*`, may have X) | **Preferred** — after stripping terminal `*` and filtering rows with internal `*` | mmseqs2 handles `X` natively; preserves the actual sequence information |
| `esm2_ready_seq` (terminal `*` stripped, `X→G` imputed) | Acceptable | The `X→G` substitution is a minor information loss; clustering will still work but with slightly inflated similarity at imputed positions |

Preprocessing pipeline:

1. Filter `protein_final` to rows with non-null `prot_seq` (Stage 1 output).
2. Drop rows where `has_internal_stop == True` (annotated by Stage 1).
3. Strip trailing `*` from `prot_seq` if present.
4. De-duplicate on `seq_hash` (one FASTA entry per unique protein
   sequence; ~375K unique from ~1.79M occurrences).
5. Write FASTA: `>{seq_hash}\n{cleaned_prot_seq}\n`.
6. Run mmseqs2 (see command below).
7. Parse cluster TSV → `(seq_hash, cluster_id)` lookup parquet.
8. Join back to all protein rows in the routing helper.

Minimum sequence length (mmseqs2 default 11) is well below the
shortest flu major (NS1 min 201 aa), so no length filter is needed.

### One-time prerequisite

Install mmseqs2 via your preferred package manager. Available on
bioconda / conda-forge (`conda install -c bioconda mmseqs2`), via
Homebrew (`brew install mmseqs2`), or as a static binary release from
the upstream GitHub. The wrapper script doesn't depend on installation
method — it expects `mmseqs` to be on PATH.

Cluster command (encapsulated by the new wrapper script):

```bash
mmseqs easy-cluster proteins.fasta out_clusters_id095 tmp_mmseqs \
    --min-seq-id 0.95 -c 0.8 --cov-mode 0
```

**Why `easy-cluster` rather than `easy-linclust`?**
mmseqs2 ships two clustering entry points with different sensitivity /
speed tradeoffs:

| Command | Algorithm | When to use |
|---|---|---|
| `easy-cluster` | Cascaded clustering with prefilter + alignment | Recommended for ≤ ~10M sequences; higher sensitivity, catches more borderline cluster members |
| `easy-linclust` | Linear-time clustering via k-mer hashing | Designed for huge datasets (UniRef-scale, ≥ 10⁸ sequences); trades sensitivity for speed |

For our ~375K unique flu proteins, `easy-cluster` runs in minutes and
the extra sensitivity is worth the modest runtime cost. `easy-linclust`
would also work but is over-engineered for this scale and may slightly
under-cluster at low identity thresholds.

### Files to create

| File | Purpose |
|---|---|
| `scripts/cluster_proteins_mmseqs.sh` | Export protein FASTA, run mmseqs, parse TSV → `data/processed/flu/{version}/protein_clusters_id{N}.parquet` (one-time per dataset version × threshold) |
| `src/analysis/protein_redundancy_per_function.py` | Pre-clustering assessment (step 1 of Method): per-function cluster-size distributions at multiple thresholds, written to a results doc |
| `src/datasets/_split_helpers.py` (new or extend existing) | `cluster_disjoint_split(pos_df, cluster_lookup, ratios, seed)`. Routing decision happens at Phase 3 (positive-pair routing); does not need `regime_targets` — those drive Phase 4/5 negative sampling, which runs on the already-partitioned positives. |
| `src/datasets/dataset_segment_pairs_v2.py` | Add `split_strategy.mode: cluster_disjoint` branch; read `dataset.cluster_id_path` |
| `conf/dataset/default.yaml` | Document the new mode |
| `conf/bundles/flu_ha_na_cluster_id{N}.yaml` (3 leaves) | One per identity threshold ∈ {0.95, 0.90, 0.80}; inherit `flu_ha_na`; override routing knobs |

### Thresholds to sweep

| Threshold (aa identity) | What collapses into one cluster | Expected outcome |
|---|---|---|
| 100% | Only character-identical sequences | Equivalent to current `seq_disjoint` (baseline control) |
| **99%** (optional) | Single-residue drift on a 100-aa protein; ≤2 aa diff on 200-aa; ≤5 aa diff on 500-aa | Catches the tightest near-neighbor leakage; minimal data loss |
| **95%** | Sequences differing by up to ~5% of residues — typical clade-level drift on flu | Standard non-redundant protein clustering; first-cut headline |
| 90% | Up to ~10% residue drift — sub-clade lineages collapse | Bigger AUC drop; data loss starts mattering on conserved functions |
| 80% | Up to ~20% residue drift — broad lineages collapse into one cluster | Largest AUC drop; conserved functions (M1, NP) likely degenerate into too few clusters to split |

Single-residue drift between two flu majors (200–760 aa long) is in
the **99–99.8% identity range**, not at 95% or below. The exact
threshold at which a 1-residue difference falls outside a cluster
depends on protein length; the table above gives the typical range.

### Outputs and success criteria

Artifacts:
- `data/processed/flu/July_2025/protein_clusters_id{95,90,80}.parquet`.
- The pre-clustering assessment doc (step 1):
  `docs/results/<date>_protein_redundancy_per_function.md`.
- Three Stage 3 runs at thresholds {0.95, 0.90, 0.80}.
- LGBM training runs per threshold (matching the existing production
  bundle settings — nt 6-mers, `unit_norm`, `unit_diff + prod`).
- Result CSV `docs/results/<date>_cluster_split_sweep.csv` with
  columns `id_threshold`, `n_pairs_train/val/test`, `auc_roc`,
  `f1`, `mcc`, `host_subtype_year_tnr`. Include the 100% baseline.

Plots:
1. **Sample size vs threshold** (sanity / cost plot). Y: number of
   surviving pairs (train + val + test), or equivalently `n_clusters`
   per function. X: identity threshold ∈ {1.00, 0.95, 0.90, 0.80}.
   One line per major protein function. Shows how aggressively the
   dataset shrinks at each threshold and which proteins degenerate
   first.
2. **Model performance vs threshold** (the headline). Y: AUC-ROC and
   `host_subtype_year` TNR. X: identity threshold. Plot two
   variants:
   - **As-is**: full surviving dataset at each threshold (largest at
     1.00, smallest at 0.80). Shows the *combined* effect of "fewer
     near-neighbors" and "less training data."
   - **Sample-size-controlled**: subsample the larger datasets down
     to the smallest surviving size, so each threshold trains on the
     same number of pairs. Isolates the "fewer near-neighbors"
     effect from the "less data" effect. Single most-informative
     plot for the leakage question.

**Success criterion:** AUC drops monotonically as threshold decreases
on both variants of plot 2. If the as-is curve drops but the
sample-size-controlled curve stays flat, the loss is driven by
dataset shrinkage rather than near-neighbor removal — re-interpret.
Quantitatively, the 95% → 100% gap on the controlled curve is the
operational measure of "how much of headline accuracy depended on
near-identical neighbors."

### Risks

- **Highly conserved functions degenerate at low thresholds.** M1
  (4,771 unique proteins across 114K isolates) may collapse into very
  few clusters at 95%, making the cluster partition trivial and
  shrinking dataset size sharply. *Mitigation:* report `n_clusters` per
  function; if any function has < 100 clusters at a threshold,
  exclude or report separately.
- **Cluster TSV / seq_hash misalignment.** If mmseqs2 output covers a
  different set of `seq_hash`es than `protein_final` (e.g. due to
  length filter mismatches), silent joins drop rows. *Mitigation:*
  hard assertion in the routing helper — every `seq_hash` in `pos_df`
  must appear in the cluster lookup.

---

## Experiment A — Cosine-controlled splits (secondary)

Same headline goal — test how much hard-regime performance depended on
near-neighbors of training pairs — but partitioned in the **model's
feature space** rather than in biological sequence space. Run only as
a complement to Experiment B, ideally to attribute leakage to
representation vs. biology.

### Goal

Build alternative train/val/test such that every test pair has
`max_train_cosine ≤ τ` for τ ∈ {0.95, 0.90, 0.80, 0.70}. Re-train under
each threshold; plot AUC vs τ.

### Method

1. **Compute joint pair features** for every pair: `concat(kmer[ctg_a],
   kmer[ctg_b])`, L2-normalized. Cache as sparse matrix.
2. **Pair-similarity graph**: for each pair, find its nearest pair in
   feature space; edge if cosine ≥ τ.
3. **Component-aware split**: connected components go to one split;
   shuffle components into train/val/test by target ratios.
4. **Sanity check**: re-compute `max_train_cosine` for each test pair
   and assert all are < τ.

### Files to create

| File | Purpose |
|---|---|
| `src/datasets/_split_helpers.py` | `cosine_controlled_split(pos_df, features, tau, ratios, seed)` |
| `src/datasets/dataset_segment_pairs_v2.py` | `split_strategy.mode: cosine_controlled` branch + `cosine_threshold` knob |
| `src/utils/similarity_utils.py` | Shared joint-feature builder + nearest-cosine math (reused by Experiment B and by Exp 3 cosine deciles) |
| `conf/bundles/flu_ha_na_cosine_t{N}.yaml` (4 leaves) | One per τ |

### Outputs

`docs/results/<date>_cosine_controlled_sweep.csv` + `.png`, same
schema as Experiment B's sweep CSV. Includes τ=1.0 (current
`seq_disjoint`) as control.

### Risks

- **Component sizes blow up at low τ** for highly conserved proteins;
  one giant component can swallow most pairs. *Mitigation:* abort and
  document when achieved test < 500 pairs.

---

## Shared infrastructure

A small refactor pays off for both experiments and for the parent
plan's Exp 3 (which already needs cosine math):

- `src/utils/similarity_utils.py` — pure functions for joint feature
  construction (k-mer or ESM-2) and pair-pair cosine math. Already
  partially exists for `exp3_cosine_deciles.py`.
- `src/datasets/_split_helpers.py` — pure functions for the new split
  modes (`cosine_controlled`, `cluster_disjoint`). `split_dataset_v2`
  stays a thin dispatcher.

---

## Execution order

1. **Pre-clustering redundancy assessment first.** Run mmseqs2 at
   {1.00, 0.95, 0.90, 0.80} on each major protein function; report
   cluster-size distributions. Decides which thresholds are feasible
   (i.e., don't degenerate into one giant cluster per function).
2. **Experiment B** (priority). mmseqs2 cluster-disjoint routing
   sweep on `flu_ha_na` and `flu_pb2_pb1` at the feasible thresholds
   from step 1.
3. **Compare to current seq_disjoint baseline** via both plot variants
   (as-is and sample-size-controlled).
4. **Experiment A only if needed.** If the cluster-disjoint sweep is
   inconclusive (e.g. modest AUC drop but unclear if biology or
   representation), run cosine-controlled to separate the two.

---

## Cross-references

- `docs/plans/2026-05-07_leakage_diagnostics_plan.md` — parent plan;
  this is the detailed design for its Exp 5.
- `docs/plans/2026-05-14_regime_aware_coverage_plan.md` — the
  regime-aware coverage proposal; sequential with this plan (cluster
  routing at Phase 3, regime-aware coverage at Phase 4). Both can be
  enabled independently or together.
- `docs/methods/leakage_definitions.md` — mode #4 (cluster leakage) is
  what this plan addresses.
- `docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md` — the
  empirical motivation: ~48% of test HA proteins have a train neighbor
  at ≥99.5% aa identity under current `seq_disjoint`.
- `docs/results/2026-05-11_exp4a_seq_disjoint_results.md` — the
  `seq_disjoint` baseline that these experiments compare against.
- `docs/methods/gto_format_reference.md` §9 — CDS reconstruction recipe
  (needed only if nt-CDS clustering is ever run as a control).
