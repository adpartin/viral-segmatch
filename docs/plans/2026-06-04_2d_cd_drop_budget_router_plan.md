# 2D-CD drop-budget router (mega-CC edge min-cut) — implementation plan

**Status: CORE IMPLEMENTED (P0–P3, 2026-06-04)** — cut module + router wiring (default-off, 8/8 bit-exact) + real build feasible + 9th regression golden. Remaining: §3.6 cut-quality validation, the balanced-routing option, Q4/Q5 relocate+rename, and P4 (train-and-score, a separate plan).

**Date.** 2026-06-04.
**Goal.** Add a routing path that makes **2D-CD** feasible below t099 by **cutting the mega-CC**: drop the minimum *straddling pairs* (edge min-cut) so the kept connected components bin-pack into 80/10/10. This operationalizes the verified result in `docs/results/2026-06-04_bigraph_megacc_structure_and_cutting.md`.
**Scope.** HA-NA and PB2-PB1, `aa` first (nt_cds via the production builder). **Out of scope:** the train-and-score sweep — that is the downstream experiment (b).

---

## 1. Background (verified, see the results doc)

- Below t099 the mega-CC exceeds the train target, so 2D-CD is infeasible (`splits.md` §1.7).
- The efficient recovery is an **edge min-cut** that drops only straddling pairs: HA-NA aa t095 reaches a feasible 80/10/10 at **0.9%** dropped (spectral) / 10.1% (KL). **Node-peel (81%) is a loose upper bound — do not use it.**
- **Mechanism-dependent:** community-structured pairs (HA-NA) cut cheaply; a conserved-collapse single-mega-cluster (PB2-PB1 at low t) does not (≥24%).
- The current router **never drops** (`splits.md` §1.3). This adds the drop-budget mode (the DataSAIL-S2 analogue named in `splits.md` §4).

## 2. Where it plugs in

The 2D-CD holdout path is `_split_helpers.py::cluster_disjoint_route_pos_df` (n_folds==1 → `route_holdout` on `component_id` atoms). The cut sits **between** "assign `component_id` to each pair" and "`route_holdout` packs the atoms":

```
pos_df → (cluster_a, cluster_b) → component_id          [existing]
       → [NEW] cut mega component(s) to feasibility:
              drop straddling pairs, re-label components → smaller CCs
       → route_holdout(LPT) on the now-feasible atom set   [existing]
```

So the cut is a pre-pass on the atom set; the bin-packer is unchanged. The dropped pairs are removed from the positives before negative sampling.

## 3. Design

### 3.1 Config surface — **decided: sub-knob (B), default off**
A `drop_budget:` sub-knob on the existing bilateral `cluster_disjoint` mode — not a new top-level mode (it is still 2D-CD with a cut step, composes with the existing dispatch, and defaults disabled so current behavior is byte-identical):
```yaml
split_strategy:
  mode: cluster_disjoint          # bilateral 2D-CD
  cluster_alphabet: aa
  cluster_id_threshold: 0.95
  drop_budget:
    enabled: false                # default off -> existing behavior unchanged
    cut_method: spectral          # spectral (default) | kl | none
    target_frac: 0.80             # largest CC must fit the train target
    drift_pp: 0.05                # LPT 80/10/10 feasibility gate
    max_drop_frac: 0.20           # refuse (with menu) if recovery needs more
    seed: 1
```

### 3.2 The cut — port from `bipartite_min_cut`
- Move the recursive-bisection cut into `src/datasets/_megacc_cut.py` (operational home), operating on the production pos_df's `(cluster_a, cluster_b)` edges. The standalone `bipartite_*` analysis scripts stay in `src/analysis` as diagnostics.
- `cut_method`: **`spectral`** (default; cheapest — 0.9% on HA-NA t095) | `kl` | `none`. **Must be deterministic** — pin the seed and confirm run-to-run identity (confirmed in P0: identical partition across two runs).
- `max_drop_frac = 0.20` guard: if recovering feasibility would drop more than 20% (e.g. the conserved-collapse pairs need ≥24%), **refuse with an informative menu** (like the D4 refuse in `splits.md` §3.4) rather than silently shedding a fifth of the data.

### 3.3 Post-cut atom routing — **decided: LPT now, balanced later**
- The cut yields many small CCs (HA-NA t095 → 397 atoms, largest 28%), which LPT packs cleanly (drift ~1.3%).
- **v1 uses LPT-greedy** (`route_holdout`, already implemented, deterministic). A **balanced multiway** option (size-matched train/val/test, no "biggest lump → train" bias) is deferred to a later increment — a separate routing primitive, not a v1 blocker.

### 3.4 Drop accounting + CC artifacts
The CC structure is currently ephemeral (built in `_split_helpers`, used for routing, only summarized). The drop-budget mode makes it first-class:
- `cluster_disjoint_audit.json`: `pairs_dropped` becomes non-zero. Add a `cut` block: `{cut_method, n_cuts, pairs_dropped, dropped_frac, largest_cc_frac_before, largest_cc_frac_after, lpt_drift, seed}`.
- **Per-CC table** (`cc_atoms.parquet`): `component_id → n_pairs, n_clusters_a, n_clusters_b, split assignment`, plus the **cut log** (edges/pairs cut, in order) and the **dropped pair_keys** — makes the split reproducible and feeds the post-hoc subtype/reassortment + cut-quality analyses without re-deriving the bigraph.
- `dataset_stats.json`: surface `dropped_frac` and a note that the split is **cross-community** (test composition can skew to rare subtypes — a property of the cut, not a bug).

### 3.5 Pair universe = the production builder
- Build the bigraph from the production `pos_df` (`create_positive_pairs_v2`), **not** `analysis.load_pair_universe`. This (a) fixes the nt_cds protein-dedup gap (the analysis loader always dedups on protein `seq_hash`), and (b) uses the actual filtered/scoped pairs — which differ from the 58,826-pair analysis universe — so feasibility is re-verified on the real set.

### 3.6 Validation — cut-quality diagnostic (tiered)

The bigraph carries no distance information (edges are co-occurrence, not similarity), so the cross-side sequence gap is a sequence-alignment operation — kept post-hoc, like 1-NN.

- **Free, every run (audit):** the a-priori floor `(1−t)·L` per slot (HA t095 ⇒ ~28 aa) + the cut structure (§3.4). Cluster-disjointness makes the floor apply between train and test (`splits.md` §2.3), but it is **soft**, not a hard bound.
- **Post-hoc, calibrated (occasional, run *with* 1-NN — not every build):** `mmseqs easy-search` of test-side cluster reps vs train-side reps (reps already on disk: `<PROT>_rep_seq.fasta`), per slot → best cross-side % identity per rep. Report min/median cross-side mismatches `=(1−id)·L` and the **count of cross-side pairs with identity ≥ t** (floor violations = near-boundary cuts, listed). Reps are a proxy — escalate to member-level only on flagged cluster pairs. Complements the 1-NN margin (rep-distance tests the *clustering* gap; 1-NN tests what the *model* exploits).
- **Optional inline flag (no per-run alignment):** precompute once per (data version, alphabet, t) a *near-boundary adjacency* (mmseqs rep all-vs-all, cluster pairs with identity in `[t−δ, t)`) as a producer artifact; the per-run check is then a set lookup (do any near-boundary pairs straddle train/test?). A k-mer-NN on reps is a lighter, uncalibrated proxy.
- **1-NN cosine margin** stays a post-hoc, occasional check (not per-run).

## 4. Bit-exact safety

- Default disabled → the existing **8 split-regression goldens** are unchanged; verify with `scripts/split_regression_harness.py check` before and after each increment.
- Once a drop-budget build is validated, **add a 9th golden** (a small HA-NA t095 drop-budget config) so the cut path is itself regression-guarded going forward.

## 5. Phasing

- **P0 ✓ (2026-06-04)** — spectral cut confirmed deterministic (identical partition; dropped=502 / 397 atoms on HA-NA t095, two seed=1 runs); baseline `harness check` **8/8 bit-exact**.
- **P1 ✓ (2026-06-04)** — `src/datasets/_megacc_cut.py` (ported from `bipartite_min_cut`); unit check reproduced the HA-NA aa t095 result **exactly** (502 pairs / 397 atoms / 7 cuts), row-accounting + determinism confirmed.
- **P2 ✓ (2026-06-04)** — wired the sub-knob `dataset_segment_pairs.py` → `split_dataset_v2` → `cluster_disjoint_route_pos_df` (cut before LPT) + `drop_budget_cut` audit; default-off **8/8 bit-exact**.
- **P3 ✓ (2026-06-04)** — full-corpus HA-NA t095 build (`flu_ha_na_cluster_t095_dropbudget`): mega-CC **97.8% → feasible 80.9/11.4/7.7**, dropping **0.51%** (299 pairs, 7 spectral cuts); `drop_budget_cut` audit + 299 dropped pair_keys persisted. **Golden captured** (`drop_budget_2d_aa`, N=20000: 94.5% → feasible 83.3/8.3/8.3, 4 cuts; round-trips bit-exact) — the cut path is the 9th regression guard.
- **Bundle** Q4 (relocate `build_mmseqs_clusters.py` → `src/preprocess`) and Q5 (rename `bipartite_*` → `bigraph_*`) into P1/P2 as the files move — one pass, no double churn.
- **P4 (separate plan)** — the train-and-score experiment (b).

## 6. Decisions (resolved 2026-06-04)

1. **Config surface:** sub-knob (B) on the existing bilateral mode, **default off** (§3.1).
2. **Default `cut_method`:** **spectral** (§3.2) — confirmed deterministic in P0 (identical partition across two seed=1 runs).
3. **Post-cut routing:** **LPT** for v1; balanced multiway deferred (§3.3).
4. **`max_drop_frac`:** **0.20**, refuse with a D4-style menu on exceed (§3.2).
5. **Subtype-awareness:** **stay blind** — the cut is unsupervised; subtype is annotated only post-hoc.

## 7. Risks

- **Determinism** of the spectral eigensolver — must pin/verify; production splits must be reproducible.
- **Mechanism-resistant pairs** (conserved-collapse) → high drop → the `max_drop_frac` guard turns this into an explicit refuse, not a silent quarter-corpus loss.
- **Cross-community test sets** — the cut routes whole subtype-communities to splits, so test can be the rare-subtype tail. Document it in the audit so downstream eval interprets the curve correctly (this is the (b) experiment's main confound).

## 8. See also

- `docs/results/2026-06-04_bigraph_megacc_structure_and_cutting.md` — the verified structure/cutting results this operationalizes.
- `docs/methods/splits.md` §1.3 (no-drop router), §1.7 (feasibility), §3.3–3.4 (D3/D4 gates), §4 (DataSAIL S2).
- `docs/methods/glossary.md` — mega-CC, straddling pair, edge min-cut, node-peel.
- `scripts/split_regression_harness.py` — the bit-exact guard.
