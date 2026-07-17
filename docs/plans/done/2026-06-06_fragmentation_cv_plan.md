# Fragmentation for cluster-disjoint CV: cluster-pair sampling + atom-fragmented folds

**Status: IN PROGRESS** (implementation underway)

Date: 2026-06-06
Owner: cluster-disjoint CV harness (`src/analysis/cluster_disjoint_cv_experiment.py` + `src/analysis/_cv_sampling.py`)

Related:
- `docs/results/2026-06-04_bigraph_megacc_structure_and_cutting.md` — mega-CC structure + edge min-cut characterization (atoms ≈ antigenic subtypes; spectral reaches 80/10/10 feasibility at 0.9% drop on HA-NA aa t095).
- `docs/plans/2026-06-04_2d_cd_drop_budget_router_plan.md` — the production drop-budget 2D-CD router (`src/datasets/_megacc_cut.py`), which uses the same edge min-cut for the splitter.
- `docs/methods/glossary.md` — canonical terms used below (**cluster pair**, **atom**, **mega-CC**, **straddling pair**, **edge min-cut**).
- Roadmap order (decided): **fragmentation → metadata-aware sampling → nt_cds → nt_ctg.** This plan is step 1.

---

## 1. Problem

The m-sweep (`cluster_disjoint_cv_experiment.py`, done — `results/flu/July_2025/runs/cluster_disjoint_cv/msweep_HA_NA_aa.png`) caps `m` positives **per connected component (CC)** and groups folds by `cc_id`. It shows F1-macro falling ~0.91→0.81 as `t` loosens t100→t095 at every `m`, while raising `m` grows N only 2–3× and barely moves F1 — evidence the t-degradation is cluster-structure-driven, not a dataset-size artifact, **within the 2–3× range the per-CC cap can reach.**

That range is small at low `t` for a structural reason. The per-CC cap is doing **two jobs at once**:

1. **redundancy control** — bounding how many pairs one component contributes, so a few large, densely-connected components don't dominate the training set; and
2. **a hidden feasibility hack** — at low `t` a single **mega-CC** swallows ~98% of pairs (HA-NA aa t095); capping it to `m` shrinks that one giant group to `m` pairs, so `GroupKFold` sees many balanced small groups instead of a single fold-sized group that cannot be split into K cluster-disjoint folds.

Because job 2 rides on job 1, the harness *cannot* unlock `N` at low `t`: drawing more pairs from the mega-CC (to grow `N`) would re-inflate it into a single fold-sized group and break the cluster-disjoint K-fold partition. So the `N` confound and the t-effect stay partly entangled, and the controlled "fix N, vary t" curve the experiment is meant to produce is not reachable with the current design.

> *Harness*, in this plan = the experiment driver `cluster_disjoint_cv_experiment.py` — the script that builds the positive/negative pair dataset, runs the CV folds, fits the k-mer models, and writes the score-vs-`t` curves. It is standalone analysis code, not the production Stage-3/4 pipeline.

**Fragmentation decouples the two jobs**, which is what unlocks the controlled experiment.

---

## 2. Conceptual model (the decoupling)

### 2.1 The substrate: three nested levels

The positives live in a three-level hierarchy on the cluster-level bigraph (nodes = clusters, edges = co-occurrences):

```
sequence pair    two co-occurring segments — one positive example   (a multigraph edge; what we train on)
  ⊂ cluster pair   a unique (cluster_a, cluster_b)                   (a simple-graph edge; the cap-m sampling unit)
      ⊂ atom        a connected component of the (cut) bigraph       (clusters + their cluster pairs; the GroupKFold split unit)
```

- A **cluster pair** is one edge `(cluster_a, cluster_b)`. Many **sequence pairs** can map to the same cluster pair — all with both endpoints ≥`t` identical to the two cluster representatives, so sequence pairs sharing a cluster pair are near-duplicates of each other (this is the redundancy the cap controls; it is sharp at the cluster-pair level, not the CC level).
- An **atom** is a connected subgraph: its **nodes are clusters**, its **edges are cluster pairs**. So a single atom (CC) contains *multiple* cluster pairs, wired together because they share cluster nodes.

*Example.* The three cluster pairs `HA_c1–NA_c1`, `HA_c1–NA_c2`, `HA_c5–NA_c2` form ONE atom over the clusters `{HA_c1, HA_c5, NA_c1, NA_c2}`: the first two share `HA_c1`, the last two share `NA_c2`, so all four clusters are path-connected. Each cluster pair carries one or more sequence pairs.

- **Disjointness is enforced on clusters (nodes), never on cluster pairs.** Connected components partition the nodes, so every cluster belongs to exactly one atom; "no cluster in two folds" is the cluster-disjoint guarantee. Cluster pairs (and the sequence pairs on them) are simply the content sitting inside one atom.

### 2.2 Two units, two jobs

Two independent units, matched to the two jobs:

| Unit | Glossary term | Role | Column |
|------|---------------|------|--------|
| `(cluster_a, cluster_b)` | **cluster pair** | **sampling** unit — cap `m` pairs per cluster pair (the biologist's "balanced representation of each cluster pair"; the *redundancy* axis) | `cluster_pair_id` |
| (sub-)component of the cluster bigraph | **atom** | **split** unit — `GroupKFold` groups (the *leakage* axis; pairwise cluster-disjoint) | `atom_id` |

- **`atom_id` is always the split group.** Under `strategy='natural'`, atoms = the natural CCs (today's `cc_id`). Under `strategy='cut'`, the **mega-CC is fragmented by edge min-cut** into smaller atoms, each still a bipartite CC of the kept graph, so still pairwise cluster-disjoint.
- **`cluster_pair_id` is the sampling unit.** Cap `m` per cluster pair = up to `m` sequence pairs from EACH cluster pair (`m=1` ⇒ exactly one pair per cluster pair; the example atom above then yields 3 positives, one per cluster pair). Available positives `N = Σ_clusterpair min(m, |cluster pair|)`. This grows `N` at low `t` (≈10K cluster pairs at t095) *without* re-inflating any one atom, **provided the atoms are fragmented to fold size first.**
- **Nesting invariant:** every cluster pair sits entirely inside one atom (both its endpoints share an atom once straddling pairs are dropped), so per-cluster-pair sampling and per-atom grouping are mutually consistent — all pairs of a cluster pair carry the same `atom_id`.

Why both are needed (neither alone suffices at low `t`):
- Fragmentation **without** cluster-pair sampling: atoms are still multi-cluster-pair; capping `m` per *atom* under-samples (a handful of atoms → a handful of positives). Useless.
- Cluster-pair sampling **without** fragmentation: at low `t` all the unlocked pairs live in one mega-CC = one atom → `GroupKFold` can't split it into K folds. Broken.

This matches the design the user stated: *"cluster pair for sampling, CCs for the split + cut."*

---

## 3. Design decisions

**D1. Split unit column.** Rename the `GroupKFold` group from `cc_id` → **`atom_id`** (glossary "atom"). Keep `cc_id` as a retained audit column (the pre-fragmentation component) so natural-vs-cut is inspectable. Under `natural`, `atom_id == cc_id`.

**D2. Sampling unit.** Add `cluster_pair_id` and make the sampling unit selectable:
- `sample_unit='cc'` + `strategy='natural'` → reproduces the existing (a) m-sweep exactly (back-compat).
- `sample_unit='cluster_pair'` + `strategy='cut'` → the new (b) fixed-N experiment.
Guard: `sample_unit='cluster_pair'` while a mega-CC exists and `strategy='natural'` produces one fold-sized atom → detect `largest_atom_frac > 1/K` and **require `strategy='cut'`** (hard error with the remedy), rather than silently emitting a degenerate fold.

**D3. Cut target = K-uniform, not 80/10/10.** The existing edge min-cut stops at LPT-feasibility for **80/10/10** (`_TARGETS` in both `bigraph_min_cut.py` and `_megacc_cut.py`). K-equal folds need a **uniform** target `{1/K, …, 1/K}`, which is **strictly tighter**: one atom at 80% is LPT-feasible for 80/10/10 (fills train exactly) but violates 5 equal folds. So K-uniform fragmentation **drops strictly more** straddling pairs than the cited 0.9% (which was the 80/10/10 figure). The exact K-uniform drop-% is unknown and must be measured — it sets both the fixed-N ceiling and the cut-bias magnitude (see §6, §8-Q1).

**D4. Cut method = spectral (default).** Spectral (Fiedler) reaches feasibility at far lower drop than KL on this graph (glossary: 0.9% vs 10.1% at t095 for the 80/10/10 target) — fewer straddling pairs dropped = less cut-bias. Expose `--cut_method {spectral,kl}`; default `spectral`. Spectral is unbalanced, so it may need more recursive cuts to drive *every* atom ≤ 1/K; report cut count + final `largest_atom_frac` and warn if `max_cuts` is hit before feasibility.

**D5. Reuse the analysis-side cut core; no cross-boundary refactor.** `_cv_sampling.py` is under `src/analysis`, so it can import `src/analysis/bigraph_min_cut.py` directly (analysis→analysis; no `datasets`→`analysis` violation). Reuse `min_cut_recursive`'s machinery — **not** `src/datasets/_megacc_cut.py` (that's the production splitter twin; importing datasets from analysis is allowed but unnecessary here and would entangle the two cut copies). The 3-way dedup of the bisection core (`bigraph_min_cut`, `_megacc_cut`, and their duplicated `_bisect`/`lpt_max_drift`) is **out of scope** — see §9.

**D6. Glossary.** All terms above already exist. The only addition: fold the **K-uniform fragmentation target** into the existing "Edge min-cut" entry (one clause: "…to an LPT-feasible partition for the target ratios — 80/10/10 for the splitter, or K-uniform 1/K for K-fold CV"). No new headword.

**D7. Positive and negative selection are separate code paths.** Positive non-random selection balances coverage across **cluster pairs** (the redundancy axis: don't over-draw from any one cluster pair). Negative non-random selection controls the **easy/hard difficulty mix** (the 8 metadata regimes; `src/datasets/_negative_regime_sampling.py`). Different objective, different inputs, different unit — they share only the abstract `strategy=` hook, not the selector body. (Answers the design question: do **not** unify the two samplers.)

---

## 4. Changes by file

### 4.1 `src/analysis/bigraph_min_cut.py` — generalize the cut target (back-compatible)

*Scope:* analysis-side recursive edge min-cut — fragments a schema pair's cluster-level bigraph by bisecting its largest connected component until the kept atoms are LPT-feasible for a target split ratio (CLI + the `min_cut_recursive` library entry point).

- `lpt_max_drift(sizes, targets=_TARGETS, ...)` already takes `targets`. Add a builder `uniform_targets(k) -> {f'f{i}': 1/k for i in range(k)}` so a K-fold caller can pass K equal bins.
- `min_cut_recursive(...)`: add a `targets: dict | None = None` parameter (default `None` → existing 80/10/10). Thread it into the `lpt_max_drift` call. **Public contract unchanged** when `targets is None` — the CLI and `bigraph_cut_subtype.py` (which call it positionally with `return_partition=True`) keep their exact behavior and return shape `(df, H_kept, dropped_edges)`.
- Factor the post-`weighted_simple` bisection loop into `fragment_weighted(H, *, targets, method, drift_pp, seed, kl_max_iter, max_cuts) -> (df, H_kept, dropped_edges)` that operates on an **already-weighted simple graph**. `min_cut_recursive` becomes `weighted_simple(G)` → `fragment_weighted(...)`. This is the entry point `_cv_sampling` calls (it builds its weighted simple graph directly from cluster-pair counts, no MultiGraph needed).

### 4.2 `src/analysis/_cv_sampling.py` — atoms + cluster-pair sampling

*Scope:* the evolving sampling half of the harness — assigns each positive pair its cluster pair / CC / atom, caps positives per sampling unit, and builds 1:1 within-fold negatives; written so the selection strategy can change without touching the fold structure.

- `assign_cc(...)` → generalize (keep the name or rename to `assign_atoms`; keep a thin `assign_cc` alias for callers). New signature:
  `assign_atoms(universe, slot_a, slot_b, alphabet, threshold, *, strategy='natural', k_folds=5, cut_method='spectral', drift_pp=0.05, seed=1) -> DataFrame`
  Output columns: `cluster_a, cluster_b, cluster_pair_id, cc_id, atom_id` (+ the hash cols already present).
  - Build cluster maps (as today), drop unmapped-endpoint pairs.
  - `cluster_pair_id` = factorize of `(cluster_a, cluster_b)`.
  - `cc_id` = natural connected components (as today).
  - `strategy='natural'`: `atom_id = cc_id`.
  - `strategy='cut'`: build the weighted simple graph (`a:<cluster_a>` / `b:<cluster_b>`, weight = #pairs on that cluster pair) directly from the cluster-pair counts; call `fragment_weighted(H, targets=uniform_targets(k_folds), method=cut_method, ...)`; atoms = `connected_components(H_kept)`; map node→atom; **drop straddling pairs** (endpoints in different atoms); `atom_id` = the kept atom.
  - **Invariant assertion** (cut path): every cluster (each `a:`/`b:` node) maps to exactly one atom — assert no cluster appears in two atoms. This is the cluster-disjoint guarantee; cheap, keep it in.
- `sample_positives(pairs, *, max_per=1, unit='cluster_pair', seed=0, strategy='random')`: group by `unit` (`'cluster_pair'` for new, `'cc'` for back-compat) instead of the hardcoded `cc_id`. Rename `max_per_cc` → `max_per` (the cap is per the chosen unit; keep `max_per_cc` as a deprecated kwarg alias for one cycle). Metadata-aware `strategy` hook stays stubbed (next roadmap item).
- `make_negatives(...)`: unchanged logic. Note in the docstring that under fragmentation a fold spans several atoms; the within-fold shuffle keeps both endpoints inside the fold, so cluster-disjointness holds at the fold boundary. (Known pre-existing limitation, unchanged: a within-fold negative can coincide with a true positive living in another fold → mild label noise; metadata-aware negatives will address it. Out of scope here.)

### 4.3 `src/analysis/cluster_disjoint_cv_experiment.py` — wire strategy + fixed-N

*Scope:* the harness orchestrator — builds the pair dataset per (schema pair, alphabet, `t`), runs cluster-disjoint `GroupKFold`, fits the k-mer models, computes metrics, and writes the score-vs-`t` curves. Standalone, not the Stage-3/4 production pipeline.

- `assign_cc(...)` call → `assign_atoms(..., strategy, k_folds, cut_method)`; compute `cluster_pair`/atom stats per `t`.
- `run_cv`: `GroupKFold(...).split(pos, groups=pos['atom_id'])` (was `cc_id`); sampling unit threaded through.
- New CLI flags: `--strategy {natural,cut}` (default `natural`), `--cut_method {spectral,kl}` (default `spectral`), `--sample_unit {cc,cluster_pair}` (default `cc`), and `--fixed_n INT|auto` for sub-experiment (b).
- **Fixed-N mode (sub-exp b):** when `--fixed_n` is set, per `t`: fragment (`strategy='cut'`) → measure available `N_avail(t)` after dropping straddlers → `fixed_n = min_t N_avail` (if `auto`) → subsample positives to `fixed_n` (uniformly over cluster pairs, respecting `m`) → run CV. Report per `t`: `n_atoms`, `largest_atom_frac`, `dropped_frac` (cut-bias), `n_avail`, `n_used`.
- Results CSV: add columns `strategy, cut_method, sample_unit, n_cluster_pairs, n_atoms, largest_atom_frac, dropped_frac, fixed_n`. Keep existing columns (`m`, `n_pos`, `f1_mean`, `f1_std`, …) so the (a) rows stay comparable.

### 4.4 Metrics + plotting (a function in `cluster_disjoint_cv_experiment.py`, not a new file)

*Scope:* per-fold scoring and the score-vs-`t` figures.

- **Metrics:** `fit_score` → `fit_eval`, returning a per-fold dict `{auc_pr, f1_macro, mcc}` (was F1-macro only). `auc_pr = average_precision_score(y, proba)` (needs `predict_proba`/decision scores), `f1_macro = f1_score(average='macro')`, `mcc = matthews_corrcoef`. Aggregate mean±std per metric over folds. (knn1's `predict_proba` is degenerate for AUC-PR; knn1 is dropped for Exp 3 anyway.) Naming follows repo convention: snake_case `auc_pr`/`f1_macro`/`mcc`; display `AUC-PR`/`F1-macro`/`MCC`.
- **Plot:** `plot_fixedN_curve(...)` — one panel per metric (AUC-PR, F1-macro, MCC) vs `t`, one line per model (mean±std), with `dropped_frac` (cut-bias) annotated per `t`. Output `scorecurve_fixedN_{A}_{B}_{alphabet}.png`. Keep `plot_m_sweep` for Exp 1–2 (its empty 4th panel can show the per-`t` `dropped_frac`/atom-count diagnostic).
- **CSV:** add `auc_pr_mean/std`, `mcc_mean/std` alongside the existing `f1_mean/std`.

---

## 5. Verification (the cluster-disjoint invariant)

1. **In-code assertion** (§4.2): no cluster in two atoms — runs every `cut` call.
2. **Fold-level check** in `run_cv` (debug/assert mode): for each fold split, assert `set(train clusters) ∩ set(test clusters) == ∅` on **both** slots — the operational definition of 2D-CD. Cheap; gate behind a `--verify` flag so the full sweep isn't slowed.
3. **Reconcile counts** [RESOLVED 2026-06-07]: 10,756 = total cluster pairs (all CCs, the harness's measurement); 10,141 = cluster pairs inside the mega-CC only (`project_changelog.md` 2026-06-01). The glossary "Cluster pair" entry conflated the two — fixed to 10,756 total. Clusters are unchanged: natural largest-CC% measures 49.0/79.6/97.8 at t100/t099/t095, matching the glossary exactly.
4. **Sanity vs the existing cut**: at the 80/10/10 target, `fragment_weighted` on HA-NA aa t095 must reproduce `bigraph_min_cut.py`'s 0.9% spectral drop (regression guard that the refactor preserved behavior).

---

## 6. The experiments

Exp 1–3 share one objective — the **score-vs-`t` trajectory** under cluster-disjoint CV — and differ only in how dataset size is controlled. Exp 4 changes objective to **hard-negative performance**.

| Exp | Positive sampling | Split unit | Dataset size `N` (positives) | Status |
|-----|-------------------|-----------|------------------------------|--------|
| 1 | one pair per CC (`m=1`, per-CC) | CC | `= #CCs`; falls with `t` | done (the `m=1` slice of Exp 2) |
| 2 | up to `m` per CC (per-CC) | CC | `> Exp 1` at each `t`; grows with `m` | done (the m-sweep) |
| 3 | up to `m` per **cluster pair**, after **cut** | **atom** | **held fixed across `t`** | **this plan** |
| 4 | strategic **negative** regime sampling (positives as in 1–3) | — | — | step 2 (metadata-aware) |

Exp 1 and 2 are one run (`strategy='natural'`, `sample_unit='cc'`, the done m-sweep); Exp 1 is just its `m=1` column.

**Exp 3 — answers to the framing questions:**
- **`N` counts positives only.** Negatives are generated 1:1 per fold (`make_negatives`), so total examples ≈ `2N`. The fixed target is set on positives; the negative count follows.
- **Only components above the `1/K` target are fragmented.** The recursive cut bisects the *largest* component each step until every atom ≤ ~`1/K`; at low `t` that is the mega-CC, while smaller CCs already ≤ `1/K` pass through untouched.
- **After the cut, sampling is per cluster pair (up to `m` each), not per atom.** Atoms are the split unit only. To hit the fixed `N` at each `t`, choose `m` (and/or subsample cluster pairs) so `Σ_clusterpair min(m, ·) = N`. Sampling per atom would under-sample (a handful of atoms → a handful of positives).

**Exp 4 — hard-negative regimes (defined now, implemented in step 2).** Reuse the existing per-regime false-positive analysis (`analyze_stage4_train.py::analyze_level1_neg_regimes`, which writes `level1_neg_regimes.{csv,png}` over the 8 metadata regimes) on the harness's negatives, and use strategic negative sampling (`_negative_regime_sampling.py`) to raise performance on **hard negatives** (high metadata-overlap regimes) without letting **easy negatives** inflate the aggregate metrics. Needs the glossary additions (Easy/Hard negative — added with this plan) and the metadata-aware negative `strategy` hook (deferred to step 2). See *Easy negative* / *Hard negative* in `docs/methods/glossary.md`.

Scope for the first run (Exp 3): HA-NA, aa, t100..t095, K=5, spectral, models = mlp + lgbm (drop knn1 — slow at the higher `N` fragmentation unlocks). nt stays gated (NotImplementedError) — roadmap step 3.

---

## 7. Step-by-step task list

1. `bigraph_min_cut.py`: add `uniform_targets`, `targets=` param, factor out `fragment_weighted`; regression-check 80/10/10 spectral drop unchanged (§5.4).
2. `_cv_sampling.py`: `assign_atoms` (strategy natural|cut, cluster_pair_id, atom_id, invariant assert); generalize `sample_positives` to `unit`/`max_per`.
3. `cluster_disjoint_cv_experiment.py`: thread `strategy`/`cut_method`/`sample_unit`; switch `GroupKFold` groups to `atom_id`; add fixed-N mode; `fit_score` → `fit_eval` returning `{auc_pr, f1_macro, mcc}` per fold + new CSV columns; `plot_fixedN_curve` (one panel per metric).
4. Add `--verify` fold-level cluster-disjoint assertion.
5. Reconcile the cluster-pair count (§5.3); measure K-uniform `dropped_frac(t)` (§8-Q1).
6. Dry-run on t100 + t095 only (cheap) to validate atoms, drop-%, fixed-N floor, and the invariant **before** the full sweep. **Do not launch the full sweep without explicit confirmation** (standing instruction).
7. Glossary one-clause update (§D6).

---

## 8. Open questions / risks

- **Q1 (blocking the ceiling): K-uniform drop-%.** Unknown and larger than the 0.9% 80/10/10 figure (§D3). It sets the fixed-N floor and the cut-bias size. Measure in step 6 before committing to a fixed_n. If spectral can't drive the largest atom ≤ ~1/K at low `t` within `max_cuts`, options: raise K? accept a looser "largest ≤ 1/K + slack"? switch low-`t` to KL? — decide from the measured numbers, don't pre-commit.
- **Q2: cut determinism across seeds.** Spectral/KL are seeded; the atom partition (hence the folds) depends on `seed`. The current harness reports single-partition fold spread, not repeated-CV variance (existing `StratifiedGroupKFold` TODO). Fragmentation adds a *second* seed dependence (the cut). Note it; full repeated-CV-over-cut-seeds is a later variance study, not this plan.
- **Q3: cut-bias direction.** Dropped straddling pairs are the cross-subtype reassortant bridges (per the 2026-06-04 cut-subtype analysis). Removing them makes atoms more subtype-pure → the fixed-N curve measures t-effect on a *slightly* easier (more homogeneous) problem than the full universe. Report `dropped_frac` so this is legible; it's a feature (cleaner atoms) as much as a confound.
- **Q4: degenerate small atoms.** Spectral can shave a tiny piece per cut; many tiny atoms + one large remainder can stall the ≤1/K target. `fragment_weighted` must bisect the *largest* component each step (it does) and cap `max_cuts`; surface a clear warning if feasibility isn't reached.

---

## 9. Out of scope (explicit)

- **Cut-core dedup** across `bigraph_min_cut.py` / `_megacc_cut.py` / duplicated `_bisect`+`lpt_max_drift`. The `_megacc_cut.py` header already flags this as a later cleanup; this plan only extends the analysis copy.
- **Metadata-aware sampling** (within-atom selection by subtype/host/year; regime-matched negatives via `_negative_regime_sampling.py`). Roadmap step 2 — the `strategy` hooks are left stubbed for it; this plan keeps `strategy='random'`.
- **nt_cds / nt_ctg.** Roadmap steps 3–4; the harness's nt path stays `NotImplementedError` (feature/cluster alphabet mismatch).
- **Repeated/stratified grouped CV** (the existing `StratifiedGroupKFold` TODO) — a separate variance study.
- **Porting the harness to `src/datasets/`.** Too early. It is an experiment driver, not a production splitter, and it imports analysis-side loaders (`load_pair_universe`, the bigraph helpers); moving it under `datasets` would invert the allowed import direction (`datasets` must not import `analysis`). Revisit only if/when cluster-disjoint CV is productionized into the Stage-3 splitter.
