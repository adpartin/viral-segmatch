# 2026-05-21 — BiCC pair-drop audit and improvement directions

## Concern (user)

We use a bipartite-CC + LPT-greedy router ("BiCC") for cluster_disjoint
splits (`src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`).
Going-in worry: the algorithm may be flawed or losing too many valid
pairs at lower identity thresholds. Two adjacent questions:

1. Are pairs being dropped at low thresholds, and if so, how many?
2. Can we redesign or extend BiCC to keep more pairs at those thresholds?

## Original response (pre-audit)

Five candidate improvement directions were proposed, ordered by likely
benefit on Flu A:

1. **Per-function (asymmetric) clustering thresholds.** Cluster HA at
   id095 (diverse), PB1 at id099 (collapses fast). Symmetric-threshold
   assumption is implicit in the current code; relaxing it is a config
   change, not an algorithm change. Highest expected yield.
2. **Pre-clustering corpus reduction at id100, THEN cluster_disjoint at
   id095.** Dedupe to one representative per id100 cluster first to
   shrink the bipartite graph. Changes what the model sees during
   training (frequency-weighted -> uniform), which may or may not be
   desirable.
3. **CC-splitting via internal min-cut** (METIS / KaHIP). Drops only
   the edges crossing the cut. Leakage guarantee changes from "no
   cluster crosses splits" to "no high-similarity edge crosses splits"
   — weaker but well-defined; equivalent to DataSAIL S2.
4. **Drop budget (DataSAIL S2-style).** Allow X% drops to escape
   mega-components. Pick budget + selector + audit unbiasedness on
   host/subtype/year.
5. **Karmarkar-Karp instead of LPT-greedy.** Tighter bin-packing for
   small N; can't help when the largest CC alone exceeds the largest
   split's quota.

The going-in recommendation: prototype #1 on PB2/PB1 first (smallest
change, preserves leakage guarantee). Use `/ultrareview` for in-tool
cross-model review of the algorithm; reserve external models for
algorithm-design audits, with a 1-page algorithm description as the
artifact that travels between reviewers. The DataSAIL bake-off (paused
on `feature/datasail-bakeoff`) is the gold-standard empirical
benchmark.

## Test performed (2026-05-21)

Read `dataset_stats.json` + `cluster_disjoint_audit.json` from 5
existing cluster_disjoint runs and summarized the audit fields
`pairs_dropped_in_routing`, `pairs_dropped_in_cluster_join`,
`achieved_pct`, and `max_target_deviation_pct`.

Runs inspected:

- `dataset_flu_ha_na_cluster_id99_20260520_211534`
- `dataset_flu_ha_na_cluster_id95_20260514_214704`
- `dataset_flu_pb2_pb1_cluster_id99_20260514_215910`
- `dataset_flu_ha_na_cluster_nt_id099_20260515_115012`
- `dataset_flu_pb2_pb1_cluster_nt_id099_20260515_115014`

### Findings

| Routing             | Train  | Val    | Test   | Max dev | Drops | Status        |
|---------------------|-------:|-------:|-------:|--------:|------:|---------------|
| HA/NA aa id99       | 80.00% | 10.00% | 10.00% | 0.0007% | 0     | clean         |
| HA/NA nt id99       | 80.00% | 10.00% | 10.00% | 0.0007% | 0     | clean         |
| PB2/PB1 nt id99     | 80.00% | 10.00% | 10.00% | 0.0011% | 0     | clean         |
| PB2/PB1 aa id99     | 87.12% |  6.44% |  6.44% | 7.12%   | 0     | marginal      |
| HA/NA aa id95       | 98.48% |  0.76% |  0.76% | 18.48%  | 0     | broken (drift)|

### Reframe

**BiCC never drops pairs.** `pairs_dropped_in_routing` and
`pairs_dropped_in_cluster_join` are 0 in every run. The "infeasibility"
manifests as **ratio drift** instead: at HA/NA aa id095 the routing
produces a 98 / 0.76 / 0.76 split. The dataset is built, the leakage
guarantee holds — but val and test get 1,107 pairs each instead of the
intended 14,597, a 92% capacity loss on val/test.

PB2/PB1 aa id099 is the marginal case the §9 narrative already flagged:
train absorbs 87.1% (vs 80% target), val/test get 6.4% each (vs 10%).
Usable, but the per-split regime composition drifts.

The user-visible problem is *"train absorbs the mega-CC, starving
val/test"*, not *"BiCC drops pairs"*. Same problem mathematically;
different framing for what to change.

## Updated recommendations

Reordered by relevance now that the no-drop status is confirmed:

1. **Per-function asymmetric thresholds (was #1, still #1).** Still
   the cheapest unlock with no leakage-guarantee change. The Flu A
   collapse profiles in §8.1 suggest the bottleneck slot for PB2/PB1
   is PB1 (collapses faster than PB2 at the same threshold). Test:
   cluster PB2 at id097 + PB1 at id099 and re-run the bipartite-CC.
   If the largest CC drops below 80%, this combination is feasible
   where symmetric id097 isn't. See direction #5 for the principled
   generalization (calibrating per-function thresholds to a uniform
   absolute mismatch budget rather than picking them ad hoc).

2. **Drop budget / CC-splitting (techniques #3 + #4 merged).** Now
   directly relevant because the alternative is the ratio drift above.
   Concrete drop sizes needed to recover 80/10/10 by hand:
   - PB2/PB1 aa id099: ~9,371 pairs (~7.1% of corpus)
   - HA/NA aa id095: ~26,975 pairs (~18.5% of corpus)
   The 7.1% case is plausibly worth a drop budget. The 18.5% case is
   on the threshold of "you may as well use a looser identity"
   (DataSAIL C2 dropped 51–71% on HA/NA, which dwarfs both).

3. **Pre-clustering reduction (was #2).** Lower priority unless we're
   willing to change the training-data distribution.

4. **Karmarkar-Karp (was #5).** Confirmed not helpful here. Cannot pack
   a 98%-of-corpus item into an 80% slot. Useful only at the marginal
   case (PB2/PB1 aa id099) where the gap is small.

5. **Absolute-mutation-tolerance clustering (generalizes #1).** Instead
   of a uniform identity threshold `t` across functions, pick a target
   absolute mismatch budget `ε` (in residues) and set per-function
   thresholds as

   ```
   t_f = 1 − ε / L_f
   ```

   where `L_f` is the median sequence length of function `f`. This makes
   every function admit roughly the same absolute number of residue
   changes per cluster, regardless of length.

   Concrete numbers for ε = 5 aa on Flu A medians (from
   `clusters.md` § 5's median-length column):

   | Function | L_f (aa) | t_f    | Equivalent idXX |
   |---|---:|---:|---|
   | PB2  | 760 | 0.9934 | ~id0993 |
   | PB1  | 758 | 0.9934 | ~id0993 |
   | PA   | 717 | 0.9930 | ~id0993 |
   | HA   | 567 | 0.9912 | ~id0991 |
   | NP   | 499 | 0.9900 | ~id099  |
   | NA   | 470 | 0.9894 | ~id0989 |
   | M1   | 253 | 0.9802 | ~id098  |
   | NS1  | 231 | 0.9784 | ~id098  |

   **Why this matters for BiCC.** At today's uniform id095, PB2 has
   26 aa clusters and PB1 has 50 — densely connected by isolates, the
   mega-CC forms. The same threshold admits ~38 aa changes per cluster
   on the polymerases, which over-pools them relative to HA/NA at the
   same threshold (which admit ~28 aa changes — 12% of HA's length vs
   PB2's 5%). Under ε = 5 absolute clustering, PB2 and PB1 each get
   thousands of clusters → the bipartite graph stays sparse → mega-CC
   needs a much looser ε before it forms. This is the principled
   version of #1: rather than picking per-function thresholds ad hoc
   ("PB2 at id097 + PB1 at id099") we calibrate them to a uniform
   biological strictness.

   **Why this matters for the classifier.** Today's uniform idXX has
   a side effect that's easy to miss: test-set sequences are at most
   `L_f × (1 − t)` residues away from their nearest train sequence,
   which varies by ~3× across functions (38 aa for PB2 vs 12 for M1
   at id095). The model is asked to generalize over different
   biological-novelty levels per protein. Under ε = 5 absolute, every
   test pair is at least ~5 aa novel on both slots — apples-to-apples
   across function pairs.

   **Caveats.**
   - `ε = 5` is itself a choice. Smaller `ε` = more clusters per
     function = more conservative leakage barrier but tighter
     feasibility ceiling. Sweep needed.
   - "Same absolute strictness across functions" assumes every aa
     substitution has equivalent biological impact across proteins,
     which isn't strictly true (a mutation in HA's receptor-binding
     domain ≠ one in M1's nuclear-export tail). A rigorous version
     would calibrate `ε_f` to per-function evolutionary rates from
     a flu phylogeny — but that's a much bigger lift, and ε-uniform
     is closer to biologically meaningful than t-uniform.
   - mmseqs2 takes a float `--min-seq-id`, so per-function `t_f`
     values fit the existing wrapper unchanged. The change is
     mostly in the config layer and the `cluster_id_threshold`
     key in `dataset.split_strategy` (which currently assumes a
     single threshold).

   Implementation cost: small. New config knob
   `dataset.split_strategy.cluster_thresholds_per_function: {PB2: t1,
   HA: t2, ...}` or `cluster_epsilon: 5`. Stage 1.5 (cluster
   generation) loops the existing easy-cluster invocation per
   function with its `t_f`.

## Next concrete steps (proposed)

- **Quick win:** prototype per-function asymmetric thresholds on
  PB2/PB1 aa. New `dataset.split_strategy.cluster_thresholds:
  {PB2: 0.97, PB1: 0.99}` knob; load two cluster parquets at the
  configured thresholds, run the bipartite-CC. ~1 day of work.
- **Documentation honesty:** update `docs/methods/clusters.md`
  §9 to clarify "infeasibility" = ratio drift, not pair drops. Add the
  achieved-train-% column to the §9 table. ~30 min of work. (Edits
  listed but not applied yet — see CLAUDE.md plan-doc convention.)
- **Algorithm review (deferred):** write a 1-page BiCC description as
  the cross-model-review artifact. Run `/ultrareview` after committing
  any algorithm changes from the prototype above.

## References

- BiCC implementation:
  `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`
- Audit JSON schema: `cluster_disjoint_audit.json` written by the
  Stage 3 saver.
- Per-segment cluster collapse: `docs/methods/clusters.md`
  §8.1.
- Feasibility ceiling table:
  `docs/methods/clusters.md` §10.2.
- DataSAIL bake-off (paused): `feature/datasail-bakeoff` branch +
  `docs/plans/2026-05-19_datasail_bakeoff_plan.md`.
