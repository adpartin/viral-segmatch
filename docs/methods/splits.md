# Split methods for viral-segmatch

Third in the methods chain: **`leakage.md` → `clusters.md` → `splits.md`**.
`leakage.md` defines the leakage modes and the "biology learning"
criterion. `clusters.md` covers mmseqs2 clustering mechanics and
Flu A corpus structure. **This doc covers everything about turning
the corpus into train / val / test partitions** — routing modes,
atoms, bin-packing, k-fold cross-validation, and the relation to
prior-art split strategies.

This doc carries **methods + structural numbers only**. Model-
performance numbers (F1, MCC, AUC) for specific routings live in
`docs/results/`; cross-refs are inline at each result-bearing claim.

---

## 1. Holdout (single split)

### 1.1 The implemented routing modes

| Mode | What it constrains | Atom | Algorithm |
|---|---|---|---|
| `random` | nothing | one isolate | random shuffle on isolates |
| `seq_disjoint` `hash_key=seq` (default) | same protein cannot span splits | bipartite CC on `(seq_hash_a, seq_hash_b)` | bipartite-CC LPT-greedy |
| `seq_disjoint` `hash_key=dna` | same full contig (UTR + CDS + intron + UTR) cannot span splits | bipartite CC on `(dna_hash_a, dna_hash_b)` | bipartite-CC LPT-greedy |
| `cluster_disjoint` bilateral (`single_slot: null`, default for the mode) | similar proteins cannot span splits on EITHER slot | bipartite CC on `(cluster_id_a, cluster_id_b)` | bipartite-CC LPT-greedy |
| `cluster_disjoint` single-slot (`single_slot: 'a'` or `'b'`) | similar proteins cannot span splits on the constrained slot only | per-cluster pair-set on the constrained slot | per-cluster atom LPT-greedy |
| `metadata_holdout` | isolate filter on metadata axes (host / year / subtype / geo / passage) | filter-defined isolate partition fed through `random` dispatch | `compute_metadata_holdout_isolates` overrides train/val/test isolate sets |

Notes:
- `seq_disjoint` does NOT have a single-slot variant; `hash_key` selects
  the hash family, not the constrained slot. Single-slot routing only
  exists under `cluster_disjoint`.
- `metadata_holdout` is mutually exclusive with the cluster_disjoint /
  seq_disjoint dispatch — it provides the isolate partition; `random`
  mode actually does the dispatch (enforced in
  `dataset_segment_pairs_v2.py:2953`).

### 1.2 Atoms — the indivisible routing unit

The router treats the corpus as a set of "atoms" that cannot be split
across train / val / test. What an atom is depends on the routing mode:

- **`random`**: atom = one **isolate** (an entire isolate's pairs go
  to one split).
- **`seq_disjoint` / bilateral `cluster_disjoint`**: atom = a
  **bipartite connected component** on the (slot A key, slot B key)
  graph. A CC can span multiple keys on each slot, linked through
  shared isolate pairs. One atom per CC.
- **Single-slot `cluster_disjoint`** (`single_slot='a'` or `'b'`):
  atom = **the pair-set of one cluster on the constrained slot**.
  All pairs whose constrained-slot sequence belongs to cluster K stay
  together; the unconstrained slot's partition is whatever those pairs
  happen to bring along (see § 1.6 for the separation consequences).
  One atom per cluster on the constrained slot.
- **`metadata_holdout`**: atoms are pre-defined by the filter
  (isolate sets per split); no bin-packing.

### 1.3 LPT-greedy bin-packing

For every mode except `random` and `metadata_holdout`, atoms are
bin-packed into 80 / 10 / 10 by **LPT-greedy** (longest-processing-time-
first deficit-fill):

```
targets:      train = 80%   val = 10%   test = 10%
biggest atom first → goes to the bin with the largest open quota →
   typically train first, then val and test fill from medium-sized atoms.
```

LPT-greedy is the "longest-processing-time-first" heuristic for bin
packing. It's optimal up to the largest atom: if the largest atom
exceeds the largest bin's target, the routing is structurally
infeasible at 80 / 10 / 10. The **largest-atom fraction** is therefore
the quantity to check before committing to a `(pair, alphabet,
threshold)` configuration (§ 1.9).

LPT-greedy is closely related to sklearn's `GroupShuffleSplit` and
`GroupKFold` — all three respect "atoms can't be split across splits"
(sklearn calls atoms "groups"). The difference is in the assignment
step: `GroupShuffleSplit` assigns atoms randomly, which can land a
30%-of-corpus atom in train and miss the 80 / 10 / 10 target by a wide
margin. LPT-greedy assigns atoms biggest-first to the most empty bin,
matching the target ratios as closely as the largest-atom constraint
allows.

> **Behavioral note — determinism.** LPT-greedy as implemented in
> `_split_helpers.py::cluster_disjoint_route_pos_df` and
> `_pair_helpers.py::seq_disjoint_route_pos_df` is **fully
> deterministic given the atom set and target ratios**. Re-running
> with a different `master_seed` reproduces the same partition.
> The `seed` parameter is accepted and recorded in audit JSON for
> traceability but is **not consumed** by the bin-packer
> (`_pair_helpers.py:631-633`, "reserved for future tie-break
> shuffling"). Per-fold variance under cluster_disjoint comes from
> the k-fold path (§ 2), not from re-shuffling holdout seeds.

**The router never drops pairs.** When the largest atom exceeds the
80 % train quota, LPT-greedy still places the whole atom in train —
train just overflows its target. The router does not split atoms and
does not discard boundary pairs; "infeasible" in this doc means "the
achieved 80 / 10 / 10 ratios drift", not "pairs are lost". The audit
JSON's `pairs_dropped` is always 0 in practice (verified across all
production builds — see `docs/results/2026-05-21_bicc_pair_drop_audit.md`).
Sub-feasibility consequence: val and test starve — e.g., HA/NA aa
t095 bilateral produces a 98.5 / 0.76 / 0.76 split rather than
80 / 10 / 10, even though every pair is routed somewhere.

### 1.4 The four cluster / hash routings — quick reference

| Mode + alphabet | Slot key | Two pairs share an atom iff … |
|---|---|---|
| `seq_disjoint` `hash_key=seq` | `seq_hash = md5(prot_seq)` exact | identical protein on a slot |
| `seq_disjoint` `hash_key=dna` | `dna_hash = md5(contig.dna)` exact | identical full contig (UTR + CDS + intron + UTR) on a slot |
| `cluster_disjoint` `cluster_alphabet=aa` | mmseqs2 aa cluster id at chosen threshold | aa-similar protein on a slot |
| `cluster_disjoint` `cluster_alphabet=nt` | mmseqs2 nt cluster id at chosen threshold, keyed on `cds_dna_hash` | nt-similar CDS DNA on a slot (UTRs and introns excluded) |

`cluster_disjoint` has the additional `single_slot` knob (§ 1.1) that
selects between bilateral and single-slot atom definitions. The four
routings span two algorithmic families both routed by bipartite-CC
LPT-greedy on the bilateral path and by per-cluster atom LPT-greedy on
the single-slot path.

### 1.5 Routing equivalence and non-equivalences

**`aa cluster_id100` ≈ `seq_disjoint hash_key=seq`** — almost identical
on Flu A. Both partition on the protein, both require exact match.
Differences come from mmseqs internals only: the `-c 0.8 --cov-mode 0`
coverage rule (§ 3.2 of `clusters.md`) may merge a longer protein and a
fragment that's covered at ≥ 80 % by the shorter at 100 % identity; an
md5 hash would not. On Flu A the length variation within a protein is
tiny (std ≤ 2.8 aa per `gto_format_reference.md` § 6.5), the X-fraction
is removed by Stage 1, so empirically the two produce essentially the
same partition. Aa t100 component count ≈ seq_disjoint seq component
count.

**`nt cluster_id100` ≠ `seq_disjoint hash_key=dna`** — they cluster
**different sequences**. `seq_disjoint hash_key=dna` keys on
`dna_hash = md5(contig.dna)` — the full contig including 5' UTR + CDS
+ intron + 3' UTR. `nt cluster_id100` keys on
`cds_dna_hash = md5(cds_dna)` — just the spliced CDS, UTRs and introns
excluded. Concrete consequences:

- Two isolates with **identical CDS but different UTRs** →
  `seq_disjoint hash_key=dna` puts them in different components
  (different contig hash); `nt id100` puts them in the same cluster
  (same CDS hash).
- For **spliced segments** (M2, NEP), the contig hash includes the
  intron and is sensitive to intron length / sequence variation;
  `cds_dna_hash` skips the intron entirely. So a synonymous variation
  in the intron flips the contig hash but not the CDS hash.

Neither is a subset of the other on the partition order; they enforce
different leakage definitions.

### 1.6 Single-slot cluster_disjoint — what it enforces

**Source.** § 3.1 (`--min-seq-id` definition), § 5 (per-protein
residue mismatch caps) of `clusters.md`, and the mmseqs2 algorithmic
construction.

**Setup.** Under **single-slot cluster_disjoint** at identity threshold
`t` on slot X (e.g., HA-only), the mmseqs clusters covering slot-X
sequences are partitioned across train / val / test: every cluster
lands in exactly one split. By construction, any train slot-X sequence
and any test slot-X sequence live in **different clusters**.

The unconstrained slot (slot Y) carries **no separation constraint at
all** — not cluster_disjoint, not even seq_disjoint. The same physical
slot-Y sequence can appear in train and test (and val) if it pairs
with multiple slot-X sequences whose clusters land in different splits.
Whether slot-Y sequences actually leak across splits depends on the
corpus's biological coupling between the two slots: tight coupling
(e.g., HA cluster ≈ NA subtype on Flu A HA-NA, Cramér's V ≈ 0.85–0.98)
drags slot-Y along with slot-X; weak coupling (e.g., PB2-PB1 cluster
coupling V ≈ 0.41–0.76 on Flu A) leaves slot-Y free to leak.

This is the group-respecting property that
`sklearn.model_selection.GroupShuffleSplit` and `GroupKFold` also
provide, with one extra feature: **mmseqs clustering at threshold `t`
gives the groups a sequence-level meaning**. `GroupShuffleSplit` and
`GroupKFold` treat group labels as opaque IDs; cluster_disjoint's
groups carry the residue-identity tolerance baked into `t`. That
promotion — from opaque ID to similarity-controlled tolerance — is the
load-bearing claim of the approach.

**Within-cluster ceiling (hard, by construction).** mmseqs at threshold
`t` guarantees that every sequence in one cluster has identity ≥ `t`
to the cluster representative **over the aligned region (≥ 80 % mutual
coverage, § 3.2 of `clusters.md`)**. For an aligned length `L_a` this
caps within-cluster mismatches at approximately `(1 − t) × L_a`
residues. Per-protein caps at each threshold: § 5 of `clusters.md`.

**Cross-cluster gap (soft, by construction).** mmseqs would have
merged two sequences into one cluster if their identity were ≥ `t`.
Therefore between any two sequences in different clusters, mismatch
count is **typically greater than `(1 − t) × L_a` residues** over the
aligned region — otherwise the clustering would have grouped them.
Under single-slot cluster_disjoint at threshold `t`, this becomes
the **typical lower bound on the residue gap between train and test
on the constrained slot**:

> train slot-X sequence vs test slot-X sequence: typically
> > `(1 − t) × L_a` mismatches.

Choosing `t` is choosing the residue-level difficulty of the
generalization test. Lower `t` → wider tolerance → wider enforced gap
→ the model is evaluated on sequences structurally more different from
its training data.

"Typically" — not "always". The bound is **soft**, not deterministic.
§ 1.7 covers the failure modes; § 1.8 covers the empirical leakage
gauge.

**Worked example — HA on Flu A** (aligned length ≈ 567 aa under the
coverage rule; exact per-threshold caps in § 5 of `clusters.md`):

| Threshold | Within-cluster max mismatches | Cross-cluster typical mismatches | What single-slot HA-only enforces between train and test |
|---:|---:|---:|---|
| t100 | 0 | 0 (byte-identical over aligned region) | Trivial separation — same sequences allowed in train and test |
| t095 | ~28 | more than ~28 | Train / test HAs typically differ by 28+ residues |
| t094 | ~34 | more than ~34 | Wider gap; cliff edge for HA-only bilateral feasibility (§ 1.9) |
| t090 | ~57 | more than ~57 | Largest enforced gap, but HA-only bilateral routing degenerates here (§ 1.9) |

Many of those 28–57 residues sit in functional regions of HA — the
receptor-binding domain (residues ~117–265) and the major antigenic
sites (Sa, Sb, Ca, Cb around residues ~140–225). The model has to
generalize across that residue-level gap to score test pairs correctly.

### 1.7 Limitations of cluster-based separation

**Source.** Steinegger & Söding 2018 (Linclust paper, Nat. Commun.
9:2542) for the algorithmic mechanism; § 6.1 of `clusters.md` for
non-monotone observations; per-dataset `cluster_disjoint_audit.json`
for unconstrained-slot leakage numbers.

The "typically more than `(1 − t) × L_a` mismatches" bound in § 1.6 is
soft. Three independent failure modes break it.

**Failure mode 1 — approximate clustering near the boundary.**
`easy-linclust` is a single-pass, hash-based algorithm rather than a
global pairwise comparison. Each sequence is hashed into `m = 20`
k-mers (default) and aligned only against the longest sequence in each
k-mer group it lands in — not pairwise against all other members. Two
sequences whose true identity exceeds `t` can fail to share any
selected k-mer, or can land against different center sequences and
pass alignment against neither. They end up in different clusters
even though they should have merged. The Steinegger & Söding 2018
paper reports a ~28 % miss rate at 50 % identity (Fig. 3c). At the
operating thresholds in this doc (t095, t094, t090) the miss rate
is smaller — the higher the identity, the more likely two
near-identical sequences share at least one of their 20 selected
k-mers — but nonzero. Practical consequence: some fraction of
cross-cluster (train, test) pairs at threshold `t` have identity > `t`
over the aligned region, eroding the typical gap.

**Failure mode 2 — non-deterministic cluster topology across nearby
thresholds.** The algorithm depends on which seeds are selected and
which "longest sequence" each k-mer group anchors on, so small changes
in `t` can shift cluster boundaries in non-monotone ways. § 6.1 of
`clusters.md` documents this empirically: NP aa drops from 526
clusters at t095 to 149 at t094 then rises to 706 at t093 before
resuming the descent. The same sequence can land in cluster A at t094
and cluster B at t093 with different cluster co-members. Down-stream,
the *identity* of which sequences a given cluster contains is
sensitive to `t` near boundary regions, not just the cluster count.

**Failure mode 3 — the unconstrained slot.** Single-slot
cluster_disjoint enforces nothing on slot Y (§ 1.6). When biological
coupling between slots X and Y is strong, the unconstrained slot
follows the constrained slot's partition by proxy and the soft gap on
slot X extends to slot Y as well. When coupling is weak, the
unconstrained slot leaks freely.

The empirical residual leakage on the unconstrained slot is captured
per-dataset in `cluster_disjoint_audit.json` (full per-split-pair
breakdown) and surfaced at the top level of `dataset_stats.json` under
`slot_leakage_summary`. For the published Flu A sweeps at the
t095–t100 range, unique seq_hash leakage on the unconstrained slot
is:

| Routing | t100 | t099 | t098 | t097 | t096 | t095 |
|---|---:|---:|---:|---:|---:|---:|
| HA-NA HA-only (slot b = NA)     | 7.5 % | 5.7 % | 4.7 % | 4.2 % | 3.6 % | 3.4 % |
| PB2-PB1 PB2-only (slot b = PB1) | 7.2 % | 5.8 % | 5.5 % | 5.3 % | 5.2 % | 5.2 % |

Both pairs show the same shape: leakage drops as `t` drops, because
looser clusters absorb more diverse partners and the same
unconstrained-slot sequence becomes less likely to span multiple
constrained-slot clusters. HA-NA's drop is steeper than PB2-PB1's
(3.4 % vs 5.2 % floor at t095) — consistent with HA-NA's tighter
HA ↔ NA coupling absorbing the unconstrained slot into the constrained
slot's partition more cleanly. Detail on the coupling per pair:
`docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md` § "Cramér's V
coupling pre-check".

These three failure modes are independent — they compound rather than
cancel. The empirical residual leakage on a real corpus has to be
measured (§ 1.8), not derived.

### 1.8 Empirical leakage gauge — 1-NN cosine margin as concept

**Source.** The 1-NN cosine margin baseline
(`src/models/baselines/knn1_margin.py`) trained alongside the MLP on
each cluster_disjoint dataset. Per-pair / per-threshold model-
performance numbers live in `docs/results/`.

The audit in § 1.7 (Failure mode 3) gives the **direct** sequence-level
leakage measurement: how many unconstrained-slot sequences appear
across multiple splits. The 1-NN diagnostic in this subsection gives
the **complementary** measurement: of the residual leakage that
remains (border-similarity on the constrained slot plus
unconstrained-slot leakage), how much can a lookup-style baseline
exploit?

**The logic.** A 1-nearest-neighbor classifier predicts each test
pair's label from its closest training pair (cosine distance on the
pair feature vector). Under single-slot cluster_disjoint at threshold
`t`, every test pair has a "wider gap" to its nearest training pair
than under seq_disjoint, *to the extent the cluster gap is hard*. If
the gap is strict, 1-NN's prediction-by-nearest-pair degrades as `t`
drops. If the gap is soft — i.e., the failure modes in § 1.7 are
active — 1-NN can still find close training neighbors and stay
competitive.

So **1-NN performance trajectory across `t` is the empirical upper
bound on residual leakage**. A 1-NN F1 that drops sharply with `t`
indicates the cluster gap is breaking lookup-style baselines. A 1-NN
F1 that stays flat indicates residual border-similarity leakage that
lookup can still exploit.

**Cross-comparison with MLP** turns this into the "biology learning"
criterion from `leakage.md` § "When we say 'the model learned biology'":
MLP > 1-NN by a meaningful margin (initial bar: > 0.02 AUC) on a
single-slot cluster_disjoint test set is evidence the MLP has
generalizable representation beyond near-neighbor lookup. MLP ≈ 1-NN
is evidence it is doing nothing more.

Per-pair / per-threshold trajectories on Flu A are reported in
`docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`
(HA-NA HA-only) and
`docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md` (PB2-PB1
PB2-only), with the extended sweep down to t080 in
`docs/results/2026-05-28_HAonly_extended_idXX_sweep.md`.

### 1.9 Feasibility on Flu A

§§ 1.6–1.8 cover single-slot cluster_disjoint conceptually. This
section covers the bilateral cluster_disjoint mode and its empirical
feasibility on the Flu A corpus, and how single-slot extends the
feasibility ceiling. The § 1.3 question — "is the largest atom small
enough for 80 / 10 / 10?" — answered per (schema pair, alphabet,
threshold).

#### 1.9.1 The bipartite-CC framework (bilateral routing)

Under bilateral cluster_disjoint, both slots' clusters are
constrained. The atom is a **bipartite connected component** on the
(slot A, slot B) cluster graph — built from positive pairs, with an
edge `HA_ck — NA_cm` iff some pair has that combination:

```
HA aa clusters (slot A)             NA aa clusters (slot B)
   ┌─────────┐
   │ HA_c1   │ ─── pair (isolate i) ───►   ┌─────────┐
   └─────────┘                              │ NA_c5   │
                                            └────┬────┘
                                                 │
                              pair (isolate j) ──┘
                                ┌─►   ┌─────────┐
   ┌─────────┐                  │     │ NA_c9   │
   │ HA_c2   │ ── pair (k) ─────┘     └─────────┘
   └─────────┘
   ┌─────────┐ ─── pair (m) ────────► ┌─────────┐
   │ HA_c3   │                        │ NA_c7   │
   └─────────┘                        └─────────┘


Connected components on this bipartite graph:
   CC #1 = { HA_c1, NA_c5 }              ← pairs in CC #1 must split together
   CC #2 = { HA_c2, NA_c9 }
   CC #3 = { HA_c3, NA_c7 }
```

All pairs inside one CC must land in the same split — otherwise a
cluster on one side would appear in both train and test, defeating
the routing's purpose. The per-protein collapse trajectory (§ 6.1 of
`clusters.md`) predicts the resulting bipartite-CC sizes. When either
slot's clusters collapse into one mega-cluster — for example PB2 aa at
t090 has only 24 clusters absorbing 99.6 % of the corpus (§ 6.3 of
`clusters.md`) — the bipartite graph collapses into a single
mega-component, and the routing becomes structurally infeasible.

**Naming.** This routing goes by several names across docs, code, and
writeups:
- **bipartite-CC LPT-greedy** — the precise technical name (bipartite
  connected components of `(slot_A_cluster, slot_B_cluster)`,
  bin-packed via Longest-Processing-Time first-fit decreasing).
- **cluster_disjoint routing** — the user-facing config name
  (`dataset.split_strategy.mode: cluster_disjoint` with the default
  `single_slot: null`).
- **BiCC-Split** — paper / prose shorthand for "Bipartite
  Connected-Component Split", abbreviated **BiCC** or **bicc** in
  inline references.

These name the same algorithm. Committed docs prefer
**bipartite-CC LPT-greedy** on first mention and **bicc** for
subsequent references; `cluster_disjoint` is for code and config;
`BiCC-Split` is for manuscript prose where a memorable name helps.

The algorithm differs from DataSAIL's split heuristic (cluster + ILP)
in two ways: (a) routing operates on bipartite-CCs as atomic units,
never dropping pairs ("CC bin-packing never splits a component",
`_split_helpers.py:267`), where DataSAIL's I2/S2 explicitly drop
pairs that straddle folds; (b) bicc's LPT-greedy is a heuristic that
hits the requested split fractions within ~0.01 % on Flu A, where
DataSAIL solves an NP-hard ILP via a heuristic clustering pre-pass.

#### 1.9.2 Bilateral feasibility on Flu A

Source: the four feasibility CSVs at
`results/flu/{version}/runs/cluster_disjoint_feasibility/feasibility_<pair>_<alphabet>.csv`,
generated by `src/analysis/cluster_disjoint_feasibility.py`.
Consolidated plot: `bipartite_largest_pct_vs_threshold.png` from
`cluster_analysis_summary.py`.

Largest bipartite-component fraction (% of deduped pairs):

| Segments | Schema pair | Alphabet | t100 | t099 | t098 | t097 | t096 | t095 | t094 | t093 | t092 | t091 | t090 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4/6 | HA/NA   | aa | 49.0 | 79.6 | 88.4 | 92.7 | 95.8 | 97.8 | 98.7 | 99.0 | 99.0 | 99.0 |  99.8 |
| 4/6 | HA/NA   | nt |  1.5 | 69.4 | 91.0 | 95.7 | 97.8 | 98.2 | 98.5 | 98.9 | 98.7 | 99.1 |  99.1 |
| 1/2 | PB2/PB1 | aa | 38.4 | 81.0 | 92.6 | 97.2 | 98.6 | 99.5 | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1/2 | PB2/PB1 | nt |  2.9 | 59.7 | 93.9 | 97.2 | 98.2 | 99.1 | 99.3 | 99.3 | 99.5 | 99.5 |  99.5 |

A cell is structurally feasible for 80 / 10 / 10 if the largest CC is
≤ 80 % (train can fit it cleanly). What "infeasible" means in
practice: the router still places every pair (see § 1.3 — `pairs_dropped`
is always 0), but train absorbs the mega-CC and val / test drift
toward zero. Reading the table by that frame:

- **t100 (every cell):** feasible. Largest CC is at most 49.0 %
  (HA/NA aa — NA's stalk-length absorption on the aa side already
  pools ~ 7 % of pairs into a single CC at t100). Routing has room.
- **t099 (marginal on aa, comfortable on nt):** HA/NA aa at 79.6 %
  lands just under the ceiling. PB2/PB1 aa at 81.0 % is 1 pp over the
  ceiling — borderline feasible. nt cells (HA/NA = 69.4 %, PB2/PB1 =
  59.7 %) sit comfortably below the ceiling. Aa is tighter than nt by
  ~ 10 pp at t099 on both pairs.
- **t098 (above the ceiling on all four cells):** aa cells are
  88.4 % (HA/NA) and 92.6 % (PB2/PB1); nt cells are 91.0 % and
  93.9 %. All four cross the ceiling but margins are slim enough that
  t098 is closer to feasibility than the deeper thresholds.
- **t097 and below:** broken on every cell — val / test get a small
  fraction of the corpus (HA/NA aa t095 produces 98.48 / 0.76 / 0.76).
  Every pair is routed; the dataset exists; it just isn't usable for
  evaluation.
- **t094..t091 yields no bilateral recovery.** Largest CC creeps
  (near-)monotonically toward 100 % on every cell.

Largest-CC % is the algorithmic *input* (and the predictor of
feasibility); achieved-train % in any built run is the realised
*output*. When largest CC ≤ 80 % the two diverge only at the 4th
decimal. When largest CC > 80 %, achieved-train % tracks largest-CC %
closely (the bin-packer can't undo what the mega-CC dictates).

**Interpretation: feasibility ceiling is corpus-controlled.** The aa-
vs-nt feasibility gap at t099 reflects the alphabet's underlying
diversity structure plus the corpus's metadata-driven bipartite linking,
not an algorithm-alphabet confound. nt clustering does not unlock
lower thresholds via synonymous diversity — even at t098 nt sits at
91–94 % — because the *bipartite linking* between HA clusters and NA
clusters is determined by which isolates carry which (HA, NA)
combinations, and Flu A's small set of dominant HxNy subtypes × host
× year cells links most pairs into one mega-component well above t098.

See also: `docs/results/2026-05-21_bicc_pair_drop_audit.md` for the
no-drop audit; `docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md`
for the algorithm choice and its validation.

#### 1.9.3 Single-slot extends the feasibility ceiling

The table above is for **bilateral** cluster_disjoint (both slots'
clusters disjoint between splits). **Single-slot routing** (only one
slot's clusters constrained — atom = per-cluster pair-count on the
constrained slot rather than the bipartite component; see § 1.1, § 1.2)
relaxes the constraint and pushes the feasibility ceiling well below
the bilateral one.

**Source.** `src/analysis/single_slot_cluster_disjoint_feasibility.py`;
CSVs at
`results/.../cluster_disjoint_feasibility/single_slot_feasibility_<pair>_<alphabet>.csv`.
The legacy feasibility heuristic — `largest_atom ≤ 80 % AND
second_atom ≤ 20 %` — is conservative; the modern D3 check (§ 3.3)
admits more configurations (the long tail of small clusters fills
val / test cleanly even when the top two atoms are larger).

aa, top-2 atom sizes (% of deduped pairs) at each threshold (legacy
heuristic verdict in parentheses):

| Pair | Constrained slot | t095 | t094 | t093 | t092 | t091 | t090 |
|---|---|---|---|---|---|---|---|
| HA/NA   | a (HA)  | 13.5 / 11.5 (✓) | 13.8 / 11.5 (✓) | 23.8 / 21.2 (✗) | 24.6 / 21.3 (✗) | 25.1 / 21.4 (✗) | 25.4 / 21.4 (✗) |
| HA/NA   | b (NA)  | 15.5 / 12.2 (✓) | 21.1 / 12.3 (✓) | 18.4 / 11.2 (✓) | 22.4 / 21.2 (✗) | 22.5 / 12.6 (✓) | 22.7 / 11.2 (✓) |
| PB2/PB1 | a (PB2) | 15.5 / 13.9 (✓) | 15.7 / 14.4 (✓) | 35.0 / 24.8 (✗) | 57.3 / 25.4 (✗) | 72.6 / 26.4 (✗) | 95.6 / 2.8 (✗) |
| PB2/PB1 | b (PB1) | 78.4 / 4.7 (✓)  | 93.1 / 0.8 (✗)  | 99.2 / 0.0 (✗) | 99.9 / 0.0 (✗) | 100 / 0 (✗) | 100 / 0 (✗) |

**Readings.**

- **Bilateral and single-slot ceilings are decoupled.** Bilateral is
  broken from t098 down on every pair / alphabet. Single-slot HA-only
  and PB2-only stay feasible well past that under D3 — empirically
  down to t080 on HA-NA HA-only at perfect 80.00 / 10.00 / 10.00
  partition (see `docs/results/2026-05-28_HAonly_extended_idXX_sweep.md`).
- **HA-only and PB2-only cliff together at t094 → t093 on the legacy
  heuristic.** The largest atom jumps from ~ 14 % to ~ 24 % (HA) /
  35 % (PB2) and the second atom crosses the legacy 20 % ceiling.
- **PB1-only cliffs one step earlier (t095 → t094).** PB1 has
  unusually few clusters at t095 (2,033) compared to PB2 (6,491),
  so the largest atom is already at 78.4 % at t095 and crosses 80 %
  at t094.
- **NA-only stays feasible through t085 with one t092 hole.**
  Largest atom stays in the 15–34 % band; second atom is below 13 %
  except at t092 where it spikes to 21.2 %. The t092 hole is
  inherited from NA's non-monotone cluster count (§ 6.1 of
  `clusters.md`).

---

## 2. Cluster-disjoint splitting (pair-weighted only)

Project-specific reference for the three cluster-disjoint routing
variants used by the splitter: `1D-CD`, `2D-CD`, `2D-CD-test`. All
three are **pair-weighted** — the splitter partitions canonical pairs
(post-`pair_key` dedup), never sequences or records. Background on
the three cluster-weighting views (unique / records / pair) is in
`clusters.md` § 6.0.

### 2.1 Why pair-weighted: the splitter partitions pairs

The dataset builder routes canonical pairs to train/val/test. The
split-size targets (e.g., 80/10/10) are measured in pairs. The
cluster's "weight" for routing decisions must therefore be measured
in pairs:

- **Unique-weighted** undercounts — a cluster with N unique seqs
  can contribute many more than N pairs if its members have
  multiple partners (the multi-partner long tail; see `glossary.md`
  "Bipartite hub").
- **Records-weighted** overcounts — high-copy sequences that pair
  with the same partner across many isolates collapse to a single
  canonical pair under `pair_key` dedup. Records-count ≠ pair-count
  (Task 4 verification on memory.md: a HA seq with copy 1,702
  contributes only 127 distinct pairs).
- **Pair-weighted** is the right currency for the splitter's
  bin-packing problem.

This applies uniformly to all three routing variants below.

### 2.2 Derivation from unique-weighted clusters storage

The cluster parquet at
`data/processed/flu/<version>/clusters_<alphabet>/id<XXX>/<protein>_cluster.parquet`
stores `{seq_hash → cluster_id}` — one row per unique sequence (see
`clusters.md` § 6.0). Pair-weighted views are derived downstream via
a JOIN against the pair universe.

Pseudocode:

```python
# Pair universe — one row per canonical pair after pair_key dedup.
# Source: cooccurrence in cds_final for the schema pair, then
# canonical_pair_key(seq_hash_a, seq_hash_b) dedup.
universe = load_pair_universe(cds_final, schema_pair=('HA', 'NA'))

# Cluster parquet per slot.
cluster_a = pd.read_parquet('clusters_aa/id095/HA_cluster.parquet')
cluster_b = pd.read_parquet('clusters_aa/id095/NA_cluster.parquet')

# Attach cluster_ids to each pair row.
df = (universe
      .merge(cluster_a.rename(columns={'cluster_id': 'cluster_id_a'}),
             left_on='seq_hash_a', right_on='seq_hash')
      .merge(cluster_b.rename(columns={'cluster_id': 'cluster_id_b'}),
             left_on='seq_hash_b', right_on='seq_hash'))

# Each row now has (canonical pair, cluster_id_a, cluster_id_b).
# This is the input to all three cluster-disjoint variants below.
```

Scripts performing this JOIN pattern:
- `src/analysis/cluster_pair_weight_topk.py` — per-cluster
  pair-weight ranking (single-slot view).
- `src/analysis/cluster_disjoint_feasibility.py` — bipartite-CC
  feasibility (pair-weighted; multigraph view).
- `src/analysis/bipartite_graph_properties.py` — per-CC stats
  including bridges and cut nodes on the simple-graph projection
  (see `glossary.md` "Simple bipartite graph").
- `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df` —
  the production splitter (Stage 3).

**TODO — pair_key alphabet question (deferred, see BACKLOG
"Methodology ideas" #3)**: the current convention is
`pair_key = canonical_pair_key(seq_hash_a, seq_hash_b)` on PROTEIN
hashes regardless of alphabet — so the pair universe is shared across
aa, nt_cds, nt_ctg. An alphabet-specific pair_key (using
`dna_hash_*` for nt_cds, `ctg_hash_*` for nt_ctg) would inflate the
nt and nt_ctg universes by distinguishing synonymous-codon and
intronic/UTR variants that currently collapse. Under the
alphabet-specific convention, § 2.3–§ 2.5 below would operate on
different-sized universes per alphabet. Decision is pending; this
section assumes the current shared-universe convention.

### 2.3 1-D cluster disjoint (`1D-CD`)

**Atom**: one slot's cluster (see `glossary.md` "Atom").
**Routing rule**: no cluster appears in more than one split FOR THE
CONSTRAINED SLOT; the other slot is unconstrained.
**Bin-packing**: LPT-greedy (§ 1.3) on per-cluster pair weights.

Code/config:
- `split_strategy.mode='cluster_disjoint'`
- `split_strategy.single_slot='a'` (constrains slot A) or `'b'`
  (constrains slot B).

Use when `2D-CD` is infeasible (mega-CC too large for the chosen
split target) and pairing sub-pattern coupling across slots is
acceptable as residual leakage. See § 1.6 for the worked example
on Flu A HA-NA.

### 2.4 2-D cluster disjoint (`2D-CD`)

**Atom**: one bipartite CC (see `glossary.md` "Connected component";
the bipartite graph operated on is the co-occurrence graph —
see `glossary.md`).
**Routing rule**: no CC appears in more than one split —
equivalently, no cluster on either slot in more than one split.
**Bin-packing**: LPT-greedy (§ 1.3) on per-CC pair weights.

Code/config:
- `split_strategy.mode='cluster_disjoint'`
- No `single_slot` knob set (default bilateral).

Use as the default. Becomes infeasible when the largest bipartite CC
(mega-CC) exceeds the train-budget fraction — see § 1.9 for the
feasibility table on Flu A.

### 2.5 Test-only cluster disjoint (`2D-CD-test`)

**Atom**: one bipartite CC (same as `2D-CD`).
**Routing rule**: cluster-disjoint constraint enforced ONLY between
train and test; val is sampled from train's CC scope (val shares
HA/NA clusters with train by construction — leaky val).
**Test integrity**: preserved (test held out under cluster_disjoint).
**Val variance via K random val subsets**: model-selection-noise
estimate, NOT generalization variance. Honest reporting requires
explicit labeling.

Code/config:
- `split_strategy.mode='cluster_disjoint_test_only'` (pending
  implementation in `_split_helpers.py`, ~50-100 LOC).

Use when `2D-CD` bilateral is infeasible AND held-out test integrity
is the priority over val/train disjointness. Active exploration
direction at time of writing (2026-06-02; see memory.md
"Split-target policy" bullet).

### 2.6 Atom definitions — quick reference

| Variant | Atom | Per-atom weight | Routing constraint |
|---|---|---|---|
| `1D-CD` | One slot's cluster | Number of pairs whose endpoint on that slot is in the cluster | No cluster of the constrained slot in multiple splits |
| `2D-CD` | One bipartite CC | Number of pairs in the CC | No cluster on either slot in multiple splits |
| `2D-CD-test` | One bipartite CC | Number of pairs in the CC | Train ↔ test cluster-disjoint only; val unconstrained |

---

## 3. K-fold cross-validation

### 3.1 The `n_folds` knob

`dataset.n_folds` controls the dataset's partition shape:

- `null` (default) or `1` → single train / val / test split (holdout).
- Integer `N ≥ 2` → `N` CV folds written as `fold_0/`, `fold_1/`, …
  subdirectories under the Stage 3 output directory; Stage 4 auto-
  detects them.

Per-fold target ratios are k-dependent. The existing CV math at
`dataset_segment_pairs_v2.py:1922` derives `val_frac = val_ratio /
(1 − 1/n_folds)` and per-fold test target = `1/k`. Canonical
0.8 / 0.1 / 0.1 holds only at k = 10; at k = 5 the effective per-fold
ratios are 0.7 / 0.1 / 0.2 (train / val / test). D3 feasibility checks
(§ 3.3) apply to these k-derived targets, not the bundle-config
defaults.

> **Behavioral note — `n_folds=1` ≡ `n_folds=null`.** Both dispatch to
> the **holdout** path; sklearn `GroupKFold` requires `n_splits ≥ 2`,
> so the k-fold path cannot route a single "fold". Practically:
> setting `n_folds: 1` in a bundle produces the same output as omitting
> the key entirely. Don't read `n_folds: 1` as "one CV fold" — it is
> "no CV, just holdout". Dispatch logic in
> `_split_helpers.py:362-367` (cluster_disjoint) and
> `dataset_segment_pairs_v2.py:2910-2960` (top-level mode + n_folds).

### 3.2 Per-mode k-fold support

Per current code dispatch (`dataset_segment_pairs_v2.py:2865-2960`):

| `split_mode` | `single_slot` | Holdout (`n_folds=null`/`1`) | k-fold (`n_folds≥2`) | Where |
|---|---|---|---|---|
| `random` | n/a | random isolate split | sklearn `KFold` on isolates | `generate_all_cv_folds_v2` (line 1883) |
| `seq_disjoint` (`hash_key=seq`/`dna`) | n/a | bipartite-CC LPT-greedy | **not built** — raises `NotImplementedError` (line 2911) | seq_disjoint plan OoS |
| `cluster_disjoint` bilateral | `None` | bipartite-CC LPT-greedy on `(cluster_id_a, cluster_id_b)` | **not built** — raises `NotImplementedError` (line 2924, requires `single_slot`) | k-fold plan OoS #5 |
| `cluster_disjoint` single-slot | `'a'` or `'b'` | per-cluster atom LPT-greedy on the constrained slot | sklearn `GroupKFold` on constrained slot's `cluster_id`, then LPT-greedy on remaining k-1 atoms for train/val | `generate_all_cluster_disjoint_cv_folds_v2` (line 1976) → `cluster_disjoint_route_pos_df(n_folds=k)` |
| `metadata_holdout` (axis-based filter) | n/a | filter-defined isolate partition fed through `random` dispatch | **not built** — raises `NotImplementedError` (line 2947) | metadata_holdout plan |

**Placeholders for the three "not built" cells** — future work tracked
in `docs/plans/2026-05-28_kfold_remaining.md`:

- **`seq_disjoint` k-fold.** Would extend naturally: bipartite CCs as
  atoms, sklearn `GroupKFold` + per-fold LPT-greedy + D3, same pattern
  as single-slot cluster_disjoint. ~ 40 LoC. Skipped because
  seq_disjoint atoms on Flu A are typically small (largest CC ~ 20 %
  of pairs at HA-NA `hash_key=seq`), so k-fold feasibility is rarely
  the bottleneck.
- **Bilateral `cluster_disjoint` k-fold.** Bipartite CC atoms collapse
  to a mega-component at most thresholds below t099 on Flu A (§ 1.9.2);
  k-fold feasibility unattainable at the interesting thresholds. At
  t100 technically feasible but redundant with seq_disjoint k-fold.
- **`metadata_holdout` k-fold.** Orthogonal extension. Could be either
  GroupKFold-on-isolates within a filter pool, or per-axis-stratified
  KFold. Design unfinished.

### 3.3 D3 two-knob feasibility check

Every partition (holdout's three bins; k-fold's `k × 3` bins) must
independently satisfy:

- **`max_acceptable_drift_pp`** (default `0.05`) — max absolute fraction
  deviation from target on any bin (train, val, OR test).
- **`min_test_frac`** (default `0.05`) — test bin ≥ 5 % of total pairs.

If any bin fails either knob, the build raises with an informative
menu listing the failing fold(s) and their measured values (D4 below).

> **Behavioral note — D3 applies to holdout, too.** The check runs
> against holdout's single partition AND against each fold's partition
> in the k-fold path. There is no holdout exemption. This is the Phase 7
> regression decision in the k-fold variance plan: holdout configs
> that previously succeeded but now violate D3 (e.g., HA-NA aa
> bilateral t095 at 98.5 / 0.76 / 0.76) raise by-design under the
> current behavior. The intent is one consistent feasibility gate
> across both partition shapes — a config that's "feasible" should
> mean the same thing whether you build it as holdout or as k-fold.
> Implementation: `_split_helpers.py::_compute_d3_check` (line 169),
> applied to holdout at line 504 and per-fold in the k-fold dispatch.

Why both knobs (partial redundancy, but they answer different
questions):

- `max_acceptable_drift_pp` catches the **train-too-small** case (e.g.
  0.65 / 0.20 / 0.15: drift 15 pp fails, test fine). Undertrained
  model means worse metrics across all folds even with adequate test
  sample.
- `min_test_frac` catches the **test-too-small for metric stability**
  case (e.g. 0.85 / 0.11 / 0.04: drift 6 pp passes at default,
  test_frac 4 % fails). A test bin below ~ 5 % means seed-to-seed
  variance dominates the signal.

All bins are checked symmetrically — val drift propagates to test
metrics via early-stopping signal noise, so val gets no exemption.

### 3.4 `max_feasible_k` and the D4 refuse menu

**Derivation.** Per-fold target = `1/k`; drift allowed up to
`drift_pp`. The test bin can absorb up to `1/k + drift_pp`. Solving
for k: `k ≤ 1 / (max_atom − drift_pp)`. So:

- `max_feasible_k_strict` = `floor(1 / max_atom_frac)` — zero-drift
  formula (the bound when `drift_pp = 0`).
- `max_feasible_k_at_default_drift` = `floor(1 / (max_atom_frac − 0.05))`
  — at the deployed default `drift_pp = 0.05`.

Both columns are emitted by Phase 1's pre-flight CSV
(`single_slot_feasibility_<pair>_<alphabet>.csv` under
`results/.../cluster_disjoint_feasibility/`). The pre-flight uses
these as **necessary conditions** for k-fold feasibility; the
authoritative gate is the per-fold drift check at build time
(necessary AND sufficient).

Worked example. HA-only t095 has `max_atom_frac = 0.135`. At
zero-drift: `floor(1 / 0.135) = 7`. At default `drift_pp = 0.05`:
`floor(1 / 0.085) = 11`.

**D4 — refuse with informative menu.** When `n_folds > max_feasible_k`
at build time, OR when any per-fold D3 check fails, the build raises
with a structured message exposing the user's options:

```
Configuration (pair=HA-NA, alphabet=aa, slot=a, threshold=id093)
  requested k=5.
  max_feasible_k = 3 at this configuration.
  Options (require explicit config change):
    - reduce k to 3 (smaller N, same threshold)
    - relax to threshold id094 (k=5 feasible)
    - accept larger min_test_frac / max_acceptable_drift_pp (loosens
      feasibility but produces noisier per-fold metrics)
```

Why refuse over (a) silent holdout fallback or (c) silent auto-
downshift to `k = max_feasible_k`:

- **(a)** replaces fold variance with seed variance and labels them
  identically downstream. A reader of "HA-NA single-slot, mean F1 ± std
  across N folds" should be able to assume N is consistent across
  thresholds. (a) breaks that silently.
- **(c)** silently produces non-comparable fold counts across
  configurations in the same result table. Same comparability break.
- **(b) refuse** forces the user to commit per-config, which is the
  right place for the trade-off.

Two trigger points: **pre-build** (cheap comparison against the
Phase 1 pre-flight column) and **mid-build** (after all k folds are
constructed in memory but before any disk write — no partial output).
The failure is **deterministic for a given (config, atom set)**
because atom ordering is pinned per D1 of the k-fold plan; re-running
with the same config produces the same pass / fail outcome.

### 3.5 Per-fold audit schema

`cluster_disjoint_audit.json` gains a `folds` array (one entry per
fold) when `k > 1`. Each fold's entry mirrors the holdout audit schema
but adds `fold_id` (0..k−1), per-fold `feasibility_check` block
(both D3 knobs with `pass`/`achieved`/`threshold`), and the renamed
`max_target_deviation_pp` (`_pp` on the 0–1 fraction scale, replacing
the legacy `_pct`).

`dataset_stats.json` gains a `kfold_summary` headline block:

```json
"kfold_summary": {
  "k": 5,
  "max_feasible_k_strict": 7,
  "max_feasible_k_at_build_drift": 11,
  "build_drift_pp": 0.05,
  "all_folds_pass": true,
  "single_slot": "a",
  "cluster_alphabet": "aa",
  "atom_ordering_key": "(-size, cluster_id)",
  "composition_mode": null,
  "per_fold": [
    {"fold_id": 0, "drift_pp": 0.024, "test_frac": 0.198, "pass": true},
    {"fold_id": 1, "drift_pp": 0.031, "test_frac": 0.205, "pass": true}
  ]
}
```

The triple `max_feasible_k_strict` / `max_feasible_k_at_build_drift` /
`build_drift_pp` removes ambiguity about which `max_feasible_k` value
D4 actually used as the gate: `max_feasible_k_at_build_drift` is the
authoritative gate value at build time, `max_feasible_k_strict`
matches Phase 1's strict-formula column for cross-reference with
pre-flight tables, and `build_drift_pp` records the resolved
`drift_pp` at build time (may differ from the config default if the
user overrode it).

`composition_mode` is **reserved for forward compatibility** with the
OoS composition lever (`cluster_disjoint(slot a)` +
`seq_disjoint(slot b)`); currently always `null`. If implemented
later, the value extends to `'cluster_a_seq_b'` or `'cluster_b_seq_a'`
without requiring a schema migration on existing audit JSONs.

### 3.6 Stage 4 integration

`scripts/stage4_sweep.sh` auto-detects `fold_*/` subdirs under the
Stage 3 output directory and dispatches one training run per fold per
seed. `train_pair_classifier.py --fold_id` (lines 1060, 1202-1203)
already resolves `--dataset_dir / "fold_{fold_id}"` — matches the
emitted layout exactly. `FIRST_SEED` guard runs baselines on seed 42
only (avoids redundant LGBM / 1-NN runs across seeds; baselines don't
take a `master_seed` override).

---

## 4. Relation to prior-art split strategies

`leakage.md` § "Relation to prior-art split taxonomies" covers the
**leakage-diagnostic** half — Park & Marcotte 2012's C1/C2/C3 test-pair
composition classes and DataSAIL's "leakage ≈ maximize OOD" framing.
This section covers the **split-strategy** half: DataSAIL's algorithmic
recipes, the segmatch ↔ DataSAIL ↔ P&M cross-reference, and the
segmatch naming convention.

### 4.1 DataSAIL R / I1 / I2 / S1 / S2 recipes

DataSAIL (Joeres et al., Nat. Commun. 2025) proposes *split strategies*
(algorithmic recipes):

- **R**: random interaction-based.
- **I1 / I2**: identity-based, one- / two-dimensional (entity holdout
  on one or both axes).
- **S1 / S2**: similarity-based, one- / two-dimensional (cluster
  holdout via similarity matrix).

For 2D modes (I2, S2), DataSAIL *drops* pairs that straddle the fold
boundary ("interactions can get lost", DataSAIL main p3). Underlying
algorithm is cluster-then-ILP minimizing total cross-fold similarity
L(π), with a class stratification constraint C (e.g., balance positives
vs negatives in each fold). DataSAIL cites P&M as reference 8 in the
PPI motivation but does not use the C1/C2/C3 vocabulary.

### 4.2 segmatch ↔ DataSAIL ↔ P&M cross-reference

| segmatch | DataSAIL | P&M (test composition produced) | Notes |
|---|---|---|---|
| `split_strategy.mode=random` | R | C1-dominated (~ 99 % in CV) | Verified P&M main p2 |
| `seq_disjoint, hash_key=seq` | I2 (2D, R=1) | C3 only | Equivalent in *intent*; differs in algorithm — bicc routes whole CCs atomically, DataSAIL drops straddling pairs |
| `cluster_disjoint cluster_alphabet=aa id100` | ≈ I2 | C3 only | mmseqs2 t100 ≈ seq_disjoint hash_key=seq (§ 1.5) |
| `cluster_disjoint id<100` (t099, t095, …) | S2 (2D, R=1) | C3 + cluster-novel | Same intent; algorithm differs (mmseqs2 cascade + bicc-route vs spectral + ILP + drop) |
| `regime_aware_coverage` (8 regimes over host × hn_subtype × year) | C-stratification (informally) | (orthogonal to C-classes) | Same goal as DataSAIL's C constraint (preserve confounder distribution); different operational level (per-cell negative sampling vs per-fold class balance) |

P&M's C1/C2/C3 and DataSAIL's R/I1/I2/S1/S2 are not synonyms: P&M
classifies *test pairs*, DataSAIL classifies *split strategies*. A
DataSAIL split strategy *produces* a test set with a specific
P&M-class composition. Earlier internal notes ("C1 is R, C2 is I1,
C3 is I2") are directionally right but conflate the two conceptual
levels; the table above is the careful version.

DataSAIL's ILP path was tested on Flu A in
`docs/results/2026-05-24_datasail_lpi_results.md` and collapsed all
routings to a partition-shape constant — not a viable primitive on
this corpus. The bicc LPT-greedy heuristic is the chosen alternative.

### 4.3 Naming convention in segmatch

segmatch retains `seq_disjoint`, `cluster_disjoint`, and the
bipartite-CC routing terminology because: (a) the route-not-drop
algorithmic property is distinguishing and worth a clear name; (b)
the threshold sweep (t100, t099, t095, …) is a knob DataSAIL
absorbs internally — naming it explicitly is useful for the
8-major-pair sweeps. On first mention in writeups, cross-refer to
DataSAIL's I2 / S2 with parenthetical: e.g., "cluster_disjoint t095
(DataSAIL S2-equivalent at 95 % identity threshold; algorithm:
bipartite-CC LPT-greedy)". See § 1.9.1 above for the naming chain
(bipartite-CC LPT-greedy ≈ cluster_disjoint routing ≈ BiCC-Split
≈ bicc).

---

## 5. See also

- `docs/methods/leakage.md` — leakage taxonomy and vocabulary
  (prerequisite for this doc).
- `docs/methods/clusters.md` — clustering mechanics and Flu A corpus
  structure (prerequisite for §§ 1.6–1.9).
- `docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md` —
  historical design log for the k-fold path (D1–D5 + Phase 1–7
  implementation log).
- `docs/plans/2026-05-28_kfold_remaining.md` — active plan for the
  three "not built" cells in § 3.2 plus the OoS items.
