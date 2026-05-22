# aa vs nt cluster membership cross-tab (linclust id099) — mechanism

**Date:** 2026-05-22
**Status:** EXPLORATORY (one-pass cross-tab; resolves the open methodology question
from `clustering_overview.md` §6.1 / `2026-05-22_aa_cluster_algorithm_validation_results.md`
§ "Open questions").
**Companion data:** `results/flu/July_2025/runs/cluster_aa_nt_crosstab/`
**Analysis script:** `src/analysis/aa_nt_cluster_crosstab.py`

---

## Headline

Under symmetric `easy-linclust` at id099, **aa and nt clusterings are not
nested** — each alphabet finds within-cluster variation the other misses.
The "nt has FEWER clusters than aa" observation on PB2/PB1/PA/HA/NS1 is
the *net* of two opposing effects, not a uniform "nt is coarser" story.

| Function | n_aa | n_nt | nt/aa | nt clusters absorbing >1 aa | aa clusters spanning >1 nt |
|---:|---:|---:|---:|---:|---:|
| PB2 | 18,354 | 11,484 | 0.63 | 2,274 (19.8%) | 1,113 (6.1%) |
| PB1 | 17,209 | 14,990 | 0.87 | 2,185 (14.6%) | 1,115 (6.5%) |
| PA  | 18,520 | 11,184 | 0.60 | 2,074 (18.5%) | 1,058 (5.7%) |
| HA  | 22,679 | 12,150 | 0.54 | 2,594 (21.3%) | 1,180 (5.2%) |
| NP  | 10,483 | 11,627 | 1.11 | 1,636 (14.1%) | 1,117 (10.7%) |
| NA  |  9,369 | 12,092 | 1.29 | 1,275 (10.5%) | 1,197 (12.8%) |
| M1  |  1,764 | 10,227 | 5.80 |   329 ( 3.2%) |   285 (16.2%) |
| NS1 | 13,508 | 12,012 | 0.89 | 1,610 (13.4%) | 1,283 ( 9.5%) |

Two patterns to notice:

1. On every function, both directions of disagreement are non-trivial. nt
   clusters absorb multi-aa-cluster groups (10–21% of nt clusters) and
   aa clusters span multi-nt-cluster groups (5–16% of aa clusters)
   *simultaneously*.
2. The two effects move in opposite directions across functions:
   PB2/PA/HA/NS1 are nt-merger-dominated (Effect A below), NP/NA/M1
   are aa-splitter-dominated (Effect B below).

---

## Setup

For each of the 8 core Flu A proteins with CDS coverage (PB2, PB1, PA, HA,
NP, NA, M1, NS1):

- Load `clusters_aa/id099/<fn>_cluster.parquet` → per-prot-seq cluster
  membership (`seq_hash` = md5 of protein; `cluster_id` = aa cluster).
- Load `clusters_nt/id099/<fn>_cluster.parquet` → per-cds-seq cluster
  membership (`seq_hash` here is md5 of CDS DNA — historical column name).
- Join with `cds_final.parquet` (per-isolate `assembly_id` ↔ `seq_hash`
  (protein) ↔ `cds_dna_hash` (CDS DNA)).
- Result: per-isolate `(assembly_id, aa_cluster_id, nt_cluster_id)`.

Total isolates per function: 108,530 (all 8 functions are complete).

For each function we compute:
- aa-per-nt: for each nt cluster, count of distinct aa clusters it contains.
- nt-per-aa: for each aa cluster, count of distinct nt clusters it spans.

Per-function histograms are written to
`<out>/<fn>_id099_aa_nt_histograms.png`; top-20 mergers per function
written to `<out>/<fn>_id099_top_nt_mergers.csv`.

---

## Effect A — nt mergers (the "swarm" picture)

**Mechanism:** mmseqs's rep-based clustering puts members within identity
threshold of the *representative*, not pairwise. At id099, two members of
the same nt cluster can be as low as ~98% nt-identical to each other
(triangle bound). On a long sequence (PB2 ~2280 nt), that ~2% gap permits
up to ~46 nt mismatches, of which a fraction are non-synonymous. Any
single non-synonymous change shifts a protein into a different aa id099
cluster (since aa id099 = ~1% gap = ~7 residues on PB2's 759 aa). So a
single nt cluster can host many aa-distinct point-mutation variants of
one parent lineage.

**Walkthrough — PB2_10902** (the largest nt-merger nt cluster on PB2):
- 12,324 isolates (11.4% of all PB2 records).
- 6,593 unique CDS sequences.
- 1,893 unique protein sequences → 1,518 distinct aa id099 clusters.
- Internal composition:
  - 1 dominant aa cluster (PB2_13817): 5,068 isolates (41% of the nt
    cluster).
  - A second sizeable aa cluster (PB2_13821): 1,405 isolates.
  - A long tail: **1,022 aa clusters are singletons** (1 isolate each)
    inside PB2_10902. Median aa-cluster-size inside PB2_10902 = 1.

This isn't "1,518 unrelated aa lineages fused" — it's "one parent lineage
plus a swarm of point-mutation variants". The variants are aa-distinct
(each crosses the 1% aa threshold via a single residue change), but they
remain ≥99% nt-identical to PB2_10902's rep because their single
non-synonymous change is swamped by the ~2280 nt context.

Where Effect A dominates: PB2, PA, HA, NS1 (n_nt < n_aa). These are
either long proteins (PB2/PA/HA: ~2280/2150/1700 nt) or under enough
positive selection that point variants accumulate without erasing the
nt signature.

---

## Effect B — aa splitters (the "synonymous fragmentation" picture)

**Mechanism:** The opposite case. When a protein is highly conserved at
the aa level but accumulates synonymous-codon variation, all the
synonymous variants land in one aa id099 cluster (zero aa change) but
fragment across multiple nt id099 clusters (the synonymous nt changes
accumulate past the nt 1% threshold).

**Walkthrough — M1_1687** (the largest aa-splitter aa cluster on M1):
- 13,466 isolates (12.4% of all M1 records).
- 500 unique protein sequences (all within the same aa id099 ball).
- 7,255 unique CDS sequences (synonymous variants).
- These 7,255 CDS fragment across **3,898 distinct nt id099 clusters**.
- 2,691 of those 3,898 nt clusters are singletons.

M1 at 252 aa is very short and very aa-conserved; almost all variation
between strains is synonymous. The nt id099 rep-based balls capture only
a tiny window of that synonymous diversity, so the same aa cluster
fragments into thousands of nt clusters.

Where Effect B dominates: NA, NP, M1 (n_nt > n_aa). These are
characterized by high aa conservation at the cluster-rep level combined
with rich synonymous diversity.

---

## Why the "nt < aa" net direction varies across functions

The two effects compete:

| Regime | Functions | Direction |
|---|---|---|
| Effect A >> Effect B | PB2, PA, HA | nt << aa (ratio 0.54–0.63) |
| Effect A > Effect B (modest) | PB1, NS1 | nt < aa (ratio 0.87–0.89) |
| Effect A < Effect B (modest) | NP, NA | nt > aa (ratio 1.11–1.29) |
| Effect A << Effect B | M1 | nt >> aa (ratio 5.80) |

The two effects are driven by independent biology:
- **Effect A magnitude** scales with protein length × aa-substitution
  opportunities under selection. Longer proteins under positive selection
  (HA antigenic sites) and long polymerase subunits (PB2/PA) have many
  rare-aa-variant singletons that share an nt parent.
- **Effect B magnitude** scales with synonymous nt diversity per aa
  cluster — high for short conserved proteins (M1, NP, NA).

Neither effect is an artifact of the algorithm switch — they were
present under the prior asymmetric (easy-cluster aa + easy-linclust nt)
setup, just less starkly visible because easy-cluster's higher
sensitivity merged some of the Effect A "swarm" together at the aa
level too.

---

## Implication for cluster_disjoint

The cluster_disjoint routing currently treats aa and nt as independent
options at the same nominal id threshold (id099 means "≥99% identity"
in both). The cross-tab shows these are **two different partitions**,
not two views of the same partition:

- aa id099 partitions by protein-level identity (point variants are
  separated).
- nt id099 partitions by full-CDS identity (synonymous variation is
  separated, but small numbers of point variants per parent are not).

For the leakage doctrine the practical question is: which alphabet
gives a *more conservative* train/test split? The cross-tab doesn't
directly answer this — neither alphabet uniformly dominates. A pair
that crosses aa cluster boundaries (training novelty under aa
cluster_disjoint) might still share an nt cluster (no nt-level
novelty), and vice versa. The 1-NN cosine results in
`2026-05-15_cluster_disjoint_nt_results.md` showed nt id099 and aa
id099 sit at similar leakage levels on Flu A, which the cross-tab now
gives a mechanistic basis for.

---

## What this does *not* prove

- The "rep-based diameter" mechanism is a structural claim about
  mmseqs's clustering semantics, validated *indirectly* by the
  swarm/fragmentation patterns in PB2_10902 and M1_1687. We did not
  directly measure pairwise nt identities within PB2_10902 to confirm
  the worst-pairwise ~98% prediction. That would be a useful follow-up
  (compute `--alignment-mode 3` pairwise on the cluster's all-vs-all)
  but isn't needed to falsify the headline "aa and nt are not nested".
- The "selection drives Effect A" framing is interpretation, not
  measurement. We have not measured dN/dS within mega-mergers (e.g.,
  PB2_10902) to confirm positive selection. The numbers are consistent
  with a positive-selection interpretation, but a neutral
  random-drift point-mutation accumulation would also produce the
  same swarm pattern.

---

## Reproduce

```bash
source /homes/apartin/miniconda3/etc/profile.d/conda.sh && conda activate segmatch
python src/analysis/aa_nt_cluster_crosstab.py \
    --threshold 0.99 \
    --functions PB2 PB1 PA HA NP NA M1 NS1 \
    --data_root data/processed/flu/July_2025 \
    --out_dir results/flu/July_2025/runs/cluster_aa_nt_crosstab
```

Outputs:
- `crosstab_summary_id099.csv` — per-function summary (the table at top).
- `<fn>_id099_aa_nt_histograms.png` — per-function aa-per-nt + nt-per-aa
  histograms.
- `<fn>_id099_top_nt_mergers.csv` — top-20 nt clusters by distinct aa
  clusters absorbed (per function).

Runtime: ~30 s on 8 functions.

---

## Followups (not in this work)

1. Pairwise nt identity within PB2_10902 (the largest merger) — verify
   the "worst-pairwise ~98%" structural prediction directly.
2. dN/dS within PB2_10902 vs other top mergers — distinguish selection
   from neutral drift as the driver of Effect A.
3. Repeat the cross-tab at id098 and id100 to confirm the trend (Effect
   A should weaken as threshold tightens to id100; Effect B should
   weaken as threshold loosens to id098).
