# Reporting

1. **Random splits.** Random splits of schema-pairs produce very high prediction scores for all 28 schema pairs --> leakage concern (sequence-level -> cluster leakage) -- *TODO*: check how this is framed in the papers
- Show: 8 x 8 matrix of scores across all schema-pairs (the 28 unordered major-protein pairs); HA-NA LightGBM (nt_ctg k-mers) score; HA-NA MLP (nt_ctg k-mers) score; explain the modeling part (k-mers, pLM, gLM)
- Questions: TBD

2. **Leakage.** The model memorizing near-duplicate sequences shared across splits -- is a known problem in sequence data (*TODO*: check how this is framed in the papers). Several methods address it by constructing cluster-disjoint splits/folds with OOD separation between them, to mitigate memorization. Such methods include GraphPart, SpanSeq, DataSAIL, etc. Only DataSAIL handles the 2-D (paired) case. However, DataSAIL does not scale to our data (it hangs on our data; their paper reports results on much smaller datasets). The idea: cluster sequences so that across clusters they are < t identical (the OOD guarantee).
* THIS PROBLEM IS CALLED IN VARIOUS WAYS IN DIFFERENT PAPERS BUT THEY GENERALLY REFER TO THE SAME PROBLEM (screenshots of papers and how they call it.)
- Show: screenshots of paper titles; a figure from each or selected papers highlighting the methods 
- Questions: need to look into these papers, and see vocabulary used to describe this "memorization"; need to determine the appropriate name to refer to this body of methods that propose generating these clusters

3. **Existing methods.** There are various tools available that can cluster sequences (list those tools). MMseqs is very popular. The papers that propose novel methods to generate OOD splits, or papers that benchmark existing methods (w/o proposing a novel method), often mention MMseqs, and sometimes benchmark their methods against it. 
- Show: TBD
- Questions: explain what's unique about MMseqs;

4. **Methods for paired data (DataSAIL).** DataSAIL does not scale to our data. Instead of debugging DataSAIL, and because MMseqs is so popular (python API, scalable, huge user base), we use MMseqs to generate OOD clusters (-> cluster-disjoint folds).

5. **MMseqs.** The default clustering in MMseqs -- set-cover (easy-cluster / easy-linclust) -- assigns each sequence to a representative, so members are within t of their representative but two clusters can still be similar: it does not guarantee OOD. The connected-component build -- union-find over the all-vs-all easy-search graph (>= t identity, >= 0.8 coverage edges) -- is OOD by construction: across different components there is no >= t edge, i.e. across clusters sequences are < t identical. Note easy-search is a heuristic (k-mer prefilter), so it is not exhaustive -- it could miss a >= t / cov hit and leave near-identical sequences in different clusters; yet verify_ood_clusters.py certifies 0 such cross-cluster violations in our clusters.
- Show: UMAP of ESM-2 embeddings colored by cluster (connected-component/search vs set-cover), explain tXXX, show 1D barplots of the clusters, show embeddings of GenSLM (codons) and/or ESM-C (AAs) or k-mers color-coded by metadata (subtype, host)
- Questions: how to demonstrate that connected-component (search) single-segment clusters are OOD while set-cover clusters are not

**Clustering results.**
- Show: 1D cluster barplots (for a specific t, eg t099); barplot of N clusters by (segment, alphabet); UMAP (ESM-2, ESMC, GenSLM); 

6. **Our method.** 2D cluster-disjoint (2D-CD) splitting; atoms = bipartite CCs. It operates on the cluster-level bigraph, whose connected components (CCs) are the atoms. Explain mega-CC. Explain that lower tXXX is expected to make folds that are more OOD. Show at even t099 dominated by the mega-CC.
- Show: TBD
- Questions: TBD

7. ****


# References

The `refs/` dir contains papers that might be relevant to our work, specifically in the context of OOD folds. We need to identify which ones are relevant, focus on those, and specify for each why it is (or isn't) relevant.

Refs (10): joeres_2025_datasail, rafi_2025_hashfrag, florensa_2024_spanseq, bernett_2024_guiding, hermann_2024_beware, bushuiev_2024_revealing, bernett_2024_cracking, teufel_2023_graphpart, steshin_2023_lohi, park_2012_pair_input_flaws

(park_2012 = the paired-interaction "pair-input flaws" paper -- directly on segmatch's 2D-pair leakage; added to the set. refs/ also holds *_supp / *_notes files -- out of scope for the scan.)

## How to run this review (staged, so it stays manageable)

Reading 10 papers (some ~30 pp) at once is not feasible in one pass. Do it in two stages, writing findings out as you go -- never hold all papers in context at once.

Stage 1 -- triage (cheap; all 10 papers):
1. Keyword scan (programmatic): per paper, extract body text (see rules), count terms, flag title/abstract hits, capture a snippet per discriminating hit -> write to `refs/keyword_scan.md` (matrix + snippets), NOT into this doc.
2. Abstract card: from each paper's abstract/intro, fill the per-paper card (below). Batch ~3-4 papers per step.
3. Score each paper on the relevance rubric (below) -> rank -> pick a shortlist (~top 4-5) to deep-read.

Stage 2 -- deep read (shortlist only):
4. One paper per step: read section-by-section, answer the specific questions with cited evidence, append that paper's card + answers here. Finish and write out one paper before starting the next.
5. Vocabulary harvest: consolidate the problem/method terms across papers into one table.
6. Synthesis: ranked relevance summary (rubric scores + one-line why relevant / not).

### Keyword scan rules
- Case-insensitive regex with stems/variants: generaliz*, out-of-distribution|OOD, leak*, memoriz*, shortcut, near[- ]duplicate, homolog*, mmseqs, cluster, disjoint, split, k?-?fold.
- Scope = the paper's OWN text only. EXCLUDE its References/Bibliography section (a term inside a cited title does not count).
- Report per term: count + whether it appears in the title/abstract (centrality). For the discriminating terms (memorization, shortcut, near-duplicate, OOD, homology), capture a short context snippet.
- cluster / split / fold are near-ubiquitous -> low-signal; weight the discriminating terms.

### Per-paper card (fields)
- thesis (1-2 sentences)
- method proposed (+ tool name), or "none (demonstrates problem / perspective)"
- data domain: 1D single-sequence vs 2D paired; protein / DNA / small-molecule
- split strategy: mmseqs / CD-HIT / graph-partition / embedding; identity threshold if any
- headline result
- vocabulary used for (a) the PROBLEM and (b) the METHOD family (verbatim terms)

### Relevance rubric (ranking; Q5)
Score each paper on these axes; composite -> rank, with a one-line "why relevant / not":
- Data structure: 2D / paired (like segmatch) > 1D single-sequence
- Split mechanism: identity-threshold clustering / mmseqs (like ours) > graph-partition > embedding-based
- Contribution: proposes a method we could adopt/benchmark > demonstrates leakage empirically > perspective / review
- Domain proximity: protein / genomic sequence > small-molecule / drug
- Directly adoptable or benchmarkable by us: yes / no

### Evidence citation (required for every claim/answer)
Cite: page number, section, and source type -- text / table / figure. Figures currently need rendering to view (poppler / PyMuPDF not yet available); mark figure-only evidence explicitly so it can be verified later.

### Keyword list
- generalization or generalize
- out-of-distribution or OOD
- leakage
- memorization
- shortcut
- near-duplicate
- homology
- mmseqs
- cluster
- disjoint
- split
- fold

### Specific questions (Stage 2, per shortlisted paper)
- Does the paper demonstrate/prove that the problem actually exists in the data (leakage, shortcut, memorization)? If yes, how?
- Does the paper cite other directly relevant methods?
- Does the paper directly benchmark its method against other relevant methods?
- Does the paper show that lower tXXX increases OOD across folds? (May not use "tXXX" terminology.)

---

## Stage 2 — deep-read synthesis (6-paper shortlist)

Six papers deep-read (opus, figures rendered via PyMuPDF) + adversarially verified (opus, figures
re-rendered); per-paper cards, four-question answers, and evidence logs are in **`refs/deep_reads.md`**;
ranked triage of all 10 refs in `refs/triage_ranked.md`. Verification: 6/6 checked, **3 SOUND / 3 MINOR,
0 MAJOR, 0 load-bearing claims refuted** (corrections logged in `deep_reads.md`).

### The four questions, across the shortlist

**Q1 — Does it prove leakage/shortcut exists, and how?** All six demonstrate it; the *method* varies.
- **Benchmark under leaky vs leakage-controlled splits:** `cracking` (2D PPI — every DL + baseline
  model collapses **0.9–0.99 → ~0.5** once train/test proteins are similarity-disjoint), `datasail`
  (lower cross-fold similarity L(π) → larger performance drop, across 1D + 2D), `spanseq` (non-aware
  test overshoots a common hold-out by ~0.07; train memorizes 0.98 vs 0.74).
- **Directly quantify train–test overlap:** `lohi` (56–88% of test within Tanimoto 0.4 of train),
  `park12` (CV test sets **>99% C1** while the population is only **19.2%** C1).
- **Prove only the mechanism (no downstream model):** `graphpart` (default set-cover/CD-HIT reduction
  *leaves* above-threshold cross-partition identities).
- **What they CALL it:** "data / information leakage" (datasail, cracking, spanseq); **three avoid the
  word** — park12 → "flaw in evaluation schemes / component-level overlap", lohi → "OOD / novelty /
  over-optimism", graphpart → "homology-driven overestimation". Concept universal, term not.

**Q2 — Cites other relevant methods?** Sparsely interconnected; citation direction tracks chronology.
- **datasail (2025)** — cites GraphPart, LoHi, CD-HIT, MMseqs2, FoldSeek, MASH, **Park & Marcotte** (the hub). Absent: KaHIP, SpanSeq, hashFrag.
- **cracking (2024)** — cites **Park & Marcotte** (its C1/C2/C3), CD-HIT, **KaHIP**. Absent: MMseqs, DataSAIL, GraphPart, LoHi, SpanSeq, hashFrag.
- **spanseq (2024)** — cites DataSAIL, GraphPart, CD-HIT, MMseqs, Petti & Eddy. Absent: Park & Marcotte, LoHi, hashFrag, KaHIP.
- **graphpart (2023)** — cites CD-HIT, MMseqs2, Hobohm-2, Petti & Eddy. Absent: everything later (Park not cited).
- **lohi (2023)** — cites the vertex-cut ILP lineage (**Cornaz et al. 2014** balanced vertex k-separator, +[63–68]), Butina. None of the sequence splitters.
- **park12 (2012)** — cites **Vert & Yamanishi 2005** (prior recognition), CD-HIT, HIPPIE.
- → Park & Marcotte is the common ancestor (datasail + cracking); DataSAIL is the modern hub; **no one cites hashFrag; only cracking cites KaHIP**.

**Q3 — Benchmarks its method against others?**
- **datasail** — vs LoHi, GraphPart, DeepChem(5 modes): lowest L(π) on 10/14 (1D); S2 hardest (2D); beats curated PLINDER-PL50 (0.0252 vs 0.0678).
- **cracking** — 6 DL + SPRINT + 8 ML baselines under leaky vs leakage-free: RF baselines match DL under leakage; all → ~0.5 without it.
- **graphpart** — vs CD-HIT + MMseqs2 (set-cover) + Hobohm-2, 8 datasets: GraphPart retains **94–99.99%** at perfect separation; set-cover discards ~28–58% *and* leaves violations.
- **lohi** — Hi-splitter vs Greedy (removes **1.5% vs 17%** on DRD2) + 10 ML models.
- **spanseq** — internal only (5 distance measures, 4 split policies; alignment-free ≈ alignment-based); no external tool head-to-head.
- **park12** — 7 PPI predictors × C1/C2/C3: CV ≈ C1 ≫ C2 > C3 for all.

**Q4 — Lower `t` → more OOD / ↓performance?** **No paper runs a `t` → OOD/performance sweep** — the
report's finding #6, now confirmed at depth across all six. The one threshold sweep that exists
(**graphpart Fig 4**) plots *sequences-retained* vs identity threshold, not OOD/performance. Everyone
else fixes one threshold and shows a *proxy* — "more separation → lower performance" via a discrete
ladder (datasail I1→S1→S2; spanseq increased-sim→random→SpanSeq; park12 C1→C3; lohi scaffold→Hi,
val PR AUC 0.872→0.603). **This is our differentiator: the `tXXX`-sweep-vs-OOD experiment is not in
this literature.**

### Cross-cutting findings (deep-read-confirmed)

1. **The literature names both our cut directions — and each has a *joint* cut+balance tool.**
   Our **edge min-cut** = DataSAIL's **minimum k-section** (edges removed; NP-hardness proven from it)
   = cracking's **KaHIP balanced weighted edge-cut**. Our planned **node-cut** = LoHi's **balanced
   vertex minimum k-cut** (ILP; Cornaz 2014 lineage) = graphpart's heuristic **"Separation"** node
   removal. **KaHIP (edge) and LoHi's ILP (vertex) each do the cut *and* the K-way balance in one
   optimization**, whereas we do cut-then-LPT in two stages — a concrete option to fuse them (caveat:
   LoHi's size constraint is lower-bound-only, so not a drop-in for exact fold sizes).
2. **Only DataSAIL does 2D out-of-the-box, and its scale ceiling is now a cited fact.** Its authors
   **excluded PCBA (>300k molecules) as "too big"** (Supp) and validated only to ~10⁵ (1D) / ~15k×13k
   (2D); the n-dependent cost is the O(n²) all-vs-all similarity + spectral clustering, not the
   constant-size ILP. So "DataSAIL doesn't scale to our data" is accurate and **scoped** — not a
   contradiction of the paper.
3. **CC-over-set-cover is empirically backed.** graphpart shows default MMseqs2 `easy-cluster` (greedy
   set-cover) discards ~28–58% *and still leaves cross-partition identity violations*, while its own
   partitioning retains 94–99.99% at perfect separation — independent support for our
   connected-component (OOD-by-construction) choice over set-cover.
4. **park12 is the foundational 2D citation** — the C1/C2/C3 (both/one/neither component seen) taxonomy
   and the >99%-C1-vs-19.2%-population argument are the original "why paired-input evaluation needs
   component-disjoint splits." (Its C1/C2/C3 → our-leakage-modes mapping is *our* reading; Park never
   says "leakage.")
5. **spanseq extends the stakes to model *development*** — leakage corrupts hyperparameter selection
   and breaks early stopping (validation stops tracking generalization), not just final scoring; its
   makespan + tabu-search fold-balancing is a concrete upgrade path for our LPT K-way step.

### Per-paper — what to cite it for
- **datasail** — the only 2D splitter; formal edge-cut (min k-section) + the edge-vs-vertex-cut dichotomy; the scalability ceiling that justifies our MMseqs route.
- **cracking** — proof that 2D paired-protein random splits collapse to chance; KaHIP as a reusable joint edge-cut+balance tool; the node-degree shortcut as a second leakage channel.
- **graphpart** — the node-cut ("Separation") algorithm reference; the empirical CC-over-set-cover evidence.
- **lohi** — the formal vertex/node-cut (our *planned* direction) with released code and a joint cut+balance ILP.
- **spanseq** — leakage corrupts model development; makespan/tabu balancing; the explicit "2D untested" that underwrites our novelty.
- **park12** — the foundational conceptual "why" for component-disjoint 2D evaluation.

### Vocabulary harvest (consolidated, verbatim)
- **Problem** — "information / data leakage" (datasail, cracking, spanseq); "similarity-induced
  leakage"; "memorization vs generalization"; "(node-degree / topological) shortcuts" (cracking);
  "overfitting / overestimated performance" (graphpart); "out-of-distribution (OOD) / distribution
  shift / novelty / over-optimism" (lohi, datasail); "near-duplicate"; "homology-driven overestimation"
  (graphpart); "flaw in evaluation schemes / component-level overlap / non-representative subsets /
  C1·C2·C3" (park12).
- **Method** — "leakage-reduced / similarity-aware splitting"; "identity- vs similarity-based
  (I1/I2/S1/S2)" (datasail); "homology partitioning vs homology reduction" (graphpart); "**minimum
  k-section**" = edge-cut (datasail); "**balanced vertex minimum k-cut / vertex separator**" = node-cut
  (lohi, Cornaz); "balanced (weighted) edge-cut / KaHIP KaFFPa" (cracking); "restricted single-linkage
  + Separation (node removal)" (graphpart); "makespan + tabu-search balancing" (spanseq); "connected
  component / giant component" (lohi, spanseq); "graph coarsening / Butina clustering" (lohi);
  "greedy set-cover clustering" (the anti-pattern; graphpart); "component-level overlap / C1/C2/C3" (park12).

*Verification: 6/6 deep reads adversarially re-checked (figures re-rendered where renderable); 0
load-bearing claims refuted. Full per-paper detail + correction log in `refs/deep_reads.md`.*

