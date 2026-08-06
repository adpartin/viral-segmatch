# Reporting
1. Random splits of schema-pairs produce very high prediction scores for all pairs --> leakage concern (sequence-level -> cluster leakage) -- *TODO*: check how this is framed in the papers
- Show: 8 x 8 matrix of scores across all schema-pairs (the 28 unordered major-protein pairs); HA-NA LightGBM (nt_ctg k-mers) score; HA-NA MLP (nt_ctg k-mers) score; explain the modeling part (k-mers, pLM, gLM)
- Questions: TBD

2. Leakage -- the model memorizing near-duplicate sequences shared across splits -- is a known problem in sequence data (*TODO*: check how this is framed in the papers). Several methods address it by constructing cluster-disjoint splits/folds with OOD separation between them, to mitigate memorization. Such methods include GraphPart, SpanSeq, DataSAIL, etc. Only DataSAIL handles the 2-D (paired) case. However, DataSAIL does not scale to our data (it hangs on our data; their paper reports results on much smaller datasets). The idea: cluster sequences so that across clusters they are < t identical (the OOD guarantee).
- Show: screenshots of paper titles; a figure from each or selected papers highlighting the methods 
- Questions: need to look into these papers, and see vocabulary used to describe this "memorization"; need to determine the appropriate name to refer to this body of methods that propose generating these clusters

3. There are various tools available that can cluster sequences (list those tools). MMseqs is very popular. The papers that propose novel methods to generate OOD splits, or papers that benchmark existing methods (w/o proposing a novel method), often mention MMseqs, and sometimes benchmark their methods against it. 
- Show: TBD
- Questions: explain what's unique about MMseqs;

4. DataSAIL does not scale to our data. Instead of debugging DataSAIL, and because MMseqs is so popular (python API, scalable, huge user base), we use MMseqs to generate OOD clusters (-> cluster-disjoint folds).

5. The default clustering in MMseqs -- set-cover (easy-cluster / easy-linclust) -- assigns each sequence to a representative, so members are within t of their representative but two clusters can still be similar: it does not guarantee OOD. The connected-component build -- union-find over the all-vs-all easy-search graph (>= t identity, >= 0.8 coverage edges) -- is OOD by construction: across different components there is no >= t edge, i.e. across clusters sequences are < t identical. Note easy-search is a heuristic (k-mer prefilter), so it is not exhaustive -- it could miss a >= t / cov hit and leave near-identical sequences in different clusters; yet verify_ood_clusters.py certifies 0 such cross-cluster violations in our clusters.
- Show: UMAP of ESM-2 embeddings colored by cluster (connected-component/search vs set-cover), explain tXXX, show 1D barplots of the clusters, show embeddings of GenSLM (codons) and/or ESM-C (AAs) or k-mers color-coded by metadata (subtype, host)
- Questions: how to demonstrate that connected-component (search) single-segment clusters are OOD while set-cover clusters are not

6. 2D cluster-disjoint (2D-CD) splitting; atoms = bipartite CCs. It operates on the cluster-level bigraph, whose connected components (CCs) are the atoms. Explain mega-CC. Explain that lower tXXX is expected to make folds that are more OOD. Show at even t099 dominated by the mega-CC.
- Show: TBD
- Questions: TBD


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

