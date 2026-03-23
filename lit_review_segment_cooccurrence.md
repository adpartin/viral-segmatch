# Literature Review: Segment Co-occurrence Prediction in Segmented RNA Viruses Using Protein Language Model Embeddings and K-mer Features

**Generated via Claude.ai on 2026-03-11.** Citations and claims should be independently verified.

**No existing method directly addresses the binary classification task of predicting whether two viral protein/genome segments originate from the same isolate using protein language model embeddings combined with k-mer features.** This represents a genuinely novel contribution sitting at the intersection of four active research areas: viral reassortment detection, protein language models for virology, pairwise embedding comparison, and influenza computational genomics. The closest existing work—HAIRANGE (Wei et al., 2025), the DCA-based droplet sequencing approach (Lacombe et al., 2023), and the SOM-based codon feature method (Gong et al., 2021)—address related questions but none combine frozen PLM embeddings with pairwise interaction features for segment-isolate matching. Below is a comprehensive review of **44 papers** across these four areas.

---

## Area 1: Viral Reassortment Detection

This area encompasses methods for identifying segment co-occurrence, reassortment events, and segment linkage in segmented viruses. The field has evolved from purely phylogenetic methods toward ML-based approaches, but the dominant paradigm remains tree-based comparison.

### 1.1 GiRaF: robust, computational identification of influenza reassortments via graph mining
**Nagarajan N. & Kingsford C.** (2011) *Nucleic Acids Research* 39(6):e34

The seminal computational tool for automated reassortment detection. GiRaF searches large collections of MCMC-sampled phylogenetic trees for groups of incompatible splits using a fast biclique enumeration algorithm with statistical tests. It analyzes all **28 pairwise segment comparisons** in influenza's 8-segment genome to catalog reassortment events. While purely phylogenetic rather than ML-based, it established the core problem formulation of determining which segments share evolutionary history—directly related to the problem of predicting segment co-occurrence. **Adjacent.**

### 1.2 Bayesian inference of reassortment networks reveals fitness benefits of reassortment in human influenza viruses
**Müller N.F., Stolz U., Dudas G., Stadler T. & Vaughan T.G.** (2020) *Proceedings of the National Academy of Sciences* 117(29):17104–17111

Introduces CoalRe, a coalescent-based model for jointly inferring reassortment networks and the embedding of segment trees within those networks using MCMC. Implemented as a BEAST2 package, it allows joint estimation of **coalescent and reassortment rates** from full genome sequence data. Represents the gold-standard Bayesian approach to modeling segment evolutionary relationships. Computationally intensive but provides ground-truth reassortment events against which ML methods can be benchmarked. **Adjacent.**

### 1.3 TreeKnit: inferring ancestral reassortment graphs of influenza viruses
**Barrat-Charlaix P., Vaughan T.G. & Neher R.A.** (2022) *PLOS Computational Biology* 18(8):e1010394

A fast method for inferring Ancestral Reassortment Graphs (ARGs) from pairs of segment trees, using topological differences and proceeding greedily to find maximally compatible clades. Orders of magnitude faster than CoalRe with comparable accuracy. TreeKnit identifies which segments share common evolutionary history for each isolate—the closest phylogenetic analog to pairwise segment co-occurrence determination. **Adjacent.**

### 1.4 Revealing reassortment in influenza A viruses with TreeSort
**Markin A., Macken C.A., Baker A.L. & Anderson T.K.** (2025) *Molecular Biology and Evolution* 42(8):msaf133

The latest large-scale phylogenetic reassortment detection tool, using a rigorous statistical framework leveraging molecular clock signal to identify both recent and ancestral events. TreeSort can process datasets with **thousands to tens of thousands** of whole-genome sequences, overcoming scalability limitations of GiRaF and CoalRe. Reports specific segments involved in reassortment and quantifies divergence from prior pairings. State-of-the-art in phylogenetic reassortment detection. **Adjacent.**

### 1.5 FluReF: an automated flu virus reassortment finder based on phylogenetic trees
**Yurovsky A. & Moret B.M.E.** (2011) *BMC Genomics* 12(Suppl 2):S3

One of the first fully automated reassortment finders, performing bottom-up search of full-genome and segment-based phylogenetic trees for candidate clades showing incongruencies. Includes a simple flu evolution simulator for benchmarking. Achieves **<10% false negative rate at 0% false positive rate** on simulated data. Important as an early scalable automated approach. **Adjacent.**

### 1.6 Non-random reassortment in human influenza A viruses
**Rabadan R., Levine A.J. & Krasnitz M.** (2008) *Influenza and Other Respiratory Viruses* 2(1):9–22

Pioneering distance-based (non-phylogenetic) method comparing **pairwise Hamming distances** in third codon positions between segments: proportional distances indicate no reassortment; disproportionate distances indicate reassortment. Demonstrated non-random reassortment patterns with preferred segment groupings. This distance-based pairwise comparison is conceptually close to pairwise interaction features in embeddings, though using simple sequence distances rather than learned representations. **Adjacent.**

### 1.7 A comprehensive analysis of reassortment in influenza A virus
**de Silva U.C., Tanaka H., Nakamura S., Goto N. & Yasunaga T.** (2012) *Biology Open* 1(4):385–390

Proposes a phylogeny-independent neighborhood-based method: for each strain and segment, finds the closest neighbors on the tree; reassortment is inferred when neighborhood sets differ substantially between segments. Conceptually similar to the target paper's approach—both assess whether two segments' evolutionary contexts are consistent with same-isolate origin. **Adjacent.**

### 1.8 HAIRANGE: deep learning predicts potential reassortments of avian H5N1 with human influenza viruses
**Wei J-Q., Zhang S., Li Y-D., Jiang S-Y., Yan S-R., Liu Y., Chen Y-H., Feng Y., Ding X., Li Y-C., Kang X-P., Liu W., Wu A., Jiang T., Tong Y-G. & Li J.** (2025) *National Science Review* 12(12):nwaf396

Introduces HAIRANGE, a deep learning framework for predicting human-adapted reassortment of H5N1 with human IAVs. Integrates a novel non-pretrained genome context embedder (**Codon2Vec**) that represents each codon and its coded residue dependent on upstream/downstream context, combined with a ResNet classifier. **Explicitly benchmarked against ESM-2, DNABERT2, and other embedders**. The most directly comparable existing work: uses learned sequence/protein embeddings, addresses segment-level compatibility via deep learning, and compares with ESM-2 representations. **🔴 Directly competing.**

### 1.9 A non-phylogeny-dependent reassortment detection method for influenza A viruses
**Gong X., Hu M., Wang B., Yang H., Jin Y., Liang L., Yue J., Chen W. & Ren H.** (2021) *Frontiers in Virology* 1:751196

Alignment-free reassortment detection using **codon features** extracted from segment sequences. Each sequence is represented as a **61-dimensional feature vector** (excluding stop codons), and self-organizing maps (SOMs) cluster sequences per segment independently. Reassortment is detected by comparing segment cluster assignments across an isolate's genome. The closest existing approach to the target paper's k-mer + embedding methodology—uses sequence-derived feature vectors without phylogenetic analysis for segment-level classification. **🔴 Directly competing.**

### 1.10 High-throughput droplet-based analysis of influenza A virus genetic reassortment by single-virus RNA sequencing
**Lacombe B. et al.** (2023) *Proceedings of the National Academy of Sciences* 120(23):e2211098120

Developed high-throughput droplet microfluidics for single-virus RNA sequencing, analyzing **18,422 viral genotypes** from H1N1pdm09/H3N2 coinfection. Applied **Direct Coupling Analysis (DCA)** to predict pairwise segment cosegregation frequencies and full reassortant genotype frequencies. Found 159/254 possible reassortant genotypes and all 112 possible pairwise segment combinations. The DCA model's prediction of pairwise segment co-occurrence is the most directly analogous existing approach to binary classification of whether two segments co-occur in the same isolate. **🔴 Directly competing.**

### 1.11 FluReassort: a database for the study of genomic reassortments among influenza viruses
**Ding X., Yuan X., Mao L., Wu A. & Jiang T.** (2020) *Briefings in Bioinformatics* 21(6):2126–2132

Curated database compiling **204 reassortment events** among 56 influenza A subtypes from 37 countries. Provides phylogenetic analysis tools and reassortment network visualization. Serves as a primary benchmark/reference database for validating computational reassortment detection methods. **Adjacent.**

### 1.12 HopPER: an adaptive model for probability estimation of influenza reassortment through host prediction
**Eng C.L.P., Tong J.C. & Tan T.W.** (2019) *BMC Medical Genomics* 12(Suppl 9):170

ML-based method estimating reassortment probabilities through host tropism prediction. Uses **147 features** derived from seven physicochemical properties of amino acids across all segments, with random forest and other classifiers predicting whether a virus is reassortant. Successfully identified 280/318 candidate reassortants. Directly relevant as it uses amino acid-level features (analogous to protein embeddings) with ML classifiers for reassortment-related prediction. **🔴 Directly competing.**

### 1.13 SegFinder: an automated tool for identifying RNA virus genome segments through co-occurrence in multiple sequenced samples
**Liu X., Kong J., Shan Y., Yang Z., Miao J., Pan Y., Luo T., Shi Z., Wang Y., Gou Q., Yang C., Li C., Li S., Zhang X., Sun Y., Holmes E.C., Guo D. & Shi M.** (2025) *Briefings in Bioinformatics* 26(4):bbaf358

Identifies virus genome segments based on their **common co-occurrence at similar abundance** within segmented viruses across multiple sequencing runs. Applied to 858 meta-transcriptomes, identifying 106 unique viral segments from 43 species across 12 orders, including 53 novel segments. While focused on novel segment discovery in metatranscriptomics rather than influenza isolate matching, SegFinder directly addresses the fundamental problem of determining which segments belong together—making it highly relevant to the target paper's task. **🔴 Directly competing.**

### 1.14 SegVir: reconstruction of complete segmented RNA viral genomes from metatranscriptomes
**Tang X., Shang J., Chen G., Chan K.H.K., Shi M. & Sun Y.** (2024) *Molecular Biology and Evolution* 41(8):msae171

A tool for identifying segmented RNA viruses and reconstructing their complete genomes from metatranscriptomes, using both close and remote homology searches (pairwise alignment and profile HMMs). Groups contigs bearing the same family label as the same segmented virus and evaluates genome completeness. Addresses the problem of determining which contigs/segments belong to the same virus—the metatranscriptomic analog of the target paper's isolate-level segment matching. **Adjacent.**

---

## Area 2: Protein Language Models for Virology

Applications of PLMs to viral proteins have grown rapidly since 2021, demonstrating that pretrained embeddings capture meaningful biological signals for viral classification, fitness prediction, and antigenic analysis.

### 2.1 Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM-2)
**Lin Z., Akin H., Rao R., Hie B., Zhu Z., Lu W., Smetanin N., Verkuil R., Kabeli O., Shmueli Y., dos Santos Costa A., Fazel-Zarandi M., Sercu T., Candido S. & Rives A.** (2023) *Science* 379(6637):1123–1130

The foundational paper for ESM-2, trained on **~65 million unique protein sequences** with masked language modeling, scaling up to 15 billion parameters. Generates rich per-residue embeddings encoding evolutionary and structural information. The direct methodological foundation for the target paper's use of frozen ESM-2 embeddings in classification. **Adjacent** (foundational reference).

### 2.2 Learning the language of viral evolution and escape
**Hie B., Zhong E.D., Berger B. & Bryson B.** (2021) *Science* 371(6526):284–288

Landmark paper applying NLP-inspired language models (BiLSTM) to viral proteins. Modeled viral escape by framing mutations as word changes that preserve "grammaticality" (fitness) while changing "semantics" (antigenicity). Applied to **influenza A hemagglutinin, HIV-1 Env, and SARS-CoV-2 spike protein**. Seminal work bridging NLP and viral evolution, demonstrating that protein language representations capture immune-relevant features of RNA virus surface proteins. **Adjacent.**

### 2.3 Evolutionary velocity with protein language models predicts evolutionary dynamics of diverse proteins
**Hie B.L., Yang K.K. & Kim P.S.** (2022) *Cell Systems* 13(4):274–285.e6

Introduces "evo-velocity" using ESM-1b to predict the direction of protein evolution. Applied to influenza HA and SARS-CoV-2 spike, recovering evolutionary trajectories and predicting viral immune escape strategies. Demonstrates that general-purpose PLM embeddings contain **evolutionary directional signals** applicable to viral proteins. **Adjacent.**

### 2.4 A protein language model for exploring viral fitness landscapes (CoVFit)
**Ito K. et al.** (2025) *Nature Communications* 16

Fine-tunes ESM-2 to predict SARS-CoV-2 variant fitness from spike protein sequences. Performs domain adaptation on ESM-2 with Coronaviridae spike sequences, then fine-tunes with genotype-fitness and deep mutational scanning data. Ranked fitness of future variants with **~15 mutations**. Directly demonstrates that fine-tuned ESM-2 can predict functional properties of RNA virus proteins. **Adjacent.**

### 2.5 From single-sequences to evolutionary trajectories: protein language models capture the evolutionary potential of SARS-CoV-2
**Lamb K.D., Hughes J., Lytras S., Koci O., Young F., Grove J., Yuan K. & Robertson D.L.** (2026) *Nature Communications*

Demonstrates that **unmodified, pretrained ESM-2** applied to SARS-CoV-2 spike via in silico deep mutational scanning captures evolutionary constraints from sequence context alone (without MSA). ESM-2 representations encode the evolutionary history between variants and distinguish variants of concern. Validates that frozen ESM-2 embeddings encode biologically meaningful signals for viral protein analysis—closely supporting the target paper's approach. **Adjacent.**

### 2.6 Prediction of virus-host associations using protein language models and multiple instance learning (EvoMIL)
**Gussow A.B. et al.** (2024) *PLOS Computational Biology* 20(11):e1012597

Combines ESM-1b protein embeddings with attention-based **multiple instance learning (MIL)** to predict virus-host associations. Each virus is treated as a "bag" of protein embeddings. ESM-1b outperforms traditional sequence composition features (amino acids, k-mers) for host prediction. Correctly identifies SARS-CoV-2 as a human virus and pinpoints spike proteins via attention. Directly relevant: PLM embeddings + MIL/MLP classifier for viral classification. **🔴 Directly competing.**

### 2.7 Language models learn to represent antigenic properties of human influenza A(H3) virus
**Durazzi F., Koopmans M.P.G., Fouchier R.A.M. & Remondini D.** (2025) *Scientific Reports* 15:21364

Compares BiLSTM, ProtBERT, and classical approaches to reconstruct antigenic map coordinates from HA1 protein sequences of influenza A(H3N2). Deep learning language model embeddings outperform classical approaches for fine-grained antigenic prediction, including **single amino acid-driven antigenic changes**. Directly relevant as it applies PLM embeddings to influenza A protein analysis. **Adjacent.**

### 2.8 Protein Set Transformer: a protein-based genome language model to power high-diversity viromics
**Martin C., Gitter A. & Anantharaman K.** (2025) *Nature Communications*

Presents PST, a genome language model using **ESM-2 protein embeddings** as inputs, contextualizing them at the genome level with a set transformer architecture. Trained on >100k viral genomes encoding >6M proteins for virus identification, host prediction, and protein annotation. Uses triplet loss and masked language modeling. Highly relevant: directly models relationships between multiple proteins within viral genomes using ESM-2 embeddings—conceptually related to determining whether protein segments originate from the same viral isolate. **🔴 Directly competing.**

### 2.9 Large language models improve annotation of prokaryotic viral proteins
**Flamholz Z.N. et al.** (2024) *Nature Microbiology* 9:537–549

Uses ESM-1b embeddings as input features to a **feed-forward neural network classifier** for functional annotation of viral proteins, particularly bacteriophage virion proteins. PLM-based classification outperforms HMM-based methods, especially for proteins with low sequence identity. Same general paradigm: PLM embeddings + neural network classifier for viral protein classification. **Adjacent.**

### 2.10 Protein language models expose viral immune mimicry
**Brandes N., Ofer D. & Linial M.** (2025) *Viruses* 17(9):1199

Uses ProtT5, ProteinBERT, and ESM-2 to distinguish viral from human proteins as a binary classification task (**99.7% AUC**). Analyzes misclassified viral proteins to reveal patterns of viral immune mimicry. Methodologically similar: binary classification using PLM embeddings on viral proteins. **Adjacent.**

### 2.11 ProtTrans: toward understanding the language of life through self-supervised learning
**Elnaggar A., Heinzinger M., Dallago C. et al.** (2022) *IEEE Transactions on Pattern Analysis and Machine Intelligence* 44(10):7112–7127

Foundational paper for the ProtTrans suite (ProtBERT, ProtT5) trained on up to **393 billion amino acids**. Demonstrates that protein LM embeddings capture biophysical features and enable downstream prediction tasks. Alternative/complementary PLM family alongside ESM-2 that many virology studies use. **Adjacent** (foundational reference).

### 2.12 Mitigating the antigenic data bottleneck: semi-supervised learning with protein language models for influenza A surveillance
**Xu Y.** (2025) *arXiv preprint* 2512.05222

Evaluates ESM-2 alongside ProtVec, ProtT5, and ProtBert with semi-supervised learning for predicting antigenicity of influenza A HA sequences across H1N1, H3N2, H5N1, and H9N2 subtypes. ESM-2 proved the most robust embedding, achieving **F1 >0.82 with only 25% labeled data**. Directly demonstrates ESM-2 utility for influenza classification. **Adjacent.**

---

## Area 3: Pairwise Sequence Comparison with Embeddings

The pairwise feature construction approach (element-wise absolute difference, concatenation) used in the target paper traces directly to NLP innovations, subsequently adopted for biological sequence comparison and protein-protein interaction prediction.

### 3.1 Supervised learning of universal sentence representations from natural language inference data (InferSent)
**Conneau A., Kiela D., Schwenk H., Barrault L. & Bordes A.** (2017) *EMNLP 2017*:670–680

The seminal NLP paper introducing the pairwise feature construction used in the target paper. For classifying relationships between sentence embeddings **u** and **v**, InferSent uses concatenation of (u, v, |u−v|, u∗v) fed into a fully-connected classifier. This three-way NLI classification approach became the **standard template** for pairwise comparison of fixed-length embeddings. Direct methodological ancestor of the target paper. **Adjacent** (NLP foundation).

### 3.2 Sentence-BERT: sentence embeddings using Siamese BERT-networks
**Reimers N. & Gurevych I.** (2019) *EMNLP-IJCNLP 2019*:3982–3992

Adapts BERT into a siamese/triplet network to produce fixed-size sentence embeddings. For classification tasks, SBERT concatenates embeddings **(u, v, |u−v|)** and passes them through a softmax classifier—the same pairwise feature construction used in the target paper. Demonstrates that frozen or fine-tuned transformer embeddings paired with lightweight pairwise classifiers perform competitively. The protein embedding + pairwise feature approach is a direct biological analog of SBERT. **Adjacent** (NLP foundation).

### 3.3 Learning protein sequence embeddings using information from structure
**Bepler T. & Berger B.** (2019) *ICLR 2019*

Trains biLSTM protein sequence embeddings using structural similarity supervision. For pairwise contact prediction, constructs pairwise features as **v_ij = [|z_i − z_j|; z_i ⊙ z_j]**—concatenation of absolute element-wise differences and element-wise products—explicitly noting this featurization's "widespread utility for pairwise comparison models in NLP." The most direct precedent for pairwise interaction features in the protein domain. **🔴 Directly competing.**

### 3.4 D-SCRIPT translates genome to phenome with sequence-based, structure-aware, genome-scale predictions of protein-protein interactions
**Sledzieski S., Singh R., Cowen L. & Berger B.** (2021) *Cell Systems* 12(10):969–982.e6

Uses frozen Bepler & Berger protein embeddings with a structure-aware neural architecture for cross-species PPI prediction. The contact module computes **absolute difference and element-wise product** between each pair of residue embeddings from two proteins, then applies convolutional filters. Demonstrates that frozen PLM embeddings can generalize across species for binary PPI classification. **🔴 Directly competing.**

### 3.5 Multifaceted protein–protein interaction prediction based on Siamese residual RCNN (PIPR)
**Chen M., Ju C.J., Zhou G., Chen X., Zhang T., Chang K.W., Zaniolo C. & Wang W.** (2019) *Bioinformatics* 35(14):i305–i314

Siamese architecture with residual recurrent CNNs for PPI prediction from sequences. Uses property-aware amino acid embeddings and a Siamese encoder to capture mutual influence of protein pairs—analogous to encoding two sequences separately then combining features for classification. Draws explicit analogy to **NLP sentence-pair modeling**. **🔴 Directly competing.**

### 3.6 Accurate prediction of virus-host protein-protein interactions via a Siamese neural network using deep protein sequence embeddings (STEP)
**Madan S., Demina V., Stapf M., Ernst O. & Fröhlich H.** (2022) *Patterns* 3(10):100551

Integrates **ProtBERT** pretrained embeddings into a Siamese neural network for predicting virus-host PPIs. Applied to SARS-CoV-2 and JCV virus-human interactions. Directly relevant as it combines deep protein language model embeddings with Siamese pairwise architectures for viral protein comparison. **🔴 Directly competing.**

### 3.7 SENSE: Siamese neural network for sequence embedding and alignment-free comparison
**Zheng W., Yang L., Genco R.J., Wactawski-Wende J., Buck M. & Sun Y.** (2019) *Bioinformatics* 35(11):1820–1828

First deep learning approach to alignment-free sequence comparison. Uses a Siamese CNN to learn an embedding function mapping biological sequences to vectors where pairwise **Euclidean distances approximate alignment distances**. Demonstrates Siamese network architecture for general biological sequence comparison, though focuses on distance regression rather than binary classification. **Adjacent.**

### 3.8 Cracking the black box of deep sequence-based protein–protein interaction prediction
**Bernett J., Blumenthal D.B. & List M.** (2024) *Briefings in Bioinformatics* 25(2):bbae076

Systematic benchmarking revealing that high reported PPI prediction accuracies are largely due to **data leakage from random train/test splits**. When leakage is minimized, deep learning methods perform near-randomly. Baseline ML models using sequence similarity match DL performance at a fraction of cost. Established a gold-standard leakage-free dataset. Critical for understanding performance claims of pairwise embedding classifiers—an important methodological caution for the target paper. **Adjacent.**

### 3.9 Deep learning models for unbiased sequence-based PPI prediction plateau at an accuracy of 0.65
**Bernett J., Blumenthal D.B. & List M.** (2025) *Bioinformatics* 41(Supplement_1):i590

Follow-up testing ESM-2 embeddings with multiple architectures on the leakage-free dataset. Re-implements per-protein embeddings → linear projection → concatenation → MLP, and the D-SCRIPT model using **absolute difference and element-wise product**. All models plateau at ~0.65 accuracy regardless of architecture complexity, suggesting **embedding quality matters more than model design**. Directly benchmarks the exact architecture used in the target paper. **🔴 Directly competing.**

### 3.10 PLM-interact: extending protein language models to predict protein-protein interactions
**Sledzieski S. et al.** (2025) *Nature Communications*

Proposes jointly encoding protein pairs through a fine-tuned PLM (ESM-2), inspired by BERT's next-sentence prediction, rather than the "frozen PLM → pairwise classifier" approach. Explicitly contrasts with the architecture where frozen embeddings are fed into a classification head. Achieves state-of-the-art on cross-species PPI and **virus-human PPI benchmarks**. Important counterpoint to the target paper's frozen-embedding approach. **🔴 Directly competing.**

### 3.11 Democratizing protein language models with parameter-efficient fine-tuning
**Sledzieski S., Khurana M., Cowen L. & Berger B.** (2024) *PNAS* 121(11):e2314853121

Introduces parameter-efficient fine-tuning (LoRA) for PLMs. Remarkably, an MLP classifier trained on **frozen ESM-2 embeddings** (the baseline) actually outperforms both PEFT and full fine-tuning for PPI prediction (AUPR ~0.684), directly validating the target paper's approach of using frozen embeddings with an MLP. **🔴 Directly competing.**

### 3.12 Contrastive learning in protein language space predicts interactions between drugs and protein targets (ConPLex)
**Singh R., Sledzieski S., Bryson B., Cowen L. & Berger B.** (2023) *PNAS* 120(24):e2220778120

Uses pretrained ProtBert embeddings with a contrastive co-embedding architecture for drug-target interaction prediction. Proteins and drugs are embedded into a shared space where distance predicts interaction—an alternative to concatenation-based MLP approaches for pairwise prediction using frozen PLM embeddings. **Adjacent.**

---

## Area 4: Influenza Computational Genomics

ML and DL methods for influenza genomics increasingly address segment-level analysis, with recent work directly modeling inter-segment relationships.

### 4.1 Machine learning methods for predicting human-adaptive influenza A virus reassortment based on intersegment constraint
**Zeng D-D., Cai Y-R., Zhang S., Yan F., Jiang T. & Li J.** (2025) *Frontiers in Microbiology* 16:1546536

Directly addresses inter-segment relationships using ML. Analyzes nucleotide composition across all 8 IAV segments, examines intersegment NC correlations, and uses **MLP and random forest classifiers** to predict adaptive IAV reassortment. Simulates reassortant IAVs by computationally combining segments from different viruses and predicting human-adaptation probability. True positive rate of **98.53%** for MLP. The most directly relevant paper from the influenza genomics perspective. **🔴 Directly competing.**

### 4.2 Flu-CNN: identifying host specificity of influenza A virus using convolutional networks
**Hu M., Luo N. & Wang B. et al.** (2025) *Human Genomics* 19:96

CNN model analyzing individual genomic segments to determine host specificity (human vs. avian), achieving **99% accuracy** on 911,098 sequences. Produces per-segment host specificity signatures and visualizes "mosaic patterns" across the 8 segments to identify zoonotic reassortment events. The segment-level analysis and identification of mixed human/avian segment signatures is directly relevant to segment-origin matching. **🔴 Directly competing.**

### 4.3 Machine learning methods for predicting human-adaptive influenza A viruses based on viral nucleotide compositions
**Li J., Zhang S., Li B. et al.** (2020) *Molecular Biology and Evolution* 37(4):1224–1236

Builds ML models using **mono- and dinucleotide compositions (60 features per segment)** to predict human-adaptation of IAV segments PB2, PB1, PA, HA, NP, and NA. Uses PCA, SVC, random forest. Identifies 9–13 optimized nucleotide features per segment. The per-segment nucleotide composition approach is conceptually similar to k-mer features. **Adjacent.**

### 4.4 VAPOR: influenza classification from short reads facilitates robust mapping pipelines and zoonotic strain detection
**Southgate J.A., Bull M.J., Brown C.M. et al.** (2020) *Bioinformatics* 36(6):1681–1688

A **k-mer-based tool (k=21)** for influenza virus classification directly from Illumina sequencing reads using De Bruijn graph queries. Achieves >99.8% identity to assemblies across 257 whole-genome sequencing samples. Relevant as a k-mer-based approach specifically designed for influenza genomic classification. **Adjacent.**

### 4.5 INFINITy: a fast machine learning-based application for human influenza A and B virus subtyping
**Cacciabue M. & Marcone D.N.** (2023) *Influenza and Other Respiratory Viruses* 17(1):e13097

Alignment-free ML tool for influenza classification into **75 clades/genetic groups** across A(H1N1)pdm09, A(H3N2), B/Victoria, and B/Yamagata. Uses only HA sequences, operating without alignment. Demonstrates feasibility of alignment-free ML for rapid influenza subtyping. **Adjacent.**

### 4.6 WaveSeekerNet: accurate prediction of influenza A virus subtypes and host source using attention-based deep learning
**Nguyen H-H., Rudar J., Lesperance N. et al.** (2025) *GigaScience* 14:giaf089

Uses chaos game representation, Fourier transform, and wavelet transform with attention mechanisms for IAV subtype and host prediction. Achieves **balanced accuracy of 1.0** for subtype prediction. Notably benchmarks against ESM-2 models and claims superior generalization at lower computational cost. **Adjacent.**

### 4.7 PACIFIC: a lightweight deep-learning classifier of SARS-CoV-2 and co-infecting RNA viruses
**Acera Mateos P., Balboa R.F., Easteal S., Eyras E. & Patel H.R.** (2021) *Scientific Reports* 11:3209

Deep learning classifier using **9-mer tokenized RNA sequences** fed into a CNN-BiLSTM architecture to classify RNA-seq reads into human, SARS-CoV-2, influenza, and other viral classes. Trained on 7.9 million 150nt fragments. Relevant for k-mer embedding approach on RNA virus sequences including influenza. **Adjacent.**

### 4.8 Viral genome deep classifier (VGDC)
**Fabijańska A. & Grabowski S.** (2019) *IEEE Access* 7:81297–81307

Universal deep CNN for virus subtyping applied to dengue, hepatitis B/C, HIV-1, and influenza A. Uses 1D convolutional layers on one-hot encoded genome sequences. For influenza A, achieves ~0.85 F1-score. Early DL approach to viral genome classification. **Adjacent.**

### 4.9 A novel data augmentation approach for influenza A subtype prediction based on HA proteins (PreIS)
**Sohrabi M. et al.** (2024) *Computers in Biology and Medicine* 172:108316

Uses pretrained protein language models and supervised data augmentation for influenza A subtype classification based on HA protein sequences. Achieves **94.54% accuracy**, outperforming previous CNN models. Demonstrates PLM embeddings effectively capture subtle differences in influenza protein sequences. **Adjacent.**

### 4.10 Influenza virus genotype to phenotype predictions through machine learning: a systematic review
**Borkenhagen L.K., Allen M.W. & Runstadler J.A.** (2021) *Emerging Microbes & Infections* 10(1):1896–1907

Comprehensive systematic review of **49 studies** employing ML for influenza A phenotype prediction, covering host discrimination, human adaptability, subtype/clade assignment, pandemic lineage assignment, infection characteristics, and antiviral drug resistance. Identifies biases in model design and gaps in wet lab validation. Essential context reference. **Adjacent.**

---

## Cross-cutting Themes

Several patterns emerge across these four areas that clarify where the target paper sits and what makes it novel.

**The segment matching gap is real.** Across all 44 papers reviewed, no existing method directly addresses binary classification of whether two specific protein/genome segments originate from the same isolate using pairwise learned embeddings. The closest approaches are: (1) HAIRANGE, which uses Codon2Vec + ResNet for assessing segment compatibility but focuses on adaptive reassortment prediction rather than pairwise isolate matching; (2) the DCA-based droplet sequencing work, which statistically models pairwise cosegregation but from experimental coinfection data; and (3) SegFinder, which uses co-occurrence abundance patterns across multiple samples rather than sequence-level features.

**The architectural lineage is well-established.** The pairwise feature approach (|u−v|, concatenation) traces from InferSent (2017) → Sentence-BERT (2019) → Bepler & Berger (2019) → D-SCRIPT (2021) → PLM-interact (2025). The target paper's combination of this approach with frozen ESM-2 embeddings follows a proven paradigm that has been validated in the PPI prediction literature, where frozen ESM-2 + MLP has been shown to be surprisingly competitive (Sledzieski et al., 2024).

**Frozen ESM-2 embeddings have been validated for viral proteins.** Multiple papers demonstrate that frozen or minimally adapted ESM-2 captures meaningful biological signals for viral sequences: evolutionary trajectories in SARS-CoV-2 (Lamb et al., 2026), fitness prediction via fine-tuning (Ito et al., 2025), antigenic prediction for influenza (Xu, 2025; Durazzi et al., 2025), and genome-level reasoning in viromics (Martin et al., 2025). However, no existing work applies frozen ESM-2 to the segment-matching task.

**The methodological caution from PPI literature applies.** Bernett et al. (2024, 2025) demonstrated that data leakage inflates performance in pairwise prediction tasks, with models plateauing at ~0.65 accuracy on leakage-free PPI data. The target paper should carefully address data partitioning strategies to avoid similar artifacts, particularly since influenza isolates from the same lineage will share significant sequence similarity.

**K-mer features for influenza are well-precedented.** VAPOR (Southgate et al., 2020), PACIFIC (Acera Mateos et al., 2021), and the nucleotide composition work (Li et al., 2020; Zeng et al., 2025) all demonstrate that k-mer or nucleotide composition features contain informative signal for influenza classification. The target paper's combination of k-mer features with ESM-2 protein embeddings represents a genuinely novel multimodal approach to segment analysis.

---

## Summary Classification Table

| Paper | Year | Area | Classification |
|---|---|---|---|
| GiRaF (Nagarajan & Kingsford) | 2011 | 1 | Adjacent |
| CoalRe (Müller et al.) | 2020 | 1 | Adjacent |
| TreeKnit (Barrat-Charlaix et al.) | 2022 | 1 | Adjacent |
| TreeSort (Markin et al.) | 2025 | 1 | Adjacent |
| FluReF (Yurovsky & Moret) | 2011 | 1 | Adjacent |
| Rabadan et al. | 2008 | 1 | Adjacent |
| de Silva et al. | 2012 | 1 | Adjacent |
| **HAIRANGE (Wei et al.)** | **2025** | **1** | **Directly competing** |
| **Gong et al. (SOM)** | **2021** | **1** | **Directly competing** |
| **Lacombe et al. (DCA)** | **2023** | **1** | **Directly competing** |
| FluReassort (Ding et al.) | 2020 | 1 | Adjacent |
| **HopPER (Eng et al.)** | **2019** | **1** | **Directly competing** |
| **SegFinder (Liu et al.)** | **2025** | **1** | **Directly competing** |
| SegVir (Tang et al.) | 2024 | 1 | Adjacent |
| ESM-2 (Lin et al.) | 2023 | 2 | Adjacent |
| Hie et al. (escape) | 2021 | 2 | Adjacent |
| Hie et al. (evo-velocity) | 2022 | 2 | Adjacent |
| CoVFit (Ito et al.) | 2025 | 2 | Adjacent |
| Lamb et al. | 2026 | 2 | Adjacent |
| **EvoMIL** | **2024** | **2** | **Directly competing** |
| Durazzi et al. | 2025 | 2 | Adjacent |
| **PST (Martin et al.)** | **2025** | **2** | **Directly competing** |
| Flamholz et al. | 2024 | 2 | Adjacent |
| Brandes et al. | 2025 | 2 | Adjacent |
| ProtTrans (Elnaggar et al.) | 2022 | 2 | Adjacent |
| Xu (ESM-2 influenza) | 2025 | 2 | Adjacent |
| InferSent (Conneau et al.) | 2017 | 3 | Adjacent |
| Sentence-BERT (Reimers & Gurevych) | 2019 | 3 | Adjacent |
| **Bepler & Berger** | **2019** | **3** | **Directly competing** |
| **D-SCRIPT (Sledzieski et al.)** | **2021** | **3** | **Directly competing** |
| **PIPR (Chen et al.)** | **2019** | **3** | **Directly competing** |
| **STEP (Madan et al.)** | **2022** | **3** | **Directly competing** |
| SENSE (Zheng et al.) | 2019 | 3 | Adjacent |
| Bernett et al. (leakage) | 2024 | 3 | Adjacent |
| **Bernett et al. (plateau)** | **2025** | **3** | **Directly competing** |
| **PLM-interact (Sledzieski et al.)** | **2025** | **3** | **Directly competing** |
| **Sledzieski et al. (PEFT)** | **2024** | **3** | **Directly competing** |
| ConPLex (Singh et al.) | 2023 | 3 | Adjacent |
| **Zeng et al. (intersegment)** | **2025** | **4** | **Directly competing** |
| **Flu-CNN (Hu et al.)** | **2025** | **4** | **Directly competing** |
| Li et al. (nucleotide) | 2020 | 4 | Adjacent |
| VAPOR (Southgate et al.) | 2020 | 4 | Adjacent |
| INFINITy (Cacciabue & Marcone) | 2023 | 4 | Adjacent |
| WaveSeekerNet (Nguyen et al.) | 2025 | 4 | Adjacent |
| PACIFIC (Acera Mateos et al.) | 2021 | 4 | Adjacent |
| VGDC (Fabijańska & Grabowski) | 2019 | 4 | Adjacent |
| PreIS (Sohrabi et al.) | 2024 | 4 | Adjacent |
| Borkenhagen et al. (review) | 2021 | 4 | Adjacent |

**18 papers classified as directly competing; 26 as adjacent.** The target paper's unique contribution lies in combining frozen ESM-2 protein embeddings and k-mer genomic features with NLP-inspired pairwise interaction features (absolute difference + concatenation) in an MLP for the specific task of segment-isolate matching—a combination that no existing paper has attempted.
