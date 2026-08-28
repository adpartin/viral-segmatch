# CLAUDE.md — Project Context for Claude Code

Always loaded. This file holds the **rules to follow**. Two companions hold the rest:
`docs/architecture.md` describes how the system works (pipeline stages, config system,
source-file map, findings, roadmap, HPC), and `.claude/memory.md` holds current project state.
Read those when you need to understand a subsystem or see where the work stands.

---

## Session Startup Checklist

1. Read `.claude/memory.md` (in the repo) for current project state.
2. If the machine-local `MEMORY.md` (path in the system prompt) doesn't exist, create it with:
   ```
   Project memory has moved into the repo for portability across machines.
   Read: .claude/memory.md (in the repo root)
   This machine-local file is no longer updated.
   ```
3. Check `docs/plans/` for in-progress plans (status != IMPLEMENTED); read and offer to resume.

**Plans** (`docs/plans/`): save new plans to `docs/plans/<descriptive_name>_plan.md` with a status
line right under the title — `**Status: IN PROGRESS**` (underway) / `**Status: IMPLEMENTED**`
(complete). When fully implemented, mark IMPLEMENTED and move the file to `docs/plans/done/`.

---

## Approval Required

Always ask for explicit confirmation before running any of these — even if it seems safe:

- `rm *` or any file/directory deletion
- `git rm *`, `git reset --hard *` / `--mixed *`, `git push --force*` / `-f *`,
  `git branch -D *`, `git rebase *`, `git clean *`
- Any command that modifies shared infrastructure, sends messages, or affects state outside this repo

(`.claude/settings.json` `permissions.deny` already hard-blocks several of these, e.g. `rm *`.)

---

## What This Project Does

**viral-segmatch** predicts whether two viral protein segments come from the same isolate
(binary classification). Frozen ESM-2 embeddings (1280-dim `esm2_t33_650M_UR50D`) or k-mer
features → pairwise interaction (e.g. `unit_diff`) → MLP or sklearn-baseline classifier. Main
virus: **Influenza A** (Bunyavirales support exists but is not maintained). Stages 1–2 run once
per `{virus}/{data_version}` and are shared; Stages 3–4 run per experiment.

Details — the stage table, Hydra bundle system, source-file map, findings, roadmap and HPC
notes — are in `docs/architecture.md`.

---

## Core Vocabulary

The terms that get confused most often. Fuller definitions are in `docs/methods/glossary.md`;
keep the two in sync. Use these exact words — do not invent synonyms.

- **atom** — the indivisible routing unit. In 2D-CD (`cluster_disjoint_cc`) atom = one **CC** (one atom per CC; `atom_id == cc_id`). An atom is NOT a row.
- **rows ≠ atoms** — a positive "pair"/row is one record; `m_pos_per_cc` caps rows-*per-atom*, NOT the atom count (the cluster threshold fixes the atom count). Reserve "atom"/"CC" for components and "pair"/"row" for records.
- **CC / mega-CC** — connected component on the (slot-A cluster, slot-B cluster) bigraph; the mega-CC is the one huge component that holds most of the pairs at low `t`.
- **pair_key_alphabet** — the positive-dedup key is built on the alphabet's hash (aa→`prot_hash`, nt_cds→`cds_dna_hash`, nt_ctg→`ctg_dna_hash`), so the positive **universe is alphabet-defined** (nt keeps codon/contig variants that aa collapses).
- **front-end (df)** — the output of `build_frontend`: `protein_final` loaded + DNA hashes attached. The `protein_final` *file* natively carries only `prot_hash`; `ctg_dna_hash`/`cds_dna_hash` are attached *after* load from sibling files (`ctg_dna_final`/`cds_dna_final`).
- **hash source-stages** — Stage 1 writes `prot_hash` + `ctg_dna_hash`; Stage 1.5 writes `cds_dna_hash`; Stage 3 reads them (no recompute).
- **within_cc vs within_fold** — CC-builder negative scope: within_cc draws negatives inside each CC (removes the cluster shortcut; hard); within_fold draws cross-CC in-split (keeps it; easier).
- **molecule ↔ alphabet** — aa↔prot / nt_cds↔cds_dna / nt_ctg↔ctg_dna. The pairing that gets mixed up most often; the `aa/nt vs protein/DNA` convention below states it in full.

---

## Conventions

- **Experiment naming**: `{virus}_{proteins}_{n_isolates}[_{modifiers}]`.
- **Timestamps**: All run directories include `YYYYMMDD_HHMMSS`.
- **Shared vs. run-specific**: Preprocessing and embeddings are shared per `{virus}/{data_version}`. Datasets and models are per run in `runs/` subdirectories.
- **Seed system**: Hierarchical — `master_seed` derives all process seeds. See `docs/SEED_SYSTEM.md`.
- **Metrics**: `metrics.csv` carries F1 (binary + macro), precision, recall, AUC-ROC, AUC-PR, MCC, Brier, BCE loss. Early-stop options: `loss`, `f1`, `auc_roc`, `auc_pr`, `mcc`. Naming: snake_case identifiers are `auc_roc` / `auc_pr`; display strings are `AUC-ROC` / `AUC-PR`. Sklearn names `roc_auc_score` / `average_precision_score` are external and left alone. Train targets neg:pos = `neg_to_pos_ratio` (default 1.0). Under the coverage-first negative sampler val/test come out ~1.07–1.20x neg-heavy, because its coverage phase overshoots the ratio; under `negative_scope: within_fold` every split is exactly on ratio.
- **Proteins**: `preprocess_flu.py` maps GTO replicon functions to standard protein names (PB2, PB1, PA, HA, NP, NA, M1, M2, NEP).
- **Threshold notation**: `tXXX`, zero-padded (e.g. `t095`), is the mmseqs identity threshold `0.XXX`. Use it everywhere — docs, plot labels, code, bundle filenames, `cluster_id_path`. Cluster parquets live at `clusters_*/tXXX/`.
- **Sequence hashes**: `prot_hash = md5(prot_seq)`, `ctg_dna_hash = md5(ctg_dna_seq)`, `cds_dna_hash = md5(cds_dna_seq)`. In pair tables: `*_hash_a` / `*_hash_b`. Per-alphabet column/file names come from one source of truth — the `SCHEMA` registry in `src/utils/schema.py` (alphabet ∈ {`aa`, `nt_cds`, `nt_ctg`}). Each hash is produced/persisted at its source stage (Stage 1 writes `prot_hash`/`ctg_dna_hash`; Stage 1.5 writes `cds_dna_hash`); Stage 3 reads them (no recompute). ESM-2 cache key uses `sha1(prot_seq)` — separate namespace, never joined back to `prot_hash`.
- **Log messages**: No emojis. Use text prefixes: `ERROR:` (fatal), `WARNING:` (non-fatal), `Done.` (success).
- **Reuse before you write**: before adding a helper, search the whole codebase for one that already does the job. Prefer one obvious implementation over a flexible one. Correctness > readability > efficiency.
- **Function docstrings**: the first sentence (two at most) states what the function does; `Args:` and `Returns:` present and correct. Every claim checked against the code — never inferred from the function name or from memory. Lean, and current-state only (no account of how the function evolved).
- **Names across functions**: the same idea gets the same name everywhere, and two different ideas never share a name (unless the name is generic). A reader should be able to work out what something means from the code, never have to guess. Where an analysis variant disagrees with production, production wins. Watch the pairs that get crossed: atom vs CC vs cluster, row/pair vs atom, `cc_id` vs `atom_id`, `aa`/`nt_cds`/`nt_ctg` vs `prot`/`cds_dna`/`ctg_dna`.
- **Function names describe current behaviour**: rename when the code has changed and the name no longer reflects what it does. Never change a function name without the user's approval.
- **Statement complexity**: break a dense statement into named steps with a brief comment. Never return an expression that also does the work — bind the call to a named variable, then return it.
- **No plan-only vocabulary in code**: code, comments, docstrings and error messages must be readable without opening a plan. Never use a label that only a plan defines (`D3`, `OoS #5`, `routing-B`, `P2`, "Phase 2") as though it were a term — say what the thing *is*, then cite the plan by full path if the derivation matters. A path pointer is fine; an undefined label is not. Canonical terms belong in `docs/methods/glossary.md`; plan labels are not canonical terms. This matters most in raised error text, which reaches users who have never seen the plan.
- **Leakage terminology**: use canonical names from `docs/plans/2026-05-07_leakage_diagnostics_plan.md` (same-pair leakage, sequence-level label imbalance, sequence-level leakage, cluster leakage, demographic shortcut leakage). New modes go in that table first.
- **aa/nt vs protein/DNA**: `aa` / `nt_cds` / `nt_ctg` name the **alphabet** (which residues); `prot` / `cds_dna` / `ctg_dna` name the **molecule** (which sequence). They pair up aa↔protein, nt_cds↔CDS DNA, nt_ctg↔contig DNA. Never pair one level's token with the other's name. `src/utils/schema.py` enforces this.
- **Reading CSVs with `function_short`**: any CSV with a `function_short` column has the literal string `'NA'` (Neuraminidase) as a value. Default `pd.read_csv()` parses `'NA'` as NaN and **silently drops Neuraminidase rows**. Always read with `keep_default_na=False, na_values=['']`. Source pipeline CSVs use full names (safe); derived `function_short` CSVs are vulnerable.
- **sklearn `GroupKFold` shuffle**: `shuffle=True` is NOT a randomized version of the balanced assignment — it *replaces* the size balancing (largest group to the lightest fold, i.e. LPT) with `np.array_split` into equal-group-**count** chunks, so folds go badly unbalanced whenever group sizes are skewed. Pass `shuffle=False` explicitly (and drop `random_state`, which sklearn then rejects). The 2D-CD router does; see `groupkfold_by_atom`.
- **Bash tool calls**: prefer single-command invocations over compound chains (`&&`, `;`, `$(...)`, `bash -c '...'`) — the allow-list matcher only auto-approves statically-parseable commands. Use compound only when atomicity matters (`git add X && git commit ...`) or it's fundamentally one shell idiom.
- **Wording**: use plain verbs (`removes`, `drops`, `reads`, `writes`, `joins`), not decorative ones (`scrubs`, `munges`, `slurps`), and no metaphors. Use the same word for the same thing throughout the repo. `docs/methods/glossary.md` is the canonical term list (graph-theory + project terms): use its exact terms, and add a new term there before using it anywhere else.
- **Docs describe current state, not history**. Method/reference docs (`docs/methods/`, `CLAUDE.md`, `.claude/memory.md`) read as a stable description of how things are now. Historical framing belongs in `docs/results/` or `docs/plans/`.
- **Check before you assert**. Anything you say about what exists (a file, function, flag, bundle), what a value currently is, or what code does must be checked against the source in the same turn — Read it, grep it, or run it. Never answer from memory or inference. If you did not check, say so in the sentence ("unverified", "likely", "would need to check X"); a flat factual claim with no evidence and no hedge is an error. If you are confused, say "I'm not sure" **before** you explain, not after you are corrected.
- **Claim exactly what you checked**. Say "checked PB2 and PB1, not the other 6" rather than rounding up to all of them. Before saying something *doesn't exist*, or is *new* / *first* / *only*, or that *all* or *none* share a property, grep the whole repo — a sample is not enough. Name the exception, not just the rule.
- **Give the number when the size is the point**. "PB2 t093 → t092: 1,085 → 112 (−90%)" beats "drops sharply". Use words like "sharply" only when the shape of a trend matters more than its size.
- **Check a name against every case it must cover**. Before naming a field, structure or API, list what it spans (slot a/b, routing modes, alphabets, splits) and check the name works for each. A name that fits today's example but not the others will lock in a wrong assumption.
- **Commits are explicit-only**. Never run `git commit` / `--amend` on Claude's own initiative. Commit only on an explicit user instruction (a specific change, or a standing batch/session authorization). Otherwise: stage, show the diff, draft the message, and stop. `git commit` is allow-listed (no prompt), so this rule is the sole guard — apply it strictly.

---

## Per-machine Git Setup

Run once after cloning on each new machine (writes to `.git/config`, not tracked):
```bash
git config pull.rebase true   # avoid "need to reconcile divergent branches" on git pull
```
