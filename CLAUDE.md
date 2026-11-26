# CLAUDE.md — Project Context for Claude Code

Always loaded. This file holds the **rules to follow**. Two companions hold the rest:
`docs/architecture.md` describes how the system works (pipeline stages, config system,
source-file map, findings, roadmap, HPC), and `.claude/memory.md` holds current project state.
Read those when you need to understand a subsystem or see where the work stands.

---

## Session Startup Checklist

1. Read `.claude/memory.md` in this repository for current project state.
2. For paper work, start with `docs/plans/2026-11-24_paper_plan_v2.md`. Open the supporting plan for the task at hand.

**Plans** (`docs/plans/`): save new plans to `docs/plans/<descriptive_name>_plan.md` with a status
line right under the title — `**Status: IN PROGRESS**` (underway) / `**Status: IMPLEMENTED**`
(complete) / `**Status: SHELVED**` (stopped before it was finished; may resume). When fully
implemented, mark IMPLEMENTED and move the file to `docs/plans/done/`. When shelving, mark
SHELVED with the date, keep the old status after it, and move the file to `docs/plans/shelved/`.

---

## Approval Required

Always ask for explicit confirmation before running any of these — even if it seems safe:

- `rm *` or any file/directory deletion
- `git rm *`, `git reset --hard *` / `--mixed *`, `git push --force*` / `-f *`,
  `git branch -D *`, `git rebase *`, `git clean *`
- Any command that modifies shared infrastructure, sends messages, or affects state outside this repo
- Stage 4 training. An OK covers the jobs in the approved command or plan, including resuming them.
  A rerun after a code or config change needs a new OK.

(`.claude/settings.json` `permissions.deny` already hard-blocks several of these, e.g. `rm *`.)

---

## What This Project Does

The paper asks whether a pair of influenza A segment sequences was observed in the same isolate.
A LightGBM classifier uses the concatenated per-site codon features from the two coding sequences.
The primary analysis uses aligned sites. The earlier standalone Task 1 evaluated pinned-length features.

See `docs/plans/2026-11-24_paper_plan_v2.md` for the study design and `docs/architecture.md`
for the pipeline stages, configuration, and source files.

---

## Core Vocabulary

The terms that get confused most often. Fuller definitions are in `docs/methods/glossary.md`;
keep the two in sync. Use these exact words — do not invent synonyms.

- **pair_key_alphabet** — the positive-dedup key is built on the alphabet's hash (aa→`prot_hash`, nt_cds→`cds_dna_hash`, nt_ctg→`ctg_dna_hash`), so the positive **universe is alphabet-defined** (nt keeps codon/contig variants that aa collapses).
- **front-end (df)** — the output of `build_frontend`: `protein_final` loaded + DNA hashes attached. The `protein_final` *file* natively carries only `prot_hash`; `ctg_dna_hash`/`cds_dna_hash` are attached *after* load from sibling files (`ctg_dna_final`/`cds_dna_final`).
- **hash source-stages** — Stage 1 writes `prot_hash` + `ctg_dna_hash`; Stage 1.5 writes `cds_dna_hash`; Stage 3 reads them (no recompute).
- **molecule ↔ alphabet** — `aa` / `nt_cds` / `nt_ctg` name the **alphabet** (which residues); `prot` / `cds_dna` / `ctg_dna` name the **molecule** (which sequence). They pair up aa↔prot, nt_cds↔cds_dna, nt_ctg↔ctg_dna. Never pair one level's token with the other's name. `src/utils/schema.py` enforces this.

---

## Conventions

- **Timestamps**: All run directories include `YYYYMMDD_HHMMSS`.
- **Shared vs. run-specific**: Preprocessing and embeddings are shared per `{virus}/{data_version}`. Datasets and models are per run in `runs/` subdirectories.
- **Seed system**: Hierarchical — `master_seed` derives all process seeds. See `docs/SEED_SYSTEM.md`.
- **Metric names**: Use `auc_roc` and `auc_pr` in code, and AUC-ROC and AUC-PR in figures and text. Keep the scikit-learn function names `roc_auc_score` and `average_precision_score`.
- **Proteins**: `preprocess_flu.py` maps GTO replicon functions to short protein names. `conf/virus/flu.yaml` knows 20 of them; the 8 selected for modelling, one per segment, are PB2, PB1, PA, HA, NP, NA, M1, NS1. The rest (M2, NEP, PA-X, PB1-F2, ...) are spliced, frameshifted or alternative-start products and are not modelled.
- **Sequence hashes**: `prot_hash = md5(prot_seq)`, `ctg_dna_hash = md5(ctg_dna_seq)`, `cds_dna_hash = md5(cds_dna_seq)`. In pair tables: `*_hash_a` / `*_hash_b`. Per-alphabet column/file names come from one source of truth — the `SCHEMA` registry in `src/utils/schema.py` (alphabet ∈ {`aa`, `nt_cds`, `nt_ctg`}).
- **Log messages**: No emojis. Use text prefixes: `ERROR:` (fatal), `WARNING:` (non-fatal), `Done.` (success).
- **Reuse before you write**: before adding a helper, search the whole codebase for one that already does the job. Prefer one obvious implementation over a flexible one. Correctness > readability > efficiency.
- **Function docstrings**: the first sentence (two at most) states what the function does; `Args:` and `Returns:` present and correct. Every claim checked against the code — never inferred from the function name or from memory. Lean, and current-state only (no account of how the function evolved).
- **Claims in code**: a comment or docstring states only what was checked against the code. Never hedge in code (`likely`, `unverified`): check the claim or leave it out. Hedges belong in chat and exploratory docs. Refer to other code by function or module name, not `file.py:N`, because line numbers go stale.
- **Experiment scripts fail loudly**: a script that builds datasets, trains or aggregates exits non-zero when required work fails or an expected output is missing; it never prints a warning and exits 0. Before writing, it checks for existing outputs and does not overwrite them unless an explicit flag allows it. It records the resolved config, seeds and code commit (`experiment_utils.get_git_info`) with its outputs.
- **Names across functions**: the same idea gets the same name everywhere, and two different ideas never share a name (unless the name is generic). A reader should be able to work out what something means from the code, never have to guess. Where an analysis variant disagrees with production, production wins.
- **Function names describe current behaviour**: rename when the code has changed and the name no longer reflects what it does. Never rename an existing function or a persisted name (an output file name, JSON key or CSV column) without the user's approval; renaming a persisted name breaks reads of earlier runs. Names added in the current uncommitted change are exempt.
- **Statement complexity**: break a dense statement into named steps with a brief comment. Never return an expression that also does the work — bind the call to a named variable, then return it.
- **No plan-only vocabulary in code**: code, comments, docstrings and error messages must be readable without opening a plan. Never use a label that only a plan defines (`D3`, `OoS #5`, `routing-B`, `P2`, "Phase 2") as though it were a term — say what the thing *is*, then cite the plan by full path if the derivation matters. A path pointer is fine; an undefined label is not. Canonical terms belong in `docs/methods/glossary.md`; plan labels are not canonical terms. This matters most in raised error text, which reaches users who have never seen the plan.
- **Leakage terminology**: `docs/methods/leakage.md` defines the five categories used in earlier analyses. For the current paper, follow the controls described in `docs/plans/2026-11-24_paper_plan_v2.md`.
- **Reading CSVs with `function_short`**: any CSV with a `function_short` column has the literal string `'NA'` (Neuraminidase) as a value. Default `pd.read_csv()` parses `'NA'` as NaN, so later filters and group-bys **silently drop Neuraminidase rows**. Always read with `keep_default_na=False, na_values=['']`. Source pipeline CSVs use full names (safe); derived `function_short` CSVs are vulnerable.
- **Bash tool calls**: prefer single-command invocations over compound chains (`&&`, `;`, `$(...)`, `bash -c '...'`) — the allow-list matcher only auto-approves statically-parseable commands. Use compound only when atomicity matters (`git add X && git commit ...`) or it's fundamentally one shell idiom.
- **Wording**: use plain verbs (`removes`, `drops`, `reads`, `writes`, `joins`), not decorative ones (`scrubs`, `munges`, `slurps`), and no metaphors. Use the same word for the same thing throughout the repo. `docs/methods/glossary.md` is the canonical term list (graph-theory + project terms): use its exact terms. Add a new technical or project-specific term (graph property, routing mode, split concept, metric, alphabet, etc.) there before using it anywhere else. Standard words, file names and local variable names need no entry; define them where they are used.
- **Explanatory prose** (plan/results docs, and the WHY-text in docstrings and comments — not CLAUDE.md itself or other terse reference lists, which are for scanning, not narration): full sentences, one claim per sentence, not a bolded headline followed by a stat fragment. Prefer splitting into two sentences, or a `because`/`so`/`,` join, over an em-dash as a clause-joiner. State a claim's limits in their own sentence rather than leaving them implied (e.g. "this does not establish X", not silence). A fact repeated in more than one place uses the same phrasing every time it appears.
  Example — not: `**Per-site features work, at a third of the width.** F1 macro 0.9192 +/- 0.0134 for nt vs 0.9094 +/- 0.0145 for k-mer.` Instead: `The nucleotide-site model obtained an F1 macro of 0.9192 ± 0.0134. The k-mer model obtained 0.9094 ± 0.0145. With only 4 folds, this does not establish either model as superior.`
- **Plain words, and no narration after a doc edit**. Write the plainest sentence that carries the claim. Cut any sentence that only sets up the next one ("Three things stand in the way, and they call for different responses", "This is the case that matters here") — start with the claim instead. Prefer the shorter word: `affects` over `is the case that matters for`, `no remedy` over `nothing can be done about it`. After editing a doc, show the diff and stop. Do not recap what changed, why it changed, or what was left out unless asked; the diff already says it.
- **Answer the question first.** Start with a direct answer in the first sentence. For yes/no questions, begin with `Yes`, `No`, or `Not always`. Add qualifications only after the answer.
- **Define the mechanism concretely.** When explaining code, name what is constrained, what is allowed to repeat, and at what scope. Do not substitute a term like `coverage`, `reuse` or `sampling behavior` for the fact unless you define it in the same sentence.
- **Use one simple example when needed.** Prefer a small example over a general or abstract explanation.
- **Separate rules from observations.** State what the code guarantees separately from what happened in one generated dataset.
- **Minimal formatting and scope.** Do not use bold merely to emphasize words inside a sentence. Use headings only when they make a long response easier to navigate. Answer only what the user asked. Do not append generic recap, open-items, or "what this does not settle" sections. Include a limitation only when it is necessary for accuracy, and place it directly after the claim it qualifies.
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
