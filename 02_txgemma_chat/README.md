# Project: TxGemma-Chat (reasoning-based chaperone identification)

**Question:** Can `txgemma-27b-chat`'s general biochemical reasoning (not a numeric
score) identify the right chaperone/corrector/stabilizer for a target, using only the
mechanism described in the prompt — and can that reasoning be fine-tuned to generalize
to orphan disease mechanisms it's never seen, without becoming an agentic pipeline (no
docking/structure-prediction tool calls — that verification stays in the user's own
downstream pipeline by design)?

## Arc
1. **try01** — smoke test, unrelated to the science: confirmed a `generate()` API bug
   (fixed) and that the model loads/runs correctly.
2. **try02** — double-blinded (drug AND target identity hidden) single-shot test:
   failed. The model substituted a generic "complex/aromatic = good binder" heuristic
   for the real, narrow iminosugar-transition-state-mimicry mechanism, picking imatinib
   for both Fabry and Gaucher. Confident, fluent, wrong.
3. **try03** — leave-one-target-out fine-tune across 4 real, verified mechanism-class
   examples (2 lysosomal enzymes + a folding corrector + an oligomer stabilizer),
   explicitly designed to test cross-mechanism transfer rather than within-target
   interpolation. Failed immediately on a `BatchEncoding` vs. list bug in the training
   batch builder (same class of transformers-version issue as try01/try02, missed in a
   different function).
4. **try04** — bug fixed, ran to completion: **0/4 folds correct**. Diffing the
   model's generated reasoning against the authored training templates showed it was
   reciting other targets' fixed reasoning paragraphs near-verbatim (e.g. CF's
   template applied to the Gaucher prompt) rather than reasoning fresh from the
   presented SMILES — because `REASONING_CHAINS` has exactly one hand-authored
   paragraph per target, making verbatim memorization the easiest fit.
5. **try05** — redesigned the completion format entirely: instead of one fixed
   paragraph, the model must state an explicit necessary condition for the mechanism,
   then give every individual candidate a satisfies/fails/partial verdict checked
   against it, built compositionally from a (scaffold class x condition) table — the
   textual analogue of the compatibility check a knowledge-graph model like TxGNN
   applies geometrically, but expressed so the transformer has to apply a rule per
   candidate instead of reciting a block. Ran to completion (job 22454842) but every
   fold came back `top_choice: None` — not a model failure, a harness bug: the new
   schema is far more verbose (9 per-candidate verdicts) and `max_new_tokens=500`
   (sized for try04's one-paragraph schema) silently truncated every completion
   mid-JSON, so nothing parsed.
6. **try06 (current)** — one-line fix: `max_new_tokens=500` -> `2048`. Submitted as
   job 22544773.

## Current state
try04 was a clean negative with a verified root cause (one memorizable reasoning
example per target). try05's redesign is directionally right but never got a real
test — its own generation budget cut off the model before it could finish an answer.
try06 is the first run that should actually test the redesign. Result pending.

## Open question, raised mid-project
"Bare TxGemma" (what every try in this project runs — a single `generate()` call, no
retrieval) is not the same system as Google's benchmarked "Agentic-Tx" (Gemini 2.0 Pro
orchestrating 18 tools, of which bare TxGemma is only one; the other 17 include
PubMed/Wikipedia/web search and gene/protein/molecular database tools, via a ReAct
loop). The published rare-disease/complex-reasoning benchmark results are Agentic-Tx's,
not bare TxGemma's. This project has only ever tested the no-retrieval configuration —
worth deciding explicitly whether closing the gap requires tool-augmentation rather
than (or in addition to) further fine-tuning.
