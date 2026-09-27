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
4. **try04 (current)** — bug fixed, ran to completion: **0/4 folds correct**. Diffing
   the model's generated reasoning against the authored training templates showed it
   was reciting other targets' fixed reasoning paragraphs near-verbatim (e.g. CF's
   template applied to the Gaucher prompt) rather than reasoning fresh from the
   presented SMILES — because `REASONING_CHAINS` has exactly one hand-authored
   paragraph per target, making verbatim memorization the easiest fit.

## Current state
First completed run, and it's a clean negative, with a verified root cause: one
authored reasoning example per target is too little diversity to force genuine
structure-grounded reasoning instead of memorization. The fix implied is not "add more
targets" but multiple independently-worded reasoning chains per target.
