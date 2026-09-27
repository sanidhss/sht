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
4. **try04 (current)** — bug fixed, ran to completion: **0/4 folds correct**. Training
   loss collapsed hard in every fold (thin 36-example base), and the `attr` fold
   reverted to the same generic "complex-molecule" heuristic seen in try02's zero-shot
   test — the fine-tune suppressed that heuristic for in-distribution mechanisms but
   didn't replace it with real transferable reasoning.

## Current state
First completed run, and it's a clean negative: leave-one-target-out fine-tuning on 3
held-in mechanisms (36 examples) does not generalize to a 4th, unseen mechanism class.
Not a bug — an honest result about how much mechanism diversity this approach needs.
