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
6. **try06** — one-line fix: `max_new_tokens=500` -> `2048`. Worked: every fold now
   parses, logic_consistency 0.56-0.78 (well above the ~0.33 chance level). Still
   0/4 on final ranking, but 3 of 4 folds correctly localized to the held-out
   target's own structural family cold (picked a real hard-negative analog of the
   true answer, not a random/cross-target compound) — partial, genuine transfer
   evidence. The failures trace to the model blending/confusing different targets'
   authored necessary-conditions when they share surface vocabulary (CF's and TTR's
   conditions are both "rigid scaffold fits an interface," and the `attr` fold fully
   borrowed CF's condition verbatim as a result).

7. **try07** — grounding-vs-retrieval diagnostic, no retraining (reuses try06's saved
   `holdout_fabry` adapter). Job 22607104, COMPLETED 04-10-2026. Clean separation: on
   an **in-training** target (gaucher) the model restates the authored
   necessary_condition character-for-character and reasons coherently from it; **cold**
   on the held-out target (fabry) it emits a splice of the right idea plus a training
   target's framing plus a confabulated descriptor present in no authored text, then
   describes every candidate in the training target's scaffold vocabulary. So the
   try06 redesign is sound in distribution — the failure is specifically
   retrieval-over-derivation under distribution shift. Caveat: n=1 per arm, and the
   verbatim in-distribution match is itself a memorization signal.
8. **try08** — stage-2 library-screen prototype. Job 22664818, COMPLETED 06-10-2026.
   Drops the 9-candidate multiple-choice format for rank-of-true-drug over 3417 ChEMBL
   approved drugs plus the authored analogs spiked in, and hands the model the authored
   condition so condition-*application* is measured separately from
   condition-*derivation*. Scores by yes/no logprob margin in one forward pass per
   candidate rather than a 2048-token generation, which is what makes a pool that size
   affordable. **First positive quantitative result in this track:** base model, no
   adapter, migalastat ranks **3 / 3420** (top 0.09%, 570x enrichment), and the model's
   entire positive set is **12 compounds** with the true drug inside it — a 285x library
   reduction. Cross-target drugs correctly rejected (lumacaftor 2361, tafamidis 2835).
   But discrimination is **class-level only**: migalastat and its own stereoisomer differ
   by 1.8e-7 (numerical noise), the yes-set is mostly plain sugars lacking the ring
   nitrogen the condition explicitly requires, and the same molecule as an HCl salt
   swings ~1.0 in margin.

## Current state
try04 was a clean negative with a verified root cause (one memorizable reasoning
example per target). try06's redesign shows real, partial transfer — family-level
recognition cold, but fine discrimination within a family (and keeping two
similarly-worded mechanisms separate) still fails. try07 then localized *why*: the
model grounds fine on targets it trained on and only substitutes a memorized condition
when the target is genuinely unseen.

**Superseded:** an earlier reading of try06 suggested the next lever was making the
authored conditions more lexically distinct from each other. try07 contradicts that —
the model grounds correctly in distribution *despite* CF and TTR sharing
interface-stabilizer vocabulary, so lexical overlap is not the mechanism of failure.
Forcing artificial distinctness would also teach a false lesson, since those mechanisms
genuinely are related. The indicated lever is instead a grounding/citation requirement
in the task prompt, forcing the stated condition to cite spans of the mechanism
description it was given.

Two measurement problems surfaced alongside this, both eval-design rather than model
failures, and both addressed in try08:
- **The 9-candidate format is far easier than the deployment task** (true drug + 3
  authored analogs + the 3 other folds' drugs + caffeine/aspirin), and top-1 over 4
  folds is only 4 bits of signal, so 0/4 vs 1/4 is unreadable. Rank over a real library
  is continuous and readable at n=1 disease. Since docking verification lives
  downstream, **recall@k is the metric that matches the role; top-1 never was** — which
  reframes try06's "right structural family 3 of 4 times" as a partial pass.
- **Condition-derivation and condition-application were fused** into one number, so a
  wrong answer could come from either. try08 isolates the second.

Note on scaling: expanding the *candidate pool* needs only SMILES and is free, but
expanding the *training* side needs hand-authored mechanism text per disease, and there
are only on the order of tens of well-documented pharmacological chaperone mechanisms.
That asymmetry is why the pool can scale now while the disease count cannot.

try08 then split the metric and got the track's first positive number: handed the
authored condition, the base model reduces 3420 approved drugs to a 12-compound positive
set containing migalastat (rank 3, 570x enrichment), and correctly dumps the other
folds' drugs. So **condition-application works at the class level** — and the earlier
0/4 results were partly a fused, low-resolution metric rather than pure model failure.

What try08 also establishes is a hard limit: the model does **no** within-class
discrimination. migalastat vs its own stereoisomer separates by 1.8e-7, and the yes-set
is populated by plain sugars that fail the condition's decisive clause (a ring position
able to carry positive charge). This is not a gap to close by fine-tuning — it is the
boundary where the textual stage should hand off to downstream docking, which resolves
stereochemistry geometrically. The two-stage split is therefore the right architecture,
with the division of labour now measured rather than assumed.

Open and unsolved: **stage 1**. try08 handed the condition in, so it says nothing about
deriving one. try07's retrieval-under-shift failure is still the live problem, and the
grounding/citation requirement is still the untested fix.

## Open question, raised mid-project
"Bare TxGemma" (what every try in this project runs — a single `generate()` call, no
retrieval) is not the same system as Google's benchmarked "Agentic-Tx" (Gemini 2.0 Pro
orchestrating 18 tools, of which bare TxGemma is only one; the other 17 include
PubMed/Wikipedia/web search and gene/protein/molecular database tools, via a ReAct
loop). The published rare-disease/complex-reasoning benchmark results are Agentic-Tx's,
not bare TxGemma's. This project has only ever tested the no-retrieval configuration —
worth deciding explicitly whether closing the gap requires tool-augmentation rather
than (or in addition to) further fine-tuning.
