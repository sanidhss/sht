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
10. **try10** — multi-disease validation of the stage-2 screen (job 23114793, COMPLETED
   24-10-2026). Runs try08's exact protocol on all four folds. **try08 does not
   generalize:** fabry rank 4/3420 (427x enrichment) but gaucher 1143, cf 993, attr 1774
   — 1.0-1.7x, at or near chance. Diagnosis: the screen converts a *physicochemical*
   specification into a ranking, and only fabry's condition is one; gaucher's is
   pharmacological/negative (yes-set = 38.5% of the library), cf's is applied correctly
   but is not selective (top-20 is kinase inhibitors and antivirals, all genuinely
   "large rigid multi-ring"). Also records the RDKit descriptor baseline and fixes a
   latent bug: ambroxol is absent from ChEMBL `max_phase=4`, so the gaucher fold would
   have silently had no true drug to rank.

9. **try09** — end-to-end composition test (job 22994689, COMPLETED 18-10-2026). Reruns
   try08's screen with one variable changed: the condition is the one the model itself
   derived cold on fabry, lifted verbatim from try07's log. **Negative and decisive:**
   migalastat falls from rank 3 to **991 / 3420**, recall@50 HIT -> miss, enrichment 570x
   -> **1.7x** (random = 1.0x). The confabulated phrase "di-metal-binding hexose" steers
   rather than dilutes — top-50 goes to 46% metal-counterion compounds vs an 8.7% library
   baseline. Stage 2 is not broken (cross-target drugs still rejected); it faithfully
   applies a corrupted specification.

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

### History
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

try09 then composed the two stages and the result is **negative and decisive**: feeding
stage 2 the condition the model itself derived cold (lifted from try07's log) drops
migalastat from rank 3 to **991 / 3420**, recall@50 from HIT to miss, and enrichment from
570x to **1.7x** against a random baseline of 1.0x. The stages do not compose.

Worse than the magnitude is the mechanism. The cold condition's confabulated phrase
"di-metal-binding hexose" **steered** the screen rather than merely degrading it: the
top-50 went to 46% metal-counterion-bearing compounds against an 8.7% library baseline
(esomeprazole magnesium, strontium ranelate, bacitracin zinc, calcium salts,
tetracyclines), a 5.3x enrichment traceable to one invented phrase. Stage 2 is not
broken — cross-target drugs are still correctly rejected — it applies a corrupted
specification faithfully. The failure is confident, internally coherent and wrong, and
produces a shortlist indistinguishable from a good one without already knowing the
answer. A self-check seen in try09's smoke test (empty positive set) did **not**
replicate on the full library and is withdrawn.

## Current state

What this track can honestly claim, and what it cannot:

- **Narrow and conditional:** given a *crisp physicochemical* condition, the model ranks
  a realistic library very well — but **on 1 of 4 mechanisms tested** (try10). fabry:
  rank 4/3420, 427x enrichment. gaucher 1143, cf 993, attr 1774 — 1.0-1.7x enrichment,
  i.e. at or near chance. try08's headline number was a single-fold result and does not
  generalize. A three-line RDKit descriptor filter lands within ~2.7x of the model on
  fabry, the one fold that works.
- **The dominant variable is condition authoring, not model capability.** The screen
  converts a *property specification* into a ranking. fabry's condition is one
  ("compact, highly polar, ring position able to carry positive charge"). gaucher's is
  pharmacological and negative ("modulatory contact rather than a clean mimic, modest
  potency") and yields a yes-set of 38.5% of the library. cf's ("larger, rigid,
  multi-ring") is applied *correctly* but describes hundreds of modern kinase inhibitors
  and antivirals, so it cannot single out lumacaftor.
- **Not supported:** end-to-end identification for an unseen disease. try09 fails this by
  ~330x in rank. Nor is a general-purpose mechanism-to-shortlist screen supported, per
  try10.
- **No self-check has survived validation.** Two candidate warning lights for a bad run
  (empty yes-set, try09; small yes-set, try10) each held on one fold and failed on the
  others. Nothing yet distinguishes a good shortlist from a bad one without ground truth.
- **Also established:** the model does no within-class discrimination (try08 —
  migalastat vs its own stereoisomer separate by 1.8e-7), which is the natural handoff
  point to the downstream docking stage rather than a gap to fine-tune away.

Open and unsolved: **stage 1**, and it is now the only thing that matters. No screening
refinement — bigger libraries, salt-form dedup, more folds, better metrics — changes
anything while the derived condition is hallucinated. The untested lever is still
try07's conclusion: require the stated condition to quote or cite spans of the provided
mechanism description, so a memorized or invented one cannot be substituted.

## Open question, raised mid-project
"Bare TxGemma" (what every try in this project runs — a single `generate()` call, no
retrieval) is not the same system as Google's benchmarked "Agentic-Tx" (Gemini 2.0 Pro
orchestrating 18 tools, of which bare TxGemma is only one; the other 17 include
PubMed/Wikipedia/web search and gene/protein/molecular database tools, via a ReAct
loop). The published rare-disease/complex-reasoning benchmark results are Agentic-Tx's,
not bare TxGemma's. This project has only ever tested the no-retrieval configuration —
worth deciding explicitly whether closing the gap requires tool-augmentation rather
than (or in addition to) further fine-tuning.
