# Try 06 — max_new_tokens fix for try05's verbose schema

**Date:** 29-09-2026
**Scripts:** `author_reasoning_examples.py` (unchanged from try05) +
`lora_finetune_reasoning_loto.py` (one-line fix) + `lora_finetune_reasoning_loto.sb`
**Job:** submitted this try, see job ID below.

## What happened in try05's actual run (job 22454842)
Completed without crashing (exit 0:0), but every fold's result was
`top_choice: None`, `logic_consistency: 0/9 (0.00)`. Checked the raw model output
before concluding anything about model capability: `generate_completion()` still used
`max_new_tokens=500`, a budget sized for try04's old schema (one short paragraph).
try05's new schema asks for a verdict on every one of ~9 candidates, each with a
templated multi-sentence reason -- the actual completion needs on the order of
1500-2000+ tokens. The raw reply for every fold is JSON truncated mid-string (e.g.
cut off mid-sentence: `"a large rigid bis-aromatic bicyclic scaffold does` with no
closing quote/brace), so `extract_json()`'s regex+`json.loads` fails on all of them.
This is a bug in the harness, not a finding about the model -- the model was never
actually given room to finish an answer.

(Also observed a non-fatal CUDA OOM warning right before the first fold's training in
try05's log -- caching allocator retry, training proceeded and completed all 4 folds
regardless. Not touched here since it didn't cause a failure; noting it in case it
recurs with the larger generation budget.)

## The fix
One line in `generate_completion()`: `max_new_tokens=500` -> `max_new_tokens=2048`.
`author_reasoning_examples.py` is unchanged from try05 (the completion design itself
isn't in question here, only the generation budget).

## Status
Submitted as job **22544773**. COMPLETED (3:34:36). The fix worked -- every fold now
parses, with real `logic_consistency` scores (0.56-0.78) well above the ~0.33 chance
level for a 3-way true/false/partial verdict. Still 0/4 folds correct on final
ranking, but this is a real result now, not an infra failure.

## Result: coherent per-candidate logic, but necessary_condition bleeds across targets
Looked at the raw `necessary_condition` the model stated for each held-out fold
(the thing it's supposed to derive fresh from the prompt, never having seen this
target's own authored condition):

| held out | stated necessary_condition | matches |
|---|---|---|
| fabry | "mimic shape/charge/stereochemistry of a sugar transition state... secondary/contact... di-metal-binding hexose..." | blend of Fabry's real condition + confabulated phrasing not in any authored text |
| gaucher | "size/shape complementarity to a hydrophobic channel... hydrophobic anchor at entrance, polar anchor at bottom" | blend of TTR's phrasing ("polar anchor at the entrance") + generic "channel" language |
| cf | "secondary/modulatory chaperone contact... domain-domain interface... transport protein, no metal-dependent activity" | blend of Gaucher's framing + CF's real facts + a confabulated "metal-binding cofactor" detail in neither |
| attr | "size/shape complementarity to a protein-protein-like interface -- larger, rigid, multi-ring scaffold..." | **verbatim CF's condition**, word-for-word |

Despite never reconstructing the true held-out condition cleanly, **3 of 4 folds
still picked a candidate from the held-out target's own correct structural family**
(fabry_analog_2, gaucher_analog_3, cf_analog_2 -- each a real hard-negative analog of
that fold's own true reference, not a random or cross-target compound). That's
partial, genuine evidence of transfer: the model is recognizing "this family of
compounds belongs to this mechanism" cold, even without training on that family --
it just can't yet resolve the fine distinction within the family (reference vs. its
own near-miss analog), the same soft-fail pattern already seen in the numeric-predict
track's own hard-negative problem.

The `attr` fold is the one clean miss: it fully defaulted to CF's answer (lumacaftor,
a real cross-target compound, not even one of TTR's own candidates), driven by
borrowing CF's condition wholesale rather than a blend. The common thread across all
4: whenever two authored conditions share surface vocabulary (CF's and TTR's are both
about "a larger rigid scaffold fitting an interface between two protein-like things"),
the model struggles to keep them separate when generalizing cold. Fabry's condition
(sugar/transition-state/stereochemistry) is lexically distinct enough from the others
that it never gets fully overwritten by another target's text, even if it gets
blended with confabulated detail.

Worth noting for the diverse-training-phase doc (`01_txgemma_predict/NEXT_PHASE_
diverse_structural_training.md` has the analogous rationale for the numeric track):
this suggests the authored conditions' *phrasing distinctiveness*, not just their
factual correctness, matters for compositional transfer -- two genuinely different
mechanisms worded similarly are harder for the model to keep apart than two different
mechanisms worded distinctly, independent of how many targets are in the training set.
