# Try 09 — end-to-end: screen the library with the model's OWN cold-derived condition

**Date:** 18-10-2026
**Scripts:** `screen_cold_condition.py` + `screen_cold_condition.sb` (copied from try08, then patched)
**Job:** 22994689. Result pending.

## The gap this closes

The pipeline has two stages:

```
mechanism description -> [1] derive necessary condition -> [2] rank candidates -> shortlist
```

Both halves have now been measured, and they disagree:

- **try07:** stage 1 is broken on an unseen disease. Cold, the adapter emits a splice of
  a memorized condition plus a confabulated descriptor.
- **try08:** stage 2 works. Handed the *authored* condition, the base model cuts 3420
  approved drugs to a 12-compound positive set containing migalastat (rank 3, 570x
  enrichment).

**Nobody has run them together.** Every encouraging number in try08 is conditional on a
correct condition that deployment will never supply — for a genuinely new disease,
deriving the condition *is* the task. If a bad condition propagates, stage 2 will
faithfully return the wrong 12 compounds with equally confident margins.

This try measures the composition, which is the first number that describes the actual
system rather than one of its halves.

## Method

Rerun try08's screen with exactly one thing changed: the condition handed to the scorer
is the one **the model itself produced** on fabry while fabry was held out, lifted
verbatim from try07's log (job 22607104, `holdout_fabry` adapter, greedy decode) rather
than retyped. Same library, same scorer, same prompt, same metrics, so the delta is
attributable to the condition alone.

Extracted to `/work/hdd/bisa/sshah21/results/reasoning_screen/cold_condition_fabry.json`,
which stores both strings; the script asserts they differ so a mis-specified file fails
loudly instead of silently re-running the baseline.

**Authored (try08 baseline):**
> the candidate must mimic the shape, charge, and exact stereochemistry of a single small
> sugar ring at its reaction transition state -- compact, highly polar, with a ring
> position capable of carrying positive charge

**Cold, model-derived (this run):**
> the candidate must mimic the shape, charge, and stereochemistry of a sugar's reaction
> transition state at the enzyme's catalytic pocket -- a secondary/contact rather than a
> direct/bridge contact-type di-metal-binding hexose transition-state mimic

The diff is the point. The cold version keeps the broad idea (mimic a sugar's reaction
transition state) but **loses the decisive clause** — "compact, highly polar, with a ring
position capable of carrying positive charge", i.e. the iminosugar ring nitrogen that
makes migalastat work — and replaces it with gaucher's secondary/contact-vs-direct/bridge
framing plus "di-metal-binding hexose", a descriptor present in no authored text for any
target.

## Prediction, recorded before the result

Partial degradation rather than collapse, for a slightly uncomfortable reason: try08
showed the model only acts on the *broad* half of a condition anyway (its yes-set was
mostly plain sugars that fail the ring-nitrogen clause). The cold condition retains that
broad half. So the clause the cold condition lost may be one the model was already
ignoring.

If that holds, it is a weak kind of good news — robustness bought by the model not
reading the condition closely, not by stage 1 working. The "di-metal-binding" phrase is
the wildcard; it could pull the ranking toward metal chelators.

## How to read the outcome

- **migalastat stays in a small yes-set** -> the stages compose, an end-to-end shortlist
  is defensible, and the track has a real result to write up. Check *why* before
  celebrating: compare the yes-set membership against try08's, since surviving because
  the model ignores conditions is a different finding from surviving because the
  condition was good enough.
- **It collapses** -> stage 1 is the whole problem. Nothing further in screening matters
  until grounding is fixed, and the untested lever is the grounding/citation requirement
  from try07's conclusion.

## Changes from try08's script

- `--condition-file` overrides the authored condition, with an assert that it actually
  differs.
- Output path gains a `cold`/`authored` tag so this cannot overwrite try08's
  `screen_fabry_base.json`.
- Batch 12 instead of 16: try08 hit a CUDA OOM warning mid-screen and only completed
  because the caching allocator retried.

## Caveats

- **n=1**, one cold condition, greedy-decoded, one disease — and fabry is the fold where
  stage 2 is weakest (pure-stereoisomer hard negatives). This measures whether
  composition destroys or degrades the result, not the pipeline's accuracy.
- **Deliberate model mismatch:** the condition came from the *adapter*, the scorer is the
  *base* model. Real deployment would use one model for both. Holding the scorer fixed at
  try08's base model is what makes the comparison attributable to the condition alone;
  adapter-as-scorer is a sensible follow-up, not this run.

## Status

Job **22994689** COMPLETED 18-10-2026 17:44 (50m44s, exit 0).
Log: `/work/hdd/bisa/sshah21/results/reasoning_screen/screen_cold_22994689.out`
Ranked output: `.../screen_fabry_cold_base.json`

## Result: the stages do NOT compose — near-total collapse

| metric | authored condition (try08) | cold, model-derived (try09) |
|---|---|---|
| migalastat rank | **3 / 3420** | **991 / 3420** (top 29%) |
| recall@10 / @50 / @100 | HIT / HIT / HIT | **miss / miss / miss** |
| enrichment over random | 570x | **1.7x** |
| yes-set size | 12 (contains migalastat) | 3 (**does not** contain migalastat) |
| margin on migalastat | +0.748 | -4.636 |

Enrichment of 1.7x against a random baseline of 1.0x means the screen is **very nearly
worthless** once the condition comes from the model instead of from an author. The
prediction recorded above — partial degradation, on the theory that the model only acts
on the broad half of a condition — was **wrong**. It acted on the hallucinated half.

## The hallucination steers the screen, it does not merely dilute it

The cold condition's confabulated phrase was "di-metal-binding hexose". The resulting
top-50 is dominated by compounds carrying **literal metal counterions**: esomeprazole
magnesium, fosamprenavir calcium, strontium ranelate, bacitracin zinc, esomeprazole
strontium, levoleucovorin calcium, oxytetracycline calcium, pitavastatin magnesium,
doxycycline calcium, rifamycin sodium, novobiocin sodium, plus tetracyclines and
rifamycins, which are classic metal chelators.

Quantified rather than eyeballed (metal-bearing disconnected fragment in the SMILES):

| | metal-containing in top 50 | library baseline |
|---|---|---|
| authored condition | 7 / 50 (14%) | 298 / 3420 (8.7%) |
| cold condition | **23 / 50 (46%)** | 298 / 3420 (8.7%) |

A **5.3x enrichment of metal-containing compounds over baseline**, traceable to a single
invented phrase. The entire cold yes-set is `['CEFTAROLINE FOSAMIL',
'ESOMEPRAZOLE MAGNESIUM', 'FOSAMPRENAVIR CALCIUM']`.

So a stage-1 hallucination does not get averaged away by stage 2. Stage 2 applies it
faithfully and redirects the search to a different chemical class. The failure is
**confident, internally coherent, and wrong** — the output looks like a clean shortlist
either way, which is the dangerous property for a pipeline whose shortlist costs months
of wet-lab time to check.

Note the cross-target drugs are still correctly rejected (lumacaftor 1596, tafamidis
2625, caffeine 3003), so stage 2 is not broken. It is working correctly on a corrupted
specification.

## Retraction: there is no cheap self-check here

The 64-candidate smoke test showed a **0-compound** yes-set under the cold condition,
which suggested "empty positive set = untrustworthy run" as a free warning light. **That
did not replicate on the full library** — the cold yes-set is 3, not 0, against 12 for
the authored condition. A 12-vs-3 difference is too weak and too uncalibrated to act on,
and nothing else separates the two runs from the outside: both produce internally
coherent, chemically sensible top-50 lists. Telling them apart requires knowing the
mechanism, which is what the pipeline was supposed to supply.

Treat the earlier smoke-test-based claim as withdrawn.

## What this means for the track

- **An end-to-end claim is not supportable.** "Fine-tuned TxGemma identifies drugs for
  an unseen disease" fails here by a factor of ~330 in rank.
- **The stage-2 claim stands and is worth keeping.** Given a correct mechanism condition,
  bare TxGemma reduces 3420 approved drugs to 12 with the true drug retained, 570x
  enrichment, no tools and no retrieval. That is a real, defensible, human-in-the-loop
  result.
- **Stage 1 is now the only thing that matters.** No screening refinement — bigger
  libraries, salt dedup, better metrics, more folds — changes anything while the
  condition is hallucinated. The untested lever remains try07's conclusion: require the
  stated condition to quote or cite spans of the provided mechanism description.

## Known bug in this run (labeling only)

`report()`'s top-20 print loop used a local named `tag`, shadowing the new
condition-label parameter of the same name. The filename therefore came out as
`screen_fabry__base.json` with `condition_tag: ''` instead of `screen_fabry_cold_base.json`
with `cold`.

Scoring was unaffected — the log confirms the cold condition was used (`condition=cold`,
provenance line, full condition text printed) and all ranks are valid. By luck the
mangled name did not collide with try08's `screen_fabry_base.json`, so the baseline
survived. Fixed after the run by renaming the loop variable to `marker` (with a comment),
and the output file was copied to the correct name with `condition_tag` corrected. The
script as committed differs from the one that ran **only** in that variable name.

## Caveats

- n=1: one cold condition, greedy-decoded, on fabry. Worth repeating on CF or TTR to
  confirm the collapse is general rather than an artifact of this particular
  hallucination. The metal-steering mechanism is specific to the phrase "di-metal-binding";
  a different hallucination would misdirect differently, possibly less severely.
- Deliberate mismatch retained from the design: condition from the adapter, scoring by
  the base model, so that the only changed variable versus try08 is the condition.
