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

Submitted as job **22994689**. Smoke test on 64 candidates first (exits non-zero on
failure), then the full 3420-candidate screen. Result pending.
