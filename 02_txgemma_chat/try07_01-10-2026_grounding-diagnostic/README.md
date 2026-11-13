# Try 07 — grounding-vs-retrieval diagnostic (not a new fine-tune)

**Date:** 01-10-2026
**Scripts:** `grounding_diagnostic.py` + `grounding_diagnostic.sb`
**Job:** see below.

## What this tests
try06's result showed the model, cold on the held-out TTR fold, stated a
necessary_condition that was verbatim CF's authored text -- not a paraphrase, not a
blend, a direct recitation of a training target's condition rather than something
derived from TTR's own mechanism description, which was sitting right there in the
prompt. Before redesigning the authored descriptions to be more lexically distinct
(the fix originally considered), this checks a more specific hypothesis: is this a
distribution-shift problem (model defaults to retrieving a memorized condition only
when the target is genuinely unseen) or a general grounding weakness (model doesn't
reliably derive conditions from the prompt's own text even for targets it trained on)?
These imply different fixes -- the first needs an explicit grounding/citation
requirement in the task prompt; the second would mean the compositional-verdict
redesign itself isn't working as intended even in-distribution.

## Method
No retraining. Loads the `holdout_fabry` adapter already saved by try06's job
22544773 (trained on gaucher/cf/attr, fabry held out) and runs it on two prompts:
- **gaucher's own prompt** (in-training -- gaucher's authored example WAS in this
  adapter's training data) -- does it correctly restate gaucher's own condition?
- **fabry's own prompt** (held-out/cold -- same as try06's actual eval) -- for direct
  side-by-side comparison against the first case, same adapter, same model weights,
  only the prompt's target differs.

## Status
Job **22607104** COMPLETED 04-10-2026 08:35 (18m27s, exit 0).
Log: `/work/hdd/bisa/sshah21/results/reasoning_loto/grounding_diag_22607104.out`

## Result: distribution shift, not a general grounding weakness

The two arms separate cleanly.

**In-training (gaucher, seen by this adapter):** the model's stated
`necessary_condition` is **character-for-character identical** to the authored ground
truth. Its per-candidate verdicts follow from that condition coherently, and the
`self_contradiction_check` is consistent with its own ranking.

**Held-out (fabry, cold -- same adapter, same weights, only the target differs):** the
condition comes back as a *splice*. It opens with roughly the right fabry idea (mimic a
sugar's reaction transition state at the catalytic pocket), then welds on gaucher's
"secondary/contact rather than direct" framing plus a "di-metal-binding hexose"
descriptor that **appears in no authored text for any target** -- a confabulation, not a
retrieval. The per-candidate reasons then describe essentially every scaffold in
gaucher's vocabulary ("aminocyclohexanol"), including self-contradictory lines that
reject a compound for lacking the very group the same sentence says it bears.

## What this rules in and out

The try06 compositional-verdict redesign is **not** broken in principle. In
distribution it reproduces the condition exactly and reasons consistently from it. The
failure is specifically **retrieval-over-derivation under distribution shift**: cold,
the model reaches for a memorized training condition and that target's scaffold
vocabulary instead of deriving from the mechanism description sitting in its own
prompt, which contains everything needed.

This is the first of the two branches this try was built to distinguish, so it supports
adding a **grounding/citation requirement** to the task prompt (force the condition to
quote or point at spans of the provided mechanism description) rather than a deeper
redesign. It also independently supports the already-made decision *not* to artificially
de-duplicate the authored mechanism descriptions: the model grounds fine when the target
is in training even though CF and TTR share interface-stabilizer vocabulary, so lexical
overlap between mechanisms is not what breaks it.

## Caveats -- read before building on this

- **n=1 per arm.** One in-training target, one held-out target, one adapter. Suggestive,
  not measured. The other three adapters are already saved and need no retraining, so
  extending this is cheap and remains worth doing.
- **The in-distribution arm being a *verbatim* match is itself a memorization signal.**
  It happens to be the correct memory here, so it reads as success, but it is not
  evidence of derivation -- only that retrieval and derivation agree in distribution. A
  stronger test would perturb the in-training mechanism description and check whether
  the stated condition follows the perturbation or the memory.
- **Fabry is the harshest possible held-out fold**, independent of grounding: its three
  authored hard negatives are pure stereoisomers of migalastat (identical constitution,
  differing only in `@`/`@@`), while the cf/attr/gaucher analogs are constitutional
  variants. The authored mechanism text requires matching the "stereochemistry" of the
  terminal alpha-linked sugar but never names that sugar's configuration, so
  exact-stereoisomer discrimination may not be derivable from the prompt at all.
  Repeating this diagnostic on a non-fabry fold is the fairer read.

## What followed

try08 does not pursue the grounding fix yet. It first isolates the *other* half of the
problem -- applying an already-stated condition to rank a realistic library -- because
try02-try07 fused condition-derivation and condition-application into a single top-1
number over 9 candidates, which is a large part of why these failures were hard to
attribute.
