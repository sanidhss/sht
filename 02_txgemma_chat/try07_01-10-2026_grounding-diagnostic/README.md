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
Submitted as job **22607104**. Result pending.
