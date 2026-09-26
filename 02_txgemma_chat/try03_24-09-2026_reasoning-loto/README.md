# Try 03 — Leave-one-target-out reasoning fine-tune

**Date:** 24-09-2026
**Scripts:** `author_reasoning_examples.py` (hand-authored training content) +
`lora_finetune_reasoning_loto.py` (fold harness) + `lora_finetune_reasoning_loto.sb`
**Job:** 22365384 (FAILED, 14m14s)

## What it does
Directly addresses try02's failure: teaches mechanism-specific reasoning in a way meant
to generalize, not memorize. Uses **4 real, independently-verified disease/mechanism
examples**, deliberately spanning 3 structurally different chaperone mechanisms (not
just lysosomal storage disorders):
- Fabry (GLA) / migalastat — active-site substrate/transition-state mimicry
- Gaucher (GBA1) / ambroxol — same broad mechanism class, but a messier real case
  (ambroxol's own evidence describes "modulation," not clean mimicry)
- **Cystic fibrosis (CFTR) / lumacaftor** — folding/trafficking correction, no catalytic
  pocket at all
- **Hereditary transthyretin amyloidosis (TTR) / tafamidis** — native-tetramer
  interfacial stabilization, also no catalytic pocket

For each of 4 folds, trains on 3 targets (never exposing the held-out target's sequence,
mechanism, or reference drug — not even as a distractor) and evaluates cold on the 4th.
Since only 3 base examples exist per fold, training data is augmented per instance:
varying mechanism-description phrasing (hand-written paraphrases, same facts), which own
hard negatives/cross-target distractors/easy negatives appear, and the blind-label
ordering — ~12 instances per training target, ~36 per fold.

Two real bugs were caught and fixed before submitting: a schema mismatch (prompt asked
for a `self_contradiction_check` field the authored completions didn't produce), and an
unreliable `next(model.parameters()).device` call for placing inputs on a
`device_map="auto"`-sharded model (verified via PEFT/transformers source that this could
silently break on a sharded model — fixed to use the proven `model.device` pattern from
try01/try02).

## Known caveat (documented in the script itself)
Only 4 verified mechanism-class examples exist, so each fold trains on 3 examples' worth
of augmented scenarios — a thin base for teaching a transferable skill. A pass is
suggestive evidence of transfer, not proof of general orphan-disease reasoning ability.

## What happened / why it failed
Crashed 14m14s in, on the very first training batch of the first fold:
`TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'` in
`make_batch()`. Root cause: the same transformers-version quirk already fixed for
`model.generate()` calls elsewhere in this project (this environment's transformers
5.17.0 wraps `apply_chat_template()`'s output in a `BatchEncoding` rather than returning
a plain token-id list) — just missed in this script's own tokenization path, which
concatenates prompt and completion token lists with `+`. Fixed in try04 by extracting
`["input_ids"]` from the returned object.

