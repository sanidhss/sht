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
Submitted as job **22544773**. Result pending.
