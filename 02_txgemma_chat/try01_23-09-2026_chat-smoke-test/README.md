# Try 01 — Chat model smoke test

**Date:** 23-09-2026 (this is the user's own script, not authored in this session)
**Script:** `chat_txgemma.py` + `chat_txgemma.sb`
**Jobs:** 22323972 (FAILED), 22344310 (COMPLETED)

## What it does
Minimal sanity check: loads `google/txgemma-27b-chat`, asks one generic ADMET question
via `tok.apply_chat_template(...)` + `model.generate(...)`, prints the reply and GPU
memory usage. No disease-specific content — just confirms the chat model loads and runs.

## What happened / why it failed (first attempt)
22323972 crashed with `AttributeError`/`KeyError: 'shape'` inside `generate()`. Root
cause: in this environment's transformers version (5.17.0), `apply_chat_template(...,
return_tensors="pt")` returns a `BatchEncoding` (dict-like object), not a raw tensor —
but the script called `model.generate(inputs, ...)` positionally, which requires a
tensor. Fixed by requesting `return_dict=True` and calling `model.generate(**inputs,
...)` instead. 22344310 (after the fix) completed cleanly with a coherent, correct
answer.
