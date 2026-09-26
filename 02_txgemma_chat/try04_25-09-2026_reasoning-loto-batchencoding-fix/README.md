# Try 04 — BatchEncoding fix applied (current state)

**Date:** 25-09-2026
**Scripts:** `author_reasoning_examples.py` + `lora_finetune_reasoning_loto.py` (fixed)
+ `lora_finetune_reasoning_loto.sb`
**Job:** not yet submitted as of this reorg.

## What changed from try03
One-line fix in `make_batch()`: `tok.apply_chat_template(messages,
add_generation_prompt=True)["input_ids"]` instead of using the returned object directly
— it's a `BatchEncoding` in this environment's transformers version, not a plain list,
so concatenating it with `+` failed immediately on the first training example.

## Status
Submitted from this folder as job **22412000** after the library reorganization was
validated. This will be the first run to actually attempt training under the
leave-one-target-out design — try01-03 in this project never got past infrastructure
bugs of one kind or another. Result pending.
