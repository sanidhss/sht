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
validated. This was the first run to actually attempt training under the
leave-one-target-out design — try01-03 in this project never got past infrastructure
bugs of one kind or another. **COMPLETED** 26-09-2026 (00:53:54 elapsed, exit 0:0) — the
first clean run in this project's history.

## Result: 0/4 folds correct
Each fold trained on 36 augmented instances (12 per target x 3 held-in targets), held
out the 4th target's mechanism entirely, and asked for a cold ranking over 9 candidate
compounds.

| held out | expected      | model's top pick        | correct |
|----------|---------------|--------------------------|---------|
| fabry    | migalastat    | (unlabeled distractor)   | No      |
| gaucher  | ambroxol      | gaucher_analog_3         | No      |
| cf       | lumacaftor    | cf_analog_2              | No      |
| attr     | tafamidis     | aspirin                  | No      |

Training loss collapsed hard within each fold (epoch 0 ~1.9-2.1 -> epoch 2 ~0.01-0.02),
consistent with the model memorizing surface patterns in that fold's 36 examples rather
than learning transferable mechanism reasoning — the same overfitting signature as the
numeric-predict track's memorization failure (see `01_txgemma_predict/try04`), but here
showing up as a genuine cross-mechanism generalization failure rather than a
same-target near-neighbor confusion.

The `attr` fold is the clearest tell: the model reverted to picking aspirin, the same
"complex/generic-looking molecule" heuristic failure documented back in try02's
zero-shot double-blind test — the fine-tune didn't durably replace that heuristic, it
just suppressed it for the 3 mechanisms it was trained on.

This is an honest, clean negative, not a bug: 3 held-in mechanisms / 36 examples total
is a thin base for teaching a 27B model genuine cross-mechanism transfer. The result
says this specific attempt doesn't generalize, not that reasoning-based generalization
is impossible for this task.
