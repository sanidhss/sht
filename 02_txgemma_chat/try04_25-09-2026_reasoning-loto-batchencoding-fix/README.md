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

Training loss collapsed hard within each fold (epoch 0 ~1.9-2.1 -> epoch 2 ~0.01-0.02).

**Correction (26-09-2026):** an earlier version of this note claimed the model "reverted
to a complex-molecule heuristic" on the `attr` fold, by pattern-matching to try02's
different (zero-shot, unrelated) failure mode without actually re-checking this run's
text. That claim was wrong. The real, verified mechanism, found by diffing the model's
generated reasoning against `author_reasoning_examples.py`'s `REASONING_CHAINS` dict:

`REASONING_CHAINS["cf"]` (fixed, hand-authored training text) says: *"The reference
compound is a bis-aromatic molecule with a rigid spirocyclopropyl-linked bicyclic core,
consistent with a scaffold built to wedge into and rigidify a specific **domain
interface**..."*

The model's actual output on the **Gaucher**-held-out fold (CF was one of its 3
training targets) says: *"The reference compound is a bis-aromatic molecule with a
rigid spirocyclopropyl-linked bicyclic core, consistent with a scaffold built to wedge
into and rigidify a specific **protein-protein-like interface**..."* — a near-verbatim
recitation of CF's canned paragraph, misapplied to describe ambroxol. Same pattern on
`attr`: its output blends fixed phrases from Fabry's authored template ("mimicking the
shape, charge... sugar ring") and Gaucher's ("aminocyclohexanol"), describing neither
aspirin (what it picked) nor tafamidis (the real answer).

Root cause: `REASONING_CHAINS` has exactly **one fixed, hand-authored paragraph per
target** — the 12x-per-target augmentation only reshuffles which candidate gets which
letter label, it never varies the reasoning text itself. With zero diversity in what
"correct reasoning" looks like for a given target, verbatim memorization of that one
paragraph is a much easier fit than learning to derive structure-function claims from
the actual SMILES in the prompt — and that's what happened. On an unseen target at test
time, there's no matching memorized paragraph, so the model splices together the
closest-sounding ones it does have.

This is an honest, clean negative, not a bug: one authored reasoning example per target
gives a fine-tune every incentive to memorize phrasing instead of learning a
transferable reasoning process. The fix implied by this diagnosis is not "add more
targets" but **multiple independently-worded authored reasoning chains per target**, so
memorizing one fixed paragraph stops being a viable shortcut.
