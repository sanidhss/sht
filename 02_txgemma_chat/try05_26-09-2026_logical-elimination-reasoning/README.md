# Try 05 — logical-elimination reasoning redesign (fixes try04's memorization)

**Date:** 26-09-2026
**Scripts:** `author_reasoning_examples.py` (rewritten) + `lora_finetune_reasoning_loto.py`
(updated) + `lora_finetune_reasoning_loto.sb`
**Job:** submitted this try, see job ID below.

## Why this try exists
try04 completed cleanly (0/4 folds correct) but the diagnosis written up there was
initially wrong, then corrected after diffing the model's actual generated text against
the training templates: the model was reciting other targets' fixed, hand-authored
`REASONING_CHAINS` paragraphs near-verbatim (e.g. CF's canned paragraph applied to the
Gaucher prompt) rather than reasoning from the candidate SMILES. Root cause: exactly one
authored reasoning paragraph existed per target, so verbatim memorization was strictly
easier than learning a transferable reasoning process.

Separately, the user asked for something structurally different from just "more
diversity": force the model into the kind of explicit, checkable elimination logic a
relational/graph model like TxGNN applies (checking a candidate against a
relation-specific compatibility criterion), but expressed through TxGemma's own
transformer/language strengths instead of trying to imitate a GNN's embeddings.

## What changed
`author_reasoning_examples.py` no longer authors one paragraph per target. Instead:

1. **`NECESSARY_CONDITION[target]`** — one explicit structural/physicochemical
   requirement the mechanism implies (e.g. Fabry: "must mimic the shape, charge, and
   exact stereochemistry of a small sugar ring at its transition state").
2. **`SCAFFOLD_CLASS[target/compound]`** — what each reference compound's (and easy
   negative's) real scaffold actually looks like.
3. **`WHY_CLASS_FAILS[target][other_scaffold]`** — a pairwise table (~12 entries) of
   why a compound from a *different* scaffold class fails a given target's necessary
   condition. This is the explicit "contraindication" logic table — the textual
   analogue of a link-prediction model's incompatible drug-disease embedding, but
   checkable and composed from (scaffold class x condition) instead of being one fixed
   narrative.
4. Own hard negatives get a **"partial"** verdict (coarse scaffold match, but the
   mechanism's dependence on exact stereochemistry is unconfirmed) — the correct nuance
   for a real, structurally-close analog rather than a hard fail.
5. The completion schema now requires a verdict **for every candidate**, not just a
   narrative about the top choice:
   ```json
   {"necessary_condition": "...",
    "candidate_verdicts": {"<label>": {"satisfies_condition": true|false|"partial", "reason": "..."}, ...},
    "ranking": [...], "top_choice": "<label>", "self_contradiction_check": "..."}
   ```
   Every fact used is a restatement of what try04's `REASONING_CHAINS` already
   established — nothing new is being claimed, it's decomposed into a compositional,
   checkable form instead of 4 monolithic paragraphs.

Since which candidates get sampled varies per augmented instance (same mechanism as
try04's random subsetting), the completion's actual JSON content now differs across
almost every training example for a given target — there is no single fixed block left
to memorize wholesale, unlike try04 where every augmentation of a target shared the
exact same reasoning text.

`lora_finetune_reasoning_loto.py` was updated to call the new `build_completion()` and
to score a second metric at cold-eval time: **logic_consistency** — the fraction of the
model's own per-candidate verdicts that match ground truth, independent of whether the
final top choice was correct. This directly measures whether the compositional rule
transferred to an unseen target, rather than only measuring the final answer.

## Verification before submitting
Two checks run before spending GPU queue time on this:
1. Ran `author_reasoning_examples.py` standalone (CPU-only, no GPU) and inspected
   `attr.json`'s completion: 9 distinct candidate verdicts, each with a
   scaffold-grounded reason, `top_choice` correctly pointing at tafamidis's label.
2. Called `make_augmented_instance()` directly (stubbing out `torch`/`transformers`/
   `peft` so the check runs in plain Python without waiting on a full ML-library
   import) for 3 different `aug_idx` values on the same target/fold: confirmed the
   generated completion JSON is genuinely different each time (`c0 != c1 != c2`), not
   just re-labeled -- the specific failure mode this redesign targets (one fixed,
   memorizable block per target) is gone at the data level, verified directly rather
   than assumed from the code reading right.

## Status
Submitted as job **22454842**. Result pending.
