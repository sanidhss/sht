# Try 03 — First LoRA fine-tune (ChEMBL, easy negatives only)

**Date:** 21-09-2026
**Scripts:** `prep_chembl_lora_data.py` (data prep) + `lora_finetune_txgemma.py`
(training + eval) + `lora_finetune_txgemma.sb`
**Job:** 22252423 (COMPLETED)

## What it does
Pulls real ChEMBL IC50 data for GLA and GBA1, normalizes to a 0-1000 scale, holds out
the known reference drug (migalastat for GLA, isofagomine's absence noted — ambroxol
substituted for GBA1 in later tries) from training, and LoRA-fine-tunes
`txgemma-9b-predict` on the rest. Trains two variants: `blind` (reference excluded) and
`with_holdout` (reference included), to test both "does it generalize" and "does it work
as a screening tool" framings. Evaluates against the reference plus 5 unrelated "easy"
negative drugs (aspirin, caffeine, metformin, ibuprofen, imatinib).

## What happened / why it (sort of) "failed"
The headline result looked like a win: in the `blind` variant, migalastat scored 206
(best of 7) for GLA — genuinely held out, correctly ranked. This was the first real
positive signal in the whole investigation.

But two problems surfaced on closer inspection:
1. **The test was too easy.** The 5 negative drugs are chemically nothing like an
   iminosugar — a model doesn't need real structure-activity understanding to reject
   them, just coarse chemotype recognition. This motivated designing "hard negatives"
   (real, structurally-near-identical analogs) in try04/try05.
2. **The `with_holdout` variant produced a backwards, unexplained result** — aspirin and
   caffeine outscored migalastat even though migalastat's real label was directly in
   the training data. Root cause was never fully confirmed (suspected shared-adapter
   state leakage from training multiple LoRA adapters sequentially on one base model
   via `add_adapter`/`set_adapter`), but the fix (reloading a fresh base model per
   target/variant instead of sharing one) was carried forward into try05 as a
   precaution, not because the exact cause was proven.
