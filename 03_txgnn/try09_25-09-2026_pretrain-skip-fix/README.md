# Try 09 — pretrain() skipped for attention=True (current state)

**Date:** 25-09-2026
**Scripts:** `txgnn_fabry.py` (current, final) + `txgnn_fabry.sb` (current, final)
**Jobs:** 22365602 (Fabry), 22365604 (Gaucher) — **cancelled before starting, as part of
this reorg** (their `.sb` referenced the pre-reorg script path; will be resubmitted
against this folder's path once the migration is validated).

## What it does (current, final version)
Same as try08, plus: the exception handlers in the installed `txgnn/model.py` were
patched to print the real underlying exception (type, message, full traceback) instead
of a bare `ValueError` — confirmed the exact root cause via source inspection
(`AttHeteroRGCNLayer.forward()` writes `Wh_<etype>` only to `G.nodes[srctype]`, which in
a DGL block is a different underlying store than the destination side, even for a
same-type relation).

**The actual fix**: skip `model.pretrain()` entirely when `attention=True`, and go
straight from `model_initialize()` to `model.finetune()`. Verified safe by reading
`TxGNN.py`'s `finetune()` source directly: it reinitializes its own optimizer and
decoder weights and only depends on `self.model` (set by `model_initialize()`) and
`self.G` — nothing set exclusively by `pretrain()`. This also matches the original
paper's own reference `train.py`, which sometimes skips/limits pretraining.

## Status
Resubmitted from this folder as jobs **22411998** (Fabry) and **22411999** (Gaucher)
after the library reorganization was validated. Results pending.
