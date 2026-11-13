# Try 01 — First TxGNN attempt, old environment

**Date:** 19-09-2026
**Script:** `txgnn_fabry.py` — **earliest version, overwritten in-place since; not
separately preserved.** At this stage it called `model_initialize()` -> `pretrain()` ->
`finetune()` -> `save_model()` -> `eval_disease_centric()` in one pass, no blindness
checking, no seeds, using `disease_idxs=[31384.]` (see try03 for why that number is
wrong).
**Env:** `envs/txgnn` (dgl-cu113==0.9.1, torch 1.12.1)
**Job:** 22226757 (FAILED)

## What it does (at this stage)
Trains and evaluates TxGNN on PrimeKG for Fabry disease, using the `complex_disease`
split, then runs `eval_disease_centric` to rank drugs.

## What happened / why it failed
`dgl._ffi.base.DGLError: Empty heterograph is not allowed`, thrown inside
`model_initialize()` while constructing the evaluation graph. Root cause: `disease_idxs
=[31384.]` was the wrong ID space (PrimeKG's raw global `node.csv` row index, not
TxGNN's internal disease-type-local index — see try03), so essentially all edges for
"disease 31384" didn't exist in the local index space, producing an empty graph
construction. Also evidence in the log of a stale cached split directory
(`disease_eval_[31384.0]`) already present, indicating this wrong ID had been used in an
even earlier, unlogged attempt.
