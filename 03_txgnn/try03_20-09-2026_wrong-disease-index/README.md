# Try 03 — New environment; Fabry succeeds, Gaucher exposes the real index bug

**Date:** 20-09-2026
**Script:** `txgnn_fabry.py` — new env, `disease_idxs=[31384.]` still in use at this
point; not separately preserved (see try09 for current script).
**Env:** `envs/txgnn2` (dgl 2.4.0, torch 2.1.0) — first use, worked mechanically.
**Job:** 22240098 (FAILED, but partially succeeded)

## What happened
**Fabry actually trained fully this time** — 2 epochs pretrain + 500 epochs finetune,
completed, checkpoint saved, `eval_disease_centric` even produced output. This is where
we first computed migalastat's real rank: **2,863rd out of 7,957 drugs** — not a crash,
but not a good result either (a genuinely disappointing, honest finding, not a bug).

**The job then crashed** (in the disease-centric evaluation step, running both Fabry and
Gaucher): `dgl._ffi.base.DGLError: The given number of nodes of node type disease must
be larger than the max ID in the data, but got 17080 and 31384`. This is the real
root-cause confirmation: `31384` is Fabry's
row number in PrimeKG's raw, global `node.csv`/`kg.csv` (spanning ALL ~129k nodes of
every type) — but TxGNN's `eval_disease_centric` expects its own internal,
disease-type-local index (0-17079, computed separately per node type during
`prepare_split()`). Traced through `kg_directed.csv`'s `x_idx`/`y_idx` columns to find
the correct value: Fabry disease's real local index is **543** (not 31384). This fix
carried forward to try05 onward.
