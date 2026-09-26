# Try 05 — Correct index used; new bug, missing edge type

**Date:** 21-09-2026
**Script:** `txgnn_fabry.py` with the index fix applied (Fabry=543) — this specific
intermediate version not separately preserved; `attention=False`, single seed=42,
`complex_disease` split for both diseases at this point.
**Job:** 22248880 (FAILED, but Fabry succeeded)

## What happened
**Fabry succeeded again** (same result as try03: migalastat rank 2,863/7,957,
checkpoint saved to disk).

**Gaucher crashed** after completing all 500 finetune epochs, in TxGNN's own internal
final "Testing..." step: `dgl._ffi.base.DGLError: Edge type ('drug', 'contraindication',
'disease') does not exist.` Root cause: Gaucher genuinely has **zero** recorded
`contraindication`/`off-label use` edges in PrimeKG (only `indication`) — confirmed by
querying `kg.csv` directly. The `complex_disease` split's test set for this run happened
to be built entirely from Gaucher-only edges, and TxGNN's decoder code
(`txgnn/model.py`) unconditionally iterates over all 3 drug-disease relation types
without checking whether an edge type actually exists in that particular graph.

**Fix applied** (patched directly in the installed `txgnn` package): filter the
decoder's relation-type list down to only types actually present in the graph being
scored, rather than crash on a type with zero real edges — mathematically correct, since
zero edges of a type means nothing to score for it. This patch persists in
`envs/txgnn2`'s installed package and is required for every subsequent try.
