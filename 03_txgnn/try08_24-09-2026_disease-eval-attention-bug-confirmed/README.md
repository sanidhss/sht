# Try 08 — Both diseases on disease_eval; attention bug confirmed identical for both

**Date:** 24-09-2026
**Script:** `txgnn_fabry.py` with both Fabry and Gaucher switched to `disease_eval`
(fixing try07's seed-leak finding) — this version superseded by try09's exception-
visibility patch, not separately preserved.
**Jobs:** 22344594 (FAILED, Fabry), 22344595 (FAILED, Gaucher)

## What happened / why it failed
Both jobs crashed at the identical point: `KeyError: 'Wh_anatomy_anatomy'` in
`pretrain()`'s minibatch attention computation — the same bug first seen in try07,
now confirmed to affect **both** diseases identically once both use `disease_eval`
(since `disease_eval_idx` doesn't change whether the whole-graph `pretrain()` stage hits
`anatomy_anatomy` blocks — that's a property of the graph overall, not the split).

At this point the library's own error handling was still swallowing the real exception
and re-raising a bare, message-less `ValueError`, making it impossible to see the actual
`KeyError` without patching the exception handling first — done next, in try09.
