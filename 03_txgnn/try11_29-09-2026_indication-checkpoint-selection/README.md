# Try 11 — indication-specific checkpoint selection (fixes seed variance)

**Date:** 29-09-2026
**Scripts:** `txgnn_fabry.py` (RESULTS_DIR changed to force retraining, see below) +
`txgnn_fabry.sb`
**Jobs:** see below.

## Why this try exists
try10's real result showed large seed-to-seed variance for Gaucher specifically:
miglustat ranked 7th on seed 2, 23rd on seed 3, but 3724th on seed 1 -- same data,
same hyperparameters, only the random seed differs. The user's hypothesis: this looks
like an optimization/checkpoint-selection problem, not a fundamental model failure,
given 2 of 3 seeds already look strong.

## Root cause, verified by reading `finetune()`'s training loop
`TxGNN.py`'s checkpoint selection (every `valid_per_n=25` epochs) picks whichever
epoch has the highest `macro_auroc` on validation. But `macro_auroc` is averaged
across **all 3** `dd_etypes` present in the validation split (contraindication +
indication + off-label -- confirmed back in try10 that Fabry/Gaucher's `disease_eval`
validation split contains all 3, unlike test which only has indication). Indication --
the only relation actually tested -- is just 1 of 3 components in that average. A
checkpoint can win "best macro AUROC" by being strong at contraindication/off-label
while mediocre at indication specifically, and which checkpoint wins that tradeoff is
exactly the kind of thing that would vary by seed.

## Free confirmation before touching any code: ensembling
Averaged raw scores across the 3 already-trained try10 seeds (no retraining) instead
of trusting any single seed's rank:
- Fabry: migalastat ensembled rank 16/7,957 (consistent with per-seed results).
- Gaucher: **miglustat ensembled rank 17/7,957** -- up from a per-seed mean of 1251
  (7th/23rd/3724th). Eliglustat and velaglucerase alfa also improved modestly.
  The 3 biologics with the exact-0.0 artifact stayed flat (~6070) even after
  ensembling, confirming that's a *different*, structural problem (likely missing
  node features, not checkpoint noise) -- ensembling only cancels random noise, and
  it didn't touch that number at all.

This strongly supports the checkpoint-selection-noise diagnosis: the real underlying
model already "knows" miglustat is a strong Gaucher candidate (both seed 2/3 and the
ensemble agree), it's specifically seed 1's *selected checkpoint* that missed it.

## The fix
Patched the installed library (`/work/hdd/bisa/sshah21/envs/txgnn2/.../txgnn/TxGNN.py`,
`finetune()`, ~line 271-286): checkpoint selection now uses the mean of
`auroc_rel[('drug','indication','disease')]` and `auroc_rel[('disease','rev_indication','drug')]`
when present (falls back to `macro_auroc` otherwise), instead of the 3-relation
macro average. Also relabeled the log line from "Best Macro AUROC" to "Best Indication
AUROC" so the printed logs don't misdescribe what's actually being selected on.

**`RESULTS_DIR` changed** to `.../txgnn_v2_indication_selection` (a new path, not
try10's) -- necessary because `run_disease_eval()` skips retraining entirely if a
checkpoint already exists at the target path, and try10's checkpoints were selected
under the *old* (macro-AUROC) criterion. Reusing them would silently defeat the whole
point of this fix. This means a full retrain, not a fast reuse-checkpoint run.

## Status
Submitted: fabry = job **22544937**, gaucher = job **22544938**.
Expect ~7-8h each (full retrain), matching try09/try10's actual training time.
