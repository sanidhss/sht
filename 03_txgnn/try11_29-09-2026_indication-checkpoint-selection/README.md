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
Submitted: fabry = job **22544937**, gaucher = job **22544938**. Both COMPLETED
(7:41h / 7:54h). Confirmed the patch actually ran (log shows "Best Indication AUROC",
not the old "Best Macro AUROC" label) before trusting the result.

## Result: hypothesis rejected by the data (01-10-2026)
**The fix did not change the outcome.** Fabry: migalastat mean rank 17.3 (12/21/19 --
essentially identical to try10's 17.7/12-22-19). Gaucher: miglustat mean rank 1233.7
(3669/7/25 -- essentially identical to try10's 1251.3/3724-7-23). Same variance,
same pattern, under the corrected checkpoint-selection criterion.

Checked seed 1 vs. seed 2's own validation trajectories directly rather than just
accept the negative result: seed 1's final indication AUROC is **0.8079**, seed 2's
is **0.7960** -- seed 1's aggregate indication-discrimination quality is comparable
to, even slightly *better than*, seed 2's. If checkpoint selection (or overall
indication-AUROC quality) were the cause, the "bad" seed should show worse validation
metrics. It doesn't.

**Conclusion: the original hypothesis was wrong.** This isn't a training/
selection bug -- it's inherent stochasticity in the learned embedding space itself.
Aggregate indication AUROC is averaged across every disease-drug indication edge in
the validation graph (hundreds of diseases); it says nothing about whether one
specific (miglustat, Gaucher) pair lands in a good relative position in a given
training run's particular embedding geometry. Two seeds can have equally good
*overall* discrimination ability while placing one specific drug very differently,
because the random initialization/sampling trajectory shapes the embedding space
itself, not just which checkpoint gets picked from it.

This means the checkpoint-selection patch is a harmless, arguably-more-correct change
(it does align selection with what's actually tested) but it does **not** solve the
variance problem it was built for. The validated mitigation remains what was already
found for free in try10: **ensembling scores across multiple independently-trained
seeds** (miglustat 17/7,957 after averaging). That -- not this patch -- is the actual
adopted fix going forward. Keeping the patch in the installed library since it's not
wrong, just insufficient on its own.
