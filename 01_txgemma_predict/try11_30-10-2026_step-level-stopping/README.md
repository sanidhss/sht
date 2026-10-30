# Try 11 — Step-level early stopping, 4 targets × 3 seeds, dual readout

**Date:** 30-10-2026
**Scripts:** `step_stopping_eval.py` + `step_stopping_{cf,gaucher,attr,fabry}.sb`
**Jobs:** 23332264 (CF) · 23332265 (Gaucher) · 23332266 (TTR) · 23332267 (Fabry) -- pending.

## Why this try exists
All 4 completed try09 runs reported `best_epoch = 0`, with validation loss rising
monotonically afterwards (TTR seed0: 2.37 → 3.10 → 3.48 → 3.56 → 3.77). But at that same
checkpoint training loss was **~1.0 against validation ~2.37** — the model had *already*
overfit inside epoch 1. So the true optimum is **earlier than the first checkpoint**, and
epoch-level evaluation is too coarse to find it.

## The five changes
1. **Validation every N steps, not per epoch.** N is set so there are ~7 evaluations per
   epoch, putting several measurement points inside epoch 1 where the minimum lives.
2. **Lower learning rate** — default **6.67e-5**, which is 1/3 of try09's 2e-4. This
   spreads the optimum over more steps so it is catchable rather than falling between two
   measurements.
3. **`N_EPOCH = 2`.** Not 1 — at least one measurement after the minimum is needed to show
   it *is* a minimum rather than just the last thing measured.
4. **Select the lowest validation loss, never the last.**
5. **The full validation curve is saved** to `step_stopping_summary.json` and printed, so
   the minimum is visible rather than asserted.

Implemented in the existing manual training loop rather than swapping to HF `Trainer`.
Same behaviour (`eval_strategy="steps"` / `load_best_model_at_end` / `metric_for_best_model
="eval_loss"`), without rewriting the training path — which keeps the only differences from
try09 the ones listed here.

**Validation set design.** Validation forwards cost the same as training forwards, so the
step-level evaluations use **one variant per validation compound** rather than a subsample
of lines. Measured augmentation is **5.00x** (fabry/gaucher/attr) and **4.92x** (CF), so
subsampling 400 *lines* would have covered only ~80 distinct molecules — and a curve
measured on 80 molecules is near-meaningless however cleanly it is measured. One variant per
compound maximises distinct molecules per unit of compute, and every target then fits under
the old budget with no subsampling at all:

| Target | distinct validation molecules in the curve |
|---|---|
| CF | 214 (100% of val compounds) |
| Gaucher | 181 |
| TTR | 72 |
| **Fabry** | **11** — the run warns explicitly |

The split was always *by compound*, so there was never post-augmentation leakage; the
line-level subsample would simply have collapsed coverage silently. The run prints the
distinct count and warns below 30.

(Incidentally, CF's 4.92x explains the 1,450-vs-1,427 count discrepancy flagged in try08:
48 CF compounds received fewer than 5 variants where random SMILES generation produced
duplicates.)

**Selection audit.** The loop keeps the **top two** checkpoints and runs full all-variants
validation on both. try09's epoch-level steps differed by ~0.7 in validation loss, which any
reasonable set resolves — but the curve *inside* epoch 1 may be far flatter, and that is
exactly where selection now happens. If full-val ranks the runner-up better, the step-level
curve mis-selected and the run says so (`SELECTION CHECK FAILED`), naming the distinct-molecule
count as the likely cause.

On disagreement the run still scores the **step-level-selected** checkpoint: selection policy
stays fixed and the check stays diagnostic, rather than silently becoming the selection rule
it was meant to audit. Both numbers go into the summary JSON.

## Kept unchanged from try09, so the runs stay comparable
- hard negatives held out of training **by scaffold** — not just the 3 molecules, since a
  near-copy sharing the skeleton would otherwise let the model recall the label anyway
- scaffold-split validation set, so held-out loss measures generalization to unseen
  frameworks rather than near-neighbour recall
- 500-decoy enrichment percentile as the category-level gate
- 3 seeds, reported as a distribution

## Both readouts, deliberately not switched
try10 established that greedy decoding returns the **modal** output rather than an estimate,
and that a digit-logit expectation fixes it (Gaucher's most-common value: 51% of novel
compounds → 5.5%). But the training change and the readout change must stay attributable, so
this run does **not** switch the readout — it scores every resulting adapter **both** ways
and reports them side by side.

Both come from **one `generate` call per molecule** (`output_scores=True`), so they derive
from the identical forward pass and cannot diverge for any reason except the readout.

## Also fixed: adapters are saved
try09 wrote summaries but never called `save_pretrained`, so its epoch-0 models were lost
and try10 had to run on the older over-trained weights. Every run here saves its adapter
*before* scoring.

## Scope and sizing
One target per job, `TARGET` explicit, so the try05 scope-coupling bug cannot recur.
4 targets × 3 seeds = **12 runs across 4 jobs**.

| Target | compounds | after scaffold holdout | ~train | ~val | job budget |
|---|---|---|---|---|---|
| CF | 1,450 | 1,424 | ~1,210 (6,050 lines) | ~214 | 10h |
| Gaucher | 1,208 | 1,202 | ~1,021 (5,105 lines) | ~181 | 10h |
| TTR (attr) | 493 | 476 | ~404 (2,020 lines) | ~72 | 8h |
| Fabry | 89 | 71 | ~60 (300 lines) | ~11 | 6h |

**Caveat on Fabry, recorded up front:** ~60 training and ~11 validation compounds is very
small. Its validation curve will be noisy and its selected step is weak evidence. Running it
for completeness, not because the result will be trustworthy on its own.

## How to read the result
The script prints the selected step per seed and warns on two specific failure modes:

| Outcome | Meaning |
|---|---|
| Selected step is mid-epoch-1, with a clear rise after | **The fix worked.** The optimum was found where epoch-level evaluation couldn't see it. These adapters are the first properly-stopped models in the project. |
| Selected step is 0 for every seed | Training never improved held-out loss at all. The adapters are effectively the base model — and that would mean fine-tuning on this data never helped, which is a major finding. |
| Selected step is ≤1 | The optimum is still at the very start. **Re-run at lr=2e-5** (1/10 of try09's) before trusting these adapters. |

The validation curve in the summary JSON is what distinguishes these, which is why it is
saved rather than summarized.

On the readouts: `greedy` vs `digit_exp` columns are reported per seed for enrichment
percentile and the hard-negative verdict. Since try10 showed greedy understates resolution,
any disagreement between the two columns is attributable to the readout alone — the training
is identical.

## Status
Submitted 30-10-2026: **23332264** (CF, 10h), **23332265** (Gaucher, 10h), **23332266**
(TTR, 8h), **23332267** (Fabry, 6h). 3 seeds each, `LR=6.67e-5`, `N_EPOCH=2`,
`EVALS_PER_EPOCH=7`. Results pending.

Pre-submission: script compiles; split sizes measured directly (see table above); leak
closure and scaffold-split logic are inherited **unchanged** from try09, where they were
verified as hard negatives 3/3 present before the holdout and 0/3 after, with zero train/val
scaffold overlap. Each run re-prints both checks in its own log, so they are confirmed per
run rather than assumed.
