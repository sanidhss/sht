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

## Result
Jobs **23332265** (Gaucher), **23332266** (TTR), **23332267** (Fabry) COMPLETED.
**23332264** (CF) **TIMED OUT** at 10h with 2 of 3 seeds done — resubmitted as **23400943**
with an 18h budget (measured cost ~4.5h/seed, so ~13.5h plus margin).

### The epoch fix worked
Selected steps are **mid-first-epoch on every run**, never step 0 and never the last — the
optimum was exactly where epoch-level evaluation could not see it:

    gaucher  steps 91 / 88 / 91   (of ~640 per epoch)   val 1.57 / 1.64 / 1.46
    attr     steps 105 / 33 / 34  (of ~250)             val 1.69 / 1.64 / 1.67
    fabry    steps 15 / 20 / 20   (of ~37)              val 2.59 / 1.78 / 1.98
    cf       step  106            (of ~740)             val 1.66

Validation loss fell from try09's 2.36–2.97 to **1.46–1.69**. Over-training was real and
step-level stopping fixes it.

### Gaucher: the first clean positive result in the project
| seed | greedy %ile | digit-exp %ile | beats held-out analogs |
|---|---|---|---|
| 0 | 0.6 | **0.2** | ✅ |
| 1 | 29.4 | **0.2** | ✅ |
| 2 | 1.2 | **0.8** | ✅ |

**3/3 seeds pass**, enrichment spread **0.6 points**. Ambroxol ranks ~1st of 500 decoys.
Leak-free, properly stopped, calibrated scoring, 3 seeds, 180–208 validation molecules, and
the selection audit agreed with the full validation set on every seed.

Mechanism, from the score distribution (seed 0): ambroxol scores **478** against a decoy
median of **691** — clearly separated, 1 of 500 beats it.

**And it is not near-neighbour recall.** Of the four targets, ambroxol has the *least*
structural support in its own training data:

| drug | best Tanimoto to anything left in training | compounds sharing its scaffold | result |
|---|---|---|---|
| **ambroxol** | **0.173** | **0 of 1,202** | **ranked ~1st** |
| tafamidis | 0.340 | 0 of 476 | failed |
| lumacaftor | 0.512 | — | pending |
| migalastat | 0.857 | — | failed |

Nothing in training is structurally close to ambroxol and nothing shares its scaffold, yet
the model ranks it first in all three seeds. That rules out "it recognised a near-copy."

### digit-logit scoring confirmed as the better readout
Gaucher greedy: 0.6 / 29.4 / 1.2 (spread 28.8). digit-exp: 0.2 / 0.2 / 0.8 (spread 0.6).
Same models, same forward passes — only the readout differs. greedy also passed only 2/3 on
the analog test where digit-exp passed 3/3.

### TTR: a real negative, and the cause is NOT yet established
    tafamidis scored 815.7   |   decoy median 815   |   254 of 500 decoys beat it
    0/3 seeds pass the analog test, under either readout

The model assigns tafamidis the same score as a median random molecule — no signal.

But the model is **not** broken on TTR. On the three analogs, held out by scaffold, it
predicts almost exactly right:

    analog 1: predicted 784  (real 781)
    analog 2: predicted 766  (real 759)
    analog 3: predicted 790  (real 774)

Within 3–16 points on held-out compounds. It then misses tafamidis by **264** (816 vs real
552). So TTR chemistry was learned; tafamidis specifically is not predictable from it.

**Four candidate causes have been tested and eliminated:**

| Hypothesis | Verdict |
|---|---|
| Label range — model can't output values as low as 552 | ✗ TTR labels span 0–1000, **33.7%** of training labels are below 552 |
| Its scaffold is absent from training | ✗ true, but **ambroxol's is also absent** (0 of 1,202) and Gaucher works |
| Its nearest neighbours were removed by the holdout | ✗ true (5 nearest all removed, best remaining 0.340), but ambroxol has **less** support at 0.173 and works |
| Too little data (493 vs Gaucher's 1,208) | ✗ try07 already showed Gaucher passes 2/3 when cut to 493 |

**Leading untested hypothesis:** base-model prior familiarity. Ambroxol is a ubiquitous
over-the-counter mucolytic with decades of literature; tafamidis is a rare orphan drug
approved in 2011 with far less text behind it. TxGemma may carry useful prior knowledge about
one and not the other, which would explain why ambroxol succeeds with *less* structural
support than tafamidis.

**The decisive test** is cheap and needs no training: score all four reference drugs plus the
decoys with the **base** model, zero-shot. If ambroxol already separates from the decoy field
before any fine-tuning and tafamidis does not, prior familiarity is the cause. Not yet run —
recorded here as the open question rather than asserted as the answer.

### Fabry: inconclusive, as flagged in advance
Enrichment 20.0 / 58.4 / 27.2 (spread 38.4), 1/3 passed. Its checking set was **10–11
molecules** against Gaucher's 180–208 — a fixed 15% fraction is useless at 71 compounds, and
Fabry additionally loses 20.2% to the scaffold holdout versus Gaucher's 0.5%. Addressed in
try12 by 6-fold cross-validation so all 71 compounds inform the stopping decision.

For the record, Fabry has never produced a positive result: try04 failed, try05 failed, try08's
apparent top-15.5% was shown in try10 to be a greedy-decoding artifact (83rd percentile under
honest scoring), try11 inconclusive.

### CF: partial (2 of 3 seeds before timeout)
    seed0: greedy  8.6 %ile | digit-exp 18.6 %ile | beats analogs: digit ✅ greedy ✗
    seed1: greedy  1.6 %ile | digit-exp 42.4 %ile | beats analogs: digit ✅ greedy ✗

Passes the analog test under digit-exp on both seeds, but enrichment is far weaker than
Gaucher's and the two seeds disagree (18.6 vs 42.4). Resubmitted for all 3 seeds.

### The selection audit fired once, as designed
`attr_seed0` tripped `SELECTION CHECK FAILED` — full all-variants validation ranked the
runner-up checkpoint better than the selected one, on a 77-molecule checking set. The check
caught it rather than hiding it, which is what it was added for.

## Status
Gaucher, TTR, Fabry complete. CF resubmitted as **23400943** (18h). Headline: step-level
stopping works, Gaucher is a clean 3/3 positive that is demonstrably not near-neighbour
recall, TTR is a real negative whose cause is narrowed to one untested hypothesis, Fabry
needs try12's cross-validation.
