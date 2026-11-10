# try12 — Fabry cross-validation: was the inconclusiveness the checking set?

**Job 23400706, 3h24m, COMPLETED 09-11-2026.**

## Why this ran

try11 gave Fabry a wide, uninterpretable spread (digit_exp enrichment 20.0 / 58.4 / 27.2,
spread 38.4). The suspected reason was the measuring instrument, not the model: Fabry has
89 compounds, 71 after the scaffold holdout, so a 15% validation split left **11
molecules**. Gaucher's curve was measured on 180–208. A checking set that small cannot
resolve a validation curve, so the selected stopping step was close to arbitrary and the
three seeds effectively stopped at three different places.

Fix: **6-fold scaffold cross-validation**, so all 71 compounds serve as a validation
molecule exactly once instead of 11 serving and 60 being wasted.

- **Phase 1** — 6 folds, each training on ~59 compounds and validating on ~12. The
  stopping point is averaged across folds, so it is chosen on all 71 compounds rather
  than one small slice.
- **Phase 2** — 3 seeds train on **all 71** compounds, stopping at the CV-chosen point.
  The optimum transfers as a **fraction of an epoch**, not a raw step count, because the
  full set has more steps per epoch than a fold does.

## What happened

The cross-validation worked exactly as designed. The averaged curve has a clean interior
minimum, not an edge:

| epochs | mean val loss |
|---|---|
| 0.00 | 14.995 |
| 0.14 | 6.678 |
| 0.28 | 2.162 |
| **0.42** | **1.956  ← minimum** |
| 0.56 | 2.072 |
| 0.69 | 2.320 |
| 0.83 | 2.652 |
| 1.11 | 3.148 |
| 1.39 | 3.481 |

Chosen stopping point: **0.420 epochs = step 18 of 44/epoch** on the full 71 compounds.

Phase 2, 3 seeds at that stopping point:

| seed | greedy %ile | digit_exp %ile | beats_hard (digit) |
|---|---|---|---|
| 0 | 76.0 | 44.0 | False |
| 1 | 45.0 | 48.6 | False |
| 2 | 28.4 | 48.0 | True |

**digit_exp spread collapsed from 38.4 to 4.6.** The instrument hypothesis was correct —
the inconclusiveness *was* the checking set.

## The result, stated plainly

**The measurement is now trustworthy, and what it reliably says is "no signal."**

44.0 / 48.6 / 48.0 is chance. Fabry is not unstable; it is consistently at the coin-flip
line. Fixing the checking set converted an uninterpretable result into an interpretable
negative one, which is the point of doing it — but it did not produce a passing target.

Two further things this run nails down:

1. **The diastereomer collapse survived every fix.** Greedy still returns one identical
   value for migalastat and all three held-out diastereomers (seed 0: 672/672/672/672).
   digit_exp separates them by only ~10–20 points on a 1,000-point scale, with real labels
   spanning 196 to 993. Stereochemistry is still unresolved after step-level stopping,
   scaffold holdout, cross-validated stopping, and the corrected readout.

2. **Fine-tuning makes Fabry worse, and this is now measurable.** try13 scored migalastat
   on the untouched base model at **15.2 percentile** zero-shot. Fine-tuning moves it to
   44–49 here, and to 83 in try11. Every fine-tuned Fabry result is worse than no
   fine-tuning at all. See `../try13_01-11-2026_zero-shot-familiarity/README.md` — that
   run also shows the stereo-collapse is present **in the base model before any
   training** (zero-shot greedy: 331 for migalastat and all three diastereomers).

## Correction to the recorded root cause

`CLAUDE.md` records Fabry's root cause as "only 89 real training compounds (13.5x less
than Gaucher)." **That is not the cause.** The collapse exists in the base model with zero
training compounds involved, and try07 already showed Gaucher passes at 493. Data volume
is not what separates these targets. The real separator is identified in try13.
