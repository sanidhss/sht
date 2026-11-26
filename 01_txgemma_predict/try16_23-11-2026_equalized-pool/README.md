# try16 — score the frozen equalized pool with saved adapters

**Jobs 24440722** (GPU array 0-5, 8h) **→ 24440723** (CPU collate, afterany).
Inference only: no training, no new fine-tunes, nothing refitted.

## What changes versus every previous try

Every enrichment number in this project so far was measured against **500 MaxMin-diverse
decoys drawn from the other targets' training compounds**. try15 showed that yardstick is
unstable — widening the pool from 4 to 8 source targets moved CF by 31.4 percentile
points without changing the model, the drug or the prompt.

This try replaces it with a **frozen, externally-defined candidate set**: `pool_v1.json`,
1,933 approved drugs with 1:1 ChEMBL↔DrugBank mappings, built and sha256-locked by
`06_equalized_pool/try01`. Every arm of the cross-architecture comparison scores this
exact file and verifies its hash first. The question becomes the deployment question:
**how deep a shortlist of real approved drugs must you carry before the right one is in
it?**

## STEP 0 — pool verification

```
expected 9d89c3ce758b7ec7c8bea2866f6448a9ea71b372f0f3c3f92c8c6b690653e308
actual   9d89c3ce758b7ec7c8bea2866f6448a9ea71b372f0f3c3f92c8c6b690653e308  MATCH
1933 records -> 1933 unique canonical structures (no duplicates)
```

`verify_pool()` exits 2 on mismatch. It runs at the start of **both** the scoring script
and the baseline script, so neither can score a silently-rebuilt pool.

## Two targets cannot be evaluated, and nothing was substituted

Checked by **canonical structure**, not by name:

| target | reference | in pool_v1? |
|---|---|---|
| cf | lumacaftor | yes — DB09280 |
| fabry | migalastat | yes — DB05018 |
| attr | tafamidis | yes — DB11644 |
| cel | imatinib | yes — DB00619 |
| mastocytosis | midostaurin | yes — DB06595 |
| tyrosinemia | nitisinone | yes — DB00348 |
| **gaucher** | **ambroxol** | **ABSENT** |
| **nsiad** | **tolvaptan** | **ABSENT** |

This matches `06_equalized_pool/try01/POOL_REPORT.md`, which reached the same conclusion
from the mapping side: ambroxol maps to CHEMBL153479 and tolvaptan to CHEMBL4303142,
neither of which is in the 3,417-drug source library.

**No salt or analogue was spiked in**, for two reasons. The premise of an equalized pool
is that every arm scores the *identical* file, and spiking breaks the sha256 contract.
And tolvaptan's UniChem partner CHEMBL4303142 is **the same ChEMBL record try14 used as
nsiad's first hard negative** — a stereo-specified variant measured at 1200 nM against
tolvaptan's 3.4 nM. Substituting it would have silently swapped a potent drug for a
350x-weaker stereoisomer and called the result tolvaptan's rank.

So the GPU array runs **6 tasks, not 8**. Gaucher and nsiad are reported as not
evaluable, with their reason, in `results_pool_v1.csv`.

## STEP 2 result — Tanimoto 5-NN on the pool (already run, CPU)

Same estimator as `05_multi_agent/prefilter_recall.py`, only the candidate set changed.

| target | drug | rank | of 1,933 | percentile | max sim to actives | |
|---|---|---|---|---|---|---|
| mastocytosis | midostaurin | **11** | 1933 | **0.57%** | 0.692 | |
| cel | imatinib | 344 | 1933 | 17.80% | 0.942 | near-trivial |
| tyrosinemia | nitisinone | 356 | 1933 | 18.42% | 0.633 | |
| attr | tafamidis | 1108 | 1933 | 57.32% | 0.435 | |
| cf | lumacaftor | 1373 | 1933 | 71.03% | 0.575 | |
| fabry | migalastat | 1659 | 1933 | 85.83% | **1.000** | **TRIVIAL** |
| gaucher | ambroxol | — | — | not in pool | 0.192 | clean |
| nsiad | tolvaptan | — | — | not in pool | **1.000** | **TRIVIAL** |

**The leak control fired on exactly the two targets it was built to catch.** fabry and
nsiad both have a training active at Tanimoto 1.000 to the reference. Morgan-2 is
stereo-blind, so migalastat's diastereomers and tolvaptan's enantiomer fingerprint
identically to the drug itself. Their 5-NN prediction is substantially a readback of
their own near-duplicates' labels, and neither row may be reported as a result.

Note the direction: fabry's leak made its baseline rank **worse** (85.83%), not better,
because the identical-fingerprint neighbours are the diastereomers with labels of
925/754/993 — far weaker than migalastat's 196. A leak is not automatically a flattering
leak.

cel at 0.942 sits just under the threshold and should be read with suspicion.

**The decoy field was systematically flattering the baseline, and not uniformly:**

| target | baseline on 500 decoys | baseline on 1,933 real drugs | shift |
|---|---|---|---|
| mastocytosis | 20.8% | **0.57%** | far better |
| cel | 5.8% | 17.80% | worse |
| tyrosinemia | 26.8% | 18.42% | better |
| attr | 31.0% | 57.32% | much worse |
| cf | 29.4% | 71.03% | much worse |
| fabry | 26.4% | 85.83% | far worse (leak) |

Shifts run in both directions and reach 56 points. This is the same lesson as try15's
decoy-pool sensitivity, now with an externally-fixed pool as the reference point: **a
percentile is a statement about a candidate set, and the candidate set has to be stated.**
The 500-decoy numbers should not appear in the thesis without the pool named next to them.

## Why `beats_hard` is absent from this try

It is not reported in `results_pool_v1.csv`, in `SUMMARY.md`, or anywhere else here.

try14 established that hard-negative **selection** is broken. The picker takes the top-3
Tanimoto neighbours with no identity or potency check, so on 3 of the 4 new targets it
selected the reference drug itself in disguise — imatinib mesylate for cel, both
nitisinone enol tautomers for tyrosinemia, staurosporine twice for mastocytosis. And on
mastocytosis all three negatives are genuinely *more* potent than midostaurin, so a
correct model must fail that test.

Reporting a flag that is simultaneously unachievable-by-correctness on one target and
asks the model to outrank itself on two others, beside an honest rank in a real drug
pool, would actively mislead. The pool rank is the number that means something. The
hard-negative test needs the prep-side fix recorded in try14's README before it is used
again.

## Design notes

- **Adapters are loaded, never refitted.** If an adapter directory is missing the script
  exits 3 and says so. 27 adapters found: 3 per target for all 8 step-stopping runs, plus
  fabry's 3 cross-validation folds. 21 are scored (gaucher's and nsiad's are skipped as
  unrankable). fabry's array task therefore does twice the work of the others.
- **`score_both` is copied verbatim** from try11/try15 — greedy and digit-logit
  expectation from one forward pass, digit width taken from the greedy decode rather than
  fixed at 3 (the try10 bug, which misread 1000 by ~900). Copied rather than imported so
  a later edit elsewhere cannot change what this run meant.
- **The adapter is asserted to have attached.** `n_lora_layers == 0` exits 5. A silently
  unapplied adapter would turn this into a second copy of try15's zero-shot run and
  nothing in the output would reveal it.
- **`base = model.unload()` between seeds**, so adapter weights cannot accumulate across
  the three seeds sharing one base-model load.
- **Ranks are best-case and ties are reported.** `rank = 1 + #{strictly better}`, plus
  `n_tied_with_ref`. Greedy collapses hard — try14 measured 54 distinct greedy values
  across 500 decoys — so breaking every tie in the reference's favour flatters greedy
  enormously. A greedy rank with hundreds of ties is not a rank, and the column makes
  that visible instead of hiding it.
- **`TARGET` is exported per array task** and the script exits 2 if unset; the try05
  scope-coupling bug came from an implicit default.
- **Collate runs under `afterany`** so a dead array task yields a report naming what is
  missing rather than no report.

## Result (complete, 25-11-2026)

All six evaluable targets finished. Tasks 2-5 ran after the time limit was cut 8h -> 2h
in place, taking 50-61 min each -- within the "strictly less work than cf's 62 min"
prediction.

digit_exp readout, mean of 3 seeds, rank among the 1,933-drug pool. Lower is better.

| target | drug | path | model rank | model %ile | seed spread | kNN rank | kNN %ile | winner |
|---|---|---|---|---|---|---|---|---|
| cel | imatinib | stepstop | **199** | **10.28** | 4.7 | 344 | 17.80 | **model** |
| fabry | migalastat | stepstop | 524 | 27.09 | 29.5 | 1659 | 85.83 | model* |
| cf | lumacaftor | stepstop | 752 | 38.89 | 43.8 | 1373 | 71.03 | **model** |
| fabry | migalastat | crossval | 783 | 40.49 | 4.4 | 1659 | 85.83 | model* |
| tyrosinemia | nitisinone | stepstop | 943 | 48.77 | 52.2 | 356 | 18.42 | kNN |
| mastocytosis | midostaurin | stepstop | 1224 | 63.34 | 26.6 | **11** | **0.57** | kNN |
| attr | tafamidis | stepstop | 1326 | 68.62 | 12.8 | 1108 | 57.32 | kNN |

`*` fabry's kNN arm is leak-damaged (Tanimoto 1.000 to its own diastereomers), so that
comparison is not a fair one -- see below.

**On a real drug pool the model does far better than the decoy field suggested.** The
500-decoy scoring had it at 1 win, 1 tie, 6 losses against the same baseline. Here it is
roughly even. This is the clearest vindication of building the equalized pool: the decoy
field was not a neutral yardstick.

### Rank alone is misleading. The label errors settle it.

| target | drug | real | nM | model predicts | abs error | verdict |
|---|---|---|---|---|---|---|
| cf | lumacaftor | 584 | 1293 | 556 | **28** | genuinely accurate |
| cel | imatinib | 441 | 84.6 | 525 | **84** | good |
| mastocytosis | midostaurin | 410 | 57.2 | 597 | 187 | poor |
| tyrosinemia | nitisinone | 234 | 37.0 | 453 | 219 | poor |
| attr | tafamidis | 552 | 838.7 | 787 | 235 | poor |
| fabry | migalastat | 196 | 40.4 | 560 | **364** | worst |

This reclassifies one result. **fabry's rank "win" is an artifact.** The model predicts
560 against a true 196 -- an error of 364, its worst anywhere. It outranks the kNN only
because the kNN is even more wrong (predicts 796, error 600) in the same direction. Two
badly wrong methods, one less wrong. Nothing was learned about migalastat.

cf and cel survive the check: best label accuracy (28 and 84) *and* better rank. Those
are the two real wins.

Revised tally: **2 genuine wins (cf, cel), 1 artifact (fabry), 3 clear losses.**

### The model compresses everything into one band — now quantified

Model predictions across the six: 453, 525, 556, 560, 597, 787 — mean 580, range 453-787.
True labels: 196, 234, 410, 441, 552, 584 — mean 403, range 196-584.

**It over-predicts the score (under-predicts potency) on 5 of 6 targets**, by +84 to
+364, and the only under-prediction is cf at -28. The true labels span 388 points; the
predictions span 334 but sit almost entirely above 450.

This is try15's "converges to a band regardless of where it started" finding, now
measured directly on predictions rather than inferred from ranks. The model has largely
learned the task's output distribution, not each target's chemistry -- which is exactly
why it does well on cf (true label 584, near the band) and worst on fabry (true label
196, far below it).

### Seed spreads make most of these differences non-significant

Spreads: tyrosinemia 52.2, cf 43.8, fabry stepstop 29.5, mastocytosis 26.6, attr 12.8,
cel 4.7, fabry crossval 4.4 percentile points.

With 3 seeds and spreads of 40-50 points, cf's and tyrosinemia's results are barely
distinguishable from noise. Only **cel** (10.28, spread 4.7) and **mastocytosis**
(63.34 against a kNN at 0.57) are large enough relative to their spread to assert.
Treat cf's win as suggestive, not established.

### mastocytosis is the clearest failure in the project

A Tanimoto 5-NN puts midostaurin **11th of 1,933** approved drugs. The fine-tuned 9B
model puts it **1,224th**. try15 measured mastocytosis zero-shot at 14.6 percentile, so
fine-tuning took a target the base model handled reasonably and destroyed it -- the same
pattern as Fabry, and the strongest single argument that LoRA on this data actively
damages good priors.

## Result — PARTIAL (2 of 6 targets, as of 25-11-2026)

Array tasks 0 (`cf`) and 1 (`fabry`) COMPLETED 25-11 at 08:03 and 08:33, after ~37h
queued. Tasks 2-5 (`attr`, `cel`, `mastocytosis`, `tyrosinemia`) are still PENDING on
priority, so `24440723` (collate) has not run and
`try16_equalized_pool/SUMMARY.md` still reports all eight as "array task did not
finish" — that file is the 23-11 dry-run collate, **not** the current state. The numbers
below are read directly from the per-seed rows in
`try17_unblinded/results_unblinded_pool_v1.csv`, whose `model_blinded` arm is this try's
output (try17's collate joins it).

The bar: the adapter arm must beat the Tanimoto rank in the table above, on the same
1,933 compounds. Note that mastocytosis's baseline is rank **11 of 1,933** — the model
has to place midostaurin in the top 11 approved drugs to beat a fingerprint lookup there.

### digit_exp, per seed, rank among 1,933 (lower is better; chance 49.90%)

| target | path | seed 0 | seed 1 | seed 2 | mean | spread | Tanimoto | chance |
|---|---|---|---|---|---|---|---|---|
| cf | stepstop | 14.54% | 43.77% | 58.36% | **38.89%** | **43.8** | 71.03% | 49.90% |
| fabry | stepstop | 15.21% | 44.70% | 21.37% | 27.09% | 29.5 | 85.83% (TRIVIAL) | 49.90% |
| fabry | crossval | 40.35% | 38.33% | 42.78% | 40.49% | **4.4** | 85.83% (TRIVIAL) | 49.90% |

fabry's headline 33.79% is the mean over all six adapters and should not be quoted as a
point estimate; the two paths disagree and only one of them is trustworthy (below).

### Finding 1 — both targets beat their Tanimoto baseline on every seed, but only one comparison is admissible

cf: all 3 seeds beat 71.03%, the worst by 12.7 points. That is a genuine win for the
adapter arm over a fingerprint lookup on a real approved-drug field — and it **inverts
the 1-win/1-tie/6-loss verdict** sections 9/10 recorded against the 500-decoy baseline.
The decoy field was flattering the baseline on cf by 41.6 points (29.4% → 71.03%),
which try16's own STEP 2 table already measured before any adapter ran.

fabry's baseline is `trivial_flag = True` (a training active at Tanimoto 1.000), so
85.83% is not a result and beating it is not evidence. **fabry has no admissible
baseline on this pool.** Against chance it is the weaker claim that still holds: all six
adapters land between 15.2% and 44.7%, i.e. better than 49.90% in every case.

### Finding 2 — cf's seed spread is 43.8 points and that is the real problem

14.54 / 43.77 / 58.36. The spread is larger than the 32.1-point distance from the mean
to the Tanimoto baseline, so "cf beats the baseline" survives on the worst seed but "cf
scores 38.89%" is not a statement about anything. This reproduces, on an externally
fixed pool, exactly what try11 recorded: **CF's step-selection check failed on 2 of 3
seeds** because its 228-254 distinct validation molecules cannot resolve its loss curve.
The instrument, not the pool, is the limit.

### Finding 3 — cross-validation stabilises fabry again, and the two paths disagree about where it lands

fabry's crossval spread is **4.4 points against stepstop's 29.5** — a 6.7x reduction,
reproducing try12's result (spread 38.4 → 4.6) on a different candidate field. The
mechanism is the one try17's dry-run then quantified: fabry's scaffold holdout removes
18 of 90 compounds (20%), so a single scaffold split is a high-variance draw and
6-fold CV averages over it.

They do not agree on the answer. crossval says 40.49% (just better than chance);
stepstop says 27.09% but with a 29.5-point spread that contains it. **Report crossval.**
Taking stepstop's lower mean would be selecting the noisier instrument because it
flatters the result, and try12 already established CV as the trustworthy measurement for
this target.

### Finding 4 — greedy's rank is not a rank, now demonstrated on a real drug field

The `n_tied_with_ref` column does what it was added for. cf seed 2: **508 of 1,933
approved drugs tie with lumacaftor** under greedy decoding; fabry stepstop seed 2: 375
ties. digit_exp ties with exactly 1 (itself) in all 12 rows. Greedy's ranks are also
wildly inconsistent with digit_exp's — cf seed 0 reads 52.98% greedy vs 14.54%
digit_exp, fabry stepstop seed 1 reads 12.73% vs 44.70%, in opposite directions. Since
greedy ranks are reported best-case, every tie broken in the reference's favour, these
are upper bounds on an already meaningless number. **digit_exp is the only readout from
this try that may be quoted.**

### Finding 5 — `abs_label_error` says cf is accurate and fabry is not, which the ranks hide

Rank is confounded with absolute potency (try17 Finding 3), and these two targets sit on
opposite sides of that confound:

| target | real label | predicted (digit_exp, per seed) | abs error |
|---|---|---|---|
| cf / lumacaftor | 584 | 540.6 / 558.8 / 567.8 | **16-43** |
| fabry / migalastat | 196 | 524.4-604.6 | **328-409** |

cf's prediction is within 3-7% of the measured value on all three seeds. fabry's is off
by a factor of ~3 on all six adapters, in the same direction every time — the model
places migalastat at ~560-600 when it measures 196. So fabry's mid-field rank is not a
near-miss on a weak binder; it is a confident wrong answer about a **potent** one
(40.4 nM, the most potent reference in the project). The rank understates how wrong it
is, because 196 is low enough that even a 400-point error still leaves it mid-pack.

### What is still missing

`attr`, `cel`, `mastocytosis` and `tyrosinemia`. Three of the four carry the baselines
the comparison most needs — mastocytosis at 0.57%, cel at 17.80%, tyrosinemia at
18.42% — so the adapter arm's hardest tests are all in the pending half. **No
cross-architecture claim should be written off cf and fabry alone**, and fabry's half of
that pair has no admissible baseline.
