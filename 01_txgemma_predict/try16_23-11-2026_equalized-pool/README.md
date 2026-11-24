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

## Result

Pending — GPU array queued. Fill in from
`/work/hdd/bisa/sshah21/results/txgemma_lora_orphan/try16_equalized_pool/SUMMARY.md`.

The bar: the adapter arm must beat the Tanimoto rank in the table above, on the same
1,933 compounds. Note that mastocytosis's baseline is rank **11 of 1,933** — the model
has to place midostaurin in the top 11 approved drugs to beat a fingerprint lookup there.
