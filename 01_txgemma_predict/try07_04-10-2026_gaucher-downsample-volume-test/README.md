# Try 07 — Gaucher downsample: is the Fabry/TTR failure about data volume?

**Date:** 04-10-2026
**Scripts:** `downsample_volume_test.py` + `downsample_volume_test.sb`
**Job:** 22660248 (submitted 04-10-2026) -- result pending; see Status below.

## Why this try exists
try06 produced a split result: CF passes, TTR fails. Lining all four targets up by
number of real training compounds, hard-negative pass/fail is perfectly monotonic:

    CF      1,427  PASS
    Gaucher 1,208  PASS
    TTR       493  FAIL
    Fabry      89  FAIL

Clean break between ~1,200 and ~500. That is a much simpler explanation than anything
mechanism-specific, and it lines up with Fabry's already-verified root cause (89
compounds, read directly from `coverage_report.json`).

But it is **confounded and n=4**. A better-studied target has both more data and
possibly cleaner data, and the four targets differ in disease mechanism. The pattern
could be volume, data quality, mechanism, or coincidence across four points.

This try breaks the confound. It takes **Gaucher — a target that passes at full size —
and retrains it on a random 493-compound subset**, matching TTR's compound count
exactly. Same target, same mechanism, same pipeline, same held-out reference
(ambroxol), same three real structural analogs. Training volume is the only variable.

## Why this is the right next experiment
It was chosen over the two alternatives on the table:

- **Over scaling to 27B.** The failures are not capacity-shaped. Fabry (89) and TTR
  (493) fail while CF (1,427) and Gaucher (1,208) pass *on the same 9B model*. TTR's
  specific failure is reciting a near-neighbour's exact training label — a sparsity
  signature, and a higher-capacity model is if anything *more* prone to memorizing a
  small pool, not less. Scaling the model does not create bioactivity data for a
  target that only has 493 measured compounds. (Separately, 27B in bf16 is ~54GB of
  weights against this run's 42GB card, and try06 already hit a recovered OOM at 9B —
  so it would need multi-GPU sharding or 4-bit quant, not a config edit.) Staying on
  9B.
- **Over building `NEXT_PHASE_diverse_structural_training.md`.** That phase was scoped
  for the premise "the general pipeline needs more than chemistry alone." CF's pass is
  evidence against that premise on the mechanism-breadth axis, so the doc is pending
  re-justification. Spending a build on it before testing the cheaper data-sufficiency
  explanation would be backwards.

This run costs one job on hardware already known to work, and reuses already-prepped
data.

## What it does
Three seeds (0, 1, 2) at 493 compounds each. Per seed: fresh base-model load, fresh
LoRA adapter, 3 epochs, then the standard easy/hard-negative validation against the
**unchanged** full-size Gaucher eval set.

Design decisions, and why:

- **No re-prep, no network.** Reads Gaucher's existing `train_blind.jsonl`
  (1,208 compounds × 5 SMILES variants = 6,040 lines). Re-running
  `prep_orphan_target_data.py` would re-hit ChEMBL/BindingDB/PubChem and risk the
  underlying data shifting under the comparison, defeating the point.
- **Downsamples at the distinct-compound level, not the line level.** Cutting 6,040
  lines to 2,465 directly would reduce augmentation depth *and* compound diversity at
  once, confounding the two. Augmented variants are regrouped by RDKit canonical
  SMILES — verified before building: recovers **exactly 1,208 groups of exactly 5, one
  consistent label each, 0 unparseable** — then whole compounds are sampled with all 5
  variants kept. Result: 493 × 5 = 2,465 lines, structurally identical to TTR's
  493 → 2,463.
- **Three seeds, not one.** A single 493-draw could be unlucky or lucky in which
  compounds it retains. A 1-of-3 or 2-of-3 split is itself informative — it would say
  493 sits *near* the threshold rather than clearly below it.
- **Hyperparameters byte-identical to try06's** `train_adapter` (r=16, alpha=32,
  lr=2e-4, 3 epochs, grad_accum=8), deliberately unchanged so any difference in
  outcome is attributable to the data and not the optimizer.
- **Writes to `gaucher_downsample/`, never `gaucher/`.** try05's passing adapter and
  eval set are the baseline this is compared against and must not be overwritten.
- **Summary re-written after every seed**, so a timeout or OOM on seed 2 still leaves
  seeds 0 and 1 on disk. try05 lost a whole run's output by only writing at the end.
- **Tracks `collapsed` separately from `beats_hard`.** "Collapse" is the specific
  TTR/Fabry signature — one identical score for the reference *and* every analog, i.e.
  the model has stopped discriminating at all. That is a different and worse failure
  than merely ranking the reference second, and conflating them would blur exactly the
  thing this run is measuring.

## Known confound in this run, stated up front
493 compounds at 5× augmentation is ~2,465 lines, therefore **~2.4× fewer gradient
steps per epoch** than full-size Gaucher. So a collapse here is consistent with
*either* "too few distinct compounds" *or* "too little training compute" — this run
cannot separate them.

That is a deliberate choice, not an oversight: matching TTR's configuration exactly is
what makes this the right test of the TTR question specifically. Separating diversity
from compute needs a matched-compute arm (493 compounds at ~12× augmentation to restore
~6,040 lines). That is the follow-up **if** this run collapses, not part of it.

## How to read the result
| Outcome | Meaning |
|---|---|
| **3/3 collapse** | Volume is the bottleneck. Fabry + TTR become *one* data-sufficiency finding rather than two mechanism-specific failures — a stronger, more generalizable thesis claim than either alone. Next: the matched-compute arm, to separate diversity from step count. |
| **0/3 collapse** | Volume is **not** a sufficient explanation. The monotonic n_training pattern across the 4 targets is then confounded or coincidental, and the Fabry/TTR failures need a mechanism- or data-quality-based account. This would redirect the next phase, so it is just as informative as a collapse. |
| **1–2/3 collapse** | 493 sits near the threshold rather than clearly below it. Suggests a volume sweep (e.g. 250 / 493 / 800) to locate the break. |

Baseline to compare against — full-size Gaucher, try05: ambroxol **predicted 691**
(real label 685), analogs **predicted 791 / 791 / 839** (real labels 791 / 679 / 839)
→ PASS. The script prints this side-by-side with each seed so the comparison does not
depend on reading two files.

**Correction worth carrying forward:** the top-level README and session briefing both
cite Gaucher's analogs as "791/791/839" in a way that reads as real labels. Those are
*predicted* scores; the real labels are 791/679/839, confirmed by reading
`gaucher/eval_set.json` directly. This matters, because **analog #2's real label (679)
is slightly more potent than ambroxol's (685)**. So "reference beats all three hard
negatives" is not strictly ground truth for Gaucher — on that one pair the full-size
model ranks *against* the real labels and is scored as passing for it.

That does not weaken this run (identical eval set on both sides, and the question is
whether discrimination survives downsampling, not whether the ranking is right), but
it does mean **Gaucher's "PASS" is a softer claim than it has been treated as** — and
Gaucher is the anchor of the whole volume hypothesis. Flagging rather than silently
fixing, since re-examining what the hard-negative test should count as success is a
judgement call, not a correction.

## Status
Submitted as job **22660248** (12h budget, gpuA100x4, 1 GPU). Result pending.

Pre-submission checks that did pass: RDKit regrouping recovers exactly 1,208 groups of
exactly 5 with one consistent label each and 0 unparseable lines; Gaucher's
`eval_set.json` schema is identical to the one try06's validation consumed (same keys,
5 easy / 3 hard negatives), so the reused eval path will not KeyError after a 35-minute
model load.
