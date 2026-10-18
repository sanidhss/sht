# Try 09 — Leak-free evaluation harness

**Date:** 18-10-2026
**Scripts:** `leak_free_eval.py` + `leak_free_eval_cf.sb` + `leak_free_eval_attr.sb`
**Jobs:** see Status below.

**Goal: make the test trustworthy.** Nothing here tries to make the model better. It
rebuilds the measurement so that a future "CF passes" can be believed.

## Why this try exists
try08 found that **all 12 hard negatives, across all four targets, were in their
training sets.** Only the reference drug was held out. So the hard-negative check
compared a *generalization* task (predict a held-out drug) against a *memorization*
task (recall three trained analogs) — not what the script claims it measures ("the test
that actually indicates the model learned the right structure-activity signal").

That one flaw accounts for most of this project's confusing history:

- The **"collapse" signature** behind the Fabry and TTR diagnoses — one identical value
  for the reference and all three analogs — is the model applying a memorized
  neighbour's label to the whole structural family. Gaucher emits **791 for 51% of 200
  diverse novel compounds**, and 791 is the real label of one of its own hard negatives.
- **Fabry's diastereomer/stereochemistry diagnosis, carried for weeks, is most likely an
  artifact of this** rather than a finding about chiral centers.

try07 independently showed single-run verdicts are noise-dominated: same target, same
data size, same hyperparameters, varying only which 493 compounds were drawn, the
held-out reference scored **785, 408, 480** (real 685; full-size 691). A 377-point swing,
with seeds on opposite sides of pass/fail.

## What it changes
Five changes, each necessary, none sufficient alone.

**1. Hard negatives held out of training — by scaffold.**
Not just the three molecules. Every training compound sharing their Bemis–Murcko
scaffold, because otherwise a near-copy (same skeleton, one extra substituent) stays in
training and the model can still effectively recall the label. Measured cost: **CF 1.8%
of compounds, TTR 3.4%, Gaucher 0.5%, Fabry 20.2%.**

**2. Scaffold split for validation, not a random split.**
A random split leaves close relatives of held-out compounds in training, so "good
validation performance" is still recall. Splitting by scaffold puts *unseen skeletons* in
validation, which is exactly the gap try08 measured (held-in resolution 72–78% vs novel
23–49%).

*Scaffolds are used for two distinct purposes above — (1) closing the leak, (2) making
validation meaningful. Same tool, different jobs.*

**3. Per-epoch validation loss, keeping the best epoch.**
There was previously **no validation set at all** — training ran a fixed 3 epochs and
never evaluated on held-out data, so nobody knew whether epoch 3 was past the overfitting
point. Training loss ending at 0.26–0.37 suggested it was, and with 5× augmentation the
model sees each exact 3-digit label 15 times (5 variants × 3 epochs). This runs up to 5
epochs and restores the weights at **minimum validation loss**.

**4. An enrichment gate, replacing the 5-molecule easy-negative test.**
The old check used 5 fixed household drugs, and try07 showed those scores are unstable
across seeds: imatinib 704/315/357, caffeine 952/652/952, metformin 632/911/653.
**Ambroxol beat metformin in 2 of 3 seeds** — so "Gaucher loses to metformin" was one
unlucky draw, not a property of the model. Those five molecules are also small and simple,
far outside the training distribution, so the model returns near-arbitrary values for them.

Replaced with: **where does the reference rank among 500 diverse decoys?** A percentile
over hundreds of molecules is stable where 5 pairwise comparisons are not.

**5. Three seeds, reported as a distribution.** The seed varies the scaffold split.

## The two metrics are now separate, and that matters
Previously one pass/fail conflated two different questions. Now:

| Metric | Question | Level |
|---|---|---|
| Enrichment percentile vs 500 decoys | Does it know this drug is special at all? | **structural class / category** |
| Leak-free hard-negative comparison | Can it tell the drug from its 3 close look-alikes? | **within-family** |

This separation is the point, because the four targets fail at *different levels* —
visible already in try08's data (200 novel compounds, single seed, leak-free):

| Target | enrichment percentile | reading |
|---|---|---|
| CF | **11.5%** | strong at category level |
| Fabry | **15.5%** | passes category, fails within-family |
| Gaucher | 38.5% | weak category; its within-family "pass" was memorization |
| TTR | **47.7%** | ~chance — fails at the category level |

A single pass/fail cannot express that, which is why all four got lumped as "two pass,
two fail" and sent the project chasing a data-volume theory that try07 then disproved.

## What is deliberately NOT in this try
- **Regularization tuning** (LoRA rank, dropout, epoch count, LR). Pointless before the
  metric works — there would be nothing trustworthy to tune against.
- **Re-testing whether 5× SMILES augmentation still helps.** It is recorded as the fix
  that got Gaucher to pass, and its original problem (verbatim string matching) was real
  — but its supporting evidence came from the broken test. Worth re-checking later, under
  a working eval, as its own change.

Changing the measurement and the model simultaneously would make any difference
uninterpretable. One at a time.

## Scope and targets
**CF and TTR first** — the informative extremes (strongest and weakest). One target per
job via an explicit `TARGET` env var, so the scope-coupling bug that cost try05 a whole
run cannot recur. Gaucher and Fabry follow once these two confirm the harness behaves.

## Decoy-set caveat, stated so the enrichment number is not over-read
There is no DrugBank or ZINC library on this system, and this script makes **no network
calls** (so the comparison cannot shift underneath it). Decoys are drawn from the other
targets' training compounds — molecules with measured bioactivity against a *different*
protein, excluded from this target's own data.

These are bioactive, drug-like molecules rather than random chemical matter, which makes
this a **conservative (harder) decoy set** than random decoys: some could legitimately be
potent here. So a good percentile is strong evidence, and a poor one is weaker evidence
than it appears. A proper absolute enrichment number needs random decoys from an external
library — separate work.

## How to read the result
Read it against try08's leaky numbers for the same target, not against zero.

| Outcome | Meaning |
|---|---|
| A target that passed before now **fails** | Its earlier pass was memorization. Expected for Gaucher; would be a significant correction for CF. |
| A target **still passes** | Real within-family discrimination. This is the result that would make CF a genuine, defensible success. |
| Enrichment ≪ 50 | Category recognition is real. <10 is strong. |
| Enrichment ≈ 50 | Chance. The model does not know the drug is special, and within-family discrimination is moot. |
| Best epoch is 0 or 1 | 3 epochs was overfitting all along, and that alone may explain much of the memorization. |
| Wide spread across seeds | Even the fixed harness is underpowered; more seeds needed before any verdict. |

## Status
Written, with a pre-submission validation of the leak fix and scaffold split. Job IDs
recorded below once submitted.
