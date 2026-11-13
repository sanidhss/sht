# Try 09 — Leak-free evaluation harness

**Date:** 18-10-2026
**Scripts:** `leak_free_eval.py` + `leak_free_eval_cf.sb` + `leak_free_eval_attr.sb`
**Jobs:** 22996487 (CF) + 22996488 (TTR) -- results pending; see Status below.

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

## Result

**TTR (22996488): COMPLETED**, 07:11:47, all 3 seeds.
**CF (22996487): TIMED OUT** at 12:00:04 with 1 of 3 seeds finished — my budget estimate
was wrong. 5 epochs plus per-epoch validation on 1,140 held-out lines costs far more than
the old 3 epochs with no validation; CF's single seed took ~7h, so 3 seeds needed ~21h.

### The finding that reframes everything: the optimal stopping point is epoch 0
`best_epoch = 0` in **every one of the 4 completed runs**, and validation loss rises
essentially monotonically after it:

    TTR seed0  val 2.37 -> 3.10 -> 3.48 -> 3.56 -> 3.77
    TTR seed1  val 2.36 -> 2.91 -> 3.22 -> 3.40 -> 4.42
    TTR seed2  val 2.97 -> 4.26 -> 3.87 -> 4.84 -> 4.01
    CF  seed0  val 2.39 -> 2.62 -> 3.46 -> 2.94 -> 3.58

Meanwhile training loss falls the whole way (1.0 -> 0.19). **The original pipeline trained
3 epochs with no validation set, so every result this project has produced came from a
model trained roughly 3x past its optimal stopping point.** That is a complete, mundane
explanation for the memorization try08 measured, and it needed no architectural theory.

### TTR, leak-free: from "fails everything" to a stable pass
| | leaky (try06) | leak-free (try09, 3 seeds) |
|---|---|---|
| beats hard negatives | no | **2 of 3 seeds** |
| reference prediction | 781 | 672 / 646 / 687 — **spread 41** |
| enrichment percentile | 47.7% (~chance) | **25.0 / 19.0 / 23.4** — spread 6.0 |
| collapsed | yes | 1 of 3 seeds |

Two things stand out. The hard negatives are now genuinely held out, so passing is
*harder* than before — and TTR passes 2 of 3 anyway. And the stability is a different
regime entirely: a **41-point** spread across seeds, against the **377-point** spread try07
measured on the old pipeline. Stopping at epoch 0 appears to remove most of the
seed-sensitivity along with the memorization.

TTR was diagnosed as the worst target in the project. It is not.

### CF, leak-free: the clean pass does not survive
    cf_seed0: reference 555 (real 584) | analogs 734, 540, 529 -> beats_hard = FALSE
              enrichment 21.4 percentile, best_epoch 0

Under the leaky test CF passed cleanly (411 vs 661/584/661). With the analogs actually held
out, **two of the three beat it** (540 and 529 vs 555). This is exactly the predicted
signature of a pass that depended on memorized analog labels.

**Caveat: n=1 seed for CF.** Given TTR went 2/3, a single CF seed cannot settle this. CF
must be re-run before the claim "CF fails" is made.

### The CF-vs-TTR ranking inverted, and the category gap was an artifact
On enrichment — the category-level gate — the two targets are now indistinguishable:

    CF   21.4 percentile  (n=1)
    TTR  19.0 / 23.4 / 25.0 percentile

try08's leaky single-seed measurements had CF at 11.5% and TTR at 47.7% (chance), which
read as a decisive gap. It was not one. (The two numbers are not strictly comparable —
different decoy sets and sizes — but the within-try09 comparison is valid, and it shows no
gap.) **"CF good, TTR hopeless" does not survive the fixed harness.**

### What early stopping did NOT fix: output degeneracy
The attractor behaviour persists. Distinct values among 500 decoys, and the most common:

    TTR seed0   92 distinct | 994 x197  (39% of all decoys)
    TTR seed1   70 distinct | 994 x165  (33%)
    TTR seed2   78 distinct | 994 x199  (40%)
    CF  seed0  111 distinct | 889 x71   (14%)

TTR still assigns 994 to roughly 40% of a diverse 500-compound set. So the degeneracy
try08 found (Gaucher's 791 on 51% of novel compounds) is **not** merely an overfitting
artifact and remains an open problem — and it is the one that most directly threatens
top-K library ranking, since a 40% tie-group makes a shortlist arbitrary.

### Also worth noting
Validation loss at its *best* is 2.26–2.97 while training loss at the same epoch is ~1.0.
The model is already overfitting *within* the first epoch. So epoch-level early stopping
may itself be too coarse — mid-epoch evaluation, a lower learning rate, or fewer steps are
the obvious things to try next.

### Consequences
1. **Re-run CF with 3 seeds.** Since `best_epoch=0` is now established across 4 runs,
   `N_EPOCH` can drop from 5 to 2 — enough to confirm epoch 0 is still the minimum while
   cutting training time ~60%, which makes 3 seeds fit comfortably.
2. **Re-run Gaucher and Fabry through this harness.** Their recorded verdicts were produced
   by over-trained, leaky models and carry no weight now. Gaucher's "pass" and Fabry's
   stereochemistry diagnosis both need redoing.
3. **Stop at epoch 0 (or earlier) everywhere**, and never again train without a validation
   set.
4. **Output degeneracy is now the main open technical problem**, separate from the eval
   fixes and not addressed by them.

## Status
TTR complete (3 seeds). CF incomplete (1 of 3 seeds, timed out) and needs a re-run with
`N_EPOCH=2`. Headline conclusions: optimal stopping is epoch 0 in every run; TTR passes
2/3 leak-free with tight seed spread; CF's leaky clean pass does not reproduce on its one
completed seed; the CF/TTR category gap was an artifact; output degeneracy survives all of
the fixes.
