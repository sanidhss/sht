# Try 08 — Output-resolution probe: how many distinct scores can each adapter emit?

**Date:** 04-10-2026
**Scripts:** `output_resolution_probe.py` + `output_resolution_probe.sb`
**Job:** 22660579 (submitted 04-10-2026) -- result pending; see Status below.

**This is a measurement, not a fix.** It trains nothing, changes no training data, and
adds no new eval criterion. It loads the four already-saved adapters and asks how many
different numbers each one is capable of producing.

## Why this try exists
The hard-negative test uses 9 molecules per target. With 9 molecules, two very
different situations are indistinguishable:

- Gaucher emitted 791 for six molecules **because those six really are similar**, or
- Gaucher emitted 791 for six molecules **because 791 is one of only a handful of
  values it can produce at all**.

Under the second reading, "ambroxol 691 beats metformin 661" is a 30-point gap on an
instrument whose smallest real step might be ~100 points — rounding noise presented as
a ranking.

This matters well beyond the hard-negative test. The downstream plan is to rank a large
library and forward a top-K shortlist to AF3 / PyRosetta / Boltz2. **A model that emits
~15 distinct values across hundreds of compounds turns a 1,000-compound ranking into
~15 giant tie-groups**, making any top-50 shortlist an arbitrary slice of a 200-way tie.
No amount of extra bioactivity data fixes that — it is a property of how the model
emits scores, not of what it learned. So this is a go/no-go check on whether the
ranking step can do the job it exists to do.

## The observation that prompted it
Distinct predicted values across each target's 9 eval molecules:

| Target | distinct | reference | unrelated drugs | real analogs |
|---|---|---|---|---|
| CF | 7 of 9 | 411 | 804, 833, 633, 926, 633 | 661, 584, 661 |
| Gaucher | **4 of 9** | 691 | **791, 791, 661, 791, 791** | **791, 791**, 839 |
| Fabry | 6 of 9 | 436 | 697, 860, 744, 742, 685 | **436, 436, 436** |
| TTR | 5 of 9 | 781 | 918, 720, 994, 660, 994 | **781, 781, 781** |

Two different failures are visible, and the existing pass/fail gate conflates them:

- **Failure A — within-family resolution.** Fabry and TTR give the reference *and all
  three close analogs* one identical value (436×4, 781×4), while resolving unrelated
  drugs perfectly well. They treat an entire structural family as a single object. This
  is the failure that actually blocks candidate ranking.
- **Failure B — output degeneracy.** Gaucher dumps 791 onto six molecules and loses to
  metformin, at 1,208 training compounds. **Not a data-volume problem** — so extra data
  will not fix it, and it is why Gaucher's recorded "pass" is softer than it reads.

This probe measures the capacity underlying both. It does not resolve which is which on
its own, but it decides whether the score scale is fine-grained enough for any of these
rankings to mean anything.

## How the probe set is chosen
Selection determines whether the result means anything: 300 *similar* molecules would
yield few distinct values trivially and prove nothing.

**Pool** — compounds in CF/TTR/Fabry training data but **not** in Gaucher's. Verified
before building: 2,030 pooled distinct, only **30** overlap with Gaucher, so ~2,000 are
novel to it. Real, measured, drug-like molecules already on disk; no network.

**Selection** — RDKit `MaxMinPicker` on Morgan fingerprints (radius 2, 2048 bits), not
random sampling. MaxMin greedily takes the molecule furthest from everything already
picked, so it spans the pool; random sampling could return a cluster by luck.

**Stratified**, because one aggregate number would not be interpretable:

| Stratum | n | What it isolates |
|---|---|---|
| MaxMin-diverse, novel to Gaucher | 200 | resolution on unseen chemistry — the headline |
| Random from the probed target's own training set | 50 | **memorization contrast** — fine resolution here but coarse on novel compounds means recall, not resolution |
| Highest-Tanimoto neighbours of that target's reference drug | 50 | **within-family** resolution (Failure A) directly |
| The existing 9 eval molecules | 9 | anchors, so output ties back to numbers already on record |

The probe set is built once with a fixed seed and saved, then reused **identically**
across all four adapters — otherwise cross-adapter comparison is meaningless.

Each molecule is prompted against **its own target's protein sequence**, recovered from
that target's `eval_set.json` by substituting the SMILES line of a known-good prompt.
Probing the Gaucher adapter with CFTR's sequence would measure nothing. Verified: all
four templates substitute cleanly with their sequences preserved.

## What gets reported
- `n_distinct / n_total`, overall and per stratum
- the value histogram — names the attractor values (is 791 one of them?)
- **`min_nonzero_gap`** — the smallest step between adjacent emitted values. This is
  *the* number: it decides whether a 30-point difference like 691-vs-661 is signal or
  rounding.
- the probe set's **own** mean pairwise Tanimoto — so if few values come out, "the
  inputs were all alike" can be ruled out as the cause
- count of generations that were not parseable integers

## How to read the result
| Outcome | Meaning |
|---|---|
| High distinct % + small `min_nonzero_gap` | Fine resolution. The 9-molecule clustering was real chemistry, rankings are meaningful, and the volume hypothesis (try07) remains the main thread. |
| **Low distinct % + large `min_nonzero_gap`** | A **resolution ceiling**. Small score differences are rounding, library ranking collapses into a few tie-groups, and more training data cannot fix it. This would make output calibration — not data volume — the primary problem, and would partly invalidate how existing pass/fail verdicts were read. |
| High on `held_in`, low on `diverse_novel` | Memorization, not resolution. The model recalls trained compounds and rounds everything else. |

If resolution differs sharply across adapters (e.g. CF fine, TTR coarse), then
resolution tracks pass/fail, and that is a finding in its own right.

## Known limitation, stated so the number is not over-read
The pool is bioactivity data for these four targets. It is drug-like and diverse
*within that space*, but it is **not** a random slice of chemical space and **not** a
substitute for a real DrugBank-wide screen. It is adequate for the question asked here
— can the model emit many distinct values at all — and not for estimating real
screening performance.

Also: a separate base-model load per adapter rather than adapter-swapping on one shared
base. try06's code records an unexplained backwards result from the shared-base
approach that was never root-caused; this probe's conclusions need to be trustworthy,
so it pays the extra load time.

## Relationship to try07
Independent and complementary, not a replacement.

- **try07** (job 22660248) tests **Failure A against data volume**: does Gaucher's
  one-of-three analog resolution survive a cut to 493 compounds?
- **try08** (this) tests whether the **score scale is fine-grained enough** for any of
  those comparisons to be meaningful — including try07's own.

If try08 finds a hard resolution ceiling, try07's result still stands as measured but
has to be read against that ceiling: "collapsed" and "merely rounded to the same
attractor value" would become the same observation, which is worth knowing before
interpreting try07.

## Result
Job **22660579**, COMPLETED 06-10-2026, elapsed 01:34:00 (faster than the ~3h estimate;
the 6h budget was ample). Probe-set diversity as built: mean pairwise Tanimoto **0.105**
on the diverse stratum, so none of the findings below can be attributed to similar inputs.

### Headline: there is no quantization ceiling, but there is severe degeneracy
| Adapter | distinct / 309 | novel stratum | held-in stratum | min_nonzero_gap |
|---|---|---|---|---|
| CF | 134 (43.4%) | **49.0%** | 72.0% | 1 |
| TTR (attr) | 123 (39.9%) | **47.2%** | 78.0% | 1 |
| Gaucher | 93 (30.1%) | **27.5%** | 74.0% | 1 |
| Fabry | 57 (18.4%) | **23.0%** | 62.0% | 1 |

**`min_nonzero_gap` is 1 for every adapter.** The models are perfectly capable of
emitting fine-grained values, so the worry that "691 vs 661 is rounding" is **not**
correct at the instrument level. The score scale is not quantized. That part of the
hypothesis is disproved.

### But the memorization control fired on every single adapter
Resolution on compounds the model trained on is far higher than on novel chemistry:

    Gaucher  held-in 74.0%  vs novel 27.5%   (2.7x)
    Fabry    held-in 62.0%  vs novel 23.0%   (2.7x)
    TTR      held-in 78.0%  vs novel 47.2%   (1.65x)
    CF       held-in 72.0%  vs novel 49.0%   (1.5x)

This is exactly the branch the control was built to detect: **fine resolution on trained
compounds, coarse on unseen ones is recall, not learned structure-activity reasoning.**
Every adapter shows it; Gaucher and Fabry worst.

### The attractor values are memorized hard-negative labels
Most frequent predictions across the 200 diverse *novel* compounds:

    Gaucher   791 x102 (51% of all novel compounds!), 679 x17
    TTR       994 x57  (29%),  687 x9
    Fabry     630 x49  (25%),  541 x17
    CF        727 x13  (6.5%), 615 x10   <- flat by comparison

**Gaucher emits 791 for half of all novel molecules.** And 791 is the real label of one
of its own hard negatives. 679, its second most common output, is the real label of
another. So Gaucher's "pass" amounts to ambroxol (691) landing below the default value
the model emits for most of chemical space -- not to discrimination between ambroxol and
the analogs.

CF is the outlier in the good direction: its most common output covers only 6.5% of novel
compounds, and it has the highest novel-compound resolution (49.0%). CF's pass looks
genuine in a way Gaucher's does not.

### The leak that explains all of it
Checked directly while interpreting this run: **all 12 hard negatives, across all four
targets, are present in their training sets.** Only the reference is held out.

    gaucher  reference in training? False | all 3 hard negatives: True
    cf       reference in training? False | all 3 hard negatives: True
    fabry    reference in training? False | all 3 hard negatives: True
    attr     reference in training? False | all 3 hard negatives: True

The script docstring only ever claimed the *reference* was held out ("reference drug held
out entirely"), so this is an omission in the test's design rather than a code bug. But it
voids the test's stated purpose -- `lora_finetune_orphan_targets.py` describes the
hard-negative check as "the test that actually indicates the model learned the right
structure-activity signal." It cannot be that, because it pits a **generalization** task
(predict a held-out drug) against a **memorization** task (recall three trained analogs).

It also explains the "collapse" signature that drove the Fabry and TTR diagnoses. TTR
emitting 781 for the reference *and* all three analogs is the model applying analog #1's
memorized label (real 781) to the whole structural family. Fabry's 436-for-everything is
the same mechanism. That is not a stereochemistry-resolution failure, which is how Fabry
was diagnosed for weeks.

### Resolution does not track pass/fail
The pass/fail ordering (CF pass, Gaucher pass, TTR fail, Fabry fail) does **not** match
the novel-resolution ordering (CF 49.0%, TTR 47.2%, Gaucher 27.5%, Fabry 23.0%). TTR
*fails* while resolving novel chemistry nearly as well as CF, which *passes*; Gaucher
*passes* with substantially worse resolution than TTR. So the verdict is not measuring
model quality on unseen compounds.

### Incidental
- Fabry's reference-neighbour stratum reached Tanimoto **1.0**, i.e. a molecule with an
  identical Morgan fingerprint to migalastat. Default Morgan fingerprints ignore
  stereochemistry, so migalastat's diastereomers are fingerprint-identical. Worth knowing
  for any future similarity-based work on Fabry: `useChirality=True` would be required.
- Fabry's held-in and neighbour strata both bottom out at exactly 436 -- the collapse
  value -- consistent with 436 being an attractor rather than a computed score.
- CF emitted 0 as a value on the novel stratum, and Fabry did too; worth a glance if CF's
  numbers are ever used quantitatively.

## Status
COMPLETED. Both objectives met: the quantization-ceiling hypothesis is disproved
(`min_gap=1` everywhere), and the memorization control fired on all four adapters,
surfacing the hard-negative training leak that reframes the project's existing verdicts.
