# Try 08 — Output-resolution probe: how many distinct scores can each adapter emit?

**Date:** 04-10-2026
**Scripts:** `output_resolution_probe.py` + `output_resolution_probe.sb`
**Job:** see Status below.

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

## Status
Not yet submitted — see Result below once it has run.
