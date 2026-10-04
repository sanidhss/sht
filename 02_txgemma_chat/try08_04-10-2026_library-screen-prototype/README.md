# Try 08 — stage-2 library-screen prototype (apply a given condition to 3.4k real drugs)

**Date:** 04-10-2026
**Scripts:** `fetch_library.py` (login node, needs network) + `screen_library.py` + `screen_library.sb`
**Job:** 22664818 (`screen_fabry`, gpuA40x4). Result pending.

## Why this exists

Two problems with the try02–try07 setup, both structural rather than model failures.

**1. The eval format didn't match the task.** Every fold handed the model 9 candidates:
the true drug, 3 authored analogs, the 3 other folds' real drugs, and caffeine + aspirin.
So "correct" meant a 1-of-9 pick against two obvious decoys. The actual goal is to take a
disease receptor and find the closest drug match out of a realistic library, which is a
very different and much harder problem. A number from the 9-candidate format wouldn't
transfer even if it were good, and top-1 accuracy over 4 folds is only 4 bits of signal —
which is why 0/4 (try06) vs some other small number is unreadable.

**2. The design fused two separable questions** into one final ranking:

1. derive a necessary condition from a mechanism description, and
2. apply that condition to rank candidates.

A wrong answer could come from either, and they are confounded in the top-1 metric. That
is part of why try04–try06 were hard to diagnose.

This try isolates **(2)**: the authored ground-truth `necessary_condition` is handed to the
model, and the only thing measured is whether it can apply that condition to rank a real
library. Nothing here tests (1) — try07 is the diagnostic for that, and its grounding
failure is independent of whatever this finds.

## Why the data scarcity doesn't block this

Expanding question (1) needs hand-authored mechanism text per disease, and there are only
on the order of tens of well-documented pharmacological chaperone mechanisms — the
interface-stabilizer class especially is thin. That scarcity is real and it caps how far
the *training* side can scale.

But question (2) needs **only SMILES**. The candidate pool can be thousands of compounds
at zero authoring cost. So the realistic-library half of the problem is reachable right
now, with the existing 4 authored diseases.

## Library

`fetch_library.py` pulls approved drugs (`max_phase=4`) with canonical SMILES from the
ChEMBL REST API: 3417 unique parseable structures after rdkit canonicalization
(`isomericSmiles=True` — stereochemistry must not be flattened, see below).

- migalastat, lumacaftor, tafamidis, caffeine, aspirin are all present.
- **ambroxol is absent** (not `max_phase=4` in ChEMBL — approved in Europe, never US FDA).
  Doesn't block the Fabry fold, but a Gaucher run will need it spiked in explicitly.
- ChEMBL's migalastat SMILES is **stereochemically identical** to the authored `Compound B`
  (`OC[C@H]1NC[C@H](O)[C@@H](O)[C@H]1O`), confirmed by canonicalized string equality — so
  the authored data is correct and ranks are directly comparable to try06's.
- The 3 authored `fabry_analog_*` stereoisomers are **not** in the approved library (they
  are synthetic), so they are spiked in as hard negatives. They are the negatives try06 and
  try07 actually failed on, so the comparison needs them present.

## Scoring method

One **forward pass** per candidate, not a generation: build a short verdict prompt
(condition + SMILES + "answer yes or no") and read the logprob margin
`logsumexp(yes tokens) − logsumexp(no tokens)` at the first generated position. Rank by
that margin.

This is the part that makes a 3.4k-candidate pool affordable at all. try07 needed ~8 min
per 2048-token completion; one generation per candidate would be hundreds of hours. Short
prompts, batched 16 at a time, is ~214 forward passes for the whole library.

Padding is left-side (decoder-only — the last position has to be the true end of every
prompt).

Runs the **base model with no adapter** by default: the honest baseline for "can bare
TxGemma apply a stated condition". `--adapter` compares a try06 LoRA.

## Metrics

Rank-of-true-drug among 3420, which is continuous and readable at n=1 disease, plus
recall@10/50/100 and enrichment over random. **Recall@50 is the metric that matches the
intended role** — docking verification lives downstream in the user's own pipeline, so
TxGemma's job is narrowing thousands to a shortlist, and top-1 was never the right target.
Under that framing try06's "wrong drug but right structural family 3 of 4 times" is a
partial pass, not a failure, since a family hit lands inside the shortlist.

Ties are reported explicitly (rank counted by strictly-greater score, plus a count of
candidates tied at the true drug's score). This matters: if the model gives nearly every
candidate the same verdict, rank is meaningless, and the tie count is what reveals it.

## The specific thing being checked

The calibration worry. The 9-candidate sets were *enriched* — they contained deliberate
junk (caffeine, aspirin) that is easy to reject. A real approved-drug library is almost
entirely plausible drug-like organics. The open question is whether the verdicts still
separate anything once the obvious decoys are gone, or whether the model leans "yes" on
nearly everything. The `calibration:` line in the output (fraction leaning yes, number of
distinct scores, margin range) is the direct read on that, and it should be checked
**before** trusting the rank.

Note also that for the Fabry fold specifically, the top of the ranking requires
distinguishing migalastat from three of its own stereoisomers. The authored mechanism text
says the candidate must match "ring shape, charge, and stereochemistry" of the terminal
alpha-linked sugar's transition state but never names the sugar's configuration, so
exact-stereoisomer discrimination may not be derivable from the prompt at all. If the
analogs rank above migalastat, that is the likely reason, and it is an eval-design issue
rather than a model one.

## Status

Submitted as job **22664818**. The `.sb` runs a 64-candidate smoke test first (subset
always retains the true drug and the analogs) and exits non-zero if it fails, then the
full 3420-candidate screen. Result pending.
