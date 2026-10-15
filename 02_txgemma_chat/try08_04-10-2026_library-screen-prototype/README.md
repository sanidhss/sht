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

Job **22664818** COMPLETED 06-10-2026 03:05 (37m30s, exit 0). Both the 64-candidate
smoke test and the full 3420-candidate screen ran.
Log: `/work/hdd/bisa/sshah21/results/reasoning_screen/screen_fabry_22664818.out`
Ranked output: `.../screen_fabry_base.json`

Asserts passed: exactly one BOS survives `add_special_tokens=False` (`bos id 2 appears
1x`), yes/no token sets disjoint, and the last-real-token index never landed on padding
across 3420 candidates.

## Result: class-level screening works, within-class discrimination does not

**The screening half works.** Base model, no adapter, authored condition handed in:

| metric | value |
|---|---|
| migalastat rank | **3 / 3420** (top 0.09%) |
| recall@10 / @50 / @100 | HIT / HIT / HIT |
| enrichment over random | 570x |
| distinct scores | 2742 / 3420 (not degenerate) |
| candidates leaning yes | 12 / 3420 (0.4%) |

The cleanest statement of the result: **the model's entire positive set is 12 compounds
out of 3420, and migalastat is in it.** The yes-set (margin > 0) is exactly the top 12 —
a 285x reduction of the library that retains the true drug. For the intended role
(narrow a library so downstream docking has something affordable to verify) that is a
genuinely useful result, and it is the first positive quantitative signal this track has
produced.

Mechanistically wrong compounds are correctly dumped: lumacaftor rank 2361, caffeine
2436, tafamidis 2835 — so the model is applying the stated condition, not ranking by
generic drug-likeness.

**But the discrimination is at the class level only, and three independent pieces of
evidence say so.**

1. **No stereochemical discrimination at all.** migalastat and `fabry_analog_2` — same
   constitution, differing only in `@`/`@@` — score `0.748397231` and `0.748397052`.
   A separation of **1.8e-7**, which is bf16 numerical noise, not a judgement. The
   rank-3-vs-4 ordering between them is a coin flip, and `analog_1` and `analog_3` sit
   at ranks 6 and 7. This confirms the caveat pre-registered in this README before the
   run: the condition demands "exact stereochemistry" and the model cannot act on that
   clause.
2. **The yes-set is mostly plain sugars that fail the condition's decisive clause.**
   Ascorbic acid, choline, glucosamine, xylose, lactitol, gluconolactone, isosorbide
   mononitrate, nitroglycerin. The condition requires "a ring position capable of
   carrying positive charge" — the iminosugar ring nitrogen that makes migalastat
   (1-deoxygalactonojirimycin) work. Almost none of the other 11 have a ring nitrogen;
   glucosamine's amine is exocyclic, and xylose/lactitol/gluconolactone have no nitrogen
   at all. So the model matched the easy surface half of the condition ("compact, highly
   polar, single small sugar ring") and ignored the half that actually discriminates.
3. **Scores are sensitive to pharmacologically irrelevant SMILES features.**
   `MIGALASTAT HYDROCHLORIDE` is byte-identical to migalastat apart from a leading
   `Cl.`, and scores **-0.250 against +0.748** — a ~1.0 margin swing from a spectator
   counterion, larger than the entire spread across the top 12. (It also means
   canonical-SMILES dedup does not collapse salt forms; a future run should strip
   counterions. It did not affect this rank, but it inflates the pool.)

## How this reads against the rest of the track

This is the same pattern try06 found — family-level recognition, no fine discrimination
— but now measured on a continuous scale instead of reported as 0/4. Reframed in these
terms, try06's "wrong drug but right structural family 3 of 4 times" was never a flat
failure; it was this same partial competence, hidden by a top-1 metric over 9 candidates.

It also means the **two-stage split is the right architecture**, and the division of
labour is now explicit: the model narrows thousands to roughly a dozen by mechanism
class, and stereochemistry — the thing it demonstrably cannot do from text — is exactly
what the downstream docking stage resolves geometrically. Nothing here argues for
teaching the model stereochemical discrimination; it argues for not asking it to.

## Caveats

- **"Rank 3" is more fortunate than it looks.** The top-12 margins are tightly packed
  (+0.999 down to +0.122) and the migalastat/analog separation is noise, so a small
  perturbation reorders that block freely. The robust claim is "inside the 12-compound
  yes-set", not "rank 3". Treat recall@12 as the real measurement and top-1 as unstable.
- **n=1 disease, and it is the hardest fold** for the ranking half (pure-stereoisomer
  hard negatives). CF or TTR is the more informative second run, since their analogs are
  constitutional and therefore *should* be discriminable from text — if the model fails
  to separate those too, the limitation is broader than stereochemistry.
- **Stage 1 is untested here.** The condition was handed in. This says nothing about
  whether the model can derive it, which is try07's problem and remains unsolved.
- A CUDA OOM warning appeared mid-run at batch 16; the caching allocator recovered and
  the run completed, but batch 16 is near the ceiling on a 4xA40 node.
