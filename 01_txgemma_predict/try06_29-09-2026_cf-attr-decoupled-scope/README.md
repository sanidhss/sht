# Try 06 — CF/TTR as their own deliberate, decoupled run

**Date:** 29-09-2026
**Scripts:** `prep_orphan_target_data.py` (target-scoping fix) +
`lora_finetune_orphan_targets.py` (target-scoping fix) +
`lora_finetune_orphan_targets.sb`
**Job:** 22546141 (COMPLETED) -- CF passes, TTR fails; see Result below.

## Why this try exists
CF and TTR were added to the shared `TARGETS` dict back in try05, originally for the
chat-reasoning pipeline's needs, and `lora_finetune_orphan_targets.py` had no target
list of its own -- it trained on whatever `coverage_report.json` marked ready, which
is shared across every target anyone has ever prepped. That's what caused try05's job
(22354716) to silently also attempt CF/TTR alongside Fabry/Gaucher and time out at its
8h budget partway through loading CF, never finishing or writing a final summary. CF
and TTR have never gotten a real, complete numeric-predict run.

## Strategic context for this run
Decided (see conversation) to pause further Fabry-specific stereochemistry engineering
(the diastereomer-collapse problem) rather than keep deepening investment in a fix
narrowly tailored to lysosomal-storage-disorder chaperones, with no evidence it would
extrapolate to a genuinely novel receptor profile. This run is the actual test of
extrapolation breadth: does the *existing, unmodified* general pipeline (ChEMBL +
BindingDB + PubChem, SMILES augmentation, real hard-negative discrimination test) hold
up on two mechanistically different disease classes -- CF (folding/trafficking
corrector, no active-site mimicry at all) and TTR (oligomer-interface stabilizer) --
or does it only work for lysosomal enzymes?

Also relevant context for interpreting whatever this produces: the real deployment
pipeline downstream of this ranking step does biochemical filtering, then structural
validation (AF3, PyRosetta, Boltz2, potential wet lab) on the surviving candidates.
The bar for this step isn't "rank the correct drug #1" -- it's "does the correct drug
survive into the top-ranked shortlist that gets forwarded to those structural tools."
Worth re-evaluating existing results (including Fabry's) against a top-K framing
rather than strict pairwise hard-negative discrimination, separately from this run.

## The fix
Both `prep_orphan_target_data.py` and `lora_finetune_orphan_targets.py` now respect a
`TARGETS_TO_RUN` env var (comma-separated). Set to `cf,attr` for this run via the
`.sb`. `prep`'s `coverage_report.json` write now merges into the existing file instead
of overwriting it, so Fabry/Gaucher's entries survive a CF/TTR-only run untouched.
Without `TARGETS_TO_RUN` set, both scripts fall back to the old (shared/coupled)
behavior -- this is additive, not a breaking change to the pattern.

Also bumped the time budget to 12h (from try05's 8h) since CF alone has 1,427 real
compounds -- more than Gaucher's 1,208 -- and CF+TTR together is comparable to or
larger than what caused the 8h budget to be exceeded before.

## Result
Job **22546141**, COMPLETED 02-10-2026 17:52, elapsed 07:04:41 (inside the raised 12h
budget; the 8h budget that killed try05 would have been marginal, not sufficient).

The target-scoping fix worked as intended: only `cf,attr` ran, and the merged
`coverage_report.json` kept Fabry's and Gaucher's entries intact rather than
overwriting them. That part of the try is settled -- `TARGETS_TO_RUN` does what it
was built to do.

**Split decision on the actual question.** Note scores here are potency-like, so
*lower is better*; CF's reference at 411 beating easy negatives at 633-926 is what
establishes the direction.

| Target | n_train | Reference (real label) | Beats easy | Beats hard | Verdict |
|---|---|---|---|---|---|
| CF (lumacaftor) | 1,427 | 411 (584) | yes | yes | **PASS** |
| TTR (tafamidis) | 493 | 781 (552) | no | no | **FAIL** |

CF hard negatives: 661 (real 684), 584 (real 584), 661 (real 663) -- reference beats
all three. Easy negatives 633-926.
TTR hard negatives: 781, 781, 781 (real 781 / 759 / 774). Easy negatives 660-994.
Training converged normally for both (CF loss 1.0214 -> 0.5030 -> 0.3668; TTR
1.1200 -> 0.4487 -> 0.2871), so neither outcome is an under-training artifact.

### Diagnosis: CF's pass is the load-bearing result
CF is a folding/trafficking corrector with no active-site mimicry whatsoever --
mechanistically about as distant from a lysosomal-enzyme chaperone as the four
targets get. The *unmodified* general pipeline handled it and beat all three real
structural analogs. **This retires the "the pipeline only works for lysosomal
enzymes" hypothesis**, which was the main thing this try was built to test.

### Diagnosis: TTR fails the same way Fabry does, only worse
Two specifics worth recording:

1. **Identical collapse signature to Fabry** -- one score (781) emitted for tafamidis
   and all three analogs alike. And 781 is *exactly* the real label of hard negative
   #1 (sim=0.43). That is near-neighbour label memorization: the model is reciting a
   training-set neighbour's label instead of reasoning over structure. This is
   precisely the failure mode SMILES augmentation was introduced in try05 to fix, and
   augmentation did run here (5.0x, 2,463 lines from 493 compounds). **Augmentation
   alone does not prevent it at this data volume.**
2. **TTR fails easy negatives too, which Fabry does not.** Caffeine (720) and
   ibuprofen (660) both score better than tafamidis (781). Fabry at least keeps its
   reference ahead of unrelated drug classes. This is a floor-level failure, and it
   matters for the pending top-K reframing: a generous top-K reading could plausibly
   excuse Fabry's 4-way tie with its own diastereomers, but it cannot excuse losing
   to caffeine. **The two failures should not be re-scored under the same rule.**

### The cross-target pattern this exposes
Sorted by training volume, pass/fail is perfectly monotonic across all four targets:

    CF      1,427  PASS
    Gaucher 1,208  PASS
    TTR       493  FAIL
    Fabry      89  FAIL

Clean break between ~1,200 and ~500. This is a simpler explanation than anything
mechanism-specific, and it is consistent with the already-verified Fabry root cause
(89 compounds, read directly from `coverage_report.json`). Caveat, stated plainly so a
future session does not over-read it: **n=4 targets, and training volume is confounded
with how well-studied each target is** -- a better-studied target has both more data
and possibly cleaner data. This is a hypothesis with suggestive support, not an
established training floor.

If it holds, it reframes Fabry and TTR as *one* finding (insufficient real
bioactivity data) rather than two separate mechanism-specific failures -- a
substantially stronger and more generalizable thesis claim than either alone. The
cheapest decisive test: downsample Gaucher's 1,208 to ~490 and check whether it
reproduces TTR's collapse. Same target, same mechanism, same pipeline, volume as the
only varied factor -- which breaks the confound.

### Consequence for the planned next phase
`NEXT_PHASE_diverse_structural_training.md` (RDKit-descriptor physics/steric
awareness) was scoped as the fix for "the general pipeline needs more than chemistry
alone." CF's pass is evidence against that premise on the mechanism-breadth axis.
The evidence now points at data sufficiency instead, which is a cheaper and more
direct fix. **That doc should be treated as pending re-justification, not as the
agreed next step.**

### Operational note
A CUDA OOM warning fired on device 0 during CF's training setup (`memory allocation
failed with OOM ... free: 87293952, total: 42405855232`). The allocator recovered and
training completed and converged normally, so results are not suspect -- but the
9B + CF's 7,135 training lines configuration is running near the memory ceiling and
a larger target could tip it.
