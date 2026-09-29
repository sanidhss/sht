# Try 06 — CF/TTR as their own deliberate, decoupled run

**Date:** 29-09-2026
**Scripts:** `prep_orphan_target_data.py` (target-scoping fix) +
`lora_finetune_orphan_targets.py` (target-scoping fix) +
`lora_finetune_orphan_targets.sb`
**Job:** see below.

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

## Status
Submitted as job **22546141**. Result pending.
