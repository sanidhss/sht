# Try 02 — Double-blinded single-shot chaperone reasoning

**Date:** 23-09-2026 (this is the user's own script, not authored in this session)
**Script:** `blind_chaperone_rank_txgemma.py` + `blind_chaperone_rank_txgemma.sb`
**Jobs:** 22289304 (CANCELLED, before this session), 22289878 (FAILED), 22323973
(FAILED), 22344312 (COMPLETED)

## What it does
Tests whether `txgemma-27b-chat`'s *general* biochemical reasoning (not the narrow
numeric predict head) can identify the right chaperone, with drug identity AND target
identity both blinded (candidates relabeled "Compound A/B/C...", target described only
by raw sequence + natural substrate chemistry, never by gene/disease name) — designed so
a correct answer requires real mechanistic reasoning, not memorized "Fabry -> migalastat"
fact recall.

## What happened / why it failed
22289878 and 22323973 failed on the same `generate()` bug diagnosed in try01 (identical
root cause, same fix applied). 22344312 (after the fix) ran to completion but the
**scientific result also failed**: the model picked **imatinib** as top choice for both
GLA and GBA1, reasoning that its "complex structure and multiple aromatic rings" would
best mimic the substrate. This is backwards for this enzyme class — real chaperones
(migalastat, isofagomine) are small, simple iminosugars that mimic a sugar transition
state; imatinib's binding strategy (aromatic stacking, kinase ATP pocket fitting) is
irrelevant to a glycosidase active site. The reasoning was fluent and confident-sounding
despite being wrong — flagged as a real risk (a wrong chat answer doesn't look uncertain
the way a low numeric score does).

This result motivated the leave-one-target-out reasoning fine-tune in try03, designed to
teach mechanism-specific reasoning that generalizes rather than a generic
complexity-implies-binding heuristic.
