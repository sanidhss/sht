# Project: TxGemma-Predict (numeric drug-target scoring)

**Question:** Can `txgemma-9b-predict`'s `BindingDB_ki`-style numeric scoring correctly
rank a known drug (migalastat/GLA, ambroxol/GBA1) above unrelated and structurally
near-identical decoys, for orphan disease targets?

## Arc
1. **try01/try02** — zero-shot: failed. Root cause is data coverage, not a broken
   pipeline: BindingDB has 0 records for GLA, 17 for GBA1, and the one real ABL1/imatinib
   hit is filed under a different sequence identity (BCR-ABL fusion) than what was asked.
2. **try03** — first LoRA fine-tune on real ChEMBL data: looked like a win (migalastat
   #1 of 7), but the test used only easy, unrelated negative drugs.
3. **try04** — added real hard negatives (structural near-misses): the try03 win didn't
   hold up. Diagnosed as SMILES-string memorization, not real structure-activity
   learning.
4. **try05 (current)** — added SMILES augmentation + more data sources (BindingDB,
   PubChem BioAssay) + EC50 support. Gaucher now genuinely passes the hard-negative
   test; Fabry improved but still fails (ties rather than misidentifies). A scope bug
   (shared target list with the chat-reasoning project) caused this run to also attempt
   2 unintended targets and time out before finishing.

5. **try06 (current)** — fixed the scope-coupling bug (`TARGETS_TO_RUN` env var, both
   scripts) and ran CF/TTR as their own deliberate, isolated run through the
   *unmodified* pipeline — the real test of whether the general approach extrapolates
   beyond lysosomal-enzyme chaperones. Job 22546141 COMPLETED: **CF passes** (beats
   all 3 real structural analogs), **TTR fails** (same one-score collapse as Fabry,
   and additionally loses to easy negatives). See try06's Result for the diagnosis.

## Current state
Passing: **Gaucher** and **CF**. Failing: **Fabry** (collapses on diastereomers) and
**TTR** (collapses on analogs *and* loses to unrelated drugs).

CF's pass is the significant one — CF is a folding/trafficking corrector with no
active-site mimicry, so the *unmodified* pipeline demonstrably extrapolates past
lysosomal-enzyme chaperones. That hypothesis is retired.

The live hypothesis is now **training-data volume, not mechanism**. Pass/fail is
monotonic in n_training across all four targets (CF 1,427 pass, Gaucher 1,208 pass,
TTR 493 fail, Fabry 89 fail), with a clean break between ~1,200 and ~500. Caveat:
n=4, and volume is confounded with how well-studied each target is. The decisive
cheap test is downsampling Gaucher to ~490 and seeing whether it reproduces TTR's
collapse — same target and mechanism, volume as the only varied factor. If it holds,
Fabry and TTR become one data-sufficiency finding rather than two mechanism-specific
failures.

Consequently `NEXT_PHASE_diverse_structural_training.md` (RDKit structural-awareness
training) is **pending re-justification** — it was scoped for "chemistry alone isn't
enough," and CF's pass is evidence against that premise on the mechanism axis.

Also open: the top-K reframing. Note it cannot be applied uniformly to Fabry and TTR
— a top-K bar could excuse Fabry's tie with its own diastereomers, but not TTR
scoring worse than caffeine.
