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
   beyond lysosomal-enzyme chaperones. Submitted as job 22546141, result pending.

## Current state
Gaucher: passing. Fabry: fails its hard-negative test (collapses on diastereomers,
traced to a 13.5x data gap vs. Gaucher, see try05's Analysis). Decided *not* to keep
deepening Fabry-specific stereochemistry fixes — no evidence they'd extrapolate to a
disease with a different receptor profile. try06 tests CF/TTR under the existing
general pipeline instead. See `NEXT_PHASE_diverse_structural_training.md` for the
planned next phase (diverse multi-disease + RDKit-computable structural-awareness
training) if try06 shows the general approach needs more than chemistry alone.
