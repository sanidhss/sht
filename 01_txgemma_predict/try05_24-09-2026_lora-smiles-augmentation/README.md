# Try 05 — SMILES augmentation (current state)

**Date:** 24-09-2026
**Scripts:** `prep_orphan_target_data.py` + `lora_finetune_orphan_targets.py` +
`lora_finetune_orphan_targets.sb` (current, final versions as of this reorg)
**Job:** 22354716 (TIMEOUT — see "scope bug" below)

## What it does
Adds to try04:
- **SMILES augmentation**: each training compound gets its canonical SMILES plus 4
  RDKit-generated random-but-equivalent SMILES (`Chem.MolToSmiles(mol, doRandom=True)`),
  all trained with the same real label — directly targets the memorization failure mode
  found in try04.
- **BindingDB and PubChem BioAssay** as additional data sources alongside ChEMBL (with a
  `safe_fetch` wrapper so one source's failure doesn't crash the whole run, an activity
  count guard against bulk/mislabeled ChEMBL target IDs like `CHEMBL612545`, and
  cross-source dedup by RDKit canonical structure so the same real compound recorded
  with/without explicit stereochemistry across databases isn't double-counted).
- IC50 **and EC50** activity types (needed once CFTR was added — correctors like
  lumacaftor have no IC50 at all, only EC50).
- Two more targets (CF/lumacaftor, TTR/tafamidis) added to `TARGETS`, originally for the
  *chat reasoning* pipeline (see `02_txgemma_chat/try03`), not this one.

## What happened / why it failed (partially)
**Real scope bug**: this script has no target list of its own — it trains on whatever
`coverage_report.json` marks `ready`, and since CF/TTR were added to the *shared*
`TARGETS` dict for the chat-reasoning use case, this run silently also tried to
numeric-fine-tune CF and TTR, which was never intended for this run. Combined with 5x
SMILES augmentation on much larger pools, the job ran its full 8h budget and **timed out
partway through loading CF**, before finishing CF/TTR or writing a final summary.

**The parts that did finish (Fabry, Gaucher) are genuinely informative:**
- Fabry: migalastat predicted 436 (real 196) — ties exactly with all 3 hard-negative
  diastereomers (all predicted 436). Better than try04's confident wrong answer, but
  still fails (can't distinguish stereoisomers at all now, rather than misidentifying).
- **Gaucher: ambroxol predicted 691 (real 685) — beats all 3 real hard negatives.
  First genuine pass on the hard-negative test for either target.**

Follow-up needed: give this script its own explicit target list (decoupled from the
shared prep pipeline's `TARGETS`), then decide separately whether a CF/TTR numeric
fine-tune is wanted as its own deliberate run.
