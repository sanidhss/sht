# Try 04 — Real hard negatives added (pre-SMILES-augmentation)

**Date:** 24-09-2026
**Scripts:** early version of `prep_orphan_target_data.py` + `lora_finetune_orphan_targets.py`
— **exact script content at this point was overwritten by later edits and not
separately preserved.** See try05 for the current/final version of these files; the
main differences at this stage were: no SMILES augmentation, no EC50 support (IC50
only), no BindingDB/PubChem sources yet integrated, and only 2 targets (fabry, gaucher).
**Job:** 22344582 (COMPLETED)

## What it does (at this stage)
Generalized version of try03's pipeline: pulls real ChEMBL data per target, verifies a
reference drug with real direct-binding evidence (migalastat for GLA, **ambroxol** for
GBA1 — chosen after confirming miglustat/eliglustat, though real approved Gaucher drugs,
work via a different target (UGCG) entirely and aren't valid references here), and adds
**real hard negatives**: the training pool's own compounds most structurally similar to
the reference (via RDKit Tanimoto similarity on Morgan fingerprints), not just unrelated
drugs.

## What happened / why it failed
This is where the try03 "win" fell apart under a fair test. With real hard negatives in
place:
- **Fabry**: migalastat predicted 754 (real label 196) — and this exact value matched
  one of its own diastereomer hard negatives' real training label. Diagnosis: the model
  was doing near-exact-string-match retrieval against memorized training examples, not
  learning real structure-activity signal — plausible because every training compound
  was seen as only ONE canonical SMILES string, an easy shortcut to memorize verbatim.
- **Gaucher**: ambroxol predicted 720 (real label 616), lost to both easy negatives and
  its own real hard negatives (839, 839, 454).

This diagnosis (SMILES-string memorization) directly motivated try05's fix: augment
training with multiple randomized-but-equivalent SMILES per compound so the model has
to learn the molecule, not one specific string.
