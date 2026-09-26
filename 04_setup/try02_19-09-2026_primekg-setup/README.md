# Try 02 — PrimeKG data preparation

**Date:** 19-09-2026
**Scripts:** `get_primekg.py` + `get_primekg.sb`
**Jobs:** 22226229 (FAILED), 22226550 (FAILED), 22226635 (COMPLETED)

## What it does
CPU job that calls `TxData(data_folder_path=...).prepare_split(split='complex_disease',
seed=42)` to materialize PrimeKG's processed knowledge graph files
(`kg_directed.csv`, the `complex_disease_42/{train,valid,test}.csv` split, etc.) that
every TxGNN run depends on.

## What happened
Two early failures (logs not preserved in current results directories — likely
environment/path setup issues before the script reached a stable state), then a
successful completion on the third attempt. The resulting processed files
(`kg_directed.csv` etc.) are what every TxGNN try in `03_txgnn/` reads from — they were
NOT regenerated per try, so any bug specific to this initial processing step would have
propagated to every later TxGNN attempt. No such bug was found; the disease-index issues
found later (see `03_txgnn/try03`) were in how `eval_disease_centric` was *called*, not
in this preprocessing step itself.
