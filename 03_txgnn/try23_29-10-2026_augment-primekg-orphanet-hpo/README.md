# try23 — augment PrimeKG with Orphanet (disease→gene) + HPO (disease→symptom)

## Why
try18/try20 showed the thesis diseases are limited mainly by how they're represented in
PrimeKG: Fabry has a mislabelled duplicate (chylomicron retention disease) carrying its
answer, and the TTR nodes are nearly empty. This builds a corrected, rare-disease-enriched
copy of PrimeKG. The original stays untouched.

## Design rules (`build_augmented_kg.py`)
- **Only adds edges between nodes that already exist in PrimeKG.** TxGNN numbers nodes by
  sorting their IDs, so an unchanged node set keeps every index (Fabry 543, etc.) and all
  existing scripts valid.
- **Never adds drug–disease edges** (those are the answers). Only genes and symptoms are added.
- Sources:
  - Orphanet product 6: gene associations with status "Assessed", excluding "Candidate gene
    tested in" and "Biomarker tested in".
  - HPO `phenotype.hpoa`: aspect P, excluding NOT.
  - IDs are mapped via MONDO cross-references, using only unambiguous ones (an Orphanet/OMIM
    ID that maps to exactly one MONDO term).
- PrimeKG predates the current MONDO release, so some of its disease terms are now obsolete.
  An obsolete term inherits its replacement's cross-references. Example: hereditary ATTR
  amyloidosis MONDO:0017132 → 0007100, which PrimeKG also has as "familial amyloid
  neuropathy", so both nodes get the data.
- **Fabry-duplicate fix:** remove migalastat ↔ chylomicron retention disease, and that
  disease's gene links duplicated from Fabry (22, incl. GLA). It keeps SAR1B, APOC2 and LPL.
- `--dry-run` writes only `report.txt`, `added_edges.csv` and `removed_edges.csv`.

## Dry run on Delta (29-10-2026; Orphanet/HPO/MONDO downloaded the same day, HPO release 2026-09-01)
Added 80,710 rows (both directions): 3,748 disease→gene and 76,962 disease→symptom.
Removed 46 rows (the fix). For the thesis diseases:

| disease | genes before→after | symptoms before→after |
|---|---|---|
| Fabry | 23→24 | 69→78 |
| Gaucher | 5→5 | 115→149 |
| CF | 11→25 | 32→51 |
| ATTRV122I amyloidosis | 1→1 | 0→21 |
| hereditary ATTR amyloidosis | 0→0 | 0→34 |
| wild type ATTR amyloidosis | 0→0 | 23→35 |
| chylomicron retention disease | 25→3 | 26→26 |

Limit: hereditary ATTR still gets no TTR gene. Orphanet attaches genes to subtypes (e.g.
ATTRV30M, ATTRV122I), not to the umbrella term. Not hand-added, to keep every edge
source-derived.

Note: our current similarity setting (`all_nodes_profile`) uses only genes + related
diseases to pick lookalikes. The added symptoms still change every node's learned
representation, but they affect lookalike choice only under `all_nodes_profile_more`.

## Running it locally (user's machine), then copying to Delta
Needs Python 3 + pandas and roughly 8–10 GB RAM (kg.csv is ~1 GB).
```bash
# 1. get the script
git pull   # in your local clone of sht
# 2. get the inputs (same versions as the dry run)
scp sshah21@login.delta.ncsa.illinois.edu:/work/hdd/bisa/sshah21/primekg/kg.csv ./primekg/
scp -r sshah21@login.delta.ncsa.illinois.edu:/work/hdd/bisa/sshah21/external ./external
# 3. build
python build_augmented_kg.py --primekg ./primekg --external ./external --out ./primekg_aug
# 4. copy the result back
ssh sshah21@login.delta.ncsa.illinois.edu mkdir -p /work/hdd/bisa/sshah21/primekg_aug
scp ./primekg_aug/{kg.csv,report.txt,added_edges.csv,removed_edges.csv} \
    sshah21@login.delta.ncsa.illinois.edu:/work/hdd/bisa/sshah21/primekg_aug/
```
The local `report.txt` should match the dry-run numbers above.

## Next (on Delta, after the copy)
Link `node.csv`/`edges.csv` from the original folder (the node set is unchanged), let TxGNN
build `kg_directed.csv`, check the indices are unchanged, run the try18 lookalike check, then
rerun Fabry/Gaucher/CF/TTR with the original settings and compare.
