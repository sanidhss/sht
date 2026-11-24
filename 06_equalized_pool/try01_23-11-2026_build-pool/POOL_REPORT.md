# Equalized candidate pool v1 (1933 drugs)

Built once and frozen. Every arm of the cross-architecture comparison scores this
exact pool, and verifies the sha256 before scoring.

## Provenance
- TxGNN drug universe: `/work/hdd/bisa/sshah21/results/txgnn_v4_new_disease_set/*/seed_1/eval_*.pkl` keys -- **7957** DrugBank ids
- Library: `/work/hdd/bisa/sshah21/results/reasoning_screen/chembl_approved.json` -- **3417** records (3417 unique ChEMBL ids)
- Mapping: UniChem whole-source src_id 1 (ChEMBL) -> src_id 2 (DrugBank), https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt.gz
  cached at `/work/hdd/bisa/sshah21/results/equalized_pool/unichem_src1src2.txt.gz`, 11168 rows

## Filtering
| step | count |
|---|---|
| UniChem pairs | 11168 |
| 1:1 pairs kept | 11152 |
| ambiguous pairs dropped (salts/mixtures; see `dropped_ambiguous.csv`) | 16 |
| library ChEMBL ids with no 1:1 DrugBank map | 1302 |
| dropped: SMILES failed RDKit parse | 0 |
| **final pool** | **1933** |

Ambiguity rule: a pair is kept only if its ChEMBL id has exactly one DrugBank
partner *and* that DrugBank id has exactly one ChEMBL partner. Arbitrarily picking
one member of an ambiguous group is the same class of error as try14's
imatinib-mesylate hard negative.

## Reference pre-check
| drug | drugbank | chembl | target | status |
|---|---|---|---|---|
| ambroxol | DB06742 | - | Gaucher (chaperone arm) | MISSING: UniChem maps it to CHEMBL153479, which is not in the library |
| dantrolene | DB01219 | CHEMBL1201288 | malignant hyperthermia | in pool |
| imatinib | DB00619 | CHEMBL941 | chronic eosinophilic leukemia | in pool |
| lumacaftor | DB09280 | CHEMBL2103870 | CF | in pool |
| midostaurin | DB06595 | CHEMBL608533 | systemic mastocytosis | in pool |
| migalastat | DB05018 | CHEMBL110458 | Fabry | in pool |
| nitisinone | DB00348 | CHEMBL1337 | tyrosinemia | in pool |
| sapropterin | DB00360 | CHEMBL1201774 | phenylketonuria | in pool |
| tafamidis | DB11644 | CHEMBL2103837 | TTR | in pool |
| tolvaptan | DB06212 | - | NSIAD | MISSING: UniChem maps it to CHEMBL4303142, which is not in the library; the library has CHEMBL344159 under this name (different ChEMBL record -- NOT substituted) |

**Targets that CANNOT be evaluated on this pool:**
- Gaucher (chaperone arm): reference ambroxol (DB06742) MISSING: UniChem maps it to CHEMBL153479, which is not in the library
- NSIAD: reference tolvaptan (DB06212) MISSING: UniChem maps it to CHEMBL4303142, which is not in the library; the library has CHEMBL344159 under this name (different ChEMBL record -- NOT substituted)

No salt or analogue was substituted.

## Freeze
- `/work/hdd/bisa/sshah21/results/equalized_pool/pool_v1.json`
- sha256: `9d89c3ce758b7ec7c8bea2866f6448a9ea71b372f0f3c3f92c8c6b690653e308`
- Records sorted by `drugbank_id`, JSON written with sorted keys, so the file is byte-stable.
