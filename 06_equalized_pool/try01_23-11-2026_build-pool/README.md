# try01 — build and freeze the equalized candidate pool

## Why
The four arms score different drug universes, so none of their current numbers are
comparable:
- TxGNN can only score PrimeKG drug nodes: **7,957**, DrugBank-keyed.
- The chat and Tanimoto arms score `chembl_approved.json`: **3,417**, ChEMBL-keyed.

This builds their intersection once, freezes it, and publishes a sha256. Every arm
scores this exact pool and verifies the hash before scoring. **Nothing is scored here.**

## What ran
`build_pool.py` (CPU, ~1 min, `envs/txgemma` — the only env with RDKit; `txgnn2` has none).

1. **TxGNN universe:** the key set of `eval_*.pkl` under
   `/work/hdd/bisa/sshah21/results/txgnn_v4_new_disease_set/<disease>/seed_1/`. One file
   per disease is read and all 6 are asserted identical (7,957 ids), which catches a
   truncated run.
2. **Mapping:** UniChem whole-source file src_id 1 (ChEMBL) → src_id 2 (DrugBank),
   `https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt.gz`,
   cached at `/work/hdd/bisa/sshah21/results/equalized_pool/unichem_src1src2.txt.gz`
   (downloaded 23-11-2026, 11,168 rows). The file is used instead of per-ID API calls: one
   request, and a citable artefact.
3. **1:1 only.** A pair is kept only if its ChEMBL id has exactly one DrugBank partner
   **and** that DrugBank id has exactly one ChEMBL partner. 11,152 kept; **16 ambiguous
   pairs dropped** (touching 24 ids), all written to `dropped_ambiguous.csv` with both id
   sets. Arbitrarily picking one member of an ambiguous group is the same class of error
   as try14's imatinib-mesylate hard negative.
4. **Intersect + canonicalise.** 1,302 library ChEMBL ids had no 1:1 DrugBank map. 0 SMILES
   failed RDKit parsing. SMILES are RDKit-canonicalised.

## Result
**Pool: 1,933 drugs.**

| | |
|---|---|
| `pool_v1.json` | `/work/hdd/bisa/sshah21/results/equalized_pool/pool_v1.json` |
| sha256 | `9d89c3ce758b7ec7c8bea2866f6448a9ea71b372f0f3c3f92c8c6b690653e308` |

Records sorted by `drugbank_id`, JSON written with sorted keys, so the file is
byte-stable: two independent runs produced the identical hash.

## Reference pre-check — 8 of 10 present, 2 targets NOT evaluable
In pool: migalastat (DB05018), lumacaftor (DB09280), tafamidis (DB11644), imatinib
(DB00619), midostaurin (DB06595), nitisinone (DB00348), dantrolene (DB01219),
sapropterin (DB00360).

**Missing — these targets cannot be evaluated on this pool:**

| drug | target | why |
|---|---|---|
| ambroxol (DB06742) | Gaucher (chaperone arm) | UniChem maps it 1:1 to CHEMBL153479, but that id is **not in `chembl_approved.json` at all** — ambroxol appears under no name in the library. It is not an approved-drug entry there. |
| tolvaptan (DB06212) | NSIAD | UniChem maps it 1:1 to **CHEMBL4303142**; the library instead carries **CHEMBL344159** under the name TOLVAPTAN. Two different ChEMBL records for the same drug, so the 1:1 join misses. **Not substituted.** |

No salt or analogue was substituted for either. Two ways forward, to be decided before
any arm scores: drop these two targets, or extend the mapping (e.g. add a curated
ChEMBL-id override table, reported explicitly) and re-freeze as `pool_v2`. Tolvaptan's
case is a plain ChEMBL-record mismatch and is defensibly fixable; ambroxol's is not —
the library genuinely lacks the drug.

Note: NSIAD is in any case a weak target — PrimeKG's tolvaptan indication for it is
medically questionable (try24), and its TxGNN rank was 6,373.

## Files
- `build_pool.py` — the whole job.
- `POOL_REPORT.md` — generated: counts, filtering table, reference pre-check, freeze info.
- `dropped_ambiguous.csv` — generated: every dropped ambiguous pair with both id groups.
