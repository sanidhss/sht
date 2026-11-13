"""
Build the screening library for the stage-2 prototype.

Pulls approved drugs (max_phase=4) with canonical SMILES from the ChEMBL REST API.
This replaces the 9-candidate hand-curated multiple-choice set used in try02-try07:
that set contained the true drug, 3 authored analogs, the 3 other folds' drugs, and
2 obvious decoys (caffeine, aspirin), which makes "correct" a 1-of-9 pick. The real
task is ranking a receptor's match out of a realistic library, so the candidate pool
has to be thousands of plausible drug-like molecules, not 9 enriched ones.

No new mechanism authoring is needed for this -- only SMILES -- which is why the
candidate pool can scale even though authored per-disease mechanism descriptions
cannot.

Run on a login node (compute nodes have no outbound network).
"""
import json
import os
import sys
import time
import urllib.request

OUT_DIR = "/work/hdd/bisa/sshah21/results/reasoning_screen"
OUT_PATH = os.path.join(OUT_DIR, "chembl_approved.json")
BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
PAGE = 1000

# The four folds' real drugs must be present and findable by name, since rank-of-true-drug
# is the metric. ChEMBL pref_name is uppercase.
REQUIRED = ["MIGALASTAT", "AMBROXOL", "LUMACAFTOR", "TAFAMIDIS"]


def fetch_page(offset):
    url = f"{BASE}?max_phase=4&limit={PAGE}&offset={offset}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  retry {attempt + 1} at offset {offset}: {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch offset {offset}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records, offset = [], 0
    while True:
        payload = fetch_page(offset)
        molecules = payload.get("molecules", [])
        if not molecules:
            break
        for m in molecules:
            structures = m.get("molecule_structures") or {}
            smiles = structures.get("canonical_smiles")
            if not smiles:
                continue
            records.append({
                "chembl_id": m.get("molecule_chembl_id"),
                "name": m.get("pref_name"),
                "smiles": smiles,
            })
        total = payload.get("page_meta", {}).get("total_count")
        offset += PAGE
        print(f"  fetched {len(records)} with SMILES (offset {offset} / total {total})", flush=True)
        if total is not None and offset >= total:
            break

    # Deduplicate on canonical SMILES so the same structure under several ChEMBL ids
    # cannot occupy multiple ranks.
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    seen, library = set(), []
    n_unparseable = 0
    for rec in records:
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None:
            n_unparseable += 1
            continue
        # isomericSmiles=True: stereochemistry is the discriminating feature for the
        # Fabry fold (migalastat vs its stereoisomers), so it must not be flattened.
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canon in seen:
            continue
        seen.add(canon)
        rec["canonical_smiles"] = canon
        library.append(rec)

    print(f"\n{len(records)} fetched -> {len(library)} unique parseable "
          f"({n_unparseable} unparseable, {len(records) - len(library) - n_unparseable} duplicate)")

    names = {(r["name"] or "").upper() for r in library}
    missing = [d for d in REQUIRED if d not in names]
    print("required drugs present:", [d for d in REQUIRED if d in names])
    if missing:
        print("WARNING missing from library:", missing)

    with open(OUT_PATH, "w") as f:
        json.dump(library, f)
    print(f"wrote {OUT_PATH}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
