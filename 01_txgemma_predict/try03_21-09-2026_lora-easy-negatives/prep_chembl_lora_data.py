import json
import math
import urllib.request
from urllib.parse import urlencode

BASE = "/projects/bisa/sshah21"
OUT_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora"

TARGETS = {
    "gla":    {"chembl_id": "CHEMBL2524", "fasta": f"{BASE}/GLA.fasta"},
    "gaucher": {"chembl_id": "CHEMBL2179", "fasta": f"{BASE}/GBA1.fasta"},
}

# compounds we test with later -- flagged so we can build a "held out" training
# variant that never saw their real label.
HELD_OUT_NAMES = {"migalastat", "isofagomine"}

PROMPT_TMPL = json.load(open("/work/hdd/bisa/sshah21/hf_cache/hub/models--google--txgemma-9b-predict/"
                              "snapshots/d8e65c274206c96631276df0ac009a5c6c24b293/tdc_prompts.json"))["BindingDB_ic50"]


def read_fasta(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


def fetch_chembl_activities(target_chembl_id):
    activities = []
    url = ("https://www.ebi.ac.uk/chembl/api/data/activity.json?" +
           urlencode({"target_chembl_id": target_chembl_id, "standard_type": "IC50",
                      "standard_units": "nM", "limit": 1000}))
    while url:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        activities.extend(data["activities"])
        url = data["page_meta"]["next"]
        if url:
            url = "https://www.ebi.ac.uk" + url
    return activities


def build_examples(name, chembl_id, sequence):
    activities = fetch_chembl_activities(chembl_id)
    # dedup by SMILES, averaging log10(IC50) across repeated measurements
    by_smiles = {}
    names_by_smiles = {}
    for a in activities:
        smi = a.get("canonical_smiles")
        val = a.get("standard_value")
        if not smi or val is None:
            continue
        try:
            val = float(val)
        except ValueError:
            continue
        if val <= 0:
            continue
        by_smiles.setdefault(smi, []).append(math.log10(val))
        mol_name = (a.get("molecule_pref_name") or "").strip().lower()
        if mol_name:
            names_by_smiles[smi] = mol_name

    log_vals = {smi: sum(v) / len(v) for smi, v in by_smiles.items()}
    lo, hi = min(log_vals.values()), max(log_vals.values())

    def normalize(v):
        if hi == lo:
            return 500
        return round(1000 * (v - lo) / (hi - lo))

    held_out, kept = [], []
    for smi, logv in log_vals.items():
        rec = {
            "target": name,
            "smiles": smi,
            "name": names_by_smiles.get(smi, ""),
            "score": normalize(logv),
            "prompt": PROMPT_TMPL.replace("{Drug SMILES}", smi).replace("{Target amino acid sequence}", sequence),
        }
        if rec["name"] in HELD_OUT_NAMES:
            held_out.append(rec)
        else:
            kept.append(rec)
    return kept, held_out


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({"prompt": r["prompt"], "answer": str(r["score"])}) + "\n")


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    all_blind, all_with_holdout, held_out_all = [], [], []
    for name, cfg in TARGETS.items():
        seq = read_fasta(cfg["fasta"])
        kept, held_out = build_examples(name, cfg["chembl_id"], seq)
        print(f"{name}: {len(kept)} training examples, {len(held_out)} held out "
              f"({[h['name'] for h in held_out]})")
        all_blind.extend(kept)
        all_with_holdout.extend(kept + held_out)
        held_out_all.extend(held_out)

    write_jsonl(os.path.join(OUT_DIR, "train_blind.jsonl"), all_blind)
    write_jsonl(os.path.join(OUT_DIR, "train_with_holdout.jsonl"), all_with_holdout)
    with open(os.path.join(OUT_DIR, "held_out_compounds.json"), "w") as f:
        json.dump(held_out_all, f, indent=2)

    print(f"\nWrote train_blind.jsonl ({len(all_blind)} examples)")
    print(f"Wrote train_with_holdout.jsonl ({len(all_with_holdout)} examples)")
    print(f"Wrote held_out_compounds.json ({len(held_out_all)} compounds)")
