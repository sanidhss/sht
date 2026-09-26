import json, os, torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "google/txgemma-9b-predict"
TASK  = "BindingDB_ki"
BASE  = "/projects/bisa/sshah21"
RESULTS_DIR = "/work/hdd/bisa/sshah21/results/txgemma"

def read_fasta(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))

targets = {
    "GLA":  read_fasta(f"{BASE}/GLA.fasta"),
    "GBA1": read_fasta(f"{BASE}/GBA1.fasta"),
    "ABL1": read_fasta(f"{BASE}/ABL1.fasta"),
}

candidates = {
    "migalastat":  "C1[C@@H]([C@H]([C@H]([C@H](N1)CO)O)O)O",
    "isofagomine": "C1[C@@H]([C@H]([C@@H](CN1)O)O)CO",
    "aspirin":     "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine":    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "metformin":   "CN(C)C(=N)N=C(N)N",
    "ibuprofen":   "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "imatinib":    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
}

tmpl  = json.load(open(hf_hub_download(repo_id=MODEL, filename="tdc_prompts.json")))[TASK]
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", dtype=torch.bfloat16)

for tname, tseq in targets.items():
    print(f"--- {tname} (len {len(tseq)}) ---")
    result_dir = os.path.join(RESULTS_DIR, tname)
    os.makedirs(result_dir, exist_ok=True)
    scores = {}
    for cname, smiles in candidates.items():
        prompt = tmpl.replace("{Drug SMILES}", smiles).replace("{Target amino acid sequence}", tseq)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=16)
        answer = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{cname}: {repr(answer)}")
        scores[cname] = answer
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump({"task": TASK, "target": tname, "scores": scores}, f, indent=2)
