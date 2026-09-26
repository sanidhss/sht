import json, re, zlib, random, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "google/txgemma-27b-chat"
BASE = "/projects/bisa/sshah21"
RESULTS_DIR = "/work/hdd/bisa/sshah21/results/blind_chaperone_rank"

# Candidate pool: the two known chaperones plus unrelated distractor drugs.
# Names are stripped before prompting the model -- it only ever sees
# "Compound A/B/C..." and a SMILES string, never a drug name or disease name.
CANDIDATES = {
    "migalastat":  "C1[C@@H]([C@H]([C@H]([C@H](N1)CO)O)O)O",
    "isofagomine": "C1[C@@H]([C@H]([C@@H](CN1)O)O)CO",
    "aspirin":     "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine":    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "metformin":   "CN(C)C(=N)N=C(N)N",
    "ibuprofen":   "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "imatinib":    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
}

# Targets are described only by sequence + natural substrate -- never by gene
# or disease name -- so a correct answer requires structural/biochemical
# reasoning rather than a memorized "disease -> drug" lookup.
TARGETS = {
    "GLA": {
        "fasta": "GLA.fasta",
        "substrate": "globotriaosylceramide (Gb3), a terminal alpha-galactose glycosphingolipid",
        "expected": "migalastat",
    },
    "GBA1": {
        "fasta": "GBA1.fasta",
        "substrate": "glucosylceramide, a terminal beta-glucose glycosphingolipid",
        "expected": "isofagomine",
    },
}

def read_fasta(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))

def blinded_labels(seed_key):
    names = list(CANDIDATES.keys())
    rng = random.Random(zlib.crc32(seed_key.encode()))
    rng.shuffle(names)
    labels = [f"Compound {chr(ord('A') + i)}" for i in range(len(names))]
    label_to_name = dict(zip(labels, names))
    return label_to_name

def build_prompt(seq, substrate, label_to_name):
    compound_lines = "\n".join(
        f"{label}: {CANDIDATES[name]}" for label, name in label_to_name.items()
    )
    return f"""You are analyzing candidate small molecules as pharmacological chaperones for a human enzyme.

Enzyme sequence (single-letter amino acids):
{seq}

Biochemical context:
- This is a lysosomal hydrolase. Destabilizing missense mutations in patients cause a lysosomal storage disorder.
- Its natural substrate is {substrate}.
- A pharmacological chaperone is a small molecule that binds reversibly in the enzyme's active site, mimicking the substrate's structure, thereby stabilizing the mutant enzyme's fold and restoring residual activity.

Candidate compounds (SMILES):
{compound_lines}

Task: using ONLY the biochemical reasoning above (substrate shape, stereochemistry, active-site mimicry) -- not any memorized drug or brand name association -- rank the candidates from most to least likely to act as a pharmacological chaperone for this specific enzyme.

Respond with strict JSON only, no other text, in this exact schema:
{{"ranking": ["<label>", "..."], "top_choice": "<label>", "reasoning": "2-4 sentences of biochemical justification for the top choice"}}
"""

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", dtype=torch.bfloat16)

    for target_name, info in TARGETS.items():
        seq = read_fasta(f"{BASE}/{info['fasta']}")
        label_to_name = blinded_labels(target_name)
        prompt = build_prompt(seq, info["substrate"], label_to_name)

        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to(model.device)
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        parsed = extract_json(reply)
        top_label = parsed.get("top_choice") if parsed else None
        top_name = label_to_name.get(top_label) if top_label else None
        correct = top_name == info["expected"]

        print(f"=== {target_name} (blinded, expected={info['expected']}) ===")
        print("label -> compound map:", label_to_name)
        print("raw reply:", reply)
        print(f"model top choice: {top_label} -> {top_name} | correct: {correct}")
        print()

        with open(os.path.join(RESULTS_DIR, f"{target_name}.json"), "w") as f:
            json.dump({
                "target": target_name,
                "expected": info["expected"],
                "label_to_name": label_to_name,
                "raw_reply": reply,
                "parsed": parsed,
                "top_choice_label": top_label,
                "top_choice_name": top_name,
                "correct": correct,
            }, f, indent=2)

if __name__ == "__main__":
    main()
