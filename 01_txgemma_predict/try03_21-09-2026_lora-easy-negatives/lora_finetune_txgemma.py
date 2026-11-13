import json
import os
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "google/txgemma-9b-predict"
TASK = "BindingDB_ic50"
BASE = "/projects/bisa/sshah21"
DATA_DIR = "/work/hdd/bisa/sshah21/results/txgemma_lora"

VARIANTS = {
    "blind":        "train_blind.jsonl",         # GLA excludes migalastat -- tests generalization
    "with_holdout": "train_with_holdout.jsonl",  # GLA includes migalastat -- tests the screening-tool framing
}

CANDIDATES = {
    "migalastat":  "C1[C@@H]([C@H]([C@H]([C@H](N1)CO)O)O)O",
    "isofagomine": "C1[C@@H]([C@H]([C@@H](CN1)O)O)CO",
    "aspirin":     "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine":    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "metformin":   "CN(C)C(=N)N=C(N)N",
    "ibuprofen":   "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "imatinib":    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
}


def read_fasta(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


TARGET_SEQS = {"gla": read_fasta(f"{BASE}/GLA.fasta"), "gaucher": read_fasta(f"{BASE}/GBA1.fasta")}
PROMPT_TMPL = json.load(open(hf_hub_download(repo_id=MODEL, filename="tdc_prompts.json")))[TASK]

LORA_CONFIG = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def load_examples(path):
    with open(os.path.join(DATA_DIR, path)) as f:
        return [json.loads(line) for line in f]


def make_batch(tok, examples, device):
    input_ids_list, labels_list = [], []
    for ex in examples:
        prompt_ids = tok(ex["prompt"], add_special_tokens=True)["input_ids"]
        answer_ids = tok(ex["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        input_ids_list.append(ids)
        labels_list.append(labels)
    maxlen = max(len(x) for x in input_ids_list)
    pad_id = tok.pad_token_id
    input_ids = torch.full((len(examples), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), maxlen), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        attn[i, :len(ids)] = 1
    return input_ids.to(device), attn.to(device), labels.to(device)


def train_adapter(model, tok, examples, n_epoch=3, lr=2e-4, micro_bs=1, grad_accum=8):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device
    for epoch in range(n_epoch):
        import random
        random.shuffle(examples)
        total_loss, step = 0.0, 0
        opt.zero_grad()
        for i in range(0, len(examples), micro_bs):
            batch = examples[i:i + micro_bs]
            input_ids, attn, labels = make_batch(tok, batch, device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            total_loss += out.loss.item()
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        print(f"  epoch {epoch}: avg loss {total_loss / step:.4f}")
    model.eval()


@torch.no_grad()
def eval_candidates(model, tok, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    for tname, tseq in TARGET_SEQS.items():
        scores = {}
        for cname, smiles in CANDIDATES.items():
            prompt = PROMPT_TMPL.replace("{Drug SMILES}", smiles).replace("{Target amino acid sequence}", tseq)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=8)
            answer = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            scores[cname] = answer
            print(f"  {tname}/{cname}: {answer!r}")
        with open(os.path.join(result_dir, f"{tname}_results.json"), "w") as f:
            json.dump({"task": TASK, "target": tname, "scores": scores}, f, indent=2)


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    model = get_peft_model(base_model, LORA_CONFIG, adapter_name="blind")
    model.add_adapter("with_holdout", LORA_CONFIG)

    for variant, fname in VARIANTS.items():
        print(f"=== Training variant: {variant} ===")
        model.set_adapter(variant)
        examples = load_examples(fname)
        train_adapter(model, tok, examples)

        result_dir = os.path.join(DATA_DIR, variant)
        model.save_pretrained(os.path.join(result_dir, "adapter"), selected_adapters=[variant])

        print(f"=== Evaluating variant: {variant} ===")
        eval_candidates(model, tok, result_dir)
